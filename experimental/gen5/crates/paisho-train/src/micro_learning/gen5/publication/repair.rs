//! Decision-based publication with explicit counterexamples. Work happens on
//! private candidates; the accepted snapshot changes only after all panels pass.
use super::*;

pub(super) const MAX_KL: f64 = 0.001;
const ERROR_TOLERANCE: f64 = 1e-9;
const FOCUS_LIMIT: usize = 32;

#[derive(Clone, Serialize, Deserialize)]
pub(super) struct Feedback {
    panel: usize,
    row: usize,
    position_sha256: String,
    raw_choice_lost: bool,
    coupled_choice_lost: bool,
    value_error_increase: f64,
    #[serde(default)]
    previous_value_error: f64,
    #[serde(default)]
    retained_publications: usize,
}

fn best(p: &[f64], valid: impl Fn(usize) -> bool) -> Option<usize> {
    (0..p.len())
        .filter(|&i| valid(i))
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
}

pub(super) fn kl(old: &Score, new: &Score) -> f64 {
    let mut sum = 0.;
    let mut count = 0;
    for (a, b) in old.priors.iter().zip(new.priors.iter()) {
        if a.is_empty() {
            continue;
        }
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        sum += a
            .iter()
            .zip(b)
            .filter(|(p, _)| **p > 0.)
            .map(|(p, q)| p * (p.max(1e-300).ln() - q.max(1e-300).ln()))
            .sum::<f64>();
        count += 1;
    }
    sum.max(0.) / count.max(1) as f64
}

fn retained(old: &Score, new: &Score) -> bool {
    old.raw
        .iter()
        .zip(&new.raw)
        .zip(old.coupled.iter().zip(&new.coupled))
        .all(|((a, b), (c, d))| (!a || *b) && (!c || *d))
}

fn reasons(old: &Score, new: &Score) -> Vec<String> {
    let mut reasons = vec![];
    if !retained(old, new) {
        reasons.push("known proof choice lost".into());
    }
    if !new.value_mse.is_finite() || new.value_mse > old.value_mse + ERROR_TOLERANCE {
        reasons.push("balanced proof value error increased".into());
    }
    let shift = kl(old, new);
    if !shift.is_finite() || shift > MAX_KL {
        reasons.push(format!(
            "known winning policy KL {shift:.6} exceeds {MAX_KL}"
        ));
    }
    reasons
}

impl Guard {
    pub fn enable_v3(&mut self) -> Result<()> {
        if self.validation.is_none() || !self.accepted.model.has_deep_value() {
            return Err(invalid(
                "transactional publication requires V2 panels and separated value",
            ));
        }
        self.state.learning_loop_v3 = true;
        Ok(())
    }

    pub fn diagnostic_measure(&self, model: &MicroModel) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"primary":self.evaluate(model)?,
            "validation":self.validation.as_deref().map(|v|v.evaluate(model)).transpose()?,
            "scope":"known positions; coupled is a prior ranking, not a full MCTS game"}))
    }

    fn panels(&self) -> impl Iterator<Item = &Guard> {
        std::iter::once(self).chain(self.validation.as_deref())
    }

    pub(super) fn checked_v3(&self, model: &MicroModel) -> Result<(Vec<Score>, Vec<String>)> {
        let _cost = transaction_bench::Cost::scope(&self.publication_cost, "checked_v3");
        let mut scores = vec![];
        let mut why = vec![];
        let joint = if self.reads.joint_panels {
            self.parallel.as_ref().map(|pool| {
                let _cost=transaction_bench::Cost::scope(&self.publication_cost,"checked_v3.joint_panels");
                read_cache::evaluate_panels(&self.panels().map(|p|(&p.reads,p.rows.as_slice(),p.beta)).collect::<Vec<_>>(),model,pool)
            }).transpose()?
        } else { None };
        let mut joint = joint.map(|v|v.into_iter());
        for (i, p) in self.panels().enumerate() {
            let panel_cost = transaction_bench::Cost::scope(&self.publication_cost, "checked_v3.panel_evaluate");
            let score = if let Some(joint) = &mut joint {joint.next().expect("missing panel score")}
                else {p.evaluate(model)?};
            drop(panel_cost);
            why.extend(
                reasons(&p.score, &score)
                    .into_iter()
                    .map(|w| format!("panel {i}: {w}")),
            );
            scores.push(score);
        }
        if let Some(book) = &self.learned_choices {
            let _registry_cost = transaction_bench::Cost::scope(&self.publication_cost, "checked_v3.registry");
            if let Some(cost) = &self.publication_cost { cost.registry_read(model); }
            let lost = book.retention_losses(model)?;
            if !lost.is_empty() {
                why.push(format!("acquired raw choices lost: {}", lost.join(",")));
            }
        }
        Ok((scores, why))
    }

    /// Capture a failed working candidate BEFORE a transaction may roll it back.
    /// Metadata survives checkpoints; examples are reconstructed from verified
    /// resident witnesses and spend the existing recall quota.
    pub fn observe_candidate(&mut self, model: &MicroModel) -> Result<Vec<Arc<MicroExample>>> {
        if !self.state.learning_loop_v3 {
            return Ok(vec![]);
        }
        let mut choices = vec![];
        let mut values = vec![];
        for (panel, p) in self.panels().enumerate() {
            let score = p.evaluate(model)?;
            for (row, witness) in p.rows.iter().enumerate() {
                let raw = p.score.raw[row] && !score.raw[row];
                let coupled = p.score.coupled[row] && !score.coupled[row];
                let increase = score.errors[row] - p.score.errors[row];
                let item = Feedback {
                    panel,
                    row,
                    position_sha256: sha256(&serde_json::to_vec(&witness.example.state)?),
                    raw_choice_lost: raw,
                    coupled_choice_lost: coupled,
                    value_error_increase: increase.max(0.),
                    previous_value_error: p.score.errors[row],
                    retained_publications: 0,
                };
                if raw || coupled {
                    choices.push(item.clone());
                }
                if increase > ERROR_TOLERANCE {
                    values.push(item);
                }
            }
        }
        values.sort_by(|a, b| b.value_error_increase.total_cmp(&a.value_error_increase));
        // Interleave error types so a large value failure cannot evict every
        // decision correction, or vice versa. Deduplicate joint failures.
        let mut selected: Vec<Feedback> = vec![];
        let mut c = choices.into_iter();
        let mut v = values.into_iter();
        loop {
            let pair = [c.next(), v.next()];
            if pair.iter().all(Option::is_none) {
                break;
            }
            for item in pair.into_iter().flatten() {
                if !selected
                    .iter()
                    .any(|x| x.panel == item.panel && x.row == item.row)
                {
                    selected.push(item);
                }
            }
        }
        // A transient successful copy does not erase unresolved saved tasks.
        if !selected.is_empty() {
            for item in &mut selected {
                if let Some(old) = self
                    .state
                    .feedback
                    .iter()
                    .find(|old| old.panel == item.panel && old.row == item.row)
                {
                    item.previous_value_error =
                        item.previous_value_error.min(old.previous_value_error);
                    item.raw_choice_lost |= old.raw_choice_lost;
                    item.coupled_choice_lost |= old.coupled_choice_lost;
                    item.value_error_increase =
                        item.value_error_increase.max(old.value_error_increase);
                }
            }
            for old in &self.state.feedback {
                if !selected
                    .iter()
                    .any(|x| x.panel == old.panel && x.row == old.row)
                {
                    selected.push(old.clone());
                }
            }
            self.state.feedback = selected;
        }
        self.correction_examples()
    }

    pub fn correction_examples(&self) -> Result<Vec<Arc<MicroExample>>> {
        let panels: Vec<_> = self.panels().collect();
        let mut out = vec![];
        for item in self
            .state
            .feedback
            .iter()
            .cycle()
            .skip(if self.state.feedback.is_empty() {
                0
            } else {
                self.state.feedback_cursor % self.state.feedback.len()
            })
            .take(FOCUS_LIMIT.min(self.state.feedback.len()))
        {
            let row = panels
                .get(item.panel)
                .and_then(|p| p.rows.get(item.row))
                .ok_or_else(|| invalid("feedback witness outside frozen panels"))?;
            if sha256(&serde_json::to_vec(&row.example.state)?) != item.position_sha256 {
                return Err(invalid("feedback witness changed on resume"));
            }
            let mut ex = row.example.as_ref().clone();
            ex.value_weight = if item.value_error_increase > ERROR_TOLERANCE {
                1.
            } else {
                0.
            };
            ex.policy_weight = if item.raw_choice_lost || item.coupled_choice_lost {
                1.
            } else {
                0.
            };
            if ex.policy_weight == 0. {
                ex.structured.clear();ex.actions.clear();
                ex.policy.clear();
            }
            ex.action_values.clear();
            ex.policy_support = ex.policy_weight > 0.;
            out.push(Arc::new(ex));
        }
        Ok(self.merge_transfer_feedback(out))
    }

    fn snapshot_repair(
        &self,
        model: MicroModel,
        parent: &Snapshot,
        steps: usize,
    ) -> Result<Arc<Snapshot>> {
        let source = parent
            .artifact
            .as_ref()
            .ok_or_else(|| invalid("repair requires learner artifact"))?;
        let a = Arc::new(MicroArtifact::new(
            &model,
            source.updates,
            serde_json::json!({"kind":"gen5-transactional-policy-repair-v1",
                "learner":parent.identity,"accepted_from":self.accepted.identity,
                "repair_steps":steps,"version":parent.version,
                "repair_steps_counted_separately_from_sgd":true}),
        ));
        let identity = a.identity();
        Ok(Arc::new(Snapshot {
            artifact: Some(a),
            model: Arc::new(model),
            version: parent.version,
            path: self
                .out
                .join("accepted")
                .join(format!("model-{identity}.json")),
            identity,
        }))
    }

    /// Signed log-probability margin matches the failed raw/coupled decision.
    /// Value parameters stay frozen during policy repair; successor values are
    /// immutable and already cached for every tested candidate.
    fn margin_gradient(&self, model: &MicroModel) -> Result<Option<(Vec<f64>, f64)>> {
        let _cost = transaction_bench::Cost::scope(&self.publication_cost, "margin_gradient");
        let mut gradient = vec![0.; model.parameters().len()];
        let mut objective = 0.;
        let mut count = 0usize;
        for p in self.panels() {
            let score = p.evaluate(model)?;
            if self.cached_margin_reads {
                // Raw-only failures do not consume coupled coefficients. Keep
                // their cells cold and leave the serial gradient fold unchanged.
                p.reads.prefill(
                    score
                        .coupled_logits
                        .iter()
                        .enumerate()
                        .filter_map(|(i, logits)| {
                            (p.score.coupled[i] && !score.coupled[i]).then_some(logits)
                        }),
                    p.parallel.as_ref(),
                );
            }
            for (i, row) in p.rows.iter().enumerate() {
                let raw = p.score.raw[i] && !score.raw[i];
                let coupled = p.score.coupled[i] && !score.coupled[i];
                if !raw && !coupled {
                    continue;
                }
                let logp: Vec<_> = score.priors[i].iter().map(|p| p.max(1e-300).ln()).collect();
                let mut rebuilt;
                let coupled_logits: &Vec<f64> = if self.cached_margin_reads && coupled {
                    score
                        .coupled_logits
                        .get(i)
                        .ok_or_else(|| invalid("missing measured coupled logits"))?
                } else {
                    rebuilt = logp.clone();
                    if coupled {
                        let inputs = row
                            .successors
                            .get()
                            .ok_or_else(|| invalid("missing measured successors"))?;
                        for (l, (sign, state)) in rebuilt.iter_mut().zip(inputs) {
                            *l += p.beta
                                * if state.is_empty() {
                                    *sign
                                } else {
                                    sign * model.value(state)
                                };
                        }
                    }
                    &rebuilt
                };
                for logits in [raw.then_some(&logp), coupled.then_some(coupled_logits)]
                    .into_iter()
                    .flatten()
                {
                    let good = best(logits, |j| row.valid[j])
                        .ok_or_else(|| invalid("proof has no winning action"))?;
                    let bad = best(logits, |j| !row.valid[j])
                        .ok_or_else(|| invalid("lost choice has no competitor"))?;
                    let mut ex = row.example.as_ref().clone();
                    ex.value_weight = 0.;
                    ex.policy_weight = 1.;
                    ex.action_values.clear();
                    ex.policy.fill(0.);
                    ex.policy[good] = 1.;
                    let loss_cost = transaction_bench::Cost::scope(&self.publication_cost, "margin_gradient.loss_pair");
                    let mut g = model.loss_gradient(&ex).map_err(invalid)?.1;
                    ex.policy[good] = 0.;
                    ex.policy[bad] = 1.;
                    let b = model.loss_gradient(&ex).map_err(invalid)?.1;
                    drop(loss_cost);
                    for (j, ((sum, a), b)) in
                        gradient.iter_mut().zip(g.iter_mut()).zip(b).enumerate()
                    {
                        if !branches::value_parameter(j) {
                            *sum += *a - b;
                        }
                    }
                    objective += logits[bad] - logits[good];
                    count += 1;
                }
            }
        }
        if count == 0 {
            if let Some(audit) = &self.reads.audit {
                audit.gradient(model, None);
            }
            return Ok(None);
        }
        for g in &mut gradient {
            *g /= count as f64;
        }
        let gap = objective / count as f64;
        if let Some(audit) = &self.reads.audit {
            audit.gradient(model, Some((&gradient, gap)));
        }
        Ok(Some((gradient, gap)))
    }

    fn repaired_candidate(
        &self,
        candidate: &Snapshot,
    ) -> Result<Option<(Arc<Snapshot>, Vec<Score>, usize)>> {
        let _cost = transaction_bench::Cost::scope(&self.publication_cost, "repaired_candidate");
        let mut model = candidate.model.as_ref().clone();
        for step in 0..=8 {
            let _step_cost = transaction_bench::Cost::scope(&self.publication_cost, "repair.step");
            let (scores, why) = self.checked_v3(&model)?;
            if why.is_empty() {
                return Ok(Some((
                    self.snapshot_repair(model, candidate, step)?,
                    scores,
                    step,
                )));
            }
            // A policy repair cannot fix an invalid value branch.
            if step == 8
                || self
                    .panels()
                    .zip(&scores)
                    .any(|(p, s)| s.value_mse > p.score.value_mse + ERROR_TOLERANCE)
            {
                break;
            }
            let Some((gradient, gap)) = self.margin_gradient(&model)? else {
                break;
            };
            if self.adaptive_margin {
                let before = margin_step::merit(self, &scores)?;
                let inside_kl = self
                    .panels()
                    .zip(&scores)
                    .all(|(p, s)| kl(&p.score, s) <= MAX_KL);
                let Some(next) =
                    margin_step::policy_step(&model, &gradient, gap, before, |trial| {
                        let (measured, why) = self.checked_v3(trial)?;
                        let mut merit = margin_step::merit(self, &measured)?;
                        // Once a trial lies in the trust region, do not leave it
                        // while repairing choices. Backtracking will try a smaller step.
                        if inside_kl
                            && self
                                .panels()
                                .zip(&measured)
                                .any(|(p, s)| kl(&p.score, s) > MAX_KL)
                        {
                            merit = f64::INFINITY;
                        }
                        Ok(margin_step::Assessment {
                            merit,
                            fully_valid: why.is_empty(),
                            evidence: measured,
                        })
                    })?
                else {
                    break;
                };
                debug_assert!((1..=6).contains(&next.trials));
                debug_assert!(next.displacement_norm > 0. && next.displacement_norm <= 0.02);
                debug_assert!(next.gradient_norm.is_finite() && next.gradient_norm > 0.);
                if next.assessment.fully_valid {
                    return Ok(Some((
                        self.snapshot_repair(next.model, candidate, step + 1)?,
                        next.assessment.evidence,
                        step + 1,
                    )));
                }
                model = next.model;
                continue;
            }
            let norm = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !norm.is_finite() || norm == 0. {
                break;
            }
            let scale = 0.002 * (10. / norm).min(1.);
            let mut next = MicroModel::from_parameters(
                model
                    .parameters()
                    .iter()
                    .zip(gradient)
                    .map(|(w, g)| w - scale * g)
                    .collect(),
            )
            .map_err(invalid)?;
            if let Some(bank) = model.sequence_memory() {
                next = next.with_sequence_memory_owned(bank.clone());
            }
            model = next;
        }
        Ok(None)
    }

    fn interpolated_candidate(
        &self,
        candidate: &Snapshot,
        preserve_value: bool,
    ) -> Result<Option<(Arc<Snapshot>, Vec<Score>, f64, usize)>> {
        let _cost = transaction_bench::Cost::scope(&self.publication_cost, "interpolated_candidate");
        for fraction in [
            0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625,
        ] {
            let _trial_cost = transaction_bench::Cost::scope(&self.publication_cost, "interpolation.trial");
            let weights = self
                .accepted
                .model
                .parameters()
                .iter()
                .zip(candidate.model.parameters())
                .enumerate()
                .map(|(i, (old, new))| {
                    if preserve_value && branches::value_parameter(i) {
                        *new
                    } else {
                        old + fraction * (new - old)
                    }
                })
                .collect();
            let mut model = MicroModel::from_parameters(weights).map_err(invalid)?;
            if let Some(bank) = candidate.model.sequence_memory() {
                model = model.with_sequence_memory_owned(bank.clone());
            }
            if self.fast_interpolation
                && !preserve_value
                && self
                    .panels()
                    .map(|p| -> Result<_> {
                        let started = p.reads.prefilter_audit.as_ref().map(|_| Instant::now());
                        let result = if p.reads.parallel_prefilter {
                            prefilter::check_ordered(
                                &p.rows,
                                &p.score,
                                &model,
                                MAX_KL,
                                p.parallel.as_ref(),
                            )
                        } else {
                            prefilter::check(&p.rows, &p.score, &model, MAX_KL)
                        }?;
                        if let (Some(audit), Some(started)) = (&p.reads.prefilter_audit, started) {
                            audit.record(&model, &result, started.elapsed().as_secs_f64());
                        }
                        Ok(result)
                    })
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .any(|r| r.reject)
            {
                continue;
            }
            // This family cannot repair a failed interpolation. A lost active
            // choice is already a necessary rejection in checked_v3, so avoid
            // full coupled/value panels for that doomed trial. Keep the complete
            // path when margin repair may recover it; do not change its search.
            if self.dynamic_interpolation_prefilter
                && (!preserve_value || !self.repair_interpolations)
                && self.learned_choices.as_ref().map_or(Ok(false), |book| -> Result<bool> {
                    Ok(!book.retention_losses(&model)?.is_empty())
                })?
            {
                continue;
            }
            let (scores, why) = self.checked_v3(&model)?;
            if why.is_empty() {
                return Ok(Some((
                    self.snapshot_repair(model, candidate, 0)?,
                    scores,
                    fraction,
                    0,
                )));
            }
            // A legal trust-region step may lose only a few previous decisions.
            // Repair those margins before shrinking away the fresh learning.
            // Value stays fixed in this family; all final criteria still apply.
            if preserve_value
                && self.repair_interpolations
                && self.panels().zip(&scores).all(|(p, s)| {
                    s.value_mse <= p.score.value_mse + ERROR_TOLERANCE && kl(&p.score, s) <= MAX_KL
                })
            {
                let trial = self.snapshot_repair(model, candidate, 0)?;
                if let Some((fixed, scores, steps)) = self.repaired_candidate(&trial)? {
                    return Ok(Some((fixed, scores, fraction, steps)));
                }
            }
        }
        Ok(None)
    }

    pub(super) fn consider_v3(
        &mut self,
        candidate: Arc<Snapshot>,
        force: bool,
    ) -> Result<Option<Vec<Arc<MicroExample>>>> {
        self.consider_v3_order(candidate, force, true)
    }

    fn consider_v3_order(
        &mut self,
        candidate: Arc<Snapshot>,
        force: bool,
        joint_first: bool,
    ) -> Result<Option<Vec<Arc<MicroExample>>>> {
        if candidate.identity == self.accepted.identity || !force && !self.due() {
            return Ok(None);
        }
        self.last = paisho_platform::training_time::now();
        self.state.checks += 1;
        self.state.feedback_cursor = self.state.feedback_cursor.wrapping_add(FOCUS_LIMIT);
        if candidate.model.parameters() == self.accepted.model.parameters() {
            self.state.last_decision = "unchanged".into();
            self.publication_fresh = None;
            return Ok(Some(self.correction_examples()?));
        }
        self.observe_candidate(&candidate.model)?;
        let proposals = [
            ("full", candidate.clone()),
            (
                "value",
                self.composed(&self.accepted, &candidate, &candidate, "value")?,
            ),
            (
                "policy",
                self.composed(&candidate, &self.accepted, &candidate, "policy")?,
            ),
        ];
        let mut attempts = vec![];
        let mut selected = None;
        let mut accepted_fraction = 1.;
        let mut accepted_value_fraction = 1.;
        for (kind, p) in &proposals {
            if p.model.parameters() == self.accepted.model.parameters() {
                continue;
            }
            let (scores, why) = self.checked_v3(&p.model)?;
            attempts.push(serde_json::json!({"kind":kind,"identity":p.identity,"reasons":why,
                "policy_kl":self.panels().zip(&scores).map(|(a,b)|kl(&a.score,b)).collect::<Vec<_>>()}));
            if why.is_empty() {
                selected = Some((p.clone(), scores, 0, *kind));
                break;
            }
            // Do not discard every learned policy coefficient just because the
            // unscaled joint step is too large. First try a verified smaller
            // joint transaction. Branch-only fallbacks remain available.
            if joint_first && *kind == "full" {
                // Preserve the full value improvement while reducing only the
                // policy/reader step. Deep-value separation proves V(s) is
                // unchanged by this interpolation; the complete guard still
                // checks the value-dependent reader and coupled decisions.
                if self.separate_step_scales
                    && self
                        .panels()
                        .zip(&scores)
                        .all(|(p, s)| s.value_mse <= p.score.value_mse + ERROR_TOLERANCE)
                {
                    if let Some((p, scores, fraction, steps)) =
                        self.interpolated_candidate(p, true)?
                    {
                        accepted_fraction = fraction;
                        selected = Some((p, scores, steps, *kind));
                        break;
                    }
                }
                if let Some((p, scores, fraction, steps)) = self.interpolated_candidate(p, false)? {
                    accepted_value_fraction = fraction;
                    accepted_fraction = fraction;
                    selected = Some((p, scores, steps, *kind));
                    break;
                }
                if let Some((p, scores, steps)) = self.repaired_candidate(p)? {
                    selected = Some((p, scores, steps, *kind));
                    break;
                }
            }
        }
        if selected.is_none() {
            for (kind, p) in &proposals {
                if joint_first && *kind == "full" {
                    continue;
                }
                if p.model.parameters() == self.accepted.model.parameters() {
                    continue;
                }
                if let Some((p, scores, fraction, steps)) = self.interpolated_candidate(p, false)? {
                    accepted_fraction = fraction;
                    selected = Some((p, scores, steps, *kind));
                    break;
                }
            }
        }
        if selected.is_none() {
            for (kind, p) in &proposals {
                if joint_first && *kind == "full" {
                    continue;
                }
                if p.model.parameters() == self.accepted.model.parameters() {
                    continue;
                }
                if let Some((p, scores, steps)) = self.repaired_candidate(p)? {
                    selected = Some((p, scores, steps, *kind));
                    break;
                }
            }
        }
        self.state.branch_attempts = attempts;
        if self.state.publication_transfer {
            let (seed, steps, kind) = selected
                .as_ref()
                .map(|(p, _, steps, kind)| (p.clone(), *steps, *kind))
                .unwrap_or_else(|| (self.accepted.clone(), 0, "retained-anchor"));
            let (transferred, summary) = self.transmit(seed, &candidate)?;
            self.state.transfer = summary;
            selected = transferred.and_then(|(p, scores)| {
                // Publishing the same identity would retire its own durable
                // file. A checked no-op is not a new actor or a publication.
                let changed = p
                    .model
                    .parameters()
                    .iter()
                    .zip(self.accepted.model.parameters())
                    .any(|(a, b)| a.to_bits() != b.to_bits());
                changed.then_some((p, scores, steps, kind))
            });
        }
        if let Some((p, scores, steps, kind)) = selected {
            // Validate the registry transaction on a private copy before any
            // actor file or publication anchor is changed.
            let next_book = if let Some(book) = &self.learned_choices {
                let mut next = book.clone();
                let relayed = self.state.transfer["relayed"].as_bool() == Some(true);
                let key = if relayed {
                    self.state.transfer["new_target"].as_str()
                } else {
                    None
                };
                let commit = next.commit_validated(p.identity.clone(), &p.model, key)?;
                let saved = next.checkpoint()?;
                Some((next, saved, serde_json::to_value(commit)?))
            } else {
                None
            };
            if kind == "value" {
                accepted_value_fraction = accepted_fraction;
            }
            if kind == "policy" {
                accepted_value_fraction = 0.;
            }
            self.publish(p, scores[0].clone(), accepted_fraction)?;
            if let Some((book, saved, commit)) = next_book {
                self.learned_choices = Some(book);
                self.state.learned_choices = saved;
                self.state.transfer["registry_commit"] = commit;
            }
            for item in &mut self.state.feedback {
                let s = &scores[item.panel];
                let decisions = (!item.raw_choice_lost || s.raw[item.row])
                    && (!item.coupled_choice_lost || s.coupled[item.row]);
                let value = item.value_error_increase <= ERROR_TOLERANCE
                    || s.errors[item.row] <= item.previous_value_error + ERROR_TOLERANCE;
                if decisions && value {
                    item.retained_publications += 1;
                } else {
                    item.retained_publications = 0;
                }
            }
            self.state
                .feedback
                .retain(|item| item.retained_publications < 2);
            if let Some(v) = &mut self.validation {
                v.score = scores[1].clone();
            }
            if let Some(v) = &mut self.validation {
                v.accepted = self.accepted.clone();
            }
            self.state.last_decision = "accepted-transaction".into();
            self.state.accepted_branch = kind.into();
            self.state.reasons.clear();
            let pre_policy_fraction = if kind == "value" || kind == "retained-anchor" {
                0.
            } else {
                accepted_fraction
            };
            if kind == "retained-anchor" {
                accepted_value_fraction = 0.;
            }
            let nonlinear = self.state.publication_transfer
                && self.state.transfer["changed_seed"].as_bool() == Some(true);
            if nonlinear {
                self.state.accepted_fraction = None;
            }
            self.state.repair = serde_json::json!({"steps":steps,"fraction":if nonlinear {None}else{Some(accepted_fraction)},
                "policy_fraction":if nonlinear {None}else{Some(pre_policy_fraction)},
                "pre_transfer_fraction":accepted_fraction,
                "pre_transfer_policy_fraction":pre_policy_fraction,"value_fraction":accepted_value_fraction,
                "transfer":self.state.transfer,"published_only_after_all_panels_pass":true});
            // Corrective rows leave priority only after two accepted publications
            // retain them. They remain part of the ordinary proof catalogue.
        } else {
            self.state.rejected += 1;
            self.state.last_decision = "rejected-transaction".into();
            self.state.reasons = self.checked_v3(&candidate.model)?.1;
            if self.state.publication_transfer
                && self.state.transfer["fresh_pass"].as_bool() == Some(false)
            {
                self.state
                    .reasons
                    .push("published actor would lose the accepted fresh gain".into());
            }
        }
        self.publication_fresh = None;
        Ok(Some(self.correction_examples()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn score(raw: bool, mass: f64, error: f64, p: Vec<f64>) -> Score {
        Score {
            raw: vec![raw],
            coupled: vec![raw],
            mass,
            value_mse: error,
            priors: Arc::new(vec![p]),
            coupled_logits: Default::default(),
            errors: Arc::new(vec![error]),
        }
    }
    #[test]
    fn confidence_dip_does_not_veto_retained_choices_and_better_value() {
        let old = score(true, 0.8, 0.2, vec![0.8, 0.2]);
        let new = score(true, 0.799, 0.1, vec![0.799, 0.201]);
        assert!(reasons(&old, &new).is_empty());
        assert!(!reasons(&old, &score(false, 0.81, 0.1, vec![0.81, 0.19])).is_empty());
        assert!(!reasons(&old, &score(true, 0.8, 0.3, vec![0.8, 0.2])).is_empty());
        assert!(!reasons(&old, &score(true, 0.99, 0.1, vec![0.99, 0.01])).is_empty());
    }
}

/// Compare the previous guard and the new decision-based guard on identical
/// frozen models and resident panels. Model loading is timed separately.
#[doc(hidden)]
pub fn verify_candidate(
    initial: &Path,
    candidate: &Path,
    manifest: &Path,
    validation: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    fs::create_dir(out)?;
    let started = Instant::now();
    let a = MicroArtifact::load(initial)?;
    let model = a.model()?;
    let first = Arc::new(Snapshot {
        identity: a.identity(),
        artifact: Some(Arc::new(a)),
        model: Arc::new(model.clone()),
        version: 0,
        path: initial.into(),
    });
    let b = MicroArtifact::load(candidate)?;
    let proposed = load_snapshot(candidate, &b.identity(), 1, &model)?;
    let mut guard = Guard::open(
        manifest,
        &out.join("old"),
        16.,
        first.clone(),
        &serde_json::Value::Null,
    )?;
    guard.enable_v2(validation)?;
    let mut corrected = Guard::open(
        manifest,
        &out.join("corrected"),
        16.,
        first.clone(),
        &serde_json::Value::Null,
    )?;
    corrected.enable_v2(validation)?;
    corrected.enable_v3()?;
    let mut previous_order = Guard::open(
        manifest,
        &out.join("previous-order"),
        16.,
        first.clone(),
        &serde_json::Value::Null,
    )?;
    previous_order.enable_v2(validation)?;
    previous_order.enable_v3()?;
    let mut uniform_joint = Guard::open(
        manifest,
        &out.join("uniform-joint"),
        16.,
        first.clone(),
        &serde_json::Value::Null,
    )?;
    uniform_joint.enable_v2(validation)?;
    uniform_joint.enable_v3()?;
    uniform_joint.separate_step_scales = false;
    let mut axes_without_margin = Guard::open(
        manifest,
        &out.join("axes-without-margin"),
        16.,
        first.clone(),
        &serde_json::Value::Null,
    )?;
    axes_without_margin.enable_v2(validation)?;
    axes_without_margin.enable_v3()?;
    axes_without_margin.repair_interpolations = false;
    let mut fixed_margin = Guard::open(
        manifest,
        &out.join("fixed-margin"),
        16.,
        first.clone(),
        &serde_json::Value::Null,
    )?;
    fixed_margin.enable_v2(validation)?;
    fixed_margin.enable_v3()?;
    fixed_margin.adaptive_margin = false;
    let loading = started.elapsed().as_secs_f64();
    // Warm both sides before the timed checks. This explicitly excludes bank
    // loading, successor construction and the first cached value computation.
    let before = corrected.diagnostic_measure(&model)?;
    let cached_margin_verification = cached_margin_verify::run(&corrected, &proposed.model)?;
    let candidate_scores = corrected.diagnostic_measure(&proposed.model)?;
    guard.diagnostic_measure(&proposed.model)?;
    previous_order.diagnostic_measure(&proposed.model)?;
    uniform_joint.diagnostic_measure(&proposed.model)?;
    uniform_joint.consider(proposed.clone(), true)?;
    let uniform_actor = uniform_joint.accepted();
    axes_without_margin.diagnostic_measure(&proposed.model)?;
    axes_without_margin.consider(proposed.clone(), true)?;
    let axes_actor = axes_without_margin.accepted();
    fixed_margin.diagnostic_measure(&proposed.model)?;
    fixed_margin.consider(proposed.clone(), true)?;
    let fixed_margin_actor = fixed_margin.accepted();
    let start = Instant::now();
    previous_order.consider_v3_order(proposed.clone(), true, false)?;
    let previous_order_seconds = start.elapsed().as_secs_f64();
    let previous_order_actor = previous_order.accepted();
    let start = Instant::now();
    guard.consider(proposed.clone(), true)?;
    let old_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    corrected.consider(proposed.clone(), true)?;
    let new_seconds = start.elapsed().as_secs_f64();
    let accepted = corrected.accepted();
    let parallel_abba =
        benchmark_candidate_parallel(first.clone(), proposed.clone(), manifest, validation, out)?;
    let margin_cache_abba =
        benchmark_margin_cache(first.clone(), proposed.clone(), manifest, validation, out)?;
    let adaptive_margin_abba =
        benchmark_margin_mode(first.clone(), proposed.clone(), manifest, validation, out)?;
    let prefilter_abba =
        benchmark_prefilter(first.clone(), proposed.clone(), manifest, validation, out)?;
    let report = serde_json::json!({"adaptive_margin_abba":adaptive_margin_abba,"cached_margin_verification":cached_margin_verification,"margin_cache_abba":margin_cache_abba,"prefilter_abba":prefilter_abba,"before":before,"candidate":candidate_scores,
        "after":corrected.diagnostic_measure(&accepted.model)?,"old":guard.progress(),"corrected":corrected.progress(),
        "accepted_model":corrected.state.accepted_path,"old_seconds":old_seconds,"corrected_seconds":new_seconds,
        "fixed_margin":fixed_margin.progress(),"fixed_margin_scores":fixed_margin.diagnostic_measure(&fixed_margin_actor.model)?,
        "axes_without_margin":axes_without_margin.progress(),"axes_without_margin_scores":axes_without_margin.diagnostic_measure(&axes_actor.model)?,
        "uniform_joint":uniform_joint.progress(),"uniform_joint_scores":uniform_joint.diagnostic_measure(&uniform_actor.model)?,
        "previous_v3_order":previous_order.progress(),"previous_v3_order_seconds":previous_order_seconds,
        "previous_v3_order_scores":previous_order.diagnostic_measure(&previous_order_actor.model)?,
        "policy_changed_parameters":accepted.model.parameters().iter().zip(model.parameters()).enumerate().filter(|(i,(a,b))|!branches::value_parameter(*i) && a.to_bits()!=b.to_bits()).count(),
        "previous_order_policy_changed_parameters":previous_order_actor.model.parameters().iter().zip(model.parameters()).enumerate().filter(|(i,(a,b))|!branches::value_parameter(*i) && a.to_bits()!=b.to_bits()).count(),
        "loading_seconds_excluded":loading,"parallel_abba":parallel_abba,
        "scope":"single frozen candidate; timings are latency observations, not campaign throughput"});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

fn benchmark_candidate_parallel(
    first: Arc<Snapshot>,
    candidate: Arc<Snapshot>,
    manifest: &Path,
    validation: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    let pools = [Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(10).build()?,
    )];
    let mut rows = vec![];
    let mut expected: Option<(Vec<u64>, serde_json::Value)> = None;
    for (i, parallel) in [false, true, true, false].into_iter().enumerate() {
        let mut guard = Guard::open(
            manifest,
            &out.join(format!("abba-{i}")),
            16.,
            first.clone(),
            &serde_json::Value::Null,
        )?;
        guard.enable_v2(validation)?;
        guard.enable_v3()?;
        if parallel {
            guard.enable_parallel(&pools);
        }
        guard.diagnostic_measure(&candidate.model)?;
        let start = Instant::now();
        guard.consider(candidate.clone(), true)?;
        let seconds = start.elapsed().as_secs_f64();
        let actor = guard.accepted();
        let bits = actor
            .model
            .parameters()
            .iter()
            .map(|w| w.to_bits())
            .collect::<Vec<_>>();
        let scores = guard.diagnostic_measure(&actor.model)?;
        if let Some((expected_bits, expected_scores)) = &expected {
            if *expected_bits != bits || *expected_scores != scores {
                return Err(invalid("parallel transaction changes parameters or scores"));
            }
        } else {
            expected = Some((bits, scores));
        }
        rows.push(serde_json::json!({"parallel":parallel,"seconds":seconds,"decision":guard.state.last_decision,
            "fraction":guard.state.accepted_fraction,"identity":actor.identity}));
    }
    Ok(
        serde_json::json!({"runs":rows,"parameters_and_scores_exact":true,"workers":10,
        "scope":"same V3 correction, serial vs ordered parallel checks; no extra actor/search workers"}),
    )
}

fn benchmark_prefilter(
    first: Arc<Snapshot>,
    candidate: Arc<Snapshot>,
    manifest: &Path,
    validation: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    let pools = [Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(10).build()?,
    )];
    let mut rows = vec![];
    let mut expected: Option<(Vec<u64>, serde_json::Value)> = None;
    for (i, fast) in [false, true, true, false].into_iter().enumerate() {
        let mut guard = Guard::open(
            manifest,
            &out.join(format!("prefilter-abba-{i}")),
            16.,
            first.clone(),
            &serde_json::Value::Null,
        )?;
        guard.enable_v2(validation)?;
        guard.enable_v3()?;
        guard.enable_parallel(&pools);
        guard.fast_interpolation = fast;
        // Here value changes at each fraction, so the necessary-condition
        // filter can avoid rebuilding successor values. For fixed-value trials
        // that table is already resident: an extra serial policy pass was slower.
        guard.separate_step_scales = false;
        guard.diagnostic_measure(&candidate.model)?;
        let start = Instant::now();
        guard.consider(candidate.clone(), true)?;
        let seconds = start.elapsed().as_secs_f64();
        let actor = guard.accepted();
        let bits = actor
            .model
            .parameters()
            .iter()
            .map(|w| w.to_bits())
            .collect::<Vec<_>>();
        let scores = guard.diagnostic_measure(&actor.model)?;
        if let Some((expected_bits, expected_scores)) = &expected {
            if *expected_bits != bits || *expected_scores != scores {
                return Err(invalid(
                    "interpolation prefilter changes parameters or scores",
                ));
            }
        } else {
            expected = Some((bits, scores));
        }
        rows.push(serde_json::json!({"prefilter":fast,"seconds":seconds,"identity":actor.identity,"decision":guard.state.last_decision,"repair":guard.state.repair}));
    }
    Ok(
        serde_json::json!({"runs":rows,"parameter_bits_and_scores_exact":true,"workers":10,"same_correction_and_resource_budget":true,"scope":"uniform interpolation fallback; fixed-value trials skip this filter"}),
    )
}

fn benchmark_margin_mode(
    first: Arc<Snapshot>,
    candidate: Arc<Snapshot>,
    manifest: &Path,
    validation: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    let pools = [Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(10).build()?,
    )];
    let mut rows = vec![];
    let mut expected = std::collections::BTreeMap::new();
    for (i, adaptive) in [false, true, true, false].into_iter().enumerate() {
        let mut guard = Guard::open(
            manifest,
            &out.join(format!("adaptive-margin-abba-{i}")),
            16.,
            first.clone(),
            &serde_json::Value::Null,
        )?;
        guard.enable_v2(validation)?;
        guard.enable_v3()?;
        guard.enable_parallel(&pools);
        guard.adaptive_margin = adaptive;
        guard.diagnostic_measure(&candidate.model)?;
        let start = Instant::now();
        guard.consider(candidate.clone(), true)?;
        let seconds = start.elapsed().as_secs_f64();
        let actor = guard.accepted();
        let bits = actor
            .model
            .parameters()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        let scores = guard.diagnostic_measure(&actor.model)?;
        let measured = (bits, scores.clone());
        if let Some(previous) = expected.get(&adaptive) {
            if previous != &measured {
                return Err(invalid("margin mode repeated outcome changed"));
            }
        } else {
            expected.insert(adaptive, measured);
        }
        rows.push(serde_json::json!({"adaptive":adaptive,"seconds":seconds,
            "identity":actor.identity,"scores":scores,"repair":guard.state.repair,
            "decision":guard.state.last_decision}));
    }
    Ok(
        serde_json::json!({"runs":rows,"repeated_mode_outcomes_exact":true,
        "scope":"fixed candidate, same full publication safeguards; algorithms may differ"}),
    )
}

fn benchmark_margin_cache(
    first: Arc<Snapshot>,
    candidate: Arc<Snapshot>,
    manifest: &Path,
    validation: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    let pools = [Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(10).build()?,
    )];
    let mut rows = vec![];
    let mut expected: Option<(Vec<u64>, serde_json::Value)> = None;
    for (i, cached) in [false, true, true, false].into_iter().enumerate() {
        let mut guard = Guard::open(
            manifest,
            &out.join(format!("margin-cache-abba-{i}")),
            16.,
            first.clone(),
            &serde_json::Value::Null,
        )?;
        guard.enable_v2(validation)?;
        guard.enable_v3()?;
        guard.enable_parallel(&pools);
        guard.cached_margin_reads = cached;
        guard.diagnostic_measure(&candidate.model)?;
        let start = Instant::now();
        guard.consider(candidate.clone(), true)?;
        let seconds = start.elapsed().as_secs_f64();
        let actor = guard.accepted();
        let bits = actor
            .model
            .parameters()
            .iter()
            .map(|w| w.to_bits())
            .collect::<Vec<_>>();
        let scores = guard.diagnostic_measure(&actor.model)?;
        if let Some((expected_bits, expected_scores)) = &expected {
            if *expected_bits != bits || *expected_scores != scores {
                return Err(invalid(
                    "margin cache changes accepted parameters or scores",
                ));
            }
        } else {
            expected = Some((bits, scores));
        }
        rows.push(serde_json::json!({"cached_margin_reads":cached,"seconds":seconds,"identity":actor.identity,"decision":guard.state.last_decision,"repair":guard.state.repair}));
    }
    Ok(
        serde_json::json!({"runs":rows,"parameter_bits_and_scores_exact":true,"workers":10,"same_correction_and_resource_budget":true}),
    )
}

#[doc(hidden)]
pub fn verify_consolidation(
    initial: &Path,
    candidate: &Path,
    manifest: &Path,
    fresh: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    fs::create_dir(out)?;
    let a = MicroArtifact::load(initial)?;
    let model = a.model()?;
    let first = Arc::new(Snapshot {
        identity: a.identity(),
        artifact: Some(Arc::new(a)),
        model: Arc::new(model.clone()),
        version: 0,
        path: initial.into(),
    });
    let b = MicroArtifact::load(candidate)?;
    let proposed = load_snapshot(candidate, &b.identity(), 1, &model)?;
    let mut guard = Guard::open(
        manifest,
        &out.join("guard"),
        16.,
        first,
        &serde_json::Value::Null,
    )?;
    guard.expand_immediate_wins()?;
    let saved: Vec<SavedMicroExample> = serde_json::from_slice(&fs::read(fresh)?)?;
    let examples = saved
        .iter()
        .take(32)
        .map(|e| e.example_for_rules(RULES).map(Arc::new))
        .collect::<Result<Vec<_>>>()?;
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(10).build()?);
    let result = protection::benchmark_transaction(
        &model,
        &proposed.model,
        &guard.reference_examples(),
        &examples,
        &[pool],
    )?;
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
