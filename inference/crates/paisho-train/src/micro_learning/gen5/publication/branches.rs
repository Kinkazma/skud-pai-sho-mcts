//! Compose only the independently parameterized V5 trunks, then measure their joint behavior.
use super::*;
pub(super) fn value_parameter(i: usize) -> bool {
    let start = (MICRO_INPUTS + 1) * MICRO_HIDDEN;
    (start..=start + MICRO_HIDDEN).contains(&i) || (MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START).contains(&i)
}
fn reasons(old: &Score, new: &Score) -> Vec<String> {
    let mut why = vec![];
    let lost = old
        .raw
        .iter()
        .zip(&new.raw)
        .zip(old.coupled.iter().zip(&new.coupled))
        .filter(|((a, b), (c, d))| **a && !**b || **c && !**d)
        .count();
    if lost > 0 {
        why.push(format!("{lost} known proof choices lost"));
    }
    if new.mass + 1e-12 < old.mass {
        why.push("mean verified policy mass decreased".into());
    }
    if new.value_mse > old.value_mse + 1e-12 {
        why.push("balanced proof value error increased".into());
    }
    why
}
impl Guard {
    pub fn verify_cached_measurements(&self, models: &[&MicroModel]) -> Result<serde_json::Value> {
        let mut old_seconds = 0.;
        let mut cached_seconds = 0.;
        let mut comparisons = 0;
        for panel in std::iter::once(self).chain(self.validation.as_deref()) {
            for model in models {
                let t = paisho_platform::training_time::now();
                let old = measure_cached(&panel.rows, model, panel.beta, false)?;
                old_seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
                let t = paisho_platform::training_time::now();
                let new = panel.evaluate(model)?;
                cached_seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
                if serde_json::to_value(&old)? != serde_json::to_value(&new)? {
                    return Err(invalid("cached publication differs"));
                }
                comparisons += 1;
            }
        }
        let bytes: usize = std::iter::once(self)
            .chain(self.validation.as_deref())
            .flat_map(|p| &p.rows)
            .map(|r| {
                r.successors.get().map_or(0, |v| {
                    v.len() * 32 + v.iter().map(|(_, s)| s.len() * 8).sum::<usize>()
                })
            })
            .sum();
        Ok(
            serde_json::json!({"comparisons":comparisons,"all_scores_exact":true,"reconstructed_seconds":old_seconds,"cached_seconds":cached_seconds,"immutable_successor_bytes":bytes}),
        )
    }
    pub fn due(&self) -> bool {
        paisho_platform::training_time::elapsed(self.last).as_secs_f64() >= 30.
    }
    pub fn reference_examples(&self) -> Vec<Arc<MicroExample>> {
        self.rows.iter().map(|r| r.example.clone()).collect()
    }
    /// Known corrective/validation rows, not the separately reserved final sources.
    /// Used by V3 consolidation and its isolated value-alignment diagnostic.
    pub fn diagnostic_validation_examples(&self) -> Result<Vec<Arc<MicroExample>>> {
        self.validation.as_deref().map(|p|p.reference_examples())
            .ok_or_else(||invalid("diagnostic requires the secondary validation panel"))
    }
    pub fn enable_v2(&mut self, path: &Path) -> Result<()> {
        if !self.accepted.model.has_deep_value() {
            return Err(invalid(
                "branch publication requires the separated V5 architecture",
            ));
        }
        let hash = sha256(&fs::read(path)?);
        if !self.state.validation_manifest.is_empty() && self.state.validation_manifest != hash {
            return Err(invalid("publication validation changed on resume"));
        }
        let mut validation = Guard::open(
            path,
            &self.out.join("validation"),
            self.beta,
            self.accepted.clone(),
            &serde_json::Value::Null,
        )?;
        let own = self
            .rows
            .iter()
            .map(|r| sha256(&serde_json::to_vec(&r.example.state).unwrap()))
            .collect::<std::collections::HashSet<_>>();
        if validation
            .rows
            .iter()
            .any(|r| own.contains(&sha256(&serde_json::to_vec(&r.example.state).unwrap())))
        {
            return Err(invalid(
                "publication validation overlaps corrective positions",
            ));
        }
        self.expand_immediate_wins()?;
        validation.expand_immediate_wins()?;
        self.state.validation_manifest = hash;
        self.validation = Some(Box::new(validation));
        Ok(())
    }
    pub(super) fn expand_immediate_wins(&mut self) -> Result<()> {
        self.reads=Default::default();
        for row in &mut self.rows {
            let row=Arc::make_mut(row);
            if row.example.value != 1. {
                continue;
            }
            for (i, a) in paisho_core::legal_actions(&row.position).iter().enumerate() {
                if !row.valid[i] {
                    let mut next = row.position.clone();
                    next.apply(*a)?;
                    row.valid[i] = next.outcome() == GameOutcome::Win(row.position.to_move());
                }
            }
            let n = row.valid.iter().filter(|v| **v).count();
            Arc::make_mut(&mut row.example).policy = row
                .valid
                .iter()
                .map(|v| if *v { 1. / n as f64 } else { 0. })
                .collect();
        }
        self.score = measure(&self.rows, &self.accepted.model, self.beta)?;
        Ok(())
    }
    pub(super) fn composed(
        &self,
        policy: &Snapshot,
        value: &Snapshot,
        learner: &Snapshot,
        kind: &str,
    ) -> Result<Arc<Snapshot>> {
        if policy.model.parameters().len()!=value.model.parameters().len(){return Err(invalid("branch composition requires matching model architectures"));}
        let weights = policy
            .model
            .parameters()
            .iter()
            .zip(value.model.parameters())
            .enumerate()
            .map(|(i, (p, v))| if value_parameter(i) { *v } else { *p })
            .collect();
        let mut model = MicroModel::from_parameters(weights).map_err(invalid)?;
        if let Some(bank) = learner.model.sequence_memory() {
            model = model.with_sequence_memory_owned(bank.clone());
        }
        let a = Arc::new(MicroArtifact::new(
            &model,
            learner
                .artifact
                .as_ref()
                .ok_or_else(|| invalid("missing learner artifact"))?
                .updates,
            serde_json::json!({"kind":"gen5-validated-branches-v1","combination":kind,"policy_parent":policy.identity,"value_parent":value.identity,"learner":learner.identity,"version":learner.version,"no_extra_optimizer_step":true}),
        ));
        let identity = a.identity();
        Ok(Arc::new(Snapshot {
            artifact: Some(a),
            model: Arc::new(model),
            version: learner.version,
            path: self
                .out
                .join("accepted")
                .join(format!("model-{identity}.json")),
            identity,
        }))
    }
    fn checked(&self, m: &MicroModel) -> Result<(Score, Vec<String>)> {
        let score = self.evaluate(m)?;
        let mut why = reasons(&self.score, &score);
        // The small focus is cheap. Do not evaluate hundreds of successor sets for
        // a candidate already rejected there.
        if why.is_empty() {
            if let Some(v) = &self.validation {
                let external = v.evaluate(m)?;
                why.extend(
                    reasons(&v.score, &external)
                        .into_iter()
                        .map(|s| format!("validation: {s}")),
                );
            }
        }
        Ok((score, why))
    }
    pub(super) fn consider_branches(
        &mut self,
        candidate: Arc<Snapshot>,
        force: bool,
    ) -> Result<Option<Vec<Arc<MicroExample>>>> {
        if candidate.identity == self.accepted.identity || !force && !self.due() {
            return Ok(None);
        }
        self.last = paisho_platform::training_time::now();
        self.state.checks += 1;
        if candidate.model.parameters() == self.accepted.model.parameters() {
            self.state.last_decision = "unchanged".into();
            self.state.reasons.clear();
            return Ok(Some(vec![]));
        }
        let full_score = self.evaluate(&candidate.model)?;
        let lost = (0..self.rows.len())
            .filter(|&i| {
                (self.score.raw[i] && !full_score.raw[i])
                    || (self.score.coupled[i] && !full_score.coupled[i])
            })
            .collect::<Vec<_>>();
        let proposals = [
            ("full", candidate.clone()),
            (
                "policy",
                self.composed(&candidate, &self.accepted, &candidate, "policy")?,
            ),
            (
                "value",
                self.composed(&self.accepted, &candidate, &candidate, "value")?,
            ),
        ];
        let mut attempts = vec![];
        let mut selected = None;
        for (kind, p) in &proposals {
            // A branch identical to the actor is not a new publication.
            if p.model.parameters() == self.accepted.model.parameters() {
                continue;
            }
            let (score, why) = self.checked(&p.model)?;
            attempts.push(serde_json::json!({"kind":kind,"identity":p.identity,"reasons":why}));
            if why.is_empty() {
                selected = Some((p.clone(), score, 1., *kind));
                break;
            }
        }
        if selected.is_none() {
            for (kind, p) in &proposals {
                let score = self.evaluate(&p.model)?;
                if score.mass + 1e-12 >= self.score.mass
                    && score.value_mse <= self.score.value_mse + 1e-12
                {
                    if let Some((p, score, fraction)) = self.project(p)? {
                        selected = Some((p, score, fraction, *kind));
                        break;
                    }
                }
            }
        }
        self.state.branch_attempts = attempts;
        self.state.reasons = reasons(&self.score, &full_score);
        if let Some((p, score, fraction, kind)) = selected {
            if let Some(v) = &mut self.validation {
                v.score = v.evaluate(&p.model)?;
                v.accepted = p.clone();
            }
            self.publish(p, score, fraction)?;
            if let Some(v)=&mut self.validation {v.accepted=self.accepted.clone();}
            self.state.last_decision = if fraction == 1. {
                "accepted-branch"
            } else {
                "projected-branch"
            }
            .into();
            self.state.accepted_branch = kind.into();
        } else {
            self.state.rejected += 1;
            self.state.last_decision = "rejected".into();
            if self.state.reasons.is_empty() {
                self.state
                    .reasons
                    .push("separate validation rejected the candidate".into());
            }
        }
        // Lost choices receive half of focus mass; remaining wins share the rest.
        // Value feedback is supplied independently by balanced recall, in its own quota.
        let mut focus = vec![];
        for i in lost.iter().copied().chain(0..self.rows.len()) {
            if self.rows[i].example.value == 1.
                && !focus
                    .iter()
                    .any(|e: &Arc<MicroExample>| Arc::ptr_eq(e, &self.rows[i].example))
            {
                focus.push(self.rows[i].example.clone());
            }
            if focus.len() == 32 {
                break;
            }
        }
        if !lost.is_empty() {
            let important = lost
                .iter()
                .filter(|&&i| self.rows[i].example.value == 1.)
                .count();
            if important > 0 && important < focus.len() {
                let extra = focus.len().saturating_sub(2 * important);
                for i in 0..extra {
                    focus.push(focus[i % important].clone());
                }
            }
        }
        Ok(Some(focus))
    }
}
