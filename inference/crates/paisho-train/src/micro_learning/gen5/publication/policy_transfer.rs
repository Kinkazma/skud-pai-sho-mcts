//! Bounded private transfer of acquired decisions. The complete Guard is final authority.
use super::gain_transfer_probe::{fresh_loss, value_equal};
use super::*;
use std::collections::BTreeSet;
// Small strict interior, including finite curvature of the policy reader.
// The final native decision tests retain their original, exact criterion.
pub(super) const EPS: f64 = 1e-5;
pub(super) const STEPS: usize = 4;
pub(super) const HALVES: usize = 6;
pub(super) const CHECKS: usize = 2;
#[derive(Clone)]
pub(super) struct OldMargin {
    row: Arc<Witness>,
    offset: Vec<f64>,
    label: String,
    // The competitor that actually broke the finite check. It can differ from
    // the largest competitor at the current (still safe) linearization point.
    competitor: Option<usize>,
}
#[derive(Clone)]
pub(super) struct FreshLimit {
    pub(super) rows: Vec<Arc<MicroExample>>,
    pub(super) ceiling: f64,
}
impl FreshLimit {
    pub(super) fn loss(&self, model: &MicroModel, guard: &Guard) -> Result<f64> {
        fresh_loss(model, &self.rows, guard.parallel.as_ref().unwrap())
    }
    fn gradient(&self, model: &MicroModel, guard: &Guard) -> Result<Vec<f64>> {
        let mut sum = vec![0.; model.parameters().len()];
        let m = model.clone();
        for chunk in self.rows.chunks(16) {
            let m = m.clone();
            let parts = guard.parallel.as_ref().unwrap().map_owned(
                chunk.to_vec(),
                |e| e.actions.len(),
                move |e| m.loss_gradient(e).map(|(_, g)| g),
            );
            for part in parts {
                for (i, (total, g)) in sum.iter_mut().zip(part.map_err(invalid)?).enumerate() {
                    if !branches::value_parameter(i) {
                        *total += g / self.rows.len() as f64;
                    }
                }
            }
        }
        Ok(sum)
    }
}
#[derive(Clone, Serialize)]
pub(super) struct Reading {
    pub(super) good: usize,
    pub(super) bad: Option<usize>,
    pub(super) gap: f64,
    pub(super) mass: f64,
    pub(super) raw: bool,
}
pub(super) fn best(p: &[f64], valid: impl Fn(usize) -> bool) -> Option<usize> {
    (0..p.len())
        .filter(|&i| valid(i))
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
}
pub(super) fn raw(model: &MicroModel, row: &Witness) -> Result<Vec<f64>> {
    let base = micro_softmax(&MicroModel::logits(
        &model.embed(&row.example.state),
        &row.example.actions,
    ))
    .map_err(invalid)?;
    model
        .memory_priors(&row.example.state, &row.example.actions, &base, 0)
        .map_err(invalid)
}
pub(super) fn reading(priors: &[f64], valid: &[bool], offset: &[f64]) -> Result<Reading> {
    if priors.len() != valid.len()
        || priors.is_empty()
        || (!offset.is_empty() && offset.len() != valid.len())
        || priors.iter().any(|p| !p.is_finite() || *p < 0.)
        || offset.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("relay probability/support/offset shape"));
    }
    let logits: Vec<_> = priors
        .iter()
        .enumerate()
        .map(|(i, p)| p.max(1e-300).ln() + offset.get(i).copied().unwrap_or(0.))
        .collect();
    let good = best(&logits, |i| valid[i]).ok_or_else(|| invalid("relay support empty"))?;
    let bad = best(&logits, |i| !valid[i]);
    if priors[good] <= 1e-300 || bad.is_some_and(|i| priors[i] <= 1e-300) {
        return Err(invalid("relay clamped decision margin"));
    }
    Ok(Reading {
        good,
        bad,
        gap: bad.map_or(-1., |i| logits[i] - logits[good]),
        mass: priors
            .iter()
            .zip(valid)
            .filter(|(_, v)| **v)
            .map(|(p, _)| p)
            .sum(),
        raw: valid[best(priors, |_| true).unwrap()],
    })
}
pub(super) fn readings(model: &MicroModel, rows: &[Arc<Witness>]) -> Result<Vec<Reading>> {
    rows.iter()
        .map(|r| reading(&raw(model, r)?, &r.valid, &[]))
        .collect()
}
pub(super) fn retained(old: &[Reading], new: &[Reading]) -> bool {
    old.len() == new.len() && old.iter().zip(new).all(|(a, b)| !a.raw || b.raw)
}
pub(super) fn top_two(items: Vec<(OldMargin, Reading)>) -> Vec<OldMargin> {
    let mut indexed = items
        .into_iter()
        .enumerate()
        .filter(|(_, (_, r))| r.bad.is_some())
        .collect::<Vec<_>>();
    indexed.sort_by(|(ia, (_, a)), (ib, (_, b))| b.gap.total_cmp(&a.gap).then_with(|| ia.cmp(ib)));
    indexed
        .into_iter()
        .take(2)
        .map(|(_, (row, _))| row)
        .collect()
}
// Only called at initialization and after a paid complete candidate check. The
// selected sources remain fixed between checks; their derivatives do not.
pub(super) fn critical(
    guard: &Guard,
    model: &MicroModel,
    scores: &[Score],
    cohort: &[Arc<Witness>],
    old64: &[Reading],
) -> Result<Vec<OldMargin>> {
    let mut items = vec![];
    for (pi, (panel, score)) in std::iter::once(guard)
        .chain(guard.validation.as_deref())
        .zip(scores)
        .enumerate()
    {
        panel.reads.prefill(
            score
                .coupled_logits
                .iter()
                .enumerate()
                .filter_map(|(i, l)| panel.score.coupled[i].then_some(l)),
            panel.parallel.as_ref(),
        );
        for (i, row) in panel.rows.iter().enumerate() {
            if panel.score.raw[i] {
                items.push((
                    OldMargin {
                        row: row.clone(),
                        offset: vec![],
                        label: format!("panel{pi}/raw/{i}"),
                        competitor: None,
                    },
                    reading(&score.priors[i], &row.valid, &[])?,
                ));
            }
            if panel.score.coupled[i] {
                // Store the exact native V offsets, not (coupled-lograw), whose
                // subtraction can introduce a second rounding at later trials.
                let offsets = score.coupled_logits[i].diagnostic_frozen_value_offsets()?;
                let r = reading(&score.priors[i], &row.valid, &offsets)?;
                items.push((
                    OldMargin {
                        row: row.clone(),
                        offset: offsets,
                        label: format!("panel{pi}/coupled/{i}"),
                        competitor: None,
                    },
                    r,
                ));
            }
        }
    }
    for (i, (row, old)) in cohort.iter().zip(old64).enumerate() {
        if old.raw {
            items.push((
                OldMargin {
                    row: row.clone(),
                    offset: vec![],
                    label: format!("cohort/raw/{i}"),
                    competitor: None,
                },
                reading(&raw(model, row)?, &row.valid, &[])?,
            ));
        }
    }
    Ok(top_two(items))
}
pub(super) fn old_constraints(
    model: &MicroModel,
    active: &[OldMargin],
) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<serde_json::Value>)> {
    let mut gs = vec![];
    let mut rhs = vec![];
    let mut trace = vec![];
    for a in active {
        let p = raw(model, &a.row)?;
        let mut r = reading(&p, &a.row.valid, &a.offset)?;
        if let Some(bad) = a.competitor {
            if bad >= p.len() || a.row.valid[bad] || p[bad] <= 1e-300 {
                return Err(invalid("invalid finite counterexample competitor"));
            }
            r.bad = Some(bad);
            r.gap = p[bad].ln() + a.offset.get(bad).copied().unwrap_or(0.)
                - p[r.good].ln()
                - a.offset.get(r.good).copied().unwrap_or(0.);
        }
        if let Some(bad) = r.bad {
            gs.push(gain_probe::margin_gradient(
                model,
                &a.row.example,
                r.good,
                bad,
            )?);
            rhs.push(r.gap + EPS);
            trace.push(serde_json::json!({"source":a.label,"reading":r,"rhs":r.gap+EPS}));
        }
    }
    Ok((gs, rhs, trace))
}
pub(super) fn local_ok(
    before: &Reading,
    after: &Reading,
    old: &[Reading],
    new: &[Reading],
) -> bool {
    retained(old, new)
        && after.mass >= before.mass
        && after.gap.is_finite()
        && after.gap < before.gap
}
pub(super) fn active_merit(model: &MicroModel, active: &[OldMargin]) -> Result<f64> {
    let mut sum = 0.;
    for a in active {
        let r = reading(&raw(model, &a.row)?, &a.row.valid, &a.offset)?;
        sum += (r.gap + EPS).max(0.);
    }
    Ok(sum)
}
pub(super) fn active_choices_retained(model: &MicroModel, active: &[OldMargin]) -> Result<bool> {
    for a in active {
        let r = reading(&raw(model, &a.row)?, &a.row.valid, &a.offset)?;
        if r.bad
            .is_some_and(|bad| r.gap > 0. || r.gap == 0. && bad < r.good)
        {
            return Ok(false);
        }
    }
    Ok(true)
}
pub(super) fn acquired_repair_ok(
    old: &[Reading],
    new: &[Reading],
    target: usize,
    before: f64,
    after: f64,
) -> bool {
    retained(old, new)
        && new[target].raw
        && before.is_finite()
        && after.is_finite()
        && after < before
}
pub(super) fn kl_constraint(panel: &Guard, model: &MicroModel) -> Result<(Vec<f64>, f64, usize)> {
    let mut score = panel.score.clone();
    let m = model.clone();
    let read = move |w: &Arc<Witness>| {
        if w.example.value == 1. {
            raw(&m, w).map_err(|e| e.to_string())
        } else {
            Ok(vec![])
        }
    };
    let rows = if let Some(pool) = &panel.parallel {
        pool.map_owned(panel.rows.clone(), |r| r.example.actions.len(), read)
    } else {
        panel.rows.iter().map(read).collect()
    };
    score.priors = Arc::new(
        rows.into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(invalid)?,
    );
    let shift = repair::kl(&panel.score, &score);
    let (gradient, calls) = gain_probe::kl_gradient(panel, model, &score)?;
    // Aim inside the SAME final bound to leave room for finite curvature.
    Ok((gradient, shift - repair::MAX_KL * 0.8, calls))
}
pub(super) fn full_admission(
    reasons: &[String],
    old: &[Reading],
    new: &[Reading],
    target: usize,
) -> bool {
    reasons.is_empty() && retained(old, new) && new.get(target).is_some_and(|r| r.raw)
}
pub(super) fn prefilter_all(
    guard: &Guard,
    model: &MicroModel,
) -> Result<Vec<prefilter::PanelResult>> {
    std::iter::once(guard)
        .chain(guard.validation.as_deref())
        .map(|p| {
            prefilter::check_ordered(
                &p.rows,
                &p.score,
                model,
                repair::MAX_KL,
                p.parallel.as_ref(),
            )
        })
        .collect()
}
pub(super) fn frozen_coupled_rows(guard: &Guard, scores: &[Score]) -> Result<Vec<OldMargin>> {
    let mut out = vec![];
    for (pi, (panel, score)) in std::iter::once(guard)
        .chain(guard.validation.as_deref())
        .zip(scores)
        .enumerate()
    {
        for (i, row) in panel.rows.iter().enumerate() {
            if panel.score.coupled[i] {
                out.push(OldMargin {
                    row: row.clone(),
                    offset: score.coupled_logits[i].diagnostic_frozen_value_offsets()?,
                    label: format!("panel{pi}/coupled/{i}"),
                    competitor: None,
                });
            }
        }
    }
    Ok(out)
}
pub(super) fn lost_coupled(
    guard: &Guard,
    model: &MicroModel,
    frozen: &[OldMargin],
) -> Result<Vec<OldMargin>> {
    let model = model.clone();
    let read = move |a: &OldMargin| -> std::result::Result<Option<OldMargin>, String> {
        let r = raw(&model, &a.row)
            .and_then(|p| reading(&p, &a.row.valid, &a.offset))
            .map_err(|e| e.to_string())?;
        if r.bad
            .is_some_and(|bad| r.gap > 0. || r.gap == 0. && bad < r.good)
        {
            let mut a = a.clone();
            a.competitor = r.bad;
            Ok(Some(a))
        } else {
            Ok(None)
        }
    };
    let rows = guard.parallel.as_ref().unwrap().map_owned(
        frozen.to_vec(),
        |a| a.row.example.actions.len(),
        read,
    );
    Ok(rows
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(invalid)?
        .into_iter()
        .flatten()
        .collect())
}
pub(super) fn relay(
    guard: &Guard,
    start: &MicroModel,
    cohort: &[Arc<Witness>],
    before: &[Reading],
    target: usize,
    adaptive_counterexamples: bool,
    adaptive_kl: bool,
    max_corrections: usize,
    fresh: Option<&FreshLimit>,
) -> Result<(MicroModel, serde_json::Value)> {
    let begin = Instant::now();
    let unchanged = guard.progress();
    let preparation = Instant::now();
    let (start_scores, why) = guard.checked_v3(start)?;
    if !why.is_empty() {
        return Err(invalid(format!(
            "published relay seed fails its PRE-boundary Guard: {why:?}"
        )));
    }
    let mut active = critical(guard, start, &start_scores, cohort, before)?;
    let frozen_coupled = if adaptive_counterexamples {
        frozen_coupled_rows(guard, &start_scores)?
    } else {
        vec![]
    };
    let preparation_seconds = preparation.elapsed().as_secs_f64();
    let mut current = start.clone();
    let mut current_rows = before.to_vec();
    let mut checks = 0;
    let mut corrections = 0;
    let mut attempts = vec![];
    let mut full_checks = vec![];
    let mut gradient_seconds = 0.;
    let mut prefilter_seconds = 0.;
    let mut final_model = start.clone();
    let mut accepted = false;
    let mut stop = None::<String>;
    let mut kl_active = BTreeSet::new();
    let mut kl_gradient_calls = 0;
    let mut current_fresh = fresh.map(|f| f.loss(&current, guard)).transpose()?;
    let mut fresh_active = fresh
        .zip(current_fresh)
        .is_some_and(|(f, l)| l > f.ceiling + 1e-12);
    let mut fresh_gradient_rows = 0;
    for step in 0..max_corrections {
        let active_before = active.len();
        let kl_before = kl_active.len();
        let fresh_before = fresh_active;
        let t = Instant::now();
        let goal = &current_rows[target];
        let Some(bad) = goal.bad else {
            stop = Some("target has no competing action".into());
            break;
        };
        let gradient =
            match gain_probe::margin_gradient(&current, &cohort[target].example, goal.good, bad) {
                Ok(v) => v,
                Err(e) => {
                    stop = Some(e.to_string());
                    break;
                }
            };
        let Some(direct) = margin_step::diagnostic_displacement(&gradient, goal.gap.max(0.)) else {
            stop = Some("target direction unavailable".into());
            break;
        };
        let (mut refs, mut rhs, mut selected) = match old_constraints(&current, &active) {
            Ok(v) => v,
            Err(e) => {
                stop = Some(e.to_string());
                break;
            }
        };
        // The new decision is an active halfspace, not merely an objective
        // that projection onto old constraints can silently erase.
        refs.push(gradient.clone());
        rhs.push(goal.gap + EPS);
        selected.push(serde_json::json!({"source":"new-proof/raw","target":target,"reading":goal,"rhs":goal.gap+EPS}));
        for (pi, panel) in std::iter::once(guard)
            .chain(guard.validation.as_deref())
            .enumerate()
        {
            if kl_active.contains(&pi) {
                let (g, bound, calls) = kl_constraint(panel, &current)?;
                refs.push(g);
                rhs.push(bound);
                kl_gradient_calls += calls;
                selected.push(serde_json::json!({"source":format!("panel{pi}/kl"),"rhs":bound,"gradient_rows":calls}));
            }
        }
        if fresh_active {
            let f = fresh.unwrap();
            refs.push(f.gradient(&current, guard)?);
            fresh_gradient_rows += f.rows.len();
            let bound = current_fresh.unwrap() - f.ceiling + 1e-8;
            rhs.push(bound);
            selected.push(serde_json::json!({"source":"fresh64","rhs":bound,"ceiling":f.ceiling}));
        }
        let Some(direction) = super::super::protection::diagnostic_affine(&direct, &refs, &rhs)
        else {
            stop = Some("old margin projection infeasible".into());
            break;
        };
        gradient_seconds += t.elapsed().as_secs_f64();
        corrections += 1;
        let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
        // Scale an oversized projection only in the same bounded line search;
        // finite prefilter and final Guard remain authoritative.
        if !norm.is_finite() || norm == 0. {
            stop = Some("invalid projected displacement".into());
            break;
        }
        let norm_scale = (0.02 / norm).min(1.);
        let repairing_acquired = adaptive_counterexamples && current_rows[target].raw;
        let merit_rows = active.clone();
        let fresh_merit =
            |v: Option<f64>| fresh.zip(v).map_or(0., |(f, l)| (l - f.ceiling).max(0.));
        let old_merit = if repairing_acquired {
            active_merit(&current, &merit_rows)? + fresh_merit(current_fresh)
        } else {
            0.
        };
        let mut next = None;
        for half in 0..HALVES {
            let scale = norm_scale * 0.5_f64.powi(half as i32);
            let trial = match gain_probe::shifted(&current, &direction, scale) {
                Ok(v) => v,
                Err(e) => {
                    attempts.push(serde_json::json!({"step":step,"half":half,"error":e.to_string(),"local_accepted":false}));
                    continue;
                }
            };
            if !value_equal(start, &trial) {
                return Err(invalid("relay changed frozen value bits"));
            }
            let t = Instant::now();
            let filtered = prefilter_all(guard, &trial);
            prefilter_seconds += t.elapsed().as_secs_f64();
            let filter = match filtered {
                Ok(v) => v,
                Err(e) => {
                    attempts.push(serde_json::json!({"step":step,"half":half,"error":e.to_string(),"local_accepted":false}));
                    continue;
                }
            };
            if filter.iter().any(|r| r.reject) {
                if adaptive_counterexamples {
                    for (panel_index, (panel, result)) in std::iter::once(guard)
                        .chain(guard.validation.as_deref())
                        .zip(&filter)
                        .enumerate()
                    {
                        if adaptive_kl
                            && result
                                .policy_kl
                                .is_some_and(|k| k.is_finite() && k > repair::MAX_KL)
                        {
                            kl_active.insert(panel_index);
                        }
                        if let Some(row_index) = result.raw_choice_lost {
                            add_counterexample(
                                &mut active,
                                OldMargin {
                                    row: panel.rows[row_index].clone(),
                                    offset: vec![],
                                    label: format!("panel{panel_index}/raw/{row_index}"),
                                    competitor: reading(
                                        &raw(&trial, &panel.rows[row_index])?,
                                        &panel.rows[row_index].valid,
                                        &[],
                                    )?
                                    .bad,
                                },
                            );
                        }
                    }
                }
                attempts.push(serde_json::json!({"step":step,"half":half,"scale":scale,"prefilter":filter,"local_accepted":false,"selected":selected}));
                continue;
            }
            let rows = match readings(&trial, cohort) {
                Ok(v) => v,
                Err(e) => {
                    attempts.push(serde_json::json!({"step":step,"half":half,"error":e.to_string(),"local_accepted":false}));
                    continue;
                }
            };
            let trial_fresh = fresh.map(|f| f.loss(&trial, guard)).transpose()?;
            fresh_active |= fresh
                .zip(trial_fresh)
                .is_some_and(|(f, l)| l > f.ceiling + 1e-12);
            let coupled_lost = if adaptive_counterexamples {
                lost_coupled(guard, &trial, &frozen_coupled)?
            } else {
                vec![]
            };
            for row in &coupled_lost {
                add_counterexample(&mut active, row.clone());
            }
            let new_merit = if repairing_acquired {
                Some(active_merit(&trial, &merit_rows)? + fresh_merit(trial_fresh))
            } else {
                None
            };
            let ok = if repairing_acquired {
                acquired_repair_ok(before, &rows, target, old_merit, new_merit.unwrap())
            } else {
                local_ok(&current_rows[target], &rows[target], before, &rows)
            };
            if adaptive_counterexamples {
                for (index, (old, new)) in before.iter().zip(&rows).enumerate() {
                    if old.raw && !new.raw {
                        add_counterexample(
                            &mut active,
                            OldMargin {
                                row: cohort[index].clone(),
                                offset: vec![],
                                label: format!("cohort/raw/{index}"),
                                competitor: new.bad,
                            },
                        );
                    }
                }
            }
            attempts.push(serde_json::json!({"step":step,"half":half,"scale":scale,"norm":norm*scale,
                "prefilter":filter,"local_accepted":ok,"goal":rows[target],"old64_retained":retained(before,&rows),"selected":selected,"value_bits_equal_start":true,
                "repairing_acquired":repairing_acquired,"old_active_merit":old_merit,"new_active_merit":new_merit}));
            if let Some(record) = attempts.last_mut() {
                record["fresh_loss"] = serde_json::json!(trial_fresh);
                record["coupled_lost"] =
                    serde_json::json!(coupled_lost.iter().map(|r| &r.label).collect::<Vec<_>>());
            }
            if ok {
                next = Some((trial, rows, trial_fresh, coupled_lost.is_empty()));
                break;
            }
        }
        let Some((trial, rows, trial_fresh, coupled_retained)) = next else {
            // A failed finite trial identifies which constraint matters along
            // this direction. Re-solve at the SAME weights with that witness,
            // within the original correction budget; do not merely halve an
            // incompatible direction until all useful progress disappears.
            if adaptive_counterexamples
                && (active.len() > active_before
                    || kl_active.len() > kl_before
                    || fresh_active && !fresh_before)
            {
                continue;
            }
            stop = Some("no finite local improvement in six trials".into());
            break;
        };
        current = trial;
        current_rows = rows;
        current_fresh = trial_fresh;
        // A complete test is spent only after a raw acquisition or on the last
        // correction. A rejected first check can update the active margins.
        let known_valid = coupled_retained
            && (!adaptive_counterexamples || active_choices_retained(&current, &active)?);
        let fresh_valid = fresh
            .zip(current_fresh)
            .map_or(true, |(f, l)| l <= f.ceiling + 1e-12);
        if current_rows[target].raw && known_valid && fresh_valid && checks < CHECKS {
            let t = Instant::now();
            checks += 1;
            let (scores, reasons) = match guard.checked_v3(&current) {
                Ok(v) => v,
                Err(e) => {
                    full_checks.push(serde_json::json!({"number":checks,"step":step,"seconds":t.elapsed().as_secs_f64(),"error":e.to_string(),"accepted":false}));
                    stop =
                        Some("complete candidate check failed numerically; seed retained".into());
                    break;
                }
            };
            let ok = full_admission(&reasons, before, &current_rows, target);
            full_checks.push(serde_json::json!({"number":checks,"step":step,"seconds":t.elapsed().as_secs_f64(),"reasons":reasons,
                "goal":current_rows[target],"old64_retained":retained(before,&current_rows),"accepted":ok,
                "kl":std::iter::once(guard).chain(guard.validation.as_deref()).zip(&scores).map(|(p,s)|repair::kl(&p.score,s)).collect::<Vec<_>>() }));
            if ok {
                final_model = current.clone();
                accepted = true;
                break;
            }
            if checks == CHECKS {
                stop = Some("two complete candidate checks exhausted".into());
                break;
            }
            let refreshed = match critical(guard, &current, &scores, cohort, before) {
                Ok(v) => v,
                Err(e) => {
                    stop = Some(format!("critical margin unavailable: {e}"));
                    break;
                }
            };
            if adaptive_counterexamples {
                for row in refreshed {
                    add_counterexample(&mut active, row);
                }
            } else {
                active = refreshed;
            }
        }
    }
    if guard.progress() != unchanged || !value_equal(start, &final_model) {
        return Err(invalid("relay mutated Guard/frozen V"));
    }
    let value_count = start
        .parameters()
        .iter()
        .enumerate()
        .filter(|(i, _)| branches::value_parameter(*i))
        .count();
    let after = if accepted {
        current_rows
    } else {
        before.to_vec()
    };
    Ok((
        final_model,
        serde_json::json!({"accepted":accepted,"target":target,"before":before,"after":after,"goal_before":before[target],"goal_after":after[target],
        "candidate_full_checks":checks,"max_candidate_full_checks":CHECKS,"seed_full_checks":1,"corrections":corrections,"max_corrections":max_corrections,"max_backtracks":HALVES,
        "preparation_seconds":preparation_seconds,"gradient_seconds":gradient_seconds,"prefilter_seconds":prefilter_seconds,
        "total_seconds":begin.elapsed().as_secs_f64(),"attempts":attempts,"full_checks":full_checks,"stop":stop,
        "value_parameters_compared":value_count,"final_value_bit_mismatches":0,"fallback_exact_published_actor":!accepted,
        "guard_anchor_unchanged":true,"fresh_loss_is_not_admission":fresh.is_none(),
        "adaptive_counterexamples":adaptive_counterexamples,"adaptive_kl":adaptive_kl,"kl_gradient_rows":kl_gradient_calls,
        "fresh_ceiling":fresh.map(|f|f.ceiling),"fresh_final_private":current_fresh,"fresh_gradient_rows":fresh_gradient_rows,
        "active_constraints_final":active.iter().map(|a|serde_json::json!({"label":a.label,"competitor":a.competitor})).collect::<Vec<_>>()}),
    ))
}
// Bound both gradient work and resident memory. Full finite checks still cover
// EVERY old witness, including those outside this active linearized subset.
pub(super) fn add_counterexample(active: &mut Vec<OldMargin>, row: OldMargin) {
    if active.len() < 10
        && !active
            .iter()
            .any(|old| old.label == row.label && old.competitor == row.competitor)
    {
        active.push(row);
    }
}
