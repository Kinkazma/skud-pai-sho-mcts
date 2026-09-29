//! Bounded linearized policy-margin repair of a private candidate.
//! This module cannot publish. Guard's full checks remain the only authority.
use super::*;

// Keep the previous fixed-step implementation's largest possible displacement:
// 0.002 * min(||g||, 10) <= 0.02. Adapt the smaller steps to the actual gap.
const MAX_DISPLACEMENT: f64 = 0.02;
const INTERIOR_MARGIN: f64 = 1e-6;
const MAX_BACKTRACKS: usize = 6;

pub(super) struct Assessment<T> {
    /// Mean hinge over a FIXED set of all previously protected decisions.
    pub merit: f64,
    /// True only after the caller's complete, unchanged publication checks.
    pub fully_valid: bool,
    pub evidence: T,
}

pub(super) struct PolicyStep<T> {
    pub model: MicroModel,
    pub assessment: Assessment<T>,
    pub trials: usize,
    pub displacement_norm: f64,
    pub gradient_norm: f64,
}

struct Direction {
    unit: Vec<f64>,
    norm: f64,
    displacement: f64,
}

fn direction(gradient: &[f64], gap: f64, frozen: impl Fn(usize) -> bool) -> Option<Direction> {
    if !gap.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let largest = gradient.iter().enumerate().filter(|(i, _)| !frozen(*i))
        .map(|(_, g)| g.abs()).fold(0., f64::max);
    if largest == 0. { return None; }
    // Scaled L2 avoids squaring tiny gradients to zero or large ones to infinity.
    let mut unit: Vec<_> = gradient.iter().enumerate()
        .map(|(i, g)| if frozen(i) { 0. } else { g / largest }).collect();
    let scaled_norm = unit.iter().map(|g| g * g).sum::<f64>().sqrt();
    let norm = largest * scaled_norm;
    if !norm.is_finite() || norm == 0. { return None; }
    for g in &mut unit { *g /= scaled_norm; }
    // alpha = (gap + interior margin) / ||g||², expressed as a displacement
    // along a unit vector. Infinity from a tiny norm is safely capped here.
    let displacement = ((gap.max(0.) + INTERIOR_MARGIN) / norm).min(MAX_DISPLACEMENT);
    if !displacement.is_finite() || displacement <= 0. { return None; }
    Some(Direction { unit, norm, displacement })
}

/// Reuse the exact bounded direction in an isolated, stricter diagnostic loop.
pub(super) fn diagnostic_displacement(gradient:&[f64], gap:f64)->Option<Vec<f64>> {
    let plan=direction(gradient,gap,branches::value_parameter)?;
    Some(plan.unit.iter().map(|u|u*plan.displacement).collect())
}

fn shifted(weights: &[f64], unit: &[f64], distance: f64, frozen: impl Fn(usize) -> bool) -> Vec<f64> {
    weights.iter().zip(unit).enumerate().map(|(i, (w, g))| {
        // Copy the frozen bits, rather than subtracting a possibly signed zero.
        if frozen(i) { *w } else { w - distance * g }
    }).collect()
}

fn backtrack<T>(
    distance: f64,
    before_merit: f64,
    mut evaluate: impl FnMut(f64) -> Result<Option<Assessment<T>>>,
) -> Result<Option<(Assessment<T>, usize, f64)>> {
    if !before_merit.is_finite() || before_merit < 0. { return Ok(None); }
    for halve in 0..MAX_BACKTRACKS {
        let step = distance * 0.5_f64.powi(halve as i32);
        let Some(score) = evaluate(step)? else { continue; };
        if score.merit.is_finite() && score.merit >= 0.
            && (score.fully_valid || score.merit < before_merit)
        {
            return Ok(Some((score, halve + 1, step)));
        }
    }
    Ok(None)
}

/// Return a private improvement, never permission to publish it. The callback
/// measures both panels and sets `fully_valid` only when every final test passes.
/// An improving but still invalid candidate may continue the bounded repair loop.
pub(super) fn policy_step<T>(
    model: &MicroModel,
    gradient: &[f64],
    gap: f64,
    before_merit: f64,
    mut evaluate: impl FnMut(&MicroModel) -> Result<Assessment<T>>,
) -> Result<Option<PolicyStep<T>>> {
    if gradient.len() != model.parameters().len() || !model.has_deep_value() {
        return Err(invalid("policy margin step requires matching separated value architecture"));
    }
    let Some(plan) = direction(gradient, gap, branches::value_parameter) else { return Ok(None); };
    let found = backtrack(plan.displacement, before_merit, |distance| {
        let weights = shifted(model.parameters(), &plan.unit, distance, branches::value_parameter);
        let Ok(mut next) = MicroModel::from_parameters(weights) else { return Ok(None); };
        if let Some(bank) = model.sequence_memory() {
            next = next.with_sequence_memory_owned(bank.clone());
        }
        let assessment = evaluate(&next)?;
        Ok(Some(Assessment {
            merit: assessment.merit,
            fully_valid: assessment.fully_valid,
            evidence: (next, assessment.evidence),
        }))
    })?;
    Ok(found.map(|(score, trials, displacement_norm)| PolicyStep {
        model: score.evidence.0,
        assessment: Assessment { merit: score.merit, fully_valid: score.fully_valid, evidence: score.evidence.1 },
        trials, displacement_norm, gradient_norm: plan.norm,
    }))
}

fn hinge(logits: &[f64], valid: &[bool]) -> Result<f64> {
    if logits.len() != valid.len() || logits.iter().any(|x| !x.is_finite()) {
        return Err(invalid("invalid measured logits in margin merit"));
    }
    let good = logits.iter().zip(valid).filter(|(_, v)| **v).map(|(x, _)| *x).reduce(f64::max)
        .ok_or_else(|| invalid("protected margin has no valid action"))?;
    let bad = logits.iter().zip(valid).filter(|(_, v)| !**v).map(|(x, _)| *x).reduce(f64::max);
    Ok(bad.map_or(0., |b| (b - good + INTERIOR_MARGIN).max(0.)))
}

/// Include newly lost decisions as well as original failures. Averaging only the
/// currently failed rows would change the denominator when an easy row is fixed,
/// and could label a genuine decrease as an increase (or the reverse).
pub(super) fn merit(guard: &Guard, scores: &[Score]) -> Result<f64> {
    let panels: Vec<_> = std::iter::once(guard).chain(guard.validation.as_deref()).collect();
    if panels.len() != scores.len() { return Err(invalid("margin merit panel count changed")); }
    let mut total = 0.;
    let mut count = 0usize;
    for (panel, score) in panels.into_iter().zip(scores) {
        if score.priors.len() != panel.rows.len() || score.coupled_logits.len() != panel.rows.len() {
            return Err(invalid("margin merit requires complete measured scores"));
        }
        // The serial loop below consumes exactly these complete coefficient
        // vectors. Proven zero hinges retain their absent successor values.
        panel.reads.prefill(
            panel.rows.iter().enumerate().filter_map(|(i, row)| {
                let logits = &score.coupled_logits[i];
                (panel.score.coupled[i] && !logits.zero_hinge(&row.valid, INTERIOR_MARGIN))
                    .then_some(logits)
            }),
            panel.parallel.as_ref(),
        );
        for (i, row) in panel.rows.iter().enumerate() {
            if panel.score.raw[i] {
                if score.priors[i].iter().any(|p| !p.is_finite() || *p < 0.) {
                    return Err(invalid("invalid measured prior in margin merit"));
                }
                let logits: Vec<_> = score.priors[i].iter().map(|p| p.max(1e-300).ln()).collect();
                total += hinge(&logits, &row.valid)?;
                count += 1;
            }
            if panel.score.coupled[i] {
                total += if score.coupled_logits[i].zero_hinge(&row.valid,INTERIOR_MARGIN) {0.}
                    else {hinge(&score.coupled_logits[i], &row.valid)?};
                count += 1;
            }
        }
    }
    Ok(total / count.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_gradient_needs_gap_scaled_step_not_eight_fixed_steps() {
        let gap = 0.001;
        let gradient = [0.1];
        let remaining_fixed = gap - 8. * 0.002 * gradient[0] * gradient[0];
        assert!(remaining_fixed > 0.0008);
        let plan = direction(&gradient, gap, |_| false).unwrap();
        assert!(plan.displacement <= MAX_DISPLACEMENT);
        let new_gap = gap - plan.displacement * gradient[0];
        assert!(new_gap < 0.);
        assert!((new_gap + INTERIOR_MARGIN).abs() < 1e-18);
    }

    #[test]
    fn backtracking_rejects_nonlinear_overshoot_before_using_smaller_step() {
        // J(d)=0.001-0.1d+12d². The linear prediction overshoots; a smaller
        // finite step improves J. This remains private and is not fully valid.
        let plan = direction(&[0.1], 0.001, |_| false).unwrap();
        let (result, trials, distance) = backtrack(plan.displacement, 0.001, |d| {
            Ok(Some(Assessment { merit: 0.001 - 0.1*d + 12.*d*d, fully_valid: false, evidence: () }))
        }).unwrap().unwrap();
        assert_eq!(trials, 2);
        assert!(distance < plan.displacement);
        assert!(result.merit < 0.001);
        assert!(!result.fully_valid);
    }

    #[test]
    fn unchanged_nonfinite_and_wrong_direction_trials_are_bounded() {
        for merit in [0.01, 0.02, f64::NAN] {
            let mut calls = 0;
            let found = backtrack(0.01, 0.01, |_| {
                calls += 1;
                Ok(Some(Assessment { merit, fully_valid: false, evidence: () }))
            }).unwrap();
            assert!(found.is_none());
            assert_eq!(calls, MAX_BACKTRACKS);
        }
        assert!(direction(&[0.], 1., |_| false).is_none());
        assert!(direction(&[f64::NAN], 1., |_| false).is_none());
        assert!(direction(&[1.], f64::INFINITY, |_| false).is_none());
    }

    #[test]
    fn masked_coefficients_keep_exact_bits_and_small_norms_remain_bounded() {
        let weights = [-0., 1., -2.];
        let plan = direction(&[1e30, 1e-200, -2e-200], 1., |i| i == 0).unwrap();
        assert_eq!(plan.displacement, MAX_DISPLACEMENT);
        let next = shifted(&weights, &plan.unit, plan.displacement, |i| i == 0);
        assert_eq!(next[0].to_bits(), weights[0].to_bits());
        let norm = next.iter().zip(weights).map(|(a,b)| (a-b).powi(2)).sum::<f64>().sqrt();
        assert!(norm <= MAX_DISPLACEMENT + 1e-15);
    }

    #[test]
    fn fixed_set_hinge_does_not_reverse_progress_when_one_failure_disappears() {
        let valid = [true, false];
        let before = [hinge(&[0., 0.001], &valid).unwrap(), hinge(&[0., 0.01], &valid).unwrap()];
        let after = [hinge(&[0., -0.001], &valid).unwrap(), hinge(&[0., 0.009], &valid).unwrap()];
        assert!(after.iter().sum::<f64>() < before.iter().sum::<f64>());
        assert!(0.009 > (0.001 + 0.01) / 2.); // active-only mean says the opposite
    }
}
