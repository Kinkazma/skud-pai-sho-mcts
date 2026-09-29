//! Diagnostic equivalence check on actual lost choices in a resident panel.
//! No model update or publication; run outside timed benchmark legs.
use super::*;

fn best(logits: &[f64], valid: impl Fn(usize) -> bool) -> Option<usize> {
    (0..logits.len())
        .filter(|&i| valid(i))
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
}

fn add_margin(
    gradient: &mut [f64],
    model: &MicroModel,
    row: &Witness,
    good: usize,
    bad: usize,
) -> Result<()> {
    let mut example = row.example.as_ref().clone();
    example.value_weight = 0.;
    example.policy_weight = 1.;
    example.action_values.clear();
    example.policy.fill(0.);
    example.policy[good] = 1.;
    let mut a = model.loss_gradient(&example).map_err(invalid)?.1;
    example.policy[good] = 0.;
    example.policy[bad] = 1.;
    let b = model.loss_gradient(&example).map_err(invalid)?.1;
    for (j, ((sum, a), b)) in gradient.iter_mut().zip(a.iter_mut()).zip(b).enumerate() {
        if !branches::value_parameter(j) {
            *sum += *a - b;
        }
    }
    Ok(())
}

pub(super) fn run(guard: &Guard, model: &MicroModel) -> Result<serde_json::Value> {
    let mut old_gradient = vec![0.; model.parameters().len()];
    let mut cached_gradient = vec![0.; model.parameters().len()];
    let mut old_objective = 0.;
    let mut cached_objective = 0.;
    let mut rows = 0usize;
    let mut raw_terms = 0usize;
    let mut coupled_terms = 0usize;
    let mut logits_compared = 0usize;
    for panel in std::iter::once(guard).chain(guard.validation.as_deref()) {
        let score = panel.evaluate(model)?;
        for (i, row) in panel.rows.iter().enumerate() {
            let raw = panel.score.raw[i] && !score.raw[i];
            let coupled = panel.score.coupled[i] && !score.coupled[i];
            if !raw && !coupled {
                continue;
            }
            rows += 1;
            let logp: Vec<_> = score.priors[i].iter().map(|p| p.max(1e-300).ln()).collect();
            let mut previous_coupled = logp.clone();
            let stored: &Vec<f64> = score
                .coupled_logits
                .get(i)
                .ok_or_else(|| invalid("margin equivalence missing measured logits"))?;
            if stored.len() != logp.len() {
                return Err(invalid("margin equivalence logit shape mismatch"));
            }
            if coupled {
                let inputs = row
                    .successors
                    .get()
                    .ok_or_else(|| invalid("margin equivalence missing measured successors"))?;
                // Preserve the original margin_gradient expression, including
                // multiplication by sign rather than the scorer's +/- branch.
                for (l, (sign, state)) in previous_coupled.iter_mut().zip(inputs) {
                    *l += panel.beta
                        * if state.is_empty() {
                            *sign
                        } else {
                            sign * model.value(state)
                        };
                }
                if previous_coupled
                    .iter()
                    .zip(stored)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err(invalid("stored/previous coupled margin logits differ"));
                }
                logits_compared += stored.len();
            }
            for (active, previous, cached) in
                [(raw, &logp, &logp), (coupled, &previous_coupled, stored)]
            {
                if !active {
                    continue;
                }
                let pair = |logits: &[f64]| -> Result<(usize, usize)> {
                    Ok((
                        best(logits, |j| row.valid[j])
                            .ok_or_else(|| invalid("margin equivalence has no good action"))?,
                        best(logits, |j| !row.valid[j])
                            .ok_or_else(|| invalid("margin equivalence has no competitor"))?,
                    ))
                };
                let (old_good, old_bad) = pair(previous)?;
                let (new_good, new_bad) = pair(cached)?;
                if (old_good, old_bad) != (new_good, new_bad) {
                    return Err(invalid("stored/previous margin action pair differs"));
                }
                add_margin(&mut old_gradient, model, row, old_good, old_bad)?;
                add_margin(&mut cached_gradient, model, row, new_good, new_bad)?;
                old_objective += previous[old_bad] - previous[old_good];
                cached_objective += cached[new_bad] - cached[new_good];
            }
            raw_terms += usize::from(raw);
            coupled_terms += usize::from(coupled);
        }
    }
    let count = raw_terms + coupled_terms;
    if count > 0 {
        for g in &mut old_gradient {
            *g /= count as f64;
        }
        for g in &mut cached_gradient {
            *g /= count as f64;
        }
        old_objective /= count as f64;
        cached_objective /= count as f64;
    }
    if old_objective.to_bits() != cached_objective.to_bits()
        || old_gradient
            .iter()
            .zip(&cached_gradient)
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(invalid("stored/previous aggregate margin gradient differs"));
    }
    Ok(serde_json::json!({
        "lost_rows":rows,"raw_terms":raw_terms,"coupled_terms":coupled_terms,
        "coupled_logit_coefficients_compared":logits_compared,
        "gradient_coefficients_compared":if count>0 {old_gradient.len()}else{0},
        "old_objective":old_objective,"cached_objective":cached_objective,
        "all_compared_bits_exact":true,
        "scope":"actual lost choices only; no optimizer update; run outside timed legs"
    }))
}
