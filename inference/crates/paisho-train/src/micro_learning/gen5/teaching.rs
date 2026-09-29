//! Search estimates improve a prior without labelling unvisited actions as losses.
use super::*;
pub(super) fn target(r: &MicroSearchReport) -> Result<(Vec<f64>, Vec<f64>)> {
    target_with_prior(r, &r.priors)
}
pub(super) fn target_with_prior(
    r: &MicroSearchReport,
    prior: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    if prior.len() != r.actions.len()
        || prior.iter().any(|v| !v.is_finite() || *v < 0.)
        || (prior.iter().sum::<f64>() - 1.).abs() > 1e-8
    {
        return Err(invalid("invalid raw teacher prior"));
    }
    let total: usize = r.visits.iter().sum();
    let completion = if total == 0 {
        r.network_value
    } else {
        r.values
            .iter()
            .zip(&r.visits)
            .map(|(q, n)| q * *n as f64 / total as f64)
            .sum()
    };
    let q: Vec<f64> = (0..r.actions.len())
        .map(|i| {
            r.proven_action_values
                .get(i)
                .copied()
                .flatten()
                .map_or_else(
                    || {
                        if r.visits[i] > 0 {
                            r.values[i]
                        } else {
                            completion
                        }
                    },
                    |p| p as f64,
                )
        })
        .collect();
    if r.proven_value.is_some() {
        return Ok((r.policy_target.clone(), q));
    }
    let allowed: Vec<usize> = (0..q.len())
        .filter(|&i| r.proven_action_values.get(i).copied().flatten() != Some(-1))
        .collect();
    if allowed.is_empty() {
        return Ok((r.policy_target.clone(), q));
    }
    let logits: Vec<f64> = allowed
        .iter()
        .map(|&i| prior[i].max(1e-300).ln() + q[i])
        .collect();
    let p = micro_softmax(&logits).map_err(invalid)?;
    let mut policy = vec![0.; q.len()];
    for (i, p) in allowed.into_iter().zip(p) {
        policy[i] = p;
    }
    Ok((policy, q))
}
#[cfg(test)]
mod tests {
    use super::*;
    fn report() -> MicroSearchReport {
        let r: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
        ))
        .parse()
        .unwrap();
        let p = r.initial_position();
        let mut session =
            MicroMctsSession::new(Arc::new(MicroModel::seeded(4).with_spatial_policy()));
        session
            .search_with_options(&p, 8, None, MicroSearchOptions::default())
            .unwrap()
    }
    #[test]
    fn raw_target_recouples_once_to_the_search_improvement() {
        let mut r = report();
        let n = r.actions.len();
        let mut raw = vec![0.1 / (n - 1) as f64; n];
        raw[0] = 0.9;
        let v = (0..n).map(|i| 0.05 * (i % 7) as f64).collect::<Vec<_>>();
        r.priors = micro_softmax(
            &raw.iter()
                .zip(&v)
                .map(|(p, v)| p.ln() + 16. * v)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        r.proven_value = None;
        r.proven_action_values = vec![None; n];
        r.visits.fill(1);
        r.values = (0..n).map(|i| 0.1 * (i % 5) as f64).collect();
        let desired = target(&r).unwrap().0;
        let corrected = target_with_prior(&r, &raw).unwrap().0;
        let played = micro_softmax(
            &corrected
                .iter()
                .zip(&v)
                .map(|(p, v)| p.ln() + 16. * v)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(played
            .iter()
            .zip(desired)
            .all(|(a, b)| (a - b).abs() < 1e-14));
        r.proven_action_values[0] = Some(-1);
        assert_eq!(target_with_prior(&r, &raw).unwrap().0[0], 0.);
        r.proven_value = Some(1);
        assert_eq!(target_with_prior(&r, &raw).unwrap().0, r.policy_target);
    }
    #[test]
    fn flat_values_preserve_ninety_percent_and_unvisited_support() {
        let mut r = report();
        let n = r.actions.len();
        r.priors = vec![0.1 / (n - 1) as f64; n];
        r.priors[0] = 0.9;
        r.values.fill(0.);
        r.proven_action_values = vec![None; n];
        r.proven_value = None;
        r.network_value = 0.;
        r.visits.fill(0);
        r.visits[1] = 8;
        let (p, _) = target(&r).unwrap();
        assert!((p[0] - 0.9).abs() < 1e-12);
        assert!(p.iter().all(|p| *p > 0.));
        r.values[1] = 0.8;
        r.visits[2] = 2;
        r.values[2] = -0.5;
        let (p, q) = target(&r).unwrap();
        let before: f64 = r.priors.iter().zip(&q).map(|(p, q)| p * q).sum();
        let after: f64 = p.iter().zip(&q).map(|(p, q)| p * q).sum();
        assert!(after >= before);
        r.proven_action_values[0] = Some(-1);
        let (p, _) = target(&r).unwrap();
        assert_eq!(p[0], 0.);
    }
}
