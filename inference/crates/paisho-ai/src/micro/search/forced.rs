//! Root forced playouts and policy target pruning, after Wu (2019), section3.2.
//! Search visits/Q stay intact; pruning changes only the supervised target.
//! Retained trees use total same-model visits and the current root's noisy prior.

pub(super) fn needs_visit(visits: usize, prior: f64, total: usize, strength: f64) -> bool {
    visits > 0 && (visits as f64).powi(2) < strength * prior * total as f64
}

pub(super) fn prune(
    visits: &[usize],
    values: &[f64],
    priors: &[f64],
    champion: usize,
    scale: f64,
    strength: f64,
) -> Vec<usize> {
    let total = visits.iter().sum::<usize>();
    let score = |i: usize, n: usize| values[i] + scale * priors[i] / (1 + n) as f64;
    let reference = score(champion, visits[champion]);
    visits
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            if i == champion || n == 0 || score(i, n) >= reference {
                return n;
            }
            let limit = (strength * priors[i] * total as f64).sqrt().floor() as usize;
            let mut lo = n.saturating_sub(limit);
            let mut hi = n;
            // Lowest remaining count with urgency still strictly below the champion.
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if score(i, mid) < reference {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            // Wu also prunes children reduced to a single playout.
            if lo == 1 && lo < n {
                0
            } else {
                lo
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forcing_needs_an_initial_visit_and_scales_sublinearly() {
        assert!(!needs_visit(0, 0.5, 100, 2.0));
        assert!(needs_visit(1, 0.01, 100, 2.0));
        assert!(!needs_visit(2, 0.01, 100, 2.0));
        assert!(!needs_visit(1, 0.5, 100, 0.0));
    }
    #[test]
    fn pruning_removes_bad_exploration_but_keeps_promising_evidence_and_champion() {
        let visits = [80, 5, 15];
        let priors = [0.8, 0.1, 0.1];
        let scale = 1.5 * 101.0_f64.sqrt();
        let corrected = prune(&visits, &[0.0, -1.0, 0.4], &priors, 0, scale, 2.0);
        assert_eq!(corrected, [80, 0, 15]);
        assert_eq!(visits, [80, 5, 15]);
        assert_eq!(
            prune(&visits, &[0.0, -1.0, 0.4], &priors, 0, scale, 0.0),
            visits
        );
    }
    #[test]
    fn binary_pruning_matches_integer_reference() {
        for n in 1..40 {
            for q in [-1.0, -0.4, 0.0, 0.8] {
                let visits = [50, n];
                let priors = [0.7, 0.3];
                let total = 50 + n;
                let scale = 1.5 * ((total + 1) as f64).sqrt();
                let reference = scale * 0.7 / 51.0;
                let limit = (2.0 * 0.3 * total as f64).sqrt().floor() as usize;
                let mut expected = n;
                for _ in 0..limit.min(n) {
                    if q + scale * 0.3 / expected as f64 >= reference {
                        break;
                    }
                    expected -= 1;
                }
                if expected == 1 && expected < n {
                    expected = 0;
                }
                assert_eq!(
                    prune(&visits, &[0.0, q], &priors, 0, scale, 2.0)[1],
                    expected
                );
            }
        }
    }
}
