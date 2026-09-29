//! Numerical interior target only. The actual finite gate is never relaxed.
use super::*;

pub(super) struct Step {
    pub correction: Option<Vec<f64>>,
    pub diagnostic: serde_json::Value,
}
fn interior_target(ceiling: f64, current: f64) -> Result<f64> {
    if !ceiling.is_finite() || !current.is_finite() {
        return Err(invalid("non-finite fresh interior target"));
    }
    // Negative numerical ceilings have no nonnegative interior. Preserve their
    // legacy affine target; the unchanged finite tolerance decides acceptance.
    if ceiling <= 0. {
        return Ok(ceiling);
    }
    let scale = 1_f64.max(ceiling.abs()).max(current.abs());
    Ok((ceiling - f64::EPSILON.sqrt() * scale).max(0.))
}
/// For theta' = theta - d, grad(F).d >= F(current) - target.
/// No extra forward/gradient, and at most two algebraic solves with these same
/// normals. An infeasible stricter target never replaces the original problem.
pub(super) fn solve(
    origin: &[f64],
    references: &[Vec<f64>],
    rhs: &[f64],
    fresh_index: usize,
    current: f64,
    ceiling: f64,
) -> Result<Step> {
    if fresh_index >= rhs.len() || rhs[fresh_index].to_bits() != (current - ceiling).to_bits() {
        return Err(invalid(
            "fresh interior must replace exactly the original fresh RHS",
        ));
    }
    let target = interior_target(ceiling, current)?;
    let started = paisho_platform::training_time::now();
    let (correction, solves, fallback) = if target < ceiling {
        let mut interior_rhs = rhs.to_vec();
        interior_rhs[fresh_index] = current - target;
        match projection::affine(origin, references, &interior_rhs) {
            Some(step) => (Some(step), 1, false),
            None => (projection::affine(origin, references, rhs), 2, true),
        }
    } else {
        (projection::affine(origin, references, rhs), 1, false)
    };
    Ok(Step {
        diagnostic: serde_json::json!({"ceiling":ceiling,"current":current,
        "target":target,"margin":ceiling-target,"strict_target_attempted":target<ceiling,
        "original_target_fallback":fallback,"algebraic_solves":solves,
        "seconds":paisho_platform::training_time::elapsed(started).as_secs_f64(),
        "feasible":correction.is_some(),"fresh_normal_index":fresh_index,
        "additional_model_reads":0,"additional_gradients":0}),
        correction,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interior_target_scales_generically_and_never_changes_acceptance() {
        for ceiling in [0., 1e-15, 0.1, 1., 1000.] {
            let current = ceiling + 0.03;
            let target = interior_target(ceiling, current).unwrap();
            assert!(target >= 0. && target <= ceiling);
            if ceiling > 0. {
                assert!(target < ceiling);
            }
            // The proposal is an affine target only. The finite outer test is
            // still the same ceiling+1e-12, including just-inside/outside cases.
            let outer = ceiling + 1e-12;
            assert!(ceiling + 0.5e-12 <= outer);
            assert!(ceiling + 2e-12 > outer);
        }
        assert_eq!(interior_target(-1e-15, 0.).unwrap(), -1e-15);
        assert!(interior_target(f64::NAN, 1.).is_err());
    }
    #[test]
    fn quadratic_remainder_crosses_boundary_without_an_interior_target() {
        // Three unrelated objective scales, no captured C1 loss or tuned gap.
        // F(x)=C+x+x², with exact derivative 1+2x. Newton aimed at C
        // leaves a positive quadratic remainder even in exact arithmetic.
        for ceiling in [0.1, 1., 1000.] {
            let margin = ceiling - interior_target(ceiling, ceiling).unwrap();
            let x = (0.5 * margin).sqrt();
            let loss = ceiling + x + x * x;
            let gradient = 1. + 2. * x;
            let old_step = (loss - ceiling) / gradient;
            let old_x = x - old_step;
            assert!(ceiling + old_x + old_x * old_x > ceiling + 1e-12);
            let step = solve(
                &[0.],
                &[vec![gradient]],
                &[loss - ceiling],
                0,
                loss,
                ceiling,
            )
            .unwrap();
            assert_eq!(step.diagnostic["original_target_fallback"], false);
            let new_x = x - step.correction.unwrap()[0];
            assert!(ceiling + new_x + new_x * new_x <= ceiling + 1e-12);
            let eps = 1e-6;
            let numeric = ((ceiling + (x + eps) + (x + eps).powi(2))
                - (ceiling + (x - eps) + (x - eps).powi(2)))
                / (2. * eps);
            assert!((numeric - gradient).abs() < 2e-7);
        }
    }
    #[test]
    fn incompatible_strict_margin_falls_back_to_the_original_feasible_boundary() {
        let refs = vec![vec![1.], vec![-1.]];
        let rhs = vec![0., 0.];
        // Old constraint requires d>=0; fresh with current=C requires d<=0.
        // A strict negative displacement is incompatible but d=0 is valid.
        let expected = projection::affine(&[0.], &refs, &rhs).unwrap();
        let got = solve(&[0.], &refs, &rhs, 1, 4., 4.).unwrap();
        assert_eq!(got.diagnostic["original_target_fallback"], true);
        assert_eq!(got.diagnostic["algebraic_solves"], 2);
        let actual = got.correction.unwrap();
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
        let impossible = solve(&[0.], &refs, &[1., 0.], 1, 4., 4.).unwrap();
        assert!(impossible.correction.is_none());
    }
    #[test]
    fn finite_violation_is_not_accepted_by_a_changed_or_misidentified_rhs() {
        assert!(solve(&[0.], &[vec![1.]], &[0.], 0, 4.001, 4.).is_err());
        assert!(solve(&[0.], &[vec![1.]], &[0.], 1, 4., 4.).is_err());
        let ceiling = 2.;
        let current = ceiling + 1e-6;
        let step = solve(
            &[0.],
            &[vec![1.]],
            &[current - ceiling],
            0,
            current,
            ceiling,
        )
        .unwrap();
        let d = step.correction.unwrap()[0];
        assert!(current - d < ceiling);
        assert!(current > ceiling + 1e-12); // the input does not become admissible
    }
}
