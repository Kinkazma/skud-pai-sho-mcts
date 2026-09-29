use super::*;
// Affine repairs need more than the homogeneous zero-step fallback: a tiny
// gradient may carry a real nonzero requirement. Scale both sides of each
// halfspace before the active-set solve, and verify the returned finite vector.
pub(super) fn affine(g: &[f64], refs: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    if refs.len() != rhs.len()
        || refs.iter().any(|r| r.len() != g.len())
        || g.iter().chain(refs.iter().flatten()).chain(rhs).any(|x| !x.is_finite())
    {
        return None;
    }
    let mut unit = Vec::with_capacity(refs.len());
    let mut bounds = Vec::with_capacity(refs.len());
    for (r, &b) in refs.iter().zip(rhs) {
        let scale = r.iter().fold(0_f64, |a, x| a.max(x.abs()));
        if scale == 0. {
            if b > 0. { return None; }
            continue;
        }
        let mut normal = r.iter().map(|x| x / scale).collect::<Vec<_>>();
        let norm = dot(&normal, &normal).sqrt();
        for x in &mut normal { *x /= norm; }
        let bound = (b / scale) / norm;
        if !bound.is_finite() { return None; }
        unit.push(normal);
        bounds.push(bound);
    }
    let candidate = project_cached_checked(g, &unit, &bounds, &gram(&unit), true)?;
    affine_feasible(&candidate, &unit, &bounds).then_some(candidate)
}
fn affine_feasible(candidate: &[f64], refs: &[Vec<f64>], bounds: &[f64]) -> bool {
    if candidate.iter().any(|x| !x.is_finite()) { return false; }
    refs.iter().zip(bounds).all(|(r, b)| {
        let error = 64. * f64::EPSILON
            * (1. + b.abs() + r.iter().zip(candidate).map(|(x, y)| (x * y).abs()).sum::<f64>());
        dot(r, candidate) >= b - error
    })
}
// Homogeneous constraints always admit zero. Keep the established arithmetic
// when it succeeds; retry in unit-normal coordinates if absolute pivot tolerances
// reject a small reference gradient. Never apply an unprotected update on failure.
pub(super) fn homogeneous(g: &[f64], refs: &[Vec<f64>], cached: &[Vec<f64>]) -> Option<Vec<f64>> {
    let rhs = vec![0.; refs.len()];
    if let Some(p) = project_cached(g, refs, &rhs, cached) {
        if p.iter().all(|x| x.is_finite()) { return Some(p); }
    }
    if g.iter().chain(refs.iter().flatten()).any(|x| !x.is_finite()) {
        return None;
    }
    let unit: Vec<Vec<f64>> = refs.iter().filter_map(|r| {
        let scale = r.iter().fold(0_f64, |a, x| a.max(x.abs()));
        if scale == 0. { return None; }
        let scaled: Vec<_> = r.iter().map(|x| x / scale).collect();
        let norm = dot(&scaled, &scaled).sqrt();
        Some(scaled.iter().map(|x| x / norm).collect())
    }).collect();
    if let Some(p) = project(g, &unit, &vec![0.; unit.len()]) {
        if p.iter().all(|x| x.is_finite()) && unit.iter().all(|r| dot(r, &p) >= -1e-10) {
            return Some(p);
        }
    }
    // Degenerate numerical systems: no change preserves every protected loss
    // exactly. The learner's existing zero_steps counter records this event.
    Some(vec![0.; g.len()])
}

#[cfg(test)]
mod homogeneous_tests {
    use super::*;
    #[test]
    fn small_reference_is_feasible_and_preserves_tangent_learning() {
        let refs = vec![vec![1e-8, 0.]];
        let g = vec![-1., 2.];
        let cached = gram(&refs);
        assert!(project_cached(&g, &refs, &[0.], &cached).is_none());
        assert_eq!(homogeneous(&g, &refs, &cached).unwrap(), vec![0., 2.]);
    }
    #[test]
    fn ordinary_projection_is_unchanged_and_nonfinite_is_rejected() {
        let refs = vec![vec![1., 0.], vec![-1., 0.]];
        let g = vec![-1., 2.];
        let cached = gram(&refs);
        assert_eq!(homogeneous(&g, &refs, &cached), project_cached(&g, &refs, &[0.; 2], &cached));
        assert!(homogeneous(&[f64::NAN, 0.], &refs, &cached).is_none());
    }
    #[test]
    fn affine_small_gradient_has_the_same_constraint_as_a_unit_gradient() {
        let refs = vec![vec![1., 0.], vec![0., 1e-8]];
        assert!(project(&[0., 0.], &refs, &[0.1, 1e-9]).is_none());
        let expected = affine(&[0., 0.], &[vec![1., 0.], vec![0., 1.]], &[0.1, 0.1]).unwrap();
        let actual = affine(&[0., 0.], &refs, &[0.1, 1e-9]).unwrap();
        for (a, b) in actual.iter().zip(expected) { assert!((a - b).abs() < 1e-15); }
        assert_eq!(affine(&[0., 0.], &refs, &[0.1, -1e-9]).unwrap(), vec![0.1, 0.]);
    }
    #[test]
    fn affine_rejects_infeasible_zero_norm_nonfinite_and_bad_dimensions() {
        assert!(affine(&[0.], &[vec![0.]], &[1e-20]).is_none());
        assert_eq!(affine(&[2.], &[vec![0.]], &[0.]).unwrap(), vec![2.]);
        assert!(affine(&[0.], &[vec![1.], vec![-1.]], &[1., 1.]).is_none());
        assert!(affine(&[f64::INFINITY], &[], &[]).is_none());
        assert!(affine(&[0.], &[vec![1., 1.]], &[1.]).is_none());
        assert!(affine(&[0.], &[vec![1.]], &[]).is_none());
    }
    #[test]
    fn affine_checks_tight_feasibility_before_choosing_the_nearest_active_set() {
        // The historical 1e-10 feasibility tolerance admits zero and makes it
        // win the distance comparison. Rejecting zero only after that comparison
        // would wrongly report infeasibility despite the active solution below.
        for rhs in [1e-12, 0.07295039242318155 - (0.07295039141925701 + 1e-9)] {
            assert!(rhs > 0.);
            assert_eq!(project(&[0.], &[vec![1.]], &[rhs]).unwrap(), vec![0.]);
            assert_eq!(affine(&[0.], &[vec![1.]], &[rhs]).unwrap(), vec![rhs]);
        }
        let rhs = [3.925e-12, -0.5];
        assert_eq!(affine(&[0., 0.], &[vec![1., 0.], vec![0., 1.]], &rhs).unwrap(), vec![rhs[0], 0.]);
    }
}
// Euclidean projection onto <=5 gradient halfspaces, enumerating active sets.
pub(super) fn project(g: &[f64], refs: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let gram=gram(refs);
    project_cached(g,refs,rhs,&gram)
}
pub(super) fn gram(refs:&[Vec<f64>])->Vec<Vec<f64>> {
    refs.iter().map(|a|refs.iter().map(|b|dot(a,b)).collect()).collect()
}
pub(super) fn project_cached(g:&[f64],refs:&[Vec<f64>],rhs:&[f64],gram:&[Vec<f64>])->Option<Vec<f64>> {
    project_cached_checked(g, refs, rhs, gram, false)
}
fn project_cached_checked(g:&[f64],refs:&[Vec<f64>],rhs:&[f64],gram:&[Vec<f64>],verify_affine:bool)->Option<Vec<f64>> {
    let n=refs.len();
    let b: Vec<_> = refs
        .iter()
        .zip(rhs)
        .map(|(a, rhs)| dot(a, g) - rhs)
        .collect();
    let mut answer = None;
    let mut checked_answer = None;
    let mut distance = f64::INFINITY;
    for mask in 0..1usize << n {
        let ids: Vec<_> = (0..n).filter(|i| mask & (1 << i) != 0).collect();
        let k = ids.len();
        let mut a: Vec<Vec<f64>> = ids
            .iter()
            .map(|&i| {
                ids.iter()
                    .map(|&j| gram[i][j])
                    .chain(std::iter::once(-b[i]))
                    .collect()
            })
            .collect();
        let mut singular = false;
        for c in 0..k {
            let pivot = (c..k)
                .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
                .unwrap();
            a.swap(c, pivot);
            let v = a[c][c];
            if v.abs() < 1e-14 {
                singular = true;
                break;
            }
            for j in c..=k {
                a[c][j] /= v;
            }
            for i in 0..k {
                if i != c {
                    let f = a[i][c];
                    for j in c..=k {
                        a[i][j] -= f * a[c][j];
                    }
                }
            }
        }
        if singular {
            continue;
        }
        let mut lambdas = vec![0.; n];
        for (i, &r) in ids.iter().enumerate() {
            lambdas[r] = a[i][k];
        }
        if lambdas.iter().any(|v| *v < -1e-10) {
            continue;
        }
        if (0..n).any(|i| b[i] + dot(&gram[i], &lambdas) < -1e-10) {
            continue;
        }
        let dist = (0..n)
            .map(|i| lambdas[i] * dot(&gram[i], &lambdas))
            .sum::<f64>();
        if dist < distance {
            if verify_affine {
                let mut out=g.to_vec();
                for (l,r) in lambdas.iter().zip(refs) {for (x,v) in out.iter_mut().zip(r) {*x+=l*v;}}
                if !affine_feasible(&out, refs, rhs) { continue; }
                checked_answer = Some(out);
            }
            distance = dist;
            answer = Some(lambdas);
        }
    }
    if verify_affine { return checked_answer; }
    answer.map(|lambdas| {
        let mut out=g.to_vec();
        for (l,r) in lambdas.iter().zip(refs) {for (x,v) in out.iter_mut().zip(r) {*x+=l*v;}}
        out
    })
}
