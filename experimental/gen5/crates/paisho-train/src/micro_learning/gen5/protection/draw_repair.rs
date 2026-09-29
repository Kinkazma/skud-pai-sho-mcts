//! Resolve the draw residuals jointly instead of linearizing only their MSE.
//! A small orthonormal row basis removes those directions from the other
//! halfspaces. Final nonlinear acceptance, freshness and choice tests still apply.
use super::*;

fn orthogonal(mut x: Vec<f64>, basis: &[Vec<f64>]) -> Vec<f64> {
    // Reorthogonalization matters when two legal draw positions are very alike.
    for _ in 0..2 {
        for q in basis {
            let a = dot(&x, q);
            for (v, b) in x.iter_mut().zip(q) {
                *v -= a * b;
            }
        }
    }
    x
}
fn basis(rows: &[Vec<f64>], rhs: &[f64]) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    let n = rows.first()?.len();
    let mut q = vec![];
    let mut step = vec![0.; n];
    for (row, target) in rows.iter().zip(rhs) {
        let mut v = orthogonal(row.clone(), &q);
        let norm = dot(&v, &v).sqrt();
        let residual = target - dot(row, &step);
        if norm <= 1e-10 * dot(row, row).sqrt().max(1e-30) {
            if residual.abs() > 1e-10 {
                return None;
            } else {
                continue;
            }
        }
        if !norm.is_finite() {
            return None;
        }
        for x in &mut v {
            *x /= norm;
        }
        for (s, x) in step.iter_mut().zip(&v) {
            *s += (residual / norm) * x;
        }
        q.push(v);
    }
    Some((q, step))
}
pub(super) fn solve(
    model: &MicroModel,
    rows: &[Arc<MicroExample>],
    refs: &[Vec<f64>],
    rhs: &[f64],
    loss: f64,
    target: f64,
    fresh: Option<(usize, f64, f64)>,
    interior: bool,
    parallel: Option<&cpu::Ordered>,
) -> Result<(Option<Vec<f64>>, serde_json::Value)> {
    let started = Instant::now();
    let draws = rows
        .iter()
        .filter(|e| e.value == 0.)
        .cloned()
        .collect::<Vec<_>>();
    if draws.is_empty() || draws.len() > 32 {
        return Ok((
            projection::affine(&vec![0.; model.parameters().len()], refs, rhs),
            serde_json::json!({"fallback":"draw population outside bound"}),
        ));
    }
    let frozen = model.clone();
    let f = move |e: &Arc<MicroExample>| -> std::result::Result<(Vec<f64>, f64), String> {
        let value = frozen.value(&e.state);
        // A far target avoids division by a near-zero training residual.
        let label = if value >= 0. { -1. } else { 1. };
        let ex = MicroExample {
            state: e.state.clone(),
            value: label,
            value_weight: 1.,
            policy_weight: 0.,
            actions: vec![],
            policy: vec![],
            action_values: vec![],
            structured: vec![],
            policy_support: false,
            sequence_source: e.sequence_source,
        };
        let (_, mut g) = frozen.loss_gradient(&ex)?;
        for x in &mut g {
            *x /= value - label;
        }
        Ok((g, value))
    };
    let parts: Vec<_> = match parallel {
        Some(p) => p.map_owned(draws, |_| 1, f),
        None => draws.iter().map(f).collect(),
    };
    let parts = parts
        .into_iter()
        .collect::<std::result::Result<Vec<_>, String>>()
        .map_err(invalid)?;
    let scale = if loss > target {
        (target / loss).max(0.).sqrt() * 0.99
    } else {
        1.
    };
    let indices = (0..model.parameters().len())
        .filter(|&i| parts.iter().any(|(g, _)| g[i] != 0.))
        .collect::<Vec<_>>();
    let gradients = parts
        .iter()
        .map(|(g, _)| indices.iter().map(|&i| g[i]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let requirements = parts
        .iter()
        .map(|(_, v)| (1. - scale) * v)
        .collect::<Vec<_>>();
    let Some((q, particular)) = basis(&gradients, &requirements) else {
        return Ok((
            None,
            serde_json::json!({"failure":"draw Jacobian inconsistent"}),
        ));
    };
    // V3 reference order is loss / DRAW / win / optional secondary, choices, fresh.
    let normals = refs
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != 1)
        .map(|(_, r)| {
            let mut out = r.clone();
            let values = orthogonal(indices.iter().map(|&i| r[i]).collect(), &q);
            for (&i, v) in indices.iter().zip(values) {
                out[i] = v;
            }
            out
        })
        .collect::<Vec<_>>();
    let bounds = refs
        .iter()
        .zip(rhs)
        .enumerate()
        .filter(|(i, _)| *i != 1)
        .map(|(_, (r, b))| {
            b - indices
                .iter()
                .zip(&particular)
                .map(|(&i, p)| r[i] * p)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();
    let origin = vec![0.; model.parameters().len()];
    let remainder = if interior && fresh.is_some() {
        let (index, value, ceiling) = fresh.unwrap();
        let margin = if ceiling > 0. {
            (f64::EPSILON.sqrt() * 1_f64.max(value.abs()).max(ceiling.abs())).min(ceiling)
        } else {
            0.
        };
        let mut tight = bounds.clone();
        tight[index - 1] += margin;
        projection::affine(&origin, &normals, &tight)
            .or_else(|| projection::affine(&origin, &normals, &bounds))
    } else {
        projection::affine(&origin, &normals, &bounds)
    };
    let step = remainder.map(|mut r| {
        for (&i, p) in indices.iter().zip(&particular) {
            r[i] += p;
        }
        r
    });
    Ok((
        step,
        serde_json::json!({"rows":parts.len(),"rank":q.len(),"scale":scale,
        "seconds":started.elapsed().as_secs_f64(),"extra_gradients":parts.len(),"acceptance_unchanged":true,"active_coordinates":indices.len()}),
    ))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coupled_rows_are_solved_and_other_directions_remain_free() {
        let rows = vec![vec![1., 1., 0.], vec![1., -1., 0.]];
        let (q, s) = basis(&rows, &[2., 0.]).unwrap();
        assert!((s[0] - 1.).abs() < 1e-12 && (s[1] - 1.).abs() < 1e-12);
        let t = orthogonal(vec![1., 2., 3.], &q);
        assert!(t[0].abs() < 1e-12 && t[1].abs() < 1e-12 && t[2] == 3.);
    }
    #[test]
    fn contradictory_duplicate_equations_fail_closed() {
        assert!(basis(&[vec![1., 0.], vec![1., 0.]], &[1., 2.]).is_none());
        assert!(basis(&[vec![1., 0.], vec![1., 0.]], &[1., 1.]).is_some());
    }
}
