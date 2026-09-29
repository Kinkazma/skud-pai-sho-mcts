use super::*;
// Euclidean projection onto <=4 gradient halfspaces, enumerating active sets.
fn project(g: &[f64], refs: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = refs.len();
    let gram: Vec<Vec<f64>> = refs
        .iter()
        .map(|a| refs.iter().map(|b| dot(a, b)).collect())
        .collect();
    let b: Vec<_> = refs
        .iter()
        .zip(rhs)
        .map(|(a, rhs)| dot(a, g) - rhs)
        .collect();
    let mut answer = None;
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
            distance = dist;
            let mut out = g.to_vec();
            for (l, r) in lambdas.iter().zip(refs) {
                for (x, v) in out.iter_mut().zip(r) {
                    *x += l * v;
                }
            }
            answer = Some(out);
        }
    }
    answer
}
fn proof_examples(m: &MicroModel, rows: &[&Proof], kind: usize) -> Vec<MicroExample> {
    rows.iter()
        .filter(|r| {
            if kind == 0 {
                r.ex.value == 1.
            } else {
                (r.ex.value as i8 + 1) as usize == kind - 1
            }
        })
        .map(|r| {
            let mut e = r.ex.clone();
            if kind == 0 {
                let p = prior(m, &e);
                let mass: f64 = p
                    .iter()
                    .zip(&r.union)
                    .filter(|(_, v)| **v)
                    .map(|(p, _)| p)
                    .sum();
                e.policy = p
                    .iter()
                    .zip(&r.union)
                    .map(|(p, v)| if *v { p / mass } else { 0. })
                    .collect();
                e.value_weight = 0.;
            } else {
                e.policy_weight = 0.;
            }
            e
        })
        .collect()
}
fn reference_losses(m: &MicroModel, rows: &[&Proof]) -> [f64; 4] {
    let mut s = [0.; 4];
    let mut ns = [0; 4];
    for r in rows {
        let k = (r.ex.value as i8 + 1) as usize + 1;
        ns[k] += 1;
        s[k] += 0.5 * (m.embed(&r.ex.state).value - r.ex.value).powi(2);
        if r.ex.value == 1. {
            let p = prior(m, &r.ex);
            s[0] -= p
                .iter()
                .zip(&r.union)
                .filter(|(_, v)| **v)
                .map(|(p, _)| p)
                .sum::<f64>()
                .max(1e-300)
                .ln();
            ns[0] += 1;
        }
    }
    for k in 0..4 {
        s[k] /= ns[k].max(1) as f64;
    }
    s
}
pub(super) fn run(d: &Data, rows: &[&Proof]) -> Result<Value> {
    assert_eq!(
        project(&[-1., 2.], &[vec![1., 0.]], &[0.]).unwrap(),
        vec![0., 2.]
    );
    assert_eq!(
        project(&[0., 0.], &[vec![1., 0.]], &[0.1]).unwrap(),
        vec![0.1, 0.]
    );
    let mut results = vec![];
    let fresh: Vec<_> = (0..256)
        .map(|i| d.fresh[i * d.fresh.len() / 256].clone())
        .collect();
    for mi in [0, 4] {
        let model = &d.models[mi];
        let start = Instant::now();
        let g = mean_gradient(model, &fresh);
        let refs: Vec<_> = (0..4)
            .map(|k| mean_gradient(model, &proof_examples(model, rows, k)))
            .collect();
        let gradient_seconds = start.elapsed().as_secs_f64();
        let t = Instant::now();
        let projected = project(&g, &refs, &[0.; 4]).unwrap();
        let projection_seconds = t.elapsed().as_secs_f64();
        let before = reference_losses(model, rows);
        let fresh_before = mean_loss(model, &fresh);
        let mut trials = vec![];
        for (name, gradient) in [("ordinary", &g), ("projected", &projected)] {
            for k in 0..8 {
                let rate = 0.02 / 2f64.powi(k);
                let candidate = step(model, gradient, rate);
                let after = reference_losses(&candidate, rows);
                let fresh_after = mean_loss(&candidate, &fresh);
                trials.push(json!({"kind":name,"rate":rate,"reference_losses":after,"fresh_loss":fresh_after,"finite_nonregression_1e_6":after.iter().zip(before).all(|(a,b)|*a<=b+1e-6),"fresh_improved":fresh_after<fresh_before}));
            }
        }
        let mut candidate = step(model, &projected, 0.02);
        let mut corrections = vec![];
        let begun = Instant::now();
        for iteration in 0..=6 {
            let current = reference_losses(&candidate, rows);
            corrections.push(json!({"iteration":iteration,"reference_losses":current,"fresh_loss":mean_loss(&candidate,&fresh)}));
            if current.iter().zip(before).all(|(a, b)| *a <= b + 1e-9) || iteration == 6 {
                break;
            }
            let refs: Vec<_> = (0..4)
                .map(|k| mean_gradient(&candidate, &proof_examples(&candidate, rows, k)))
                .collect();
            let rhs: Vec<_> = current
                .iter()
                .zip(before)
                .map(|(a, b)| a - b + 1e-9)
                .collect();
            let Some(correction) = project(&vec![0.; g.len()], &refs, &rhs) else {
                break;
            };
            candidate = clone_weights(
                &candidate,
                candidate
                    .parameters()
                    .iter()
                    .zip(correction)
                    .map(|(w, g)| w - g)
                    .collect(),
            );
        }
        let before_choices: Vec<_> = rows
            .iter()
            .filter(|r| r.ex.value == 1.)
            .map(|r| r.union[best(&prior(model, &r.ex))])
            .collect();
        let after_choices: Vec<_> = rows
            .iter()
            .filter(|r| r.ex.value == 1.)
            .map(|r| r.union[best(&prior(&candidate, &r.ex))])
            .collect();
        results.push(json!({"model":mi,"before":before,"fresh_before":fresh_before,"gradient_seconds":gradient_seconds,"projection_seconds":projection_seconds,"norm_ratio":(dot(&projected,&projected)/dot(&g,&g)).sqrt(),"raw_dots":refs.iter().map(|r|dot(r,&g)).collect::<Vec<_>>(),"projected_dots":refs.iter().map(|r|dot(r,&projected)).collect::<Vec<_>>(),"trials":trials,"finite_corrections":corrections,"finite_correction_seconds":begun.elapsed().as_secs_f64(),"old_verified_choices":before_choices.iter().filter(|v|**v).count(),"new_verified_choices":after_choices.iter().filter(|v|**v).count(),"verified_choices_lost":before_choices.iter().zip(&after_choices).filter(|(a,b)|**a && !**b).count()}));
    }
    Ok(
        json!({"runs":results,"loss_order":["any_verified_win_policy_mass","proved_loss_value","proved_draw_value","proved_win_value"],"scope":"Average proof losses only; tangent constraint is first-order, finite rates explicitly checked; no per-position or global strength guarantee."}),
    )
}
