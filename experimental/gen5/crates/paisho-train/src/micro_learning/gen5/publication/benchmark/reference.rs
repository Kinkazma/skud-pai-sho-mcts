//! Frozen V35 sequential scoring reference, diagnostic only.
use super::*;
pub(super) fn measure_cached(rows: &[Arc<Witness>], model: &MicroModel, beta: f64, cached: bool) -> Result<Score> {
    let mut raw = vec![];
    let mut coupled = vec![];
    let mut mass = 0.;
    let mut errors = [0.; 3];
    let mut counts = [0usize; 3];
    for r in rows {
        let e = model.embed(&r.example.state);
        if cached && r.example.value != 1. {
            // These two flags are identically false, regardless of the policy.
            raw.push(false);
            coupled.push(false);
            let k = (r.example.value as i8 + 1) as usize;
            errors[k] += (e.value - r.example.value).powi(2);
            counts[k] += 1;
            continue;
        }
        let base = micro_softmax(&MicroModel::logits(&e, &r.example.actions)).map_err(invalid)?;
        let p = model
            .memory_priors(&r.example.state, &r.example.actions, &base, 0)
            .map_err(invalid)?;
        let mut logits: Vec<f64> = p.iter().map(|p| p.max(1e-300).ln()).collect();
        let best = |p: &[f64]| {
            (0..p.len())
                .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
                .unwrap()
        };
        raw.push(r.example.value == 1. && r.valid[best(&p)]);
        if r.example.value == 1. {
            mass += p
                .iter()
                .zip(&r.valid)
                .filter(|(_, v)| **v)
                .map(|(p, _)| *p)
                .sum::<f64>();
        }
        let rebuilt;
        let inputs = if cached {
            if r.successors.get().is_none() {
                let _ = r.successors.set(successor_inputs(r, model)?);
            }
            r.successors.get().unwrap()
        } else {
            rebuilt = successor_inputs(r, model)?;
            &rebuilt
        };
        for (l, (sign, state)) in logits.iter_mut().zip(inputs) {
            let q = if state.is_empty() {
                *sign
            } else {
                let v = model.embed(state).value;
                if *sign == 1. {
                    v
                } else {
                    -v
                }
            };
            *l += beta * q;
        }
        coupled.push(r.example.value == 1. && r.valid[best(&logits)]);
        let k = (r.example.value as i8 + 1) as usize;
        errors[k] += (e.value - r.example.value).powi(2);
        counts[k] += 1;
    }
    let value_mse = errors
        .iter()
        .zip(counts)
        .filter(|(_, n)| *n > 0)
        .map(|(e, n)| e / n as f64)
        .sum::<f64>()
        / counts.iter().filter(|n| **n > 0).count() as f64;
    Ok(Score {
        raw,
        coupled,
        mass: mass / rows.iter().filter(|r| r.example.value == 1.).count().max(1) as f64,
        value_mse,
        priors: Default::default(),
        coupled_logits: Default::default(),
        errors: Default::default(),
    })
}
