use super::*;
fn norm(g: &[f64]) -> f64 {
    g.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn mean_gradient(
    model: &MicroModel,
    examples: &[MicroExample],
    policy: bool,
    value: bool,
) -> Result<Vec<f64>> {
    let mut sum = vec![0.; model.parameters().len()];
    for ex in examples {
        let mut ex = ex.clone();
        if !policy {
            ex.policy_weight = 0.;
        }
        if !value {
            ex.value_weight = 0.;
        }
        let (_, g) = model.loss_gradient(&ex)?;
        for (s, g) in sum.iter_mut().zip(g) {
            *s += g / examples.len() as f64;
        }
    }
    Ok(sum)
}
fn average_loss(
    model: &MicroModel,
    examples: &[MicroExample],
    policy: bool,
    value: bool,
) -> Result<f64> {
    let mut sum = 0.;
    for ex in examples {
        let mut ex = ex.clone();
        if !policy {
            ex.policy_weight = 0.;
        }
        if !value {
            ex.value_weight = 0.;
        }
        let (loss, _) = model.loss_gradient(&ex)?;
        sum += loss.total(ex.policy_weight) / examples.len() as f64;
    }
    Ok(sum)
}
// This gradient measures the probability of ANY verified win, rather than an
// arbitrary distribution among certificate children. Immediate wins are included.
fn win_mass_gradient(model: &MicroModel, examples: &[MicroExample]) -> Result<Vec<f64>> {
    let mut targets = vec![];
    for ex in examples {
        let p = prior(model, ex)?;
        let mass: f64 = p
            .iter()
            .zip(&ex.policy)
            .filter(|(_, t)| **t > 0.)
            .map(|(p, _)| p)
            .sum();
        let mut target = ex.clone();
        target.policy = p
            .iter()
            .zip(&ex.policy)
            .map(|(p, t)| if *t > 0. { *p / mass } else { 0. })
            .collect();
        targets.push(target);
    }
    mean_gradient(model, &targets, true, false)
}
fn win_mass_loss(model: &MicroModel, examples: &[MicroExample]) -> Result<f64> {
    let mut total = 0.;
    for ex in examples {
        let p = prior(model, ex)?;
        let mass: f64 = p
            .iter()
            .zip(&ex.policy)
            .filter(|(_, t)| **t > 0.)
            .map(|(p, _)| p)
            .sum();
        total -= mass.max(1e-300).ln() / examples.len() as f64;
    }
    Ok(total)
}
pub fn run(
    model: &MicroModel,
    fresh: &[MicroExample],
    sets: &BTreeMap<String, Vec<MicroExample>>,
) -> Result<Value> {
    let mut gradients = BTreeMap::new();
    gradients.insert("fresh_policy", mean_gradient(model, fresh, true, false)?);
    gradients.insert("fresh_value", mean_gradient(model, fresh, false, true)?);
    gradients.insert(
        "guard_policy",
        mean_gradient(model, &sets["guard_1"], true, false)?,
    );
    gradients.insert(
        "outside_policy",
        mean_gradient(model, &sets["outside_1"], true, false)?,
    );
    gradients.insert(
        "outside_any_win_mass",
        win_mass_gradient(model, &sets["outside_any_1"])?,
    );
    gradients.insert(
        "outside_win_value",
        mean_gradient(model, &sets["outside_1"], false, true)?,
    );
    gradients.insert(
        "outside_loss_value",
        mean_gradient(model, &sets["outside_-1"], false, true)?,
    );
    let mut relations = vec![];
    for (a, ga) in &gradients {
        for (b, gb) in &gradients {
            relations.push(json!({"step_source":a,"test_loss":b,"gradient_dot":dot(ga,gb),"cosine":dot(ga,gb)/(norm(ga)*norm(gb)).max(1e-300),"first_order_loss_change_per_unit_step":-dot(ga,gb)}));
        }
    }
    let mut checks = vec![];
    for source in ["fresh_policy", "fresh_value", "guard_policy"] {
        let g = &gradients[source];
        let step = 1e-5;
        let next = MicroModel::from_parameters(
            model
                .parameters()
                .iter()
                .zip(g)
                .map(|(w, g)| w - step * g)
                .collect(),
        )?
        .with_sequence_memory(model.sequence_memory().unwrap().clone());
        for (name, key, policy, value) in [
            ("outside_policy", "outside_1", true, false),
            ("outside_win_value", "outside_1", false, true),
            ("guard_policy", "guard_1", true, false),
        ] {
            let before = average_loss(model, &sets[key], policy, value)?;
            let after = average_loss(&next, &sets[key], policy, value)?;
            let expected = -step * dot(g, &gradients[name]);
            checks.push(json!({"source":source,"test":name,"before":before,"after":after,"actual_delta":after-before,"first_order_delta":expected,"step":step}));
        }
        let before = win_mass_loss(model, &sets["outside_any_1"])?;
        let after = win_mass_loss(&next, &sets["outside_any_1"])?;
        checks.push(json!({"source":source,"test":"outside_any_win_mass","before":before,"after":after,"actual_delta":after-before,"first_order_delta":-step*dot(g,&gradients["outside_any_win_mass"]),"step":step}));
    }
    let mut batch_norms = vec![];
    for batch in fresh.chunks(64).take(64) {
        let g = mean_gradient(model, batch, true, true)?;
        batch_norms.push(norm(&g));
    }
    Ok(
        json!({"norms":gradients.iter().map(|(name,g)|(*name,norm(g))).collect::<BTreeMap<_,_>>(),
  "relations":relations,"directional_checks":checks,"diagnostic_batch_norms":batch_norms,"batches_clipped":batch_norms.iter().filter(|&&x|x>10.).count(),
  "scope":"analytical means of sampled fresh positions and proof panels, not reconstruction of the live mixed minibatches"}),
    )
}
