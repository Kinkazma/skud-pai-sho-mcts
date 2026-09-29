//! Frozen analytical diagnostics. No production writes or learning campaign.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use std::{fs, path::Path};
fn norm(g: &[f64]) -> f64 {
    g.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn activation(a: &[f64]) -> Value {
    json!({"n":a.len(),"saturated99":a.iter().filter(|x|x.abs()>0.99).count(),"mean_derivative":a.iter().map(|x|1.-x*x).sum::<f64>()/a.len() as f64})
}
fn local(board: &[f64], a: &[f64; 32]) -> [f64; 18] {
    let mut out = [0.; 18];
    for (ep, (offset, present)) in [(18, a[26] != 0.), (20, a[2] == 0.)]
        .into_iter()
        .enumerate()
    {
        if !present {
            continue;
        }
        let cx = (a[offset] * 8.).round() as i32 + 8;
        let cy = (a[offset + 1] * 8.).round() as i32 + 8;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y) = (cx + dx, cy + dy);
                if (0..17).contains(&x) && (0..17).contains(&y) {
                    out[ep * 9 + ((dy + 1) * 3 + dx + 1) as usize] = board[(y * 17 + x) as usize];
                }
            }
        }
    }
    out
}
fn layer(x: &[f64], w: &[f64], start: usize, bias: usize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            (w[bias + j]
                + x.iter()
                    .zip(&w[start + j * x.len()..start + (j + 1) * x.len()])
                    .map(|(a, b)| a * b)
                    .sum::<f64>())
            .tanh()
        })
        .collect()
}
fn forward(model: &MicroModel, ex: &MicroExample) -> (f64, Vec<f64>) {
    let e = model.embed(&ex.state);
    let prior = if ex.actions.is_empty() {
        vec![]
    } else {
        model
            .memory_priors(
                &ex.state,
                &ex.actions,
                &micro_softmax(&MicroModel::logits(&e, &ex.actions)).unwrap(),
                ex.sequence_source,
            )
            .unwrap()
    };
    (e.value, prior)
}
fn scalar_loss(model: &MicroModel, ex: &MicroExample) -> f64 {
    let (v, p) = forward(model, ex);
    0.5 * (v - ex.value).powi(2)
        - ex.policy_weight
            * p.iter()
                .zip(&ex.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(p, t)| t * p.ln())
                .sum::<f64>()
}
fn clone_weights(model: &MicroModel, w: Vec<f64>) -> MicroModel {
    MicroModel::from_parameters(w)
        .unwrap()
        .with_sequence_memory(model.sequence_memory().unwrap().clone())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("MODELS PANEL_MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let specs: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let panel: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut models = vec![];
    for m in specs.as_array().ok_or("models")? {
        let a: MicroArtifact =
            serde_json::from_slice(&fs::read(m["path"].as_str().ok_or("path")?)?)?;
        models.push(a.model()?);
    }
    let mut output = vec![];
    let mut examples = vec![];
    let mut info = vec![];
    let mut finite_checks = vec![];
    for (index, spec) in panel["positions"]
        .as_array()
        .ok_or("positions")?
        .iter()
        .enumerate()
    {
        let path = Path::new(spec["path"].as_str().ok_or("path")?);
        let (record, cert): (GameRecord, Option<MicroProofCertificate>) =
            if spec["kind"] == "certificate" {
                let d: Value = serde_json::from_slice(&fs::read(path)?)?;
                (
                    d["prefix"].as_str().ok_or("prefix")?.parse()?,
                    Some(serde_json::from_value(d["certificate"].clone())?),
                )
            } else {
                (fs::read_to_string(path)?.parse()?, None)
            };
        let p = record.replay()?;
        let actions = legal_actions(&p);
        let features = actions
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect::<Vec<_>>();
        let immediate = actions
            .iter()
            .map(|a| {
                let mut n = p.clone();
                n.apply(*a).unwrap();
                n.outcome() == GameOutcome::Win(p.to_move())
            })
            .collect::<Vec<_>>();
        let mut target = vec![0.; actions.len()];
        let value;
        if let Some(cert) = cert {
            cert.verify(&p)?;
            let sign = if p.to_move() == Player::Host { 1 } else { -1 };
            value = (cert.outcome * sign) as f64;
            if value >= 0. {
                for (a, c) in cert.children {
                    if c.outcome == cert.outcome {
                        let a: Action = a.parse()?;
                        target[actions.iter().position(|x| *x == a).unwrap()] = 1.;
                    }
                }
            }
        } else if immediate.iter().any(|v| *v) {
            value = 1.;
            for (t, w) in target.iter_mut().zip(&immediate) {
                *t = f64::from(*w);
            }
        } else {
            continue;
        }
        let mass: f64 = target.iter().sum();
        if mass > 0. {
            for t in &mut target {
                *t /= mass;
            }
        }
        let ex = MicroExample { policy_support: false, action_values: vec![], 
            state: models[0].state_features(&p),
            actions: if mass > 0. { features } else { vec![] },
            policy: if mass > 0. { target } else { vec![] },
            value,
            policy_weight: if mass > 0. { 1. } else { 0. },
            value_weight: 1.0, sequence_source: 0,
        };
        info.push(json!({"position":index,"target":value,"immediate":immediate.iter().filter(|v|**v).count()}));
        for (mi, model) in models.iter().enumerate() {
            let w = model.parameters();
            let emb = model.embed(&ex.state);
            let (loss, g) = model.loss_gradient(&ex)?;
            let mut vex = ex.clone();
            vex.policy_weight = 0.;
            let (_, vg) = model.loss_gradient(&vex)?;
            let pg = g.iter().zip(&vg).map(|(a, b)| a - b).collect::<Vec<_>>();
            assert_eq!(pg.iter().zip(&vg).map(|(a, b)| a * b).sum::<f64>(), 0.);
            let mut residual = vec![];
            for a in &ex.actions {
                let local = local(&ex.state[128..], a);
                let h: Vec<f64> = (0..16)
                    .map(|j| {
                        let context = w[6241 + j]
                            + emb
                                .hidden
                                .iter()
                                .zip(&w[5729 + j * 32..5729 + (j + 1) * 32])
                                .map(|(x, y)| x * y)
                                .sum::<f64>();
                        (context
                            + a.iter()
                                .zip(&w[5217 + j * 32..5217 + (j + 1) * 32])
                                .map(|(x, y)| x * y)
                                .sum::<f64>()
                            + local
                                .iter()
                                .zip(
                                    &w[MICRO_SPATIAL_LOCAL + j * 18
                                        ..MICRO_SPATIAL_LOCAL + (j + 1) * 18],
                                )
                                .map(|(x, y)| x * y)
                                .sum::<f64>())
                        .tanh()
                    })
                    .collect();
                let raw = a
                    .iter()
                    .zip(emb.policy_context)
                    .map(|(x, y)| x * y)
                    .sum::<f64>()
                    + w[6273]
                    + h.iter()
                        .zip(&w[6257..6273])
                        .map(|(x, y)| x * y)
                        .sum::<f64>();
                assert!((raw - MicroModel::logit(&emb, a)).abs() < 1e-10);
                residual.extend(h);
            }
            let one = layer(&ex.state, w, 29198, 82574, 128);
            let two = layer(&one, w, 82702, 90894, 64);
            let three = layer(&two, w, 90958, 93006, 32);
            let (_, prior) = forward(model, &ex);
            let mut neutral = w.to_vec();
            neutral[93038..].fill(0.);
            let neutral = clone_weights(model, neutral);
            let (without_deep, neutral_prior) = forward(&neutral, &ex);
            assert_eq!(prior, neutral_prior);
            let mut stepped = model.clone();
            stepped.train_step(&ex, 0.02 / 64., 0.)?;
            let (next_value, next_prior) = forward(&stepped, &ex);
            let immediate_mass = if prior.is_empty() {
                None
            } else {
                Some(
                    prior
                        .iter()
                        .zip(&immediate)
                        .filter(|(_, ok)| **ok)
                        .map(|(v, _)| v)
                        .sum::<f64>(),
                )
            };
            let next_mass = if next_prior.is_empty() {
                None
            } else {
                Some(
                    next_prior
                        .iter()
                        .zip(&immediate)
                        .filter(|(_, ok)| **ok)
                        .map(|(v, _)| v)
                        .sum::<f64>(),
                )
            };
            output.push(json!({"model":mi,"position":index,"value":emb.value,"target":value,"loss_value":loss.value,"loss_policy":loss.policy,
                "gradient_norm":norm(&g),"value_gradient_norm":norm(&vg),"policy_gradient_norm":norm(&pg),"deep_gradient_norm":norm(&g[29198..]),
                "policy_board_gradient_norm":norm(&g[6286..15534]),"policy_local_gradient_norm":norm(&g[15534..15822]),
                "policy_hidden":activation(&emb.hidden),"residual":if residual.is_empty(){Value::Null}else{activation(&residual)},
                "deep_first":activation(&one),"deep_second":activation(&two),"deep_third":activation(&three),"value_without_deep":without_deep,
                "one_step_value":next_value,"one_step_loss":scalar_loss(&stepped,&ex),"immediate_mass":immediate_mass,"one_step_immediate_mass":next_mass}));
            if mi == models.len() - 1 && index < 4 {
                for (start, end) in [
                    (0, 4096),
                    (4096, 4128),
                    (4128, 4161),
                    (4161, 5217),
                    (5217, 6274),
                    (6274, 6286),
                    (6286, 15534),
                    (15534, 15822),
                    (15822, 29198),
                    (29198, 82574),
                    (82574, 82702),
                    (82702, 90894),
                    (90894, 90958),
                    (90958, 93038),
                    (93038, 93071),
                ] {
                    let k = (start..end)
                        .max_by(|&a, &b| g[a].abs().total_cmp(&g[b].abs()))
                        .unwrap();
                    let eps = 1e-6;
                    let mut a = w.to_vec();
                    a[k] += eps;
                    let mut b = w.to_vec();
                    b[k] -= eps;
                    let numeric = (scalar_loss(&clone_weights(model, a), &ex)
                        - scalar_loss(&clone_weights(model, b), &ex))
                        / (2. * eps);
                    assert!((numeric - g[k]).abs() < 2e-7);
                    finite_checks.push(
                        json!({"position":index,"coordinate":k,"analytic":g[k],"numeric":numeric}),
                    );
                }
            }
        }
        examples.push(ex);
        if index % 64 == 0 {
            eprintln!("diagnostic positions {}", index + 1);
        }
    }
    // Real-size frozen batches: clipping depends on the averaged gradient, not
    // on per-example norms. Use deterministic panels of 64 examples.
    let mut batches = vec![];
    let model = models.last().unwrap();
    for chunk in examples.chunks(64) {
        let mut g = vec![0.; model.parameters().len()];
        for ex in chunk {
            let (_, gx) = model.loss_gradient(ex)?;
            for (a, b) in g.iter_mut().zip(gx) {
                *a += b / chunk.len() as f64;
            }
        }
        let objective = |m: &MicroModel| {
            chunk.iter().map(|x| scalar_loss(m, x)).sum::<f64>() / chunk.len() as f64
                + 0.5e-5 * m.parameters().iter().map(|x| x * x).sum::<f64>()
        };
        let before = objective(model);
        let mut steps = vec![];
        for rate in [0.02, 0.01, 0.005, 0.0025] {
            let mut candidate = model.clone();
            let refs = chunk.iter().collect::<Vec<_>>();
            candidate.train_batch_inline(&refs, rate * chunk.len() as f64 / 64., 1e-5)?;
            steps.push(json!({"base_rate":rate,"after":objective(&candidate)}));
        }
        batches.push(json!({"n":chunk.len(),"norm":norm(&g),"clipping_scale":(10./norm(&g)).min(1.),"objective_before":before,"steps":steps}));
    }
    fs::write(
        &args[3],
        serde_json::to_vec(
            &json!({"rows":output,"positions":info,"finite_differences":finite_checks,"batches":batches,"policy_value_gradient_overlap":0,"deep_value_changes_policy":false,"production_writes":0}),
        )?,
    )?;
    Ok(())
}
