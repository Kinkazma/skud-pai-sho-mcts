//! Bounded frozen-clone interventions, with a source-disjoint diagnostic split.
//! The archive was exposed to campaign training; this is not a held-out Elo test.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, sync::Arc};

struct Row {
    example: MicroExample,
    train: bool,
    source: String,
    index: usize,
}
fn metrics(model: &MicroModel, rows: &[Row], train: bool) -> Value {
    let mut groups = vec![];
    for target in [-1., 1.] {
        let examples: Vec<_> = rows
            .iter()
            .filter(|r| r.train == train && r.example.value == target)
            .collect();
        let predictions: Vec<_> = examples
            .iter()
            .map(|r| model.embed(&r.example.state).value)
            .collect();
        groups.push(json!({"target":target,"n":predictions.len(),
            "mse":predictions.iter().map(|v|(v-target).powi(2)).sum::<f64>()/predictions.len() as f64,
            "mean_value":predictions.iter().sum::<f64>()/predictions.len() as f64,
            "wrong_sign":predictions.iter().filter(|v|**v*target<=0.).count()}));
    }
    json!(groups)
}
fn policy_metrics(model: &MicroModel, rows: &[Row], train: bool) -> Value {
    let mut loss = 0.;
    let mut top = 0;
    let mut mass = 0.;
    let mut n = 0;
    for r in rows
        .iter()
        .filter(|r| r.train == train && r.example.value == 1.)
    {
        let ex = &r.example;
        let e = model.embed(&ex.state);
        let p = model
            .memory_priors(
                &ex.state,
                &ex.actions,
                &micro_softmax(&MicroModel::logits(&e, &ex.actions)).unwrap(),
                0,
            )
            .unwrap();
        let best = (0..p.len())
            .max_by(|a, b| p[*a].total_cmp(&p[*b]).then_with(|| b.cmp(a)))
            .unwrap();
        top += usize::from(ex.policy[best] > 0.);
        n += 1;
        loss -= p
            .iter()
            .zip(&ex.policy)
            .filter(|(_, t)| **t > 0.)
            .map(|(v, t)| t * v.ln())
            .sum::<f64>();
        mass += p
            .iter()
            .zip(&ex.policy)
            .filter(|(_, t)| **t > 0.)
            .map(|(v, _)| v)
            .sum::<f64>();
    }
    json!({"n":n,"cross_entropy":loss/n as f64,"certified_top":top,"certified_mass":mass/n as f64})
}
fn aliases(
    original: &MicroModel,
    manifest: &Value,
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let mut alias_rows = vec![];
    for (index, spec) in manifest["positions"].as_array().unwrap().iter().enumerate() {
        if spec["kind"] == "certificate" {
            continue;
        }
        let record: GameRecord = fs::read_to_string(spec["path"].as_str().unwrap())?.parse()?;
        let p = record.replay()?;
        let actions = legal_actions(&p);
        let mut policy = actions
            .iter()
            .map(|a| {
                let mut next = p.clone();
                next.apply(*a).unwrap();
                f64::from(next.outcome() == GameOutcome::Win(p.to_move()))
            })
            .collect::<Vec<_>>();
        let mass: f64 = policy.iter().sum();
        if mass == 0. {
            continue;
        }
        for x in &mut policy {
            *x /= mass;
        }
        alias_rows.push(Row {
            train: true,
            source: format!("witness-{index}"),
            index,
            example: MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
                state: original.state_features(&p),
                actions: actions
                    .iter()
                    .map(|a| micro_action_features(&p, *a))
                    .collect(),
                policy,
                value: 1.,
                policy_weight: 1.,
                value_weight: 1.0, sequence_source: 0,
            },
        });
    }
    assert_eq!(alias_rows.len(), 4);
    Ok(alias_rows)
}
fn alias_fit(
    original: &MicroModel,
    alias_rows: &[Row],
    rows: &[Row],
    mixed: bool,
) -> Result<Vec<Value>, Box<dyn std::error::Error>> {
    let wins: Vec<_> = rows
        .iter()
        .filter(|r| r.train && r.example.value == 1.)
        .collect();
    let mut alias_trials = vec![];
    for seed in [37, 913, 4421] {
        let mut rng = StableRng::new(seed);
        let mut old_rng = StableRng::new(seed ^ 0x726563616c6c);
        let mut model = original.clone();
        let mut trace = vec![];
        for step in 0..=32768 {
            if [0, 64, 256, 1024, 4096, 8192, 16384, 32768].contains(&step) {
                trace.push(json!({"step":step,"witnesses":policy_metrics(&model,&alias_rows,true),"old_archive":policy_metrics(&model,&rows,false)}));
            }
            if step == 32768 {
                break;
            }
            let i = rng.index(alias_rows.len());
            let ex = if mixed && step % 2 == 1 {
                &wins[old_rng.index(wins.len())].example
            } else {
                &alias_rows[i].example
            };
            model.train_policy_step(ex, 0.02 / 64.)?;
        }
        alias_trials.push(json!({"seed":seed,"trace":trace,"old_archive_after":policy_metrics(&model,&rows,false)}));
    }
    Ok(alias_trials)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if ![4, 5].contains(&args.len()) {
        return Err("MODEL PANEL_MANIFEST OUTPUT [mixed]".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&args[1])?)?;
    let original = artifact.model()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut rows = vec![];
    for (index, spec) in manifest["positions"]
        .as_array()
        .ok_or("positions")?
        .iter()
        .enumerate()
    {
        if spec["kind"] != "certificate" {
            continue;
        }
        let data: Value = serde_json::from_slice(&fs::read(spec["path"].as_str().ok_or("path")?)?)?;
        let source = data["human_source"]
            .as_str()
            .ok_or("human_source")?
            .to_owned();
        let train = Sha256::digest(source.as_bytes())[0] % 3 != 0;
        let record: GameRecord = data["prefix"].as_str().ok_or("prefix")?.parse()?;
        let p = record.replay()?;
        let cert: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
        cert.verify(&p)?;
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let value = (cert.outcome * sign) as f64;
        if value == 0. {
            continue;
        }
        let actions = legal_actions(&p);
        let mut policy = vec![0.; actions.len()];
        if value == 1. {
            for (a, c) in cert.children {
                if c.outcome == sign {
                    let a: Action = a.parse()?;
                    policy[actions
                        .iter()
                        .position(|x| *x == a)
                        .ok_or("illegal proof")?] = 1.;
                }
            }
        }
        let mass: f64 = policy.iter().sum();
        if mass > 0. {
            for p in &mut policy {
                *p /= mass;
            }
        }
        rows.push(Row {
            train,
            source,
            index,
            example: MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
                state: original.state_features(&p),
                actions: actions
                    .iter()
                    .map(|a| micro_action_features(&p, *a))
                    .collect(),
                policy,
                value,
                policy_weight: f64::from(value == 1.),
                value_weight: 1.0, sequence_source: 0,
            },
        });
    }
    let alias_rows = aliases(&original, &manifest)?;
    if args.get(4).map(String::as_str) == Some("mixed") {
        let mut output: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
        output["alias_mixed_trials"] = json!(alias_fit(&original, &alias_rows, &rows, true)?);
        fs::write(&args[3], serde_json::to_vec(&output)?)?;
        return Ok(());
    }
    let wins: Vec<_> = rows
        .iter()
        .filter(|r| r.train && r.example.value == 1.)
        .collect();
    let losses: Vec<_> = rows
        .iter()
        .filter(|r| r.train && r.example.value == -1.)
        .collect();
    assert!(!wins.is_empty() && !losses.is_empty());
    let mut trials = vec![];
    for seed in [37, 913, 4421] {
        for positive_only in [false, true] {
            let mut rng = StableRng::new(seed);
            let mut model = original.clone();
            let mut trace = vec![];
            for step in 0..=96 {
                if step % 16 == 0 {
                    trace.push(json!({"step":step,"train":metrics(&model,&rows,true),"validation":metrics(&model,&rows,false)}));
                }
                if step == 96 {
                    break;
                }
                let mut batch = vec![];
                // Same 48 balanced base examples and RNG calls in both arms.
                // The final quarter contains either sixteen wins or eight of each sign.
                for slot in 0..64 {
                    let wi = rng.index(wins.len());
                    let li = rng.index(losses.len());
                    let choose_win = slot % 2 == 0 || (slot >= 48 && positive_only);
                    let mut ex = if choose_win {
                        wins[wi].example.clone()
                    } else {
                        losses[li].example.clone()
                    };
                    ex.policy_weight = 0.;
                    ex.structured.clear();ex.actions.clear();
                    ex.policy.clear();
                    batch.push(ex);
                }
                // Actual SGD/clipping implementation. L2 omitted equally in both arms
                // to isolate the contribution of value supervision to interference.
                model.train_batch_inline(&batch.iter().collect::<Vec<_>>(), 0.02, 0.)?;
            }
            trials.push(json!({"seed":seed,"positive_only_quarter":positive_only,"trace":trace}));
        }
    }
    let mut policy_trials = vec![];
    for seed in [37, 913, 4421] {
        let mut rng = StableRng::new(seed);
        let mut model = original.clone();
        let mut trace = vec![];
        for step in 0..=128 {
            if step % 16 == 0 {
                trace.push(json!({"step":step,"train":policy_metrics(&model,&rows,true),"validation":policy_metrics(&model,&rows,false)}));
            }
            if step == 128 {
                break;
            }
            // Only verified positive policies; no search and no empirical labels.
            for _ in 0..8 {
                model.train_policy_step(&wins[rng.index(wins.len())].example, 0.02 / 64.)?;
            }
        }
        policy_trials.push(json!({"seed":seed,"trace":trace}));
    }
    let alias_trials = alias_fit(&original, &alias_rows, &rows, false)?;
    let original_arc = Arc::new(original);
    fs::write(
        &args[3],
        serde_json::to_vec(&json!({"model_parameters":original_arc.parameters().len(),
        "source_split":rows.iter().map(|r|json!({"index":r.index,"source":r.source,"train":r.train,"target":r.example.value})).collect::<Vec<_>>(),
        "value_trials":trials,"policy_trials":policy_trials,"alias_fit_trials":alias_trials,"production_writes":0,
        "scope":"source-disjoint diagnostic intervention on campaign-exposed archive; not a campaign replay or held-out strength estimate"}))?,
    )?;
    Ok(())
}
