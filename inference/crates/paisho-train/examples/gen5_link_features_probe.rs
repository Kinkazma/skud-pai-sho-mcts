//! Export frozen policy and action-dependent deep-value signals for proposal tests.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("MODEL DISTILLATION_RESULTS OLD_PANEL OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&a[1])?)?;
    let model = artifact.model()?;
    let trial: Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let old: Value = serde_json::from_slice(&fs::read(&a[3])?)?;
    let mut inputs: HashMap<String, (GameRecord, Option<Value>)> = HashMap::new();
    for t in trial["teacher_searches"].as_array().unwrap() {
        let r: GameRecord = t["prefix"].as_str().unwrap().parse()?;
        inputs.insert(hash(r.to_string().as_bytes()), (r, Some(t.clone())));
    }
    for t in old["positions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["kind"] == "certificate")
    {
        let b = fs::read(t["path"].as_str().unwrap())?;
        assert_eq!(hash(&b), t["sha256"]);
        let d: Value = serde_json::from_slice(&b)?;
        let r: GameRecord = d["prefix"].as_str().unwrap().parse()?;
        inputs
            .entry(hash(r.to_string().as_bytes()))
            .or_insert((r, Some(d)));
    }
    let mut rows = vec![];
    for meta in trial["rows"].as_array().unwrap() {
        let (record, data) = &inputs[meta["prefix"].as_str().unwrap()];
        let p = record.replay()?;
        let data = data.as_ref().unwrap();
        let cert: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
        assert_eq!(cert.verify(&p)?, GameOutcome::Win(p.to_move()));
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let actions = legal_actions(&p);
        let state = model.state_features(&p);
        let features: Vec<_> = actions
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect();
        let e = model.embed(&state);
        let prior = model.memory_priors(
            &state,
            &features,
            &micro_softmax(&MicroModel::logits(&e, &features))?,
            0,
        )?;
        let mut valid = vec![false; actions.len()];
        for (a, c) in &cert.children {
            if c.outcome == sign {
                let a: Action = a.parse()?;
                valid[actions.iter().position(|x| *x == a).unwrap()] = true;
            }
        }
        if let Some(v) = data["valid"].as_array() {
            for (i, v) in v.iter().enumerate() {
                valid[i] |= v.as_bool().unwrap();
            }
        }
        let values: Vec<_> = actions
            .iter()
            .map(|a| {
                let mut n = p.clone();
                n.apply(*a).unwrap();
                match n.outcome() {
                    GameOutcome::Win(w) => {
                        if w == p.to_move() {
                            1.
                        } else {
                            -1.
                        }
                    }
                    GameOutcome::Draw => 0.,
                    _ => {
                        let v = model.embed(&model.state_features(&n)).value;
                        if n.to_move() == p.to_move() {
                            v
                        } else {
                            -v
                        }
                    }
                }
            })
            .collect();
        let target = if data["target"].is_array() {
            data["target"].clone()
        } else {
            let n = valid.iter().filter(|v| **v).count() as f64;
            json!(valid
                .iter()
                .map(|v| if *v { 1. / n } else { 0. })
                .collect::<Vec<_>>())
        };
        rows.push(json!({"meta":meta,"state":state,"prior":prior,"value":e.value,"action_features":features,"successor_values":values,"valid":valid,"target":target}));
    }
    fs::write(
        &a[4],
        serde_json::to_vec(
            &json!({"model_identity":artifact.identity(),"rows":rows,"production_writes":0}),
        )?,
    )?;
    Ok(())
}
