//! Offline checkpoint census. No training, search, campaign write or mutable proof catalogue.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let specs = manifest["models"].as_array().ok_or("models")?;
    let mut models: Vec<MicroModel> = vec![];
    let mut identities = vec![];
    for m in specs {
        let a: MicroArtifact =
            serde_json::from_slice(&fs::read(m["path"].as_str().ok_or("path")?)?)?;
        let model = a.model()?;
        assert_eq!(model.parameters().len(), 93071);
        if let Some(first) = models.first() {
            assert!(std::sync::Arc::ptr_eq(
                first.sequence_memory().unwrap(),
                model.sequence_memory().unwrap()
            ));
        }
        identities.push(json!({"identity":a.identity(),"updates":a.updates}));
        models.push(model);
    }
    let mut positions = vec![];
    let mut results = vec![];
    for (index, spec) in manifest["positions"]
        .as_array()
        .ok_or("positions")?
        .iter()
        .enumerate()
    {
        let (record, cert, source): (GameRecord, Option<MicroProofCertificate>, String) =
            if spec["kind"] == "certificate" {
                let v: Value =
                    serde_json::from_slice(&fs::read(spec["path"].as_str().ok_or("path")?)?)?;
                (
                    v["prefix"].as_str().ok_or("prefix")?.parse()?,
                    Some(serde_json::from_value(v["certificate"].clone())?),
                    v["human_source"].as_str().unwrap_or("").to_owned(),
                )
            } else {
                (
                    fs::read_to_string(spec["path"].as_str().ok_or("path")?)?.parse()?,
                    None,
                    format!("alias-{}", index / 2),
                )
            };
        let p = record.replay()?;
        assert_eq!(p.rule_profile(), RuleProfileId::SkudPaiShoGen5V1);
        assert_eq!(p.outcome(), GameOutcome::Ongoing);
        let actions = legal_actions(&p);
        let features = actions
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect::<Vec<_>>();
        let mut certified = vec![false; actions.len()];
        let mut target = None;
        if let Some(c) = &cert {
            c.verify(&p)?;
            let sign = if p.to_move() == Player::Host { 1 } else { -1 };
            target = Some(c.outcome as f64 * sign as f64);
            if c.outcome == sign {
                for (a, child) in &c.children {
                    if child.outcome == sign {
                        let action: Action = a.parse()?;
                        certified[actions
                            .iter()
                            .position(|a| *a == action)
                            .ok_or("illegal certificate")?] = true;
                    }
                }
                assert!(certified.iter().any(|b| *b));
            }
        }
        let successors = actions
            .iter()
            .map(|a| {
                let mut n = p.clone();
                n.apply(*a).unwrap();
                n
            })
            .collect::<Vec<_>>();
        let immediate = successors
            .iter()
            .map(|n| n.outcome() == GameOutcome::Win(p.to_move()))
            .collect::<Vec<_>>();
        let successor_features = successors
            .iter()
            .map(|n| models[0].state_features(n))
            .collect::<Vec<_>>();
        let state = models[0].state_features(&p);
        assert_eq!(state.len(), 417);
        let best = |v: &[f64]| {
            (0..v.len())
                .max_by(|&a, &b| v[a].total_cmp(&v[b]).then_with(|| b.cmp(&a)))
                .unwrap()
        };
        for (mi, model) in models.iter().enumerate() {
            let e = model.embed(&state);
            let pure = micro_softmax(&MicroModel::logits(&e, &features))?;
            let prior = model.memory_priors(&state, &features, &pure, 0)?;
            let pick = best(&prior);
            let coupled = if specs[mi]["coupled"] == true {
                let logits = prior
                    .iter()
                    .zip(&successors)
                    .zip(&successor_features)
                    .map(|((prob, n), s)| {
                        let q = match n.outcome() {
                            GameOutcome::Win(w) => {
                                if w == p.to_move() {
                                    1.
                                } else {
                                    -1.
                                }
                            }
                            GameOutcome::Draw => 0.,
                            _ => {
                                model.embed(s).value
                                    * if n.to_move() == p.to_move() { 1. } else { -1. }
                            }
                        };
                        prob.max(1e-300).ln() + 16. * q
                    })
                    .collect::<Vec<_>>();
                Some(best(&logits))
            } else {
                None
            };
            results.push(json!({"model":mi,"position":index,"value":e.value,
                "certified_mass":prior.iter().zip(&certified).filter(|(_,b)|**b).map(|(v,_)|v).sum::<f64>(),
                "certified_selected":certified[pick],"pure_certified_selected":certified[best(&pure)],
                "immediate_selected":immediate[pick],"coupled_certified_selected":coupled.map(|i|certified[i]),
                "coupled_immediate_selected":coupled.map(|i|immediate[i]),
                "selected":actions[pick].to_string(),"coupled_selected":coupled.map(|i|actions[i].to_string())}));
        }
        positions.push(json!({"source":source,"target":target,"legal":actions.len(),"certified_actions":certified.iter().filter(|b|**b).count(),
            "immediate_actions":immediate.iter().filter(|b|**b).count(),"prefix_sha256":format!("{:x}",Sha256::digest(record.to_string().as_bytes()))}));
        if index % 32 == 0 {
            eprintln!("positions verified/evaluated: {}", index + 1);
        }
    }
    fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"identities":identities,"positions":positions,"results":results,
        "shared_bank":true,"new_games":0,"learning_updates":0}),
        )?,
    )?;
    Ok(())
}
