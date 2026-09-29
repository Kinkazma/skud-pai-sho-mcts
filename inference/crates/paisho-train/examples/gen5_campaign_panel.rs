//! Read-only fixed-position checkpoint audit; no training or new match generation.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use std::{fs, path::Path};

fn depth(c: &MicroProofCertificate) -> usize {
    c.children.iter().map(|(_, c)| depth(c) + 1).max().unwrap_or(0)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 { return Err("MANIFEST OUTPUT".into()); }
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut models = Vec::new();
    let mut identities = Vec::new();
    // Keep the bank resident: deserialize once per artifact, construct once;
    // shared immutable banks remain alive throughout the checkpoint panel.
    for m in manifest["models"].as_array().ok_or("models")? {
        let a: MicroArtifact = serde_json::from_slice(&fs::read(m["path"].as_str().ok_or("path")?)?)?;
        let model = a.model()?;
        assert_eq!(model.parameters().len(), 93071);
        if let Some(first) = models.first() {
            let first: &MicroModel = first;
            assert!(std::sync::Arc::ptr_eq(first.sequence_memory().unwrap(), model.sequence_memory().unwrap()));
        }
        identities.push(json!({"identity":a.identity(), "updates":a.updates}));
        models.push(model);
    }
    let mut positions = Vec::new();
    let mut results = Vec::new();
    for (index, spec) in manifest["positions"].as_array().ok_or("positions")?.iter().enumerate() {
        let path = Path::new(spec["path"].as_str().ok_or("path")?);
        let (record, cert, source): (GameRecord, Option<MicroProofCertificate>, String) =
            if spec["kind"] == "certificate" {
                let v: Value = serde_json::from_slice(&fs::read(path)?)?;
                (v["prefix"].as_str().ok_or("prefix")?.parse()?,
                 Some(serde_json::from_value(v["certificate"].clone())?),
                 v["human_source"].as_str().unwrap_or("").to_owned())
            } else { (fs::read_to_string(path)?.parse()?, None, format!("alias-{}", index / 2)) };
        let p = record.replay()?;
        assert_eq!(p.rule_profile(), RuleProfileId::SkudPaiShoGen5V1);
        assert_eq!(p.outcome(), GameOutcome::Ongoing);
        let actions = legal_actions(&p);
        let features = actions.iter().map(|a| micro_action_features(&p, *a)).collect::<Vec<_>>();
        let mut certified = vec![false; actions.len()];
        let mut target = None;
        if let Some(c) = &cert {
            c.verify(&p).map_err(|e| format!("{}: {e}", path.display()))?;
            let sign = if p.to_move() == Player::Host { 1 } else { -1 };
            target = Some(c.outcome as f64 * sign as f64);
            if c.outcome == sign {
                for (a, child) in &c.children {
                    if child.outcome == sign {
                        let action: Action = a.parse()?;
                        certified[actions.iter().position(|a| *a == action).ok_or("illegal certificate child")?] = true;
                    }
                }
                assert!(certified.iter().any(|x| *x));
            }
        }
        let immediate = actions.iter().map(|a| {
            let mut next = p.clone(); next.apply(*a).unwrap();
            next.outcome() == GameOutcome::Win(p.to_move())
        }).collect::<Vec<_>>();
        if cert.is_none() && immediate.iter().any(|x| *x) { target = Some(1.); }
        let state = models[0].state_features(&p);
        assert_eq!(state.len(), 417);
        for (mi, model) in models.iter().enumerate() {
            let embedding = model.embed(&state);
            let pure = micro_softmax(&MicroModel::logits(&embedding, &features))?;
            let prior = model.memory_priors(&state, &features, &pure, 0)?;
            let pick = (0..actions.len()).max_by(|a,b| prior[*a].total_cmp(&prior[*b]).then_with(|| b.cmp(a))).unwrap();
            results.push(json!({"model":mi,"position":index,"value":embedding.value,
                "certified_mass":prior.iter().zip(&certified).filter(|(_, ok)| **ok).map(|(v,_)| v).sum::<f64>(),
                "immediate_mass":prior.iter().zip(&immediate).filter(|(_, ok)| **ok).map(|(v,_)| v).sum::<f64>(),
                "certified_selected":certified[pick],"immediate_selected":immediate[pick],
                "selected":actions[pick].to_string(),"selected_prior":prior[pick],
                "entropy":-prior.iter().filter(|&&v|v>0.).map(|v|v*v.ln()).sum::<f64>()}));
        }
        positions.push(json!({"source":source,"target":target,"legal":actions.len(),
            "certified_actions":certified.iter().filter(|x|**x).count(),
            "immediate_actions":immediate.iter().filter(|x|**x).count(),
            "certificate_depth":cert.as_ref().map(depth),"chooser":format!("{:?}",p.to_move())}));
        if index % 32 == 0 { eprintln!("verified/evaluated position {}", index + 1); }
    }
    let baseline = models[0].parameters();
    let drift = models.iter().map(|m| {
        let changes = m.parameters().iter().zip(baseline).map(|(a,b)| a-b).collect::<Vec<_>>();
        json!({"changed":changes.iter().filter(|&&x| x!=0.).count(),
            "rms":(changes.iter().map(|x|x*x).sum::<f64>()/changes.len() as f64).sqrt(),
            "max":changes.iter().map(|x|x.abs()).fold(0.,f64::max)})
    }).collect::<Vec<_>>();
    fs::write(&args[2], serde_json::to_vec(&json!({"identities":identities,"parameter_drift":drift,
        "positions":positions,"results":results,"shared_bank":true,"new_games":0,"learning_updates":0}))?)?;
    Ok(())
}
