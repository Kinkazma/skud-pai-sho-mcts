//! Frozen native diagnostic for optional root value coupling.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use std::{fs, sync::Arc};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 6 {
        return Err("MODEL DISTILLATION_RESULTS OPENING_MANIFEST BETA OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&a[1])?)?;
    let model = Arc::new(artifact.model()?);
    let teachers: Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let openings: Value = serde_json::from_slice(&fs::read(&a[3])?)?;
    let beta: f64 = a[4].parse()?;
    let mut specs = teachers["teacher_searches"]
        .as_array()
        .ok_or("teachers")?
        .clone();
    specs.extend(openings["games"].as_array().ok_or("games")?.iter().cloned());
    let mut rows = Vec::new();
    for spec in specs {
        let record: GameRecord = spec["prefix"].as_str().ok_or("prefix")?.parse()?;
        let p = record.replay()?;
        let mut session = MicroMctsSession::new(model.clone());
        session.set_root_value_strength(beta)?;
        let r = session.search_with_options(
            &p,
            512,
            None,
            MicroSearchOptions {
                proof_search: true,
                seed: 37,
                ..Default::default()
            },
        )?;
        let c = session.certificate(10000);
        if let Some(c) = &c {
            c.verify(&p)?;
        }
        let selected = r.actions[r.selected_index].to_string();
        let old_verified = spec["actions"]
            .as_array()
            .zip(spec["valid"].as_array())
            .map_or(false, |(aa, vv)| {
                aa.iter()
                    .zip(vv)
                    .any(|(a, v)| a.as_str() == Some(&selected) && v == true)
            });
        let fresh_verified = r.proven_action_values[r.selected_index] == Some(1);
        rows.push(json!({"source":spec["source"],"cohort":spec["cohort"],"selected":selected,
            "known_verified_win":old_verified,"fresh_verified_win":fresh_verified,
            "actions":r.actions.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "priors":r.priors,"search_priors":r.search_priors,"target":r.policy_target,
            "visits":r.visits,"values":r.values,"proof":r.proven_value,"proof_actions":r.proven_action_values,
            "simulations":r.simulations,"tactical_evaluations":r.tactical_evaluations,
            "evaluations":r.inference_evaluations,"cache_hits":r.inference_cache_hits,"certificate":c}));
    }
    fs::write(
        &a[5],
        serde_json::to_vec(&json!({"beta":beta,"model_identity":artifact.identity(),"rows":rows}))?,
    )?;
    Ok(())
}
