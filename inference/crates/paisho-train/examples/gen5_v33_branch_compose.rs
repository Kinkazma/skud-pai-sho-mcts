//! Diagnostic composition of already-separated V5 branches, never a campaign activation.
use paisho_ai::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::json;
use std::fs;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("POLICY_ARTIFACT VALUE_ARTIFACT OUTPUT".into());
    }
    let policy: MicroArtifact = serde_json::from_slice(&fs::read(&args[1])?)?;
    let value: MicroArtifact = serde_json::from_slice(&fs::read(&args[2])?)?;
    assert_eq!(policy.parameters.len(), 93071);
    assert_eq!(value.parameters.len(), 93071);
    assert_eq!(
        serde_json::to_value(&policy.sequence_memory)?,
        serde_json::to_value(&value.sequence_memory)?
    );
    let p = policy.model()?;
    let start = (MICRO_INPUTS + 1) * MICRO_HIDDEN;
    let weights = policy
        .parameters
        .iter()
        .zip(&value.parameters)
        .enumerate()
        .map(|(i, (p, v))| {
            if (start..=start + MICRO_HIDDEN).contains(&i) || i >= MICRO_VALUE_TRUNK {
                *v
            } else {
                *p
            }
        })
        .collect();
    let model = MicroModel::from_parameters(weights)?
        .with_sequence_memory(p.sequence_memory().unwrap().clone());
    let artifact = MicroArtifact::new(
        &model,
        value.updates,
        json!({"diagnostic_only":true,"kind":"independent-v5-branches","policy_parent":policy.identity(),"value_parent":value.identity(),"not_resume_ready":true}),
    );
    fs::write(&args[3], serde_json::to_vec(&artifact)?)?;
    println!(
        "{}",
        json!({"identity":artifact.identity(),"parameters":artifact.parameters.len(),"production_writes":0})
    );
    Ok(())
}
