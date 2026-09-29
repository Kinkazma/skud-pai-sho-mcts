//! Exercise the actual direct-recall implementation against isolated proof files.
use paisho_train::micro_learning::certificate_action_values;
use paisho_train::micro_learning as action_values;
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const RULES: RuleProfileId = RuleProfileId::SkudPaiShoGen5V1;
fn invalid(s: impl Into<String>) -> Box<dyn std::error::Error> {
    s.into().into()
}
fn sha256(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
#[path = "../src/micro_learning/gen5/durable/winning.rs"]
mod actual_winning;
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("MODEL PROOF_A PROOF_B OUTPUT_DIRECTORY".into());
    }
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&args[1])?)?;
    let model = artifact.model()?;
    let root = Path::new(&args[4]).join(format!("isolated-catalogue-{}", std::process::id()));
    fs::create_dir(&root)?;
    fs::create_dir(root.join("proofs"))?;
    let a: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let b: Value = serde_json::from_slice(&fs::read(&args[3])?)?;
    let record: GameRecord = b["prefix"].as_str().ok_or("prefix")?.parse()?;
    let target = model.state_features(&record.replay()?);
    fs::write(
        root.join("proofs").join(format!(
            "{}.json",
            sha256(a["prefix"].as_str().ok_or("prefix")?.as_bytes())
        )),
        serde_json::to_vec(&a)?,
    )?;
    let mut pool = actual_winning::Winning::default();
    pool.policy_only=true;
    pool.add_root(&root)?;
    let mut rng = StableRng::new(42017);
    pool.draw(64, &mut rng, &model)?;
    // New correct proof arrives on disk after startup, without a priority admission.
    fs::write(
        root.join("proofs").join(format!(
            "{}.json",
            sha256(b["prefix"].as_str().ok_or("prefix")?.as_bytes())
        )),
        serde_json::to_vec(&b)?,
    )?;
    let mut missed = 0;
    for _ in 0..64 {
        missed += pool
            .draw(64, &mut rng, &model)?
            .iter()
            .inspect(|x|assert_eq!(x.value_weight,0.))
            .filter(|x| x.state == target)
            .count();
    }
    assert_eq!(missed, 0);
    let before = pool.progress();
    assert_eq!(before["catalogue"], 1);
    // Publish this one already-verified proof; never rescan the catalogue.
    let path = root.join("proofs").join(format!(
        "{}.json",
        sha256(b["prefix"].as_str().unwrap().as_bytes())
    ));
    assert!(pool.register_persisted_proof(path.clone()));
    for _ in 0..200 {
        assert!(!pool.register_persisted_proof(path.clone()));
    }
    let mut found = 0;
    for _ in 0..8 {
        found += pool
            .draw(64, &mut rng, &model)?
            .iter()
            .inspect(|x|assert_eq!(x.value_weight,0.))
            .filter(|x| x.state == target)
            .count();
    }
    assert!(found > 0);
    let after = pool.progress();
    assert_eq!(after["catalogue"], 2);
    fs::write(
        Path::new(&args[4]).join("recall-candidate-results.json"),
        serde_json::to_vec_pretty(
            &json!({"policy_only_value_weight_zero":true,"uses_production_source":"gen5/durable/winning.rs","new_directory_scans":0,"duplicate_admissions_checked":200,
        "new_proof_on_disk":true,"draws_before_refresh":4096,"new_proof_draws_before_refresh":missed,"before":before,
        "draws_after_refresh":512,"new_proof_draws_after_refresh":found,"after":after,"production_writes":0}),
        )?,
    )?;
    Ok(())
}
