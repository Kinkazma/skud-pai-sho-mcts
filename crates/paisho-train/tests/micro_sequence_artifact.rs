use paisho_ai::*;
use paisho_train::micro_learning::*;
use sha2::{Digest, Sha256};
use std::sync::Arc;
#[test]
fn bank_dependency_is_hash_bound_shared_and_required() {
    let mut e = SequenceEntry {
        key: [0.; 64],
        patterns: [[0; 32]; 4],
        source: 1,
        game: 0,
        decision: 0,
        end_decision: 20,
        outcome: 1,
        phase: 0,
    };
    e.key[0] = 1.;
    let bank = SequenceBank::build(vec![e], 1, 1, 1);
    let mut bytes = vec![];
    bank.write_to(&mut bytes).unwrap();
    let dir = std::env::temp_dir().join(format!("sequence-artifact-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("memory.bin");
    std::fs::write(&path, &bytes).unwrap();
    let spec = SequenceMemorySpec {
        path: path.to_string_lossy().into(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let b = load_sequence_memory(&spec).unwrap();
    let again = load_sequence_memory(&spec).unwrap();
    assert!(Arc::ptr_eq(&b, &again));
    let m = MicroModel::seeded(1).with_sequence_memory(b.clone());
    let a = MicroArtifact::new(&m, 7, serde_json::json!({}));
    let rebuilt = a.model().unwrap();
    assert!(Arc::ptr_eq(rebuilt.sequence_memory().unwrap(), &b));
    a.save(&dir.join("model.json")).unwrap();
    let loaded = MicroArtifact::load(&dir.join("model.json")).unwrap();
    assert_eq!(a.identity(), loaded.identity());
    let mut missing = a.clone();
    missing.sequence_memory = None;
    assert!(missing.model().is_err());
    let mut bad = spec.clone();
    bad.sha256 = "0".repeat(64);
    assert!(load_sequence_memory(&bad).is_err());
    let mut corrupt = bytes.clone();
    corrupt[32] ^= 1;
    assert!(sequence_memory_from_bytes(spec, &corrupt).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
