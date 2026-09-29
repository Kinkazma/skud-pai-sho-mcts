use paisho_ai::*;
use paisho_train::{compact_learning::ModelArtifact, gen32::*};
use std::fs;
#[test]
fn explicit_migration_roundtrip_training_and_legacy_rejection() {
    let d = std::env::temp_dir().join(format!("gen34-migration-{}", std::process::id()));
    fs::create_dir_all(&d).unwrap();
    let parent = d.join("compact.json");
    fs::write(
        &parent,
        serde_json::to_vec(&ModelArtifact::legacy()).unwrap(),
    )
    .unwrap();
    let old = d.join("old.json");
    bootstrap(&parent, &old, None).unwrap();
    let mut a = Artifact::load(&old).unwrap();
    a.schema = "paisho-gen34-value128-memory-v1".into();
    a.memory_scope = Some("bonus-nodes".into());
    a.value128_extra = Some(vec![0.; 64]);
    a.generation = "3.4".into();
    let p = d.join("new.json");
    a.save(&p).unwrap();
    let mut m = Artifact::load(&p).unwrap().model().unwrap();
    assert_eq!(m.memory_scope, Gen3MemoryScope::BonusNodes);
    assert_eq!(
        m.value.weights(),
        Artifact::load(&old)
            .unwrap()
            .model()
            .unwrap()
            .value
            .weights()
    );
    let e = MicroExample {
        sequence_source: 0,
        state: [0.1; 128],
        actions: vec![],
        policy: vec![],
        value: -0.8,
        policy_weight: 0.,
    };
    m.train(&e, 0.01).unwrap();
    a.updated(&m, a.updates + 1).save(&p).unwrap();
    let restored = Artifact::load(&p).unwrap().model().unwrap();
    assert_eq!(restored.value_extra, m.value_extra);
    assert_ne!(restored.value_extra, Some([0.; 64]));
    assert_eq!(restored.value.weights(), m.value.weights());
    a.schema = "paisho-gen3-policy-memory-v1".into();
    assert!(a.model().is_err());
    a.schema = "paisho-gen34-value128-memory-v1".into();
    a.value128_extra = Some(vec![0.; 63]);
    assert!(a.model().is_err());
    a.value128_extra = Some(vec![0.; 64]);
    a.memory_scope = Some("typo".into());
    assert!(a.model().is_err());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn historical_pool_has_exact_ten_percent_balanced_generations_and_no_time_cap() {
    use sha2::{Digest, Sha256};
    let d = std::env::temp_dir().join(format!("gen34-pool-{}", std::process::id()));
    fs::create_dir_all(&d).unwrap();
    let parent = d.join("reference.json");
    let bytes = serde_json::to_vec(&ModelArtifact::legacy()).unwrap();
    fs::write(&parent, &bytes).unwrap();
    let model = d.join("model.json");
    bootstrap(&parent, &model, None).unwrap();
    let pool: Vec<_> = ["Gen3.1", "Gen3.2", "Gen3.3"]
        .into_iter()
        .map(|g| HistoricalExpert {
            generation: g.into(),
            budget: 8,
            path: parent.clone(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            solver: false,
        })
        .collect();
    let o = Options {
        model,
        output: d.join("run"),
        historical_pool: pool.clone(),
        historical_every: 10,
        games: 60,
        threads: 1,
        actors: 1,
        budgets: vec![8],
        caps: vec![1e-9],
        decisions: 2,
        seconds: 30.,
        learn: false,
        ..Default::default()
    };
    run(o.clone()).unwrap();
    let rows: Vec<serde_json::Value> = fs::read_to_string(o.output.join("receipts.jsonl"))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(rows.len(), 60);
    assert_eq!(rows.iter().filter(|r| r["historical"] == true).count(), 6);
    for g in ["Gen3.1", "Gen3.2", "Gen3.3"] {
        let r: Vec<_> = rows.iter().filter(|r| r["opponent"] == g).collect();
        assert_eq!(r.len(), 2);
        assert_ne!(r[0]["candidate_seat"], r[1]["candidate_seat"]);
        for row in r {
            assert!(row["game_cap_seconds"].is_null());
            assert_eq!(row["reference_sha256"], pool[0].sha256);
            assert_eq!(row["decisions"], 2);
        }
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(o.output.join("reference-pool.json")).unwrap()).unwrap();
    assert_eq!(meta["unique_models_loaded"], 1);
    let mut bad = o.clone();
    bad.output = d.join("bad");
    bad.historical_pool[0].sha256 = "changed".into();
    assert!(run(bad).is_err());
    assert!(!d.join("bad").exists());
    fs::remove_dir_all(d).unwrap();
}
