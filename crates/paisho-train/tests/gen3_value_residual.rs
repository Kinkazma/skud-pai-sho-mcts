use paisho_ai::*;
use paisho_train::{compact_learning::ModelArtifact, gen32::*};
use std::fs;

#[test]
fn explicit_upgrade_roundtrip_preserves_lineage_weights_and_requires_schema() {
    let dir = std::env::temp_dir().join(format!("gen3-value-residual-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let compact = dir.join("compact.json");
    fs::write(
        &compact,
        serde_json::to_vec(&ModelArtifact::legacy()).unwrap(),
    )
    .unwrap();
    let parent = dir.join("parent.json");
    bootstrap(&compact, &parent, None).unwrap();
    let mut a = Artifact::load(&parent).unwrap();
    a.generation = "3.4".into();
    a.schema = "paisho-gen34-value128-memory-v1".into();
    a.value128_extra = Some(vec![0.01; 64]);
    a.memory_scope = Some("all-nodes".into());
    a.save(&parent).unwrap();
    let original = fs::read(&parent).unwrap();
    let output = dir.join("residual.json");
    upgrade_value_residual(&parent, &output, 71).unwrap();
    assert_eq!(original, fs::read(&parent).unwrap());
    let upgraded = Artifact::load(&output).unwrap();
    assert_eq!(upgraded.schema, GEN3_VALUE_RESIDUAL_SCHEMA);
    assert_eq!(upgraded.generation, a.generation);
    assert_eq!(upgraded.updates, a.updates);
    assert_eq!(upgraded.compact.weights, a.compact.weights);
    assert_eq!(upgraded.value128_extra, a.value128_extra);
    assert_eq!(upgraded.policy.parameters, a.policy.parameters);
    assert_eq!(upgraded.memory_scope, a.memory_scope);
    assert_eq!(upgraded.value_residual.as_ref().unwrap().len(), 2081);
    assert!(upgrade_value_residual(&parent, &output, 7).is_err());
    assert!(upgrade_value_residual(&output, &dir.join("again.json"), 7).is_err());
    let mut m = upgraded.model().unwrap();
    m.train(
        &MicroExample {
            sequence_source: 0,
            state: [0.2; 128],
            actions: vec![],
            policy: vec![],
            value: -0.8,
            policy_weight: 0.,
        },
        0.01,
    )
    .unwrap();
    upgraded
        .updated(&m, upgraded.updates + 1)
        .save(&output)
        .unwrap();
    let learned = Artifact::load(&output).unwrap();
    assert_eq!(learned.model().unwrap().value_residual, m.value_residual);
    assert_eq!(learned.model().unwrap().value_extra, m.value_extra);
    assert!(learned.model().unwrap().value_residual.unwrap().active());
    for malformed in 0..4 {
        let mut bad = learned.clone();
        match malformed {
            0 => bad.schema = "paisho-gen34-value128-memory-v1".into(),
            1 => bad.value_residual = None,
            2 => bad.value_residual = Some(vec![0.; 2080]),
            _ => bad.value128_extra = None,
        }
        assert!(bad.model().is_err());
    }
    // The real runtime must retain the new model on checkpoint and resume, even
    // in an inference-only smoke (no synthetic targets mixed into production).
    let o = Options {
        model: output,
        output: dir.join("run"),
        games: 2,
        threads: 1,
        actors: 1,
        budgets: vec![8],
        caps: vec![0.5],
        decisions: 8,
        seconds: 10.,
        learn: false,
        solver: true,
        ..Default::default()
    };
    run(o.clone()).unwrap();
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(o.output.join("checkpoint.json")).unwrap()).unwrap();
    let saved =
        Artifact::load(std::path::Path::new(checkpoint["model"].as_str().unwrap())).unwrap();
    assert_eq!(saved.value_residual, learned.value_residual);
    assert_eq!(saved.generation, "3.4");
    let resumed = Options {
        model: checkpoint["model"].as_str().unwrap().into(),
        replay_index: Some(checkpoint["replay_index"].as_str().unwrap().into()),
        output: dir.join("resume"),
        games: 1,
        ..o
    };
    run(resumed.clone()).unwrap();
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(resumed.output.join("checkpoint.json")).unwrap()).unwrap();
    let saved =
        Artifact::load(std::path::Path::new(checkpoint["model"].as_str().unwrap())).unwrap();
    assert_eq!(saved.value_residual, learned.value_residual);
    for path in [
        dir.join("run/games/game-0000000.psr"),
        dir.join("resume/games/game-0000000.psr"),
    ] {
        let record: paisho_core::GameRecord = fs::read_to_string(path).unwrap().parse().unwrap();
        record.replay().unwrap();
    }
    fs::remove_dir_all(dir).unwrap();
}
