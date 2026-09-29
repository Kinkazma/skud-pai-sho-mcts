use paisho_train::{compact_learning::ModelArtifact, gen32::*};
use std::{fs, path::PathBuf};
fn directory(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("gen32-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&p);
    fs::create_dir_all(&p).unwrap();
    p
}
#[test]
fn bootstrap_never_reinterprets_gen5_memory_as_gen3() {
    let d = directory("memory");
    let parent = d.join("parent.json");
    fs::write(
        &parent,
        serde_json::to_vec(&ModelArtifact::legacy()).unwrap(),
    )
    .unwrap();
    fs::write(
        d.join("summary.json"),
        r#"{"rules":"skud-pai-sho-gen5-v1"}"#,
    )
    .unwrap();
    assert!(bootstrap(&parent, &d.join("new.json"), Some(&d)).is_err());
    assert!(!d.join("new.json").exists());
    bootstrap(&parent, &d.join("new.json"), None).unwrap();
    let a = Artifact::load(&d.join("new.json")).unwrap();
    assert_eq!(a.compact.weights, ModelArtifact::legacy().weights);
    fs::remove_dir_all(d).unwrap();
}
#[test]
fn bounded_run_archives_truthful_psrs_and_resumes_the_exact_checkpoint() {
    let d = directory("run");
    let parent = d.join("parent.json");
    fs::write(
        &parent,
        serde_json::to_vec(&ModelArtifact::legacy()).unwrap(),
    )
    .unwrap();
    let initial = d.join("initial.json");
    bootstrap(&parent, &initial, None).unwrap();
    let o = Options {
        model: initial,
        output: d.join("run"),
        seconds: 15.,
        games: 4,
        threads: 2,
        actors: 2,
        budgets: vec![8],
        caps: vec![3.],
        decisions: 60,
        ..Default::default()
    };
    run(o.clone()).unwrap();
    let checkpoint: serde_json::Value =
        serde_json::from_slice(&fs::read(o.output.join("checkpoint.json")).unwrap()).unwrap();
    let count = checkpoint["completed"].as_u64().unwrap();
    assert_eq!(count, 4);
    for id in 0..count {
        let psr: paisho_core::GameRecord =
            fs::read_to_string(o.output.join(format!("games/game-{id:07}.psr")))
                .unwrap()
                .parse()
                .unwrap();
        assert_eq!(psr.rules(), RULES);
        psr.replay().unwrap();
    }
    let second = Options {
        model: checkpoint["model"].as_str().unwrap().into(),
        replay_index: Some(checkpoint["replay_index"].as_str().unwrap().into()),
        output: d.join("resume"),
        games: 1,
        ..o
    };
    run(second.clone()).unwrap();
    assert!(second.output.join("summary.json").exists());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn frozen_reference_is_hash_checked_and_replaces_only_every_tenth_game() {
    use sha2::{Digest, Sha256};
    let d = directory("frozen-reference");
    let parent = d.join("parent.json");
    let bytes = serde_json::to_vec(&ModelArtifact::legacy()).unwrap();
    fs::write(&parent, &bytes).unwrap();
    let initial = d.join("initial.json");
    bootstrap(&parent, &initial, None).unwrap();
    let o = Options {
        model: initial,
        output: d.join("run"),
        seconds: 20.,
        games: 20,
        threads: 1,
        actors: 1,
        budgets: vec![8],
        caps: vec![0.000000001],
        decisions: 2,
        learn: false,
        historical_reference: Some(parent.clone()),
        historical_reference_sha256: Some(format!("{:x}", Sha256::digest(&bytes))),
        historical_every: 10,
        ..Default::default()
    };
    run(o.clone()).unwrap();
    let rows: Vec<serde_json::Value> = fs::read_to_string(o.output.join("receipts.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(rows.len(), 20);
    let history: Vec<_> = rows.iter().filter(|r| r["historical"] == true).collect();
    assert_eq!(history.len(), 2);
    for r in &rows {
        assert_ne!(r["opponent"], "historical-heuristic");
        assert!(r["collector_sha256"].as_str().unwrap().len() == 64);
        let psr: paisho_core::GameRecord = fs::read_to_string(
            o.output
                .join(format!("games/game-{:07}.psr", r["id"].as_u64().unwrap())),
        )
        .unwrap()
        .parse()
        .unwrap();
        psr.replay().unwrap();
    }
    for r in &history {
        assert_eq!(r["opponent"], "Gen3.1");
        assert_eq!(
            r["reference_sha256"],
            o.historical_reference_sha256.as_deref().unwrap()
        );
        assert_eq!(r["decisions"], 2); // Tiny selfplay cutoff must not cut reference games.
        assert!(r["game_cap_seconds"].is_null());
        assert_eq!(r["eligible"], 0); // No fictitious result for decision-limited games.
    }
    assert_ne!(history[0]["candidate_seat"], history[1]["candidate_seat"]);
    fs::write(&parent, b"changed").unwrap();
    assert!(run(Options {
        output: d.join("bad"),
        ..o
    })
    .is_err());
    assert!(!d.join("bad/progress.json").exists());
    fs::remove_dir_all(d).unwrap();
}
