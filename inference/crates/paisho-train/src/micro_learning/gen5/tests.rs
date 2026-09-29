use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
static NEXT: AtomicUsize = AtomicUsize::new(0);
fn directory() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "paisho-gen5-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&p).unwrap();
    p
}
fn base(root: &Path) -> Options {
    let p = root.join("parent.json");
    MicroArtifact::new(&MicroModel::seeded(5), 0, serde_json::json!({"test":true}))
        .save(&p)
        .unwrap();
    Options {
        model: p,
        output: root.join("run"),
        threads: 1,
        actors: 1,
        history_interval: 0.0,
        historical: false,
        learn: false,
        human_fraction: 0.0,
        budgets: vec![(1, 1.0)],
        // These existing fixtures exercise the original collector recipe.
        forced_playout_strength: 0.0,
        games: 4,
        decision_limit: 2,
        seconds: 10.0,
        ..Default::default()
    }
}
#[test]
fn unlimited_reference_ignores_game_timeout_but_honors_campaign_end() {
    for budget in [32, 64, 128] {
        let root = directory();
        let mut o = base(&root);
        for (_, seconds) in &mut o.historical_seconds {
            *seconds = 0.000000001;
        }
        assert_eq!(o.historical_unlimited_budgets, vec![32, 64, 128]);
        o.validate().unwrap();
        let artifact = MicroArtifact::load(&o.model).unwrap();
        let snapshot = Arc::new(Snapshot {
            artifact: None,
            version: 0,
            identity: artifact.identity(),
            model: Arc::new(artifact.model().unwrap()),
            path: o.model.clone(),
        });
        let old = crate::compact_learning::ModelArtifact::legacy()
            .model()
            .unwrap();
        let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
        let game = collector::play(
            0,
            snapshot.clone(),
            snapshot.clone(),
            Some((&old, budget)),
            &o,
            paisho_platform::training_time::now() + Duration::from_secs(10),
            &pool,
            "test",
        );
        assert_eq!(game.cap_seconds, None);
        assert_eq!(game.record.actions().len(), 2);
        assert_eq!(game.termination, "decision-limit");
        assert!(!game.campaign_censored);
        let ended = collector::play(
            1,
            snapshot.clone(),
            snapshot.clone(),
            Some((&old, budget)),
            &o,
            paisho_platform::training_time::now(),
            &pool,
            "test",
        );
        assert_eq!(ended.termination, "wall-limit");
        assert!(ended.campaign_censored);
        o.historical_unlimited_budgets.clear();
        let capped = collector::play(
            2,
            snapshot.clone(),
            snapshot,
            Some((&old, budget)),
            &o,
            paisho_platform::training_time::now() + Duration::from_secs(10),
            &pool,
            "test",
        );
        assert_eq!(capped.termination, "wall-limit");
        assert!(!capped.campaign_censored);
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn evaluation_limits_are_separate_from_training_and_persist_in_options() {
    let root = directory();
    let mut o = base(&root);
    assert!(o.search(1, true).unwrap().proof_search);
    assert!(o.search(1, false).unwrap().proof_search);
    assert_eq!(o.correction_replay_fraction, 0.05);
    o.proof_search = false;
    assert!(!o.search(1, true).unwrap().proof_search);
    o.proof_search = true;
    assert_eq!(
        Options::default()
            .search(1, true)
            .unwrap()
            .forced_playout_strength,
        2.0
    );
    assert_eq!(o.search(1, true).unwrap().forced_playout_strength, 0.0);
    o.forced_playout_strength = 2.0;
    assert_eq!(o.search(1, true).unwrap().forced_playout_strength, 2.0);
    assert_eq!(o.search(1, false).unwrap().forced_playout_strength, 0.0);
    o.mode = "gumbel".into();
    assert_eq!(o.search(1, true).unwrap().forced_playout_strength, 0.0);
    o.mode = "puct".into();
    assert_eq!(o.history_initial_pairs, 50);
    assert_eq!(o.history_decision_limit, 800);
    assert_eq!(o.decision_limit, 2);
    let restored: Options = serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
    assert_eq!(restored.history_initial_pairs * 2, 100);
    assert_eq!(restored.history_decision_limit, 800);
    o.history_initial_pairs = 0;
    assert!(o.validate().is_err());
    o.history_initial_pairs = 50;
    o.history_decision_limit = 0;
    assert!(o.validate().is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn assessment_uses_its_own_pair_count_and_decision_limit() {
    let root = directory();
    let mut o = base(&root);
    o.history_initial_pairs = 2;
    o.history_decision_limit = 1;
    o.decision_limit = 800;
    o.reference = root.join("reference.json");
    crate::compact_learning::save_model_new(
        &o.reference,
        &crate::compact_learning::ModelArtifact::legacy(),
    )
    .unwrap();
    let artifact = MicroArtifact::load(&o.model).unwrap();
    let initial = Arc::new(Snapshot {
        artifact: None,
        version: 0,
        identity: artifact.identity(),
        model: Arc::new(artifact.model().unwrap()),
        path: o.model.clone(),
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    assert!(history::assess(
        initial.clone(),
        &initial,
        &o,
        paisho_platform::training_time::now() + Duration::from_secs(30),
        &pool,
        0
    )
    .unwrap()
    .is_none());
    let dir = o.output.join("history/sweep-0000/initial");
    let plan: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("plan.json")).unwrap()).unwrap();
    assert_eq!(plan["options"]["pairs"], 2);
    assert_eq!(plan["options"]["decisions"], 1);
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["games"].as_array().unwrap().len(), 4);
    for id in 0..4 {
        let record: GameRecord = fs::read_to_string(dir.join(format!("game-{id:04}.psr")))
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(record.actions().len(), 1);
    }
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn controlled_resume_keeps_counters_replay_and_the_original_elo_anchor() {
    let root = directory();
    let mut o = base(&root);
    o.games = 102;
    let anchor = MicroArtifact::new(
        &MicroModel::seeded(99),
        0,
        serde_json::json!({"anchor":true}),
    );
    let anchor_path = root.join("anchor.json");
    anchor.save(&anchor_path).unwrap();
    o.evaluation_anchor = Some(anchor_path);
    let replay = root.join("replay.json");
    memory::Memory::new(&o).save(&replay).unwrap();
    o.replay_index = Some(replay);
    let progress = root.join("resume.json");
    save_json_new(&progress, &serde_json::json!({
        "completed":4,"version":17,"updates":0,"next_game_id":100,"next_history_index":2,
        "main_started":4,"historical_started":0,"fresh_terminal_used":0,"human_used":0,
        "elapsed_seconds":20.0,"learner_seconds":0.0,"archive_seconds":0.0,"publish_seconds":0.0,
        "replay_positions":0,"lanes":{}
    })).unwrap();
    o.resume_progress = Some(progress);
    run(o.clone()).unwrap();
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(o.output.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["completed"], 6);
    assert_eq!(report["version"], 17);
    assert!(report["elapsed_seconds"].as_f64().unwrap() >= 20.0);
    assert!(o.output.join("games/game-0000100.json").exists());
    assert_eq!(
        MicroArtifact::load(&o.output.join("evaluation-anchor.json"))
            .unwrap()
            .identity(),
        anchor.identity()
    );
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn gen5_limits_exclude_large_budgets_and_expired_deadlines() {
    let root = directory();
    let mut o = base(&root);
    o.seconds = 14.0 * 3600.0;
    o.validate().unwrap();
    o.seconds = 86401.0;
    assert!(o.validate().is_err());
    o.seconds = 10.0;
    o.budgets = vec![(2049, 1.0)];
    assert!(run(o.clone()).is_err());
    assert!(!o.output.exists());
    o.budgets = vec![(2048, 1.0)];
    o.end_unix_seconds = Some(1.0);
    assert!(run(o.clone()).is_err());
    assert!(!o.output.exists());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn bounded_collector_replays_all_records_and_never_trains_on_truncation() {
    let root = directory();
    let mut o = base(&root);
    o.learn = true;
    run(o.clone()).unwrap();
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(o.output.join("report.json")).unwrap()).unwrap();
    assert_eq!(report["completed"], 4);
    assert_eq!(report["updates"], 0);
    assert_eq!(report["fresh_terminal_used"], 0);
    for id in 0..4 {
        let record: GameRecord =
            fs::read_to_string(o.output.join(format!("games/game-{id:07}.psr")))
                .unwrap()
                .parse()
                .unwrap();
        assert_eq!(record.rules(), RULES);
        assert_eq!(record.actions().len(), 2);
        assert_eq!(record.replay().unwrap().outcome(), GameOutcome::Ongoing);
    }
    let index = o.output.join("replay-final.index.json");
    let mut replay = memory::Memory::new(&o);
    replay.load(&index).unwrap();
    assert_eq!(replay.len(), 0);
    assert!(run(o).is_err());
    fs::remove_dir_all(root).unwrap();
}
#[test]
fn historical_lane_has_separate_pool_and_cannot_exceed_start_quota() {
    let root = directory();
    let mut o = base(&root);
    o.threads = 2;
    o.actors = 1;
    o.historical = true;
    o.games = 60;
    o.reference = root.join("reference.json");
    crate::compact_learning::save_model_new(
        &o.reference,
        &crate::compact_learning::ModelArtifact::legacy(),
    )
    .unwrap();
    run(o.clone()).unwrap();
    let progress: serde_json::Value =
        serde_json::from_slice(&fs::read(o.output.join("progress.json")).unwrap()).unwrap();
    assert_eq!(progress["main_cpu_capacity"], 1);
    assert_eq!(progress["historical_cpu_capacity"], 1);
    assert!(
        progress["historical_started"].as_u64().unwrap()
            <= progress["main_started"].as_u64().unwrap() / 19
    );
    assert_eq!(progress["completed"], 60);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn calibration_histories_are_frozen_and_use_the_requested_budget() {
    let root = directory();
    let mut o = base(&root);
    o.calibration_reference_budget = Some(64);
    o.reference = root.join("reference.json");
    crate::compact_learning::save_model_new(
        &o.reference,
        &crate::compact_learning::ModelArtifact::legacy(),
    )
    .unwrap();
    let mut bad = o.clone();
    bad.learn = true;
    assert!(run(bad).is_err());
    run(o.clone()).unwrap();
    for id in 0..4 {
        let r: serde_json::Value = serde_json::from_slice(
            &fs::read(o.output.join(format!("games/game-{id:07}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(r["lane"], "Historical");
        assert_eq!(r["reference_budget"], 64);
        // Stable seed95000 fixtures; the persisted field must name the Gen5 seat.
        assert_eq!(r["candidate_seat"], ["Guest", "Host", "Guest", "Host"][id]);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn restored_history_is_allowed_but_existing_training_output_is_rejected() {
    let root = directory();
    let mut o = base(&root);
    o.games = 1;
    fs::create_dir_all(o.output.join("history")).unwrap();
    fs::write(o.output.join(".DS_Store"), b"Finder metadata").unwrap();
    fs::write(o.output.join("history/retained.json"), b"retained").unwrap();
    run(o.clone()).unwrap();
    assert_eq!(
        fs::read(o.output.join("history/retained.json")).unwrap(),
        b"retained"
    );
    let before = fs::read(o.output.join("progress.json")).unwrap();
    assert!(run(o.clone()).is_err());
    assert_eq!(fs::read(o.output.join("progress.json")).unwrap(), before);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn legacy_cases_use_requested_reference_and_seat_without_a_clock_cap() {
    let root = directory();
    let mut o = base(&root);
    o.legacy_replay = true;
    o.decision_limit = 2;
    let artifact = MicroArtifact::load(&o.model).unwrap();
    let snapshot = Arc::new(Snapshot {
        artifact: None,
        version: 0,
        identity: artifact.identity(),
        model: Arc::new(artifact.model().unwrap()),
        path: o.model.clone(),
    });
    let old = crate::compact_learning::ModelArtifact::legacy()
        .model()
        .unwrap();
    let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
    for budget in ladder::BUDGETS {
        for seat in [Player::Host, Player::Guest] {
            o.candidate_seat = Some(seat);
            let game = collector::play(
                10,
                snapshot.clone(),
                snapshot.clone(),
                Some((&old, budget)),
                &o,
                paisho_platform::training_time::now() + Duration::from_secs(10),
                &pool,
                "test",
            );
            assert_eq!(game.reference_budget, Some(budget));
            assert_eq!(game.cap_seconds, None);
            assert_eq!(game.candidate_seat, seat);
            assert_eq!(game.lane, Lane::Historical);
            assert!(game.error.is_none());
            game.record.replay().unwrap();
        }
    }
    o.stop_signal = Some(Arc::new(std::sync::atomic::AtomicBool::new(true)));
    let stopped = collector::play(
        11,
        snapshot.clone(),
        snapshot,
        Some((&old, 8)),
        &o,
        paisho_platform::training_time::now() + Duration::from_secs(10),
        &pool,
        "test",
    );
    assert_eq!(stopped.termination, "user-stop");
    assert_eq!(stopped.outcome, GameOutcome::Ongoing);
    assert!(stopped.error.is_none());
    assert!(stopped.record.actions().is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn consolidation_capture_is_explicit_and_legacy_options_keep_their_serialization() {
    let old:Options=serde_json::from_value(serde_json::json!({})).unwrap();
    assert!(!old.diagnostic_consolidation_capture);
    assert!(serde_json::to_value(&old).unwrap().get("diagnostic_consolidation_capture").is_none());
    let mut invalid=old.clone();invalid.diagnostic_consolidation_capture=true;
    assert!(invalid.validate().unwrap_err().to_string().contains("requires transactional V3"));
    let restored:Options=serde_json::from_value(serde_json::to_value(&invalid).unwrap()).unwrap();
    assert!(restored.diagnostic_consolidation_capture);
}
