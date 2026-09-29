use super::*;

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "paisho-compact-selfplay-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn args(extra: &[&str]) -> Vec<String> {
    [
        vec!["--model", "model.json", "--output", "new-output"],
        extra.to_vec(),
    ]
    .concat()
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn options(directory: &Directory) -> Options {
    Options {
        games: 2,
        seconds: 30.0,
        game_seconds: None,
        learn: true,
        replay_input: None,
        replay_capacity: 16,
        replay_ratio: 0,
        repetition_cycles: 0,
        reuse_search: false,
        workers: 2,
        simulations: 1,
        budgets: Vec::new(),
        decision_limit: 1,
        samples: 32,
        learning_rate: 0.01,
        lambda: 0.5,
        seed: 1,
        model: directory.0.join("parent.json"),
        output: directory.0.join("run"),
    }
}

pub(super) fn sample(perspective: Player, decision: usize) -> RootSample {
    let position = Position::from_standard_setup(StandardSetup::balanced(BASIC_FLOWERS[0]));
    RootSample {
        decision,
        perspective: Seat::from_player(perspective),
        features: CompactValueFeatures::extract(&position, perspective)
            .values()
            .to_vec(),
        q: 0.25,
        selected_visits: 1,
        search_simulations: 1,
    }
}

#[test]
fn parser_defaults_and_finite_bounds() {
    let options = Options::parse(&args(&[])).unwrap();
    assert_eq!(
        (
            options.games,
            options.workers,
            options.simulations,
            options.samples
        ),
        (128, 6, 8, 32)
    );
    assert_eq!(
        (options.seconds, options.learning_rate, options.lambda),
        (60.0, 0.01, 0.5)
    );
    assert_eq!(options.game_seconds, None);
    assert!(options.learn);
    let frozen = Options::parse(&args(&["--learn", "false", "--game-seconds", "0.25"])).unwrap();
    assert!(!frozen.learn);
    assert_eq!(frozen.game_seconds, Some(0.25));
    for invalid in [
        vec!["--games", "0"],
        vec!["--workers", "0"],
        vec!["--samples", "0"],
        vec!["--seconds", "NaN"],
        vec!["--seconds", "0"],
        vec!["--game-seconds", "NaN"],
        vec!["--game-seconds", "0"],
        vec!["--learn", "maybe"],
        vec!["--lambda", "1.1"],
        vec!["--learning-rate", "-1"],
        vec!["--unknown", "2"],
        vec!["--games", "2", "--games", "3"],
    ] {
        assert!(Options::parse(&args(&invalid)).is_err(), "{invalid:?}");
    }
    assert!(Options::parse(&["--model".to_string()]).is_err());
}

#[test]
fn collection_cutoffs_are_fixed_per_budget_and_allow_explicit_override() {
    for (budget, seconds) in [(32, 8.0), (64, 17.5), (128, 25.0), (256, 34.0), (512, 58.0)] {
        let budget = budget.to_string();
        let parsed = Options::parse(&args(&["--simulations", &budget])).unwrap();
        assert_eq!(parsed.game_seconds, Some(seconds));
        let explicit =
            Options::parse(&args(&["--simulations", &budget, "--game-seconds", "90"])).unwrap();
        assert_eq!(explicit.game_seconds, Some(90.0));
    }
    assert_eq!(
        Options::parse(&args(&["--simulations", "1024"]))
            .unwrap()
            .game_seconds,
        None
    );
}

#[test]
fn target_uses_real_player_not_move_parity() {
    let outcome = GameOutcome::Win(Player::Guest);
    assert_eq!(
        mixed_target(&sample(Player::Guest, 1), outcome, 0.5),
        Some(0.625)
    );
    assert_eq!(
        mixed_target(&sample(Player::Guest, 2), outcome, 0.5),
        Some(0.625)
    );
    assert_eq!(
        mixed_target(&sample(Player::Host, 2), outcome, 0.5),
        Some(-0.375)
    );
    assert_eq!(
        mixed_target(&sample(Player::Host, 1), GameOutcome::Draw, 0.5),
        Some(0.125)
    );
    assert_eq!(
        mixed_target(&sample(Player::Guest, 1), GameOutcome::Ongoing, 0.5),
        None
    );
}

#[test]
fn bonus_roots_keep_the_actual_players_terminal_sign() {
    let record: GameRecord =
        include_str!("../../../../benchmarks/results/human-mcts-bridge-2026-09-07/human.psr")
            .parse()
            .unwrap();
    let outcome = record.replay().unwrap().outcome();
    let mut position = record.initial_position();
    let mut bonuses = 0;
    for (index, action) in record.actions().iter().enumerate() {
        let before = position.to_move();
        position.apply(*action).unwrap();
        if position.phase() == paisho_core::TurnPhase::HarmonyBonus
            && position.outcome() == GameOutcome::Ongoing
        {
            assert_eq!(position.to_move(), before);
            let expected = if before == Player::Guest {
                0.625
            } else {
                -0.375
            };
            assert_eq!(
                mixed_target(&sample(before, index + 2), outcome, 0.5),
                Some(expected)
            );
            bonuses += 1;
        }
    }
    assert!(bonuses > 0);
}

#[test]
fn reservoir_is_bounded_distinct_and_spans_the_game() {
    let mut selected = Vec::new();
    let mut rng = StableRng::new(1);
    for decision in 1..=1000 {
        reservoir_sample(
            &mut selected,
            sample(Player::Guest, decision),
            decision,
            32,
            &mut rng,
        );
    }
    let decisions: std::collections::BTreeSet<_> = selected.iter().map(|s| s.decision).collect();
    assert_eq!((selected.len(), decisions.len()), (32, 32));
    assert!(*decisions.first().unwrap() < 200);
    assert!(*decisions.last().unwrap() > 800);
}

#[test]
fn tiny_unresolved_run_preserves_every_game_and_never_updates() {
    let directory = Directory::new();
    let options = options(&directory);
    let parent = ModelArtifact::legacy();
    save_model_new(&options.model, &parent).unwrap();
    run_options(options.clone()).unwrap();
    let final_model = load_model(&options.output.join("final-model.json")).unwrap();
    assert_eq!(final_model.weights, parent.weights);
    assert_eq!(final_model.training_steps, parent.training_steps);
    let summary: serde_json::Value =
        serde_json::from_slice(&fs::read(options.output.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["progress"]["consumed_games"], 2);
    assert_eq!(summary["progress"]["unresolved_games"], 2);
    assert_eq!(summary["progress"]["updates"], 0);
    assert_eq!(summary["progress"]["self_play_games"], 1);
    assert_eq!(summary["progress"]["legacy_games"], 1);
    for id in 0..2 {
        let psr = options
            .output
            .join("games")
            .join(format!("game-{id:08}.psr"));
        let record: GameRecord = fs::read_to_string(psr).unwrap().parse().unwrap();
        assert_eq!(record.actions().len(), 1);
        assert_eq!(record.replay().unwrap().outcome(), GameOutcome::Ongoing);
        let facts: serde_json::Value = serde_json::from_slice(
            &fs::read(
                options
                    .output
                    .join("games")
                    .join(format!("game-{id:08}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            facts["completed_turns"],
            record.replay().unwrap().completed_turns()
        );
        let receipt: serde_json::Value = serde_json::from_slice(
            &fs::read(
                options
                    .output
                    .join("games")
                    .join(format!("game-{id:08}.learning.json")),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["targets"], serde_json::json!([]));
        assert_eq!(receipt["learner_version_after"], 0);
    }
    assert!(run_options(options).is_err());
}

#[test]
fn terminal_update_is_durable_before_published_snapshot() {
    let directory = Directory::new();
    let options = options(&directory);
    fs::create_dir_all(options.output.join("models")).unwrap();
    fs::create_dir_all(options.output.join("games")).unwrap();
    let parent = ModelArtifact::legacy();
    let mut model = parent.model().unwrap();
    let shared: SharedModel = Arc::new(RwLock::new((0, Arc::new(model.clone()))));
    let old_snapshot = Arc::clone(&shared.read().unwrap().1);
    let record: GameRecord =
        include_str!("../../../../benchmarks/results/human-mcts-bridge-2026-09-07/human.psr")
            .parse()
            .unwrap();
    let game = PlayedGame {
        id: 0,
        worker: 0,
        version: 0,
        simulations: options.simulations,
        requested_game_seconds: options.game_seconds,
        schedule_id: 0,
        weights_sha256: weights_hash(&model),
        seeds: [1, 2, 3],
        self_play: true,
        learner_seat: None,
        outcome: record.replay().unwrap().outcome(),
        completed_turns: record.replay().unwrap().completed_turns(),
        record,
        repetition_loss: None,
        termination: "terminal",
        error: None,
        samples: vec![sample(Player::Guest, 1), sample(Player::Host, 2)],
        eligible_roots: 2,
        actual_simulations: vec![],
        reused_root_visits: 0,
        reused_candidate_positions: 0,
        reused_leaf_values: 0,
        elapsed_seconds: 0.0,
        timing: GameTiming::default(),
        effective_deadline_seconds_from_game_start: 0.0,
        game_deadline_overshoot_seconds: None,
        campaign_deadline_overshoot_seconds: 0.0,
    };
    let mut progress = Progress::default();
    learn_game(
        &game,
        &options,
        &parent,
        &mut model,
        &shared,
        &mut progress,
        &mut ReplayMemory::new(16, 1),
    )
    .unwrap();
    assert_eq!(progress.updates, 2);
    let snapshot = shared.read().unwrap();
    assert_eq!(snapshot.0, 1);
    let durable = load_model(&options.output.join("models/version-00000001.json")).unwrap();
    assert_eq!(durable.training_steps, 2);
    assert_eq!(durable.weights, snapshot.1.weights().to_vec());
    assert_eq!(old_snapshot.weights(), parent.model().unwrap().weights());
    assert_ne!(old_snapshot.weights(), snapshot.1.weights());
}

#[test]
fn expired_game_deadline_returns_a_replayable_unresolved_record() {
    let directory = Directory::new();
    let options = options(&directory);
    let game = play_game(
        &options,
        GameIdentity {
            id: 0,
            worker: 0,
            schedule_id: 0,
        },
        7,
        &CompactValueModel::default(),
        Instant::now(),
        &AtomicBool::new(false),
    );
    assert_eq!(game.termination, "campaign-deadline");
    assert_eq!(game.version, 7);
    assert!(game.samples.is_empty());
    assert!(game.record.actions().is_empty());
    assert_eq!(
        game.record.replay().unwrap().outcome(),
        GameOutcome::Ongoing
    );
}

#[test]
fn per_game_deadline_stops_only_the_game_and_preserves_nonterminal_status() {
    let directory = Directory::new();
    let mut options = options(&directory);
    options.game_seconds = Some(1e-9);
    let game = play_game(
        &options,
        GameIdentity {
            id: 0,
            worker: 0,
            schedule_id: 0,
        },
        0,
        &CompactValueModel::default(),
        Instant::now() + Duration::from_secs(30),
        &AtomicBool::new(false),
    );
    assert_eq!(game.termination, "game-deadline");
    assert!(game.record.actions().is_empty());
    assert_eq!(
        game.record.replay().unwrap().outcome(),
        GameOutcome::Ongoing
    );
    assert!(game.game_deadline_overshoot_seconds.unwrap() >= 0.0);
    assert_eq!(game.campaign_deadline_overshoot_seconds, 0.0);
    assert!(game.elapsed_seconds >= game.effective_deadline_seconds_from_game_start);
}

#[test]
fn frozen_terminal_collection_never_updates_even_when_samples_are_eligible() {
    let directory = Directory::new();
    let mut options = options(&directory);
    options.learn = false;
    fs::create_dir_all(options.output.join("models")).unwrap();
    fs::create_dir_all(options.output.join("games")).unwrap();
    let parent = ModelArtifact::legacy();
    let mut model = parent.model().unwrap();
    let shared: SharedModel = Arc::new(RwLock::new((0, Arc::new(model.clone()))));
    let record: GameRecord =
        include_str!("../../../../benchmarks/results/human-mcts-bridge-2026-09-07/human.psr")
            .parse()
            .unwrap();
    let game = PlayedGame {
        id: 0,
        worker: 0,
        version: 0,
        simulations: options.simulations,
        requested_game_seconds: options.game_seconds,
        schedule_id: 0,
        weights_sha256: weights_hash(&model),
        seeds: [1, 2, 3],
        self_play: true,
        learner_seat: None,
        outcome: record.replay().unwrap().outcome(),
        completed_turns: record.replay().unwrap().completed_turns(),
        record,
        repetition_loss: None,
        termination: "terminal",
        error: None,
        samples: vec![sample(Player::Guest, 1), sample(Player::Host, 2)],
        eligible_roots: 2,
        actual_simulations: vec![],
        reused_root_visits: 0,
        reused_candidate_positions: 0,
        reused_leaf_values: 0,
        elapsed_seconds: 0.0,
        timing: GameTiming::default(),
        effective_deadline_seconds_from_game_start: 30.0,
        game_deadline_overshoot_seconds: None,
        campaign_deadline_overshoot_seconds: 0.0,
    };
    let mut progress = Progress::default();
    learn_game(
        &game,
        &options,
        &parent,
        &mut model,
        &shared,
        &mut progress,
        &mut ReplayMemory::new(16, 1),
    )
    .unwrap();
    assert_eq!(model.weights(), parent.model().unwrap().weights());
    assert_eq!(progress.updates, 0);
    assert_eq!(progress.published_version, 0);
    assert_eq!(shared.read().unwrap().0, 0);
    assert_eq!(
        fs::read_dir(options.output.join("models")).unwrap().count(),
        0
    );
    let receipt: serde_json::Value = serde_json::from_slice(
        &fs::read(options.output.join("games/game-00000000.learning.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["learning_enabled"], false);
    assert_eq!(receipt["updates"], 0);
    assert_eq!(receipt["targets"], serde_json::json!([]));
}

#[test]
fn frozen_game_trajectories_do_not_depend_on_worker_identity() {
    let directory = Directory::new();
    let mut options = options(&directory);
    options.learn = false;
    options.decision_limit = 3;
    let model = CompactValueModel::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    for id in [0, 1, 3] {
        let first = pool.install(|| {
            play_game(
                &options,
                GameIdentity {
                    id,
                    worker: 0,
                    schedule_id: id,
                },
                0,
                &model,
                deadline,
                &AtomicBool::new(false),
            )
        });
        let second = pool.install(|| {
            play_game(
                &options,
                GameIdentity {
                    id,
                    worker: 7,
                    schedule_id: id,
                },
                0,
                &model,
                deadline,
                &AtomicBool::new(false),
            )
        });
        assert_eq!(first.record, second.record);
        assert_eq!(first.weights_sha256, second.weights_sha256);
        assert_eq!(first.seeds, second.seeds);
        assert_eq!(
            serde_json::to_value(first.samples).unwrap(),
            serde_json::to_value(second.samples).unwrap()
        );
    }
}

#[test]
fn heterogeneous_budgets_assign_workers_and_resolve_each_cutoff() {
    let options =
        Options::parse(&args(&["--budgets", "32,64,128,256,512", "--workers", "8"])).unwrap();
    assert_eq!(options.game_seconds, None);
    for (worker, (budget, cutoff)) in [
        (32, 8.0),
        (64, 17.5),
        (128, 25.0),
        (256, 34.0),
        (512, 58.0),
        (32, 8.0),
        (64, 17.5),
        (128, 25.0),
    ]
    .into_iter()
    .enumerate()
    {
        let actor = options.worker_options(worker);
        assert_eq!(actor.simulations, budget);
        assert_eq!(actor.game_seconds, Some(cutoff));
    }
    let explicit = Options::parse(&args(&["--budgets", "32,512", "--game-seconds", "90"])).unwrap();
    assert_eq!(explicit.worker_options(0).game_seconds, Some(90.0));
    assert_eq!(explicit.worker_options(1).game_seconds, Some(90.0));
    for invalid in [
        vec!["--budgets", ""],
        vec!["--budgets", "32,32"],
        vec!["--budgets", "32,0"],
        vec!["--budgets", "32,foo"],
        vec!["--budgets", "32,64", "--simulations", "8"],
        vec!["--budgets", "32,64", "--workers", "1"],
        vec!["--budgets", "32,64", "--games", "1"],
    ] {
        assert!(Options::parse(&args(&invalid)).is_err(), "{invalid:?}");
    }
}

#[test]
fn mixed_run_archives_actual_budget_and_counts_all_games_once() {
    let directory = Directory::new();
    let mut options = options(&directory);
    options.budgets = vec![1, 2];
    options.games = 2;
    save_model_new(&options.model, &ModelArtifact::legacy()).unwrap();
    run_options(options.clone()).unwrap();
    let read = |path: PathBuf| -> serde_json::Value {
        serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
    };
    let plan = read(options.output.join("plan.json"));
    assert_eq!(plan["worker_assignments"][0]["simulations"], 1);
    assert_eq!(plan["worker_assignments"][1]["simulations"], 2);
    assert_eq!(plan["search"]["simulations"], serde_json::Value::Null);
    let progress = read(options.output.join("progress.json"));
    assert_eq!(progress["consumed_games"], 2);
    for (id, budget) in [(0, 1), (1, 2)] {
        let game = read(options.output.join(format!("games/game-{id:08}.json")));
        let receipt = read(
            options
                .output
                .join(format!("games/game-{id:08}.learning.json")),
        );
        assert_eq!(game["simulations_requested_per_decision"], budget);
        assert_eq!(game["actual_simulations"], serde_json::json!([budget]));
        assert_eq!(game["schedule_id"], 0);
        assert_eq!(receipt["source_simulations_requested_per_decision"], budget);
        let stats = &progress["per_budget"][budget.to_string()];
        assert_eq!(stats["consumed_games"], 1);
        assert_eq!(stats["unresolved_games"], 1);
        assert_eq!(stats["updates"], 0);
        assert_eq!(stats["decisions"], 1);
    }
}

#[test]
fn local_schedule_cycles_opponents_and_standard_setups_independent_of_global_id() {
    let directory = Directory::new();
    let options = options(&directory);
    for (ordinal, expected_seat) in [
        (0, None),
        (1, Some(Seat::Host)),
        (2, None),
        (3, Some(Seat::Guest)),
    ] {
        // Already-expired deadline inspects assignment without doing a search.
        let game = play_game(
            &options,
            GameIdentity {
                id: 400,
                worker: 2,
                schedule_id: ordinal,
            },
            9,
            &CompactValueModel::default(),
            Instant::now(),
            &AtomicBool::new(false),
        );
        assert_eq!(game.self_play, ordinal % 2 == 0);
        assert_eq!(game.learner_seat, expected_seat);
        assert_eq!(
            game.record.setup(),
            StandardSetup::balanced(BASIC_FLOWERS[(ordinal / 2) % BASIC_FLOWERS.len()])
        );
        assert_eq!(game.schedule_id, ordinal);
        assert_eq!(game.version, 9);
    }
}

#[test]
fn repetition_loss_penalizes_only_cycle_closer_and_keeps_rule_outcome() {
    let directory = Directory::new();
    let mut options = options(&directory);
    options.replay_ratio = 4;
    let parent = ModelArtifact::legacy();
    let mut model = parent.model().unwrap();
    let mut game = play_game(
        &options,
        GameIdentity {
            id: 0,
            worker: 0,
            schedule_id: 0,
        },
        0,
        &model,
        Instant::now(),
        &AtomicBool::new(false),
    );
    game.samples = vec![
        sample(Player::Guest, 1),
        sample(Player::Host, 25),
        sample(Player::Guest, 26),
    ];
    game.repetition_loss = Some(RepetitionLoss {
        loser: Seat::Guest,
        first_decision: 10,
        last_decision: 26,
        period: 6,
        cycles: 4,
    });
    game.termination = "repetition-training-loss";
    assert!(training_example(&game, &game.samples[0], 0.5, "test-run").is_none());
    assert!(training_example(&game, &game.samples[1], 0.5, "test-run").is_none());
    assert_eq!(
        training_example(&game, &game.samples[2], 0.5, "test-run")
            .unwrap()
            .target,
        -1.0
    );
    assert_eq!(game.outcome, GameOutcome::Ongoing);
    assert_eq!(
        game.record.replay().unwrap().outcome(),
        GameOutcome::Ongoing
    );
    fs::create_dir_all(options.output.join("games")).unwrap();
    fs::create_dir_all(options.output.join("models")).unwrap();
    let shared = Arc::new(RwLock::new((0, Arc::new(model.clone()))));
    let mut progress = Progress::default();
    let mut replay = ReplayMemory::new(16, 1);
    learn_game(
        &game,
        &options,
        &parent,
        &mut model,
        &shared,
        &mut progress,
        &mut replay,
    )
    .unwrap();
    assert_eq!(progress.updates, 5);
    assert_eq!(replay.len(), 1);
    let receipt: serde_json::Value = serde_json::from_slice(
        &fs::read(options.output.join("games/game-00000000.learning.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["fresh_updates"], 1);
    assert_eq!(receipt["replay_updates"], 4);
    replay.persist(&options.output).unwrap();
    let mut restored = ReplayMemory::new(16, 99);
    restored
        .load(&options.output.join("replay-final.json"))
        .unwrap();
    assert_eq!(restored.len(), 1);
    let remembered = restored.draw().unwrap();
    assert_eq!(remembered.target, -1.0);
    assert_eq!(remembered.reason, "repetition-training-loss");
    assert_eq!(remembered.actor_weights_sha256, game.weights_sha256);

    for target in receipt["targets"].as_array().unwrap() {
        assert_eq!(target["target"], -1.0);
        assert_eq!(target["perspective"], "guest");
    }
}

#[test]
fn blocked_gates_never_supply_a_terminal_training_target() {
    let record: GameRecord = include_str!(
        "../../../paisho-core/tests/fixtures/no-legal-actions-v2.psr"
    ).parse().unwrap();
    let position = record.replay().unwrap();
    assert!(legal_actions(&position).is_empty());
    assert_eq!(position.outcome(), GameOutcome::Ongoing);
    for player in [Player::Host, Player::Guest] {
        assert_eq!(mixed_target(&sample(player, 63), position.outcome(), 0.5), None);
    }
}
