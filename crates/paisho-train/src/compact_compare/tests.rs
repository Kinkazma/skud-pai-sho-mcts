use super::*;

fn args(extra: &[&str]) -> Vec<String> {
    ["--candidate", "candidate.json", "--output", "new-results"]
        .into_iter()
        .chain(extra.iter().copied())
        .map(str::to_owned)
        .collect()
}

#[test]
fn parser_defaults_and_bounds_are_explicit() {
    let defaults = Options::parse(&args(&[])).unwrap();
    assert_eq!(
        (
            defaults.pairs,
            defaults.simulations,
            defaults.workers,
            defaults.seconds
        ),
        (12, 32, 6, 120)
    );
    let high = Options::parse(&args(&["--simulations", "8192", "--reference-simulations", "512"])).unwrap();
    assert_eq!((high.simulations, high.reference_simulations), (8192, 512));
    assert_eq!(defaults.first_pair, 100000);
    assert_eq!(defaults.reference_simulations, defaults.simulations);
    assert_eq!(defaults.move_ms, None);
    assert_eq!(defaults.reference_model, None);
    assert!(!defaults.candidate_reuse && !defaults.reference_reuse);
    for extra in [
        vec!["--pairs", "0"],
        vec!["--seconds", "3601"],
        vec!["--simulations", "8193"],
        vec!["--reference-simulations", "0"],
        vec!["--reference-simulations", "8193"],
        vec!["--workers", "65"],
        vec!["--decision-limit", "8193"],
        vec!["--move-ms", "0"],
        vec!["--reference-model", ""],
        vec!["--candidate-reuse", "1"],
        vec!["--reference-reuse", "yes"],
        vec!["--pairs", "1", "--pairs", "2"],
        vec!["--unknown", "1"],
        vec!["--seconds"],
        vec!["--first-pair", "18446744073709551615"],
    ] {
        assert!(Options::parse(&args(&extra)).is_err(), "{extra:?}");
    }
    let custom = Options::parse(&args(&[
        "--pairs",
        "1",
        "--simulations",
        "2048",
        "--move-ms",
        "3",
        "--first-pair",
        "0",
        "--reference-model",
        "parent.json",
        "--candidate-reuse",
        "true",
        "--reference-reuse",
        "false",
    ]))
    .unwrap();
    assert_eq!(
        (
            custom.pairs,
            custom.simulations,
            custom.move_ms,
            custom.first_pair
        ),
        (1, 2048, Some(3), 0)
    );
    assert_eq!(custom.reference_model, Some(PathBuf::from("parent.json")));
    assert!(custom.candidate_reuse && !custom.reference_reuse);
}

#[test]
fn reference_budget_follows_reference_through_seat_reversal() {
    let options = Options::parse(&args(&[
        "--simulations",
        "64",
        "--reference-simulations",
        "256",
        "--decision-limit",
        "2",
    ]))
    .unwrap();
    let model = CompactValueModel::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for index in 0..2 {
        let result = pool.install(|| {
            play_game(
                index,
                &options,
                &model,
                &CpuMctsEvaluator,
                Instant::now() + Duration::from_secs(30),
            )
        });
        assert!(matches!(
            result.metadata.termination,
            Termination::DecisionLimit
        ));
        assert!(result.metadata.candidate.completed_searches > 0);
        assert!(result.metadata.reference.completed_searches > 0);
        assert_eq!(
            result.metadata.candidate.simulations,
            result.metadata.candidate.completed_searches * 64
        );
        assert_eq!(
            result.metadata.reference.simulations,
            result.metadata.reference.completed_searches * 256
        );
    }
}

#[test]
fn retained_search_follows_roles_and_only_counts_new_simulations() {
    let model = CompactValueModel::default();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for (candidate_reuse, reference_reuse) in [(true, false), (false, true), (true, true)] {
        let mut options = Options::parse(&args(&[
            "--simulations",
            "32",
            "--reference-simulations",
            "16",
            "--decision-limit",
            "16",
        ]))
        .unwrap();
        options.candidate_reuse = candidate_reuse;
        options.reference_reuse = reference_reuse;
        for index in 0..2 {
            let result = pool.install(|| {
                play_game(
                    index,
                    &options,
                    &model,
                    &model,
                    Instant::now() + Duration::from_secs(30),
                )
            });
            assert!(!matches!(
                result.metadata.termination,
                Termination::Error { .. }
            ));
            assert!(result.metadata.simulation_counts_complete);
            for (side, enabled, budget) in [
                (&result.metadata.candidate, candidate_reuse, 32),
                (&result.metadata.reference, reference_reuse, 16),
            ] {
                assert!(side.completed_searches > 0);
                assert_eq!(side.simulations, side.completed_searches * budget);
                assert_eq!(
                    side.retained_searches,
                    if enabled { side.completed_searches } else { 0 }
                );
                assert_eq!(side.reused_candidate_positions > 0, enabled);
                assert_eq!(side.maximum_retained_tree_bytes > 0, enabled);
            }
            let position = result.record.unwrap().replay().unwrap();
            assert_eq!(position.completed_turns(), result.metadata.completed_turns);
        }
    }
}

#[test]
fn retained_telemetry_sums_work_but_takes_peak_memory() {
    let mut total = SideTelemetry::default();
    for peak in [20, 10] {
        let side = SideTelemetry {
            retained_searches: 2,
            inherited_root_visits: 11,
            reused_candidate_positions: 3,
            reused_leaf_values: 4,
            maximum_retained_tree_bytes: peak,
            retained_memory_resets: 1,
            ..SideTelemetry::default()
        };
        total.add(&side);
    }
    assert_eq!(total.retained_searches, 4);
    assert_eq!(total.inherited_root_visits, 22);
    assert_eq!(total.reused_candidate_positions, 6);
    assert_eq!(total.reused_leaf_values, 8);
    assert_eq!(total.maximum_retained_tree_bytes, 20);
    assert_eq!(total.retained_memory_resets, 2);
}

#[test]
fn reversed_legs_preserve_seat_seeds_and_setup() {
    let options = Options::parse(&args(&[])).unwrap();
    let first = initial_metadata(0, &options);
    let second = initial_metadata(1, &options);
    let next = initial_metadata(2, &options);
    assert_eq!(first.pair_id, second.pair_id);
    assert_eq!(first.starting_flower, second.starting_flower);
    assert_eq!(first.host_seed, second.host_seed);
    assert_eq!(first.guest_seed, second.guest_seed);
    assert_ne!(first.host_seed, first.guest_seed);
    assert_ne!(first.host_seed, next.host_seed);
    assert!(first.candidate_host && !second.candidate_host);
}

#[test]
fn time_limits_and_unplayed_games_are_not_draws() {
    for stop in [
        Termination::WallLimit,
        Termination::DecisionLimit,
        Termination::NoLegalActions,
        Termination::Error {
            message: "unavailable".into(),
        },
        Termination::NotPlayed {
            reason: "deadline".into(),
        },
    ] {
        assert_eq!(stop.candidate_score(true), None);
        assert_eq!(stop.candidate_score(false), None);
        assert!(!serde_json::to_string(&stop).unwrap().contains("draw"));
    }
    let options = Options::parse(&args(&[])).unwrap();
    let result = play_game(
        0,
        &options,
        &CompactValueModel::default(),
        &CpuMctsEvaluator,
        Instant::now(),
    );
    assert!(matches!(
        result.metadata.termination,
        Termination::NotPlayed { .. }
    ));
    assert!(result.record.is_none());
    assert_eq!(result.metadata.decisions, 0);
}

#[test]
fn provided_reference_evaluator_follows_its_role_in_both_seats() {
    struct FailingReference;
    impl MctsEvaluator for FailingReference {
        fn evaluate(
            &self,
            _: &[Position],
            _: Player,
            _: paisho_ai::HeuristicWeights,
        ) -> std::result::Result<Vec<f32>, String> {
            Err("the supplied reference was called".into())
        }
    }
    let options = Options::parse(&args(&["--simulations", "1", "--decision-limit", "2"])).unwrap();
    for index in 0..2 {
        let result = play_game(
            index,
            &options,
            &CompactValueModel::default(),
            &FailingReference,
            Instant::now() + Duration::from_secs(30),
        );
        assert!(
            matches!(result.metadata.termination, Termination::Error { ref message }
            if message.contains("the supplied reference was called"))
        );
        assert_eq!(result.metadata.reference.attempted_decisions, 1);
        let starts_host = Position::from_standard_setup(StandardSetup::balanced(
            BASIC_FLOWERS[(result.metadata.pair_id % 6) as usize],
        ))
        .to_move()
            == Player::Host;
        assert_eq!(
            result.metadata.candidate.completed_decisions,
            usize::from(result.metadata.candidate_host == starts_host)
        );
    }
}

#[test]
fn scoring_uses_only_complete_pairs_and_keeps_conservative_exclusions() {
    let options = Options::parse(&args(&[])).unwrap();
    let mut games: Vec<_> = (0..6)
        .map(|index| initial_metadata(index, &options))
        .collect();
    games[0].termination = Termination::Rules {
        outcome: TerminalOutcome::HostWin,
    };
    games[1].termination = Termination::Rules {
        outcome: TerminalOutcome::Draw,
    };
    games[2].termination = Termination::Rules {
        outcome: TerminalOutcome::HostWin,
    };
    games[3].termination = Termination::WallLimit;
    games[4].termination = Termination::DecisionLimit;
    games[5].termination = Termination::Rules {
        outcome: TerminalOutcome::HostWin,
    };
    let result = summarize(&games);
    let comparison = &result["paired_comparison"];
    assert_eq!(comparison["rated_pairs"], 1);
    assert_eq!(comparison["one_and_half"], 1);
    assert_eq!(comparison["score"], 0.75);
    assert_eq!(comparison["wins"], 1);
    assert_eq!(comparison["draws"], 1);
    assert_eq!(comparison["losses"], 0);
    assert_eq!(comparison["excluded"], 2);
    assert_eq!(comparison["excluded_pessimistic_ties"], 1);
    assert_eq!(comparison["excluded_pessimistic_losses"], 1);
    assert_eq!(
        result["pessimistic_score_missing_games_as_candidate_losses"],
        json!(5.0 / 12.0)
    );
    let no_pairs = summarize(&games[2..]);
    assert!(no_pairs["paired_comparison"]["score"].is_null());
}

#[test]
fn game_errors_fail_the_result_even_when_the_archive_can_be_complete() {
    let options = Options::parse(&args(&[])).unwrap();
    let mut games: Vec<_> = (0..2)
        .map(|index| initial_metadata(index, &options))
        .collect();
    games[0].termination = Termination::Error {
        message: "learned evaluation failed".into(),
    };
    games[1].termination = Termination::Rules {
        outcome: TerminalOutcome::Draw,
    };
    let summary = summarize(&games);
    assert_eq!(summary["status"], "failed");
    assert_eq!(summary["game_errors"][0]["game_index"], 0);
    assert_eq!(
        summary["game_errors"][0]["message"],
        "learned evaluation failed"
    );
    assert_eq!(summary["game_terminations"]["error"], 1);
    assert_eq!(summary["paired_comparison"]["excluded"], 1);
    assert!(summary["paired_comparison"]["score"].is_null());
}

#[test]
fn bounded_run_saves_replayable_partial_games_and_rejects_existing_output() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "paisho-compact-compare-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&root).unwrap();
    let candidate = root.join("candidate.json");
    let reference = root.join("parent.json");
    let output = root.join("comparison");
    crate::compact_learning::save_model_new(&candidate, &ModelArtifact::legacy()).unwrap();
    let mut parent = ModelArtifact::legacy();
    parent.training_steps = 17;
    parent.weights[0] += 0.01;
    crate::compact_learning::save_model_new(&reference, &parent).unwrap();
    let args = vec![
        "--candidate".into(),
        candidate.to_string_lossy().into_owned(),
        "--reference-model".into(),
        reference.to_string_lossy().into_owned(),
        "--output".into(),
        output.to_string_lossy().into_owned(),
        "--pairs".into(),
        "1".into(),
        "--simulations".into(),
        "1".into(),
        "--workers".into(),
        "2".into(),
        "--decision-limit".into(),
        "1".into(),
        "--seconds".into(),
        "2".into(),
        "--candidate-reuse".into(),
        "true".into(),
        "--reference-reuse".into(),
        "true".into(),
    ];
    run(&args).unwrap();
    let plan: Value = serde_json::from_slice(&fs::read(output.join("plan.json")).unwrap()).unwrap();
    assert_eq!(plan["reference_kind"], "compact-model");
    assert_eq!(plan["candidate_reuse"], true);
    assert_eq!(plan["reference_reuse"], true);
    assert_eq!(plan["evaluators"]["candidate_reuse"], true);
    assert_eq!(plan["reference_model"]["training_steps"], 17);
    assert_eq!(
        fs::read(output.join("reference.json")).unwrap(),
        fs::read(&reference).unwrap()
    );
    assert_ne!(
        plan["reference_model"]["sha256"],
        plan["candidate"]["sha256"]
    );
    let summary: Value =
        serde_json::from_slice(&fs::read(output.join("summary.json")).unwrap()).unwrap();
    assert_eq!(summary["status"], "completed");
    assert_eq!(summary["complete_archive"], true);
    assert_eq!(summary["game_terminations"]["decision_limit"], 2);
    assert!(summary["paired_comparison"]["score"].is_null());
    assert_eq!(summary["paired_comparison"]["excluded"], 1);
    for index in 0..2 {
        let metadata: GameMetadata = serde_json::from_slice(
            &fs::read(output.join("games").join(format!("game-{index:08}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(metadata.evaluators.as_ref().unwrap()).unwrap(),
            plan["evaluators"]
        );
        assert_eq!(metadata.candidate_host, index == 0);
        let bytes = fs::read(output.join(metadata.record.unwrap())).unwrap();
        assert_eq!(metadata.record_sha256.unwrap(), sha256(&bytes));
        let record: GameRecord = std::str::from_utf8(&bytes).unwrap().parse().unwrap();
        assert_eq!(record.actions().len(), 1);
        assert_eq!(record.replay().unwrap().outcome(), GameOutcome::Ongoing);
        assert_eq!(
            metadata.candidate.simulations + metadata.reference.simulations,
            1
        );
    }
    assert!(run(&args).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn captured_flowers_and_occupied_gates_stop_before_search_without_result() {
    let record: GameRecord = include_str!(
        "../../../paisho-core/tests/fixtures/no-legal-actions-v2.psr"
    ).parse().unwrap();
    let mut position = record.initial_position();
    for action in record.actions() {
        assert!(actions_for_search(&position).is_ok());
        position.apply(*action).unwrap();
    }
    assert_eq!(position.to_move(), Player::Host);
    assert_eq!(position.completed_turns(), 53);
    assert_eq!(position.outcome(), GameOutcome::Ongoing);
    assert!(matches!(actions_for_search(&position), Err(Termination::NoLegalActions)));
    assert_eq!(Termination::NoLegalActions.candidate_score(true), None);
    assert_eq!(Termination::NoLegalActions.candidate_score(false), None);
}
