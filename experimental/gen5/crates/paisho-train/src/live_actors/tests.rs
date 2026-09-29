use super::*;
use paisho_ai::PolicyValueOutput;
use paisho_core::{GameOutcome, GameRecord, TurnPhase};
use paisho_model::{encode_action_v1, InferenceExampleV1};
use paisho_replay::PolicyTargetKindV1;

#[derive(Clone)]
struct Uniform;
impl PolicyValueEvaluator for Uniform {
    fn evaluate(&self, example: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError> {
        Ok(PolicyValueOutput::new(
            vec![1.0 / example.legal_actions().len() as f32; example.legal_actions().len()],
            [0.25, 0.5, 0.25],
        )
        .unwrap())
    }
}

#[derive(Clone)]
struct Broken;
impl PolicyValueEvaluator for Broken {
    fn evaluate(&self, _: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError> {
        Err(AgentError::new("mock evaluator unavailable"))
    }
}

fn config() -> LiveActorsConfiguration {
    LiveActorsConfiguration {
        actors: 2,
        target_games: 2,
        maximum_attempts: 5,
        first_game_id: 501,
        decision_soft_limit: 1,
        neutral_start: None,
        ..Default::default()
    }
}

fn pool(workers: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .unwrap()
}

#[test]
fn scheduling_matches_cli_pairs_flowers_and_relative_seats() {
    let config = config();
    let tasks = build_tasks(&config, 0, 12).unwrap();
    let producer = ReplayDigestV1::from_bytes([1; 32]);
    let opponent = ReplayDigestV1::from_bytes([2; 32]);
    for (slot, pair) in tasks.chunks_exact(2).enumerate() {
        assert_eq!(pair[0].game_id + 1, pair[1].game_id);
        assert_eq!(pair[0].setup, pair[1].setup);
        assert_eq!(pair[0].setup.starting_flower, BASIC_FLOWERS[slot]);
        assert_eq!(
            make_agent(&config, Uniform, producer, opponent, &pair[0], Player::Host).digest(),
            producer
        );
        assert_eq!(
            make_agent(
                &config,
                Uniform,
                producer,
                opponent,
                &pair[1],
                Player::Guest
            )
            .digest(),
            producer
        );
        assert_eq!(
            make_agent(&config, Uniform, producer, opponent, &pair[1], Player::Host).digest(),
            opponent
        );
    }
    assert_eq!(build_tasks(&config, 2, 4).unwrap(), tasks[2..6]);
    assert!(build_tasks(&config, 1, 2).is_err());
    assert!(build_tasks(&config, 0, 3).is_err());
}

#[test]
fn neutral_start_is_shared_by_both_legs() {
    let config = LiveActorsConfiguration {
        neutral_start: Some(NeutralStartConfigurationV1::new(20_260_905, 16, 4096, 16).unwrap()),
        ..config()
    };
    let tasks = pool(2).install(|| build_tasks(&config, 0, 2)).unwrap();
    assert!(!tasks[0].starting_actions.is_empty());
    assert_eq!(tasks[0].starting_actions, tasks[1].starting_actions);
    assert_eq!(tasks[0].neutral_start, tasks[1].neutral_start);
    assert!(
        tasks[0]
            .neutral_start
            .unwrap()
            .source_remaining_decisions()
            .abs_diff(16)
            <= 1
    );
}

#[test]
fn collection_finishes_paired_batches_and_preserves_failure_report() {
    let pool = pool(2);
    let producer = ReplayDigestV1::from_bytes([1; 32]);
    let report = collect_live_games(&config(), &pool, Uniform, producer).unwrap();
    assert_eq!(report.attempts, 4); // Odd remaining budget never starts half a pair.
    assert_eq!(report.interrupted, 4);
    assert_eq!(report.excluded_pairs, 2);
    assert!(!report.target_reached);
    assert!(report.retained.is_empty());
    assert!(report.abort_reason.is_none());
    let failed = collect_live_games(&config(), &pool, Broken, producer).unwrap();
    assert_eq!(failed.attempts, 2);
    assert_eq!(failed.failed, 1); // Other leg reaches the limit on the random side's first move.
    assert_eq!(failed.excluded_pairs, 1);
    assert!(failed
        .abort_reason
        .unwrap()
        .contains("mock evaluator unavailable"));
}

// A terminal fixture supplies a CPU-only evaluator for the last complete turn.
#[derive(Clone)]
struct Scripted(Vec<(InferenceExampleV1, usize)>);
impl PolicyValueEvaluator for Scripted {
    fn evaluate(&self, example: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError> {
        let index = self
            .0
            .iter()
            .find(|(state, _)| *state == example)
            .map(|(_, index)| *index)
            .ok_or_else(|| AgentError::new("unexpected fixture position"))?;
        let mut policy = vec![0.0; example.legal_actions().len()];
        policy[index] = 1.0;
        Ok(PolicyValueOutput::new(policy, [0.25, 0.5, 0.25]).unwrap())
    }
}

fn terminal_batch(workers: usize) -> Vec<ReplayMatchResult> {
    let record: GameRecord = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr"
    ))
    .parse()
    .unwrap();
    let mut position = record.initial_position();
    let mut states = Vec::new();
    let mut last_main = 0;
    for (index, &action) in record.actions().iter().enumerate() {
        if position.phase() == TurnPhase::Main {
            last_main = index;
        }
        let inference = InferenceExampleV1::from_position(&position).unwrap();
        let encoded = encode_action_v1(action, position.to_move()).unwrap();
        let chosen = inference
            .legal_actions()
            .iter()
            .position(|a| *a == encoded)
            .unwrap();
        states.push((inference, chosen));
        position.apply(action).unwrap();
    }
    assert_ne!(position.outcome(), GameOutcome::Ongoing);
    let config = LiveActorsConfiguration {
        opponent: LiveActorsOpponent::SelfPlay,
        policy: NetworkPolicy::Argmax,
        first_game_id: 0,
        ..config()
    };
    let tasks = (0..2)
        .map(|game_id| ReplayMatchTask {
            game_id,
            setup: record.setup(),
            starting_actions: record.actions()[..last_main].to_vec(),
            neutral_start: None,
        })
        .collect::<Vec<_>>();
    let evaluator = Scripted(states);
    let producer = ReplayDigestV1::from_bytes([9; 32]);
    pool(workers)
        .install(|| {
            play_replay_parallel(
                &tasks,
                ReplayMatchConfiguration {
                    decision_soft_limit: 16,
                },
                |task| {
                    make_agent(
                        &config,
                        evaluator.clone(),
                        producer,
                        producer,
                        task,
                        Player::Host,
                    )
                },
                |task| {
                    make_agent(
                        &config,
                        evaluator.clone(),
                        producer,
                        producer,
                        task,
                        Player::Guest,
                    )
                },
            )
        })
        .matches
        .into_iter()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn terminal_network_targets_are_reproducible_and_partial_pairs_are_discarded() {
    let matches = terminal_batch(1);
    let repeated = terminal_batch(2);
    for (first, second) in matches.iter().zip(&repeated) {
        assert_eq!(first.game, second.game);
        assert!(first.starting_decisions > 0);
        for decision in first.game.decisions() {
            assert!(decision.decision_index() >= first.starting_decisions);
            assert_eq!(decision.policy().kind(), PolicyTargetKindV1::Behavior);
            assert_eq!(
                decision.policy().producer(),
                ReplayDigestV1::from_bytes([9; 32])
            );
            assert_eq!(decision.behavior_value(), Some(0.0));
        }
    }
    let mut report = LiveActorsReport::default();
    absorb_matches(
        &mut report,
        vec![
            Ok(matches[0].clone()),
            Err(ReplayMatchError::DecisionLimit {
                game_id: 1,
                decisions: 1,
            }),
        ],
        LiveActorsOpponent::Random,
    );
    assert!(report.retained.is_empty());
    assert_eq!(
        (report.completed, report.interrupted, report.excluded_pairs),
        (1, 1, 1)
    );
    absorb_matches(
        &mut report,
        matches.into_iter().map(Ok).collect(),
        LiveActorsOpponent::Random,
    );
    assert_eq!(report.retained.len(), 2);
    assert_eq!(report.into_games().len(), 2);
}

#[test]
fn invalid_configuration_is_rejected_before_evaluation() {
    let pool = pool(1);
    for opponent in [
        LiveActorsOpponent::Mcts { simulations: 0 },
        LiveActorsOpponent::Mcts { simulations: 513 },
    ] {
        let bad = LiveActorsConfiguration {
            opponent,
            ..config()
        };
        assert!(
            collect_live_games(&bad, &pool, Broken, ReplayDigestV1::from_bytes([0; 32])).is_err()
        );
    }
    assert!(LiveActorsConfiguration {
        actors: 3,
        ..config()
    }
    .validate()
    .is_err());
    assert!(LiveActorsConfiguration {
        first_game_id: u64::MAX,
        ..config()
    }
    .validate()
    .is_err());
    assert!(LiveActorsConfiguration {
        opponent: LiveActorsOpponent::SelfPlay,
        actors: 1,
        target_games: 1,
        ..config()
    }
    .validate()
    .is_ok());
}

#[test]
fn mixed_collection_has_exact_separate_quotas_and_disjoint_ids() {
    let mut config = config();
    config.actors = 4;
    config.target_games = 4;
    config.maximum_attempts = 80;
    config.decision_soft_limit = 256;
    config.neutral_start = Some(NeutralStartConfigurationV1::new(17, 8, 16384, 16).unwrap());
    let producer = ReplayDigestV1::from_bytes([1; 32]);
    let report = collect_live_mixed_games(&config, &pool(2), Uniform, producer).unwrap();
    assert!(report.target_reached, "{report:?}");
    let own = report
        .retained
        .iter()
        .filter(|r| r.game.host_agent() == producer && r.game.guest_agent() == producer)
        .count();
    assert_eq!(own, 2);
    assert_eq!(report.retained.len(), 4);
    let ids = report
        .retained
        .iter()
        .map(|r| r.game.game_id())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(ids.len(), 4);
    let single = collect_live_mixed_games(&config, &pool(1), Uniform, producer).unwrap();
    assert_eq!(
        single.retained.iter().map(|r| &r.game).collect::<Vec<_>>(),
        report.retained.iter().map(|r| &r.game).collect::<Vec<_>>()
    );
    config.target_games = 6;
    assert!(collect_live_mixed_games(&config, &pool(2), Uniform, producer).is_err());
}

#[test]
fn weighted_quotas_preserve_pairs_and_worker_independent_trajectories() {
    let config = LiveActorsConfiguration {
        actors: 4,
        target_games: 8,
        maximum_attempts: 160,
        decision_soft_limit: 256,
        neutral_start: Some(NeutralStartConfigurationV1::new(17, 8, 16384, 16).unwrap()),
        ..config()
    };
    let producer = ReplayDigestV1::from_bytes([1; 32]);
    for quota in [0, 2, 6, 8] {
        let a = collect_live_weighted_games(&config, quota, &pool(2), Uniform, producer).unwrap();
        assert!(a.target_reached, "quota={quota}: {a:?}");
        assert!(a.attempts <= config.maximum_attempts);
        let external = a
            .retained
            .iter()
            .filter(|r| r.game.host_agent() != producer || r.game.guest_agent() != producer)
            .count();
        assert_eq!(external, quota);
        assert_eq!(
            a.retained
                .iter()
                .map(|r| r.game.game_id())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            8
        );
        let b = collect_live_weighted_games(&config, quota, &pool(1), Uniform, producer).unwrap();
        assert_eq!(a.into_games(), b.into_games());
    }
    for quota in [1, 9, 10] {
        assert!(collect_live_weighted_games(&config, quota, &pool(2), Uniform, producer).is_err());
    }
}

#[test]
fn weighted_failure_preserves_partial_reports_and_attempt_budget() {
    let config = LiveActorsConfiguration {
        actors: 4,
        target_games: 8,
        maximum_attempts: 12,
        ..config()
    };
    let report = collect_live_weighted_games(
        &config,
        2,
        &pool(2),
        Uniform,
        ReplayDigestV1::from_bytes([1; 32]),
    )
    .unwrap();
    assert_eq!(report.attempts, 12);
    assert!(!report.target_reached);
    assert!(report.abort_reason.is_none());
    let report = collect_live_weighted_games(
        &config,
        2,
        &pool(2),
        Broken,
        ReplayDigestV1::from_bytes([1; 32]),
    )
    .unwrap();
    assert!(report.abort_reason.is_some());
    assert!(report.attempts <= 12);
}
