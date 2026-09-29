use std::sync::Mutex;
use std::time::{Duration, Instant};

use paisho_ai::{
    CpuMctsEvaluator, HeuristicWeights, MctsAgent, MctsConfig, MctsEvaluator, SearchReport,
};
use paisho_core::{
    legal_actions, Action, BasicFlower, GameOutcome, GameRecord, Player, Position, StandardSetup,
};

struct LegacyOrdering;

impl MctsEvaluator for LegacyOrdering {
    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        CpuMctsEvaluator.evaluate(positions, perspective, weights)
    }
}

struct ObservedLeaf {
    result: Result<f32, String>,
    positions: Mutex<Vec<(Position, Player)>>,
    legacy_ordering: bool,
}

impl ObservedLeaf {
    fn new(result: Result<f32, String>) -> Self {
        Self {
            result,
            positions: Mutex::new(Vec::new()),
            legacy_ordering: false,
        }
    }
}

impl MctsEvaluator for ObservedLeaf {
    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        if self.legacy_ordering {
            return CpuMctsEvaluator.evaluate(positions, perspective, weights);
        }
        // Ordering is deliberately different from leaf values so a regression
        // that substitutes ordering scores for learned leaves is observable.
        Ok(vec![0.0; positions.len()])
    }

    fn evaluate_leaf(
        &self,
        position: &Position,
        perspective: Player,
        _: HeuristicWeights,
    ) -> Result<f32, String> {
        self.positions
            .lock()
            .unwrap()
            .push((position.clone(), perspective));
        self.result.clone()
    }
}

fn initial_position() -> Position {
    Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3))
}

fn config(simulations: usize) -> MctsConfig {
    MctsConfig {
        simulations,
        maximum_tree_depth: 4,
        action_rank_batch_size: 4,
        ..MctsConfig::default()
    }
}

fn assert_report_accounting(report: &SearchReport, actions: &[Action]) {
    assert_eq!(
        report
            .actions
            .iter()
            .map(|action| action.visits)
            .sum::<usize>(),
        report.simulations
    );
    assert!(report.selected_index < actions.len());
    assert!(report.actions[report.selected_index].visits > 0);
    assert!(report.actions.iter().all(|action| {
        action.value_sum.is_finite() && action.value_sum.abs() <= action.visits as f64
    }));
}

#[test]
fn default_leaf_and_disabled_deadline_preserve_legacy_search() {
    // A one-worker pool also makes occupancy telemetry deterministic.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    pool.install(|| {
        let position = initial_position();
        let actions = legal_actions(&position);
        for (trees, depth, rollout) in [(1, 1, 0), (3, 4, 0), (1, 4, 2)] {
            let config = MctsConfig {
                independent_trees: trees,
                maximum_tree_depth: depth,
                rollout_depth: rollout,
                ..config(12)
            };
            let reference = MctsAgent::new(347, config)
                .unwrap()
                .search(&position, &actions);
            let default_leaf = MctsAgent::new(347, config)
                .unwrap()
                .search_with_evaluator(&position, &actions, &LegacyOrdering)
                .unwrap();
            let no_deadline = MctsAgent::new(347, config)
                .unwrap()
                .search_with_evaluator_until(&position, &actions, &LegacyOrdering, None)
                .unwrap();
            assert_eq!(default_leaf, reference);
            assert_eq!(no_deadline, reference);
        }
    });
}

#[test]
fn learned_leaf_is_used_when_a_new_node_is_expanded() {
    let position = initial_position();
    let action = legal_actions(&position)[0];
    let mut child = position.clone();
    child.apply(action).unwrap();
    let evaluator = ObservedLeaf::new(Ok(0.375));
    let report = MctsAgent::new(47, config(1))
        .unwrap()
        .search_with_evaluator(&position, &[action], &evaluator)
        .unwrap();
    assert_eq!(report.actions[0].value_sum, 0.375);
    assert_eq!(report.expanded_nodes, 1);
    assert_eq!(
        *evaluator.positions.lock().unwrap(),
        vec![(child, position.to_move())]
    );
}

#[test]
fn learned_leaf_is_used_again_at_the_tree_depth_limit() {
    let position = initial_position();
    let action = legal_actions(&position)[0];
    let evaluator = ObservedLeaf::new(Ok(-0.625));
    let report = MctsAgent::new(
        47,
        MctsConfig {
            maximum_tree_depth: 1,
            ..config(4)
        },
    )
    .unwrap()
    .search_with_evaluator(&position, &[action], &evaluator)
    .unwrap();
    assert_eq!(report.expanded_nodes, 1);
    assert_eq!(report.maximum_depth, 1);
    assert_eq!(report.actions[0].value_sum, -2.5);
    let calls = evaluator.positions.lock().unwrap();
    assert_eq!(calls.len(), 4);
    assert!(calls.windows(2).all(|pair| pair[0] == pair[1]));
}

#[test]
fn learned_leaf_evaluates_the_rollout_endpoint() {
    let position = initial_position();
    let action = legal_actions(&position)[0];
    let mut child = position.clone();
    child.apply(action).unwrap();
    let evaluator = ObservedLeaf::new(Ok(0.25));
    let report = MctsAgent::new(
        87,
        MctsConfig {
            rollout_depth: 2,
            ..config(1)
        },
    )
    .unwrap()
    .search_with_evaluator(&position, &[action], &evaluator)
    .unwrap();
    assert_eq!(report.rollout_steps, 2);
    assert_eq!(report.actions[0].value_sum, 0.25);
    let calls = evaluator.positions.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_ne!(calls[0].0, child);
    assert!(calls[0].0.completed_turns() > child.completed_turns());
    assert_eq!(calls[0].1, position.to_move());
}

#[test]
fn leaf_errors_and_unbounded_or_nonfinite_values_are_rejected() {
    let position = initial_position();
    let action = legal_actions(&position)[0];
    for result in [
        Err("learned snapshot unavailable".into()),
        Ok(f32::NAN),
        Ok(f32::INFINITY),
        Ok(f32::NEG_INFINITY),
        Ok(1.001),
        Ok(-1.001),
    ] {
        for rollout_depth in [0, 2] {
            let evaluator = ObservedLeaf::new(result.clone());
            let error = MctsAgent::new(
                47,
                MctsConfig {
                    rollout_depth,
                    ..config(1)
                },
            )
            .unwrap()
            .search_with_evaluator(&position, &[action], &evaluator)
            .unwrap_err();
            assert!(error.contains(
                result
                    .as_ref()
                    .err()
                    .map_or("invalid MCTS leaf evaluator response", String::as_str)
            ));
        }
    }
}

#[test]
fn rule_terminal_outcomes_bypass_custom_leaf_values() {
    let records = [
        include_str!("fixtures/site_bot_v1_ring_finish.psr"),
        include_str!("fixtures/site_bot_v1_reserve_finish.psr"),
        include_str!("../../../benchmarks/results/mcts-128-vs-32-2156132-block-2024/records/pair-00000000000000002024-high-host.psr"),
    ];
    let mut saw_draw = false;
    let mut saw_win = false;
    for text in records {
        let record: GameRecord = text.parse().unwrap();
        let (last, prefix) = record.actions().split_last().unwrap();
        let mut position = record.initial_position();
        for action in prefix {
            position.apply(*action).unwrap();
        }
        let mut terminal = position.clone();
        terminal.apply(*last).unwrap();
        let expected = match terminal.outcome() {
            GameOutcome::Win(winner) => {
                saw_win = true;
                if winner == position.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
            GameOutcome::Draw => {
                saw_draw = true;
                0.0
            }
            GameOutcome::Ongoing => panic!("fixture must reach a rule terminal outcome"),
        };
        for rollout_depth in [0, 2] {
            let evaluator = ObservedLeaf::new(Err("terminal must bypass the model".into()));
            let report = MctsAgent::new(
                47,
                MctsConfig {
                    rollout_depth,
                    maximum_tree_depth: 1,
                    ..config(3)
                },
            )
            .unwrap()
            .search_with_evaluator(&position, &[*last], &evaluator)
            .unwrap();
            assert_eq!(report.actions[0].mean_value(), expected);
            assert!(evaluator.positions.lock().unwrap().is_empty());
        }
    }
    assert!(saw_win && saw_draw);
}

#[test]
fn an_opponent_terminal_win_is_backed_up_as_a_root_loss() {
    let record: GameRecord = include_str!("fixtures/site_bot_v1_ring_finish.psr")
        .parse()
        .unwrap();
    let mut position = record.initial_position();
    for action in &record.actions()[..9] {
        position.apply(*action).unwrap();
    }
    assert_eq!(position.to_move(), Player::Host);
    let blunder = record.actions()[9];
    let mut evaluator = ObservedLeaf::new(Ok(0.25));
    evaluator.legacy_ordering = true;
    let report = MctsAgent::new(
        47,
        MctsConfig {
            maximum_tree_depth: 2,
            action_rank_batch_size: usize::MAX,
            ..config(2)
        },
    )
    .unwrap()
    .search_with_evaluator(&position, &[blunder], &evaluator)
    .unwrap();
    // The first simulation evaluates the new nonterminal child as +0.25.
    // The next sees Guest's rule-terminal win, which is -1 for Host.
    assert_eq!(report.actions[0].value_sum, -0.75);
    assert_eq!(report.expanded_nodes, 2);
    assert_eq!(evaluator.positions.lock().unwrap().len(), 1);
}

#[test]
fn an_expired_soft_deadline_keeps_one_simulation_per_tree_and_a_legal_move() {
    let position = initial_position();
    let actions = legal_actions(&position);
    for (simulations, trees, expected) in [(20, 3, 3), (2, 8, 2), (1, 1, 1)] {
        let report = MctsAgent::new(
            347,
            MctsConfig {
                independent_trees: trees,
                ..config(simulations)
            },
        )
        .unwrap()
        .search_with_evaluator_until(&position, &actions, &CpuMctsEvaluator, Some(Instant::now()))
        .unwrap();
        assert_eq!(report.simulations, expected);
        assert_eq!(report.trees, expected);
        assert_report_accounting(&report, &actions);
        position
            .clone()
            .apply(actions[report.selected_index])
            .unwrap();
    }
}

#[test]
fn a_future_deadline_completes_the_normal_fixed_budget() {
    let position = initial_position();
    let actions = legal_actions(&position);
    let report = MctsAgent::new(
        347,
        MctsConfig {
            independent_trees: 3,
            ..config(10)
        },
    )
    .unwrap()
    .search_with_evaluator_until(
        &position,
        &actions,
        &CpuMctsEvaluator,
        Some(Instant::now() + Duration::from_secs(300)),
    )
    .unwrap();
    assert_eq!(report.simulations, 10);
    assert_eq!(report.trees, 3);
    assert_report_accounting(&report, &actions);
}
