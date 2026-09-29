use super::*;
use crate::CompactValueModel;
use paisho_core::{BasicFlower, StandardSetup, TurnPhase};
use std::sync::atomic::{AtomicUsize, Ordering};

fn position() -> Position {
    Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3))
}
fn config() -> MctsConfig {
    MctsConfig {
        simulations: 32,
        ..MctsConfig::default()
    }
}

fn assert_same_search(mut actual: SearchReport, expected: SearchReport) {
    // Distinct Rayon worker occupancy is runtime telemetry, not search semantics.
    actual.action_ranking_workers = expected.action_ranking_workers;
    assert_eq!(actual, expected);
}

#[test]
fn fresh_cached_tree_has_exact_legacy_search_statistics() {
    for budget in [8, 32, 64, 128] {
        for rollout_depth in [0, 2] {
            let config = MctsConfig {
                simulations: budget,
                rollout_depth,
                ..config()
            };
            let position = position();
            let actions = legal_actions(&position);
            let mut original = MctsAgent::new(47, config).unwrap();
            let mut session = MctsSession::new(47, config, &CpuMctsEvaluator).unwrap();
            let expected = original
                .search_with_evaluator(&position, &actions, &CpuMctsEvaluator)
                .unwrap();
            let actual = session.search_until(&position, &actions, None).unwrap();
            assert_same_search(actual.clone(), expected);
            assert_eq!(
                session.reuse_statistics().reused_candidate_positions,
                actual.expanded_nodes
            );
            if rollout_depth == 0 {
                assert!(session.reuse_statistics().reused_leaf_values > 0);
            }
        }
    }
}

#[test]
fn learned_fresh_cache_parity_and_actual_action_retention() {
    let model = CompactValueModel::default();
    let mut position = position();
    let actions = legal_actions(&position);
    let mut original = MctsAgent::new(8, config()).unwrap();
    let mut session = MctsSession::new(8, config(), &model).unwrap();
    let expected = original
        .search_with_evaluator(&position, &actions, &model)
        .unwrap();
    let report = session.search_until(&position, &actions, None).unwrap();
    assert_same_search(report.clone(), expected);
    let chosen = report.actions[report.selected_index];
    assert!(session.advance(chosen.action));
    assert_eq!(session.retained_visits(), chosen.visits);
    position.apply(chosen.action).unwrap();
    let inherited = session.retained_visits();
    let report = session
        .search_until(&position, &legal_actions(&position), None)
        .unwrap();
    assert_eq!(report.simulations, 32);
    assert_eq!(session.reuse_statistics().inherited_root_visits, inherited);
    assert_eq!(session.retained_visits(), inherited + 32);
    assert!(report
        .actions
        .iter()
        .all(|stat| stat.mean_value().abs() <= 1.0));
}

#[test]
fn mismatch_discards_tree_and_new_snapshot_requires_new_session() {
    let first = position();
    let mut next = first.clone();
    next.apply(legal_actions(&first)[0]).unwrap();
    let mut session = MctsSession::new(1, config(), &CpuMctsEvaluator).unwrap();
    session
        .search_until(&first, &legal_actions(&first), None)
        .unwrap();
    session
        .search_until(&next, &legal_actions(&next), None)
        .unwrap();
    assert_eq!(session.reuse_statistics().inherited_root_visits, 0);
    session.clear();
    assert_eq!(session.retained_visits(), 0);
}

struct SplitEvaluator {
    leaves: AtomicUsize,
}
impl MctsEvaluator for SplitEvaluator {
    fn evaluate(
        &self,
        positions: &[Position],
        _: Player,
        _: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        Ok(vec![0.9; positions.len()])
    }
    fn evaluate_leaf(&self, _: &Position, _: Player, _: HeuristicWeights) -> Result<f32, String> {
        self.leaves.fetch_add(1, Ordering::Relaxed);
        Ok(-0.5)
    }
}

#[test]
fn ordering_values_are_not_assumed_to_be_leaf_values() {
    let evaluator = SplitEvaluator {
        leaves: AtomicUsize::new(0),
    };
    let mut session = MctsSession::new(9, config(), &evaluator).unwrap();
    let position = position();
    let report = session
        .search_until(&position, &legal_actions(&position), None)
        .unwrap();
    assert!(evaluator.leaves.load(Ordering::Relaxed) > 0);
    assert_eq!(session.reuse_statistics().reused_leaf_values, 0);
    assert!(report
        .actions
        .iter()
        .filter(|stat| stat.visits > 0)
        .all(|stat| stat.mean_value() == -0.5));
}

#[test]
fn soft_deadline_counts_new_visits_separately_from_retained_visits() {
    let mut session = MctsSession::new(1, config(), &CpuMctsEvaluator).unwrap();
    let position = position();
    let actions = legal_actions(&position);
    session.search_until(&position, &actions, None).unwrap();
    let report = session
        .search_until(&position, &actions, Some(std::time::Instant::now()))
        .unwrap();
    assert_eq!(report.simulations, 1);
    assert_eq!(session.retained_visits(), 33);
    assert_eq!(session.reuse_statistics().inherited_root_visits, 32);
}

#[test]
fn bonus_and_opponent_transitions_keep_fixed_tree_perspective() {
    let mut position = position();
    let mut session = MctsSession::new(
        18,
        MctsConfig {
            simulations: 8,
            ..config()
        },
        &CpuMctsEvaluator,
    )
    .unwrap();
    let initial_player = position.to_move();
    let mut saw_bonus = false;
    let mut saw_other_player = false;
    for _ in 0..160 {
        if position.outcome() != GameOutcome::Ongoing {
            break;
        }
        let actions = legal_actions(&position);
        let report = session.search_until(&position, &actions, None).unwrap();
        let root = session.root.as_ref().unwrap();
        let sign = if session.perspective.unwrap() == position.to_move() {
            1.0
        } else {
            -1.0
        };
        for stat in &report.actions {
            if let Some(child) = root
                .children
                .iter()
                .find(|child| child.action == stat.action)
            {
                assert_eq!(stat.value_sum, sign * child.node.value_sum);
            }
        }
        let chosen = actions[report.selected_index];
        assert!(session.advance(chosen));
        position.apply(chosen).unwrap();
        saw_bonus |= position.phase() == TurnPhase::HarmonyBonus;
        saw_other_player |= position.to_move() != initial_player;
        assert_eq!(session.perspective, Some(initial_player));
        assert_eq!(session.root.as_ref().unwrap().position, position);
        if saw_bonus && saw_other_player {
            break;
        }
    }
    assert!(saw_bonus && saw_other_player);
}

#[test]
fn retained_memory_limit_releases_tree_without_changing_the_completed_report() {
    let position = position();
    let actions = legal_actions(&position);
    let mut bounded = MctsSession::new(12, config(), &CpuMctsEvaluator).unwrap();
    bounded.set_memory_limit_bytes(1);
    let actual = bounded.search_until(&position, &actions, None).unwrap();
    let mut original = MctsAgent::new(12, config()).unwrap();
    assert_eq!(
        actual,
        original
            .search_with_evaluator(&position, &actions, &CpuMctsEvaluator)
            .unwrap()
    );
    assert_eq!(bounded.retained_bytes(), 0);
    assert!(bounded.reuse_statistics().tree_bytes_before_limit > 1);
    assert!(bounded.reuse_statistics().memory_limit_reset);
}

#[test]
fn prepared_candidate_memory_is_limited_per_node() {
    let mut position = position();
    let mut rng = StableRng::new(139);
    for _ in 0..30 {
        let actions = legal_actions(&position);
        if position.outcome() != GameOutcome::Ongoing {
            break;
        }
        if actions.len() > 64 {
            let mut session = MctsSession::new(1, config(), &CpuMctsEvaluator).unwrap();
            session.search_until(&position, &actions, None).unwrap();
            let root = session.root.as_ref().unwrap();
            assert!(root.prepared.len() <= 64);
            assert!(root.prepared.capacity() <= 64);
            assert!(session.retained_bytes() < 256 * 1024 * 1024);
            return;
        }
        position.apply(actions[rng.index(actions.len())]).unwrap();
    }
    panic!("test trajectory should reach more than 64 legal candidates");
}

struct FailingLeaf;
impl MctsEvaluator for FailingLeaf {
    fn evaluate(
        &self,
        positions: &[Position],
        _: Player,
        _: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        Ok(vec![0.0; positions.len()])
    }
    fn evaluate_leaf(&self, _: &Position, _: Player, _: HeuristicWeights) -> Result<f32, String> {
        Err("intentional leaf failure".into())
    }
}

#[test]
fn failed_simulation_does_not_leave_a_partially_mutated_reusable_tree() {
    let position = position();
    let mut session = MctsSession::new(12, config(), &FailingLeaf).unwrap();
    assert!(session
        .search_until(&position, &legal_actions(&position), None)
        .is_err());
    assert!(session.root.is_none());
    assert!(session.perspective.is_none());
}

#[test]
fn widening_beyond_cached_candidates_preserves_exact_search() {
    let mut position = position();
    let mut rng = StableRng::new(139);
    for _ in 0..30 {
        let actions = legal_actions(&position);
        if actions.len() > 64 {
            let config = MctsConfig {
                simulations: 128,
                maximum_tree_depth: 1,
                root_widening_factor: 100.0,
                ..config()
            };
            let mut session = MctsSession::new(91, config, &CpuMctsEvaluator).unwrap();
            let actual = session.search_until(&position, &actions, None).unwrap();
            let expected = MctsAgent::new(91, config)
                .unwrap()
                .search_with_evaluator(&position, &actions, &CpuMctsEvaluator)
                .unwrap();
            assert_same_search(actual.clone(), expected);
            assert_eq!(session.reuse_statistics().reused_candidate_positions, 64);
            assert!(actual.expanded_nodes > 64);
            return;
        }
        position.apply(actions[rng.index(actions.len())]).unwrap();
    }
    panic!("test trajectory should reach more than 64 legal candidates");
}
