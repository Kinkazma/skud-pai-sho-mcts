use paisho_ai::{
    evaluate_position, play_match, Agent, GreedyAgent, HeuristicWeights, MatchConfig, MatchResult,
    MatchTask, MctsAgent, MctsConfig,
};
use paisho_core::{
    legal_actions, Action, BasicFlower, GameOutcome, GameRecord, Player, Position, StandardSetup,
    TurnPhase,
};

const RING_FINISH_FIXTURE: &str = include_str!("fixtures/site_bot_v1_ring_finish.psr");

fn initial_position() -> Position {
    Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3))
}

#[test]
fn heuristic_evaluation_is_zero_sum() {
    let mut position = initial_position();
    let weights = HeuristicWeights::default();
    for step in 0..8 {
        let host = evaluate_position(&position, Player::Host, weights);
        let guest = evaluate_position(&position, Player::Guest, weights);
        assert!((host + guest).abs() < f32::EPSILON);
        let actions = legal_actions(&position);
        position.apply(actions[step % actions.len()]).unwrap();
    }
}

#[test]
fn finite_extreme_weights_still_produce_a_finite_bounded_value() {
    let mut position = initial_position();
    let plant = legal_actions(&position)
        .into_iter()
        .find(|action| matches!(action, paisho_core::Action::Plant { .. }))
        .unwrap();
    position.apply(plant).unwrap();
    let weights = HeuristicWeights {
        harmony: f32::MAX,
        midline_harmony: f32::MAX,
        blooming_flower: f32::MAX,
        total_flower: f32::MAX,
        basic_reserve_progress: f32::MAX,
    };

    let guest = evaluate_position(&position, Player::Guest, weights);
    let host = evaluate_position(&position, Player::Host, weights);
    assert!(guest.is_finite() && (-1.0..=1.0).contains(&guest));
    assert!(host.is_finite() && (-1.0..=1.0).contains(&host));
    assert_eq!(guest, -host);
}

#[test]
fn greedy_agent_returns_a_legal_reproducible_action() {
    let position = initial_position();
    let actions = legal_actions(&position);
    let mut first = GreedyAgent::new(42, HeuristicWeights::default());
    let mut second = GreedyAgent::new(42, HeuristicWeights::default());

    let first_index = first.select_action(&position, &actions).unwrap();
    let second_index = second.select_action(&position, &actions).unwrap();
    assert_eq!(first_index, second_index);
    let mut next = position;
    next.apply(actions[first_index]).unwrap();
}

#[test]
fn root_parallel_mcts_accounts_for_every_simulation() {
    let position = initial_position();
    let actions = legal_actions(&position);
    let config = MctsConfig {
        simulations: 24,
        independent_trees: 3,
        maximum_tree_depth: 12,
        action_rank_batch_size: 64,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 1,
        exploration: 1.25,
        heuristic_weights: HeuristicWeights::default(),
    };
    let mut agent = MctsAgent::new(91, config).unwrap();

    let report = agent.search(&position, &actions);

    assert_eq!(report.simulations, 24);
    assert_eq!(report.trees, config.independent_trees);
    assert_eq!(
        report.worker_capacity,
        rayon::current_num_threads().min(config.independent_trees)
    );
    assert!((1..=report.worker_capacity).contains(&report.workers));
    assert_eq!(
        report.action_ranking_worker_capacity,
        rayon::current_num_threads()
    );
    assert!((1..=report.action_ranking_worker_capacity).contains(&report.action_ranking_workers));
    assert_eq!(
        report
            .actions
            .iter()
            .map(|entry| entry.visits)
            .sum::<usize>(),
        config.simulations
    );
    let visited_actions = report.visited_root_actions();
    let largest_tree_budget = (config.simulations + report.trees - 1) / report.trees;
    let expected_width = (largest_tree_budget as f64).sqrt().ceil() as usize;
    assert_eq!(
        visited_actions,
        actions.len().min(expected_width),
        "root trees should share the progressive heuristic expansion order"
    );
    assert!(report.expanded_nodes > 0);
    assert!(report.evaluated_actions >= actions.len());
    assert!(report.generated_nodes >= report.trees);
    assert!(report.generated_nodes <= report.expanded_nodes + report.trees);
    assert!(report.generated_actions >= actions.len() * report.trees);
    assert!(report.mean_branching_factor() >= actions.len() as f64);
    assert!(report.maximum_depth > 0);
    assert!(report.maximum_depth <= config.maximum_tree_depth);
    assert!(report.rollout_steps <= report.expanded_nodes * config.rollout_depth);
    assert!(report.selected_index < actions.len());
    let mut next = position;
    next.apply(actions[report.selected_index]).unwrap();
}

#[test]
fn root_parallel_budget_distribution_covers_remainders_and_excess_trees() {
    let position = initial_position();
    let actions = legal_actions(&position);
    for (simulations, independent_trees, expected_trees) in [(10, 3, 3), (2, 8, 2), (1, 8, 1)] {
        let config = MctsConfig {
            simulations,
            independent_trees,
            maximum_tree_depth: 8,
            action_rank_batch_size: 64,
            root_widening_factor: 1.0,
            progressive_widening_factor: 1.0,
            rollout_depth: 0,
            exploration: 1.0,
            heuristic_weights: HeuristicWeights::default(),
        };
        let report = MctsAgent::new(0x4255_4447_4554, config)
            .unwrap()
            .search(&position, &actions);
        assert_eq!(report.trees, expected_trees);
        assert_eq!(
            report
                .actions
                .iter()
                .map(|statistics| statistics.visits)
                .sum::<usize>(),
            simulations
        );
    }
}

#[test]
fn mcts_finds_an_immediate_ring_win() {
    let record: GameRecord = RING_FINISH_FIXTURE.parse().unwrap();
    let (winning_action, prefix) = record.actions().split_last().unwrap();
    let mut position = record.initial_position();
    for action in prefix {
        position.apply(*action).unwrap();
    }
    let actions = legal_actions(&position);
    assert!(actions.contains(winning_action));
    let config = MctsConfig {
        simulations: 1,
        independent_trees: 1,
        maximum_tree_depth: 8,
        action_rank_batch_size: usize::MAX,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: 1.0,
        heuristic_weights: HeuristicWeights::default(),
    };
    let report = MctsAgent::new(0x5249_4e47, config)
        .unwrap()
        .search(&position, &actions);
    assert_eq!(actions[report.selected_index], *winning_action);
}

#[test]
fn mcts_blocks_an_immediate_ring_threat_when_a_safe_turn_exists() {
    let record: GameRecord = RING_FINISH_FIXTURE.parse().unwrap();
    let mut position = record.initial_position();
    for action in &record.actions()[..9] {
        position.apply(*action).unwrap();
    }
    assert_eq!(position.to_move(), Player::Host);
    assert!(turn_has_safe_completion(&position));
    let mut recorded_blunder = position.clone();
    recorded_blunder.apply(record.actions()[9]).unwrap();
    assert!(current_player_has_immediate_ring_win(&recorded_blunder));

    let config = MctsConfig {
        simulations: 512,
        independent_trees: 1,
        maximum_tree_depth: 16,
        action_rank_batch_size: usize::MAX,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    };
    let mut agent = MctsAgent::new(0x0042_4c4f_434b, config).unwrap();
    apply_agent_decision(&mut position, &mut agent);
    if position.phase() == TurnPhase::HarmonyBonus {
        apply_agent_decision(&mut position, &mut agent);
    }
    assert!(
        completed_turn_is_safe_for(&position, Player::Host),
        "MCTS(512) left an immediate Ring win after a safe turn was available"
    );
}

#[test]
fn mcts_keeps_maximizing_during_the_same_players_bonus() {
    let record: GameRecord = RING_FINISH_FIXTURE.parse().unwrap();
    let mut position = record.initial_position();
    for action in &record.actions()[..4] {
        position.apply(*action).unwrap();
    }
    assert_eq!(position.phase(), TurnPhase::Main);
    assert_eq!(position.to_move(), Player::Guest);

    let arrangement = record.actions()[4];
    assert!(legal_actions(&position).contains(&arrangement));
    let mut after_arrangement = position.clone();
    after_arrangement.apply(arrangement).unwrap();
    assert_eq!(after_arrangement.phase(), TurnPhase::HarmonyBonus);
    assert_eq!(after_arrangement.to_move(), Player::Guest);

    let weights = HeuristicWeights::default();
    let before_bonus_value = evaluate_position(&after_arrangement, Player::Guest, weights);
    let mut bonus_values: Vec<_> = legal_actions(&after_arrangement)
        .into_iter()
        .map(|action| {
            let mut candidate = after_arrangement.clone();
            candidate.apply(action).unwrap();
            evaluate_position(&candidate, Player::Guest, weights)
        })
        .collect();
    bonus_values.sort_by(|left, right| right.total_cmp(left));
    assert!(bonus_values.len() >= 2);
    let best_bonus_value = bonus_values[0];
    let first_lower_bonus = bonus_values
        .iter()
        .position(|value| *value < best_bonus_value)
        .expect("fixture needs at least two distinct bonus values");

    // The first root simulation reaches the bonus node. Continue until
    // progressive widening has exposed the first inferior bonus and the next
    // simulation must select between it and the better children.
    let mut simulations = 1;
    let mut bonus_node_visits = 1;
    let mut expanded_bonuses = 0;
    let mut expected_value_sum = f64::from(before_bonus_value);
    loop {
        simulations += 1;
        let widening_limit = ((bonus_node_visits + 1) as f64).sqrt().ceil() as usize;
        if expanded_bonuses < widening_limit {
            expected_value_sum += f64::from(bonus_values[expanded_bonuses]);
            expanded_bonuses += 1;
        } else {
            expected_value_sum += f64::from(best_bonus_value);
            if expanded_bonuses > first_lower_bonus {
                break;
            }
        }
        bonus_node_visits += 1;
    }
    assert!(simulations < 1_024, "bonus fixture became too expensive");

    let config = MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 2,
        action_rank_batch_size: usize::MAX,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: 0.0,
        heuristic_weights: weights,
    };
    let report = MctsAgent::new(0x0042_4f4e_5553, config)
        .unwrap()
        .search(&position, &[arrangement]);
    assert_eq!(report.actions[0].visits, simulations);
    assert_eq!(
        report.actions[0].value_sum, expected_value_sum,
        "selection after widening must revisit the best bonus for the same player"
    );
}

fn turn_has_safe_completion(position: &Position) -> bool {
    legal_actions(position)
        .into_iter()
        .any(|main| action_has_safe_completion(position, main))
}

fn action_has_safe_completion(position: &Position, main: Action) -> bool {
    let defender = position.to_move();
    let mut after_main = position.clone();
    after_main.apply(main).unwrap();
    if after_main.outcome() != GameOutcome::Ongoing {
        return completed_turn_is_safe_for(&after_main, defender);
    }
    if after_main.phase() == TurnPhase::HarmonyBonus {
        legal_actions(&after_main).into_iter().any(|bonus| {
            let mut after_bonus = after_main.clone();
            after_bonus.apply(bonus).unwrap();
            completed_turn_is_safe_for(&after_bonus, defender)
        })
    } else {
        completed_turn_is_safe_for(&after_main, defender)
    }
}

fn completed_turn_is_safe_for(position: &Position, defender: Player) -> bool {
    match position.outcome() {
        GameOutcome::Win(winner) => winner == defender,
        GameOutcome::Draw => true,
        GameOutcome::Ongoing => {
            position.phase() == TurnPhase::Main && !current_player_has_immediate_ring_win(position)
        }
    }
}

fn current_player_has_immediate_ring_win(position: &Position) -> bool {
    if position.outcome() != GameOutcome::Ongoing || position.phase() != TurnPhase::Main {
        return false;
    }
    let player = position.to_move();
    legal_actions(position).into_iter().any(|action| {
        let mut candidate = position.clone();
        candidate.apply(action).unwrap();
        candidate.outcome() == GameOutcome::Win(player)
    })
}

fn apply_agent_decision(position: &mut Position, agent: &mut MctsAgent) -> Action {
    let actions = legal_actions(position);
    let selected = agent.select_action(position, &actions).unwrap();
    let action = actions[selected];
    position.apply(action).unwrap();
    action
}

#[test]
fn mcts_is_reproducible_across_parallel_runs() {
    let position = initial_position();
    let actions = legal_actions(&position);
    let config = MctsConfig {
        simulations: 16,
        independent_trees: 2,
        maximum_tree_depth: 8,
        action_rank_batch_size: 64,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: 1.0,
        heuristic_weights: HeuristicWeights::default(),
    };
    let mut first = MctsAgent::new(1234, config).unwrap();
    let mut second = MctsAgent::new(1234, config).unwrap();

    let mut first_report = first.search(&position, &actions);
    let mut second_report = second.search(&position, &actions);
    first_report.workers = 0;
    first_report.worker_capacity = 0;
    first_report.action_ranking_workers = 0;
    first_report.action_ranking_worker_capacity = 0;
    second_report.workers = 0;
    second_report.worker_capacity = 0;
    second_report.action_ranking_workers = 0;
    second_report.action_ranking_worker_capacity = 0;
    assert_eq!(first_report, second_report);
}

#[test]
fn singleton_internal_batches_account_for_only_demanded_rankings() {
    let position = initial_position();
    let actions = legal_actions(&position);
    let config = MctsConfig {
        simulations: 32,
        independent_trees: 1,
        maximum_tree_depth: 12,
        action_rank_batch_size: 1,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: 1.0,
        heuristic_weights: HeuristicWeights::default(),
    };
    let report = MctsAgent::new(81, config)
        .unwrap()
        .search(&position, &actions);

    let internal_expansions = report.expanded_nodes - report.visited_root_actions();
    assert_eq!(
        report.evaluated_actions,
        actions.len() + internal_expansions,
        "each non-root expansion should rank exactly one new candidate"
    );
}

#[test]
fn logical_tree_result_is_independent_of_available_cpu_workers() {
    let position = initial_position();
    let actions = legal_actions(&position);
    let config = MctsConfig {
        simulations: 16,
        independent_trees: 2,
        maximum_tree_depth: 8,
        action_rank_batch_size: 64,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: 1.0,
        heuristic_weights: HeuristicWeights::default(),
    };
    let search = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                MctsAgent::new(1234, config)
                    .unwrap()
                    .search(&position, &actions)
            })
    };

    let mut one_worker = search(1);
    let mut two_workers = search(2);
    assert_eq!(one_worker.workers, 1);
    assert_eq!(one_worker.worker_capacity, 1);
    assert_eq!(one_worker.action_ranking_workers, 1);
    assert_eq!(one_worker.action_ranking_worker_capacity, 1);
    assert!((1..=2).contains(&two_workers.workers));
    assert_eq!(two_workers.worker_capacity, 2);
    assert!((1..=2).contains(&two_workers.action_ranking_workers));
    assert_eq!(two_workers.action_ranking_worker_capacity, 2);
    one_worker.workers = 0;
    two_workers.workers = 0;
    one_worker.worker_capacity = 0;
    two_workers.worker_capacity = 0;
    one_worker.action_ranking_workers = 0;
    two_workers.action_ranking_workers = 0;
    one_worker.action_ranking_worker_capacity = 0;
    two_workers.action_ranking_worker_capacity = 0;
    assert_eq!(one_worker, two_workers);
}

#[test]
fn bounded_mcts_trajectory_is_independent_of_available_cpu_workers() {
    let config = MctsConfig {
        simulations: 16,
        independent_trees: 2,
        maximum_tree_depth: 8,
        action_rank_batch_size: 64,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: 1.0,
        heuristic_weights: HeuristicWeights::default(),
    };
    let play = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                play_match(
                    MatchTask {
                        id: 93,
                        setup: StandardSetup::balanced(BasicFlower::White3),
                    },
                    MatchConfig {
                        decision_soft_limit: 16,
                    },
                    &mut MctsAgent::new(11, config).unwrap(),
                    &mut MctsAgent::new(12, config).unwrap(),
                )
                .unwrap()
            })
    };

    let mut one_worker = play(1);
    let mut two_workers = play(2);
    assert_eq!(one_worker.host_telemetry.maximum_action_ranking_workers, 1);
    assert_eq!(one_worker.guest_telemetry.maximum_action_ranking_workers, 1);
    assert!((1..=2).contains(&two_workers.host_telemetry.maximum_action_ranking_workers));
    assert!((1..=2).contains(&two_workers.guest_telemetry.maximum_action_ranking_workers));
    assert_eq!(
        two_workers
            .host_telemetry
            .maximum_action_ranking_worker_capacity,
        2
    );
    assert_eq!(
        two_workers
            .guest_telemetry
            .maximum_action_ranking_worker_capacity,
        2
    );
    clear_hardware_telemetry(&mut one_worker);
    clear_hardware_telemetry(&mut two_workers);
    assert_eq!(one_worker, two_workers);
}

fn clear_hardware_telemetry(result: &mut MatchResult) {
    for telemetry in [&mut result.host_telemetry, &mut result.guest_telemetry] {
        telemetry.maximum_search_workers = 0;
        telemetry.maximum_search_worker_capacity = 0;
        telemetry.maximum_action_ranking_workers = 0;
        telemetry.maximum_action_ranking_worker_capacity = 0;
    }
}

#[test]
fn two_mcts_agents_can_self_play_without_an_illegal_action() {
    let config = MctsConfig {
        simulations: 8,
        independent_trees: 1,
        maximum_tree_depth: 8,
        action_rank_batch_size: 64,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 1,
        exploration: 1.0,
        heuristic_weights: HeuristicWeights::default(),
    };
    let result = play_match(
        MatchTask {
            id: 77,
            setup: StandardSetup::balanced(BasicFlower::White3),
        },
        MatchConfig {
            decision_soft_limit: 16,
        },
        &mut MctsAgent::new(1, config).unwrap(),
        &mut MctsAgent::new(2, config).unwrap(),
    )
    .unwrap();

    assert_eq!(result.record.replay().unwrap(), result.final_position);
    assert!(result.host_telemetry.simulations > 0);
    assert!(result.guest_telemetry.simulations > 0);
    assert!(result.host_telemetry.expanded_nodes > 0);
    assert!(result.guest_telemetry.expanded_nodes > 0);
    assert!(result.host_telemetry.generated_nodes > 0);
    assert!(result.guest_telemetry.generated_nodes > 0);
    assert!(result.host_telemetry.maximum_search_depth > 0);
    assert!(result.guest_telemetry.maximum_search_depth > 0);
    assert!(result.host_telemetry.evaluated_actions > 0);
    assert!(result.guest_telemetry.evaluated_actions > 0);
    assert_eq!(result.host_telemetry.maximum_search_trees, 1);
    assert_eq!(result.guest_telemetry.maximum_search_trees, 1);
    assert_eq!(result.host_telemetry.maximum_search_workers, 1);
    assert_eq!(result.guest_telemetry.maximum_search_workers, 1);
    assert_eq!(result.host_telemetry.maximum_search_worker_capacity, 1);
    assert_eq!(result.guest_telemetry.maximum_search_worker_capacity, 1);
    assert!(result.host_telemetry.maximum_action_ranking_workers >= 1);
    assert!(result.guest_telemetry.maximum_action_ranking_workers >= 1);
    assert_eq!(
        result.host_telemetry.maximum_action_ranking_worker_capacity,
        rayon::current_num_threads()
    );
    assert_eq!(
        result
            .guest_telemetry
            .maximum_action_ranking_worker_capacity,
        rayon::current_num_threads()
    );
}
