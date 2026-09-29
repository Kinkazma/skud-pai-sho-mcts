use std::collections::VecDeque;

use paisho_ai::{
    Agent, AgentError, AgentTelemetry, HeuristicWeights, MctsAgent, MctsConfig, NetworkPolicy,
    PolicyValueEvaluator, PolicyValueOutput, PureNetworkAgent,
};
use paisho_core::{
    legal_actions, Action, BasicFlower, GameRecord, Player, Position, StandardSetup,
};
use paisho_model::InferenceExampleV1;
use paisho_replay::{PolicyTargetKindV1, ReplayDigestV1};
use paisho_train::{
    play_replay_match, play_replay_parallel, MctsReplayAgent, PlayedActionReplayAgent,
    RecordedNetworkAgent, ReplayActorAgent, ReplayMatchConfiguration, ReplayMatchError,
    ReplayMatchTask, UnrecordedReplayAgent,
};

const TERMINAL_RING: &str =
    include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");

struct ScriptedAgent {
    actions: VecDeque<Action>,
    decisions: usize,
}

impl ScriptedAgent {
    fn new(actions: Vec<Action>) -> Self {
        Self {
            actions: actions.into(),
            decisions: 0,
        }
    }
}

impl Agent for ScriptedAgent {
    fn select_action(
        &mut self,
        _position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        let expected = self
            .actions
            .pop_front()
            .ok_or_else(|| AgentError::new("script ended before the game"))?;
        self.decisions += 1;
        legal_actions
            .iter()
            .position(|action| *action == expected)
            .ok_or_else(|| AgentError::new(format!("scripted action {expected} is not legal")))
    }

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry {
            decisions: self.decisions,
            ..AgentTelemetry::default()
        }
    }

    fn reset_telemetry(&mut self) {
        self.decisions = 0;
    }
}

#[derive(Clone)]
struct UniformEvaluator;

impl PolicyValueEvaluator for UniformEvaluator {
    fn evaluate(&self, example: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError> {
        let probability = 1.0 / example.legal_actions().len() as f32;
        let policy = vec![probability; example.legal_actions().len()];
        PolicyValueOutput::new(policy, [0.6, 0.1, 0.3])
            .map_err(|source| AgentError::new(source.to_string()))
    }
}

fn terminal_scripts() -> (GameRecord, Vec<Action>, Vec<Action>) {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let mut position = record.initial_position();
    let mut host = Vec::new();
    let mut guest = Vec::new();
    for &action in record.actions() {
        match position.to_move() {
            Player::Host => host.push(action),
            Player::Guest => guest.push(action),
        }
        position.apply(action).unwrap();
    }
    (record, host, guest)
}

#[test]
fn terminal_actor_match_records_replayable_training_decisions() {
    let (record, host_actions, guest_actions) = terminal_scripts();
    let mut host = PlayedActionReplayAgent::new(
        ReplayDigestV1::from_bytes([1; 32]),
        ScriptedAgent::new(host_actions),
    );
    let mut guest = PlayedActionReplayAgent::new(
        ReplayDigestV1::from_bytes([2; 32]),
        ScriptedAgent::new(guest_actions),
    );
    let result = play_replay_match(
        ReplayMatchTask::standard(41, record.setup()),
        ReplayMatchConfiguration {
            decision_soft_limit: 512,
        },
        &mut host,
        &mut guest,
    )
    .unwrap();

    assert_eq!(result.game.record(), &record);
    assert_eq!(
        result.game.record().rules(),
        paisho_core::RuleProfileId::SkudPaiSho2022
    );
    assert_eq!(result.starting_decisions, 0);
    assert!(result.neutral_start.is_none());
    assert_eq!(result.game.decisions().len(), record.actions().len());
    assert_eq!(
        result.host_telemetry.decisions + result.guest_telemetry.decisions,
        record.actions().len()
    );
    assert_eq!(
        result.game.materialize_training_examples().unwrap().len(),
        record.actions().len()
    );
}

#[test]
fn interrupted_actor_game_is_never_made_into_training_data() {
    let digest = ReplayDigestV1::from_bytes([3; 32]);
    let mut host = UnrecordedReplayAgent::new(digest, paisho_ai::RandomAgent::new(1));
    let mut guest = PlayedActionReplayAgent::new(digest, paisho_ai::RandomAgent::new(2));
    let error = play_replay_match(
        ReplayMatchTask::standard(99, StandardSetup::balanced(BasicFlower::Red3)),
        ReplayMatchConfiguration {
            decision_soft_limit: 0,
        },
        &mut host,
        &mut guest,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        ReplayMatchError::DecisionLimit {
            game_id: 99,
            decisions: 0
        }
    ));
}

#[test]
fn network_and_mcts_wrappers_preserve_informative_policy_targets() {
    let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::White4));
    let legal = legal_actions(&position);
    let network_digest = ReplayDigestV1::from_bytes([4; 32]);
    let network = PureNetworkAgent::new(
        UniformEvaluator,
        71,
        NetworkPolicy::Sample {
            temperature: 1.0,
            uniform_mix: 0.0,
        },
    )
    .unwrap();
    let mut network = RecordedNetworkAgent::new(network_digest, network);
    let network_choice = network.select_for_replay(&position, &legal).unwrap();
    let network_target = network_choice.policy_target().unwrap();
    assert_eq!(network_target.kind(), PolicyTargetKindV1::Behavior);
    assert_eq!(network_target.producer(), network_digest);
    assert_eq!(network_target.entries().len(), legal.len());
    assert!((network_choice.behavior_value().unwrap() - 0.3).abs() < 1.0e-6);

    let mcts_digest = ReplayDigestV1::from_bytes([5; 32]);
    let mcts = MctsAgent::new(
        72,
        MctsConfig {
            simulations: 8,
            independent_trees: 1,
            maximum_tree_depth: 8,
            action_rank_batch_size: usize::MAX,
            root_widening_factor: 1.0,
            progressive_widening_factor: 1.0,
            rollout_depth: 0,
            exploration: 1.0,
            heuristic_weights: HeuristicWeights::default(),
        },
    )
    .unwrap();
    let mut mcts = MctsReplayAgent::new(mcts_digest, mcts);
    let mcts_choice = mcts.select_for_replay(&position, &legal).unwrap();
    let mcts_target = mcts_choice.policy_target().unwrap();
    assert_eq!(mcts_target.kind(), PolicyTargetKindV1::MctsVisit);
    assert_eq!(mcts_target.producer(), mcts_digest);
    assert_eq!(mcts_choice.behavior_value(), None);
    let total = mcts_target
        .entries()
        .iter()
        .map(|entry| entry.probability())
        .sum::<f32>();
    assert!((total - 1.0).abs() < 1.0e-5);
}

#[test]
fn parallel_actor_batch_preserves_task_order() {
    let (record, host_actions, guest_actions) = terminal_scripts();
    let tasks = (0..8)
        .map(|game_id| ReplayMatchTask::standard(game_id, record.setup()))
        .collect::<Vec<_>>();
    let results = play_replay_parallel(
        &tasks,
        ReplayMatchConfiguration {
            decision_soft_limit: 512,
        },
        |_| {
            PlayedActionReplayAgent::new(
                ReplayDigestV1::from_bytes([6; 32]),
                ScriptedAgent::new(host_actions.clone()),
            )
        },
        |_| {
            PlayedActionReplayAgent::new(
                ReplayDigestV1::from_bytes([7; 32]),
                ScriptedAgent::new(guest_actions.clone()),
            )
        },
    );

    assert!(results.workers >= 1);
    assert!(results.workers <= results.worker_capacity);
    for (expected, result) in (0_u64..).zip(results.matches) {
        assert_eq!(result.unwrap().game.game_id(), expected);
    }
}

#[test]
fn an_advanced_prefix_is_replayed_but_never_becomes_a_training_target() {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let mut position = record.initial_position();
    let mut cut = None;
    for (index, &action) in record.actions().iter().enumerate() {
        position.apply(action).unwrap();
        if index >= 5
            && position.outcome() == paisho_core::GameOutcome::Ongoing
            && position.phase() == paisho_core::TurnPhase::Main
        {
            cut = Some(index + 1);
            break;
        }
    }
    let cut = cut.expect("terminal fixture exposes an advanced main-phase boundary");
    let mut host_actions = Vec::new();
    let mut guest_actions = Vec::new();
    for &action in &record.actions()[cut..] {
        match position.to_move() {
            Player::Host => host_actions.push(action),
            Player::Guest => guest_actions.push(action),
        }
        position.apply(action).unwrap();
    }
    let mut host = PlayedActionReplayAgent::new(
        ReplayDigestV1::from_bytes([8; 32]),
        ScriptedAgent::new(host_actions),
    );
    let mut guest = PlayedActionReplayAgent::new(
        ReplayDigestV1::from_bytes([9; 32]),
        ScriptedAgent::new(guest_actions),
    );
    let result = play_replay_match(
        ReplayMatchTask {
            game_id: 123,
            setup: record.setup(),
            starting_actions: record.actions()[..cut].to_vec(),
            neutral_start: None,
        },
        ReplayMatchConfiguration {
            decision_soft_limit: 512,
        },
        &mut host,
        &mut guest,
    )
    .unwrap();

    assert_eq!(result.game.record(), &record);
    assert_eq!(result.starting_decisions, cut);
    assert_eq!(result.game.decisions().len(), record.actions().len() - cut);
    assert!(result
        .game
        .decisions()
        .iter()
        .all(|decision| decision.decision_index() >= cut));
}
