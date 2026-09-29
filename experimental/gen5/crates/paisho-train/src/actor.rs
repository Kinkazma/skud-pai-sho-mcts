use core::fmt;
use std::collections::HashSet;
use std::sync::Mutex;

use paisho_ai::{
    Agent, AgentError, AgentTelemetry, MctsAgent, PolicyValueEvaluator, PureNetworkAgent,
};
use paisho_core::{
    legal_actions, Action, ApplyError, GameOutcome, GameRecord, Player, Position, RuleProfileId,
    StandardSetup, TurnPhase,
};
use paisho_model::{encode_action_v1, ActionEncodingError};
use paisho_replay::{
    PolicyEntryV1, PolicyTargetKindV1, PolicyTargetV1, PolicyTargetV1Error, ReplayDecisionV1,
    ReplayDigestV1, ReplayGameV1, ReplayValidationError,
};
use rayon::prelude::*;

use crate::{NeutralStartProvenanceV1, NeutralStartV1};

pub struct ReplayActorChoice {
    selected_index: usize,
    policy_target: Option<PolicyTargetV1>,
    behavior_value: Option<f32>,
}

impl ReplayActorChoice {
    pub const fn new(selected_index: usize, policy_target: Option<PolicyTargetV1>) -> Self {
        Self {
            selected_index,
            policy_target,
            behavior_value: None,
        }
    }

    pub const fn with_behavior_value(mut self, behavior_value: f32) -> Self {
        self.behavior_value = Some(behavior_value);
        self
    }

    pub const fn selected_index(&self) -> usize {
        self.selected_index
    }

    pub const fn policy_target(&self) -> Option<&PolicyTargetV1> {
        self.policy_target.as_ref()
    }

    pub const fn behavior_value(&self) -> Option<f32> {
        self.behavior_value
    }
}

pub trait ReplayActorAgent {
    fn digest(&self) -> ReplayDigestV1;

    fn select_for_replay(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<ReplayActorChoice, AgentError>;

    fn telemetry(&self) -> AgentTelemetry;

    fn reset_telemetry(&mut self);
}

pub struct RecordedNetworkAgent<Evaluator> {
    digest: ReplayDigestV1,
    agent: PureNetworkAgent<Evaluator>,
}

impl<Evaluator> RecordedNetworkAgent<Evaluator> {
    pub const fn new(digest: ReplayDigestV1, agent: PureNetworkAgent<Evaluator>) -> Self {
        Self { digest, agent }
    }

    pub const fn agent(&self) -> &PureNetworkAgent<Evaluator> {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut PureNetworkAgent<Evaluator> {
        &mut self.agent
    }
}

impl<Evaluator> ReplayActorAgent for RecordedNetworkAgent<Evaluator>
where
    Evaluator: PolicyValueEvaluator,
{
    fn digest(&self) -> ReplayDigestV1 {
        self.digest
    }

    fn select_for_replay(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<ReplayActorChoice, AgentError> {
        let decision = self
            .agent
            .decide(position, legal_actions)
            .map_err(|source| AgentError::new(format!("pure-network actor failed: {source}")))?;
        let target = policy_from_probabilities(
            PolicyTargetKindV1::Behavior,
            self.digest,
            position,
            legal_actions,
            decision.behavior_policy(),
        )
        .map_err(|source| AgentError::new(format!("cannot record network policy: {source}")))?;
        let values = decision.value_probabilities();
        Ok(
            ReplayActorChoice::new(decision.selected_index(), Some(target))
                .with_behavior_value(values[0] - values[2]),
        )
    }

    fn telemetry(&self) -> AgentTelemetry {
        self.agent.telemetry()
    }

    fn reset_telemetry(&mut self) {
        self.agent.reset_telemetry();
    }
}

pub struct UnrecordedReplayAgent<Inner> {
    digest: ReplayDigestV1,
    agent: Inner,
}

impl<Inner> UnrecordedReplayAgent<Inner> {
    pub const fn new(digest: ReplayDigestV1, agent: Inner) -> Self {
        Self { digest, agent }
    }
}

impl<Inner> ReplayActorAgent for UnrecordedReplayAgent<Inner>
where
    Inner: Agent,
{
    fn digest(&self) -> ReplayDigestV1 {
        self.digest
    }

    fn select_for_replay(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<ReplayActorChoice, AgentError> {
        self.agent
            .select_action(position, legal_actions)
            .map(|selected| ReplayActorChoice::new(selected, None))
    }

    fn telemetry(&self) -> AgentTelemetry {
        self.agent.telemetry()
    }

    fn reset_telemetry(&mut self) {
        self.agent.reset_telemetry();
    }
}

pub struct PlayedActionReplayAgent<Inner> {
    digest: ReplayDigestV1,
    agent: Inner,
}

impl<Inner> PlayedActionReplayAgent<Inner> {
    pub const fn new(digest: ReplayDigestV1, agent: Inner) -> Self {
        Self { digest, agent }
    }
}

impl<Inner> ReplayActorAgent for PlayedActionReplayAgent<Inner>
where
    Inner: Agent,
{
    fn digest(&self) -> ReplayDigestV1 {
        self.digest
    }

    fn select_for_replay(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<ReplayActorChoice, AgentError> {
        let selected = self.agent.select_action(position, legal_actions)?;
        let action = legal_actions.get(selected).copied().ok_or_else(|| {
            AgentError::new(format!(
                "recorded agent selected {selected} from {} legal actions",
                legal_actions.len()
            ))
        })?;
        let encoded = encode_action_v1(action, position.to_move())
            .map_err(|source| AgentError::new(format!("cannot encode played action: {source}")))?;
        let target = PolicyTargetV1::one_hot(self.digest, encoded)
            .map_err(|source| AgentError::new(format!("cannot record played action: {source}")))?;
        Ok(ReplayActorChoice::new(selected, Some(target)))
    }

    fn telemetry(&self) -> AgentTelemetry {
        self.agent.telemetry()
    }

    fn reset_telemetry(&mut self) {
        self.agent.reset_telemetry();
    }
}

pub struct MctsReplayAgent {
    digest: ReplayDigestV1,
    agent: MctsAgent,
}

impl MctsReplayAgent {
    pub const fn new(digest: ReplayDigestV1, agent: MctsAgent) -> Self {
        Self { digest, agent }
    }
}

impl ReplayActorAgent for MctsReplayAgent {
    fn digest(&self) -> ReplayDigestV1 {
        self.digest
    }

    fn select_for_replay(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<ReplayActorChoice, AgentError> {
        let selected = self.agent.select_action(position, legal_actions)?;
        let report = self
            .agent
            .last_report()
            .expect("MCTS selection always retains its search report");
        let visits = report
            .actions
            .iter()
            .map(|entry| entry.visits)
            .sum::<usize>();
        if visits == 0 {
            return Err(AgentError::new("MCTS search produced no root visits"));
        }
        let probabilities = report
            .actions
            .iter()
            .map(|entry| entry.visits as f32 / visits as f32)
            .collect::<Vec<_>>();
        let target = policy_from_probabilities(
            PolicyTargetKindV1::MctsVisit,
            self.digest,
            position,
            legal_actions,
            &probabilities,
        )
        .map_err(|source| AgentError::new(format!("cannot record MCTS visits: {source}")))?;
        Ok(ReplayActorChoice::new(selected, Some(target)))
    }

    fn telemetry(&self) -> AgentTelemetry {
        self.agent.telemetry()
    }

    fn reset_telemetry(&mut self) {
        self.agent.reset_telemetry();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayMatchConfiguration {
    pub decision_soft_limit: usize,
}

impl Default for ReplayMatchConfiguration {
    fn default() -> Self {
        Self {
            decision_soft_limit: 2_048,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayMatchTask {
    pub game_id: u64,
    pub setup: StandardSetup,
    pub starting_actions: Vec<Action>,
    pub neutral_start: Option<NeutralStartProvenanceV1>,
}

impl ReplayMatchTask {
    pub fn standard(game_id: u64, setup: StandardSetup) -> Self {
        Self {
            game_id,
            setup,
            starting_actions: Vec::new(),
            neutral_start: None,
        }
    }

    pub fn from_neutral(game_id: u64, start: &NeutralStartV1) -> Self {
        Self {
            game_id,
            setup: start.prefix().setup(),
            starting_actions: start.prefix().actions().to_vec(),
            neutral_start: Some(start.provenance()),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayMatchResult {
    pub game: ReplayGameV1,
    pub host_telemetry: AgentTelemetry,
    pub guest_telemetry: AgentTelemetry,
    pub starting_decisions: usize,
    pub neutral_start: Option<NeutralStartProvenanceV1>,
}

#[derive(Debug)]
pub struct ParallelReplayResults {
    pub workers: usize,
    pub worker_capacity: usize,
    pub matches: Vec<Result<ReplayMatchResult, ReplayMatchError>>,
}

pub fn play_replay_match<Host: ReplayActorAgent, Guest: ReplayActorAgent>(
    task: ReplayMatchTask,
    configuration: ReplayMatchConfiguration,
    host: &mut Host,
    guest: &mut Guest,
) -> Result<ReplayMatchResult, ReplayMatchError> {
    host.reset_telemetry();
    guest.reset_telemetry();
    let host_digest = host.digest();
    let guest_digest = guest.digest();
    // The retired PPO pipeline and its encoded checkpoints remain on V1.
    let mut record = GameRecord::with_rules(task.setup, RuleProfileId::SkudPaiSho2022);
    let mut position = record.initial_position();
    for (index, action) in task.starting_actions.iter().copied().enumerate() {
        position
            .apply(action)
            .map_err(|source| ReplayMatchError::StartingActionRejected {
                game_id: task.game_id,
                action_number: index + 1,
                action,
                source,
            })?;
        record.push(action);
    }
    let starting_decisions = record.actions().len();
    if position.outcome() != GameOutcome::Ongoing {
        return Err(ReplayMatchError::StartingPositionTerminal {
            game_id: task.game_id,
            decisions: starting_decisions,
        });
    }
    if position.phase() != TurnPhase::Main {
        return Err(ReplayMatchError::StartingPositionInBonus {
            game_id: task.game_id,
            decisions: starting_decisions,
        });
    }
    let mut retained = Vec::new();
    let mut played_decisions = 0_usize;
    let mut record_decision = starting_decisions;

    loop {
        if position.outcome() != GameOutcome::Ongoing {
            let game = ReplayGameV1::new(task.game_id, host_digest, guest_digest, record, retained)
                .map_err(ReplayMatchError::Replay)?;
            return Ok(ReplayMatchResult {
                game,
                host_telemetry: host.telemetry(),
                guest_telemetry: guest.telemetry(),
                starting_decisions,
                neutral_start: task.neutral_start,
            });
        }
        if played_decisions >= configuration.decision_soft_limit
            && position.phase() == TurnPhase::Main
        {
            return Err(ReplayMatchError::DecisionLimit {
                game_id: task.game_id,
                decisions: played_decisions,
            });
        }

        let legal = legal_actions(&position);
        let player = position.to_move();
        if legal.is_empty() {
            return Err(ReplayMatchError::NoLegalAction { player });
        }
        let choice = match player {
            Player::Host => host.select_for_replay(&position, &legal),
            Player::Guest => guest.select_for_replay(&position, &legal),
        }
        .map_err(|source| ReplayMatchError::AgentFailure { player, source })?;
        let action = legal.get(choice.selected_index).copied().ok_or(
            ReplayMatchError::AgentChoiceOutOfRange {
                player,
                selected: choice.selected_index,
                legal_action_count: legal.len(),
            },
        )?;
        if let Some(policy) = choice.policy_target {
            let mut decision = ReplayDecisionV1::new(record_decision, policy);
            if let Some(value) = choice.behavior_value {
                decision = decision.with_behavior_value(value);
            }
            retained.push(decision);
        }
        position
            .apply(action)
            .map_err(|source| ReplayMatchError::GeneratedActionRejected {
                player,
                action,
                source,
            })?;
        record.push(action);
        played_decisions = played_decisions
            .checked_add(1)
            .ok_or(ReplayMatchError::DecisionCounterOverflow)?;
        record_decision = record_decision
            .checked_add(1)
            .ok_or(ReplayMatchError::DecisionCounterOverflow)?;
    }
}

pub fn play_replay_parallel<Host, Guest, MakeHost, MakeGuest>(
    tasks: &[ReplayMatchTask],
    configuration: ReplayMatchConfiguration,
    make_host: MakeHost,
    make_guest: MakeGuest,
) -> ParallelReplayResults
where
    Host: ReplayActorAgent + Send,
    Guest: ReplayActorAgent + Send,
    MakeHost: Fn(&ReplayMatchTask) -> Host + Sync,
    MakeGuest: Fn(&ReplayMatchTask) -> Guest + Sync,
{
    let observed_workers = Mutex::new(HashSet::new());
    let matches = tasks
        .par_iter()
        .map(|task| {
            if let Some(index) = rayon::current_thread_index() {
                observed_workers
                    .lock()
                    .expect("worker observation lock is not poisoned")
                    .insert(index);
            }
            let mut host = make_host(task);
            let mut guest = make_guest(task);
            play_replay_match(task.clone(), configuration, &mut host, &mut guest)
        })
        .collect();
    ParallelReplayResults {
        workers: observed_workers
            .into_inner()
            .expect("worker observation lock is not poisoned")
            .len(),
        worker_capacity: rayon::current_num_threads(),
        matches,
    }
}

fn policy_from_probabilities(
    kind: PolicyTargetKindV1,
    producer: ReplayDigestV1,
    position: &Position,
    legal_actions: &[Action],
    probabilities: &[f32],
) -> Result<PolicyTargetV1, RecordedPolicyError> {
    if legal_actions.len() != probabilities.len() {
        return Err(RecordedPolicyError::LengthMismatch {
            legal_actions: legal_actions.len(),
            probabilities: probabilities.len(),
        });
    }
    let entries = legal_actions
        .iter()
        .copied()
        .zip(probabilities.iter().copied())
        .enumerate()
        .filter_map(|(index, (action, probability))| {
            (probability > 0.0).then_some((index, action, probability))
        })
        .map(|(index, action, probability)| {
            let action = encode_action_v1(action, position.to_move())
                .map_err(|source| RecordedPolicyError::ActionEncoding { index, source })?;
            PolicyEntryV1::new(action, probability).map_err(RecordedPolicyError::Target)
        })
        .collect::<Result<Vec<_>, _>>()?;
    PolicyTargetV1::new(kind, producer, entries).map_err(RecordedPolicyError::Target)
}

#[derive(Debug)]
enum RecordedPolicyError {
    LengthMismatch {
        legal_actions: usize,
        probabilities: usize,
    },
    ActionEncoding {
        index: usize,
        source: ActionEncodingError,
    },
    Target(PolicyTargetV1Error),
}

impl fmt::Display for RecordedPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                legal_actions,
                probabilities,
            } => write!(
                formatter,
                "policy has {probabilities} probabilities for {legal_actions} legal actions"
            ),
            Self::ActionEncoding { index, source } => {
                write!(formatter, "cannot encode policy action {index}: {source}")
            }
            Self::Target(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for RecordedPolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ActionEncoding { source, .. } => Some(source),
            Self::Target(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ReplayMatchError {
    StartingActionRejected {
        game_id: u64,
        action_number: usize,
        action: Action,
        source: ApplyError,
    },
    StartingPositionTerminal {
        game_id: u64,
        decisions: usize,
    },
    StartingPositionInBonus {
        game_id: u64,
        decisions: usize,
    },
    DecisionLimit {
        game_id: u64,
        decisions: usize,
    },
    NoLegalAction {
        player: Player,
    },
    AgentFailure {
        player: Player,
        source: AgentError,
    },
    AgentChoiceOutOfRange {
        player: Player,
        selected: usize,
        legal_action_count: usize,
    },
    GeneratedActionRejected {
        player: Player,
        action: Action,
        source: ApplyError,
    },
    DecisionCounterOverflow,
    Replay(ReplayValidationError),
}

impl fmt::Display for ReplayMatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartingActionRejected {
                game_id,
                action_number,
                action,
                source,
            } => write!(
                formatter,
                "replay game {game_id} rejects starting action {action_number} ({action}): {source}"
            ),
            Self::StartingPositionTerminal { game_id, decisions } => write!(
                formatter,
                "replay game {game_id} starts from a terminal prefix of {decisions} decisions"
            ),
            Self::StartingPositionInBonus { game_id, decisions } => write!(
                formatter,
                "replay game {game_id} starts inside a Harmony Bonus after {decisions} decisions"
            ),
            Self::DecisionLimit { game_id, decisions } => write!(
                formatter,
                "replay game {game_id} did not finish within {decisions} decisions"
            ),
            Self::NoLegalAction { player } => {
                write!(formatter, "{player:?} has no legal action in an ongoing replay game")
            }
            Self::AgentFailure { player, source } => {
                write!(formatter, "{player:?} replay actor failed: {source}")
            }
            Self::AgentChoiceOutOfRange {
                player,
                selected,
                legal_action_count,
            } => write!(
                formatter,
                "{player:?} replay actor selected {selected}, but only {legal_action_count} actions are legal"
            ),
            Self::GeneratedActionRejected {
                player,
                action,
                source,
            } => write!(
                formatter,
                "the engine rejected replay action {action} for {player:?}: {source}"
            ),
            Self::DecisionCounterOverflow => {
                formatter.write_str("replay decision counter overflow")
            }
            Self::Replay(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for ReplayMatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StartingActionRejected { source, .. }
            | Self::GeneratedActionRejected { source, .. } => Some(source),
            Self::AgentFailure { source, .. } => Some(source),
            Self::Replay(source) => Some(source),
            _ => None,
        }
    }
}
