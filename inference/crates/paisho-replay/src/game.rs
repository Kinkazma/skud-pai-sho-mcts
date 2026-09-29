use core::fmt;
use std::collections::HashMap;

use paisho_core::{
    Action, ApplyError, GameOutcome, GameRecord, Player, Position, ReplayError, RuleProfileId,
};
use paisho_model::{
    encode_action_v1, ActionEncodingError, ActionEncodingV1, InferenceExampleEncodingError,
    InferenceExampleV1, TerminalPpoExampleV1, TerminalPpoWireError, TrainingExampleV1,
    TrainingWireError, ValueClassV1, ValueTargetError, VALUE_CLASS_COUNT_V1,
};

use crate::{PolicyTargetKindV1, PolicyTargetV1, ReplayDigestV1};

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayDecisionV1 {
    decision_index: usize,
    policy: PolicyTargetV1,
    behavior_value: Option<f32>,
}

impl ReplayDecisionV1 {
    pub const fn new(decision_index: usize, policy: PolicyTargetV1) -> Self {
        Self {
            decision_index,
            policy,
            behavior_value: None,
        }
    }

    /// Attach the frozen behavior network's `P(win) - P(loss)` prediction.
    /// Validation remains centralized in `ReplayGameV1::new` so decoded and
    /// locally produced decisions follow the same contract.
    pub const fn with_behavior_value(mut self, behavior_value: f32) -> Self {
        self.behavior_value = Some(behavior_value);
        self
    }

    pub const fn decision_index(&self) -> usize {
        self.decision_index
    }

    pub const fn policy(&self) -> &PolicyTargetV1 {
        &self.policy
    }

    pub const fn behavior_value(&self) -> Option<f32> {
        self.behavior_value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayGameV1 {
    game_id: u64,
    host_agent: ReplayDigestV1,
    guest_agent: ReplayDigestV1,
    record: GameRecord,
    decisions: Vec<ReplayDecisionV1>,
    outcome: GameOutcome,
}

impl ReplayGameV1 {
    pub fn new(
        game_id: u64,
        host_agent: ReplayDigestV1,
        guest_agent: ReplayDigestV1,
        record: GameRecord,
        mut decisions: Vec<ReplayDecisionV1>,
    ) -> Result<Self, ReplayValidationError> {
        decisions.sort_by_key(ReplayDecisionV1::decision_index);
        validate_decision_indices(record.actions().len(), &decisions)?;
        let initial_position = record.initial_position();
        require_rule_profile(&initial_position, record.rules())?;
        let final_position = record
            .replay()
            .map_err(ReplayValidationError::InvalidRecord)?;
        let outcome = final_position.outcome();
        if outcome == GameOutcome::Ongoing {
            return Err(ReplayValidationError::NonTerminalGame(game_id));
        }
        let game = Self {
            game_id,
            host_agent,
            guest_agent,
            record,
            decisions,
            outcome,
        };
        game.visit_training_examples(drop)?;
        Ok(game)
    }

    pub const fn game_id(&self) -> u64 {
        self.game_id
    }

    pub const fn host_agent(&self) -> ReplayDigestV1 {
        self.host_agent
    }

    pub const fn guest_agent(&self) -> ReplayDigestV1 {
        self.guest_agent
    }

    pub const fn record(&self) -> &GameRecord {
        &self.record
    }

    pub fn decisions(&self) -> &[ReplayDecisionV1] {
        &self.decisions
    }

    pub const fn outcome(&self) -> GameOutcome {
        self.outcome
    }

    pub fn materialize_training_examples(
        &self,
    ) -> Result<Vec<ReplayTrainingExampleV1>, ReplayValidationError> {
        let mut examples = Vec::with_capacity(self.decisions.len());
        self.visit_training_examples(|example| examples.push(example))?;
        Ok(examples)
    }

    fn visit_training_examples(
        &self,
        mut visit: impl FnMut(ReplayTrainingExampleV1),
    ) -> Result<(), ReplayValidationError> {
        let mut position = self.record.initial_position();
        require_rule_profile(&position, self.record.rules())?;

        let mut next_decision = 0;
        for (decision_index, played_action) in self.record.actions().iter().copied().enumerate() {
            if self
                .decisions
                .get(next_decision)
                .is_some_and(|decision| decision.decision_index == decision_index)
            {
                let decision = &self.decisions[next_decision];
                visit(materialize_decision(
                    self.game_id,
                    decision,
                    played_action,
                    &position,
                    self.agent_for(position.to_move()),
                    self.outcome,
                    self.record.actions().len(),
                )?);
                next_decision += 1;
            }
            position
                .apply(played_action)
                .map_err(|source| ReplayValidationError::Apply {
                    decision_index,
                    action: played_action,
                    source,
                })?;
        }
        debug_assert_eq!(next_decision, self.decisions.len());
        Ok(())
    }

    pub fn materialize_training_example(
        &self,
        selected_decision: usize,
    ) -> Result<ReplayTrainingExampleV1, ReplayValidationError> {
        let decision = self.decisions.get(selected_decision).ok_or(
            ReplayValidationError::SelectedDecisionOutOfRange {
                selected_decision,
                decision_count: self.decisions.len(),
            },
        )?;
        let mut position = self.record.initial_position();
        require_rule_profile(&position, self.record.rules())?;
        for (decision_index, &action) in self.record.actions()[..decision.decision_index()]
            .iter()
            .enumerate()
        {
            position
                .apply(action)
                .map_err(|source| ReplayValidationError::Apply {
                    decision_index,
                    action,
                    source,
                })?;
        }
        materialize_decision(
            self.game_id,
            decision,
            self.record.actions()[decision.decision_index()],
            &position,
            self.agent_for(position.to_move()),
            self.outcome,
            self.record.actions().len(),
        )
    }

    const fn agent_for(&self, player: Player) -> ReplayDigestV1 {
        match player {
            Player::Host => self.host_agent,
            Player::Guest => self.guest_agent,
        }
    }
}

fn validate_decision_indices(
    action_count: usize,
    decisions: &[ReplayDecisionV1],
) -> Result<(), ReplayValidationError> {
    if decisions.is_empty() {
        return Err(ReplayValidationError::NoTrainingDecision);
    }
    for pair in decisions.windows(2) {
        if pair[0].decision_index == pair[1].decision_index {
            return Err(ReplayValidationError::DuplicateDecisionIndex(
                pair[0].decision_index,
            ));
        }
    }
    if let Some(decision) = decisions
        .iter()
        .find(|decision| decision.decision_index >= action_count)
    {
        return Err(ReplayValidationError::DecisionOutOfRange {
            decision_index: decision.decision_index,
            action_count,
        });
    }
    for decision in decisions {
        let Some(value) = decision.behavior_value else {
            continue;
        };
        if decision.policy.kind() != PolicyTargetKindV1::Behavior {
            return Err(ReplayValidationError::BehaviorValueWithoutBehaviorPolicy {
                decision_index: decision.decision_index,
                policy_kind: decision.policy.kind(),
            });
        }
        if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
            return Err(ReplayValidationError::InvalidBehaviorValue {
                decision_index: decision.decision_index,
                value,
            });
        }
    }
    Ok(())
}

fn require_rule_profile(
    position: &Position,
    record_profile: RuleProfileId,
) -> Result<(), ReplayValidationError> {
    if position.rule_profile() == record_profile {
        Ok(())
    } else {
        Err(ReplayValidationError::RuleProfileMismatch {
            record: record_profile,
            engine: position.rule_profile(),
        })
    }
}

fn materialize_decision(
    game_id: u64,
    decision: &ReplayDecisionV1,
    played_action: Action,
    position: &Position,
    acting_agent: ReplayDigestV1,
    final_outcome: GameOutcome,
    game_decisions: usize,
) -> Result<ReplayTrainingExampleV1, ReplayValidationError> {
    let perspective = position.to_move();
    let inference = InferenceExampleV1::from_position(position).map_err(|source| {
        ReplayValidationError::Encoding {
            decision_index: decision.decision_index,
            source,
        }
    })?;
    let mut legal_indices = HashMap::with_capacity(inference.legal_actions().len());
    for (index, &action) in inference.legal_actions().iter().enumerate() {
        if legal_indices.insert(action, index).is_some() {
            return Err(ReplayValidationError::DuplicateLegalAddress {
                decision_index: decision.decision_index,
                action,
            });
        }
    }
    let played_action = encode_action_v1(played_action, perspective).map_err(|source| {
        ReplayValidationError::PlayedActionEncoding {
            decision_index: decision.decision_index,
            source,
        }
    })?;
    let played_action_index = legal_indices.get(&played_action).copied().ok_or(
        ReplayValidationError::PlayedActionNotLegal {
            decision_index: decision.decision_index,
            action: played_action,
        },
    )?;
    if decision.policy.kind() == PolicyTargetKindV1::PlayedAction
        && decision.policy.entries()[0].action() != played_action
    {
        return Err(ReplayValidationError::PlayedActionTargetMismatch {
            decision_index: decision.decision_index,
            played: played_action,
            target: decision.policy.entries()[0].action(),
        });
    }
    let mut policy_target = vec![0.0; inference.legal_actions().len()];
    for entry in decision.policy.entries() {
        let action = entry.action();
        let Some(&index) = legal_indices.get(&action) else {
            return Err(ReplayValidationError::PolicyActionNotLegal {
                decision_index: decision.decision_index,
                action,
            });
        };
        policy_target[index] = entry.probability();
    }
    if decision.policy.kind() == PolicyTargetKindV1::Behavior
        && policy_target[played_action_index] <= 0.0
    {
        return Err(ReplayValidationError::BehaviorExcludesPlayedAction {
            decision_index: decision.decision_index,
            action: played_action,
        });
    }
    if decision.policy.kind() == PolicyTargetKindV1::Behavior
        && decision.policy.producer() != acting_agent
    {
        return Err(ReplayValidationError::BehaviorProducerMismatch {
            decision_index: decision.decision_index,
            acting_agent,
            producer: decision.policy.producer(),
        });
    }
    let value_class = ValueClassV1::from_terminal_outcome(final_outcome, perspective)
        .map_err(ReplayValidationError::ValueTarget)?;
    Ok(ReplayTrainingExampleV1 {
        game_id,
        remaining_decisions: game_decisions - decision.decision_index,
        decision_index: decision.decision_index,
        perspective,
        played_action,
        played_action_index,
        inference,
        policy_target,
        value_target: value_class.one_hot(),
        value_class,
        policy_kind: decision.policy.kind(),
        policy_producer: decision.policy.producer(),
        behavior_value: decision.behavior_value,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayTrainingExampleV1 {
    game_id: u64,
    remaining_decisions: usize,
    decision_index: usize,
    perspective: Player,
    played_action: ActionEncodingV1,
    played_action_index: usize,
    inference: InferenceExampleV1,
    policy_target: Vec<f32>,
    value_target: [f32; VALUE_CLASS_COUNT_V1],
    value_class: ValueClassV1,
    policy_kind: PolicyTargetKindV1,
    policy_producer: ReplayDigestV1,
    behavior_value: Option<f32>,
}

impl ReplayTrainingExampleV1 {
    pub const fn remaining_decisions(&self) -> usize {
        self.remaining_decisions
    }

    pub const fn game_id(&self) -> u64 {
        self.game_id
    }

    pub const fn decision_index(&self) -> usize {
        self.decision_index
    }

    pub const fn perspective(&self) -> Player {
        self.perspective
    }

    pub const fn played_action(&self) -> ActionEncodingV1 {
        self.played_action
    }

    pub const fn played_action_index(&self) -> usize {
        self.played_action_index
    }

    pub const fn inference(&self) -> &InferenceExampleV1 {
        &self.inference
    }

    pub fn policy_target(&self) -> &[f32] {
        &self.policy_target
    }

    pub const fn value_target(&self) -> &[f32; VALUE_CLASS_COUNT_V1] {
        &self.value_target
    }

    pub const fn value_class(&self) -> ValueClassV1 {
        self.value_class
    }

    pub const fn terminal_return(&self) -> f32 {
        self.value_class.signed_return()
    }

    pub fn played_behavior_probability(&self) -> Option<f32> {
        (self.policy_kind == PolicyTargetKindV1::Behavior)
            .then_some(self.policy_target[self.played_action_index])
    }

    pub const fn policy_kind(&self) -> PolicyTargetKindV1 {
        self.policy_kind
    }

    pub const fn policy_producer(&self) -> ReplayDigestV1 {
        self.policy_producer
    }

    /// Frozen behavior-network baseline captured during actor inference.
    /// Legacy replay shards legitimately return `None` and are recomputed by
    /// the learner through its compatibility path.
    pub const fn behavior_value(&self) -> Option<f32> {
        self.behavior_value
    }

    pub fn to_training_example(&self) -> Result<TrainingExampleV1, TrainingWireError> {
        TrainingExampleV1::new(
            self.inference.clone(),
            self.policy_target.clone(),
            self.value_target,
        )
    }

    pub fn to_terminal_ppo_example(
        &self,
        actor_value: f32,
    ) -> Result<TerminalPpoExampleV1, ReplayTerminalPpoExampleError> {
        let behavior_probability = self.played_behavior_probability().ok_or(
            ReplayTerminalPpoExampleError::NotBehaviorPolicy(self.policy_kind),
        )?;
        TerminalPpoExampleV1::new(
            self.inference.clone(),
            self.played_action_index,
            behavior_probability,
            self.value_class,
            actor_value,
        )
        .map_err(ReplayTerminalPpoExampleError::Wire)
    }
}

#[derive(Debug)]
pub enum ReplayTerminalPpoExampleError {
    NotBehaviorPolicy(PolicyTargetKindV1),
    Wire(TerminalPpoWireError),
}

impl fmt::Display for ReplayTerminalPpoExampleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotBehaviorPolicy(kind) => write!(
                formatter,
                "terminal PPO requires a Behavior policy, got {kind:?}"
            ),
            Self::Wire(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for ReplayTerminalPpoExampleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Wire(source) => Some(source),
            Self::NotBehaviorPolicy(_) => None,
        }
    }
}

#[derive(Debug)]
pub enum ReplayValidationError {
    NoTrainingDecision,
    DuplicateDecisionIndex(usize),
    DecisionOutOfRange {
        decision_index: usize,
        action_count: usize,
    },
    SelectedDecisionOutOfRange {
        selected_decision: usize,
        decision_count: usize,
    },
    RuleProfileMismatch {
        record: RuleProfileId,
        engine: RuleProfileId,
    },
    InvalidRecord(ReplayError),
    NonTerminalGame(u64),
    Encoding {
        decision_index: usize,
        source: InferenceExampleEncodingError,
    },
    PlayedActionEncoding {
        decision_index: usize,
        source: ActionEncodingError,
    },
    DuplicateLegalAddress {
        decision_index: usize,
        action: ActionEncodingV1,
    },
    PlayedActionNotLegal {
        decision_index: usize,
        action: ActionEncodingV1,
    },
    BehaviorExcludesPlayedAction {
        decision_index: usize,
        action: ActionEncodingV1,
    },
    BehaviorProducerMismatch {
        decision_index: usize,
        acting_agent: ReplayDigestV1,
        producer: ReplayDigestV1,
    },
    BehaviorValueWithoutBehaviorPolicy {
        decision_index: usize,
        policy_kind: PolicyTargetKindV1,
    },
    InvalidBehaviorValue {
        decision_index: usize,
        value: f32,
    },
    PlayedActionTargetMismatch {
        decision_index: usize,
        played: ActionEncodingV1,
        target: ActionEncodingV1,
    },
    PolicyActionNotLegal {
        decision_index: usize,
        action: ActionEncodingV1,
    },
    ValueTarget(ValueTargetError),
    Apply {
        decision_index: usize,
        action: Action,
        source: ApplyError,
    },
}

impl fmt::Display for ReplayValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoTrainingDecision => {
                formatter.write_str("a replay game needs at least one training decision")
            }
            Self::DuplicateDecisionIndex(index) => {
                write!(formatter, "replay game repeats decision index {index}")
            }
            Self::DecisionOutOfRange {
                decision_index,
                action_count,
            } => write!(
                formatter,
                "training decision {decision_index} is outside a record of {action_count} actions"
            ),
            Self::SelectedDecisionOutOfRange {
                selected_decision,
                decision_count,
            } => write!(
                formatter,
                "selected training decision {selected_decision} is outside {decision_count} retained decisions"
            ),
            Self::RuleProfileMismatch { record, engine } => write!(
                formatter,
                "record rule profile {record} does not match engine profile {engine}"
            ),
            Self::InvalidRecord(source) => {
                write!(formatter, "invalid replay game record: {source}")
            }
            Self::NonTerminalGame(game_id) => {
                write!(formatter, "replay game {game_id} has no terminal result")
            }
            Self::Encoding {
                decision_index,
                source,
            } => write!(
                formatter,
                "cannot encode replay decision {decision_index}: {source}"
            ),
            Self::PlayedActionEncoding {
                decision_index,
                source,
            } => write!(
                formatter,
                "cannot encode played action at replay decision {decision_index}: {source}"
            ),
            Self::DuplicateLegalAddress {
                decision_index,
                action,
            } => write!(
                formatter,
                "decision {decision_index} generated duplicate legal address {:?}",
                action.slots()
            ),
            Self::PlayedActionNotLegal {
                decision_index,
                action,
            } => write!(
                formatter,
                "played action {:?} is absent from decision {decision_index}'s legal set",
                action.slots()
            ),
            Self::BehaviorExcludesPlayedAction {
                decision_index,
                action,
            } => write!(
                formatter,
                "behavior policy assigns zero probability to played action {:?} at decision {decision_index}",
                action.slots()
            ),
            Self::BehaviorProducerMismatch {
                decision_index,
                acting_agent,
                producer,
            } => write!(
                formatter,
                "behavior producer {producer} does not match acting agent {acting_agent} at decision {decision_index}"
            ),
            Self::BehaviorValueWithoutBehaviorPolicy {
                decision_index,
                policy_kind,
            } => write!(
                formatter,
                "decision {decision_index} stores a behavior value for {policy_kind:?} policy data"
            ),
            Self::InvalidBehaviorValue {
                decision_index,
                value,
            } => write!(
                formatter,
                "decision {decision_index} stores invalid behavior value {value}; expected a finite value in [-1, 1]"
            ),
            Self::PlayedActionTargetMismatch {
                decision_index,
                played,
                target,
            } => write!(
                formatter,
                "played-action target {:?} does not match played action {:?} at decision {decision_index}",
                target.slots(),
                played.slots()
            ),
            Self::PolicyActionNotLegal {
                decision_index,
                action,
            } => write!(
                formatter,
                "policy action {:?} is not legal at decision {decision_index}",
                action.slots()
            ),
            Self::ValueTarget(source) => source.fmt(formatter),
            Self::Apply {
                decision_index,
                action,
                source,
            } => write!(
                formatter,
                "cannot apply replay decision {decision_index} (`{action}`): {source}"
            ),
        }
    }
}

impl std::error::Error for ReplayValidationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRecord(source) => Some(source),
            Self::Encoding { source, .. } => Some(source),
            Self::PlayedActionEncoding { source, .. } => Some(source),
            Self::ValueTarget(source) => Some(source),
            Self::Apply { source, .. } => Some(source),
            _ => None,
        }
    }
}
