use std::path::PathBuf;

use paisho_model::{TerminalPpoParametersV1, TERMINAL_PPO_OBJECTIVE_V1};
use paisho_replay::ReplayDigestV1;

pub const SUPERVISED_OBJECTIVE_V1: &str = "supervised-policy-value-v1";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalPpoLearnerObjectiveV1 {
    behavior_producer: ReplayDigestV1,
    actor_checkpoint_sha256: Option<ReplayDigestV1>,
    parameters: TerminalPpoParametersV1,
}

impl TerminalPpoLearnerObjectiveV1 {
    pub const fn new(
        behavior_producer: ReplayDigestV1,
        actor_checkpoint_sha256: Option<ReplayDigestV1>,
        parameters: TerminalPpoParametersV1,
    ) -> Self {
        Self {
            behavior_producer,
            actor_checkpoint_sha256,
            parameters,
        }
    }

    pub const fn behavior_producer(self) -> ReplayDigestV1 {
        self.behavior_producer
    }

    pub const fn actor_checkpoint_sha256(self) -> Option<ReplayDigestV1> {
        self.actor_checkpoint_sha256
    }

    pub const fn parameters(self) -> TerminalPpoParametersV1 {
        self.parameters
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LearnerObjectiveV1 {
    SupervisedPolicyValue,
    TerminalPpo(TerminalPpoLearnerObjectiveV1),
}

impl LearnerObjectiveV1 {
    pub const fn identifier(self) -> &'static str {
        match self {
            Self::SupervisedPolicyValue => SUPERVISED_OBJECTIVE_V1,
            Self::TerminalPpo(_) => TERMINAL_PPO_OBJECTIVE_V1,
        }
    }

    pub const fn terminal_ppo(self) -> Option<TerminalPpoLearnerObjectiveV1> {
        match self {
            Self::TerminalPpo(objective) => Some(objective),
            Self::SupervisedPolicyValue => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum LearnerObjectiveConfiguration {
    SupervisedPolicyValue,
    TerminalPpo {
        behavior_producer: ReplayDigestV1,
        actor_checkpoint: Option<PathBuf>,
        parameters: TerminalPpoParametersV1,
    },
}

impl LearnerObjectiveConfiguration {
    pub const fn supervised_policy_value() -> Self {
        Self::SupervisedPolicyValue
    }

    pub fn terminal_ppo(
        behavior_producer: ReplayDigestV1,
        actor_checkpoint: Option<PathBuf>,
        parameters: TerminalPpoParametersV1,
    ) -> Self {
        Self::TerminalPpo {
            behavior_producer,
            actor_checkpoint,
            parameters,
        }
    }
}

impl Default for LearnerObjectiveConfiguration {
    fn default() -> Self {
        Self::SupervisedPolicyValue
    }
}
