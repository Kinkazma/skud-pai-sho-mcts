use core::fmt;

use paisho_core::{Action, Position};

use crate::StableRng;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AgentTelemetry {
    pub decisions: usize,
    pub simulations: usize,
    pub evaluated_actions: usize,
    pub expanded_nodes: usize,
    pub generated_nodes: usize,
    pub generated_actions: usize,
    pub maximum_search_depth: usize,
    pub maximum_search_trees: usize,
    pub maximum_search_workers: usize,
    pub maximum_search_worker_capacity: usize,
    pub maximum_action_ranking_workers: usize,
    pub maximum_action_ranking_worker_capacity: usize,
    pub rollout_steps: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentError {
    message: String,
}

impl AgentError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for AgentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AgentError {}

pub trait Agent {
    /// Returns an index into the non-empty legal-action slice.
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError>;

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry::default()
    }

    /// Clears counters reported by `telemetry` without changing playing state.
    ///
    /// This method is required so a new agent cannot silently opt out of the
    /// per-match telemetry contract.
    fn reset_telemetry(&mut self);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomAgent {
    rng: StableRng,
    decisions: usize,
}

impl RandomAgent {
    pub const fn new(seed: u64) -> Self {
        Self {
            rng: StableRng::new(seed),
            decisions: 0,
        }
    }
}

impl Agent for RandomAgent {
    fn select_action(
        &mut self,
        _position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        self.decisions += 1;
        Ok(self.rng.index(legal_actions.len()))
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
