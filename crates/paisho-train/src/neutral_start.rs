use core::fmt;
use std::cmp::Reverse;

use paisho_ai::StableRng;
use paisho_core::{
    legal_actions, ApplyError, GameOutcome, GameRecord, Player, RuleProfileId, StandardSetup,
    TurnPhase,
};

pub const NEUTRAL_START_POLICY_V1: &str = "uniform-random-terminal-horizon-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeutralStartConfigurationV1 {
    base_seed: u64,
    target_remaining_decisions: usize,
    source_decision_soft_limit: usize,
    maximum_source_attempts: usize,
}

impl NeutralStartConfigurationV1 {
    pub fn new(
        base_seed: u64,
        target_remaining_decisions: usize,
        source_decision_soft_limit: usize,
        maximum_source_attempts: usize,
    ) -> Result<Self, NeutralStartError> {
        if target_remaining_decisions == 0 {
            return Err(NeutralStartError::ZeroTargetHorizon);
        }
        if source_decision_soft_limit == 0 {
            return Err(NeutralStartError::ZeroSourceDecisionLimit);
        }
        if maximum_source_attempts == 0 {
            return Err(NeutralStartError::ZeroSourceAttempts);
        }
        Ok(Self {
            base_seed,
            target_remaining_decisions,
            source_decision_soft_limit,
            maximum_source_attempts,
        })
    }

    pub const fn base_seed(self) -> u64 {
        self.base_seed
    }

    pub const fn target_remaining_decisions(self) -> usize {
        self.target_remaining_decisions
    }

    pub const fn source_decision_soft_limit(self) -> usize {
        self.source_decision_soft_limit
    }

    pub const fn maximum_source_attempts(self) -> usize {
        self.maximum_source_attempts
    }

    pub fn generate(
        self,
        setup: StandardSetup,
        ordinal: u64,
    ) -> Result<NeutralStartV1, NeutralStartError> {
        for attempt in 0..self.maximum_source_attempts {
            let attempt = u64::try_from(attempt).map_err(|_| NeutralStartError::CounterOverflow)?;
            let source_seed = source_seed(self.base_seed, ordinal, attempt);
            if let Some(start) = generate_one_source(self, setup, source_seed, attempt)? {
                return Ok(start);
            }
        }
        Err(NeutralStartError::AttemptsExhausted {
            ordinal,
            attempts: self.maximum_source_attempts,
            decision_limit: self.source_decision_soft_limit,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NeutralStartProvenanceV1 {
    source_seed: u64,
    source_attempt: u64,
    source_decisions: usize,
    prefix_decisions: usize,
    source_remaining_decisions: usize,
}

impl NeutralStartProvenanceV1 {
    pub const fn source_seed(self) -> u64 {
        self.source_seed
    }

    pub const fn source_attempt(self) -> u64 {
        self.source_attempt
    }

    pub const fn source_decisions(self) -> usize {
        self.source_decisions
    }

    pub const fn prefix_decisions(self) -> usize {
        self.prefix_decisions
    }

    pub const fn source_remaining_decisions(self) -> usize {
        self.source_remaining_decisions
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NeutralStartV1 {
    prefix: GameRecord,
    provenance: NeutralStartProvenanceV1,
}

impl NeutralStartV1 {
    pub const fn prefix(&self) -> &GameRecord {
        &self.prefix
    }

    pub const fn provenance(&self) -> NeutralStartProvenanceV1 {
        self.provenance
    }
}

fn generate_one_source(
    configuration: NeutralStartConfigurationV1,
    setup: StandardSetup,
    source_seed: u64,
    source_attempt: u64,
) -> Result<Option<NeutralStartV1>, NeutralStartError> {
    let mut rng = StableRng::new(source_seed);
    // These sources are replayed by the archived V1 PPO/promotion pipeline.
    let mut source = GameRecord::with_rules(setup, RuleProfileId::SkudPaiSho2022);
    let mut position = source.initial_position();
    let mut main_boundaries = Vec::new();

    while position.outcome() == GameOutcome::Ongoing {
        if source.actions().len() >= configuration.source_decision_soft_limit
            && position.phase() == TurnPhase::Main
        {
            return Ok(None);
        }
        let actions = legal_actions(&position);
        if actions.is_empty() {
            return Err(NeutralStartError::NoLegalAction {
                source_seed,
                player: position.to_move(),
                decision: source.actions().len(),
            });
        }
        let action = actions[rng.index(actions.len())];
        let decision = source.actions().len();
        position
            .apply(action)
            .map_err(|source| NeutralStartError::GeneratedActionRejected {
                source_seed,
                decision,
                source,
            })?;
        source.push(action);
        if position.outcome() == GameOutcome::Ongoing && position.phase() == TurnPhase::Main {
            main_boundaries.push(source.actions().len());
        }
    }

    let source_decisions = source.actions().len();
    let Some(prefix_decisions) = main_boundaries.into_iter().min_by_key(|&boundary| {
        let remaining = source_decisions - boundary;
        (
            remaining.abs_diff(configuration.target_remaining_decisions),
            Reverse(remaining),
            boundary,
        )
    }) else {
        return Ok(None);
    };
    let mut prefix = GameRecord::with_rules(setup, source.rules());
    for &action in &source.actions()[..prefix_decisions] {
        prefix.push(action);
    }
    let source_remaining_decisions = source_decisions - prefix_decisions;
    Ok(Some(NeutralStartV1 {
        prefix,
        provenance: NeutralStartProvenanceV1 {
            source_seed,
            source_attempt,
            source_decisions,
            prefix_decisions,
            source_remaining_decisions,
        },
    }))
}

fn source_seed(base: u64, ordinal: u64, attempt: u64) -> u64 {
    let mut rng = StableRng::new(
        base ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15)
            ^ attempt.wrapping_mul(0xd1b5_4a32_d192_ed03),
    );
    rng.next_u64()
}

#[derive(Debug)]
pub enum NeutralStartError {
    ZeroTargetHorizon,
    ZeroSourceDecisionLimit,
    ZeroSourceAttempts,
    CounterOverflow,
    AttemptsExhausted {
        ordinal: u64,
        attempts: usize,
        decision_limit: usize,
    },
    NoLegalAction {
        source_seed: u64,
        player: Player,
        decision: usize,
    },
    GeneratedActionRejected {
        source_seed: u64,
        decision: usize,
        source: ApplyError,
    },
}

impl fmt::Display for NeutralStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTargetHorizon => {
                formatter.write_str("neutral-start target horizon must be positive")
            }
            Self::ZeroSourceDecisionLimit => {
                formatter.write_str("neutral-start source decision limit must be positive")
            }
            Self::ZeroSourceAttempts => {
                formatter.write_str("neutral-start source attempts must be positive")
            }
            Self::CounterOverflow => formatter.write_str("neutral-start counter overflow"),
            Self::AttemptsExhausted {
                ordinal,
                attempts,
                decision_limit,
            } => write!(
                formatter,
                "neutral start {ordinal} found no usable terminal random source in {attempts} attempts of {decision_limit} decisions"
            ),
            Self::NoLegalAction {
                source_seed,
                player,
                decision,
            } => write!(
                formatter,
                "neutral random source {source_seed} has no legal action for {player:?} at decision {decision}"
            ),
            Self::GeneratedActionRejected {
                source_seed,
                decision,
                source,
            } => write!(
                formatter,
                "neutral random source {source_seed} generated a rejected action at decision {decision}: {source}"
            ),
        }
    }
}

impl std::error::Error for NeutralStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::GeneratedActionRejected { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use paisho_core::BasicFlower;

    use super::*;

    #[test]
    fn generated_start_is_deterministic_advanced_and_near_the_requested_horizon() {
        let configuration = NeutralStartConfigurationV1::new(71, 32, 4_096, 16).unwrap();
        let setup = StandardSetup::balanced(BasicFlower::Red3);
        let first = configuration.generate(setup, 5).unwrap();
        let second = configuration.generate(setup, 5).unwrap();

        assert_eq!(first, second);
        assert!(!first.prefix().actions().is_empty());
        let position = first.prefix().replay().unwrap();
        assert_eq!(first.prefix().rules(), RuleProfileId::SkudPaiSho2022);
        assert_eq!(position.rule_profile(), RuleProfileId::SkudPaiSho2022);
        assert_eq!(position.outcome(), GameOutcome::Ongoing);
        assert_eq!(position.phase(), TurnPhase::Main);
        let provenance = first.provenance();
        assert_eq!(
            provenance.prefix_decisions(),
            first.prefix().actions().len()
        );
        assert_eq!(
            provenance.source_decisions() - provenance.prefix_decisions(),
            provenance.source_remaining_decisions()
        );
        assert!(provenance.source_remaining_decisions().abs_diff(32) <= 1);
    }

    #[test]
    fn configuration_rejects_zero_controls() {
        assert!(matches!(
            NeutralStartConfigurationV1::new(1, 0, 10, 1),
            Err(NeutralStartError::ZeroTargetHorizon)
        ));
        assert!(matches!(
            NeutralStartConfigurationV1::new(1, 10, 0, 1),
            Err(NeutralStartError::ZeroSourceDecisionLimit)
        ));
        assert!(matches!(
            NeutralStartConfigurationV1::new(1, 10, 10, 0),
            Err(NeutralStartError::ZeroSourceAttempts)
        ));
    }
}
