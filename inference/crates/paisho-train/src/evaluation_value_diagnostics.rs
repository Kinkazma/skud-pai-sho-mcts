use core::fmt;

use paisho_core::{ApplyError, GameOutcome, GameRecord, Player, ReplayError};
use paisho_model::{
    InferenceExampleEncodingError, InferenceExampleV1, ValueClassV1, ValueTargetError,
};

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateValueExampleV1 {
    inference: InferenceExampleV1,
    target: ValueClassV1,
}

impl CandidateValueExampleV1 {
    pub const fn inference(&self) -> &InferenceExampleV1 {
        &self.inference
    }

    pub const fn target(&self) -> ValueClassV1 {
        self.target
    }
}

/// Reconstruct only the candidate's decisions from the measured continuation
/// of a terminal evaluation game. The random neutral prefix remains context,
/// never a labelled example.
pub fn materialize_candidate_value_examples_v1(
    record: &GameRecord,
    candidate: Player,
    continuation_decisions: usize,
) -> Result<Vec<CandidateValueExampleV1>, EvaluationValueMaterializationError> {
    let final_position = record
        .replay()
        .map_err(EvaluationValueMaterializationError::InvalidRecord)?;
    let outcome = final_position.outcome();
    if outcome == GameOutcome::Ongoing {
        return Err(EvaluationValueMaterializationError::NonTerminal);
    }
    let first_continuation = record
        .actions()
        .len()
        .checked_sub(continuation_decisions)
        .ok_or(EvaluationValueMaterializationError::ContinuationTooLong {
            continuation_decisions,
            record_decisions: record.actions().len(),
        })?;
    let target = ValueClassV1::from_terminal_outcome(outcome, candidate)
        .map_err(EvaluationValueMaterializationError::ValueTarget)?;
    let mut position = record.initial_position();
    let mut examples = Vec::new();
    for (decision_index, action) in record.actions().iter().copied().enumerate() {
        if decision_index >= first_continuation && position.to_move() == candidate {
            let inference = InferenceExampleV1::from_position(&position).map_err(|source| {
                EvaluationValueMaterializationError::Encoding {
                    decision_index,
                    source,
                }
            })?;
            examples.push(CandidateValueExampleV1 { inference, target });
        }
        position
            .apply(action)
            .map_err(|source| EvaluationValueMaterializationError::Apply {
                decision_index,
                source,
            })?;
    }
    Ok(examples)
}

#[derive(Debug)]
pub enum EvaluationValueMaterializationError {
    InvalidRecord(ReplayError),
    NonTerminal,
    ContinuationTooLong {
        continuation_decisions: usize,
        record_decisions: usize,
    },
    ValueTarget(ValueTargetError),
    Encoding {
        decision_index: usize,
        source: InferenceExampleEncodingError,
    },
    Apply {
        decision_index: usize,
        source: ApplyError,
    },
}

impl fmt::Display for EvaluationValueMaterializationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRecord(source) => write!(formatter, "invalid evaluation record: {source}"),
            Self::NonTerminal => {
                formatter.write_str("evaluation value diagnostics require a terminal record")
            }
            Self::ContinuationTooLong {
                continuation_decisions,
                record_decisions,
            } => write!(
                formatter,
                "evaluation continuation has {continuation_decisions} decisions, but its record has only {record_decisions}"
            ),
            Self::ValueTarget(source) => source.fmt(formatter),
            Self::Encoding {
                decision_index,
                source,
            } => write!(
                formatter,
                "cannot encode evaluation decision {decision_index}: {source}"
            ),
            Self::Apply {
                decision_index,
                source,
            } => write!(
                formatter,
                "cannot replay evaluation decision {decision_index}: {source}"
            ),
        }
    }
}

impl std::error::Error for EvaluationValueMaterializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRecord(source) => Some(source),
            Self::ValueTarget(source) => Some(source),
            Self::Encoding { source, .. } => Some(source),
            Self::Apply { source, .. } => Some(source),
            Self::NonTerminal | Self::ContinuationTooLong { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TERMINAL_RING: &str =
        include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");

    #[test]
    fn continuation_selects_only_the_requested_players_decisions() {
        let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
        let outcome = record.replay().unwrap().outcome();
        let host =
            materialize_candidate_value_examples_v1(&record, Player::Host, record.actions().len())
                .unwrap();
        let guest =
            materialize_candidate_value_examples_v1(&record, Player::Guest, record.actions().len())
                .unwrap();

        assert_eq!(host.len() + guest.len(), record.actions().len());
        assert!(host.iter().all(|example| {
            example.target() == ValueClassV1::from_terminal_outcome(outcome, Player::Host).unwrap()
        }));
        assert!(guest.iter().all(|example| {
            example.target() == ValueClassV1::from_terminal_outcome(outcome, Player::Guest).unwrap()
        }));
    }

    #[test]
    fn prefix_is_context_and_malformed_ranges_are_rejected() {
        let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
        let host = materialize_candidate_value_examples_v1(&record, Player::Host, 1).unwrap();
        let guest = materialize_candidate_value_examples_v1(&record, Player::Guest, 1).unwrap();
        assert_eq!(host.len() + guest.len(), 1);
        assert!(matches!(
            materialize_candidate_value_examples_v1(
                &record,
                Player::Host,
                record.actions().len() + 1
            ),
            Err(EvaluationValueMaterializationError::ContinuationTooLong { .. })
        ));
        assert!(matches!(
            materialize_candidate_value_examples_v1(
                &GameRecord::with_rules(record.setup(), record.rules()),
                Player::Host,
                0
            ),
            Err(EvaluationValueMaterializationError::NonTerminal)
        ));
    }
}
