use core::fmt;
use std::collections::HashSet;
use std::path::Path;

use paisho_replay::{
    ReplayDatasetV1, ReplayDatasetV1Error, ReplayDigestV1, ReplaySnapshotV1, ReplaySnapshotV1Error,
    ReplayTrainingExampleV1, ReplayValidationError,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayClassStatisticsV1 {
    pub games: u64,
    pub examples: u64,
    pub mean_examples_per_game: Option<f64>,
    pub mean_behavior_probability: Option<f64>,
    pub mean_legal_actions: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayRangeStatisticsV1 {
    pub minimum: f64,
    pub mean: f64,
    pub maximum: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BehaviorReplayStatisticsV1 {
    pub snapshot_sha256: ReplayDigestV1,
    pub behavior_producer: ReplayDigestV1,
    pub games: u64,
    pub examples: u64,
    pub wins: ReplayClassStatisticsV1,
    pub draws: ReplayClassStatisticsV1,
    pub losses: ReplayClassStatisticsV1,
    pub behavior_probability: ReplayRangeStatisticsV1,
    pub legal_actions: ReplayRangeStatisticsV1,
}

pub fn analyze_behavior_replay_v1(
    snapshot_path: &Path,
    replay_directory: &Path,
    behavior_producer: ReplayDigestV1,
) -> Result<BehaviorReplayStatisticsV1, ReplayStatisticsError> {
    let snapshot =
        ReplaySnapshotV1::read(snapshot_path).map_err(ReplayStatisticsError::Snapshot)?;
    let dataset =
        ReplayDatasetV1::from_snapshot_for_behavior(&snapshot, replay_directory, behavior_producer)
            .map_err(ReplayStatisticsError::Dataset)?;
    let examples = dataset
        .materialize_all()
        .map_err(ReplayStatisticsError::Validation)?;
    summarize_examples(snapshot.digest(), behavior_producer, &examples)
}

fn summarize_examples(
    snapshot_sha256: ReplayDigestV1,
    behavior_producer: ReplayDigestV1,
    examples: &[ReplayTrainingExampleV1],
) -> Result<BehaviorReplayStatisticsV1, ReplayStatisticsError> {
    let mut accumulator = StatisticsAccumulator::default();
    for example in examples {
        accumulator.record(example)?;
    }
    accumulator.finish(snapshot_sha256, behavior_producer)
}

#[derive(Default)]
struct StatisticsAccumulator {
    game_ids: HashSet<u64>,
    classes: [ClassAccumulator; 3],
    examples: u64,
    probability_sum: f64,
    probability_minimum: Option<f32>,
    probability_maximum: Option<f32>,
    legal_action_sum: u64,
    legal_action_minimum: Option<u64>,
    legal_action_maximum: Option<u64>,
}

impl StatisticsAccumulator {
    fn record(&mut self, example: &ReplayTrainingExampleV1) -> Result<(), ReplayStatisticsError> {
        let behavior_probability = example.played_behavior_probability().ok_or(
            ReplayStatisticsError::NotBehaviorPolicy {
                game_id: example.game_id(),
                decision_index: example.decision_index(),
            },
        )?;
        let legal_actions = u64::try_from(example.inference().legal_actions().len())
            .map_err(|_| ReplayStatisticsError::CountOverflow("legal actions"))?;

        self.game_ids.insert(example.game_id());
        self.examples = checked_increment(self.examples, "examples")?;
        self.probability_sum += f64::from(behavior_probability);
        self.probability_minimum = Some(
            self.probability_minimum
                .map_or(behavior_probability, |value| {
                    value.min(behavior_probability)
                }),
        );
        self.probability_maximum = Some(
            self.probability_maximum
                .map_or(behavior_probability, |value| {
                    value.max(behavior_probability)
                }),
        );
        self.legal_action_sum = self
            .legal_action_sum
            .checked_add(legal_actions)
            .ok_or(ReplayStatisticsError::CountOverflow("legal action sum"))?;
        self.legal_action_minimum = Some(
            self.legal_action_minimum
                .map_or(legal_actions, |value| value.min(legal_actions)),
        );
        self.legal_action_maximum = Some(
            self.legal_action_maximum
                .map_or(legal_actions, |value| value.max(legal_actions)),
        );
        self.classes[example.value_class().index()].record(
            example.game_id(),
            behavior_probability,
            legal_actions,
        )
    }

    fn finish(
        self,
        snapshot_sha256: ReplayDigestV1,
        behavior_producer: ReplayDigestV1,
    ) -> Result<BehaviorReplayStatisticsV1, ReplayStatisticsError> {
        let games = u64::try_from(self.game_ids.len())
            .map_err(|_| ReplayStatisticsError::CountOverflow("games"))?;
        let probability_minimum = self
            .probability_minimum
            .ok_or(ReplayStatisticsError::Empty)?;
        let probability_maximum = self
            .probability_maximum
            .ok_or(ReplayStatisticsError::Empty)?;
        let legal_action_minimum = self
            .legal_action_minimum
            .ok_or(ReplayStatisticsError::Empty)?;
        let legal_action_maximum = self
            .legal_action_maximum
            .ok_or(ReplayStatisticsError::Empty)?;
        let denominator = self.examples as f64;
        let [wins, draws, losses] = self.classes;

        Ok(BehaviorReplayStatisticsV1 {
            snapshot_sha256,
            behavior_producer,
            games,
            examples: self.examples,
            wins: wins.finish()?,
            draws: draws.finish()?,
            losses: losses.finish()?,
            behavior_probability: ReplayRangeStatisticsV1 {
                minimum: f64::from(probability_minimum),
                mean: self.probability_sum / denominator,
                maximum: f64::from(probability_maximum),
            },
            legal_actions: ReplayRangeStatisticsV1 {
                minimum: legal_action_minimum as f64,
                mean: self.legal_action_sum as f64 / denominator,
                maximum: legal_action_maximum as f64,
            },
        })
    }
}

#[derive(Default)]
struct ClassAccumulator {
    game_ids: HashSet<u64>,
    examples: u64,
    probability_sum: f64,
    legal_action_sum: u64,
}

impl ClassAccumulator {
    fn record(
        &mut self,
        game_id: u64,
        behavior_probability: f32,
        legal_actions: u64,
    ) -> Result<(), ReplayStatisticsError> {
        self.game_ids.insert(game_id);
        self.examples = checked_increment(self.examples, "class examples")?;
        self.probability_sum += f64::from(behavior_probability);
        self.legal_action_sum = self.legal_action_sum.checked_add(legal_actions).ok_or(
            ReplayStatisticsError::CountOverflow("class legal action sum"),
        )?;
        Ok(())
    }

    fn finish(self) -> Result<ReplayClassStatisticsV1, ReplayStatisticsError> {
        let games = u64::try_from(self.game_ids.len())
            .map_err(|_| ReplayStatisticsError::CountOverflow("class games"))?;
        let example_denominator = (self.examples > 0).then_some(self.examples as f64);
        Ok(ReplayClassStatisticsV1 {
            games,
            examples: self.examples,
            mean_examples_per_game: (games > 0).then_some(self.examples as f64 / games as f64),
            mean_behavior_probability: example_denominator
                .map(|denominator| self.probability_sum / denominator),
            mean_legal_actions: example_denominator
                .map(|denominator| self.legal_action_sum as f64 / denominator),
        })
    }
}

fn checked_increment(value: u64, field: &'static str) -> Result<u64, ReplayStatisticsError> {
    value
        .checked_add(1)
        .ok_or(ReplayStatisticsError::CountOverflow(field))
}

#[derive(Debug)]
pub enum ReplayStatisticsError {
    Empty,
    CountOverflow(&'static str),
    NotBehaviorPolicy { game_id: u64, decision_index: usize },
    Snapshot(ReplaySnapshotV1Error),
    Dataset(ReplayDatasetV1Error),
    Validation(ReplayValidationError),
}

impl fmt::Display for ReplayStatisticsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("behavior replay statistics require examples"),
            Self::CountOverflow(field) => write!(formatter, "{field} count overflow"),
            Self::NotBehaviorPolicy {
                game_id,
                decision_index,
            } => write!(
                formatter,
                "game {game_id} decision {decision_index} is not a behavior-policy example"
            ),
            Self::Snapshot(source) => source.fmt(formatter),
            Self::Dataset(source) => source.fmt(formatter),
            Self::Validation(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for ReplayStatisticsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Snapshot(source) => Some(source),
            Self::Dataset(source) => Some(source),
            Self::Validation(source) => Some(source),
            Self::Empty | Self::CountOverflow(_) | Self::NotBehaviorPolicy { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_statistics_distinguish_games_from_decisions() {
        let mut accumulator = ClassAccumulator::default();
        accumulator.record(7, 0.25, 8).unwrap();
        accumulator.record(7, 0.50, 12).unwrap();
        accumulator.record(9, 0.75, 10).unwrap();

        assert_eq!(
            accumulator.finish().unwrap(),
            ReplayClassStatisticsV1 {
                games: 2,
                examples: 3,
                mean_examples_per_game: Some(1.5),
                mean_behavior_probability: Some(0.5),
                mean_legal_actions: Some(10.0),
            }
        );
    }
}
