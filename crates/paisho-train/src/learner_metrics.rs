use core::fmt;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use paisho_replay::{ReplayDigestV1, ReplayDigestV1Error};
use serde::{Deserialize, Serialize};

use crate::atomic_file;

const DOCUMENT_V1: &str = "paisho-terminal-ppo-metrics-v1";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalPpoBatchMetricsV1 {
    pub training_step: u64,
    pub replay_index: u64,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub total_loss: f32,
    pub mean_advantage: f32,
    pub mean_importance_ratio: f32,
    pub mean_squared_ratio_deviation: f32,
}

impl TerminalPpoBatchMetricsV1 {
    pub fn validate(self) -> Result<Self, LearnerMetricsError> {
        if self.training_step == 0 {
            return Err(LearnerMetricsError::ZeroTrainingStep);
        }
        for (name, value) in self.metric_values() {
            if !value.is_finite() {
                return Err(LearnerMetricsError::NonFiniteMetric { name, value });
            }
        }
        if self.mean_squared_ratio_deviation < 0.0 {
            return Err(LearnerMetricsError::NegativeSquaredRatioDeviation(
                self.mean_squared_ratio_deviation,
            ));
        }
        Ok(self)
    }

    fn metric_values(self) -> [(&'static str, f32); 7] {
        [
            ("policy loss", self.policy_loss),
            ("value loss", self.value_loss),
            ("entropy", self.entropy),
            ("total loss", self.total_loss),
            ("mean advantage", self.mean_advantage),
            ("mean importance ratio", self.mean_importance_ratio),
            (
                "mean squared ratio deviation",
                self.mean_squared_ratio_deviation,
            ),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScalarMetricSummaryV1 {
    pub minimum: f64,
    pub mean: f64,
    pub maximum: f64,
    pub last: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalPpoMetricsSummaryV1 {
    pub batches: u64,
    pub first_training_step: u64,
    pub last_training_step: u64,
    pub policy_loss: ScalarMetricSummaryV1,
    pub value_loss: ScalarMetricSummaryV1,
    pub entropy: ScalarMetricSummaryV1,
    pub total_loss: ScalarMetricSummaryV1,
    pub mean_advantage: ScalarMetricSummaryV1,
    pub mean_importance_ratio: ScalarMetricSummaryV1,
    pub mean_squared_ratio_deviation: ScalarMetricSummaryV1,
}

impl TerminalPpoMetricsSummaryV1 {
    pub fn from_batches(
        batches: &[TerminalPpoBatchMetricsV1],
    ) -> Result<Option<Self>, LearnerMetricsError> {
        if batches.is_empty() {
            return Ok(None);
        }
        validate_batches(batches)?;
        let batches_count =
            u64::try_from(batches.len()).map_err(|_| LearnerMetricsError::BatchCountOverflow)?;
        Ok(Some(Self {
            batches: batches_count,
            first_training_step: batches[0].training_step,
            last_training_step: batches[batches.len() - 1].training_step,
            policy_loss: summarize(batches, |batch| batch.policy_loss),
            value_loss: summarize(batches, |batch| batch.value_loss),
            entropy: summarize(batches, |batch| batch.entropy),
            total_loss: summarize(batches, |batch| batch.total_loss),
            mean_advantage: summarize(batches, |batch| batch.mean_advantage),
            mean_importance_ratio: summarize(batches, |batch| batch.mean_importance_ratio),
            mean_squared_ratio_deviation: summarize(batches, |batch| {
                batch.mean_squared_ratio_deviation
            }),
        }))
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalPpoMetricSegmentV1 {
    document: String,
    generation: u64,
    checkpoint_file: String,
    checkpoint_sha256: String,
    through_training_step: u64,
    batches: Vec<TerminalPpoBatchMetricsV1>,
}

impl TerminalPpoMetricSegmentV1 {
    fn new(
        generation: u64,
        checkpoint_file: String,
        checkpoint_sha256: [u8; 32],
        through_training_step: u64,
        batches: Vec<TerminalPpoBatchMetricsV1>,
    ) -> Result<Self, LearnerMetricsError> {
        let segment = Self {
            document: DOCUMENT_V1.to_owned(),
            generation,
            checkpoint_file,
            checkpoint_sha256: ReplayDigestV1::from_bytes(checkpoint_sha256).to_string(),
            through_training_step,
            batches,
        };
        segment.validate()?;
        Ok(segment)
    }

    fn validate(&self) -> Result<(), LearnerMetricsError> {
        if self.document != DOCUMENT_V1 {
            return Err(LearnerMetricsError::WrongDocument(self.document.clone()));
        }
        validate_file_name(&self.checkpoint_file)?;
        self.checkpoint_sha256
            .parse::<ReplayDigestV1>()
            .map_err(LearnerMetricsError::Digest)?;
        if self.batches.is_empty() {
            return Err(LearnerMetricsError::EmptySegment);
        }
        validate_batches(&self.batches)?;
        let actual = self.batches[self.batches.len() - 1].training_step;
        if actual != self.through_training_step {
            return Err(LearnerMetricsError::ThroughStepMismatch {
                expected: self.through_training_step,
                actual,
            });
        }
        Ok(())
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn checkpoint_file(&self) -> &str {
        &self.checkpoint_file
    }

    pub fn checkpoint_sha256(&self) -> Result<ReplayDigestV1, ReplayDigestV1Error> {
        self.checkpoint_sha256.parse()
    }

    pub const fn through_training_step(&self) -> u64 {
        self.through_training_step
    }

    pub fn batches(&self) -> &[TerminalPpoBatchMetricsV1] {
        &self.batches
    }

    pub fn summary(&self) -> Result<TerminalPpoMetricsSummaryV1, LearnerMetricsError> {
        TerminalPpoMetricsSummaryV1::from_batches(&self.batches)?
            .ok_or(LearnerMetricsError::EmptySegment)
    }
}

pub(crate) fn write_terminal_ppo_metric_segment_v1(
    checkpoint_path: &Path,
    checkpoint_sha256: [u8; 32],
    generation: u64,
    through_training_step: u64,
    batches: &[TerminalPpoBatchMetricsV1],
) -> Result<Option<PathBuf>, LearnerMetricsError> {
    if batches.is_empty() {
        return Ok(None);
    }
    let checkpoint_file = checkpoint_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| LearnerMetricsError::InvalidCheckpointPath(checkpoint_path.to_owned()))?;
    let segment = TerminalPpoMetricSegmentV1::new(
        generation,
        checkpoint_file.to_owned(),
        checkpoint_sha256,
        through_training_step,
        batches.to_vec(),
    )?;
    let mut bytes = serde_json::to_vec_pretty(&segment).map_err(LearnerMetricsError::Json)?;
    bytes.push(b'\n');
    let path = checkpoint_path.with_file_name(format!("{checkpoint_file}.ppo-metrics.json"));
    atomic_file::write_idempotently(&path, &bytes).map_err(LearnerMetricsError::Io)?;
    Ok(Some(path))
}

pub fn read_terminal_ppo_metric_segment_v1(
    path: &Path,
) -> Result<TerminalPpoMetricSegmentV1, LearnerMetricsError> {
    let bytes = fs::read(path).map_err(LearnerMetricsError::Io)?;
    let segment = serde_json::from_slice::<TerminalPpoMetricSegmentV1>(&bytes)
        .map_err(LearnerMetricsError::Json)?;
    segment.validate()?;
    Ok(segment)
}

fn validate_batches(batches: &[TerminalPpoBatchMetricsV1]) -> Result<(), LearnerMetricsError> {
    for &batch in batches {
        batch.validate()?;
    }
    for pair in batches.windows(2) {
        let expected = pair[0]
            .training_step
            .checked_add(1)
            .ok_or(LearnerMetricsError::TrainingStepOverflow)?;
        if pair[1].training_step != expected {
            return Err(LearnerMetricsError::NonConsecutiveTrainingSteps {
                previous: pair[0].training_step,
                next: pair[1].training_step,
            });
        }
        if pair[1].replay_index <= pair[0].replay_index {
            return Err(LearnerMetricsError::NonIncreasingReplayIndex {
                previous: pair[0].replay_index,
                next: pair[1].replay_index,
            });
        }
    }
    Ok(())
}

fn summarize(
    batches: &[TerminalPpoBatchMetricsV1],
    value: impl Fn(TerminalPpoBatchMetricsV1) -> f32,
) -> ScalarMetricSummaryV1 {
    let first = f64::from(value(batches[0]));
    let mut minimum = first;
    let mut maximum = first;
    let mut sum = 0.0;
    for &batch in batches {
        let value = f64::from(value(batch));
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        sum += value;
    }
    ScalarMetricSummaryV1 {
        minimum,
        mean: sum / batches.len() as f64,
        maximum,
        last: f64::from(value(batches[batches.len() - 1])),
    }
}

fn validate_file_name(file_name: &str) -> Result<(), LearnerMetricsError> {
    let mut components = Path::new(file_name).components();
    let valid = matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
        && !file_name.is_empty()
        && !file_name.chars().any(char::is_control);
    if valid {
        Ok(())
    } else {
        Err(LearnerMetricsError::InvalidCheckpointFileName(
            file_name.to_owned(),
        ))
    }
}

#[derive(Debug)]
pub enum LearnerMetricsError {
    ZeroTrainingStep,
    BatchCountOverflow,
    TrainingStepOverflow,
    EmptySegment,
    NonFiniteMetric { name: &'static str, value: f32 },
    NegativeSquaredRatioDeviation(f32),
    NonConsecutiveTrainingSteps { previous: u64, next: u64 },
    NonIncreasingReplayIndex { previous: u64, next: u64 },
    ThroughStepMismatch { expected: u64, actual: u64 },
    WrongDocument(String),
    InvalidCheckpointPath(PathBuf),
    InvalidCheckpointFileName(String),
    Digest(ReplayDigestV1Error),
    Json(serde_json::Error),
    Io(io::Error),
}

impl fmt::Display for LearnerMetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTrainingStep => formatter.write_str("PPO metric training step is zero"),
            Self::BatchCountOverflow => formatter.write_str("PPO metric batch count overflow"),
            Self::TrainingStepOverflow => formatter.write_str("PPO metric training step overflow"),
            Self::EmptySegment => formatter.write_str("PPO metric segment is empty"),
            Self::NonFiniteMetric { name, value } => {
                write!(formatter, "PPO metric {name} is not finite: {value}")
            }
            Self::NegativeSquaredRatioDeviation(value) => write!(
                formatter,
                "PPO mean squared importance-ratio deviation is negative: {value}"
            ),
            Self::NonConsecutiveTrainingSteps { previous, next } => write!(
                formatter,
                "PPO metric training steps are not consecutive: {previous} then {next}"
            ),
            Self::NonIncreasingReplayIndex { previous, next } => write!(
                formatter,
                "PPO metric replay index does not increase: {previous} then {next}"
            ),
            Self::ThroughStepMismatch { expected, actual } => write!(
                formatter,
                "PPO metric segment ends at step {actual}; expected {expected}"
            ),
            Self::WrongDocument(document) => {
                write!(formatter, "unknown PPO metric document {document}")
            }
            Self::InvalidCheckpointPath(path) => {
                write!(
                    formatter,
                    "invalid PPO metric checkpoint path {}",
                    path.display()
                )
            }
            Self::InvalidCheckpointFileName(name) => {
                write!(formatter, "invalid PPO metric checkpoint file name {name}")
            }
            Self::Digest(source) => source.fmt(formatter),
            Self::Json(source) => write!(formatter, "PPO metric JSON failed: {source}"),
            Self::Io(source) => write!(formatter, "PPO metric I/O failed: {source}"),
        }
    }
}

impl std::error::Error for LearnerMetricsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Digest(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_bound_segment_round_trips_and_summarizes() {
        let directory = std::env::temp_dir().join(format!(
            "paisho-ppo-metrics-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        if directory.exists() {
            fs::remove_dir_all(&directory).unwrap();
        }
        fs::create_dir(&directory).unwrap();
        let checkpoint = directory.join("checkpoint.psckpt");
        let batches = [batch(7, 64, 1.0), batch(8, 128, 3.0)];

        let path = write_terminal_ppo_metric_segment_v1(&checkpoint, [0x42; 32], 2, 8, &batches)
            .unwrap()
            .unwrap();
        let restored = read_terminal_ppo_metric_segment_v1(&path).unwrap();
        assert_eq!(restored.generation(), 2);
        assert_eq!(restored.checkpoint_file(), "checkpoint.psckpt");
        assert_eq!(
            restored.checkpoint_sha256().unwrap().as_bytes(),
            &[0x42; 32]
        );
        assert_eq!(restored.batches(), batches);
        let summary = restored.summary().unwrap();
        assert_eq!(summary.batches, 2);
        assert_eq!(summary.policy_loss.mean, 2.0);
        assert_eq!(summary.policy_loss.last, 3.0);

        write_terminal_ppo_metric_segment_v1(&checkpoint, [0x42; 32], 2, 8, &batches).unwrap();
        fs::remove_dir_all(directory).unwrap();
    }

    fn batch(training_step: u64, replay_index: u64, value: f32) -> TerminalPpoBatchMetricsV1 {
        TerminalPpoBatchMetricsV1 {
            training_step,
            replay_index,
            policy_loss: value,
            value_loss: value,
            entropy: value,
            total_loss: value,
            mean_advantage: value,
            mean_importance_ratio: value,
            mean_squared_ratio_deviation: value,
        }
    }
}
