use core::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{verify_checkpoint_file, MpsGraphClientError, NetworkPreset, OptimizationLevel};

const CHECKPOINT_MAGIC_V2: &[u8] = b"PAISHO-CKPT-V2\n";
const MAXIMUM_METADATA_BYTES: u32 = 16 * 1_024 * 1_024;
const TENSOR_SCHEMA_V1: &str = "paisho-neural-encoding-v1";
const RULE_PROFILE_V1: &str = "skud-pai-sho-2022-03-14";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckpointMetadataV2 {
    content_sha256: [u8; 32],
    network_preset: NetworkPreset,
    optimization: OptimizationLevel,
    training_step: u64,
    generation: u64,
    replay_index: u64,
    replay_snapshot_sha256: [u8; 32],
    learning_rate: f32,
}

impl CheckpointMetadataV2 {
    pub const fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }

    pub const fn network_preset(&self) -> NetworkPreset {
        self.network_preset
    }

    pub const fn optimization(&self) -> OptimizationLevel {
        self.optimization
    }

    pub const fn training_step(&self) -> u64 {
        self.training_step
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn replay_index(&self) -> u64 {
        self.replay_index
    }

    pub const fn replay_snapshot_sha256(&self) -> [u8; 32] {
        self.replay_snapshot_sha256
    }

    pub const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }
}

pub fn read_checkpoint_metadata(
    path: &Path,
) -> Result<CheckpointMetadataV2, CheckpointMetadataError> {
    let content_sha256 = verify_checkpoint_file(path).map_err(CheckpointMetadataError::Verify)?;
    let mut file = File::open(path).map_err(CheckpointMetadataError::Io)?;
    let mut magic = [0_u8; CHECKPOINT_MAGIC_V2.len()];
    file.read_exact(&mut magic)
        .map_err(CheckpointMetadataError::Io)?;
    if magic != CHECKPOINT_MAGIC_V2 {
        return Err(CheckpointMetadataError::InvalidMagic(path.to_owned()));
    }
    let mut length = [0_u8; 4];
    file.read_exact(&mut length)
        .map_err(CheckpointMetadataError::Io)?;
    let length = u32::from_le_bytes(length);
    if length > MAXIMUM_METADATA_BYTES {
        return Err(CheckpointMetadataError::MetadataTooLarge(length));
    }
    let mut bytes = vec![0_u8; length as usize];
    file.read_exact(&mut bytes)
        .map_err(CheckpointMetadataError::Io)?;
    let stored: StoredMetadata =
        serde_json::from_slice(&bytes).map_err(CheckpointMetadataError::Json)?;
    stored.validate(content_sha256)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredMetadata {
    configuration: StoredConfiguration,
    format_version: u32,
    optimization: String,
    progress: StoredProgress,
    rule_profile: String,
    tensor_schema: String,
    training_step: u64,
}

impl StoredMetadata {
    fn validate(
        self,
        content_sha256: [u8; 32],
    ) -> Result<CheckpointMetadataV2, CheckpointMetadataError> {
        if self.format_version != 2 {
            return Err(CheckpointMetadataError::UnsupportedFormatVersion(
                self.format_version,
            ));
        }
        if self.tensor_schema != TENSOR_SCHEMA_V1 {
            return Err(CheckpointMetadataError::UnsupportedTensorSchema(
                self.tensor_schema,
            ));
        }
        if self.rule_profile != RULE_PROFILE_V1 {
            return Err(CheckpointMetadataError::UnsupportedRuleProfile(
                self.rule_profile,
            ));
        }
        if self.progress.scheduler.completed_steps != self.training_step {
            return Err(CheckpointMetadataError::SchedulerStepMismatch {
                training_step: self.training_step,
                scheduler_step: self.progress.scheduler.completed_steps,
            });
        }
        if !self.progress.scheduler.learning_rate.is_finite()
            || self.progress.scheduler.learning_rate <= 0.0
        {
            return Err(CheckpointMetadataError::InvalidLearningRate(
                self.progress.scheduler.learning_rate,
            ));
        }
        let network_preset = self.configuration.preset()?;
        let optimization = match self.optimization.as_str() {
            "level0" => OptimizationLevel::Level0,
            "level1" => OptimizationLevel::Level1,
            _ => {
                return Err(CheckpointMetadataError::UnsupportedOptimization(
                    self.optimization,
                ))
            }
        };
        let replay_snapshot_sha256 = decode_digest(&self.progress.replay_snapshot_sha256)?;
        Ok(CheckpointMetadataV2 {
            content_sha256,
            network_preset,
            optimization,
            training_step: self.training_step,
            generation: self.progress.generation,
            replay_index: self.progress.replay_index,
            replay_snapshot_sha256,
            learning_rate: self.progress.scheduler.learning_rate,
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredConfiguration {
    trunk_channels: usize,
    residual_blocks: usize,
    policy_embedding_channels: usize,
    value_hidden_channels: usize,
    normalization_epsilon: f32,
}

impl StoredConfiguration {
    fn preset(&self) -> Result<NetworkPreset, CheckpointMetadataError> {
        let common_epsilon = self.normalization_epsilon.to_bits() == 1.0e-5_f32.to_bits();
        match (
            self.trunk_channels,
            self.residual_blocks,
            self.policy_embedding_channels,
            self.value_hidden_channels,
            common_epsilon,
        ) {
            (20, 3, 8, 32, true) => Ok(NetworkPreset::Micro),
            (160, 10, 32, 128, true) => Ok(NetworkPreset::Pure),
            _ => Err(CheckpointMetadataError::UnsupportedNetworkConfiguration),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredProgress {
    generation: u64,
    replay_index: u64,
    #[serde(rename = "replaySnapshotSHA256")]
    replay_snapshot_sha256: String,
    scheduler: StoredScheduler,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredScheduler {
    completed_steps: u64,
    learning_rate: f32,
}

fn decode_digest(text: &str) -> Result<[u8; 32], CheckpointMetadataError> {
    if text.len() != 64 {
        return Err(CheckpointMetadataError::InvalidReplaySnapshotDigest);
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in text.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_nibble(pair[0])?;
        let low = decode_nibble(pair[1])?;
        digest[index] = high * 16 + low;
    }
    Ok(digest)
}

fn decode_nibble(byte: u8) -> Result<u8, CheckpointMetadataError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(CheckpointMetadataError::InvalidReplaySnapshotDigest),
    }
}

#[derive(Debug)]
pub enum CheckpointMetadataError {
    InvalidMagic(PathBuf),
    MetadataTooLarge(u32),
    UnsupportedFormatVersion(u32),
    UnsupportedTensorSchema(String),
    UnsupportedRuleProfile(String),
    UnsupportedOptimization(String),
    UnsupportedNetworkConfiguration,
    SchedulerStepMismatch {
        training_step: u64,
        scheduler_step: u64,
    },
    InvalidLearningRate(f32),
    InvalidReplaySnapshotDigest,
    Verify(MpsGraphClientError),
    Json(serde_json::Error),
    Io(io::Error),
}

impl fmt::Display for CheckpointMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic(path) => {
                write!(formatter, "checkpoint {} has an invalid V2 magic", path.display())
            }
            Self::MetadataTooLarge(length) => {
                write!(formatter, "checkpoint metadata has {length} bytes")
            }
            Self::UnsupportedFormatVersion(version) => {
                write!(formatter, "unsupported checkpoint format version {version}")
            }
            Self::UnsupportedTensorSchema(schema) => {
                write!(formatter, "unsupported checkpoint tensor schema {schema}")
            }
            Self::UnsupportedRuleProfile(profile) => {
                write!(formatter, "unsupported checkpoint rule profile {profile}")
            }
            Self::UnsupportedOptimization(level) => {
                write!(formatter, "unsupported checkpoint optimization {level}")
            }
            Self::UnsupportedNetworkConfiguration => {
                formatter.write_str("checkpoint network is not a known Paisho preset")
            }
            Self::SchedulerStepMismatch {
                training_step,
                scheduler_step,
            } => write!(
                formatter,
                "checkpoint training step {training_step} differs from scheduler step {scheduler_step}"
            ),
            Self::InvalidLearningRate(rate) => {
                write!(formatter, "checkpoint learning rate {rate} is invalid")
            }
            Self::InvalidReplaySnapshotDigest => {
                formatter.write_str("checkpoint replay snapshot digest is not hexadecimal SHA-256")
            }
            Self::Verify(source) => source.fmt(formatter),
            Self::Json(source) => write!(formatter, "checkpoint metadata JSON is invalid: {source}"),
            Self::Io(source) => write!(formatter, "cannot read checkpoint metadata: {source}"),
        }
    }
}

impl std::error::Error for CheckpointMetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Verify(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use sha2::{Digest, Sha256};

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn reads_verified_v2_metadata_without_loading_parameter_tensors() {
        let checkpoint = TestCheckpoint::write(&metadata_json(41, 41, "AB", "level1"));
        let metadata = read_checkpoint_metadata(&checkpoint.path).unwrap();
        assert_eq!(metadata.content_sha256(), checkpoint.digest);
        assert_eq!(metadata.network_preset(), NetworkPreset::Pure);
        assert_eq!(metadata.optimization(), OptimizationLevel::Level1);
        assert_eq!(metadata.training_step(), 41);
        assert_eq!(metadata.generation(), 7);
        assert_eq!(metadata.replay_index(), 1_024);
        assert_eq!(metadata.replay_snapshot_sha256(), [0xAB; 32]);
        assert_eq!(metadata.learning_rate(), 2.5e-5);
    }

    #[test]
    fn rejects_scheduler_progress_that_disagrees_with_global_adam_step() {
        let checkpoint = TestCheckpoint::write(&metadata_json(41, 40, "ab", "level1"));
        assert!(matches!(
            read_checkpoint_metadata(&checkpoint.path),
            Err(CheckpointMetadataError::SchedulerStepMismatch {
                training_step: 41,
                scheduler_step: 40
            })
        ));
    }

    #[test]
    fn rejects_a_checkpoint_whose_content_checksum_was_changed() {
        let checkpoint = TestCheckpoint::write(&metadata_json(41, 41, "ab", "level1"));
        let mut bytes = fs::read(&checkpoint.path).unwrap();
        bytes[0] ^= 1;
        fs::write(&checkpoint.path, bytes).unwrap();
        assert!(matches!(
            read_checkpoint_metadata(&checkpoint.path),
            Err(CheckpointMetadataError::Verify(
                MpsGraphClientError::CheckpointChecksumMismatch(_)
            ))
        ));
    }

    fn metadata_json(training_step: u64, scheduler_step: u64, byte: &str, level: &str) -> String {
        format!(
            concat!(
                "{{\"configuration\":{{\"normalizationEpsilon\":0.00001,",
                "\"policyEmbeddingChannels\":32,\"residualBlocks\":10,",
                "\"trunkChannels\":160,\"valueHiddenChannels\":128}},",
                "\"executionShape\":{{\"batchSize\":64,\"legalActionCapacity\":1024}},",
                "\"formatVersion\":2,\"optimization\":\"{}\",",
                "\"progress\":{{\"generation\":7,\"randomStates\":[],",
                "\"replayIndex\":1024,\"replaySnapshotSHA256\":\"{}\",",
                "\"scheduler\":{{\"completedSteps\":{},\"learningRate\":0.000025}}}},",
                "\"ruleProfile\":\"skud-pai-sho-2022-03-14\",",
                "\"tensorSchema\":\"paisho-neural-encoding-v1\",\"trainingStep\":{}}}"
            ),
            level,
            byte.repeat(32),
            scheduler_step,
            training_step
        )
    }

    struct TestCheckpoint {
        path: PathBuf,
        digest: [u8; 32],
    }

    impl TestCheckpoint {
        fn write(metadata: &str) -> Self {
            let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "paisho-checkpoint-metadata-test-{}-{sequence}.psckpt",
                std::process::id()
            ));
            let mut bytes = CHECKPOINT_MAGIC_V2.to_vec();
            bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
            bytes.extend_from_slice(metadata.as_bytes());
            bytes.extend_from_slice(b"parameter payload is intentionally not decoded");
            let digest: [u8; 32] = Sha256::digest(&bytes).into();
            bytes.extend_from_slice(&digest);
            fs::write(&path, bytes).unwrap();
            Self { path, digest }
        }
    }

    impl Drop for TestCheckpoint {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.path);
        }
    }
}
