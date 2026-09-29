use core::fmt;
use core::str::FromStr;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use paisho_core::RuleProfileId;
use paisho_model::{TerminalPpoParametersV1, TerminalPpoWireError, TERMINAL_PPO_OBJECTIVE_V1};
use paisho_mpsgraph_client::{
    verify_checkpoint_file, MpsGraphClientError, NetworkPreset, OptimizationLevel,
};
use paisho_replay::{ReplayDigestV1, ReplayDigestV1Error};
use sha2::{Digest, Sha256};

use crate::atomic_file;
use crate::{LearnerObjectiveV1, TerminalPpoLearnerObjectiveV1, SUPERVISED_OBJECTIVE_V1};

const MAGIC_V1: &str = "PAISHO-LEARNER-COMMIT\t1";
const MAGIC_V2: &str = "PAISHO-LEARNER-COMMIT\t2";
const MAGIC_V3: &str = "PAISHO-LEARNER-COMMIT\t3";
const COMMIT_SUFFIX: &str = ".pslearn";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommitFormat {
    V1,
    V2,
    V3,
}

impl CommitFormat {
    const fn magic(self) -> &'static str {
        match self {
            Self::V1 => MAGIC_V1,
            Self::V2 => MAGIC_V2,
            Self::V3 => MAGIC_V3,
        }
    }

    const fn version(self) -> u8 {
        match self {
            Self::V1 => 1,
            Self::V2 => 2,
            Self::V3 => 3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnerIdentityV1 {
    pub replay_snapshot: ReplayDigestV1,
    pub network_preset: NetworkPreset,
    pub optimization: OptimizationLevel,
    pub batch_size: usize,
    pub legal_action_capacity: usize,
    pub model_seed: u64,
    pub sampler_seed: u64,
    pub generation: u64,
    pub learning_rate: f32,
    pub objective: LearnerObjectiveV1,
}

impl LearnerIdentityV1 {
    pub fn validate(self) -> Result<Self, LearnerCommitError> {
        if self.batch_size == 0 {
            return Err(LearnerCommitError::InvalidNumber("batch-size"));
        }
        if self.legal_action_capacity == 0 {
            return Err(LearnerCommitError::InvalidNumber("action-capacity"));
        }
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(LearnerCommitError::InvalidLearningRate(self.learning_rate));
        }
        u64::try_from(self.batch_size)
            .map_err(|_| LearnerCommitError::InvalidNumber("batch-size"))?;
        u64::try_from(self.legal_action_capacity)
            .map_err(|_| LearnerCommitError::InvalidNumber("action-capacity"))?;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LearnerCommit {
    format: CommitFormat,
    identity: LearnerIdentityV1,
    starting_training_step: u64,
    training_step: u64,
    next_replay_index: u64,
    next_request_id: u64,
    checkpoint_file: String,
    checkpoint_sha256: [u8; 32],
}

impl LearnerCommit {
    pub fn new(
        identity: LearnerIdentityV1,
        starting_training_step: u64,
        training_step: u64,
        next_replay_index: u64,
        next_request_id: u64,
        checkpoint_file: impl Into<String>,
        checkpoint_sha256: [u8; 32],
    ) -> Result<Self, LearnerCommitError> {
        Self::new_with_format(
            CommitFormat::V3,
            identity,
            starting_training_step,
            training_step,
            next_replay_index,
            next_request_id,
            checkpoint_file,
            checkpoint_sha256,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_format(
        format: CommitFormat,
        identity: LearnerIdentityV1,
        starting_training_step: u64,
        training_step: u64,
        next_replay_index: u64,
        next_request_id: u64,
        checkpoint_file: impl Into<String>,
        checkpoint_sha256: [u8; 32],
    ) -> Result<Self, LearnerCommitError> {
        let identity = identity.validate()?;
        if format == CommitFormat::V1 && starting_training_step != 0 {
            return Err(LearnerCommitError::LegacyCommitHasGenerationStart(
                starting_training_step,
            ));
        }
        if format != CommitFormat::V3
            && identity.objective != LearnerObjectiveV1::SupervisedPolicyValue
        {
            return Err(LearnerCommitError::LegacyCommitHasTerminalPpoObjective);
        }
        let checkpoint_file = checkpoint_file.into();
        validate_file_name(&checkpoint_file)?;
        let local_training_steps = training_step.checked_sub(starting_training_step).ok_or(
            LearnerCommitError::TrainingStepPrecedesGeneration {
                starting: starting_training_step,
                actual: training_step,
            },
        )?;
        let expected_index = local_training_steps
            .checked_mul(identity.batch_size as u64)
            .ok_or(LearnerCommitError::ReplayIndexOverflow)?;
        if next_replay_index != expected_index {
            return Err(LearnerCommitError::ReplayIndexMismatch {
                expected: expected_index,
                actual: next_replay_index,
            });
        }
        let (checkpoint_generation, checkpoint_step, _) =
            parse_checkpoint_file_name(&checkpoint_file)?;
        if checkpoint_generation != identity.generation || checkpoint_step != training_step {
            return Err(LearnerCommitError::CheckpointIdentityMismatch {
                expected_generation: identity.generation,
                actual_generation: checkpoint_generation,
                expected_step: training_step,
                actual_step: checkpoint_step,
            });
        }
        Ok(Self {
            format,
            identity,
            starting_training_step,
            training_step,
            next_replay_index,
            next_request_id,
            checkpoint_file,
            checkpoint_sha256,
        })
    }

    pub const fn identity(&self) -> LearnerIdentityV1 {
        self.identity
    }

    pub const fn format_version(&self) -> u8 {
        self.format.version()
    }

    pub const fn training_step(&self) -> u64 {
        self.training_step
    }

    pub const fn starting_training_step(&self) -> u64 {
        self.starting_training_step
    }

    pub const fn next_replay_index(&self) -> u64 {
        self.next_replay_index
    }

    pub const fn next_request_id(&self) -> u64 {
        self.next_request_id
    }

    pub fn checkpoint_file(&self) -> &str {
        &self.checkpoint_file
    }

    pub const fn checkpoint_sha256(&self) -> [u8; 32] {
        self.checkpoint_sha256
    }

    pub fn checkpoint_path(&self, directory: &Path) -> PathBuf {
        directory.join(&self.checkpoint_file)
    }

    pub fn write_idempotently(&self, directory: &Path) -> Result<PathBuf, LearnerCommitError> {
        let destination = directory.join(commit_file_name(self.training_step));
        atomic_file::write_idempotently(&destination, self.encode().as_bytes())
            .map_err(LearnerCommitError::Io)?;
        Ok(destination)
    }

    pub fn read(source: &Path) -> Result<Self, LearnerCommitError> {
        let bytes = fs::read(source).map_err(LearnerCommitError::Io)?;
        let text = core::str::from_utf8(&bytes).map_err(|_| LearnerCommitError::InvalidUtf8)?;
        let commit = decode(text)?;
        if commit.encode().as_bytes() != bytes {
            return Err(LearnerCommitError::NonCanonicalEncoding);
        }
        let expected_name = commit_file_name(commit.training_step);
        if source.file_name().and_then(|name| name.to_str()) != Some(expected_name.as_str()) {
            return Err(LearnerCommitError::UnexpectedCommitFile {
                expected: expected_name,
                actual: source.to_owned(),
            });
        }
        Ok(commit)
    }

    fn encode(&self) -> String {
        use core::fmt::Write as _;

        let mut text = self.body();
        let digest: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        writeln!(text, "sha256\t{}", encode_digest(digest)).unwrap();
        text
    }

    fn body(&self) -> String {
        use core::fmt::Write as _;

        let identity = self.identity;
        let mut text = String::new();
        writeln!(text, "{}", self.format.magic()).unwrap();
        writeln!(text, "rules\t{}", RuleProfileId::SkudPaiSho2022.as_str()).unwrap();
        writeln!(text, "snapshot\t{}", identity.replay_snapshot).unwrap();
        writeln!(text, "network\t{}", preset_name(identity.network_preset)).unwrap();
        writeln!(
            text,
            "optimization\t{}",
            optimization_name(identity.optimization)
        )
        .unwrap();
        writeln!(text, "batch-size\t{}", identity.batch_size).unwrap();
        writeln!(text, "action-capacity\t{}", identity.legal_action_capacity).unwrap();
        writeln!(text, "model-seed\t{}", identity.model_seed).unwrap();
        writeln!(text, "sampler-seed\t{}", identity.sampler_seed).unwrap();
        writeln!(text, "generation\t{}", identity.generation).unwrap();
        writeln!(
            text,
            "learning-rate-bits\t{:08x}",
            identity.learning_rate.to_bits()
        )
        .unwrap();
        if self.format == CommitFormat::V3 {
            write_objective(&mut text, identity.objective);
        }
        writeln!(text, "training-step\t{}", self.training_step).unwrap();
        if matches!(self.format, CommitFormat::V2 | CommitFormat::V3) {
            writeln!(
                text,
                "starting-training-step\t{}",
                self.starting_training_step
            )
            .unwrap();
        }
        writeln!(text, "next-replay-index\t{}", self.next_replay_index).unwrap();
        writeln!(text, "next-request-id\t{}", self.next_request_id).unwrap();
        writeln!(text, "checkpoint\t{}", self.checkpoint_file).unwrap();
        writeln!(
            text,
            "checkpoint-sha256\t{}",
            encode_digest(self.checkpoint_sha256)
        )
        .unwrap();
        text
    }
}

fn write_objective(text: &mut String, objective: LearnerObjectiveV1) {
    use core::fmt::Write as _;

    writeln!(text, "objective\t{}", objective.identifier()).unwrap();
    let Some(terminal) = objective.terminal_ppo() else {
        return;
    };
    writeln!(text, "behavior-producer\t{}", terminal.behavior_producer()).unwrap();
    match terminal.actor_checkpoint_sha256() {
        Some(digest) => writeln!(text, "actor-checkpoint-sha256\t{digest}").unwrap(),
        None => writeln!(text, "actor-checkpoint-sha256\tNONE").unwrap(),
    }
    let parameters = terminal.parameters();
    writeln!(
        text,
        "policy-temperature-bits\t{:08x}",
        parameters.policy_temperature().to_bits()
    )
    .unwrap();
    writeln!(
        text,
        "uniform-mix-bits\t{:08x}",
        parameters.uniform_mix().to_bits()
    )
    .unwrap();
    writeln!(
        text,
        "ppo-clip-epsilon-bits\t{:08x}",
        parameters.clip_epsilon().to_bits()
    )
    .unwrap();
    writeln!(
        text,
        "ppo-value-loss-weight-bits\t{:08x}",
        parameters.value_loss_weight().to_bits()
    )
    .unwrap();
    writeln!(
        text,
        "ppo-entropy-weight-bits\t{:08x}",
        parameters.entropy_weight().to_bits()
    )
    .unwrap();
}

pub(crate) fn checkpoint_file_name(generation: u64, training_step: u64, attempt: u64) -> String {
    format!("checkpoint-g{generation:020}-s{training_step:020}-a{attempt:020}.psckpt")
}

pub fn discover_latest_commit(
    directory: &Path,
    expected_identity: LearnerIdentityV1,
) -> Result<Option<LearnerCommit>, LearnerCommitError> {
    let expected_identity = expected_identity.validate()?;
    if !directory.exists() {
        return Ok(None);
    }
    let mut commit_paths = Vec::new();
    for entry in fs::read_dir(directory).map_err(LearnerCommitError::Io)? {
        let entry = entry.map_err(LearnerCommitError::Io)?;
        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some(&COMMIT_SUFFIX[1..]) {
            continue;
        }
        commit_paths.push((parse_commit_step(&path)?, path));
    }
    commit_paths.sort_by_key(|(step, _)| *step);
    let Some((_, latest_path)) = commit_paths.pop() else {
        return Ok(None);
    };
    let latest =
        LearnerCommit::read(&latest_path).map_err(|source| LearnerCommitError::InvalidCommit {
            path: latest_path.clone(),
            source: Box::new(source),
        })?;
    if latest.identity != expected_identity {
        return Err(LearnerCommitError::IdentityMismatch(latest_path));
    }
    let checkpoint_path = latest.checkpoint_path(directory);
    let actual =
        verify_checkpoint_file(&checkpoint_path).map_err(LearnerCommitError::Checkpoint)?;
    if actual != latest.checkpoint_sha256 {
        return Err(LearnerCommitError::CheckpointDigestMismatch {
            path: checkpoint_path,
            expected: latest.checkpoint_sha256,
            actual,
        });
    }
    Ok(Some(latest))
}

pub(crate) fn commit_file_name(training_step: u64) -> String {
    format!("commit-s{training_step:020}{COMMIT_SUFFIX}")
}

fn parse_commit_step(path: &Path) -> Result<u64, LearnerCommitError> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| LearnerCommitError::InvalidCommitFile(path.to_owned()))?;
    let digits = file_name
        .strip_prefix("commit-s")
        .and_then(|value| value.strip_suffix(COMMIT_SUFFIX))
        .filter(|digits| digits.len() == 20 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .ok_or_else(|| LearnerCommitError::InvalidCommitFile(path.to_owned()))?;
    digits
        .parse()
        .map_err(|_| LearnerCommitError::InvalidCommitFile(path.to_owned()))
}

fn parse_checkpoint_file_name(file_name: &str) -> Result<(u64, u64, u64), LearnerCommitError> {
    validate_file_name(file_name)?;
    let body = file_name
        .strip_prefix("checkpoint-g")
        .and_then(|value| value.strip_suffix(".psckpt"))
        .ok_or_else(|| LearnerCommitError::InvalidCheckpointFile(file_name.to_owned()))?;
    let mut fields = body.split('-');
    let generation = parse_padded_file_number(fields.next(), None, file_name)?;
    let step = parse_padded_file_number(fields.next(), Some('s'), file_name)?;
    let attempt = parse_padded_file_number(fields.next(), Some('a'), file_name)?;
    if fields.next().is_some() {
        return Err(LearnerCommitError::InvalidCheckpointFile(
            file_name.to_owned(),
        ));
    }
    Ok((generation, step, attempt))
}

fn parse_padded_file_number(
    field: Option<&str>,
    prefix: Option<char>,
    file_name: &str,
) -> Result<u64, LearnerCommitError> {
    let field =
        field.ok_or_else(|| LearnerCommitError::InvalidCheckpointFile(file_name.to_owned()))?;
    let digits = match prefix {
        Some(prefix) => field.strip_prefix(prefix),
        None => Some(field),
    }
    .filter(|digits| digits.len() == 20 && digits.bytes().all(|byte| byte.is_ascii_digit()))
    .ok_or_else(|| LearnerCommitError::InvalidCheckpointFile(file_name.to_owned()))?;
    digits
        .parse()
        .map_err(|_| LearnerCommitError::InvalidCheckpointFile(file_name.to_owned()))
}

fn decode(text: &str) -> Result<LearnerCommit, LearnerCommitError> {
    let without_final_newline = text
        .strip_suffix('\n')
        .ok_or(LearnerCommitError::NonCanonicalEncoding)?;
    let checksum_start = without_final_newline
        .rfind('\n')
        .map(|index| index + 1)
        .ok_or(LearnerCommitError::Truncated)?;
    let checksum_line = &without_final_newline[checksum_start..];
    let body = &text[..checksum_start];
    let stored = parse_digest(field_value(checksum_line, "sha256")?)?;
    let actual: [u8; 32] = Sha256::digest(body.as_bytes()).into();
    if stored != actual {
        return Err(LearnerCommitError::ChecksumMismatch);
    }

    let mut lines = body.lines();
    let format = match lines.next() {
        Some(MAGIC_V1) => CommitFormat::V1,
        Some(MAGIC_V2) => CommitFormat::V2,
        Some(MAGIC_V3) => CommitFormat::V3,
        _ => return Err(LearnerCommitError::InvalidField("signature")),
    };
    require_line(
        lines.next(),
        &format!("rules\t{}", RuleProfileId::SkudPaiSho2022.as_str()),
        "rules",
    )?;
    let snapshot = parse_value::<ReplayDigestV1>(&mut lines, "snapshot")?;
    let network_preset = parse_preset(next_value(&mut lines, "network")?)?;
    let optimization = parse_optimization(next_value(&mut lines, "optimization")?)?;
    let batch_size = parse_value(&mut lines, "batch-size")?;
    let legal_action_capacity = parse_value(&mut lines, "action-capacity")?;
    let model_seed = parse_value(&mut lines, "model-seed")?;
    let sampler_seed = parse_value(&mut lines, "sampler-seed")?;
    let generation = parse_value(&mut lines, "generation")?;
    let learning_rate_bits = u32::from_str_radix(next_value(&mut lines, "learning-rate-bits")?, 16)
        .map_err(|_| LearnerCommitError::InvalidNumber("learning-rate-bits"))?;
    let objective = match format {
        CommitFormat::V1 | CommitFormat::V2 => LearnerObjectiveV1::SupervisedPolicyValue,
        CommitFormat::V3 => parse_objective(&mut lines)?,
    };
    let training_step: u64 = parse_value(&mut lines, "training-step")?;
    let starting_training_step = match format {
        CommitFormat::V1 => 0,
        CommitFormat::V2 | CommitFormat::V3 => parse_value(&mut lines, "starting-training-step")?,
    };
    let next_replay_index: u64 = parse_value(&mut lines, "next-replay-index")?;
    let next_request_id = parse_value(&mut lines, "next-request-id")?;
    let checkpoint_file = next_value(&mut lines, "checkpoint")?.to_owned();
    let checkpoint_sha256 = parse_digest(next_value(&mut lines, "checkpoint-sha256")?)?;
    if lines.next().is_some() {
        return Err(LearnerCommitError::TrailingLines);
    }
    let identity = LearnerIdentityV1 {
        replay_snapshot: snapshot,
        network_preset,
        optimization,
        batch_size,
        legal_action_capacity,
        model_seed,
        sampler_seed,
        generation,
        learning_rate: f32::from_bits(learning_rate_bits),
        objective,
    }
    .validate()?;
    LearnerCommit::new_with_format(
        format,
        identity,
        starting_training_step,
        training_step,
        next_replay_index,
        next_request_id,
        checkpoint_file,
        checkpoint_sha256,
    )
}

fn parse_objective<'text>(
    lines: &mut impl Iterator<Item = &'text str>,
) -> Result<LearnerObjectiveV1, LearnerCommitError> {
    match next_value(lines, "objective")? {
        SUPERVISED_OBJECTIVE_V1 => Ok(LearnerObjectiveV1::SupervisedPolicyValue),
        TERMINAL_PPO_OBJECTIVE_V1 => {
            let behavior_producer = parse_value(lines, "behavior-producer")?;
            let actor_checkpoint_sha256 = match next_value(lines, "actor-checkpoint-sha256")? {
                "NONE" => None,
                digest => Some(
                    digest
                        .parse::<ReplayDigestV1>()
                        .map_err(LearnerCommitError::Digest)?,
                ),
            };
            let parameters = TerminalPpoParametersV1::with_behavior(
                parse_float_bits(lines, "policy-temperature-bits")?,
                parse_float_bits(lines, "uniform-mix-bits")?,
                parse_float_bits(lines, "ppo-clip-epsilon-bits")?,
                parse_float_bits(lines, "ppo-value-loss-weight-bits")?,
                parse_float_bits(lines, "ppo-entropy-weight-bits")?,
            )
            .map_err(LearnerCommitError::TerminalPpo)?;
            Ok(LearnerObjectiveV1::TerminalPpo(
                TerminalPpoLearnerObjectiveV1::new(
                    behavior_producer,
                    actor_checkpoint_sha256,
                    parameters,
                ),
            ))
        }
        _ => Err(LearnerCommitError::InvalidField("objective")),
    }
}

fn parse_float_bits<'text>(
    lines: &mut impl Iterator<Item = &'text str>,
    field: &'static str,
) -> Result<f32, LearnerCommitError> {
    u32::from_str_radix(next_value(lines, field)?, 16)
        .map(f32::from_bits)
        .map_err(|_| LearnerCommitError::InvalidNumber(field))
}

fn require_line(
    actual: Option<&str>,
    expected: &str,
    field: &'static str,
) -> Result<(), LearnerCommitError> {
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(LearnerCommitError::InvalidField(field))
    }
}

fn next_value<'text>(
    lines: &mut impl Iterator<Item = &'text str>,
    field: &'static str,
) -> Result<&'text str, LearnerCommitError> {
    lines
        .next()
        .and_then(|line| field_value(line, field).ok())
        .ok_or(LearnerCommitError::InvalidField(field))
}

fn field_value<'text>(
    line: &'text str,
    field: &'static str,
) -> Result<&'text str, LearnerCommitError> {
    line.strip_prefix(field)
        .and_then(|value| value.strip_prefix('\t'))
        .filter(|value| !value.is_empty() && !value.contains('\t'))
        .ok_or(LearnerCommitError::InvalidField(field))
}

fn parse_value<'text, T: FromStr>(
    lines: &mut impl Iterator<Item = &'text str>,
    field: &'static str,
) -> Result<T, LearnerCommitError> {
    next_value(lines, field)?
        .parse()
        .map_err(|_| LearnerCommitError::InvalidNumber(field))
}

fn preset_name(preset: NetworkPreset) -> &'static str {
    match preset {
        NetworkPreset::Micro => "micro",
        NetworkPreset::Pure => "pure",
    }
}

fn parse_preset(value: &str) -> Result<NetworkPreset, LearnerCommitError> {
    match value {
        "micro" => Ok(NetworkPreset::Micro),
        "pure" => Ok(NetworkPreset::Pure),
        _ => Err(LearnerCommitError::InvalidField("network")),
    }
}

fn optimization_name(level: OptimizationLevel) -> &'static str {
    match level {
        OptimizationLevel::Level0 => "0",
        OptimizationLevel::Level1 => "1",
    }
}

fn parse_optimization(value: &str) -> Result<OptimizationLevel, LearnerCommitError> {
    match value {
        "0" => Ok(OptimizationLevel::Level0),
        "1" => Ok(OptimizationLevel::Level1),
        _ => Err(LearnerCommitError::InvalidField("optimization")),
    }
}

fn validate_file_name(file_name: &str) -> Result<(), LearnerCommitError> {
    let mut components = Path::new(file_name).components();
    let one_normal =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if file_name.is_empty()
        || !one_normal
        || file_name
            .chars()
            .any(|character| character.is_control() || character == '\t')
    {
        Err(LearnerCommitError::InvalidCheckpointFile(
            file_name.to_owned(),
        ))
    } else {
        Ok(())
    }
}

fn encode_digest(digest: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(encoded, "{byte:02x}").unwrap();
    }
    encoded
}

fn parse_digest(text: &str) -> Result<[u8; 32], LearnerCommitError> {
    Ok(*text
        .parse::<ReplayDigestV1>()
        .map_err(LearnerCommitError::Digest)?
        .as_bytes())
}

#[derive(Debug)]
pub enum LearnerCommitError {
    InvalidUtf8,
    NonCanonicalEncoding,
    Truncated,
    InvalidField(&'static str),
    InvalidNumber(&'static str),
    InvalidLearningRate(f32),
    InvalidCheckpointFile(String),
    ChecksumMismatch,
    TrailingLines,
    ReplayIndexOverflow,
    LegacyCommitHasGenerationStart(u64),
    LegacyCommitHasTerminalPpoObjective,
    TrainingStepPrecedesGeneration {
        starting: u64,
        actual: u64,
    },
    ReplayIndexMismatch {
        expected: u64,
        actual: u64,
    },
    CheckpointIdentityMismatch {
        expected_generation: u64,
        actual_generation: u64,
        expected_step: u64,
        actual_step: u64,
    },
    UnexpectedCommitFile {
        expected: String,
        actual: PathBuf,
    },
    InvalidCommitFile(PathBuf),
    IdentityMismatch(PathBuf),
    CheckpointDigestMismatch {
        path: PathBuf,
        expected: [u8; 32],
        actual: [u8; 32],
    },
    InvalidCommit {
        path: PathBuf,
        source: Box<LearnerCommitError>,
    },
    Digest(ReplayDigestV1Error),
    TerminalPpo(TerminalPpoWireError),
    Checkpoint(MpsGraphClientError),
    Io(io::Error),
}

impl fmt::Display for LearnerCommitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => formatter.write_str("learner commit is not UTF-8"),
            Self::NonCanonicalEncoding => formatter.write_str("learner commit is not canonical"),
            Self::Truncated => formatter.write_str("learner commit is truncated"),
            Self::InvalidField(field) => write!(formatter, "invalid learner commit field {field}"),
            Self::InvalidNumber(field) => {
                write!(formatter, "invalid number in learner field {field}")
            }
            Self::InvalidLearningRate(rate) => write!(formatter, "invalid learning rate {rate}"),
            Self::InvalidCheckpointFile(file) => {
                write!(formatter, "invalid checkpoint file name {file:?}")
            }
            Self::ChecksumMismatch => formatter.write_str("learner commit checksum mismatch"),
            Self::TrailingLines => formatter.write_str("learner commit has trailing lines"),
            Self::ReplayIndexOverflow => {
                formatter.write_str("training step and batch size overflow the replay index")
            }
            Self::LegacyCommitHasGenerationStart(step) => write!(
                formatter,
                "learner commit V1 cannot start a generation at global step {step}"
            ),
            Self::LegacyCommitHasTerminalPpoObjective => {
                formatter.write_str("learner commit V1/V2 cannot carry a terminal PPO objective")
            }
            Self::TrainingStepPrecedesGeneration { starting, actual } => write!(
                formatter,
                "training step {actual} precedes generation start step {starting}"
            ),
            Self::ReplayIndexMismatch { expected, actual } => write!(
                formatter,
                "learner commit replay index is {actual}; expected {expected}"
            ),
            Self::CheckpointIdentityMismatch {
                expected_generation,
                actual_generation,
                expected_step,
                actual_step,
            } => write!(
                formatter,
                "checkpoint claims generation {actual_generation}, step {actual_step}; expected generation {expected_generation}, step {expected_step}"
            ),
            Self::UnexpectedCommitFile { expected, actual } => write!(
                formatter,
                "learner commit {} should be named {expected}",
                actual.display()
            ),
            Self::InvalidCommitFile(path) => write!(
                formatter,
                "invalid learner commit file name {}",
                path.display()
            ),
            Self::IdentityMismatch(path) => write!(
                formatter,
                "learner commit {} belongs to a different run",
                path.display()
            ),
            Self::CheckpointDigestMismatch { path, .. } => write!(
                formatter,
                "checkpoint {} does not match its learner commit digest",
                path.display()
            ),
            Self::InvalidCommit { path, source } => write!(
                formatter,
                "invalid learner commit {}: {source}",
                path.display()
            ),
            Self::Digest(source) => source.fmt(formatter),
            Self::TerminalPpo(source) => source.fmt(formatter),
            Self::Checkpoint(source) => source.fmt(formatter),
            Self::Io(source) => write!(formatter, "learner commit I/O failed: {source}"),
        }
    }
}

impl std::error::Error for LearnerCommitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCommit { source, .. } => Some(source),
            Self::Digest(source) => Some(source),
            Self::TerminalPpo(source) => Some(source),
            Self::Checkpoint(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terminal_objective() -> LearnerObjectiveV1 {
        LearnerObjectiveV1::TerminalPpo(TerminalPpoLearnerObjectiveV1::new(
            ReplayDigestV1::from_bytes([0x88; 32]),
            Some(ReplayDigestV1::from_bytes([0x99; 32])),
            TerminalPpoParametersV1::with_behavior(0.8, 0.05, 0.2, 0.5, 0.01).unwrap(),
        ))
    }

    #[test]
    fn decoded_zero_batch_is_an_error_instead_of_a_division_panic() {
        let identity = LearnerIdentityV1 {
            replay_snapshot: ReplayDigestV1::from_bytes([0x44; 32]),
            network_preset: NetworkPreset::Micro,
            optimization: OptimizationLevel::Level1,
            batch_size: 8,
            legal_action_capacity: 1_024,
            model_seed: 17,
            sampler_seed: 23,
            generation: 0,
            learning_rate: 1.0e-4,
            objective: LearnerObjectiveV1::SupervisedPolicyValue,
        };
        let commit = LearnerCommit::new(
            identity,
            0,
            1,
            8,
            2,
            checkpoint_file_name(0, 1, 0),
            [0x55; 32],
        )
        .unwrap();
        let body = commit.body().replace("batch-size\t8", "batch-size\t0");
        let checksum: [u8; 32] = Sha256::digest(body.as_bytes()).into();
        let malformed = format!("{body}sha256\t{}\n", encode_digest(checksum));
        assert!(matches!(
            decode(&malformed),
            Err(LearnerCommitError::InvalidNumber("batch-size"))
        ));
    }

    #[test]
    fn legacy_v1_manifest_is_read_canonically_with_a_zero_generation_start() {
        let identity = LearnerIdentityV1 {
            replay_snapshot: ReplayDigestV1::from_bytes([0x66; 32]),
            network_preset: NetworkPreset::Micro,
            optimization: OptimizationLevel::Level1,
            batch_size: 8,
            legal_action_capacity: 1_024,
            model_seed: 17,
            sampler_seed: 23,
            generation: 0,
            learning_rate: 1.0e-4,
            objective: LearnerObjectiveV1::SupervisedPolicyValue,
        };
        let legacy = LearnerCommit::new_with_format(
            CommitFormat::V1,
            identity,
            0,
            2,
            16,
            3,
            checkpoint_file_name(0, 2, 0),
            [0x77; 32],
        )
        .unwrap();
        let encoded = legacy.encode();
        assert!(encoded.starts_with("PAISHO-LEARNER-COMMIT\t1\n"));
        assert!(!encoded.contains("starting-training-step"));
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, legacy);
        assert_eq!(decoded.format_version(), 1);
        assert_eq!(decoded.starting_training_step(), 0);
    }

    #[test]
    fn v3_manifest_round_trips_terminal_ppo_identity() {
        let mut identity = LearnerIdentityV1 {
            replay_snapshot: ReplayDigestV1::from_bytes([0x66; 32]),
            network_preset: NetworkPreset::Micro,
            optimization: OptimizationLevel::Level1,
            batch_size: 8,
            legal_action_capacity: 1_024,
            model_seed: 17,
            sampler_seed: 23,
            generation: 2,
            learning_rate: 1.0e-4,
            objective: terminal_objective(),
        };
        let commit = LearnerCommit::new(
            identity,
            100,
            102,
            16,
            3,
            checkpoint_file_name(2, 102, 0),
            [0x77; 32],
        )
        .unwrap();
        let encoded = commit.encode();
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, commit);
        assert_eq!(decoded.identity().objective, terminal_objective());

        identity.objective = LearnerObjectiveV1::SupervisedPolicyValue;
        assert!(LearnerCommit::new_with_format(
            CommitFormat::V2,
            identity,
            100,
            102,
            16,
            3,
            checkpoint_file_name(2, 102, 0),
            [0x77; 32],
        )
        .is_ok());
    }

    #[test]
    fn legacy_manifest_cannot_silently_drop_terminal_ppo_identity() {
        let identity = LearnerIdentityV1 {
            replay_snapshot: ReplayDigestV1::from_bytes([0x66; 32]),
            network_preset: NetworkPreset::Micro,
            optimization: OptimizationLevel::Level1,
            batch_size: 8,
            legal_action_capacity: 1_024,
            model_seed: 17,
            sampler_seed: 23,
            generation: 2,
            learning_rate: 1.0e-4,
            objective: terminal_objective(),
        };
        assert!(matches!(
            LearnerCommit::new_with_format(
                CommitFormat::V2,
                identity,
                100,
                102,
                16,
                3,
                checkpoint_file_name(2, 102, 0),
                [0x77; 32],
            ),
            Err(LearnerCommitError::LegacyCommitHasTerminalPpoObjective)
        ));
    }
}
