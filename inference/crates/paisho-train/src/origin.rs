use core::fmt;
use core::str::FromStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use paisho_core::RuleProfileId;
use paisho_model::{TerminalPpoParametersV1, TerminalPpoWireError, TERMINAL_PPO_OBJECTIVE_V1};
use paisho_mpsgraph_client::{NetworkPreset, OptimizationLevel};
use paisho_replay::{ReplayDigestV1, ReplayDigestV1Error};
use sha2::{Digest, Sha256};

use crate::atomic_file;
use crate::{
    LearnerCommitError, LearnerIdentityV1, LearnerObjectiveV1, TerminalPpoLearnerObjectiveV1,
    SUPERVISED_OBJECTIVE_V1,
};

const MAGIC_V1: &str = "PAISHO-LEARNER-ORIGIN\t1";
const MAGIC_V2: &str = "PAISHO-LEARNER-ORIGIN\t2";
const FILE_NAME: &str = "origin.psorigin";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OriginFormat {
    V1,
    V2,
}

impl OriginFormat {
    const fn magic(self) -> &'static str {
        match self {
            Self::V1 => MAGIC_V1,
            Self::V2 => MAGIC_V2,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParentCheckpointV1 {
    content_sha256: [u8; 32],
    generation: u64,
    training_step: u64,
}

impl ParentCheckpointV1 {
    pub const fn new(content_sha256: [u8; 32], generation: u64, training_step: u64) -> Self {
        Self {
            content_sha256,
            generation,
            training_step,
        }
    }

    pub const fn content_sha256(&self) -> [u8; 32] {
        self.content_sha256
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn training_step(&self) -> u64 {
        self.training_step
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LearnerOriginV1 {
    format: OriginFormat,
    identity: LearnerIdentityV1,
    parent: Option<ParentCheckpointV1>,
}

impl LearnerOriginV1 {
    pub fn new(
        identity: LearnerIdentityV1,
        parent: Option<ParentCheckpointV1>,
    ) -> Result<Self, LearnerOriginV1Error> {
        Self::new_with_format(OriginFormat::V2, identity, parent)
    }

    fn new_with_format(
        format: OriginFormat,
        identity: LearnerIdentityV1,
        parent: Option<ParentCheckpointV1>,
    ) -> Result<Self, LearnerOriginV1Error> {
        let identity = identity
            .validate()
            .map_err(LearnerOriginV1Error::Identity)?;
        if format == OriginFormat::V1
            && identity.objective != LearnerObjectiveV1::SupervisedPolicyValue
        {
            return Err(LearnerOriginV1Error::LegacyOriginHasTerminalPpoObjective);
        }
        Ok(Self {
            format,
            identity,
            parent,
        })
    }

    pub const fn identity(&self) -> LearnerIdentityV1 {
        self.identity
    }

    pub const fn parent(&self) -> Option<ParentCheckpointV1> {
        self.parent
    }

    pub const fn starting_training_step(&self) -> u64 {
        match self.parent {
            Some(parent) => parent.training_step,
            None => 0,
        }
    }

    pub fn path(directory: &Path) -> PathBuf {
        directory.join(FILE_NAME)
    }

    pub fn establish(&self, directory: &Path) -> Result<PathBuf, LearnerOriginV1Error> {
        let destination = Self::path(directory);
        if let Err(error) = atomic_file::write_idempotently(&destination, self.encode().as_bytes())
        {
            if error.kind() == io::ErrorKind::AlreadyExists {
                return Err(LearnerOriginV1Error::Mismatch(destination));
            }
            return Err(LearnerOriginV1Error::Io(error));
        }
        let stored = Self::read(&destination)?;
        if stored != *self {
            return Err(LearnerOriginV1Error::Mismatch(destination));
        }
        Ok(destination)
    }

    pub fn read(source: &Path) -> Result<Self, LearnerOriginV1Error> {
        let bytes = fs::read(source).map_err(LearnerOriginV1Error::Io)?;
        let text = core::str::from_utf8(&bytes).map_err(|_| LearnerOriginV1Error::InvalidUtf8)?;
        let origin = decode(text)?;
        if origin.encode().as_bytes() != bytes {
            return Err(LearnerOriginV1Error::NonCanonicalEncoding);
        }
        Ok(origin)
    }

    fn encode(&self) -> String {
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
        if self.format == OriginFormat::V2 {
            write_objective(&mut text, identity.objective);
        }
        match self.parent {
            Some(parent) => {
                writeln!(
                    text,
                    "parent-checkpoint-sha256\t{}",
                    encode_digest(parent.content_sha256)
                )
                .unwrap();
                writeln!(text, "parent-generation\t{}", parent.generation).unwrap();
                writeln!(text, "starting-training-step\t{}", parent.training_step).unwrap();
            }
            None => {
                writeln!(text, "parent-checkpoint-sha256\tNONE").unwrap();
                writeln!(text, "parent-generation\tNONE").unwrap();
                writeln!(text, "starting-training-step\t0").unwrap();
            }
        }
        let checksum: [u8; 32] = Sha256::digest(text.as_bytes()).into();
        writeln!(text, "sha256\t{}", encode_digest(checksum)).unwrap();
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

fn decode(text: &str) -> Result<LearnerOriginV1, LearnerOriginV1Error> {
    let without_final_newline = text
        .strip_suffix('\n')
        .ok_or(LearnerOriginV1Error::NonCanonicalEncoding)?;
    let checksum_start = without_final_newline
        .rfind('\n')
        .map(|index| index + 1)
        .ok_or(LearnerOriginV1Error::Truncated)?;
    let checksum_line = &without_final_newline[checksum_start..];
    let body = &text[..checksum_start];
    let stored = parse_digest(field_value(checksum_line, "sha256")?)?;
    let actual: [u8; 32] = Sha256::digest(body.as_bytes()).into();
    if stored != actual {
        return Err(LearnerOriginV1Error::ChecksumMismatch);
    }

    let mut lines = body.lines();
    let format = match lines.next() {
        Some(MAGIC_V1) => OriginFormat::V1,
        Some(MAGIC_V2) => OriginFormat::V2,
        _ => return Err(LearnerOriginV1Error::InvalidField("signature")),
    };
    require_line(
        lines.next(),
        &format!("rules\t{}", RuleProfileId::SkudPaiSho2022.as_str()),
        "rules",
    )?;
    let replay_snapshot = parse_value(&mut lines, "snapshot")?;
    let network_preset = parse_preset(next_value(&mut lines, "network")?)?;
    let optimization = parse_optimization(next_value(&mut lines, "optimization")?)?;
    let batch_size = parse_value(&mut lines, "batch-size")?;
    let legal_action_capacity = parse_value(&mut lines, "action-capacity")?;
    let model_seed = parse_value(&mut lines, "model-seed")?;
    let sampler_seed = parse_value(&mut lines, "sampler-seed")?;
    let generation = parse_value(&mut lines, "generation")?;
    let learning_rate_bits = u32::from_str_radix(next_value(&mut lines, "learning-rate-bits")?, 16)
        .map_err(|_| LearnerOriginV1Error::InvalidNumber("learning-rate-bits"))?;
    let objective = match format {
        OriginFormat::V1 => LearnerObjectiveV1::SupervisedPolicyValue,
        OriginFormat::V2 => parse_objective(&mut lines)?,
    };
    let parent_digest = next_value(&mut lines, "parent-checkpoint-sha256")?;
    let parent_generation = next_value(&mut lines, "parent-generation")?;
    let starting_training_step = parse_value(&mut lines, "starting-training-step")?;
    if lines.next().is_some() {
        return Err(LearnerOriginV1Error::TrailingLines);
    }
    let parent = match (parent_digest, parent_generation) {
        ("NONE", "NONE") if starting_training_step == 0 => None,
        ("NONE", "NONE") => return Err(LearnerOriginV1Error::InvalidParent),
        (digest, generation) if digest != "NONE" && generation != "NONE" => {
            Some(ParentCheckpointV1::new(
                parse_digest(digest)?,
                generation
                    .parse()
                    .map_err(|_| LearnerOriginV1Error::InvalidNumber("parent-generation"))?,
                starting_training_step,
            ))
        }
        _ => return Err(LearnerOriginV1Error::InvalidParent),
    };
    LearnerOriginV1::new_with_format(
        format,
        LearnerIdentityV1 {
            replay_snapshot,
            network_preset,
            optimization,
            batch_size,
            legal_action_capacity,
            model_seed,
            sampler_seed,
            generation,
            learning_rate: f32::from_bits(learning_rate_bits),
            objective,
        },
        parent,
    )
}

fn parse_objective<'text>(
    lines: &mut impl Iterator<Item = &'text str>,
) -> Result<LearnerObjectiveV1, LearnerOriginV1Error> {
    match next_value(lines, "objective")? {
        SUPERVISED_OBJECTIVE_V1 => Ok(LearnerObjectiveV1::SupervisedPolicyValue),
        TERMINAL_PPO_OBJECTIVE_V1 => {
            let behavior_producer = parse_value(lines, "behavior-producer")?;
            let actor_checkpoint_sha256 = match next_value(lines, "actor-checkpoint-sha256")? {
                "NONE" => None,
                digest => Some(
                    digest
                        .parse::<ReplayDigestV1>()
                        .map_err(LearnerOriginV1Error::Digest)?,
                ),
            };
            let parameters = TerminalPpoParametersV1::with_behavior(
                parse_float_bits(lines, "policy-temperature-bits")?,
                parse_float_bits(lines, "uniform-mix-bits")?,
                parse_float_bits(lines, "ppo-clip-epsilon-bits")?,
                parse_float_bits(lines, "ppo-value-loss-weight-bits")?,
                parse_float_bits(lines, "ppo-entropy-weight-bits")?,
            )
            .map_err(LearnerOriginV1Error::TerminalPpo)?;
            Ok(LearnerObjectiveV1::TerminalPpo(
                TerminalPpoLearnerObjectiveV1::new(
                    behavior_producer,
                    actor_checkpoint_sha256,
                    parameters,
                ),
            ))
        }
        _ => Err(LearnerOriginV1Error::InvalidField("objective")),
    }
}

fn parse_float_bits<'text>(
    lines: &mut impl Iterator<Item = &'text str>,
    field: &'static str,
) -> Result<f32, LearnerOriginV1Error> {
    u32::from_str_radix(next_value(lines, field)?, 16)
        .map(f32::from_bits)
        .map_err(|_| LearnerOriginV1Error::InvalidNumber(field))
}

fn require_line(
    actual: Option<&str>,
    expected: &str,
    field: &'static str,
) -> Result<(), LearnerOriginV1Error> {
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(LearnerOriginV1Error::InvalidField(field))
    }
}

fn next_value<'text>(
    lines: &mut impl Iterator<Item = &'text str>,
    field: &'static str,
) -> Result<&'text str, LearnerOriginV1Error> {
    lines
        .next()
        .and_then(|line| field_value(line, field).ok())
        .ok_or(LearnerOriginV1Error::InvalidField(field))
}

fn field_value<'text>(
    line: &'text str,
    field: &'static str,
) -> Result<&'text str, LearnerOriginV1Error> {
    line.strip_prefix(field)
        .and_then(|value| value.strip_prefix('\t'))
        .filter(|value| !value.is_empty() && !value.contains('\t'))
        .ok_or(LearnerOriginV1Error::InvalidField(field))
}

fn parse_value<'text, T: FromStr>(
    lines: &mut impl Iterator<Item = &'text str>,
    field: &'static str,
) -> Result<T, LearnerOriginV1Error> {
    next_value(lines, field)?
        .parse()
        .map_err(|_| LearnerOriginV1Error::InvalidNumber(field))
}

fn preset_name(preset: NetworkPreset) -> &'static str {
    match preset {
        NetworkPreset::Micro => "micro",
        NetworkPreset::Pure => "pure",
    }
}

fn parse_preset(value: &str) -> Result<NetworkPreset, LearnerOriginV1Error> {
    match value {
        "micro" => Ok(NetworkPreset::Micro),
        "pure" => Ok(NetworkPreset::Pure),
        _ => Err(LearnerOriginV1Error::InvalidField("network")),
    }
}

fn optimization_name(level: OptimizationLevel) -> &'static str {
    match level {
        OptimizationLevel::Level0 => "0",
        OptimizationLevel::Level1 => "1",
    }
}

fn parse_optimization(value: &str) -> Result<OptimizationLevel, LearnerOriginV1Error> {
    match value {
        "0" => Ok(OptimizationLevel::Level0),
        "1" => Ok(OptimizationLevel::Level1),
        _ => Err(LearnerOriginV1Error::InvalidField("optimization")),
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

fn parse_digest(text: &str) -> Result<[u8; 32], LearnerOriginV1Error> {
    Ok(*text
        .parse::<ReplayDigestV1>()
        .map_err(LearnerOriginV1Error::Digest)?
        .as_bytes())
}

#[derive(Debug)]
pub enum LearnerOriginV1Error {
    InvalidUtf8,
    NonCanonicalEncoding,
    Truncated,
    InvalidField(&'static str),
    InvalidNumber(&'static str),
    InvalidParent,
    LegacyOriginHasTerminalPpoObjective,
    ChecksumMismatch,
    TrailingLines,
    Mismatch(PathBuf),
    Identity(LearnerCommitError),
    Digest(ReplayDigestV1Error),
    TerminalPpo(TerminalPpoWireError),
    Io(io::Error),
}

impl fmt::Display for LearnerOriginV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUtf8 => formatter.write_str("learner origin is not UTF-8"),
            Self::NonCanonicalEncoding => formatter.write_str("learner origin is not canonical"),
            Self::Truncated => formatter.write_str("learner origin is truncated"),
            Self::InvalidField(field) => write!(formatter, "invalid learner origin field {field}"),
            Self::InvalidNumber(field) => {
                write!(formatter, "invalid number in learner origin field {field}")
            }
            Self::InvalidParent => {
                formatter.write_str("learner origin has inconsistent parent fields")
            }
            Self::LegacyOriginHasTerminalPpoObjective => {
                formatter.write_str("learner origin V1 cannot carry a terminal PPO objective")
            }
            Self::ChecksumMismatch => formatter.write_str("learner origin checksum mismatch"),
            Self::TrailingLines => formatter.write_str("learner origin has trailing lines"),
            Self::Mismatch(path) => write!(
                formatter,
                "learner origin {} differs from the requested generation",
                path.display()
            ),
            Self::Identity(source) => source.fmt(formatter),
            Self::Digest(source) => source.fmt(formatter),
            Self::TerminalPpo(source) => source.fmt(formatter),
            Self::Io(source) => write!(formatter, "learner origin I/O failed: {source}"),
        }
    }
}

impl std::error::Error for LearnerOriginV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Identity(source) => Some(source),
            Self::Digest(source) => Some(source),
            Self::TerminalPpo(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn parent_origin_round_trips_and_is_idempotent() {
        let temporary = TemporaryDirectory::new();
        let parent = ParentCheckpointV1::new([0xA5; 32], 8, 1_234);
        let origin = LearnerOriginV1::new(identity(), Some(parent)).unwrap();
        let path = origin.establish(&temporary.path).unwrap();
        origin.establish(&temporary.path).unwrap();
        assert_eq!(path, LearnerOriginV1::path(&temporary.path));
        assert_eq!(LearnerOriginV1::read(&path).unwrap(), origin);
        assert_eq!(origin.starting_training_step(), 1_234);
    }

    #[test]
    fn established_origin_rejects_a_different_parent() {
        let temporary = TemporaryDirectory::new();
        LearnerOriginV1::new(
            identity(),
            Some(ParentCheckpointV1::new([0x11; 32], 3, 100)),
        )
        .unwrap()
        .establish(&temporary.path)
        .unwrap();
        let conflict = LearnerOriginV1::new(
            identity(),
            Some(ParentCheckpointV1::new([0x22; 32], 3, 100)),
        )
        .unwrap();
        assert!(matches!(
            conflict.establish(&temporary.path),
            Err(LearnerOriginV1Error::Mismatch(path))
                if path == LearnerOriginV1::path(&temporary.path)
        ));
    }

    #[test]
    fn checksum_detects_parent_provenance_tampering() {
        let temporary = TemporaryDirectory::new();
        let origin = LearnerOriginV1::new(identity(), None).unwrap();
        let path = origin.establish(&temporary.path).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            text.replace("starting-training-step\t0", "starting-training-step\t1"),
        )
        .unwrap();
        assert!(matches!(
            LearnerOriginV1::read(&path),
            Err(LearnerOriginV1Error::ChecksumMismatch)
        ));
    }

    #[test]
    fn v2_origin_round_trips_terminal_ppo_identity() {
        let temporary = TemporaryDirectory::new();
        let parent = ParentCheckpointV1::new([0xA5; 32], 8, 1_234);
        let mut learner_identity = identity();
        learner_identity.objective =
            LearnerObjectiveV1::TerminalPpo(TerminalPpoLearnerObjectiveV1::new(
                ReplayDigestV1::from_bytes([0x77; 32]),
                Some(ReplayDigestV1::from_bytes([0xA5; 32])),
                TerminalPpoParametersV1::with_behavior(0.8, 0.05, 0.2, 0.5, 0.01).unwrap(),
            ));
        let origin = LearnerOriginV1::new(learner_identity, Some(parent)).unwrap();
        let path = origin.establish(&temporary.path).unwrap();
        let restored = LearnerOriginV1::read(&path).unwrap();
        assert_eq!(restored, origin);
        let text = fs::read_to_string(path).unwrap();
        assert!(text.starts_with("PAISHO-LEARNER-ORIGIN\t2\n"));
        assert!(text.contains("objective\tppo-terminal-v1\n"));
        assert!(text.contains(&format!(
            "actor-checkpoint-sha256\t{}\n",
            ReplayDigestV1::from_bytes([0xA5; 32])
        )));
    }

    #[test]
    fn v1_origin_remains_canonical_for_supervised_runs() {
        let origin = LearnerOriginV1::new_with_format(OriginFormat::V1, identity(), None).unwrap();
        let encoded = origin.encode();
        assert!(encoded.starts_with("PAISHO-LEARNER-ORIGIN\t1\n"));
        assert!(!encoded.contains("objective\t"));
        assert_eq!(decode(&encoded).unwrap(), origin);
    }

    fn identity() -> LearnerIdentityV1 {
        LearnerIdentityV1 {
            replay_snapshot: ReplayDigestV1::from_bytes([0x33; 32]),
            network_preset: NetworkPreset::Pure,
            optimization: OptimizationLevel::Level1,
            batch_size: 64,
            legal_action_capacity: 1_024,
            model_seed: 17,
            sampler_seed: 23,
            generation: 9,
            learning_rate: 1.0e-4,
            objective: LearnerObjectiveV1::SupervisedPolicyValue,
        }
    }

    struct TemporaryDirectory {
        path: PathBuf,
    }

    impl TemporaryDirectory {
        fn new() -> Self {
            let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "paisho-origin-test-{}-{nonce}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
