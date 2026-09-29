use core::fmt;
use core::str::FromStr;
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use paisho_core::RuleProfileId;
use sha2::{Digest, Sha256};

use crate::atomic_file;
use crate::{
    ReplayDigestV1, ReplayDigestV1Error, ReplayShardV1, ReplayShardV1Error,
    REPLAY_ENCODING_SCHEMA_V1,
};

const MAGIC: &str = "PAISHO-REPLAY-SNAPSHOT\t1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayShardReferenceV1 {
    rule_profile: RuleProfileId,
    file_name: String,
    shard_index: u64,
    digest: ReplayDigestV1,
    game_count: u64,
    example_count: u64,
}

impl ReplayShardReferenceV1 {
    pub fn new(
        file_name: impl Into<String>,
        shard_index: u64,
        digest: ReplayDigestV1,
        game_count: u64,
        example_count: u64,
    ) -> Result<Self, ReplaySnapshotV1Error> {
        Self::with_rules(
            file_name,
            shard_index,
            digest,
            game_count,
            example_count,
            RuleProfileId::CURRENT,
        )
    }

    pub fn with_rules(
        file_name: impl Into<String>,
        shard_index: u64,
        digest: ReplayDigestV1,
        game_count: u64,
        example_count: u64,
        rule_profile: RuleProfileId,
    ) -> Result<Self, ReplaySnapshotV1Error> {
        let file_name = file_name.into();
        validate_file_name(&file_name)?;
        Ok(Self {
            rule_profile,
            file_name,
            shard_index,
            digest,
            game_count,
            example_count,
        })
    }

    pub fn from_shard(
        file_name: impl Into<String>,
        shard: &ReplayShardV1,
    ) -> Result<Self, ReplaySnapshotV1Error> {
        let game_count = u64::try_from(shard.games().len())
            .map_err(|_| ReplaySnapshotV1Error::CountOverflow("games"))?;
        let example_count = u64::try_from(shard.training_example_count()?)
            .map_err(|_| ReplaySnapshotV1Error::CountOverflow("examples"))?;
        Self::with_rules(
            file_name,
            shard.shard_index(),
            shard.digest()?,
            game_count,
            example_count,
            shard.rule_profile(),
        )
    }

    pub const fn rule_profile(&self) -> RuleProfileId {
        self.rule_profile
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub const fn shard_index(&self) -> u64 {
        self.shard_index
    }

    pub const fn digest(&self) -> ReplayDigestV1 {
        self.digest
    }

    pub const fn game_count(&self) -> u64 {
        self.game_count
    }

    pub const fn example_count(&self) -> u64 {
        self.example_count
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplaySnapshotV1 {
    rule_profile: RuleProfileId,
    shards: Vec<ReplayShardReferenceV1>,
}

impl ReplaySnapshotV1 {
    pub fn new(shards: Vec<ReplayShardReferenceV1>) -> Result<Self, ReplaySnapshotV1Error> {
        let rule_profile = shards
            .first()
            .map_or(RuleProfileId::CURRENT, ReplayShardReferenceV1::rule_profile);
        Self::with_rules(shards, rule_profile)
    }

    pub fn with_rules(
        mut shards: Vec<ReplayShardReferenceV1>,
        rule_profile: RuleProfileId,
    ) -> Result<Self, ReplaySnapshotV1Error> {
        shards.sort_by_key(ReplayShardReferenceV1::shard_index);
        let mut indices = HashSet::with_capacity(shards.len());
        let mut file_names = HashSet::with_capacity(shards.len());
        for shard in &shards {
            require_shard_profile(shard.file_name(), rule_profile, shard.rule_profile())?;
            validate_file_name(shard.file_name())?;
            if !indices.insert(shard.shard_index()) {
                return Err(ReplaySnapshotV1Error::DuplicateShardIndex(
                    shard.shard_index(),
                ));
            }
            if !file_names.insert(shard.file_name().to_owned()) {
                return Err(ReplaySnapshotV1Error::DuplicateFileName(
                    shard.file_name().to_owned(),
                ));
            }
        }
        Ok(Self {
            rule_profile,
            shards,
        })
    }

    pub const fn rule_profile(&self) -> RuleProfileId {
        self.rule_profile
    }

    pub fn shards(&self) -> &[ReplayShardReferenceV1] {
        &self.shards
    }

    pub fn digest(&self) -> ReplayDigestV1 {
        digest_bytes(snapshot_body(&self.shards, self.rule_profile).as_bytes())
    }

    pub fn game_count(&self) -> Result<u64, ReplaySnapshotV1Error> {
        checked_sum(
            self.shards.iter().map(ReplayShardReferenceV1::game_count),
            "games",
        )
    }

    pub fn example_count(&self) -> Result<u64, ReplaySnapshotV1Error> {
        checked_sum(
            self.shards
                .iter()
                .map(ReplayShardReferenceV1::example_count),
            "examples",
        )
    }

    pub fn write_new(&self, destination: &Path) -> Result<ReplayDigestV1, ReplaySnapshotV1Error> {
        if destination.exists() {
            return Err(ReplaySnapshotV1Error::DestinationExists(
                destination.to_owned(),
            ));
        }
        let (text, digest) = encode_snapshot(&self.shards, self.rule_profile);
        atomic_file::write_new(destination, text.as_bytes()).map_err(ReplaySnapshotV1Error::Io)?;
        Ok(digest)
    }

    pub fn read(source: &Path) -> Result<Self, ReplaySnapshotV1Error> {
        let bytes = fs::read(source).map_err(ReplaySnapshotV1Error::Io)?;
        let text = core::str::from_utf8(&bytes).map_err(|_| ReplaySnapshotV1Error::InvalidUtf8)?;
        let (shards, rule_profile, _) = decode_snapshot(text)?;
        let snapshot = Self::with_rules(shards, rule_profile)?;
        let (canonical, _) = encode_snapshot(&snapshot.shards, snapshot.rule_profile);
        if canonical.as_bytes() != bytes {
            return Err(ReplaySnapshotV1Error::NonCanonicalEncoding);
        }
        Ok(snapshot)
    }

    pub fn verify_directory(
        &self,
        directory: &Path,
    ) -> Result<ReplaySnapshotVerificationV1, ReplaySnapshotV1Error> {
        self.load_verified_shards(directory)?;
        Ok(ReplaySnapshotVerificationV1 {
            shard_count: self.shards.len(),
            game_count: self.game_count()?,
            example_count: self.example_count()?,
            digest: self.digest(),
        })
    }

    /// Verifies that every immutable shard still has the exact digest sealed by
    /// this snapshot without decoding and replaying all of its games.
    pub fn verify_directory_integrity(
        &self,
        directory: &Path,
    ) -> Result<ReplaySnapshotVerificationV1, ReplaySnapshotV1Error> {
        for reference in &self.shards {
            let path = directory.join(reference.file_name());
            let (digest, rule_profile) =
                ReplayShardV1::verify_file_digest(&path).map_err(|source| {
                    ReplaySnapshotV1Error::InvalidShard {
                        file_name: reference.file_name().to_owned(),
                        source,
                    }
                })?;
            require_shard_profile(reference.file_name(), self.rule_profile, rule_profile)?;
            if digest != reference.digest() {
                return Err(ReplaySnapshotV1Error::ShardDigestMismatch {
                    file_name: reference.file_name().to_owned(),
                });
            }
        }
        Ok(ReplaySnapshotVerificationV1 {
            shard_count: self.shards.len(),
            game_count: self.game_count()?,
            example_count: self.example_count()?,
            digest: self.digest(),
        })
    }

    pub(crate) fn load_verified_shards(
        &self,
        directory: &Path,
    ) -> Result<Vec<ReplayShardV1>, ReplaySnapshotV1Error> {
        let mut shards = Vec::with_capacity(self.shards.len());
        let mut game_owners = std::collections::HashMap::new();
        for reference in &self.shards {
            let path = directory.join(reference.file_name());
            let shard = ReplayShardV1::read(&path).map_err(|source| {
                ReplaySnapshotV1Error::InvalidShard {
                    file_name: reference.file_name().to_owned(),
                    source,
                }
            })?;
            if shard.shard_index() != reference.shard_index() {
                return Err(ReplaySnapshotV1Error::ShardIndexMismatch {
                    file_name: reference.file_name().to_owned(),
                    expected: reference.shard_index(),
                    actual: shard.shard_index(),
                });
            }
            require_shard_profile(
                reference.file_name(),
                self.rule_profile,
                shard.rule_profile(),
            )?;
            let digest = shard.digest()?;
            if digest != reference.digest() {
                return Err(ReplaySnapshotV1Error::ShardDigestMismatch {
                    file_name: reference.file_name().to_owned(),
                });
            }
            let game_count = shard.games().len() as u64;
            let example_count = u64::try_from(shard.training_example_count()?)
                .map_err(|_| ReplaySnapshotV1Error::CountOverflow("examples"))?;
            if game_count != reference.game_count() || example_count != reference.example_count() {
                return Err(ReplaySnapshotV1Error::ShardCountMismatch {
                    file_name: reference.file_name().to_owned(),
                    expected_games: reference.game_count(),
                    actual_games: game_count,
                    expected_examples: reference.example_count(),
                    actual_examples: example_count,
                });
            }
            for game in shard.games() {
                if let Some(first_file) =
                    game_owners.insert(game.game_id(), reference.file_name().to_owned())
                {
                    return Err(ReplaySnapshotV1Error::DuplicateGameId {
                        game_id: game.game_id(),
                        first_file,
                        second_file: reference.file_name().to_owned(),
                    });
                }
            }
            shards.push(shard);
        }
        Ok(shards)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplaySnapshotVerificationV1 {
    pub shard_count: usize,
    pub game_count: u64,
    pub example_count: u64,
    pub digest: ReplayDigestV1,
}

fn validate_file_name(file_name: &str) -> Result<(), ReplaySnapshotV1Error> {
    let mut components = Path::new(file_name).components();
    let is_one_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if file_name.is_empty()
        || !is_one_normal_component
        || file_name
            .chars()
            .any(|character| character.is_control() || character == '\t')
    {
        Err(ReplaySnapshotV1Error::InvalidFileName(file_name.to_owned()))
    } else {
        Ok(())
    }
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    field: &'static str,
) -> Result<u64, ReplaySnapshotV1Error> {
    values.into_iter().try_fold(0_u64, |sum, value| {
        sum.checked_add(value)
            .ok_or(ReplaySnapshotV1Error::CountOverflow(field))
    })
}

fn snapshot_body(shards: &[ReplayShardReferenceV1], rule_profile: RuleProfileId) -> String {
    use core::fmt::Write as _;

    let mut text = String::new();
    writeln!(text, "{MAGIC}").unwrap();
    writeln!(text, "encoding\t{REPLAY_ENCODING_SCHEMA_V1}").unwrap();
    writeln!(text, "rules\t{}", rule_profile.as_str()).unwrap();
    writeln!(text, "shards\t{}", shards.len()).unwrap();
    for shard in shards {
        writeln!(
            text,
            "shard\t{}\t{}\t{}\t{}\t{}",
            shard.shard_index, shard.file_name, shard.digest, shard.game_count, shard.example_count
        )
        .unwrap();
    }
    text
}

fn encode_snapshot(
    shards: &[ReplayShardReferenceV1],
    rule_profile: RuleProfileId,
) -> (String, ReplayDigestV1) {
    use core::fmt::Write as _;

    let mut text = snapshot_body(shards, rule_profile);
    let digest = digest_bytes(text.as_bytes());
    writeln!(text, "sha256\t{digest}").unwrap();
    (text, digest)
}

fn decode_snapshot(
    text: &str,
) -> Result<(Vec<ReplayShardReferenceV1>, RuleProfileId, ReplayDigestV1), ReplaySnapshotV1Error> {
    let without_final_newline = text
        .strip_suffix('\n')
        .ok_or(ReplaySnapshotV1Error::NonCanonicalEncoding)?;
    let checksum_start = without_final_newline
        .rfind('\n')
        .map(|index| index + 1)
        .ok_or(ReplaySnapshotV1Error::Truncated)?;
    let checksum_line = &without_final_newline[checksum_start..];
    let body = &text[..checksum_start];
    let stored_digest = checksum_line
        .strip_prefix("sha256\t")
        .ok_or(ReplaySnapshotV1Error::MissingChecksum)?
        .parse::<ReplayDigestV1>()
        .map_err(ReplaySnapshotV1Error::Digest)?;
    let actual_digest = digest_bytes(body.as_bytes());
    if stored_digest != actual_digest {
        return Err(ReplaySnapshotV1Error::ChecksumMismatch);
    }

    let mut lines = body.lines();
    require_line(lines.next(), MAGIC, "signature")?;
    require_line(
        lines.next(),
        &format!("encoding\t{REPLAY_ENCODING_SCHEMA_V1}"),
        "encoding",
    )?;
    let rule_profile = lines
        .next()
        .and_then(|line| line.strip_prefix("rules\t"))
        .ok_or(ReplaySnapshotV1Error::InvalidHeader("rules"))?
        .parse::<RuleProfileId>()
        .map_err(|_| ReplaySnapshotV1Error::InvalidHeader("rules"))?;
    let shard_count = parse_header_count(lines.next(), "shards")?;
    let mut shards = Vec::with_capacity(shard_count);
    for _ in 0..shard_count {
        shards.push(parse_shard_line(
            lines.next().ok_or(ReplaySnapshotV1Error::Truncated)?,
            rule_profile,
        )?);
    }
    if lines.next().is_some() {
        return Err(ReplaySnapshotV1Error::TrailingLines);
    }
    Ok((shards, rule_profile, actual_digest))
}

fn require_line(
    actual: Option<&str>,
    expected: &str,
    field: &'static str,
) -> Result<(), ReplaySnapshotV1Error> {
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(ReplaySnapshotV1Error::InvalidHeader(field))
    }
}

fn parse_header_count(
    line: Option<&str>,
    field: &'static str,
) -> Result<usize, ReplaySnapshotV1Error> {
    line.and_then(|line| line.strip_prefix(&format!("{field}\t")))
        .ok_or(ReplaySnapshotV1Error::InvalidHeader(field))?
        .parse()
        .map_err(|_| ReplaySnapshotV1Error::InvalidHeader(field))
}

fn parse_shard_line(
    line: &str,
    rule_profile: RuleProfileId,
) -> Result<ReplayShardReferenceV1, ReplaySnapshotV1Error> {
    let fields = line.split('\t').collect::<Vec<_>>();
    if fields.len() != 6 || fields[0] != "shard" {
        return Err(ReplaySnapshotV1Error::InvalidShardLine);
    }
    ReplayShardReferenceV1::with_rules(
        fields[2],
        parse_u64(fields[1])?,
        ReplayDigestV1::from_str(fields[3]).map_err(ReplaySnapshotV1Error::Digest)?,
        parse_u64(fields[4])?,
        parse_u64(fields[5])?,
        rule_profile,
    )
}

fn require_shard_profile(
    file_name: &str,
    expected: RuleProfileId,
    actual: RuleProfileId,
) -> Result<(), ReplaySnapshotV1Error> {
    if actual != expected {
        return Err(ReplaySnapshotV1Error::RuleProfileMismatch {
            file_name: file_name.to_owned(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn parse_u64(value: &str) -> Result<u64, ReplaySnapshotV1Error> {
    value
        .parse()
        .map_err(|_| ReplaySnapshotV1Error::InvalidShardLine)
}

fn digest_bytes(bytes: &[u8]) -> ReplayDigestV1 {
    ReplayDigestV1::from_bytes(Sha256::digest(bytes).into())
}

#[derive(Debug)]
pub enum ReplaySnapshotV1Error {
    InvalidFileName(String),
    DuplicateShardIndex(u64),
    DuplicateFileName(String),
    CountOverflow(&'static str),
    DestinationExists(PathBuf),
    InvalidUtf8,
    Truncated,
    MissingChecksum,
    ChecksumMismatch,
    NonCanonicalEncoding,
    InvalidHeader(&'static str),
    InvalidShardLine,
    TrailingLines,
    Digest(ReplayDigestV1Error),
    Shard(ReplayShardV1Error),
    InvalidShard {
        file_name: String,
        source: ReplayShardV1Error,
    },
    ShardIndexMismatch {
        file_name: String,
        expected: u64,
        actual: u64,
    },
    ShardDigestMismatch {
        file_name: String,
    },
    RuleProfileMismatch {
        file_name: String,
        expected: RuleProfileId,
        actual: RuleProfileId,
    },
    ShardCountMismatch {
        file_name: String,
        expected_games: u64,
        actual_games: u64,
        expected_examples: u64,
        actual_examples: u64,
    },
    DuplicateGameId {
        game_id: u64,
        first_file: String,
        second_file: String,
    },
    Io(io::Error),
}

impl fmt::Display for ReplaySnapshotV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFileName(name) => write!(formatter, "invalid replay shard file name {name}"),
            Self::DuplicateShardIndex(index) => {
                write!(formatter, "replay snapshot repeats shard index {index}")
            }
            Self::DuplicateFileName(name) => {
                write!(formatter, "replay snapshot repeats file name {name}")
            }
            Self::CountOverflow(field) => write!(formatter, "replay snapshot {field} overflow"),
            Self::DestinationExists(path) => {
                write!(formatter, "replay snapshot {} already exists", path.display())
            }
            Self::InvalidUtf8 => formatter.write_str("replay snapshot is not UTF-8"),
            Self::Truncated => formatter.write_str("truncated replay snapshot"),
            Self::MissingChecksum => formatter.write_str("replay snapshot checksum is missing"),
            Self::ChecksumMismatch => formatter.write_str("replay snapshot SHA-256 mismatch"),
            Self::NonCanonicalEncoding => {
                formatter.write_str("replay snapshot is not canonically encoded")
            }
            Self::InvalidHeader(field) => write!(formatter, "invalid replay snapshot {field}"),
            Self::InvalidShardLine => formatter.write_str("invalid replay snapshot shard line"),
            Self::TrailingLines => formatter.write_str("replay snapshot contains trailing lines"),
            Self::Digest(source) => source.fmt(formatter),
            Self::Shard(source) => source.fmt(formatter),
            Self::InvalidShard { file_name, source } => {
                write!(formatter, "invalid replay shard {file_name}: {source}")
            }
            Self::ShardIndexMismatch {
                file_name,
                expected,
                actual,
            } => write!(
                formatter,
                "replay shard {file_name} has index {actual}; expected {expected}"
            ),
            Self::ShardDigestMismatch { file_name } => {
                write!(formatter, "replay shard {file_name} has the wrong digest")
            }
            Self::RuleProfileMismatch { file_name, expected, actual } => write!(
                formatter, "replay shard {file_name} uses rule profile {actual}; snapshot declares {expected}"
            ),
            Self::ShardCountMismatch {
                file_name,
                expected_games,
                actual_games,
                expected_examples,
                actual_examples,
            } => write!(
                formatter,
                "replay shard {file_name} has {actual_games}/{actual_examples} games/examples; expected {expected_games}/{expected_examples}"
            ),
            Self::DuplicateGameId {
                game_id,
                first_file,
                second_file,
            } => write!(
                formatter,
                "replay game {game_id} occurs in both {first_file} and {second_file}"
            ),
            Self::Io(source) => write!(formatter, "replay snapshot I/O failed: {source}"),
        }
    }
}

impl std::error::Error for ReplaySnapshotV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Digest(source) => Some(source),
            Self::Shard(source) => Some(source),
            Self::InvalidShard { source, .. } => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

impl From<ReplayShardV1Error> for ReplaySnapshotV1Error {
    fn from(source: ReplayShardV1Error) -> Self {
        Self::Shard(source)
    }
}
