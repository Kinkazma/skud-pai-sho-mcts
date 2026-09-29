use core::fmt;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Read, Seek};
use std::path::{Path, PathBuf};

use paisho_core::RuleProfileId;
use sha2::{Digest, Sha256};

use crate::atomic_file;
use crate::codec::{self, CodecError};
use crate::{ReplayDigestV1, ReplayGameV1, ReplayTrainingExampleV1, ReplayValidationError};

pub const REPLAY_ENCODING_SCHEMA_V1: &str = "paisho-neural-encoding-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct ReplayShardV1 {
    format_version: u32,
    shard_index: u64,
    games: Vec<ReplayGameV1>,
}

impl ReplayShardV1 {
    pub fn new(shard_index: u64, mut games: Vec<ReplayGameV1>) -> Result<Self, ReplayShardV1Error> {
        if games.is_empty() {
            return Err(ReplayShardV1Error::Empty);
        }
        games.sort_by_key(ReplayGameV1::game_id);
        let rule_profile = games[0].record().rules();
        let mut game_ids = HashSet::with_capacity(games.len());
        for game in &games {
            if game.record().rules() != rule_profile {
                return Err(ReplayShardV1Error::Codec(CodecError::RuleProfileMismatch {
                    game_id: game.game_id(),
                    expected: rule_profile,
                    actual: game.record().rules(),
                }));
            }
            if !game_ids.insert(game.game_id()) {
                return Err(ReplayShardV1Error::DuplicateGameId(game.game_id()));
            }
        }
        Ok(Self {
            format_version: codec::CURRENT_FORMAT_VERSION,
            shard_index,
            games,
        })
    }

    pub const fn shard_index(&self) -> u64 {
        self.shard_index
    }

    pub fn rule_profile(&self) -> RuleProfileId {
        self.games[0].record().rules()
    }

    pub fn games(&self) -> &[ReplayGameV1] {
        &self.games
    }

    pub fn training_examples(&self) -> Result<Vec<ReplayTrainingExampleV1>, ReplayShardV1Error> {
        let mut examples = Vec::new();
        for game in &self.games {
            examples.extend(
                game.materialize_training_examples()
                    .map_err(ReplayShardV1Error::Validation)?,
            );
        }
        Ok(examples)
    }

    pub fn training_example_count(&self) -> Result<usize, ReplayShardV1Error> {
        self.games.iter().try_fold(0_usize, |count, game| {
            count
                .checked_add(game.decisions().len())
                .ok_or(ReplayShardV1Error::ExampleCountOverflow)
        })
    }

    pub fn digest(&self) -> Result<ReplayDigestV1, ReplayShardV1Error> {
        codec::encode(self.format_version, self.shard_index, &self.games)
            .map(|(_, digest)| digest)
            .map_err(ReplayShardV1Error::Codec)
    }

    pub fn write_new(&self, destination: &Path) -> Result<ReplayDigestV1, ReplayShardV1Error> {
        if destination.exists() {
            return Err(ReplayShardV1Error::DestinationExists(
                destination.to_owned(),
            ));
        }
        let (bytes, digest) = codec::encode(self.format_version, self.shard_index, &self.games)
            .map_err(ReplayShardV1Error::Codec)?;
        atomic_file::write_new(destination, &bytes).map_err(ReplayShardV1Error::Io)?;
        Ok(digest)
    }

    pub fn read(source: &Path) -> Result<Self, ReplayShardV1Error> {
        let bytes = fs::read(source).map_err(ReplayShardV1Error::Io)?;
        let (format_version, shard_index, games, _) =
            codec::decode(&bytes).map_err(ReplayShardV1Error::Codec)?;
        let mut shard = Self::new(shard_index, games)?;
        shard.format_version = format_version;
        let (canonical, _) = codec::encode(shard.format_version, shard.shard_index, &shard.games)
            .map_err(ReplayShardV1Error::Codec)?;
        if canonical != bytes {
            return Err(ReplayShardV1Error::NonCanonicalEncoding);
        }
        Ok(shard)
    }

    pub(crate) fn verify_file_digest(
        source: &Path,
    ) -> Result<(ReplayDigestV1, RuleProfileId), ReplayShardV1Error> {
        const DIGEST_BYTES: u64 = 32;

        let mut file = File::open(source).map_err(ReplayShardV1Error::Io)?;
        let length = file.metadata().map_err(ReplayShardV1Error::Io)?.len();
        if length < DIGEST_BYTES {
            return Err(ReplayShardV1Error::Codec(CodecError::Truncated));
        }
        let mut remaining = length - DIGEST_BYTES;
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 64 * 1_024];
        while remaining > 0 {
            let wanted = usize::try_from(remaining.min(buffer.len() as u64))
                .expect("bounded replay read length fits usize");
            let read = file
                .read(&mut buffer[..wanted])
                .map_err(ReplayShardV1Error::Io)?;
            if read == 0 {
                return Err(ReplayShardV1Error::Codec(CodecError::Truncated));
            }
            hasher.update(&buffer[..read]);
            remaining -= read as u64;
        }
        let mut stored = [0_u8; DIGEST_BYTES as usize];
        file.read_exact(&mut stored)
            .map_err(ReplayShardV1Error::Io)?;
        let actual = ReplayDigestV1::from_bytes(hasher.finalize().into());
        if stored != *actual.as_bytes() {
            return Err(ReplayShardV1Error::Codec(CodecError::ChecksumMismatch));
        }
        let mut trailing = [0_u8; 1];
        if file.read(&mut trailing).map_err(ReplayShardV1Error::Io)? != 0 {
            return Err(ReplayShardV1Error::NonCanonicalEncoding);
        }
        file.rewind().map_err(ReplayShardV1Error::Io)?;
        // Both supported canonical headers fit well within this bounded prefix.
        // Integrity verification remains independent of replaying the games.
        let mut header = Vec::new();
        file.take(256)
            .read_to_end(&mut header)
            .map_err(ReplayShardV1Error::Io)?;
        let profile = codec::header_rule_profile(&header).map_err(ReplayShardV1Error::Codec)?;
        Ok((actual, profile))
    }
}

#[derive(Debug)]
pub enum ReplayShardV1Error {
    Empty,
    DuplicateGameId(u64),
    ExampleCountOverflow,
    DestinationExists(PathBuf),
    NonCanonicalEncoding,
    Validation(ReplayValidationError),
    Codec(CodecError),
    Io(io::Error),
}

impl fmt::Display for ReplayShardV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("a replay shard cannot be empty"),
            Self::DuplicateGameId(game_id) => {
                write!(formatter, "replay shard repeats game id {game_id}")
            }
            Self::ExampleCountOverflow => {
                formatter.write_str("replay shard training-example count overflow")
            }
            Self::DestinationExists(path) => {
                write!(formatter, "replay shard {} already exists", path.display())
            }
            Self::NonCanonicalEncoding => {
                formatter.write_str("replay shard is valid but not canonically encoded")
            }
            Self::Validation(source) => source.fmt(formatter),
            Self::Codec(source) => source.fmt(formatter),
            Self::Io(source) => write!(formatter, "replay shard I/O failed: {source}"),
        }
    }
}

impl std::error::Error for ReplayShardV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(source) => Some(source),
            Self::Codec(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

impl From<ReplayValidationError> for ReplayShardV1Error {
    fn from(source: ReplayValidationError) -> Self {
        Self::Validation(source)
    }
}
