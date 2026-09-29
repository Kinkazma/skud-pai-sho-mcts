use core::fmt;
use core::str::FromStr;

use paisho_core::{GameRecord, RecordParseError, RuleProfileId};
use paisho_model::{ActionEncodingError, ActionEncodingV1};
use sha2::{Digest, Sha256};

use crate::{
    PolicyEntryV1, PolicyTargetKindV1, PolicyTargetV1, PolicyTargetV1Error, ReplayDecisionV1,
    ReplayDigestV1, ReplayGameV1, ReplayValidationError, REPLAY_ENCODING_SCHEMA_V1,
};

const MAGIC: [u8; 8] = *b"PSRPLY01";
pub(crate) const LEGACY_FORMAT_VERSION: u32 = 1;
pub(crate) const CURRENT_FORMAT_VERSION: u32 = 2;
const DIGEST_BYTES: usize = 32;

pub(crate) fn encode(
    format_version: u32,
    shard_index: u64,
    games: &[ReplayGameV1],
) -> Result<(Vec<u8>, ReplayDigestV1), CodecError> {
    if !matches!(
        format_version,
        LEGACY_FORMAT_VERSION | CURRENT_FORMAT_VERSION
    ) {
        return Err(CodecError::UnsupportedVersion(format_version));
    }
    let rule_profile = games
        .first()
        .map_or(RuleProfileId::CURRENT, |game| game.record().rules());
    for game in games {
        require_rule_profile(game, rule_profile)?;
    }
    let mut writer = BinaryWriter::new();
    writer.bytes(&MAGIC);
    writer.u32(format_version);
    writer.string(REPLAY_ENCODING_SCHEMA_V1, "encoding schema")?;
    writer.string(rule_profile.as_str(), "rule profile")?;
    writer.u64(shard_index);
    writer.count(games.len(), "game count")?;
    for game in games {
        writer.u64(game.game_id());
        writer.bytes(game.host_agent().as_bytes());
        writer.bytes(game.guest_agent().as_bytes());
        writer.string(&game.record().to_string(), "game record")?;
        writer.count(game.decisions().len(), "decision count")?;
        for decision in game.decisions() {
            writer.index(decision.decision_index(), "decision index")?;
            writer.u8(decision.policy().kind().code());
            writer.bytes(decision.policy().producer().as_bytes());
            writer.count(decision.policy().entries().len(), "policy entry count")?;
            for entry in decision.policy().entries() {
                for slot in entry.action().slots() {
                    writer.u16(slot);
                }
                writer.u32(entry.probability().to_bits());
            }
            if format_version >= CURRENT_FORMAT_VERSION {
                match decision.behavior_value() {
                    Some(value) => {
                        writer.u8(1);
                        writer.u32(value.to_bits());
                    }
                    None => writer.u8(0),
                }
            } else if decision.behavior_value().is_some() {
                return Err(CodecError::BehaviorValueUnsupportedByVersion(
                    format_version,
                ));
            }
        }
    }
    let mut bytes = writer.finish();
    let digest = sha256(&bytes);
    bytes.extend_from_slice(digest.as_bytes());
    Ok((bytes, digest))
}

pub(crate) fn decode(
    bytes: &[u8],
) -> Result<(u32, u64, Vec<ReplayGameV1>, ReplayDigestV1), CodecError> {
    if bytes.len() < DIGEST_BYTES {
        return Err(CodecError::Truncated);
    }
    let (content, stored_digest) = bytes.split_at(bytes.len() - DIGEST_BYTES);
    let actual_digest = sha256(content);
    if stored_digest != actual_digest.as_bytes() {
        return Err(CodecError::ChecksumMismatch);
    }
    let mut reader = BinaryReader::new(content);
    let (version, rule_profile) = decode_header(&mut reader)?;
    let shard_index = reader.u64()?;
    let game_count = reader.u32()? as usize;
    let mut games = Vec::with_capacity(game_count);
    for _ in 0..game_count {
        let game = decode_game(&mut reader, version)?;
        require_rule_profile(&game, rule_profile)?;
        games.push(game);
    }
    if !reader.is_empty() {
        return Err(CodecError::TrailingBytes(reader.remaining()));
    }
    Ok((version, shard_index, games, actual_digest))
}

pub(crate) fn header_rule_profile(bytes: &[u8]) -> Result<RuleProfileId, CodecError> {
    decode_header(&mut BinaryReader::new(bytes)).map(|(_, profile)| profile)
}

fn decode_header(reader: &mut BinaryReader<'_>) -> Result<(u32, RuleProfileId), CodecError> {
    let magic = reader.array::<8>()?;
    if magic != MAGIC {
        return Err(CodecError::InvalidMagic(magic));
    }
    let version = reader.u32()?;
    if !matches!(version, LEGACY_FORMAT_VERSION | CURRENT_FORMAT_VERSION) {
        return Err(CodecError::UnsupportedVersion(version));
    }
    let schema = reader.string()?;
    if schema != REPLAY_ENCODING_SCHEMA_V1 {
        return Err(CodecError::UnsupportedEncodingSchema(schema));
    }
    let rule_profile_text = reader.string()?;
    let rule_profile = RuleProfileId::from_str(&rule_profile_text)
        .map_err(|_| CodecError::UnsupportedRuleProfile(rule_profile_text))?;
    Ok((version, rule_profile))
}

fn require_rule_profile(game: &ReplayGameV1, expected: RuleProfileId) -> Result<(), CodecError> {
    let actual = game.record().rules();
    if actual != expected {
        return Err(CodecError::RuleProfileMismatch {
            game_id: game.game_id(),
            expected,
            actual,
        });
    }
    Ok(())
}

fn decode_game(
    reader: &mut BinaryReader<'_>,
    format_version: u32,
) -> Result<ReplayGameV1, CodecError> {
    let game_id = reader.u64()?;
    let host_agent = ReplayDigestV1::from_bytes(reader.array()?);
    let guest_agent = ReplayDigestV1::from_bytes(reader.array()?);
    let record = reader
        .string()?
        .parse::<GameRecord>()
        .map_err(CodecError::Record)?;
    let decision_count = reader.u32()? as usize;
    let mut decisions = Vec::with_capacity(decision_count);
    for _ in 0..decision_count {
        let decision_index = reader.u32()? as usize;
        let code = reader.u8()?;
        let kind =
            PolicyTargetKindV1::from_code(code).ok_or(CodecError::UnknownPolicyTargetKind(code))?;
        let producer = ReplayDigestV1::from_bytes(reader.array()?);
        let entry_count = reader.u32()? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let action = ActionEncodingV1::from_slots([
                reader.u16()?,
                reader.u16()?,
                reader.u16()?,
                reader.u16()?,
            ])
            .map_err(CodecError::Action)?;
            let probability = f32::from_bits(reader.u32()?);
            entries.push(PolicyEntryV1::new(action, probability).map_err(CodecError::Policy)?);
        }
        let policy = PolicyTargetV1::new(kind, producer, entries).map_err(CodecError::Policy)?;
        let mut decision = ReplayDecisionV1::new(decision_index, policy);
        if format_version >= CURRENT_FORMAT_VERSION {
            decision = match reader.u8()? {
                0 => decision,
                1 => decision.with_behavior_value(f32::from_bits(reader.u32()?)),
                code => return Err(CodecError::InvalidOptionalBehaviorValue(code)),
            };
        }
        decisions.push(decision);
    }
    ReplayGameV1::new(game_id, host_agent, guest_agent, record, decisions)
        .map_err(CodecError::Validation)
}

fn sha256(bytes: &[u8]) -> ReplayDigestV1 {
    ReplayDigestV1::from_bytes(Sha256::digest(bytes).into())
}

struct BinaryWriter {
    bytes: Vec<u8>,
}

impl BinaryWriter {
    fn new() -> Self {
        Self { bytes: Vec::new() }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn bytes(&mut self, values: &[u8]) {
        self.bytes.extend_from_slice(values);
    }

    fn count(&mut self, value: usize, field: &'static str) -> Result<(), CodecError> {
        self.u32(u32::try_from(value).map_err(|_| CodecError::FieldTooLarge(field))?);
        Ok(())
    }

    fn index(&mut self, value: usize, field: &'static str) -> Result<(), CodecError> {
        self.count(value, field)
    }

    fn string(&mut self, value: &str, field: &'static str) -> Result<(), CodecError> {
        self.count(value.len(), field)?;
        self.bytes(value.as_bytes());
        Ok(())
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> BinaryReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        Ok(self.array::<1>()?[0])
    }

    fn u16(&mut self) -> Result<u16, CodecError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CodecError> {
        let end = self.cursor.checked_add(N).ok_or(CodecError::Truncated)?;
        let slice = self
            .bytes
            .get(self.cursor..end)
            .ok_or(CodecError::Truncated)?;
        self.cursor = end;
        Ok(slice.try_into().expect("slice length was checked"))
    }

    fn string(&mut self) -> Result<String, CodecError> {
        let length = self.u32()? as usize;
        let end = self
            .cursor
            .checked_add(length)
            .ok_or(CodecError::Truncated)?;
        let bytes = self
            .bytes
            .get(self.cursor..end)
            .ok_or(CodecError::Truncated)?;
        self.cursor = end;
        String::from_utf8(bytes.to_vec()).map_err(|_| CodecError::InvalidUtf8)
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.bytes.len()
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.cursor
    }
}

#[derive(Debug)]
pub enum CodecError {
    Truncated,
    ChecksumMismatch,
    InvalidMagic([u8; 8]),
    UnsupportedVersion(u32),
    UnsupportedEncodingSchema(String),
    UnsupportedRuleProfile(String),
    RuleProfileMismatch {
        game_id: u64,
        expected: RuleProfileId,
        actual: RuleProfileId,
    },
    InvalidUtf8,
    TrailingBytes(usize),
    FieldTooLarge(&'static str),
    BehaviorValueUnsupportedByVersion(u32),
    InvalidOptionalBehaviorValue(u8),
    UnknownPolicyTargetKind(u8),
    Record(RecordParseError),
    Action(ActionEncodingError),
    Policy(PolicyTargetV1Error),
    Validation(ReplayValidationError),
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => formatter.write_str("truncated replay shard"),
            Self::ChecksumMismatch => formatter.write_str("replay shard SHA-256 mismatch"),
            Self::InvalidMagic(magic) => write!(formatter, "invalid replay shard magic {magic:?}"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported replay shard version {version}")
            }
            Self::UnsupportedEncodingSchema(schema) => {
                write!(formatter, "unsupported replay encoding schema {schema}")
            }
            Self::UnsupportedRuleProfile(profile) => {
                write!(formatter, "unsupported replay rule profile {profile}")
            }
            Self::RuleProfileMismatch {
                game_id,
                expected,
                actual,
            } => write!(
                formatter,
                "replay game {game_id} uses rule profile {actual}; shard declares {expected}"
            ),
            Self::InvalidUtf8 => formatter.write_str("replay shard contains invalid UTF-8"),
            Self::TrailingBytes(count) => {
                write!(formatter, "replay shard contains {count} trailing bytes")
            }
            Self::FieldTooLarge(field) => write!(formatter, "replay {field} exceeds V1"),
            Self::BehaviorValueUnsupportedByVersion(version) => write!(
                formatter,
                "replay shard version {version} cannot store behavior values"
            ),
            Self::InvalidOptionalBehaviorValue(code) => write!(
                formatter,
                "invalid optional behavior-value marker {code} in replay shard"
            ),
            Self::UnknownPolicyTargetKind(code) => {
                write!(formatter, "unknown replay policy target kind {code}")
            }
            Self::Record(source) => write!(formatter, "invalid game record: {source}"),
            Self::Action(source) => write!(formatter, "invalid policy action: {source}"),
            Self::Policy(source) => source.fmt(formatter),
            Self::Validation(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for CodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Record(source) => Some(source),
            Self::Action(source) => Some(source),
            Self::Policy(source) => Some(source),
            Self::Validation(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use paisho_core::GameRecord;
    use paisho_model::encode_action_v1;

    use super::*;

    const TERMINAL_RING: &str =
        include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");

    #[test]
    fn legacy_v1_encoding_remains_canonical_and_readable() {
        let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
        let position = record.initial_position();
        let agent = ReplayDigestV1::from_bytes([31; 32]);
        let action = encode_action_v1(record.actions()[0], position.to_move()).unwrap();
        let policy = PolicyTargetV1::one_hot(agent, action).unwrap();
        let game = ReplayGameV1::new(
            91,
            agent,
            agent,
            record,
            vec![ReplayDecisionV1::new(0, policy)],
        )
        .unwrap();

        let (bytes, digest) = encode(LEGACY_FORMAT_VERSION, 7, &[game.clone()]).unwrap();
        assert_eq!(
            digest.to_string(),
            "61e62f845a6a80d2dc185b75f96d6ce14a2b0a2b1dcb7e4e7ae7050262271e2c"
        );
        let (version, index, games, decoded_digest) = decode(&bytes).unwrap();
        assert_eq!(version, LEGACY_FORMAT_VERSION);
        assert_eq!(index, 7);
        assert_eq!(games, vec![game]);
        assert_eq!(decoded_digest, digest);
        assert_eq!(encode(version, index, &games).unwrap().0, bytes);
    }
}
