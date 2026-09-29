use core::fmt;
use std::collections::HashSet;

pub const CHECKPOINT_REQUEST_MAGIC_V1: [u8; 8] = *b"PSCREQ01";
pub const CHECKPOINT_RESPONSE_MAGIC_V1: [u8; 8] = *b"PSCRSP01";
pub const CHECKPOINT_ERROR_MAGIC_V1: [u8; 8] = *b"PSCERR01";
pub const MAXIMUM_CHECKPOINT_REQUEST_PAYLOAD_V1: usize = 65_536;

const MAXIMUM_RANDOM_STATES_V1: usize = 1_024;
const MAXIMUM_RANDOM_STATE_NAME_BYTES_V1: usize = 4_096;
const MAXIMUM_DESTINATION_BYTES_V1: usize = 16_384;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointRandomStateV1 {
    name: String,
    state: u64,
}

impl CheckpointRandomStateV1 {
    pub fn new(name: impl Into<String>, state: u64) -> Result<Self, CheckpointWireError> {
        let name = name.into();
        if name.is_empty() {
            return Err(CheckpointWireError::EmptyRandomStateName);
        }
        if name.as_bytes().len() > MAXIMUM_RANDOM_STATE_NAME_BYTES_V1
            || u32::try_from(name.as_bytes().len()).is_err()
        {
            return Err(CheckpointWireError::RandomStateNameTooLong(
                name.as_bytes().len(),
            ));
        }
        Ok(Self { name, state })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn state(&self) -> u64 {
        self.state
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointRequestV1 {
    request_id: u64,
    expected_training_step: u64,
    replay_snapshot_sha256: [u8; 32],
    replay_index: u64,
    generation: u64,
    learning_rate: f32,
    random_states: Vec<CheckpointRandomStateV1>,
    destination: String,
}

impl CheckpointRequestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: u64,
        expected_training_step: u64,
        replay_snapshot_sha256: [u8; 32],
        replay_index: u64,
        generation: u64,
        learning_rate: f32,
        random_states: Vec<CheckpointRandomStateV1>,
        destination: impl Into<String>,
    ) -> Result<Self, CheckpointWireError> {
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(CheckpointWireError::InvalidLearningRate(learning_rate));
        }
        if random_states.is_empty() {
            return Err(CheckpointWireError::MissingRandomState);
        }
        if random_states.len() > MAXIMUM_RANDOM_STATES_V1
            || u32::try_from(random_states.len()).is_err()
        {
            return Err(CheckpointWireError::TooManyRandomStates(
                random_states.len(),
            ));
        }
        let mut names = HashSet::with_capacity(random_states.len());
        for random_state in &random_states {
            if !names.insert(random_state.name()) {
                return Err(CheckpointWireError::DuplicateRandomState(
                    random_state.name().to_owned(),
                ));
            }
        }
        let destination = destination.into();
        if destination.is_empty() || destination.contains('\0') {
            return Err(CheckpointWireError::InvalidDestination);
        }
        if destination.as_bytes().len() > MAXIMUM_DESTINATION_BYTES_V1
            || u32::try_from(destination.as_bytes().len()).is_err()
        {
            return Err(CheckpointWireError::DestinationTooLong(
                destination.as_bytes().len(),
            ));
        }
        let request = Self {
            request_id,
            expected_training_step,
            replay_snapshot_sha256,
            replay_index,
            generation,
            learning_rate,
            random_states,
            destination,
        };
        let payload_size = request.encoded_payload_size()?;
        if payload_size > MAXIMUM_CHECKPOINT_REQUEST_PAYLOAD_V1 {
            return Err(CheckpointWireError::PayloadExceedsMaximum(payload_size));
        }
        Ok(request)
    }

    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    pub const fn expected_training_step(&self) -> u64 {
        self.expected_training_step
    }

    pub const fn replay_snapshot_sha256(&self) -> &[u8; 32] {
        &self.replay_snapshot_sha256
    }

    pub const fn replay_index(&self) -> u64 {
        self.replay_index
    }

    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn random_states(&self) -> &[CheckpointRandomStateV1] {
        &self.random_states
    }

    pub fn destination(&self) -> &str {
        &self.destination
    }

    pub fn encode_payload(&self) -> Result<Vec<u8>, CheckpointWireError> {
        let mut writer = WireWriter::with_capacity(self.encoded_payload_size()?);
        writer.bytes(&CHECKPOINT_REQUEST_MAGIC_V1);
        writer.u64(self.request_id);
        writer.u64(self.expected_training_step);
        writer.bytes(&self.replay_snapshot_sha256);
        writer.u64(self.replay_index);
        writer.u64(self.generation);
        writer.f32(self.learning_rate);
        writer.u32(self.random_states.len() as u32);
        for random_state in &self.random_states {
            writer.u32(random_state.name.as_bytes().len() as u32);
            writer.bytes(random_state.name.as_bytes());
            writer.u64(random_state.state);
        }
        writer.u32(self.destination.as_bytes().len() as u32);
        writer.bytes(self.destination.as_bytes());
        Ok(writer.finish())
    }

    fn encoded_payload_size(&self) -> Result<usize, CheckpointWireError> {
        let fixed = 8usize + 8 + 8 + 32 + 8 + 8 + 4 + 4 + 4;
        let random_states = self.random_states.iter().try_fold(0usize, |total, state| {
            total
                .checked_add(4)
                .and_then(|value| value.checked_add(state.name.as_bytes().len()))
                .and_then(|value| value.checked_add(8))
                .ok_or(CheckpointWireError::PayloadTooLarge)
        })?;
        fixed
            .checked_add(random_states)
            .and_then(|value| value.checked_add(self.destination.as_bytes().len()))
            .ok_or(CheckpointWireError::PayloadTooLarge)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointResponseV1 {
    request_id: u64,
    completed_training_step: u64,
    completed_replay_index: u64,
    content_sha256: [u8; 32],
}

impl CheckpointResponseV1 {
    pub fn decode_payload(
        payload: &[u8],
        request: &CheckpointRequestV1,
    ) -> Result<Self, CheckpointWireError> {
        let mut reader = WireReader::new(payload);
        let magic = reader.array::<8>()?;
        if magic == CHECKPOINT_ERROR_MAGIC_V1 {
            return decode_service_error(reader, request.request_id);
        }
        if magic != CHECKPOINT_RESPONSE_MAGIC_V1 {
            return Err(CheckpointWireError::InvalidResponseMagic(magic));
        }
        let request_id = reader.u64()?;
        require_request_id(request.request_id, request_id)?;
        let completed_training_step = reader.u64()?;
        if completed_training_step != request.expected_training_step {
            return Err(CheckpointWireError::ResponseTrainingStepMismatch {
                expected: request.expected_training_step,
                actual: completed_training_step,
            });
        }
        let completed_replay_index = reader.u64()?;
        if completed_replay_index != request.replay_index {
            return Err(CheckpointWireError::ResponseReplayIndexMismatch {
                expected: request.replay_index,
                actual: completed_replay_index,
            });
        }
        let content_sha256 = reader.array()?;
        reader.finish()?;
        Ok(Self {
            request_id,
            completed_training_step,
            completed_replay_index,
            content_sha256,
        })
    }

    pub const fn request_id(self) -> u64 {
        self.request_id
    }

    pub const fn completed_training_step(self) -> u64 {
        self.completed_training_step
    }

    pub const fn completed_replay_index(self) -> u64 {
        self.completed_replay_index
    }

    pub const fn content_sha256(self) -> [u8; 32] {
        self.content_sha256
    }
}

fn require_request_id(expected: u64, actual: u64) -> Result<(), CheckpointWireError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CheckpointWireError::ResponseRequestIdMismatch { expected, actual })
    }
}

fn decode_service_error(
    mut reader: WireReader<'_>,
    expected_request_id: u64,
) -> Result<CheckpointResponseV1, CheckpointWireError> {
    let request_id = reader.u64()?;
    require_request_id(expected_request_id, request_id)?;
    let length = reader.u32()? as usize;
    let message = String::from_utf8(reader.bytes(length)?.to_vec())
        .map_err(|_| CheckpointWireError::InvalidErrorMessage)?;
    reader.finish()?;
    Err(CheckpointWireError::Service(message))
}

struct WireWriter {
    bytes: Vec<u8>,
}

impl WireWriter {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(capacity),
        }
    }

    fn bytes(&mut self, values: &[u8]) {
        self.bytes.extend_from_slice(values);
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    fn finish(self) -> Vec<u8> {
        self.bytes
    }
}

struct WireReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> WireReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], CheckpointWireError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(CheckpointWireError::TruncatedResponse)?;
        let values = self
            .bytes
            .get(self.cursor..end)
            .ok_or(CheckpointWireError::TruncatedResponse)?;
        self.cursor = end;
        Ok(values)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CheckpointWireError> {
        self.bytes(N)?
            .try_into()
            .map_err(|_| CheckpointWireError::TruncatedResponse)
    }

    fn u32(&mut self) -> Result<u32, CheckpointWireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, CheckpointWireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn finish(self) -> Result<(), CheckpointWireError> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(CheckpointWireError::TrailingResponseBytes(
                self.bytes.len() - self.cursor,
            ))
        }
    }
}

#[derive(Debug)]
pub enum CheckpointWireError {
    EmptyRandomStateName,
    RandomStateNameTooLong(usize),
    MissingRandomState,
    TooManyRandomStates(usize),
    DuplicateRandomState(String),
    InvalidLearningRate(f32),
    InvalidDestination,
    DestinationTooLong(usize),
    PayloadTooLarge,
    PayloadExceedsMaximum(usize),
    TruncatedResponse,
    TrailingResponseBytes(usize),
    InvalidResponseMagic([u8; 8]),
    ResponseRequestIdMismatch { expected: u64, actual: u64 },
    ResponseTrainingStepMismatch { expected: u64, actual: u64 },
    ResponseReplayIndexMismatch { expected: u64, actual: u64 },
    InvalidErrorMessage,
    Service(String),
}

impl fmt::Display for CheckpointWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRandomStateName => formatter.write_str("checkpoint random-state name is empty"),
            Self::RandomStateNameTooLong(length) => write!(
                formatter,
                "checkpoint random-state name has {length} bytes; maximum is {MAXIMUM_RANDOM_STATE_NAME_BYTES_V1}"
            ),
            Self::MissingRandomState => {
                formatter.write_str("checkpoint requires at least one random state")
            }
            Self::TooManyRandomStates(count) => write!(
                formatter,
                "checkpoint has {count} random states; maximum is {MAXIMUM_RANDOM_STATES_V1}"
            ),
            Self::DuplicateRandomState(name) => {
                write!(formatter, "checkpoint repeats random state {name}")
            }
            Self::InvalidLearningRate(value) => write!(
                formatter,
                "checkpoint learning rate {value} is not positive and finite"
            ),
            Self::InvalidDestination => {
                formatter.write_str("checkpoint destination must be non-empty and contain no NUL")
            }
            Self::DestinationTooLong(length) => write!(
                formatter,
                "checkpoint destination has {length} bytes; maximum is {MAXIMUM_DESTINATION_BYTES_V1}"
            ),
            Self::PayloadTooLarge => formatter.write_str("checkpoint payload size overflow"),
            Self::PayloadExceedsMaximum(actual) => write!(
                formatter,
                "checkpoint payload has {actual} bytes; maximum is {MAXIMUM_CHECKPOINT_REQUEST_PAYLOAD_V1}"
            ),
            Self::TruncatedResponse => formatter.write_str("truncated checkpoint response"),
            Self::TrailingResponseBytes(count) => {
                write!(formatter, "checkpoint response has {count} trailing bytes")
            }
            Self::InvalidResponseMagic(value) => {
                write!(formatter, "invalid checkpoint response magic {value:?}")
            }
            Self::ResponseRequestIdMismatch { expected, actual } => write!(
                formatter,
                "checkpoint response request id {actual} does not match {expected}"
            ),
            Self::ResponseTrainingStepMismatch { expected, actual } => write!(
                formatter,
                "checkpoint response step {actual} does not match {expected}"
            ),
            Self::ResponseReplayIndexMismatch { expected, actual } => write!(
                formatter,
                "checkpoint response replay index {actual} does not match {expected}"
            ),
            Self::InvalidErrorMessage => {
                formatter.write_str("checkpoint service error is not UTF-8")
            }
            Self::Service(message) => write!(formatter, "checkpoint service failed: {message}"),
        }
    }
}

impl std::error::Error for CheckpointWireError {}
