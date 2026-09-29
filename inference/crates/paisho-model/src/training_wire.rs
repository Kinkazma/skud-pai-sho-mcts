use core::fmt;

use crate::{
    InferenceExampleV1, GLOBAL_FEATURE_COUNT_V1, SPATIAL_VALUE_COUNT_V1, VALUE_CLASS_COUNT_V1,
};

pub const TRAINING_REQUEST_MAGIC_V1: [u8; 8] = *b"PSTREQ01";
pub const TRAINING_RESPONSE_MAGIC_V1: [u8; 8] = *b"PSTRSP01";
pub const TRAINING_ERROR_MAGIC_V1: [u8; 8] = *b"PSTERR01";
const DISTRIBUTION_TOLERANCE: f32 = 1.0e-5;

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingExampleV1 {
    inference: InferenceExampleV1,
    policy_targets: Vec<f32>,
    value_targets: [f32; VALUE_CLASS_COUNT_V1],
}

impl TrainingExampleV1 {
    pub fn new(
        inference: InferenceExampleV1,
        policy_targets: Vec<f32>,
        value_targets: [f32; VALUE_CLASS_COUNT_V1],
    ) -> Result<Self, TrainingWireError> {
        if policy_targets.len() != inference.legal_actions().len() {
            return Err(TrainingWireError::PolicyTargetCountMismatch {
                expected: inference.legal_actions().len(),
                actual: policy_targets.len(),
            });
        }
        validate_distribution("policy", &policy_targets)?;
        validate_distribution("value", &value_targets)?;
        Ok(Self {
            inference,
            policy_targets,
            value_targets,
        })
    }

    pub const fn inference(&self) -> &InferenceExampleV1 {
        &self.inference
    }

    pub fn policy_targets(&self) -> &[f32] {
        &self.policy_targets
    }

    pub const fn value_targets(&self) -> &[f32; VALUE_CLASS_COUNT_V1] {
        &self.value_targets
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingRequestV1 {
    request_id: u64,
    expected_training_step: u64,
    learning_rate: f32,
    legal_action_capacity: usize,
    replay_snapshot_sha256: [u8; 32],
    start_replay_index: u64,
    next_replay_index: u64,
    examples: Vec<TrainingExampleV1>,
}

impl TrainingRequestV1 {
    pub fn new(
        request_id: u64,
        expected_training_step: u64,
        learning_rate: f32,
        legal_action_capacity: usize,
        replay_snapshot_sha256: [u8; 32],
        start_replay_index: u64,
        examples: Vec<TrainingExampleV1>,
    ) -> Result<Self, TrainingWireError> {
        if examples.is_empty() {
            return Err(TrainingWireError::EmptyBatch);
        }
        if expected_training_step == u64::MAX {
            return Err(TrainingWireError::TrainingStepOverflow);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(TrainingWireError::InvalidLearningRate(learning_rate));
        }
        if legal_action_capacity == 0 {
            return Err(TrainingWireError::ZeroActionCapacity);
        }
        u32::try_from(examples.len())
            .map_err(|_| TrainingWireError::BatchTooLarge(examples.len()))?;
        u32::try_from(legal_action_capacity)
            .map_err(|_| TrainingWireError::ActionCapacityTooLarge(legal_action_capacity))?;
        let next_replay_index = start_replay_index
            .checked_add(examples.len() as u64)
            .ok_or(TrainingWireError::ReplayIndexOverflow)?;
        for (row, example) in examples.iter().enumerate() {
            if example.inference.legal_actions().len() > legal_action_capacity {
                return Err(TrainingWireError::TooManyLegalActions {
                    row,
                    capacity: legal_action_capacity,
                    actual: example.inference.legal_actions().len(),
                });
            }
        }
        Ok(Self {
            request_id,
            expected_training_step,
            learning_rate,
            legal_action_capacity,
            replay_snapshot_sha256,
            start_replay_index,
            next_replay_index,
            examples,
        })
    }

    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    pub const fn expected_training_step(&self) -> u64 {
        self.expected_training_step
    }

    pub const fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn batch_size(&self) -> usize {
        self.examples.len()
    }

    pub const fn legal_action_capacity(&self) -> usize {
        self.legal_action_capacity
    }

    pub const fn replay_snapshot_sha256(&self) -> &[u8; 32] {
        &self.replay_snapshot_sha256
    }

    pub const fn start_replay_index(&self) -> u64 {
        self.start_replay_index
    }

    pub const fn next_replay_index(&self) -> u64 {
        self.next_replay_index
    }

    pub fn examples(&self) -> &[TrainingExampleV1] {
        &self.examples
    }

    pub fn encode_payload(&self) -> Result<Vec<u8>, TrainingWireError> {
        let mut writer = WireWriter::with_capacity(self.encoded_payload_size()?);
        writer.bytes(&TRAINING_REQUEST_MAGIC_V1);
        writer.u64(self.request_id);
        writer.u64(self.expected_training_step);
        writer.f32(self.learning_rate);
        writer.u32(self.batch_size() as u32);
        writer.u32(self.legal_action_capacity as u32);
        writer.bytes(&self.replay_snapshot_sha256);
        writer.u64(self.start_replay_index);
        writer.u64(self.next_replay_index);
        for example in &self.examples {
            writer.f32_slice(example.inference.state().spatial_nhwc());
            writer.f32_slice(example.inference.state().global());
            writer.u32(example.inference.legal_actions().len() as u32);
            for action in example.inference.legal_actions() {
                for slot in action.slots() {
                    writer.u16(slot);
                }
            }
            writer.f32_slice(&example.policy_targets);
            writer.f32_slice(&example.value_targets);
        }
        Ok(writer.finish())
    }

    fn encoded_payload_size(&self) -> Result<usize, TrainingWireError> {
        let fixed_header = 8usize + 8 + 8 + 4 + 4 + 4 + 32 + 8 + 8;
        let state_bytes = (SPATIAL_VALUE_COUNT_V1 + GLOBAL_FEATURE_COUNT_V1)
            .checked_mul(4)
            .ok_or(TrainingWireError::PayloadTooLarge)?;
        self.examples
            .iter()
            .try_fold(fixed_header, |total, example| {
                let action_count = example.inference.legal_actions().len();
                let action_bytes = action_count
                    .checked_mul(4 * 2)
                    .ok_or(TrainingWireError::PayloadTooLarge)?;
                let policy_bytes = action_count
                    .checked_mul(4)
                    .ok_or(TrainingWireError::PayloadTooLarge)?;
                total
                    .checked_add(state_bytes)
                    .and_then(|value| value.checked_add(4))
                    .and_then(|value| value.checked_add(action_bytes))
                    .and_then(|value| value.checked_add(policy_bytes))
                    .and_then(|value| value.checked_add(VALUE_CLASS_COUNT_V1 * 4))
                    .ok_or(TrainingWireError::PayloadTooLarge)
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrainingResponseV1 {
    request_id: u64,
    completed_training_step: u64,
    completed_replay_index: u64,
    policy_loss: f32,
    value_loss: f32,
    total_loss: f32,
}

impl TrainingResponseV1 {
    pub fn decode_payload(
        payload: &[u8],
        request: &TrainingRequestV1,
    ) -> Result<Self, TrainingWireError> {
        let mut reader = WireReader::new(payload);
        let magic = reader.array::<8>()?;
        if magic == TRAINING_ERROR_MAGIC_V1 {
            return decode_service_error(reader, request.request_id);
        }
        if magic != TRAINING_RESPONSE_MAGIC_V1 {
            return Err(TrainingWireError::InvalidResponseMagic(magic));
        }
        let request_id = reader.u64()?;
        require_request_id(request.request_id, request_id)?;
        let completed_training_step = reader.u64()?;
        let expected_step = request.expected_training_step + 1;
        if completed_training_step != expected_step {
            return Err(TrainingWireError::ResponseTrainingStepMismatch {
                expected: expected_step,
                actual: completed_training_step,
            });
        }
        let completed_replay_index = reader.u64()?;
        if completed_replay_index != request.next_replay_index {
            return Err(TrainingWireError::ResponseReplayIndexMismatch {
                expected: request.next_replay_index,
                actual: completed_replay_index,
            });
        }
        let policy_loss = reader.f32()?;
        let value_loss = reader.f32()?;
        let total_loss = reader.f32()?;
        for (name, value) in [
            ("policy", policy_loss),
            ("value", value_loss),
            ("total", total_loss),
        ] {
            if !value.is_finite() {
                return Err(TrainingWireError::NonFiniteLoss { name, value });
            }
        }
        reader.finish()?;
        Ok(Self {
            request_id,
            completed_training_step,
            completed_replay_index,
            policy_loss,
            value_loss,
            total_loss,
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

    pub const fn policy_loss(self) -> f32 {
        self.policy_loss
    }

    pub const fn value_loss(self) -> f32 {
        self.value_loss
    }

    pub const fn total_loss(self) -> f32 {
        self.total_loss
    }
}

fn validate_distribution(name: &'static str, values: &[f32]) -> Result<(), TrainingWireError> {
    for &value in values {
        if !value.is_finite() || value < 0.0 {
            return Err(TrainingWireError::InvalidProbability { name, value });
        }
    }
    let sum = values.iter().sum::<f32>();
    if (sum - 1.0).abs() > DISTRIBUTION_TOLERANCE {
        return Err(TrainingWireError::InvalidDistribution { name, sum });
    }
    Ok(())
}

fn require_request_id(expected: u64, actual: u64) -> Result<(), TrainingWireError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TrainingWireError::ResponseRequestIdMismatch { expected, actual })
    }
}

fn decode_service_error(
    mut reader: WireReader<'_>,
    expected_request_id: u64,
) -> Result<TrainingResponseV1, TrainingWireError> {
    let request_id = reader.u64()?;
    require_request_id(expected_request_id, request_id)?;
    let length = reader.u32()? as usize;
    let message = String::from_utf8(reader.bytes(length)?.to_vec())
        .map_err(|_| TrainingWireError::InvalidErrorMessage)?;
    reader.finish()?;
    Err(TrainingWireError::Service(message))
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

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
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

    fn f32_slice(&mut self, values: &[f32]) {
        for &value in values {
            self.f32(value);
        }
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

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], TrainingWireError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(TrainingWireError::TruncatedResponse)?;
        let values = self
            .bytes
            .get(self.cursor..end)
            .ok_or(TrainingWireError::TruncatedResponse)?;
        self.cursor = end;
        Ok(values)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], TrainingWireError> {
        self.bytes(N)?
            .try_into()
            .map_err(|_| TrainingWireError::TruncatedResponse)
    }

    fn u32(&mut self) -> Result<u32, TrainingWireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, TrainingWireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, TrainingWireError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn finish(self) -> Result<(), TrainingWireError> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(TrainingWireError::TrailingResponseBytes(
                self.bytes.len() - self.cursor,
            ))
        }
    }
}

#[derive(Debug)]
pub enum TrainingWireError {
    PolicyTargetCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidProbability {
        name: &'static str,
        value: f32,
    },
    InvalidDistribution {
        name: &'static str,
        sum: f32,
    },
    EmptyBatch,
    TrainingStepOverflow,
    InvalidLearningRate(f32),
    ZeroActionCapacity,
    BatchTooLarge(usize),
    ActionCapacityTooLarge(usize),
    ReplayIndexOverflow,
    TooManyLegalActions {
        row: usize,
        capacity: usize,
        actual: usize,
    },
    PayloadTooLarge,
    TruncatedResponse,
    TrailingResponseBytes(usize),
    InvalidResponseMagic([u8; 8]),
    ResponseRequestIdMismatch {
        expected: u64,
        actual: u64,
    },
    ResponseTrainingStepMismatch {
        expected: u64,
        actual: u64,
    },
    ResponseReplayIndexMismatch {
        expected: u64,
        actual: u64,
    },
    NonFiniteLoss {
        name: &'static str,
        value: f32,
    },
    InvalidErrorMessage,
    Service(String),
}

impl fmt::Display for TrainingWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PolicyTargetCountMismatch { expected, actual } => write!(
                formatter,
                "training policy has {actual} targets; expected {expected}"
            ),
            Self::InvalidProbability { name, value } => {
                write!(
                    formatter,
                    "training {name} contains invalid probability {value}"
                )
            }
            Self::InvalidDistribution { name, sum } => {
                write!(formatter, "training {name} sums to {sum}, not 1")
            }
            Self::EmptyBatch => formatter.write_str("training request batch is empty"),
            Self::TrainingStepOverflow => formatter.write_str("training step would overflow"),
            Self::InvalidLearningRate(value) => {
                write!(
                    formatter,
                    "training learning rate {value} is not positive and finite"
                )
            }
            Self::ZeroActionCapacity => {
                formatter.write_str("training action capacity must be positive")
            }
            Self::BatchTooLarge(value) => write!(formatter, "training batch {value} exceeds V1"),
            Self::ActionCapacityTooLarge(value) => {
                write!(formatter, "training action capacity {value} exceeds V1")
            }
            Self::ReplayIndexOverflow => formatter.write_str("training replay index overflow"),
            Self::TooManyLegalActions {
                row,
                capacity,
                actual,
            } => write!(
                formatter,
                "training row {row} has {actual} legal actions; capacity is {capacity}"
            ),
            Self::PayloadTooLarge => formatter.write_str("training payload size overflow"),
            Self::TruncatedResponse => formatter.write_str("truncated training response"),
            Self::TrailingResponseBytes(count) => {
                write!(formatter, "training response has {count} trailing bytes")
            }
            Self::InvalidResponseMagic(value) => {
                write!(formatter, "invalid training response magic {value:?}")
            }
            Self::ResponseRequestIdMismatch { expected, actual } => write!(
                formatter,
                "training response request id {actual} does not match {expected}"
            ),
            Self::ResponseTrainingStepMismatch { expected, actual } => write!(
                formatter,
                "training response completed step {actual}; expected {expected}"
            ),
            Self::ResponseReplayIndexMismatch { expected, actual } => write!(
                formatter,
                "training response completed replay index {actual}; expected {expected}"
            ),
            Self::NonFiniteLoss { name, value } => {
                write!(formatter, "training {name} loss is non-finite: {value}")
            }
            Self::InvalidErrorMessage => formatter.write_str("training service error is not UTF-8"),
            Self::Service(message) => write!(formatter, "training service failed: {message}"),
        }
    }
}

impl std::error::Error for TrainingWireError {}
