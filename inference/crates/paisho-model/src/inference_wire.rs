use core::fmt;

use paisho_core::{legal_actions, GameOutcome, Position};

use crate::{
    encode_action_v1, state::encode_state_features_v1, ActionEncodingError, ActionEncodingV1,
    StateEncodingError, StateEncodingV1, GLOBAL_FEATURE_COUNT_V1, SPATIAL_VALUE_COUNT_V1,
    VALUE_CLASS_COUNT_V1,
};

pub const INFERENCE_REQUEST_MAGIC_V1: [u8; 8] = *b"PSIREQ01";
pub const INFERENCE_RESPONSE_MAGIC_V1: [u8; 8] = *b"PSIRSP01";
pub const INFERENCE_ERROR_MAGIC_V1: [u8; 8] = *b"PSIERR01";
const PROBABILITY_TOLERANCE: f32 = 1.0e-4;

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceExampleV1 {
    state: StateEncodingV1,
    legal_actions: Vec<ActionEncodingV1>,
}

impl InferenceExampleV1 {
    pub fn from_position(position: &Position) -> Result<Self, InferenceExampleEncodingError> {
        if position.outcome() != GameOutcome::Ongoing {
            return Err(InferenceExampleEncodingError::State(
                StateEncodingError::TerminalPosition,
            ));
        }
        let native_actions = legal_actions(position);
        if native_actions.is_empty() {
            return Err(InferenceExampleEncodingError::State(
                StateEncodingError::NoLegalAction,
            ));
        }
        let state = encode_state_features_v1(position);
        let legal_actions = native_actions
            .into_iter()
            .map(|action| encode_action_v1(action, position.to_move()))
            .collect::<Result<_, _>>()
            .map_err(InferenceExampleEncodingError::Action)?;
        Ok(Self {
            state,
            legal_actions,
        })
    }

    pub fn state(&self) -> &StateEncodingV1 {
        &self.state
    }

    pub fn legal_actions(&self) -> &[ActionEncodingV1] {
        &self.legal_actions
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceRequestV1 {
    request_id: u64,
    legal_action_capacity: usize,
    examples: Vec<InferenceExampleV1>,
}

impl InferenceRequestV1 {
    pub fn from_positions(
        request_id: u64,
        positions: &[Position],
        legal_action_capacity: usize,
    ) -> Result<Self, InferenceWireError> {
        let mut examples = Vec::with_capacity(positions.len());
        for (row, position) in positions.iter().enumerate() {
            examples.push(
                InferenceExampleV1::from_position(position)
                    .map_err(|source| InferenceWireError::ExampleEncoding { row, source })?,
            );
        }
        Self::from_examples(request_id, examples, legal_action_capacity)
    }

    pub fn from_examples(
        request_id: u64,
        examples: Vec<InferenceExampleV1>,
        legal_action_capacity: usize,
    ) -> Result<Self, InferenceWireError> {
        if examples.is_empty() {
            return Err(InferenceWireError::EmptyBatch);
        }
        if legal_action_capacity == 0 {
            return Err(InferenceWireError::ZeroActionCapacity);
        }
        u32::try_from(examples.len())
            .map_err(|_| InferenceWireError::BatchTooLarge(examples.len()))?;
        u32::try_from(legal_action_capacity)
            .map_err(|_| InferenceWireError::ActionCapacityTooLarge(legal_action_capacity))?;
        for (row, example) in examples.iter().enumerate() {
            if example.legal_actions.len() > legal_action_capacity {
                return Err(InferenceWireError::TooManyLegalActions {
                    row,
                    capacity: legal_action_capacity,
                    actual: example.legal_actions.len(),
                });
            }
        }
        Ok(Self {
            request_id,
            legal_action_capacity,
            examples,
        })
    }

    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    pub fn batch_size(&self) -> usize {
        self.examples.len()
    }

    pub const fn legal_action_capacity(&self) -> usize {
        self.legal_action_capacity
    }

    pub fn examples(&self) -> &[InferenceExampleV1] {
        &self.examples
    }

    pub fn encode_payload(&self) -> Result<Vec<u8>, InferenceWireError> {
        let capacity = self.encoded_payload_size()?;
        let mut writer = WireWriter::with_capacity(capacity);
        writer.bytes(&INFERENCE_REQUEST_MAGIC_V1);
        writer.u64(self.request_id);
        writer.u32(self.batch_size() as u32);
        writer.u32(self.legal_action_capacity as u32);
        for example in &self.examples {
            writer.f32_slice(example.state.spatial_nhwc());
            writer.f32_slice(example.state.global());
            writer.u32(example.legal_actions.len() as u32);
            for action in &example.legal_actions {
                for slot in action.slots() {
                    writer.u16(slot);
                }
            }
        }
        Ok(writer.finish())
    }

    fn encoded_payload_size(&self) -> Result<usize, InferenceWireError> {
        let fixed_header = 8usize + 8 + 4 + 4;
        let state_bytes = (SPATIAL_VALUE_COUNT_V1 + GLOBAL_FEATURE_COUNT_V1)
            .checked_mul(4)
            .ok_or(InferenceWireError::PayloadTooLarge)?;
        self.examples
            .iter()
            .try_fold(fixed_header, |total, example| {
                let action_bytes = example
                    .legal_actions
                    .len()
                    .checked_mul(4 * 2)
                    .ok_or(InferenceWireError::PayloadTooLarge)?;
                total
                    .checked_add(state_bytes)
                    .and_then(|value| value.checked_add(4))
                    .and_then(|value| value.checked_add(action_bytes))
                    .ok_or(InferenceWireError::PayloadTooLarge)
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceExampleEncodingError {
    State(StateEncodingError),
    Action(ActionEncodingError),
}

impl fmt::Display for InferenceExampleEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::State(source) => source.fmt(formatter),
            Self::Action(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for InferenceExampleEncodingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::State(source) => Some(source),
            Self::Action(source) => Some(source),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceOutputV1 {
    policy_probabilities: Vec<f32>,
    value_probabilities: [f32; VALUE_CLASS_COUNT_V1],
}

impl InferenceOutputV1 {
    pub fn policy_probabilities(&self) -> &[f32] {
        &self.policy_probabilities
    }

    pub const fn value_probabilities(&self) -> &[f32; VALUE_CLASS_COUNT_V1] {
        &self.value_probabilities
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceResponseV1 {
    request_id: u64,
    outputs: Vec<InferenceOutputV1>,
}

impl InferenceResponseV1 {
    pub fn decode_payload(
        payload: &[u8],
        request: &InferenceRequestV1,
    ) -> Result<Self, InferenceWireError> {
        let mut reader = WireReader::new(payload);
        let magic = reader.array::<8>()?;
        if magic == INFERENCE_ERROR_MAGIC_V1 {
            return decode_service_error(reader, request.request_id);
        }
        if magic != INFERENCE_RESPONSE_MAGIC_V1 {
            return Err(InferenceWireError::InvalidResponseMagic(magic));
        }
        let request_id = reader.u64()?;
        require_request_id(request.request_id, request_id)?;
        let batch_size = reader.u32()? as usize;
        let legal_action_capacity = reader.u32()? as usize;
        if batch_size != request.batch_size()
            || legal_action_capacity != request.legal_action_capacity
        {
            return Err(InferenceWireError::ResponseShapeMismatch {
                expected_batch: request.batch_size(),
                actual_batch: batch_size,
                expected_capacity: request.legal_action_capacity,
                actual_capacity: legal_action_capacity,
            });
        }

        let mut policies = Vec::with_capacity(batch_size);
        for (row, example) in request.examples.iter().enumerate() {
            let mut legal = Vec::with_capacity(example.legal_actions.len());
            for column in 0..legal_action_capacity {
                let probability = reader.f32()?;
                validate_probability("policy", row, probability)?;
                if column < example.legal_actions.len() {
                    legal.push(probability);
                } else if probability.abs() > PROBABILITY_TOLERANCE {
                    return Err(InferenceWireError::NonZeroPadding {
                        row,
                        index: column,
                        value: probability,
                    });
                }
            }
            validate_distribution("policy", row, &legal)?;
            policies.push(legal);
        }

        let mut outputs = Vec::with_capacity(batch_size);
        for (row, policy_probabilities) in policies.into_iter().enumerate() {
            let mut value_probabilities = [0.0; VALUE_CLASS_COUNT_V1];
            for value in &mut value_probabilities {
                *value = reader.f32()?;
                validate_probability("value", row, *value)?;
            }
            validate_distribution("value", row, &value_probabilities)?;
            outputs.push(InferenceOutputV1 {
                policy_probabilities,
                value_probabilities,
            });
        }
        reader.finish()?;
        Ok(Self {
            request_id,
            outputs,
        })
    }

    pub const fn request_id(&self) -> u64 {
        self.request_id
    }

    pub fn outputs(&self) -> &[InferenceOutputV1] {
        &self.outputs
    }

    pub fn into_outputs(self) -> Vec<InferenceOutputV1> {
        self.outputs
    }
}

fn require_request_id(expected: u64, actual: u64) -> Result<(), InferenceWireError> {
    if expected == actual {
        Ok(())
    } else {
        Err(InferenceWireError::ResponseRequestIdMismatch { expected, actual })
    }
}

fn decode_service_error(
    mut reader: WireReader<'_>,
    expected_request_id: u64,
) -> Result<InferenceResponseV1, InferenceWireError> {
    let request_id = reader.u64()?;
    require_request_id(expected_request_id, request_id)?;
    let length = reader.u32()? as usize;
    let message = String::from_utf8(reader.bytes(length)?.to_vec())
        .map_err(|_| InferenceWireError::InvalidErrorMessage)?;
    reader.finish()?;
    Err(InferenceWireError::Service(message))
}

fn validate_probability(
    name: &'static str,
    row: usize,
    value: f32,
) -> Result<(), InferenceWireError> {
    if !value.is_finite() || value < 0.0 {
        Err(InferenceWireError::InvalidProbability { name, row, value })
    } else {
        Ok(())
    }
}

fn validate_distribution(
    name: &'static str,
    row: usize,
    values: &[f32],
) -> Result<(), InferenceWireError> {
    let sum = values.iter().sum::<f32>();
    if (sum - 1.0).abs() <= PROBABILITY_TOLERANCE {
        Ok(())
    } else {
        Err(InferenceWireError::InvalidDistribution { name, row, sum })
    }
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

    fn f32_slice(&mut self, values: &[f32]) {
        for value in values {
            self.u32(value.to_bits());
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

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], InferenceWireError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(InferenceWireError::TruncatedResponse)?;
        let values = self
            .bytes
            .get(self.cursor..end)
            .ok_or(InferenceWireError::TruncatedResponse)?;
        self.cursor = end;
        Ok(values)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], InferenceWireError> {
        self.bytes(N)?
            .try_into()
            .map_err(|_| InferenceWireError::TruncatedResponse)
    }

    fn u32(&mut self) -> Result<u32, InferenceWireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, InferenceWireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, InferenceWireError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn finish(self) -> Result<(), InferenceWireError> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(InferenceWireError::TrailingResponseBytes(
                self.bytes.len() - self.cursor,
            ))
        }
    }
}

#[derive(Debug)]
pub enum InferenceWireError {
    EmptyBatch,
    ZeroActionCapacity,
    BatchTooLarge(usize),
    ActionCapacityTooLarge(usize),
    ExampleEncoding {
        row: usize,
        source: InferenceExampleEncodingError,
    },
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
    ResponseShapeMismatch {
        expected_batch: usize,
        actual_batch: usize,
        expected_capacity: usize,
        actual_capacity: usize,
    },
    InvalidProbability {
        name: &'static str,
        row: usize,
        value: f32,
    },
    InvalidDistribution {
        name: &'static str,
        row: usize,
        sum: f32,
    },
    NonZeroPadding {
        row: usize,
        index: usize,
        value: f32,
    },
    InvalidErrorMessage,
    Service(String),
}

impl fmt::Display for InferenceWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("inference request batch is empty"),
            Self::ZeroActionCapacity => {
                formatter.write_str("inference action capacity must be positive")
            }
            Self::BatchTooLarge(value) => write!(formatter, "batch size {value} exceeds V1"),
            Self::ActionCapacityTooLarge(value) => {
                write!(formatter, "action capacity {value} exceeds V1")
            }
            Self::ExampleEncoding { row, source } => {
                write!(formatter, "inference encoding failed for row {row}: {source}")
            }
            Self::TooManyLegalActions {
                row,
                capacity,
                actual,
            } => write!(
                formatter,
                "row {row} has {actual} legal actions; request capacity is {capacity}"
            ),
            Self::PayloadTooLarge => formatter.write_str("inference payload size overflow"),
            Self::TruncatedResponse => formatter.write_str("truncated inference response"),
            Self::TrailingResponseBytes(count) => {
                write!(formatter, "inference response has {count} trailing bytes")
            }
            Self::InvalidResponseMagic(value) => {
                write!(formatter, "invalid inference response magic {value:?}")
            }
            Self::ResponseRequestIdMismatch { expected, actual } => write!(
                formatter,
                "response request id {actual} does not match {expected}"
            ),
            Self::ResponseShapeMismatch {
                expected_batch,
                actual_batch,
                expected_capacity,
                actual_capacity,
            } => write!(
                formatter,
                "response shape {actual_batch}x{actual_capacity} does not match {expected_batch}x{expected_capacity}"
            ),
            Self::InvalidProbability { name, row, value } => {
                write!(formatter, "{name} row {row} contains invalid probability {value}")
            }
            Self::InvalidDistribution { name, row, sum } => {
                write!(formatter, "{name} row {row} sums to {sum}, not 1")
            }
            Self::NonZeroPadding { row, index, value } => write!(
                formatter,
                "policy padding {index} in row {row} is non-zero ({value})"
            ),
            Self::InvalidErrorMessage => {
                formatter.write_str("inference service returned non-UTF-8 error text")
            }
            Self::Service(message) => write!(formatter, "inference service error: {message}"),
        }
    }
}

impl std::error::Error for InferenceWireError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ExampleEncoding { source, .. } => Some(source),
            _ => None,
        }
    }
}
