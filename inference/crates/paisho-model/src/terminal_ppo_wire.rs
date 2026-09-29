use core::fmt;

use crate::{InferenceExampleV1, ValueClassV1, GLOBAL_FEATURE_COUNT_V1, SPATIAL_VALUE_COUNT_V1};

pub const TERMINAL_PPO_OBJECTIVE_V1: &str = "ppo-terminal-v1";
pub const TERMINAL_PPO_REQUEST_MAGIC_V1: [u8; 8] = *b"PSTREQ02";
pub const TERMINAL_PPO_RESPONSE_MAGIC_V1: [u8; 8] = *b"PSTRSP03";
pub const TERMINAL_PPO_ERROR_MAGIC_V1: [u8; 8] = *b"PSTERR02";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalPpoParametersV1 {
    policy_temperature: f32,
    uniform_mix: f32,
    clip_epsilon: f32,
    value_loss_weight: f32,
    entropy_weight: f32,
}

impl TerminalPpoParametersV1 {
    pub fn new(
        clip_epsilon: f32,
        value_loss_weight: f32,
        entropy_weight: f32,
    ) -> Result<Self, TerminalPpoWireError> {
        Self::with_behavior(1.0, 0.0, clip_epsilon, value_loss_weight, entropy_weight)
    }

    pub fn with_behavior(
        policy_temperature: f32,
        uniform_mix: f32,
        clip_epsilon: f32,
        value_loss_weight: f32,
        entropy_weight: f32,
    ) -> Result<Self, TerminalPpoWireError> {
        if !policy_temperature.is_finite() || policy_temperature <= 0.0 {
            return Err(TerminalPpoWireError::InvalidPolicyTemperature(
                policy_temperature,
            ));
        }
        if !uniform_mix.is_finite() || !(0.0..=1.0).contains(&uniform_mix) {
            return Err(TerminalPpoWireError::InvalidUniformMix(uniform_mix));
        }
        if !clip_epsilon.is_finite() || !(0.0..1.0).contains(&clip_epsilon) {
            return Err(TerminalPpoWireError::InvalidClipEpsilon(clip_epsilon));
        }
        validate_non_negative_weight("value", value_loss_weight)?;
        validate_non_negative_weight("entropy", entropy_weight)?;
        Ok(Self {
            policy_temperature,
            uniform_mix,
            clip_epsilon,
            value_loss_weight,
            entropy_weight,
        })
    }

    pub const fn policy_temperature(self) -> f32 {
        self.policy_temperature
    }

    pub const fn uniform_mix(self) -> f32 {
        self.uniform_mix
    }

    pub const fn clip_epsilon(self) -> f32 {
        self.clip_epsilon
    }

    pub const fn value_loss_weight(self) -> f32 {
        self.value_loss_weight
    }

    pub const fn entropy_weight(self) -> f32 {
        self.entropy_weight
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerminalPpoExampleV1 {
    inference: InferenceExampleV1,
    played_action_index: usize,
    behavior_probability: f32,
    terminal_value: ValueClassV1,
    actor_value: f32,
    policy_return: Option<f32>,
}

impl TerminalPpoExampleV1 {
    pub fn new(
        inference: InferenceExampleV1,
        played_action_index: usize,
        behavior_probability: f32,
        terminal_value: ValueClassV1,
        actor_value: f32,
    ) -> Result<Self, TerminalPpoWireError> {
        if played_action_index >= inference.legal_actions().len() {
            return Err(TerminalPpoWireError::PlayedActionOutOfRange {
                played: played_action_index,
                legal_actions: inference.legal_actions().len(),
            });
        }
        if !behavior_probability.is_finite()
            || behavior_probability <= 0.0
            || behavior_probability > 1.0
        {
            return Err(TerminalPpoWireError::InvalidBehaviorProbability(
                behavior_probability,
            ));
        }
        if !actor_value.is_finite() || !(-1.0..=1.0).contains(&actor_value) {
            return Err(TerminalPpoWireError::InvalidActorValue(actor_value));
        }
        Ok(Self {
            inference,
            played_action_index,
            behavior_probability,
            terminal_value,
            actor_value,
            policy_return: None,
        })
    }

    pub const fn inference(&self) -> &InferenceExampleV1 {
        &self.inference
    }

    pub const fn played_action_index(&self) -> usize {
        self.played_action_index
    }

    pub const fn behavior_probability(&self) -> f32 {
        self.behavior_probability
    }

    pub const fn terminal_value(&self) -> ValueClassV1 {
        self.terminal_value
    }

    pub const fn actor_value(&self) -> f32 {
        self.actor_value
    }

    /// Bounded win-duration utility; WDL targets and the measured baseline stay unchanged.
    pub fn with_win_duration(mut self, remaining_decisions: usize) -> Self {
        self.policy_return = Some(if self.terminal_value.signed_return() > 0.0 {
            0.9 + 0.1 * 256.0 / (256.0 + remaining_decisions as f32)
        } else {
            self.terminal_value.signed_return()
        });
        self
    }

    pub fn advantage(&self) -> f32 {
        self.policy_return
            .unwrap_or(self.terminal_value.signed_return())
            - self.actor_value
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TerminalPpoRequestV1 {
    request_id: u64,
    expected_training_step: u64,
    learning_rate: f32,
    parameters: TerminalPpoParametersV1,
    legal_action_capacity: usize,
    replay_snapshot_sha256: [u8; 32],
    start_replay_index: u64,
    next_replay_index: u64,
    examples: Vec<TerminalPpoExampleV1>,
}

impl TerminalPpoRequestV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: u64,
        expected_training_step: u64,
        learning_rate: f32,
        parameters: TerminalPpoParametersV1,
        legal_action_capacity: usize,
        replay_snapshot_sha256: [u8; 32],
        start_replay_index: u64,
        examples: Vec<TerminalPpoExampleV1>,
    ) -> Result<Self, TerminalPpoWireError> {
        if examples.is_empty() {
            return Err(TerminalPpoWireError::EmptyBatch);
        }
        if expected_training_step == u64::MAX {
            return Err(TerminalPpoWireError::TrainingStepOverflow);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err(TerminalPpoWireError::InvalidLearningRate(learning_rate));
        }
        if legal_action_capacity == 0 {
            return Err(TerminalPpoWireError::ZeroActionCapacity);
        }
        u32::try_from(examples.len())
            .map_err(|_| TerminalPpoWireError::BatchTooLarge(examples.len()))?;
        u32::try_from(legal_action_capacity)
            .map_err(|_| TerminalPpoWireError::ActionCapacityTooLarge(legal_action_capacity))?;
        let next_replay_index = start_replay_index
            .checked_add(examples.len() as u64)
            .ok_or(TerminalPpoWireError::ReplayIndexOverflow)?;
        for (row, example) in examples.iter().enumerate() {
            if example.inference.legal_actions().len() > legal_action_capacity {
                return Err(TerminalPpoWireError::TooManyLegalActions {
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
            parameters,
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

    pub const fn parameters(&self) -> TerminalPpoParametersV1 {
        self.parameters
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

    pub fn examples(&self) -> &[TerminalPpoExampleV1] {
        &self.examples
    }

    pub fn encode_payload(&self) -> Result<Vec<u8>, TerminalPpoWireError> {
        let mut writer = WireWriter::with_capacity(self.encoded_payload_size()?);
        let shaped = self.examples.iter().any(|e| e.policy_return.is_some());
        writer.bytes(if shaped {
            b"PSTREQ03"
        } else {
            &TERMINAL_PPO_REQUEST_MAGIC_V1
        });
        writer.u64(self.request_id);
        writer.u64(self.expected_training_step);
        writer.f32(self.learning_rate);
        writer.f32(self.parameters.policy_temperature);
        writer.f32(self.parameters.uniform_mix);
        writer.f32(self.parameters.clip_epsilon);
        writer.f32(self.parameters.value_loss_weight);
        writer.f32(self.parameters.entropy_weight);
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
            writer.u32(example.played_action_index as u32);
            writer.f32(example.behavior_probability);
            writer.u32(example.terminal_value.index() as u32);
            writer.f32(example.actor_value);
            if shaped {
                writer.f32(
                    example
                        .policy_return
                        .unwrap_or(example.terminal_value.signed_return()),
                );
            }
        }
        Ok(writer.finish())
    }

    fn encoded_payload_size(&self) -> Result<usize, TerminalPpoWireError> {
        let extra = if self.examples.iter().any(|e| e.policy_return.is_some()) {
            4
        } else {
            0
        };
        let fixed_header = 8usize + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 32 + 8 + 8;
        let state_bytes = (SPATIAL_VALUE_COUNT_V1 + GLOBAL_FEATURE_COUNT_V1)
            .checked_mul(4)
            .ok_or(TerminalPpoWireError::PayloadTooLarge)?;
        self.examples
            .iter()
            .try_fold(fixed_header, |total, example| {
                let action_bytes = example
                    .inference
                    .legal_actions()
                    .len()
                    .checked_mul(4 * 2)
                    .ok_or(TerminalPpoWireError::PayloadTooLarge)?;
                total
                    .checked_add(state_bytes)
                    .and_then(|value| value.checked_add(4))
                    .and_then(|value| value.checked_add(action_bytes))
                    .and_then(|value| value.checked_add(4 + 4 + 4 + 4 + extra))
                    .ok_or(TerminalPpoWireError::PayloadTooLarge)
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerminalPpoResponseV1 {
    request_id: u64,
    completed_training_step: u64,
    completed_replay_index: u64,
    policy_loss: f32,
    value_loss: f32,
    entropy: f32,
    total_loss: f32,
    mean_advantage: f32,
    mean_importance_ratio: f32,
    mean_squared_ratio_deviation: f32,
}

impl TerminalPpoResponseV1 {
    pub fn decode_payload(
        payload: &[u8],
        request: &TerminalPpoRequestV1,
    ) -> Result<Self, TerminalPpoWireError> {
        let mut reader = WireReader::new(payload);
        let magic = reader.array::<8>()?;
        if magic == TERMINAL_PPO_ERROR_MAGIC_V1 {
            return decode_service_error(reader, request.request_id);
        }
        if magic != TERMINAL_PPO_RESPONSE_MAGIC_V1 {
            return Err(TerminalPpoWireError::InvalidResponseMagic(magic));
        }
        let request_id = reader.u64()?;
        require_request_id(request.request_id, request_id)?;
        let completed_training_step = reader.u64()?;
        let expected_step = request.expected_training_step + 1;
        if completed_training_step != expected_step {
            return Err(TerminalPpoWireError::ResponseTrainingStepMismatch {
                expected: expected_step,
                actual: completed_training_step,
            });
        }
        let completed_replay_index = reader.u64()?;
        if completed_replay_index != request.next_replay_index {
            return Err(TerminalPpoWireError::ResponseReplayIndexMismatch {
                expected: request.next_replay_index,
                actual: completed_replay_index,
            });
        }
        let policy_loss = reader.f32()?;
        let value_loss = reader.f32()?;
        let entropy = reader.f32()?;
        let total_loss = reader.f32()?;
        let mean_advantage = reader.f32()?;
        let mean_importance_ratio = reader.f32()?;
        let mean_squared_ratio_deviation = reader.f32()?;
        for (name, value) in [
            ("policy", policy_loss),
            ("value", value_loss),
            ("entropy", entropy),
            ("total", total_loss),
            ("mean advantage", mean_advantage),
            ("mean importance ratio", mean_importance_ratio),
            ("mean squared ratio deviation", mean_squared_ratio_deviation),
        ] {
            if !value.is_finite() {
                return Err(TerminalPpoWireError::NonFiniteMetric { name, value });
            }
        }
        reader.finish()?;
        Ok(Self {
            request_id,
            completed_training_step,
            completed_replay_index,
            policy_loss,
            value_loss,
            entropy,
            total_loss,
            mean_advantage,
            mean_importance_ratio,
            mean_squared_ratio_deviation,
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

    pub const fn entropy(self) -> f32 {
        self.entropy
    }

    pub const fn total_loss(self) -> f32 {
        self.total_loss
    }

    pub const fn mean_advantage(self) -> f32 {
        self.mean_advantage
    }

    pub const fn mean_importance_ratio(self) -> f32 {
        self.mean_importance_ratio
    }

    pub const fn mean_squared_ratio_deviation(self) -> f32 {
        self.mean_squared_ratio_deviation
    }
}

fn validate_non_negative_weight(
    name: &'static str,
    value: f32,
) -> Result<(), TerminalPpoWireError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(TerminalPpoWireError::InvalidLossWeight { name, value })
    }
}

fn require_request_id(expected: u64, actual: u64) -> Result<(), TerminalPpoWireError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TerminalPpoWireError::ResponseRequestIdMismatch { expected, actual })
    }
}

fn decode_service_error(
    mut reader: WireReader<'_>,
    expected_request_id: u64,
) -> Result<TerminalPpoResponseV1, TerminalPpoWireError> {
    let request_id = reader.u64()?;
    require_request_id(expected_request_id, request_id)?;
    let length = reader.u32()? as usize;
    let message = String::from_utf8(reader.bytes(length)?.to_vec())
        .map_err(|_| TerminalPpoWireError::InvalidErrorMessage)?;
    reader.finish()?;
    Err(TerminalPpoWireError::Service(message))
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

    fn bytes(&mut self, count: usize) -> Result<&'a [u8], TerminalPpoWireError> {
        let end = self
            .cursor
            .checked_add(count)
            .ok_or(TerminalPpoWireError::TruncatedResponse)?;
        let values = self
            .bytes
            .get(self.cursor..end)
            .ok_or(TerminalPpoWireError::TruncatedResponse)?;
        self.cursor = end;
        Ok(values)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], TerminalPpoWireError> {
        self.bytes(N)?
            .try_into()
            .map_err(|_| TerminalPpoWireError::TruncatedResponse)
    }

    fn u32(&mut self) -> Result<u32, TerminalPpoWireError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, TerminalPpoWireError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, TerminalPpoWireError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn finish(self) -> Result<(), TerminalPpoWireError> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(TerminalPpoWireError::TrailingResponseBytes(
                self.bytes.len() - self.cursor,
            ))
        }
    }
}

#[derive(Debug)]
pub enum TerminalPpoWireError {
    PlayedActionOutOfRange {
        played: usize,
        legal_actions: usize,
    },
    InvalidBehaviorProbability(f32),
    InvalidActorValue(f32),
    InvalidPolicyTemperature(f32),
    InvalidUniformMix(f32),
    InvalidClipEpsilon(f32),
    InvalidLossWeight {
        name: &'static str,
        value: f32,
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
    NonFiniteMetric {
        name: &'static str,
        value: f32,
    },
    InvalidErrorMessage,
    Service(String),
}

impl fmt::Display for TerminalPpoWireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlayedActionOutOfRange {
                played,
                legal_actions,
            } => write!(
                formatter,
                "played action index {played} is outside {legal_actions} legal actions"
            ),
            Self::InvalidBehaviorProbability(value) => write!(
                formatter,
                "behavior probability must be finite and in (0, 1], got {value}"
            ),
            Self::InvalidActorValue(value) => {
                write!(
                    formatter,
                    "actor value must be finite and in [-1, 1], got {value}"
                )
            }
            Self::InvalidPolicyTemperature(value) => write!(
                formatter,
                "PPO policy temperature must be finite and positive, got {value}"
            ),
            Self::InvalidUniformMix(value) => write!(
                formatter,
                "PPO uniform mix must be finite and in [0, 1], got {value}"
            ),
            Self::InvalidClipEpsilon(value) => write!(
                formatter,
                "PPO clip epsilon must be finite and in (0, 1), got {value}"
            ),
            Self::InvalidLossWeight { name, value } => write!(
                formatter,
                "PPO {name} weight must be finite and non-negative, got {value}"
            ),
            Self::EmptyBatch => formatter.write_str("terminal PPO request batch is empty"),
            Self::TrainingStepOverflow => formatter.write_str("training step would overflow"),
            Self::InvalidLearningRate(value) => write!(
                formatter,
                "training learning rate {value} is not positive and finite"
            ),
            Self::ZeroActionCapacity => {
                formatter.write_str("terminal PPO action capacity must be positive")
            }
            Self::BatchTooLarge(value) => {
                write!(formatter, "terminal PPO batch {value} exceeds V1")
            }
            Self::ActionCapacityTooLarge(value) => {
                write!(formatter, "terminal PPO action capacity {value} exceeds V1")
            }
            Self::ReplayIndexOverflow => formatter.write_str("terminal PPO replay index overflow"),
            Self::TooManyLegalActions {
                row,
                capacity,
                actual,
            } => write!(
                formatter,
                "terminal PPO row {row} has {actual} legal actions; capacity is {capacity}"
            ),
            Self::PayloadTooLarge => formatter.write_str("terminal PPO payload size overflow"),
            Self::TruncatedResponse => formatter.write_str("truncated terminal PPO response"),
            Self::TrailingResponseBytes(count) => {
                write!(
                    formatter,
                    "terminal PPO response has {count} trailing bytes"
                )
            }
            Self::InvalidResponseMagic(value) => {
                write!(formatter, "invalid terminal PPO response magic {value:?}")
            }
            Self::ResponseRequestIdMismatch { expected, actual } => write!(
                formatter,
                "terminal PPO response request id {actual} does not match {expected}"
            ),
            Self::ResponseTrainingStepMismatch { expected, actual } => write!(
                formatter,
                "terminal PPO response completed step {actual}; expected {expected}"
            ),
            Self::ResponseReplayIndexMismatch { expected, actual } => write!(
                formatter,
                "terminal PPO response completed replay index {actual}; expected {expected}"
            ),
            Self::NonFiniteMetric { name, value } => {
                write!(
                    formatter,
                    "terminal PPO {name} metric is non-finite: {value}"
                )
            }
            Self::InvalidErrorMessage => {
                formatter.write_str("terminal PPO service error is not UTF-8")
            }
            Self::Service(message) => write!(formatter, "terminal PPO service failed: {message}"),
        }
    }
}

impl std::error::Error for TerminalPpoWireError {}
