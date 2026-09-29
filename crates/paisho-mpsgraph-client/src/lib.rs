//! Persistent child-process bridge to the macOS MPSGraph inference/training service.

mod broker;
mod capacity;
mod checkpoint_metadata;
pub mod training_cycle;
pub mod weights;
pub use weights::{WeightImportAck, WeightSnapshot};

use std::fmt;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, ExitStatus, Stdio};

use paisho_model::{
    CheckpointRequestV1, CheckpointResponseV1, CheckpointWireError, InferenceRequestV1,
    InferenceResponseV1, InferenceWireError, TerminalPpoRequestV1, TerminalPpoResponseV1,
    TerminalPpoWireError, TrainingRequestV1, TrainingResponseV1, TrainingWireError,
};
use sha2::{Digest, Sha256};

pub use broker::{
    InferenceBroker, InferenceBrokerClient, InferenceBrokerConfiguration, InferenceBrokerError,
    InferenceBrokerTelemetry,
};
pub use capacity::{
    CapacityClassConfiguration, CapacityClassTelemetry, CapacityInferenceBroker,
    CapacityInferenceBrokerClient, CapacityInferenceTelemetry,
};
pub use checkpoint_metadata::{
    read_checkpoint_metadata, CheckpointMetadataError, CheckpointMetadataV2,
};
pub use training_cycle::{
    TrainingCycleProgressV1, TrainingCycleRandomStateV1, TrainingCycleRequestV1,
    TrainingCycleResponseV1, TrainingCycleSchedulerV1,
};

const FRAME_LENGTH_BYTES: usize = 8;
const MAXIMUM_ERROR_RESPONSE_BYTES: usize = 1_048_576;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum NetworkPreset {
    Micro,
    Pure,
}

impl NetworkPreset {
    const fn argument(self) -> &'static str {
        match self {
            Self::Micro => "micro",
            Self::Pure => "pure",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub enum OptimizationLevel {
    Level0,
    Level1,
}

impl OptimizationLevel {
    const fn argument(self) -> &'static str {
        match self {
            Self::Level0 => "0",
            Self::Level1 => "1",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ServiceConfiguration {
    pub executable: PathBuf,
    pub preset: NetworkPreset,
    pub batch_size: usize,
    pub legal_action_capacity: usize,
    pub inference_slots: usize,
    pub optimization: OptimizationLevel,
    pub seed: u64,
    pub checkpoint: Option<PathBuf>,
}

impl ServiceConfiguration {
    pub fn validate(&self) -> Result<(), MpsGraphClientError> {
        if self.batch_size == 0 {
            return Err(MpsGraphClientError::ZeroBatchSize);
        }
        if self.legal_action_capacity == 0 {
            return Err(MpsGraphClientError::ZeroActionCapacity);
        }
        if self.inference_slots == 0 {
            return Err(MpsGraphClientError::ZeroInferenceSlots);
        }
        u32::try_from(self.batch_size)
            .map_err(|_| MpsGraphClientError::BatchTooLarge(self.batch_size))?;
        u32::try_from(self.legal_action_capacity)
            .map_err(|_| MpsGraphClientError::ActionCapacityTooLarge(self.legal_action_capacity))?;
        let metadata = fs::metadata(&self.executable).map_err(|source| {
            MpsGraphClientError::UnavailableExecutable {
                path: self.executable.clone(),
                source,
            }
        })?;
        if !metadata.is_file() {
            return Err(MpsGraphClientError::ExecutableIsNotAFile(
                self.executable.clone(),
            ));
        }
        if let Some(checkpoint) = &self.checkpoint {
            let metadata = fs::metadata(checkpoint).map_err(|source| {
                MpsGraphClientError::UnavailableCheckpoint {
                    path: checkpoint.clone(),
                    source,
                }
            })?;
            if !metadata.is_file() {
                return Err(MpsGraphClientError::CheckpointIsNotAFile(
                    checkpoint.clone(),
                ));
            }
        }
        response_payload_limit(self.batch_size, self.legal_action_capacity)?;
        Ok(())
    }

    pub(crate) fn runs_same_model_as(&self, other: &Self) -> bool {
        self.executable == other.executable
            && self.preset == other.preset
            && self.optimization == other.optimization
            && self.checkpoint == other.checkpoint
            && (self.checkpoint.is_some() || self.seed == other.seed)
    }
}

pub struct MpsGraphProcess {
    configuration: ServiceConfiguration,
    child: Option<Child>,
    input: Option<ChildStdin>,
    output: Option<ChildStdout>,
    response_payload_limit: usize,
}

impl MpsGraphProcess {
    pub fn launch(configuration: ServiceConfiguration) -> Result<Self, MpsGraphClientError> {
        Self::launch_with_mode(configuration, false)
    }

    pub fn launch_new_generation(
        configuration: ServiceConfiguration,
    ) -> Result<Self, MpsGraphClientError> {
        if configuration.checkpoint.is_none() {
            return Err(MpsGraphClientError::NewGenerationWithoutCheckpoint);
        }
        Self::launch_with_mode(configuration, true)
    }

    fn launch_with_mode(
        configuration: ServiceConfiguration,
        new_generation: bool,
    ) -> Result<Self, MpsGraphClientError> {
        configuration.validate()?;
        let response_payload_limit = response_payload_limit(
            configuration.batch_size,
            configuration.legal_action_capacity,
        )?;
        let mut command = Command::new(&configuration.executable);
        command
            .arg("--preset")
            .arg(configuration.preset.argument())
            .arg("--batch")
            .arg(configuration.batch_size.to_string())
            .arg("--actions")
            .arg(configuration.legal_action_capacity.to_string())
            .arg("--inference-slots")
            .arg(configuration.inference_slots.to_string())
            .arg("--level")
            .arg(configuration.optimization.argument())
            .arg("--seed")
            .arg(configuration.seed.to_string());
        if let Some(checkpoint) = &configuration.checkpoint {
            command.arg("--checkpoint").arg(checkpoint);
        }
        if new_generation {
            command.arg("--checkpoint-mode").arg("new-generation");
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|source| MpsGraphClientError::Launch {
                path: configuration.executable.clone(),
                source,
            })?;
        let input = child
            .stdin
            .take()
            .ok_or(MpsGraphClientError::MissingChildPipe("stdin"))?;
        let output = child
            .stdout
            .take()
            .ok_or(MpsGraphClientError::MissingChildPipe("stdout"))?;
        Ok(Self {
            configuration,
            child: Some(child),
            input: Some(input),
            output: Some(output),
            response_payload_limit,
        })
    }

    pub fn configuration(&self) -> &ServiceConfiguration {
        &self.configuration
    }

    pub fn process_id(&self) -> Option<u32> {
        self.child.as_ref().map(Child::id)
    }

    pub fn infer(
        &mut self,
        request: &InferenceRequestV1,
    ) -> Result<InferenceResponseV1, MpsGraphClientError> {
        self.send_inference(request)?;
        self.receive_inference(request)
    }

    pub(crate) fn send_inference(
        &mut self,
        request: &InferenceRequestV1,
    ) -> Result<(), MpsGraphClientError> {
        if request.batch_size() != self.configuration.batch_size
            || request.legal_action_capacity() != self.configuration.legal_action_capacity
        {
            return Err(MpsGraphClientError::RequestShapeMismatch {
                expected_batch: self.configuration.batch_size,
                actual_batch: request.batch_size(),
                expected_capacity: self.configuration.legal_action_capacity,
                actual_capacity: request.legal_action_capacity(),
            });
        }
        let payload = request.encode_payload()?;
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &payload,
        )?;
        Ok(())
    }

    pub(crate) fn receive_inference(
        &mut self,
        request: &InferenceRequestV1,
    ) -> Result<InferenceResponseV1, MpsGraphClientError> {
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            self.response_payload_limit,
        )?;
        InferenceResponseV1::decode_payload(&response, request).map_err(Into::into)
    }

    pub fn train(
        &mut self,
        request: &TrainingRequestV1,
    ) -> Result<TrainingResponseV1, MpsGraphClientError> {
        if request.batch_size() != self.configuration.batch_size
            || request.legal_action_capacity() != self.configuration.legal_action_capacity
        {
            return Err(MpsGraphClientError::RequestShapeMismatch {
                expected_batch: self.configuration.batch_size,
                actual_batch: request.batch_size(),
                expected_capacity: self.configuration.legal_action_capacity,
                actual_capacity: request.legal_action_capacity(),
            });
        }
        let payload = request.encode_payload()?;
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &payload,
        )?;
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            self.response_payload_limit,
        )?;
        TrainingResponseV1::decode_payload(&response, request).map_err(Into::into)
    }

    pub fn train_terminal_ppo(
        &mut self,
        request: &TerminalPpoRequestV1,
    ) -> Result<TerminalPpoResponseV1, MpsGraphClientError> {
        if request.batch_size() != self.configuration.batch_size
            || request.legal_action_capacity() != self.configuration.legal_action_capacity
        {
            return Err(MpsGraphClientError::RequestShapeMismatch {
                expected_batch: self.configuration.batch_size,
                actual_batch: request.batch_size(),
                expected_capacity: self.configuration.legal_action_capacity,
                actual_capacity: request.legal_action_capacity(),
            });
        }
        let payload = request.encode_payload()?;
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &payload,
        )?;
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            self.response_payload_limit,
        )?;
        TerminalPpoResponseV1::decode_payload(&response, request).map_err(Into::into)
    }

    pub fn publish_checkpoint(
        &mut self,
        request: &CheckpointRequestV1,
    ) -> Result<CheckpointResponseV1, MpsGraphClientError> {
        let payload = request.encode_payload()?;
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &payload,
        )?;
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            self.response_payload_limit,
        )?;
        let response = CheckpointResponseV1::decode_payload(&response, request)?;
        let stored_digest = verify_checkpoint_file(Path::new(request.destination()))?;
        if stored_digest != response.content_sha256() {
            return Err(MpsGraphClientError::CheckpointResponseDigestMismatch {
                response: response.content_sha256(),
                stored: stored_digest,
            });
        }
        Ok(response)
    }

    /// Rebind training progress in the live service, retaining its PID, weights and Adam state.
    pub fn begin_new_generation(
        &mut self,
        request: &TrainingCycleRequestV1,
    ) -> Result<TrainingCycleResponseV1, MpsGraphClientError> {
        let payload = request.encode_payload()?;
        write_frame(
            self.input
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            &payload,
        )?;
        let response = read_frame(
            self.output
                .as_mut()
                .ok_or(MpsGraphClientError::ProcessClosed)?,
            training_cycle::MAXIMUM_TRAINING_CYCLE_PAYLOAD_BYTES,
        )?;
        TrainingCycleResponseV1::decode_payload(&response, request)
    }

    pub fn shutdown(mut self) -> Result<ExitStatus, MpsGraphClientError> {
        drop(self.input.take());
        drop(self.output.take());
        let mut child = self
            .child
            .take()
            .ok_or(MpsGraphClientError::ProcessClosed)?;
        child.wait().map_err(MpsGraphClientError::Io)
    }
}

impl Drop for MpsGraphProcess {
    fn drop(&mut self) {
        drop(self.input.take());
        drop(self.output.take());
        if let Some(mut child) = self.child.take() {
            if child.try_wait().ok().flatten().is_none() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

fn write_frame(writer: &mut impl Write, payload: &[u8]) -> Result<(), MpsGraphClientError> {
    let length = u64::try_from(payload.len()).map_err(|_| MpsGraphClientError::FrameTooLarge)?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(payload)?;
    writer.flush()?;
    Ok(())
}

fn read_frame(
    reader: &mut impl Read,
    maximum_payload: usize,
) -> Result<Vec<u8>, MpsGraphClientError> {
    let mut length = [0; FRAME_LENGTH_BYTES];
    reader.read_exact(&mut length)?;
    let length = u64::from_le_bytes(length);
    if length > maximum_payload as u64 || length > usize::MAX as u64 {
        return Err(MpsGraphClientError::ResponseFrameTooLarge {
            actual: length,
            maximum: maximum_payload,
        });
    }
    let mut payload = vec![0; length as usize];
    reader.read_exact(&mut payload)?;
    Ok(payload)
}

fn response_payload_limit(
    batch_size: usize,
    legal_action_capacity: usize,
) -> Result<usize, MpsGraphClientError> {
    let policy_values = batch_size
        .checked_mul(legal_action_capacity)
        .ok_or(MpsGraphClientError::ResponseSizeOverflow)?;
    let value_values = batch_size
        .checked_mul(3)
        .ok_or(MpsGraphClientError::ResponseSizeOverflow)?;
    let success_bytes = policy_values
        .checked_add(value_values)
        .and_then(|count| count.checked_mul(4))
        .and_then(|bytes| bytes.checked_add(24))
        .ok_or(MpsGraphClientError::ResponseSizeOverflow)?;
    Ok(success_bytes.max(MAXIMUM_ERROR_RESPONSE_BYTES))
}

pub fn verify_checkpoint_file(path: &Path) -> Result<[u8; 32], MpsGraphClientError> {
    let mut file = fs::File::open(path)?;
    let length = file.metadata()?.len();
    if length < 32 {
        return Err(MpsGraphClientError::CheckpointFileTooShort {
            path: path.to_owned(),
            length,
        });
    }
    let content_length = length - 32;
    let mut remaining = content_length;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1_024];
    while remaining > 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| MpsGraphClientError::ResponseSizeOverflow)?;
        let read = file.read(&mut buffer[..wanted])?;
        if read == 0 {
            return Err(MpsGraphClientError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "checkpoint ended before its checksum",
            )));
        }
        hasher.update(&buffer[..read]);
        remaining -= read as u64;
    }
    let mut stored = [0_u8; 32];
    file.read_exact(&mut stored)?;
    let computed: [u8; 32] = hasher.finalize().into();
    if computed != stored {
        return Err(MpsGraphClientError::CheckpointChecksumMismatch(
            path.to_owned(),
        ));
    }
    Ok(stored)
}

#[derive(Debug)]
pub enum MpsGraphClientError {
    ZeroBatchSize,
    ZeroActionCapacity,
    ZeroInferenceSlots,
    BatchTooLarge(usize),
    ActionCapacityTooLarge(usize),
    UnavailableExecutable {
        path: PathBuf,
        source: io::Error,
    },
    ExecutableIsNotAFile(PathBuf),
    UnavailableCheckpoint {
        path: PathBuf,
        source: io::Error,
    },
    CheckpointIsNotAFile(PathBuf),
    NewGenerationWithoutCheckpoint,
    Launch {
        path: PathBuf,
        source: io::Error,
    },
    MissingChildPipe(&'static str),
    ProcessClosed,
    RequestShapeMismatch {
        expected_batch: usize,
        actual_batch: usize,
        expected_capacity: usize,
        actual_capacity: usize,
    },
    FrameTooLarge,
    ResponseFrameTooLarge {
        actual: u64,
        maximum: usize,
    },
    ResponseSizeOverflow,
    Wire(InferenceWireError),
    TrainingWire(TrainingWireError),
    TerminalPpoWire(TerminalPpoWireError),
    CheckpointWire(CheckpointWireError),
    TrainingCycleWire(String),
    WeightsWire(String),
    CheckpointFileTooShort {
        path: PathBuf,
        length: u64,
    },
    CheckpointChecksumMismatch(PathBuf),
    CheckpointResponseDigestMismatch {
        response: [u8; 32],
        stored: [u8; 32],
    },
    Io(io::Error),
}

impl fmt::Display for MpsGraphClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroBatchSize => formatter.write_str("MPSGraph service batch must be positive"),
            Self::ZeroActionCapacity => {
                formatter.write_str("MPSGraph service action capacity must be positive")
            }
            Self::ZeroInferenceSlots => {
                formatter.write_str("MPSGraph service inference slots must be positive")
            }
            Self::BatchTooLarge(value) => write!(formatter, "service batch {value} exceeds V1"),
            Self::ActionCapacityTooLarge(value) => {
                write!(formatter, "service action capacity {value} exceeds V1")
            }
            Self::UnavailableExecutable { path, source } => {
                write!(formatter, "cannot inspect {}: {source}", path.display())
            }
            Self::ExecutableIsNotAFile(path) => {
                write!(formatter, "{} is not a service executable file", path.display())
            }
            Self::UnavailableCheckpoint { path, source } => {
                write!(formatter, "cannot inspect checkpoint {}: {source}", path.display())
            }
            Self::CheckpointIsNotAFile(path) => {
                write!(formatter, "{} is not a checkpoint file", path.display())
            }
            Self::NewGenerationWithoutCheckpoint => {
                formatter.write_str("a new-generation service launch requires a parent checkpoint")
            }
            Self::Launch { path, source } => {
                write!(formatter, "cannot launch {}: {source}", path.display())
            }
            Self::MissingChildPipe(name) => write!(formatter, "child process has no {name} pipe"),
            Self::ProcessClosed => formatter.write_str("MPSGraph service process is closed"),
            Self::RequestShapeMismatch {
                expected_batch,
                actual_batch,
                expected_capacity,
                actual_capacity,
            } => write!(
                formatter,
                "request shape {actual_batch}x{actual_capacity} does not match service {expected_batch}x{expected_capacity}"
            ),
            Self::FrameTooLarge => formatter.write_str("request frame length exceeds u64"),
            Self::ResponseFrameTooLarge { actual, maximum } => write!(
                formatter,
                "response frame has {actual} bytes; maximum is {maximum}"
            ),
            Self::ResponseSizeOverflow => formatter.write_str("response size overflows usize"),
            Self::Wire(source) => source.fmt(formatter),
            Self::TrainingWire(source) => source.fmt(formatter),
            Self::TerminalPpoWire(source) => source.fmt(formatter),
            Self::CheckpointWire(source) => source.fmt(formatter),
            Self::TrainingCycleWire(message) => write!(formatter, "training cycle wire: {message}"),
            Self::WeightsWire(message) => write!(formatter, "weights wire: {message}"),
            Self::CheckpointFileTooShort { path, length } => write!(
                formatter,
                "checkpoint {} has only {length} bytes",
                path.display()
            ),
            Self::CheckpointChecksumMismatch(path) => write!(
                formatter,
                "checkpoint {} failed its independent SHA-256 check",
                path.display()
            ),
            Self::CheckpointResponseDigestMismatch { response, stored } => write!(
                formatter,
                "checkpoint response digest {response:?} does not match stored digest {stored:?}"
            ),
            Self::Io(source) => write!(formatter, "MPSGraph service I/O failed: {source}"),
        }
    }
}

impl std::error::Error for MpsGraphClientError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::UnavailableExecutable { source, .. }
            | Self::UnavailableCheckpoint { source, .. }
            | Self::Launch { source, .. }
            | Self::Io(source) => Some(source),
            Self::Wire(source) => Some(source),
            Self::TrainingWire(source) => Some(source),
            Self::TerminalPpoWire(source) => Some(source),
            Self::CheckpointWire(source) => Some(source),
            _ => None,
        }
    }
}

impl From<InferenceWireError> for MpsGraphClientError {
    fn from(source: InferenceWireError) -> Self {
        Self::Wire(source)
    }
}

impl From<TrainingWireError> for MpsGraphClientError {
    fn from(source: TrainingWireError) -> Self {
        Self::TrainingWire(source)
    }
}

impl From<TerminalPpoWireError> for MpsGraphClientError {
    fn from(source: TerminalPpoWireError) -> Self {
        Self::TerminalPpoWire(source)
    }
}

impl From<CheckpointWireError> for MpsGraphClientError {
    fn from(source: CheckpointWireError) -> Self {
        Self::CheckpointWire(source)
    }
}

impl From<io::Error> for MpsGraphClientError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

pub fn default_service_path(repository: &Path) -> PathBuf {
    repository.join("apple/paisho-mpsgraph/.build/release/paisho-mpsgraph-service")
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_CHECKPOINT_TEST: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn frame_round_trip_is_little_endian_and_bounded() {
        let payload = b"paisho";
        let mut framed = Vec::new();
        write_frame(&mut framed, payload).unwrap();
        assert_eq!(&framed[..8], &6u64.to_le_bytes());
        assert_eq!(read_frame(&mut Cursor::new(framed), 6).unwrap(), payload);
    }

    #[test]
    fn oversized_response_is_rejected_before_allocation() {
        let framed = 9u64.to_le_bytes();
        assert!(matches!(
            read_frame(&mut Cursor::new(framed), 8),
            Err(MpsGraphClientError::ResponseFrameTooLarge {
                actual: 9,
                maximum: 8
            })
        ));
    }

    #[test]
    fn checkpoint_content_is_rehashed_instead_of_trusting_the_response() {
        let sequence = NEXT_CHECKPOINT_TEST.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "paisho-checkpoint-digest-test-{}-{sequence}.psckpt",
            std::process::id()
        ));
        let mut file = b"checkpoint-content".to_vec();
        let digest: [u8; 32] = Sha256::digest(&file).into();
        file.extend_from_slice(&digest);
        fs::write(&path, &file).unwrap();
        assert_eq!(verify_checkpoint_file(&path).unwrap(), digest);

        file[0] ^= 1;
        fs::write(&path, file).unwrap();
        assert!(matches!(
            verify_checkpoint_file(&path),
            Err(MpsGraphClientError::CheckpointChecksumMismatch(actual)) if actual == path
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn new_generation_launch_requires_a_parent_checkpoint_before_process_start() {
        let configuration = ServiceConfiguration {
            executable: PathBuf::from("missing-service"),
            preset: NetworkPreset::Micro,
            batch_size: 1,
            legal_action_capacity: 1,
            inference_slots: 1,
            optimization: OptimizationLevel::Level1,
            seed: 1,
            checkpoint: None,
        };
        assert!(matches!(
            MpsGraphProcess::launch_new_generation(configuration),
            Err(MpsGraphClientError::NewGenerationWithoutCheckpoint)
        ));
    }
}
