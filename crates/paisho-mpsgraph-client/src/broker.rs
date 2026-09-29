use std::collections::VecDeque;
use std::fmt;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender, SyncSender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use paisho_model::{
    InferenceExampleEncodingError, InferenceExampleV1, InferenceOutputV1, InferenceRequestV1,
};

use crate::{MpsGraphClientError, MpsGraphProcess, ServiceConfiguration};

mod prefetch;

#[cfg(test)]
#[path = "broker/controls_tests.rs"]
mod controls_tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InferenceBrokerConfiguration {
    pub maximum_batch_wait: Duration,
    pub maximum_in_flight_batches: usize,
    pub prepare_ahead: bool,
}

impl Default for InferenceBrokerConfiguration {
    fn default() -> Self {
        Self {
            maximum_batch_wait: Duration::from_millis(5),
            maximum_in_flight_batches: 1,
            prepare_ahead: false,
        }
    }
}

pub struct InferenceBroker {
    sender: Option<Sender<BrokerMessage>>,
    worker: Option<JoinHandle<Result<InferenceBrokerTelemetry, String>>>,
    legal_action_capacity: usize,
}

impl InferenceBroker {
    pub fn launch(
        service_configuration: ServiceConfiguration,
        broker_configuration: InferenceBrokerConfiguration,
    ) -> Result<Self, InferenceBrokerError> {
        if broker_configuration.maximum_in_flight_batches == 0 {
            return Err(InferenceBrokerError::ZeroInFlightBatches);
        }
        if service_configuration.inference_slots < broker_configuration.maximum_in_flight_batches {
            return Err(InferenceBrokerError::InsufficientInferenceSlots {
                slots: service_configuration.inference_slots,
                in_flight: broker_configuration.maximum_in_flight_batches,
            });
        }
        let batch_size = service_configuration.batch_size;
        let legal_action_capacity = service_configuration.legal_action_capacity;
        let process = MpsGraphProcess::launch(service_configuration)
            .map_err(|error| InferenceBrokerError::Worker(error.to_string()))?;
        let (sender, receiver) = mpsc::channel();
        let worker = thread::Builder::new()
            .name(format!("paisho-mpsgraph-broker-{legal_action_capacity}"))
            .spawn(move || {
                if broker_configuration.prepare_ahead {
                    return prefetch::run(
                        process,
                        receiver,
                        batch_size,
                        legal_action_capacity,
                        broker_configuration.maximum_batch_wait,
                    );
                }
                run_worker(
                    process,
                    receiver,
                    batch_size,
                    legal_action_capacity,
                    broker_configuration.maximum_batch_wait,
                    broker_configuration.maximum_in_flight_batches,
                )
            })
            .map_err(InferenceBrokerError::Spawn)?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
            legal_action_capacity,
        })
    }

    pub fn client(&self) -> Result<InferenceBrokerClient, InferenceBrokerError> {
        Ok(InferenceBrokerClient {
            sender: self
                .sender
                .as_ref()
                .ok_or(InferenceBrokerError::Closed)?
                .clone(),
            legal_action_capacity: self.legal_action_capacity,
        })
    }

    pub fn shutdown(mut self) -> Result<InferenceBrokerTelemetry, InferenceBrokerError> {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(BrokerMessage::Shutdown);
        }
        join_worker(self.worker.take())
    }

    /// Update a persistent service between actor rounds. Callers must stop issuing
    /// game requests first so a single game never observes two model versions.
    pub fn update_service(
        &mut self,
        update: impl FnOnce(&mut MpsGraphProcess) -> Result<(), MpsGraphClientError> + Send + 'static,
    ) -> Result<(), InferenceBrokerError> {
        let (response, receiver) = mpsc::sync_channel(1);
        self.sender
            .as_ref()
            .ok_or(InferenceBrokerError::Closed)?
            .send(BrokerMessage::Control(ServiceOperation {
                update: Box::new(update),
                response,
            }))
            .map_err(|_| InferenceBrokerError::Closed)?;
        receiver
            .recv()
            .map_err(|_| InferenceBrokerError::Closed)?
            .map_err(InferenceBrokerError::Worker)
    }
}

impl Drop for InferenceBroker {
    fn drop(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(BrokerMessage::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
pub struct InferenceBrokerClient {
    sender: Sender<BrokerMessage>,
    legal_action_capacity: usize,
}

impl InferenceBrokerClient {
    pub fn infer(
        &self,
        position: &paisho_core::Position,
    ) -> Result<InferenceOutputV1, InferenceBrokerError> {
        let example =
            InferenceExampleV1::from_position(position).map_err(InferenceBrokerError::Encoding)?;
        self.infer_encoded(example)
    }

    pub fn infer_encoded(
        &self,
        example: InferenceExampleV1,
    ) -> Result<InferenceOutputV1, InferenceBrokerError> {
        if example.legal_actions().len() > self.legal_action_capacity {
            return Err(InferenceBrokerError::TooManyLegalActions {
                capacity: self.legal_action_capacity,
                actual: example.legal_actions().len(),
            });
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(BrokerMessage::Infer(InferenceJob {
                example,
                response: sender,
            }))
            .map_err(|_| InferenceBrokerError::Closed)?;
        receiver.recv().map_err(|_| InferenceBrokerError::Closed)?
    }
}

struct InferenceJob {
    example: InferenceExampleV1,
    response: SyncSender<Result<InferenceOutputV1, InferenceBrokerError>>,
}

struct PendingBatch {
    request: InferenceRequestV1,
    responses: Vec<SyncSender<Result<InferenceOutputV1, InferenceBrokerError>>>,
    real_count: usize,
}

enum BrokerMessage {
    Infer(InferenceJob),
    Control(ServiceOperation),
    Shutdown,
}

type ServiceUpdate =
    Box<dyn FnOnce(&mut MpsGraphProcess) -> Result<(), MpsGraphClientError> + Send>;
struct ServiceOperation {
    update: ServiceUpdate,
    response: SyncSender<Result<(), String>>,
}

fn apply_control(operation: ServiceOperation, process: &mut MpsGraphProcess) -> Result<(), String> {
    let result = (operation.update)(process).map_err(|error| error.to_string());
    let _ = operation.response.send(result.clone());
    result
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InferenceBrokerTelemetry {
    pub batches: u64,
    pub full_batches: u64,
    pub requested_positions: u64,
    pub executed_positions: u64,
    pub padded_positions: u64,
    pub maximum_observed_batch: usize,
    pub maximum_observed_in_flight_batches: usize,
}

fn run_worker(
    mut process: MpsGraphProcess,
    receiver: Receiver<BrokerMessage>,
    batch_size: usize,
    legal_action_capacity: usize,
    maximum_batch_wait: Duration,
    maximum_in_flight_batches: usize,
) -> Result<InferenceBrokerTelemetry, String> {
    let mut telemetry = InferenceBrokerTelemetry::default();
    let mut request_id = 0u64;
    let mut stopping = false;
    let mut pending = VecDeque::with_capacity(maximum_in_flight_batches);
    let mut deferred = None;

    while !stopping || !pending.is_empty() {
        while !stopping && pending.len() < maximum_in_flight_batches {
            let message = if let Some(message) = deferred.take() {
                Ok(message)
            } else if pending.is_empty() {
                receiver.recv().map_err(|_| TryRecvError::Disconnected)
            } else {
                receiver.try_recv()
            };
            let first = match message {
                Ok(BrokerMessage::Infer(job)) => job,
                Ok(BrokerMessage::Control(operation)) => {
                    if !pending.is_empty() {
                        deferred = Some(BrokerMessage::Control(operation));
                        break;
                    }
                    apply_control(operation, &mut process)?;
                    continue;
                }
                Ok(BrokerMessage::Shutdown) | Err(TryRecvError::Disconnected) => {
                    stopping = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
            };
            let (jobs, next_message) =
                collect_jobs(first, &receiver, batch_size, maximum_batch_wait);
            deferred = next_message;
            let real_count = jobs.len();
            let (mut examples, responses): (Vec<_>, Vec<_>) = jobs
                .into_iter()
                .map(|job| (job.example, job.response))
                .unzip();
            examples.resize(batch_size, examples[0].clone());
            let request = match InferenceRequestV1::from_examples(
                request_id,
                examples,
                legal_action_capacity,
            ) {
                Ok(request) => request,
                Err(error) => {
                    let message = error.to_string();
                    send_failure(&responses, message.clone());
                    send_pending_failures(&pending, &message);
                    return Err(message);
                }
            };
            if let Err(error) = process.send_inference(&request) {
                let message = error.to_string();
                send_failure(&responses, message.clone());
                send_pending_failures(&pending, &message);
                return Err(message);
            }
            pending.push_back(PendingBatch {
                request,
                responses,
                real_count,
            });
            telemetry.maximum_observed_in_flight_batches = telemetry
                .maximum_observed_in_flight_batches
                .max(pending.len());
            request_id = match request_id.checked_add(1) {
                Some(next) => next,
                None => {
                    let message = "MPSGraph broker request id overflow".to_owned();
                    send_pending_failures(&pending, &message);
                    return Err(message);
                }
            };
        }

        let Some(completed) = pending.pop_front() else {
            continue;
        };
        let response = match process.receive_inference(&completed.request) {
            Ok(response) => response,
            Err(error) => {
                let message = error.to_string();
                send_failure(&completed.responses, message.clone());
                send_pending_failures(&pending, &message);
                return Err(message);
            }
        };
        let outputs = response.into_outputs();
        for (response, output) in completed.responses.into_iter().zip(outputs.into_iter()) {
            let _ = response.send(Ok(output));
        }

        telemetry.batches += 1;
        telemetry.full_batches += u64::from(completed.real_count == batch_size);
        telemetry.requested_positions += completed.real_count as u64;
        telemetry.executed_positions += batch_size as u64;
        telemetry.padded_positions += (batch_size - completed.real_count) as u64;
        telemetry.maximum_observed_batch =
            telemetry.maximum_observed_batch.max(completed.real_count);
    }

    let status = process.shutdown().map_err(|error| error.to_string())?;
    if status.success() {
        Ok(telemetry)
    } else {
        Err(format!("MPSGraph service exited with {status}"))
    }
}

fn collect_jobs(
    first: InferenceJob,
    receiver: &Receiver<BrokerMessage>,
    batch_size: usize,
    maximum_batch_wait: Duration,
) -> (Vec<InferenceJob>, Option<BrokerMessage>) {
    let mut jobs = Vec::with_capacity(batch_size);
    jobs.push(first);
    let batch_started = Instant::now();
    while jobs.len() < batch_size {
        let remaining = maximum_batch_wait.saturating_sub(batch_started.elapsed());
        match receiver.recv_timeout(remaining) {
            Ok(BrokerMessage::Infer(job)) => jobs.push(job),
            Ok(message) => return (jobs, Some(message)),
            Err(RecvTimeoutError::Disconnected) => return (jobs, Some(BrokerMessage::Shutdown)),
            Err(RecvTimeoutError::Timeout) => break,
        }
    }
    (jobs, None)
}

fn send_failure(
    responses: &[SyncSender<Result<InferenceOutputV1, InferenceBrokerError>>],
    message: String,
) {
    for response in responses {
        let _ = response.send(Err(InferenceBrokerError::Worker(message.clone())));
    }
}

fn send_pending_failures(pending: &VecDeque<PendingBatch>, message: &str) {
    for batch in pending {
        send_failure(&batch.responses, message.to_owned());
    }
}

fn join_worker(
    worker: Option<JoinHandle<Result<InferenceBrokerTelemetry, String>>>,
) -> Result<InferenceBrokerTelemetry, InferenceBrokerError> {
    worker
        .ok_or(InferenceBrokerError::Closed)?
        .join()
        .map_err(|_| InferenceBrokerError::WorkerPanicked)?
        .map_err(InferenceBrokerError::Worker)
}

#[derive(Debug)]
pub enum InferenceBrokerError {
    Spawn(std::io::Error),
    Encoding(InferenceExampleEncodingError),
    NoCapacityClasses,
    DuplicateCapacityClass(usize),
    ZeroCapacityClassLanes { capacity: usize },
    ZeroInFlightBatches,
    InsufficientInferenceSlots { slots: usize, in_flight: usize },
    InconsistentCapacityModel { capacity: usize },
    TooManyLegalActions { capacity: usize, actual: usize },
    Closed,
    Worker(String),
    WorkerPanicked,
}

impl fmt::Display for InferenceBrokerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn(source) => write!(formatter, "cannot spawn inference broker: {source}"),
            Self::Encoding(source) => source.fmt(formatter),
            Self::NoCapacityClasses => {
                formatter.write_str("capacity inference broker needs at least one class")
            }
            Self::DuplicateCapacityClass(capacity) => {
                write!(formatter, "duplicate inference capacity class {capacity}")
            }
            Self::ZeroCapacityClassLanes { capacity } => write!(
                formatter,
                "inference capacity class {capacity} needs at least one lane"
            ),
            Self::ZeroInFlightBatches => {
                formatter.write_str("inference broker needs at least one in-flight batch")
            }
            Self::InsufficientInferenceSlots { slots, in_flight } => write!(
                formatter,
                "inference service has {slots} slots but broker requests {in_flight} batches in flight"
            ),
            Self::InconsistentCapacityModel { capacity } => write!(
                formatter,
                "inference capacity class {capacity} does not run the same model"
            ),
            Self::TooManyLegalActions { capacity, actual } => write!(
                formatter,
                "inference job has {actual} legal actions; service capacity is {capacity}"
            ),
            Self::Closed => formatter.write_str("inference broker is closed"),
            Self::Worker(message) => write!(formatter, "inference broker failed: {message}"),
            Self::WorkerPanicked => formatter.write_str("inference broker worker panicked"),
        }
    }
}

impl std::error::Error for InferenceBrokerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn(source) => Some(source),
            Self::Encoding(source) => Some(source),
            _ => None,
        }
    }
}
