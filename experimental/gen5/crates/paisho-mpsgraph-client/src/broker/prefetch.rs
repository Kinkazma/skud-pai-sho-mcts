//! Prepare the next batch on CPU while the service executes the current one.
use super::*;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

enum Prepared {
    Batch(PendingBatch),
    Control(ServiceOperation),
}

pub(super) fn run(
    mut process: MpsGraphProcess,
    receiver: Receiver<BrokerMessage>,
    batch_size: usize,
    capacity: usize,
    wait: Duration,
) -> Result<InferenceBrokerTelemetry, String> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let cancellation = cancelled.clone();
    let (prepared, batches) = mpsc::sync_channel(1);
    let producer = thread::Builder::new()
        .name(format!("paisho-pack-{capacity}"))
        .spawn(move || {
            let mut request_id = 0;
            while !cancellation.load(Ordering::Relaxed) {
                let first = match receiver.recv_timeout(Duration::from_millis(10)) {
                    Ok(BrokerMessage::Infer(job)) => job,
                    Ok(BrokerMessage::Control(operation)) => {
                        if prepared.send(Prepared::Control(operation)).is_err() {
                            break;
                        }
                        continue;
                    }
                    Ok(BrokerMessage::Shutdown) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => continue,
                };
                let (jobs, following) = collect_jobs(first, &receiver, batch_size, wait);
                let real_count = jobs.len();
                let (mut examples, responses): (Vec<_>, Vec<_>) = jobs
                    .into_iter()
                    .map(|job| (job.example, job.response))
                    .unzip();
                examples.resize(batch_size, examples[0].clone());
                let request =
                    match InferenceRequestV1::from_examples(request_id, examples, capacity) {
                        Ok(request) => request,
                        Err(error) => {
                            send_failure(&responses, error.to_string());
                            return Err(error.to_string());
                        }
                    };
                if prepared
                    .send(Prepared::Batch(PendingBatch {
                        request,
                        responses,
                        real_count,
                    }))
                    .is_err()
                {
                    break;
                }
                request_id = request_id
                    .checked_add(1)
                    .ok_or("broker request id overflow")?;
                match following {
                    Some(BrokerMessage::Shutdown) => break,
                    Some(BrokerMessage::Control(operation)) => {
                        if prepared.send(Prepared::Control(operation)).is_err() {
                            break;
                        }
                    }
                    Some(BrokerMessage::Infer(_)) => {
                        unreachable!("collector returns only control messages")
                    }
                    None => {}
                }
            }
            Ok::<(), String>(())
        })
        .map_err(|error| error.to_string())?;

    let result: Result<InferenceBrokerTelemetry, String> = (|| {
        let mut telemetry = InferenceBrokerTelemetry::default();
        while let Ok(prepared) = batches.recv() {
            let batch = match prepared {
                Prepared::Batch(batch) => batch,
                Prepared::Control(operation) => {
                    apply_control(operation, &mut process)?;
                    continue;
                }
            };
            let response = process.infer(&batch.request).map_err(|error| {
                send_failure(&batch.responses, error.to_string());
                error.to_string()
            })?;
            for (destination, output) in batch.responses.into_iter().zip(response.into_outputs()) {
                let _ = destination.send(Ok(output));
            }
            telemetry.batches += 1;
            telemetry.full_batches += u64::from(batch.real_count == batch_size);
            telemetry.requested_positions += batch.real_count as u64;
            telemetry.executed_positions += batch_size as u64;
            telemetry.padded_positions += (batch_size - batch.real_count) as u64;
            telemetry.maximum_observed_batch =
                telemetry.maximum_observed_batch.max(batch.real_count);
            telemetry.maximum_observed_in_flight_batches = 1;
        }
        Ok(telemetry)
    })();
    cancelled.store(true, Ordering::Relaxed);
    drop(batches); // Also wakes a producer blocked on the bounded queue after an error.
    let producer_result = producer
        .join()
        .map_err(|_| "batch producer panicked".to_owned())?;
    let telemetry = result?;
    producer_result?;
    let status = process.shutdown().map_err(|error| error.to_string())?;
    if !status.success() {
        return Err(format!("MPSGraph service exited with {status}"));
    }
    Ok(telemetry)
}
