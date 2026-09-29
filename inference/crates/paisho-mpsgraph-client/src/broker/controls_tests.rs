//! Opt-in Metal tests. Run with the campaign pause wrapper and
//! PAISHO_WEIGHTS_TEST_SERVICE pointing to the built Swift service.
use super::*;
use crate::{NetworkPreset, OptimizationLevel};
use paisho_core::{BasicFlower, Position, StandardSetup};

const DEADLINE: Duration = Duration::from_secs(30);

fn assert_output(actual: &InferenceOutputV1, expected: &InferenceOutputV1) {
    assert_eq!(
        actual.policy_probabilities().len(),
        expected.policy_probabilities().len()
    );
    for (actual, expected) in actual
        .policy_probabilities()
        .iter()
        .chain(actual.value_probabilities())
        .zip(
            expected
                .policy_probabilities()
                .iter()
                .chain(expected.value_probabilities()),
        )
    {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "output differs: {actual} versus {expected}"
        );
    }
}

fn queued_infer(
    sender: &Sender<BrokerMessage>,
    example: &InferenceExampleV1,
) -> Receiver<Result<InferenceOutputV1, InferenceBrokerError>> {
    let (response, receiver) = mpsc::sync_channel(1);
    sender
        .send(BrokerMessage::Infer(InferenceJob {
            example: example.clone(),
            response,
        }))
        .unwrap();
    receiver
}

fn exercise_controls(prepare_ahead: bool) {
    let service = std::env::var_os("PAISHO_WEIGHTS_TEST_SERVICE")
        .expect("set PAISHO_WEIGHTS_TEST_SERVICE to the built Swift service");
    let configuration = ServiceConfiguration {
        executable: service.into(),
        preset: NetworkPreset::Micro,
        batch_size: 2,
        legal_action_capacity: 128,
        inference_slots: 2,
        optimization: OptimizationLevel::Level1,
        seed: 809,
        checkpoint: None,
    };
    let mut source_configuration = configuration.clone();
    source_configuration.seed = 811;
    source_configuration.batch_size = 1;
    let mut source = MpsGraphProcess::launch(source_configuration).unwrap();
    let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let example = InferenceExampleV1::from_position(&position).unwrap();
    let request = InferenceRequestV1::from_examples(1, vec![example.clone()], 128).unwrap();
    let expected_new = source.infer(&request).unwrap().into_outputs().remove(0);
    let new_weights = source.export_weights(2).unwrap();
    assert!(source.shutdown().unwrap().success());

    let mut broker = InferenceBroker::launch(
        configuration,
        InferenceBrokerConfiguration {
            maximum_batch_wait: Duration::from_millis(2),
            maximum_in_flight_batches: 2,
            prepare_ahead,
        },
    )
    .unwrap();
    let client = broker.client().unwrap();

    // Control on an empty queue, including before the first inference.
    let (snapshot_sender, snapshot_receiver) = mpsc::sync_channel(1);
    broker
        .update_service(move |process| {
            snapshot_sender
                .send((process.process_id(), process.export_weights(3)?))
                .unwrap();
            Ok(())
        })
        .unwrap();
    let (pid, old_weights) = snapshot_receiver.recv_timeout(DEADLINE).unwrap();
    assert_ne!(old_weights.content_sha256(), new_weights.content_sha256());
    let expected_old = client.infer_encoded(example.clone()).unwrap();
    let replacement = new_weights.clone();
    broker
        .update_service(move |process| {
            assert_eq!(process.process_id(), pid);
            process.import_weights(4, &replacement)?;
            assert_eq!(process.export_weights(5)?, replacement);
            Ok(())
        })
        .unwrap();
    assert_output(
        &client.infer_encoded(example.clone()).unwrap(),
        &expected_new,
    );

    // Hold a control at the consumer to enqueue an exact FIFO sequence without sleeps:
    // five old-version requests (two full batches + a partial), import, three new requests.
    let sender = broker.sender.as_ref().unwrap();
    let (entered, entered_receiver) = mpsc::sync_channel(1);
    let (release, release_receiver) = mpsc::sync_channel(1);
    let (hold_response, hold_receiver) = mpsc::sync_channel(1);
    sender
        .send(BrokerMessage::Control(ServiceOperation {
            update: Box::new(move |_| {
                entered.send(()).unwrap();
                release_receiver.recv_timeout(DEADLINE).unwrap();
                Ok(())
            }),
            response: hold_response,
        }))
        .unwrap();
    entered_receiver.recv_timeout(DEADLINE).unwrap();
    let before: Vec<_> = (0..5).map(|_| queued_infer(sender, &example)).collect();
    let (update_response, update_receiver) = mpsc::sync_channel(1);
    sender
        .send(BrokerMessage::Control(ServiceOperation {
            update: Box::new(move |process| {
                assert_eq!(process.process_id(), pid);
                process.import_weights(6, &old_weights)?;
                assert_eq!(process.export_weights(7)?, old_weights);
                Ok(())
            }),
            response: update_response,
        }))
        .unwrap();
    let after: Vec<_> = (0..3).map(|_| queued_infer(sender, &example)).collect();
    release.send(()).unwrap();
    hold_receiver.recv_timeout(DEADLINE).unwrap().unwrap();
    for result in before {
        assert_output(
            &result.recv_timeout(DEADLINE).unwrap().unwrap(),
            &expected_new,
        );
    }
    update_receiver.recv_timeout(DEADLINE).unwrap().unwrap();
    for result in after {
        assert_output(
            &result.recv_timeout(DEADLINE).unwrap().unwrap(),
            &expected_old,
        );
    }
    // Public control path and public infer still work after the queue-draining transition.
    broker
        .update_service(move |process| {
            assert_eq!(process.process_id(), pid);
            process.import_weights(8, &new_weights)?;
            Ok(())
        })
        .unwrap();
    assert_output(&client.infer_encoded(example).unwrap(), &expected_new);
    drop(client);
    let telemetry = broker.shutdown().unwrap();
    assert_eq!(telemetry.requested_positions, 11);
}

#[test]
#[ignore = "Metal broker controls: run with campaign pause wrapper"]
fn service_controls_default_empty_and_draining_queue() {
    exercise_controls(false);
}

#[test]
#[ignore = "Metal broker controls: run with campaign pause wrapper"]
fn service_controls_prefetch_empty_and_draining_queue() {
    exercise_controls(true);
}
