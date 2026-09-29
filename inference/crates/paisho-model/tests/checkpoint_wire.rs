use paisho_model::{
    CheckpointRandomStateV1, CheckpointRequestV1, CheckpointResponseV1, CheckpointWireError,
    CHECKPOINT_ERROR_MAGIC_V1, CHECKPOINT_REQUEST_MAGIC_V1, CHECKPOINT_RESPONSE_MAGIC_V1,
};

fn request() -> CheckpointRequestV1 {
    CheckpointRequestV1::new(
        41,
        12,
        [0xab; 32],
        9_876,
        7,
        1.0e-4,
        vec![
            CheckpointRandomStateV1::new("actor-seed-cursor", 91).unwrap(),
            CheckpointRandomStateV1::new("replay-sampler", 92).unwrap(),
        ],
        "/tmp/generation-0007-step-0012.psckpt",
    )
    .unwrap()
}

#[test]
fn request_serializes_checkpoint_progress_in_v1_order() {
    let request = request();
    let payload = request.encode_payload().unwrap();
    let mut cursor = 0;

    assert_eq!(
        take::<8>(&payload, &mut cursor),
        CHECKPOINT_REQUEST_MAGIC_V1
    );
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 41);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 12);
    assert_eq!(take::<32>(&payload, &mut cursor), [0xab; 32]);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 9_876);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 7);
    assert_eq!(
        f32::from_bits(u32::from_le_bytes(take(&payload, &mut cursor))),
        1.0e-4
    );
    assert_eq!(u32::from_le_bytes(take(&payload, &mut cursor)), 2);
    for expected in request.random_states() {
        let length = u32::from_le_bytes(take(&payload, &mut cursor)) as usize;
        assert_eq!(
            bytes(&payload, &mut cursor, length),
            expected.name().as_bytes()
        );
        assert_eq!(
            u64::from_le_bytes(take(&payload, &mut cursor)),
            expected.state()
        );
    }
    let destination_length = u32::from_le_bytes(take(&payload, &mut cursor)) as usize;
    assert_eq!(
        bytes(&payload, &mut cursor, destination_length),
        request.destination().as_bytes()
    );
    assert_eq!(cursor, payload.len());
}

#[test]
fn response_is_bound_to_the_requested_durable_progress() {
    let request = request();
    let mut payload = Vec::new();
    payload.extend_from_slice(&CHECKPOINT_RESPONSE_MAGIC_V1);
    payload.extend_from_slice(&request.request_id().to_le_bytes());
    payload.extend_from_slice(&request.expected_training_step().to_le_bytes());
    payload.extend_from_slice(&request.replay_index().to_le_bytes());
    payload.extend_from_slice(&[0xcd; 32]);

    let response = CheckpointResponseV1::decode_payload(&payload, &request).unwrap();
    assert_eq!(response.request_id(), 41);
    assert_eq!(response.completed_training_step(), 12);
    assert_eq!(response.completed_replay_index(), 9_876);
    assert_eq!(response.content_sha256(), [0xcd; 32]);

    payload[24..32].copy_from_slice(&9_877_u64.to_le_bytes());
    assert!(matches!(
        CheckpointResponseV1::decode_payload(&payload, &request),
        Err(CheckpointWireError::ResponseReplayIndexMismatch {
            expected: 9_876,
            actual: 9_877,
        })
    ));
}

#[test]
fn malformed_metadata_and_service_errors_are_explicit() {
    let random_state = CheckpointRandomStateV1::new("same", 1).unwrap();
    assert!(matches!(
        CheckpointRequestV1::new(
            1,
            0,
            [0; 32],
            0,
            0,
            1.0e-4,
            vec![random_state.clone(), random_state],
            "/tmp/a.psckpt",
        ),
        Err(CheckpointWireError::DuplicateRandomState(name)) if name == "same"
    ));
    assert!(matches!(
        CheckpointRequestV1::new(1, 0, [0; 32], 0, 0, f32::NAN, Vec::new(), ""),
        Err(CheckpointWireError::InvalidLearningRate(value)) if value.is_nan()
    ));
    let oversized_states = (0..16)
        .map(|index| {
            let mut name = format!("{index:04}");
            name.push_str(&"x".repeat(4_092));
            CheckpointRandomStateV1::new(name, index).unwrap()
        })
        .collect();
    assert!(matches!(
        CheckpointRequestV1::new(
            1,
            0,
            [0; 32],
            0,
            0,
            1.0e-4,
            oversized_states,
            "/tmp/a.psckpt",
        ),
        Err(CheckpointWireError::PayloadExceedsMaximum(_))
    ));

    let request = request();
    let mut error = Vec::new();
    error.extend_from_slice(&CHECKPOINT_ERROR_MAGIC_V1);
    error.extend_from_slice(&request.request_id().to_le_bytes());
    error.extend_from_slice(&4_u32.to_le_bytes());
    error.extend_from_slice(b"boom");
    assert!(matches!(
        CheckpointResponseV1::decode_payload(&error, &request),
        Err(CheckpointWireError::Service(message)) if message == "boom"
    ));
}

fn bytes<'a>(payload: &'a [u8], cursor: &mut usize, count: usize) -> &'a [u8] {
    let value = &payload[*cursor..*cursor + count];
    *cursor += count;
    value
}

fn take<const N: usize>(payload: &[u8], cursor: &mut usize) -> [u8; N] {
    let value = payload[*cursor..*cursor + N].try_into().unwrap();
    *cursor += N;
    value
}
