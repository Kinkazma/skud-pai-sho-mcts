use paisho_core::{legal_actions, BasicFlower, Position, StandardSetup};
use paisho_model::{
    InferenceRequestV1, InferenceResponseV1, InferenceWireError, GLOBAL_FEATURE_COUNT_V1,
    INFERENCE_ERROR_MAGIC_V1, INFERENCE_REQUEST_MAGIC_V1, INFERENCE_RESPONSE_MAGIC_V1,
    SPATIAL_VALUE_COUNT_V1,
};

fn two_positions() -> [Position; 2] {
    let mut next = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let opening = next.clone();
    next.apply(legal_actions(&next)[0]).unwrap();
    [opening, next]
}

#[test]
fn request_serializes_real_engine_encodings_in_v1_order() {
    let positions = two_positions();
    let request = InferenceRequestV1::from_positions(77, &positions, 512).unwrap();
    let payload = request.encode_payload().unwrap();
    let mut cursor = 0;

    assert_eq!(take::<8>(&payload, &mut cursor), INFERENCE_REQUEST_MAGIC_V1);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 77);
    assert_eq!(u32::from_le_bytes(take(&payload, &mut cursor)), 2);
    assert_eq!(u32::from_le_bytes(take(&payload, &mut cursor)), 512);
    for example in request.examples() {
        for expected in example.state().spatial_nhwc() {
            assert_eq!(
                f32::from_bits(u32::from_le_bytes(take(&payload, &mut cursor))),
                *expected
            );
        }
        for expected in example.state().global() {
            assert_eq!(
                f32::from_bits(u32::from_le_bytes(take(&payload, &mut cursor))),
                *expected
            );
        }
        assert_eq!(
            u32::from_le_bytes(take(&payload, &mut cursor)) as usize,
            example.legal_actions().len()
        );
        for action in example.legal_actions() {
            for expected in action.slots() {
                assert_eq!(u16::from_le_bytes(take(&payload, &mut cursor)), expected);
            }
        }
    }
    assert_eq!(cursor, payload.len());
    assert_eq!(request.batch_size(), 2);
    assert_eq!(request.legal_action_capacity(), 512);
    assert_eq!(
        request.examples()[0].state().spatial_nhwc().len(),
        SPATIAL_VALUE_COUNT_V1
    );
    assert_eq!(
        request.examples()[0].state().global().len(),
        GLOBAL_FEATURE_COUNT_V1
    );
}

#[test]
fn response_is_matched_to_request_and_padding_is_removed() {
    let positions = two_positions();
    let request = InferenceRequestV1::from_positions(81, &positions, 512).unwrap();
    let payload = valid_response(&request);
    let response = InferenceResponseV1::decode_payload(&payload, &request).unwrap();

    assert_eq!(response.request_id(), 81);
    assert_eq!(response.outputs().len(), 2);
    for (output, example) in response.outputs().iter().zip(request.examples()) {
        assert_eq!(
            output.policy_probabilities().len(),
            example.legal_actions().len()
        );
        assert!((output.policy_probabilities().iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert_eq!(output.value_probabilities(), &[0.2, 0.3, 0.5]);
    }
}

#[test]
fn malformed_and_error_responses_are_rejected() {
    let positions = two_positions();
    let request = InferenceRequestV1::from_positions(91, &positions, 512).unwrap();

    let mut wrong_id = valid_response(&request);
    wrong_id[8..16].copy_from_slice(&92u64.to_le_bytes());
    assert!(matches!(
        InferenceResponseV1::decode_payload(&wrong_id, &request),
        Err(InferenceWireError::ResponseRequestIdMismatch {
            expected: 91,
            actual: 92
        })
    ));

    let mut bad_padding = valid_response(&request);
    let first_padding = 24 + request.examples()[0].legal_actions().len() * 4;
    bad_padding[first_padding..first_padding + 4].copy_from_slice(&0.5f32.to_bits().to_le_bytes());
    assert!(matches!(
        InferenceResponseV1::decode_payload(&bad_padding, &request),
        Err(InferenceWireError::NonZeroPadding { row: 0, .. })
    ));

    assert!(matches!(
        InferenceResponseV1::decode_payload(&valid_response(&request)[..20], &request),
        Err(InferenceWireError::TruncatedResponse)
    ));

    let mut service_error = Vec::new();
    service_error.extend_from_slice(&INFERENCE_ERROR_MAGIC_V1);
    service_error.extend_from_slice(&request.request_id().to_le_bytes());
    service_error.extend_from_slice(&4u32.to_le_bytes());
    service_error.extend_from_slice(b"boom");
    assert!(matches!(
        InferenceResponseV1::decode_payload(&service_error, &request),
        Err(InferenceWireError::Service(message)) if message == "boom"
    ));
}

#[test]
fn request_rejects_a_capacity_below_the_real_legal_count() {
    let positions = two_positions();
    let actual = legal_actions(&positions[0]).len();
    let error = InferenceRequestV1::from_positions(1, &positions[..1], actual - 1).unwrap_err();
    assert!(matches!(
        error,
        InferenceWireError::TooManyLegalActions {
            row: 0,
            capacity,
            actual: found
        } if capacity == actual - 1 && found == actual
    ));
}

fn valid_response(request: &InferenceRequestV1) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(&INFERENCE_RESPONSE_MAGIC_V1);
    payload.extend_from_slice(&request.request_id().to_le_bytes());
    payload.extend_from_slice(&(request.batch_size() as u32).to_le_bytes());
    payload.extend_from_slice(&(request.legal_action_capacity() as u32).to_le_bytes());
    for example in request.examples() {
        let probability = 1.0 / example.legal_actions().len() as f32;
        for index in 0..request.legal_action_capacity() {
            let value = if index < example.legal_actions().len() {
                probability
            } else {
                0.0
            };
            payload.extend_from_slice(&f32::to_bits(value).to_le_bytes());
        }
    }
    for _ in request.examples() {
        for value in [0.2f32, 0.3, 0.5] {
            payload.extend_from_slice(&value.to_bits().to_le_bytes());
        }
    }
    payload
}

fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> [u8; N] {
    let end = *cursor + N;
    let value = bytes[*cursor..end].try_into().unwrap();
    *cursor = end;
    value
}
