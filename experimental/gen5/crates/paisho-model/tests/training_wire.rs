use paisho_core::{legal_actions, BasicFlower, Position, StandardSetup};
use paisho_model::{
    InferenceExampleV1, TrainingExampleV1, TrainingRequestV1, TrainingResponseV1,
    TrainingWireError, GLOBAL_FEATURE_COUNT_V1, SPATIAL_VALUE_COUNT_V1, TRAINING_ERROR_MAGIC_V1,
    TRAINING_REQUEST_MAGIC_V1, TRAINING_RESPONSE_MAGIC_V1,
};

fn training_examples() -> Vec<TrainingExampleV1> {
    let mut position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let mut examples = Vec::new();
    for value_target in [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] {
        let inference = InferenceExampleV1::from_position(&position).unwrap();
        let mut policy = vec![0.0; inference.legal_actions().len()];
        policy[0] = 1.0;
        examples.push(TrainingExampleV1::new(inference, policy, value_target).unwrap());
        let action = legal_actions(&position)[0];
        position.apply(action).unwrap();
    }
    examples
}

#[test]
fn request_serializes_real_training_examples_in_v1_order() {
    let request =
        TrainingRequestV1::new(71, 12, 1.0e-4, 512, [0xab; 32], 900, training_examples()).unwrap();
    let payload = request.encode_payload().unwrap();
    let mut cursor = 0;

    assert_eq!(take::<8>(&payload, &mut cursor), TRAINING_REQUEST_MAGIC_V1);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 71);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 12);
    assert_eq!(f32_value(&payload, &mut cursor), 1.0e-4);
    assert_eq!(u32::from_le_bytes(take(&payload, &mut cursor)), 2);
    assert_eq!(u32::from_le_bytes(take(&payload, &mut cursor)), 512);
    assert_eq!(take::<32>(&payload, &mut cursor), [0xab; 32]);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 900);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 902);
    for example in request.examples() {
        assert_eq!(
            example.inference().state().spatial_nhwc().len(),
            SPATIAL_VALUE_COUNT_V1
        );
        assert_eq!(
            example.inference().state().global().len(),
            GLOBAL_FEATURE_COUNT_V1
        );
        for &expected in example.inference().state().spatial_nhwc() {
            assert_eq!(f32_value(&payload, &mut cursor), expected);
        }
        for &expected in example.inference().state().global() {
            assert_eq!(f32_value(&payload, &mut cursor), expected);
        }
        assert_eq!(
            u32::from_le_bytes(take(&payload, &mut cursor)) as usize,
            example.inference().legal_actions().len()
        );
        for action in example.inference().legal_actions() {
            for expected in action.slots() {
                assert_eq!(u16::from_le_bytes(take(&payload, &mut cursor)), expected);
            }
        }
        for &expected in example.policy_targets() {
            assert_eq!(f32_value(&payload, &mut cursor), expected);
        }
        for &expected in example.value_targets() {
            assert_eq!(f32_value(&payload, &mut cursor), expected);
        }
    }
    assert_eq!(cursor, payload.len());
}

#[test]
fn response_is_bound_to_request_and_next_optimizer_step() {
    let request =
        TrainingRequestV1::new(81, 12, 1.0e-4, 512, [0xcd; 32], 1_000, training_examples())
            .unwrap();
    let response = TrainingResponseV1::decode_payload(&valid_response(&request), &request).unwrap();
    assert_eq!(response.request_id(), 81);
    assert_eq!(response.completed_training_step(), 13);
    assert_eq!(response.completed_replay_index(), 1_002);
    assert_eq!(response.policy_loss(), 1.25);
    assert_eq!(response.value_loss(), 0.75);
    assert_eq!(response.total_loss(), 2.0);

    let mut wrong_step = valid_response(&request);
    wrong_step[16..24].copy_from_slice(&14_u64.to_le_bytes());
    assert!(matches!(
        TrainingResponseV1::decode_payload(&wrong_step, &request),
        Err(TrainingWireError::ResponseTrainingStepMismatch {
            expected: 13,
            actual: 14,
        })
    ));
}

#[test]
fn invalid_targets_and_service_errors_are_explicit() {
    let inference = training_examples().remove(0).inference().clone();
    assert!(matches!(
        TrainingExampleV1::new(inference.clone(), vec![1.0], [1.0, 0.0, 0.0]),
        Err(TrainingWireError::PolicyTargetCountMismatch { .. })
    ));
    let mut negative = vec![0.0; inference.legal_actions().len()];
    negative[0] = 1.1;
    negative[1] = -0.1;
    assert!(matches!(
        TrainingExampleV1::new(inference, negative, [1.0, 0.0, 0.0]),
        Err(TrainingWireError::InvalidProbability { name: "policy", .. })
    ));

    let request =
        TrainingRequestV1::new(91, 0, 1.0e-4, 512, [0xef; 32], 0, training_examples()).unwrap();
    let mut error = Vec::new();
    error.extend_from_slice(&TRAINING_ERROR_MAGIC_V1);
    error.extend_from_slice(&91_u64.to_le_bytes());
    error.extend_from_slice(&4_u32.to_le_bytes());
    error.extend_from_slice(b"boom");
    assert!(matches!(
        TrainingResponseV1::decode_payload(&error, &request),
        Err(TrainingWireError::Service(message)) if message == "boom"
    ));
}

fn valid_response(request: &TrainingRequestV1) -> Vec<u8> {
    let mut response = Vec::new();
    response.extend_from_slice(&TRAINING_RESPONSE_MAGIC_V1);
    response.extend_from_slice(&request.request_id().to_le_bytes());
    response.extend_from_slice(&(request.expected_training_step() + 1).to_le_bytes());
    response.extend_from_slice(&request.next_replay_index().to_le_bytes());
    for value in [1.25_f32, 0.75, 2.0] {
        response.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    response
}

fn f32_value(bytes: &[u8], cursor: &mut usize) -> f32 {
    f32::from_bits(u32::from_le_bytes(take(bytes, cursor)))
}

fn take<const N: usize>(bytes: &[u8], cursor: &mut usize) -> [u8; N] {
    let value = bytes[*cursor..*cursor + N].try_into().unwrap();
    *cursor += N;
    value
}
