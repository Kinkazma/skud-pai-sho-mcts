use paisho_core::{legal_actions, BasicFlower, Position, StandardSetup};
use paisho_model::{
    InferenceExampleV1, TerminalPpoExampleV1, TerminalPpoParametersV1, TerminalPpoRequestV1,
    TerminalPpoResponseV1, TerminalPpoWireError, ValueClassV1, GLOBAL_FEATURE_COUNT_V1,
    SPATIAL_VALUE_COUNT_V1, TERMINAL_PPO_ERROR_MAGIC_V1, TERMINAL_PPO_REQUEST_MAGIC_V1,
    TERMINAL_PPO_RESPONSE_MAGIC_V1,
};

fn ppo_examples() -> Vec<TerminalPpoExampleV1> {
    let mut position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let mut examples = Vec::new();
    for (terminal_value, actor_value) in [(ValueClassV1::Win, 0.25), (ValueClassV1::Loss, -0.5)] {
        let inference = InferenceExampleV1::from_position(&position).unwrap();
        examples.push(
            TerminalPpoExampleV1::new(inference, 0, 0.125, terminal_value, actor_value).unwrap(),
        );
        let action = legal_actions(&position)[0];
        position.apply(action).unwrap();
    }
    examples
}

fn request() -> TerminalPpoRequestV1 {
    TerminalPpoRequestV1::new(
        71,
        12,
        1.0e-4,
        TerminalPpoParametersV1::new(0.2, 0.5, 0.01).unwrap(),
        512,
        [0xab; 32],
        900,
        ppo_examples(),
    )
    .unwrap()
}

#[test]
fn request_serializes_terminal_ppo_examples_in_v1_order() {
    let request = request();
    let payload = request.encode_payload().unwrap();
    let mut cursor = 0;

    assert_eq!(
        take::<8>(&payload, &mut cursor),
        TERMINAL_PPO_REQUEST_MAGIC_V1
    );
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 71);
    assert_eq!(u64::from_le_bytes(take(&payload, &mut cursor)), 12);
    assert_eq!(f32_value(&payload, &mut cursor), 1.0e-4);
    assert_eq!(f32_value(&payload, &mut cursor), 1.0);
    assert_eq!(f32_value(&payload, &mut cursor), 0.0);
    assert_eq!(f32_value(&payload, &mut cursor), 0.2);
    assert_eq!(f32_value(&payload, &mut cursor), 0.5);
    assert_eq!(f32_value(&payload, &mut cursor), 0.01);
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
        assert_eq!(
            u32::from_le_bytes(take(&payload, &mut cursor)) as usize,
            example.played_action_index()
        );
        assert_eq!(
            f32_value(&payload, &mut cursor),
            example.behavior_probability()
        );
        assert_eq!(
            u32::from_le_bytes(take(&payload, &mut cursor)) as usize,
            example.terminal_value().index()
        );
        assert_eq!(f32_value(&payload, &mut cursor), example.actor_value());
    }
    assert_eq!(cursor, payload.len());
    assert_eq!(request.examples()[0].advantage(), 0.75);
    assert_eq!(request.examples()[1].advantage(), -0.5);
}

#[test]
fn response_is_bound_to_request_and_reports_objective_metrics() {
    let request = request();
    let response =
        TerminalPpoResponseV1::decode_payload(&valid_response(&request), &request).unwrap();

    assert_eq!(response.request_id(), 71);
    assert_eq!(response.completed_training_step(), 13);
    assert_eq!(response.completed_replay_index(), 902);
    assert_eq!(response.policy_loss(), -0.125);
    assert_eq!(response.value_loss(), 0.75);
    assert_eq!(response.entropy(), 1.5);
    assert_eq!(response.total_loss(), 0.235);
    assert_eq!(response.mean_advantage(), 0.125);
    assert_eq!(response.mean_importance_ratio(), 1.25);
    assert_eq!(response.mean_squared_ratio_deviation(), 0.0625);

    let mut wrong_index = valid_response(&request);
    wrong_index[24..32].copy_from_slice(&903_u64.to_le_bytes());
    assert!(matches!(
        TerminalPpoResponseV1::decode_payload(&wrong_index, &request),
        Err(TerminalPpoWireError::ResponseReplayIndexMismatch {
            expected: 902,
            actual: 903,
        })
    ));
}

#[test]
fn invalid_policy_gradient_inputs_and_service_errors_are_explicit() {
    let inference = ppo_examples().remove(0).inference().clone();
    assert!(matches!(
        TerminalPpoExampleV1::new(
            inference.clone(),
            inference.legal_actions().len(),
            0.5,
            ValueClassV1::Win,
            0.0,
        ),
        Err(TerminalPpoWireError::PlayedActionOutOfRange { .. })
    ));
    assert!(matches!(
        TerminalPpoExampleV1::new(inference, 0, 0.0, ValueClassV1::Loss, 0.0),
        Err(TerminalPpoWireError::InvalidBehaviorProbability(value)) if value == 0.0
    ));
    assert!(matches!(
        TerminalPpoParametersV1::new(1.0, 0.5, 0.01),
        Err(TerminalPpoWireError::InvalidClipEpsilon(value)) if value == 1.0
    ));

    let request = request();
    let mut error = Vec::new();
    error.extend_from_slice(&TERMINAL_PPO_ERROR_MAGIC_V1);
    error.extend_from_slice(&request.request_id().to_le_bytes());
    error.extend_from_slice(&4_u32.to_le_bytes());
    error.extend_from_slice(b"boom");
    assert!(matches!(
        TerminalPpoResponseV1::decode_payload(&error, &request),
        Err(TerminalPpoWireError::Service(message)) if message == "boom"
    ));
}

fn valid_response(request: &TerminalPpoRequestV1) -> Vec<u8> {
    let mut response = Vec::new();
    response.extend_from_slice(&TERMINAL_PPO_RESPONSE_MAGIC_V1);
    response.extend_from_slice(&request.request_id().to_le_bytes());
    response.extend_from_slice(&(request.expected_training_step() + 1).to_le_bytes());
    response.extend_from_slice(&request.next_replay_index().to_le_bytes());
    for value in [-0.125_f32, 0.75, 1.5, 0.235, 0.125, 1.25, 0.0625] {
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

#[test]
fn duration_reward_preserves_outcomes_and_serializes_explicit_utility() {
    let source = ppo_examples();
    let fast = source[0].clone().with_win_duration(1);
    let slow = source[0].clone().with_win_duration(600);
    assert!(fast.advantage() > slow.advantage());
    assert!(slow.advantage() + slow.actor_value() >= 0.9);
    assert_eq!(
        source[1].clone().with_win_duration(1).advantage(),
        source[1].clone().with_win_duration(600).advantage()
    );
    let draw = TerminalPpoExampleV1::new(
        source[0].inference().clone(),
        0,
        0.5,
        ValueClassV1::Draw,
        0.0,
    )
    .unwrap();
    assert_eq!(draw.with_win_duration(1).advantage(), 0.0);
    let request = TerminalPpoRequestV1::new(
        71,
        12,
        1e-4,
        TerminalPpoParametersV1::new(0.2, 0.5, 0.01).unwrap(),
        512,
        [0; 32],
        0,
        vec![slow],
    )
    .unwrap();
    let bytes = request.encode_payload().unwrap();
    assert_eq!(&bytes[..8], b"PSTREQ03");
    let utility = f32::from_le_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
    assert!((utility - (0.9 + 0.1 * 256.0 / 856.0)).abs() < 1e-6);
}
