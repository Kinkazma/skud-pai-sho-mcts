use std::collections::HashSet;

use paisho_core::{
    legal_actions, Accent, AccentPlacement, Action, BasicFlower, Coordinate, GameOutcome,
    GameRecord, Player, Position, SpecialFlower, StandardSetup, TileKind, TurnPhase, EAST_GATE,
    NORTH_GATE, PLAYABLE_POINT_COUNT, SOUTH_GATE, WEST_GATE,
};
use paisho_model::{
    current_tile_channel_v1, encode_action_v1, encode_legal_actions_v1, encode_state_v1,
    opponent_tile_channel_v1, ActionEncodingError, ActionEncodingV1, ActionFamilyV1,
    StateEncodingError, CURRENT_RESERVE_START_V1, GATE_CHANNEL_V1, GLOBAL_FEATURE_COUNT_V1,
    HARMONY_BONUS_PHASE_FEATURE_V1, MAIN_PHASE_FEATURE_V1, NO_COORDINATE_V1, NO_TILE_V1,
    OPPONENT_RESERVE_START_V1, PLAYABLE_CHANNEL_V1, SPATIAL_CHANNEL_COUNT_V1,
    SPATIAL_VALUE_COUNT_V1, TILE_KINDS_V1,
};

const RING_FINISH_RECORD: &str = "PAISHO-RECORD 1
rules skud-pai-sho-2022-03-14
start R4
host-accents 1,1,1,1
guest-accents 1,1,1,1
actions
arrange 0,-8 -2,-6
arrange 0,8 1,8
plant R5 0,-8
plant R4 -8,0
arrange 0,-8 1,-6
bonus-plant W3 0,8
arrange -8,0 -7,1
arrange 0,8 1,6
bonus-plant R5 0,8
plant W4 0,-8
arrange 0,8 -2,6
";

fn at(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}

fn assert_close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1.0e-6, "{actual} != {expected}");
}

#[test]
fn numeric_v1_schema_is_pinned() {
    assert_eq!(paisho_model::BOARD_SIZE_V1, 17);
    assert_eq!(paisho_model::BOARD_CELL_COUNT_V1, 289);
    assert_eq!(paisho_model::TILE_KIND_COUNT_V1, 12);
    assert_eq!(SPATIAL_CHANNEL_COUNT_V1, 29);
    assert_eq!(GLOBAL_FEATURE_COUNT_V1, 26);
    assert_eq!(NO_TILE_V1, 12);
    assert_eq!(NO_COORDINATE_V1, 289);
    for (slot, kind) in TILE_KINDS_V1.into_iter().enumerate() {
        assert_eq!(paisho_model::tile_slot_v1(kind), slot);
        assert_eq!(paisho_model::tile_kind_v1(slot as u16), Some(kind));
    }
    assert_eq!(paisho_model::tile_kind_v1(NO_TILE_V1), None);
    assert_eq!(ActionFamilyV1::PlantBasicMain.index(), 0);
    assert_eq!(ActionFamilyV1::PlantBasicBonus.index(), 6);
}

#[test]
fn opening_state_has_exact_shape_topology_pieces_and_reserves() {
    let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let encoded = encode_state_v1(&position).unwrap();

    assert_eq!(encoded.perspective(), Player::Guest);
    assert_eq!(
        encoded.spatial_shape_nhwc(),
        [17, 17, SPATIAL_CHANNEL_COUNT_V1]
    );
    assert_eq!(encoded.spatial_nhwc().len(), SPATIAL_VALUE_COUNT_V1);
    assert_eq!(encoded.global().len(), GLOBAL_FEATURE_COUNT_V1);

    let playable_sum: f32 = paisho_core::all_coordinates()
        .map(|coordinate| {
            encoded
                .spatial_value(PLAYABLE_CHANNEL_V1, coordinate)
                .unwrap()
        })
        .sum();
    let gate_sum: f32 = paisho_core::all_coordinates()
        .map(|coordinate| encoded.spatial_value(GATE_CHANNEL_V1, coordinate).unwrap())
        .sum();
    assert_eq!(playable_sum, PLAYABLE_POINT_COUNT as f32);
    assert_eq!(gate_sum, 4.0);

    let red3 = TileKind::Basic(BasicFlower::Red3);
    assert_eq!(
        encoded.spatial_value(current_tile_channel_v1(red3), SOUTH_GATE),
        Some(1.0)
    );
    assert_eq!(
        encoded.spatial_value(opponent_tile_channel_v1(red3), NORTH_GATE),
        Some(1.0)
    );
    assert_close(
        encoded.global()[CURRENT_RESERVE_START_V1 + red3.index()],
        2.0 / 3.0,
    );
    assert_close(
        encoded.global()[OPPONENT_RESERVE_START_V1 + red3.index()],
        2.0 / 3.0,
    );
    assert_close(
        encoded.global()[CURRENT_RESERVE_START_V1 + TileKind::Accent(Accent::Rock).index()],
        0.5,
    );
    assert_eq!(encoded.global()[MAIN_PHASE_FEATURE_V1], 1.0);
    assert_eq!(encoded.global()[HARMONY_BONUS_PHASE_FEATURE_V1], 0.0);
}

#[test]
fn host_to_move_is_rotated_into_the_same_canonical_side() {
    let mut position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    position
        .apply(Action::Plant {
            flower: BasicFlower::White5,
            gate: EAST_GATE,
        })
        .unwrap();
    assert_eq!(position.to_move(), Player::Host);

    let encoded = encode_state_v1(&position).unwrap();
    let red3 = TileKind::Basic(BasicFlower::Red3);
    let white5 = TileKind::Basic(BasicFlower::White5);
    assert_eq!(encoded.perspective(), Player::Host);
    assert_eq!(
        encoded.spatial_value(current_tile_channel_v1(red3), SOUTH_GATE),
        Some(1.0)
    );
    assert_eq!(
        encoded.spatial_value(opponent_tile_channel_v1(red3), NORTH_GATE),
        Some(1.0)
    );
    assert_eq!(
        encoded.spatial_value(opponent_tile_channel_v1(white5), WEST_GATE),
        Some(1.0)
    );
    assert_close(
        encoded.global()[CURRENT_RESERVE_START_V1 + white5.index()],
        1.0,
    );
    assert_close(
        encoded.global()[OPPONENT_RESERVE_START_V1 + white5.index()],
        2.0 / 3.0,
    );
}

#[test]
fn phase_is_encoded_and_terminal_positions_are_rejected() {
    let record: GameRecord = RING_FINISH_RECORD.parse().unwrap();
    let mut position = record.initial_position();
    for action in &record.actions()[..5] {
        position.apply(*action).unwrap();
    }
    assert_eq!(position.phase(), TurnPhase::HarmonyBonus);

    let encoded = encode_state_v1(&position).unwrap();
    assert_eq!(encoded.global()[MAIN_PHASE_FEATURE_V1], 0.0);
    assert_eq!(encoded.global()[HARMONY_BONUS_PHASE_FEATURE_V1], 1.0);

    for action in &record.actions()[5..] {
        position.apply(*action).unwrap();
    }
    assert_ne!(position.outcome(), GameOutcome::Ongoing);
    assert_eq!(
        encode_state_v1(&position),
        Err(StateEncodingError::TerminalPosition)
    );
}

#[test]
fn ongoing_positions_without_a_legal_decision_are_rejected() {
    let mut position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    for notation in [
        "arrange 0,-8 0,-6",
        "plant W3 0,-8",
        "arrange 0,-6 1,-6",
        "arrange 0,-8 1,-6",
        "plant R3 0,-8",
        "plant R5 8,0",
        "arrange 0,-8 -1,-7",
        "plant R5 0,-8",
        "arrange -1,-7 -2,-7",
        "plant R5 -8,0",
        "arrange -2,-7 0,-7",
        "arrange 1,-6 0,-7",
    ] {
        let action: Action = notation.parse().unwrap();
        assert!(legal_actions(&position).contains(&action));
        position.apply(action).unwrap();
    }

    assert_eq!(position.outcome(), GameOutcome::Ongoing);
    assert!(legal_actions(&position).is_empty());
    assert_eq!(
        encode_state_v1(&position),
        Err(StateEncodingError::NoLegalAction)
    );
}

#[test]
fn every_action_shape_round_trips_in_both_perspectives() {
    let actions = [
        Action::Plant {
            flower: BasicFlower::Red3,
            gate: NORTH_GATE,
        },
        Action::Arrange {
            from: at(-2, 1),
            to: at(2, 1),
        },
        Action::SkipHarmonyBonus,
        Action::PlayAccent {
            accent: Accent::Rock,
            placement: AccentPlacement::At(at(0, 0)),
        },
        Action::PlayAccent {
            accent: Accent::Boat,
            placement: AccentPlacement::BoatMove {
                flower: at(-1, 1),
                destination: at(0, 1),
            },
        },
        Action::PlantSpecial {
            flower: SpecialFlower::Orchid,
            gate: EAST_GATE,
        },
        Action::BonusPlantBasic {
            flower: BasicFlower::White5,
            gate: WEST_GATE,
        },
    ];

    for perspective in [Player::Guest, Player::Host] {
        for action in actions {
            let encoded = encode_action_v1(action, perspective).unwrap();
            assert_eq!(encoded.decode(perspective), Ok(action));
            assert_eq!(ActionEncodingV1::from_slots(encoded.slots()), Ok(encoded));
        }
    }
}

#[test]
fn host_action_coordinates_are_half_turned_in_policy_space() {
    let action = Action::Arrange {
        from: NORTH_GATE,
        to: at(2, 6),
    };
    let guest = encode_action_v1(action, Player::Guest).unwrap();
    let host = encode_action_v1(action, Player::Host).unwrap();

    assert_eq!(guest.source(), Some(NORTH_GATE));
    assert_eq!(guest.destination(), Some(at(2, 6)));
    assert_eq!(host.source(), Some(SOUTH_GATE));
    assert_eq!(host.destination(), Some(at(-2, -6)));
    assert_eq!(host.decode(Player::Host), Ok(action));
}

#[test]
fn malformed_policy_addresses_are_rejected() {
    assert_eq!(
        ActionEncodingV1::from_slots([7, NO_TILE_V1, NO_COORDINATE_V1, NO_COORDINATE_V1]),
        Err(ActionEncodingError::InvalidFamily(7))
    );
    assert_eq!(
        ActionEncodingV1::from_slots([
            ActionFamilyV1::SkipHarmonyBonus as u16,
            NO_TILE_V1 + 1,
            NO_COORDINATE_V1,
            NO_COORDINATE_V1,
        ]),
        Err(ActionEncodingError::InvalidTile(NO_TILE_V1 + 1))
    );
    assert_eq!(
        ActionEncodingV1::from_slots([
            ActionFamilyV1::Arrange as u16,
            NO_TILE_V1,
            NO_COORDINATE_V1 + 1,
            at(0, 0).dense_index() as u16,
        ]),
        Err(ActionEncodingError::InvalidCoordinate(NO_COORDINATE_V1 + 1))
    );
    assert_eq!(
        ActionEncodingV1::from_slots([
            ActionFamilyV1::Arrange as u16,
            NO_TILE_V1,
            at(-8, 8).dense_index() as u16,
            at(0, 0).dense_index() as u16,
        ]),
        Err(ActionEncodingError::NonPlayableCoordinate(
            at(-8, 8).dense_index() as u16
        ))
    );
    assert_eq!(
        ActionEncodingV1::from_slots([
            ActionFamilyV1::PlantBasicMain as u16,
            TileKind::Basic(BasicFlower::Red3).index() as u16,
            NO_COORDINATE_V1,
            at(0, 0).dense_index() as u16,
        ]),
        Err(ActionEncodingError::InvalidShape(
            ActionFamilyV1::PlantBasicMain
        ))
    );
    assert_eq!(
        encode_action_v1(
            Action::PlayAccent {
                accent: Accent::Rock,
                placement: AccentPlacement::BoatMove {
                    flower: at(0, 0),
                    destination: at(0, 1),
                },
            },
            Player::Guest,
        ),
        Err(ActionEncodingError::InvalidBoatAccent(Accent::Rock))
    );
}

#[test]
fn every_legal_address_is_unique_and_decodes_to_its_engine_action() {
    let record: GameRecord = RING_FINISH_RECORD.parse().unwrap();
    let mut position = record.initial_position();
    let mut seen_families = HashSet::new();

    for recorded_action in record.actions() {
        let actions = legal_actions(&position);
        let encoded = encode_legal_actions_v1(&position).unwrap();
        assert_eq!(encoded.len(), actions.len());
        assert_eq!(
            encoded.iter().copied().collect::<HashSet<_>>().len(),
            encoded.len()
        );
        for address in &encoded {
            seen_families.insert(address.family());
        }
        for (action, address) in actions.iter().zip(&encoded) {
            assert_eq!(address.decode(position.to_move()), Ok(*action));
        }
        assert!(encoded
            .iter()
            .any(|address| address.decode(position.to_move()) == Ok(*recorded_action)));
        position.apply(*recorded_action).unwrap();
    }

    assert_eq!(seen_families.len(), paisho_model::ACTION_FAMILY_COUNT_V1);
    assert!(legal_actions(&position).is_empty());
    assert!(encode_legal_actions_v1(&position).unwrap().is_empty());
}
