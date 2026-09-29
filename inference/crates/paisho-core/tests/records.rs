use paisho_core::{
    Accent, AccentLoadout, AccentPlacement, Action, ActionNotationError, BasicFlower, Coordinate,
    GameRecord, Player, RuleProfileId, SpecialFlower, StandardSetup, TileKind, EAST_GATE,
    NORTH_GATE, WEST_GATE,
};

fn at(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}

#[test]
fn every_action_shape_round_trips_through_stable_notation() {
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

    for action in actions {
        let notation = action.to_string();
        assert_eq!(notation.parse::<Action>(), Ok(action), "{notation}");
    }
}

#[test]
fn malformed_action_notation_is_rejected_precisely() {
    assert_eq!(
        "accent R move 0,0 1,1".parse::<Action>(),
        Err(ActionNotationError::InvalidAccentPlacement)
    );
    assert_eq!(
        "plant L 0,8".parse::<Action>(),
        Err(ActionNotationError::InvalidBasicFlower)
    );
    assert_eq!(
        "arrange 0,0 20,0".parse::<Action>(),
        Err(ActionNotationError::InvalidCoordinate(
            paisho_core::CoordinateError::OutsideEnvelope { x: 20, y: 0 }
        ))
    );
}

#[test]
fn a_record_round_trips_and_replays_to_the_same_position() {
    let setup = StandardSetup {
        host_accents: AccentLoadout::new(2, 0, 1, 1).unwrap(),
        guest_accents: AccentLoadout::new(0, 2, 1, 1).unwrap(),
        starting_flower: BasicFlower::White4,
    };
    let actions = [
        Action::Plant {
            flower: BasicFlower::Red3,
            gate: WEST_GATE,
        },
        Action::Plant {
            flower: BasicFlower::White5,
            gate: EAST_GATE,
        },
    ];
    let mut record = GameRecord::new(setup);
    let mut expected = paisho_core::Position::from_standard_setup(setup);
    for action in actions {
        expected.apply(action).unwrap();
        record.push(action);
    }

    let encoded = record.to_string();
    let decoded: GameRecord = encoded.parse().unwrap();

    assert_eq!(decoded, record);
    assert_eq!(decoded.replay().unwrap(), expected);
    assert_eq!(decoded.rules(), RuleProfileId::CURRENT);
    assert_eq!(decoded.actions().len(), 2);
    assert_eq!(
        decoded
            .replay()
            .unwrap()
            .reserve(Player::Host)
            .count(TileKind::Basic(BasicFlower::White5)),
        2
    );
}

#[test]
fn replay_reports_the_first_illegal_action_and_its_number() {
    let mut record = GameRecord::new(StandardSetup::balanced(BasicFlower::Red3));
    let illegal = Action::Plant {
        flower: BasicFlower::White3,
        gate: NORTH_GATE,
    };
    record.push(illegal);

    let error = record.replay().unwrap_err();
    assert_eq!(error.action_number, 1);
    assert_eq!(error.action, illegal);
}

#[test]
fn the_standard_profile_is_stable_and_self_describing() {
    let id = "skud-pai-sho-2022-03-14".parse::<RuleProfileId>().unwrap();
    let profile = id.profile();
    assert_eq!(profile.id.to_string(), "skud-pai-sho-2022-03-14");
    assert_eq!(
        profile.reference_source_commit,
        "b849dbdabb1138ff0f6d609adf38b301c2f875ae"
    );
    assert!(profile.formal_same_flower_start);
    assert!(profile.limited_harmony_bonus_gates);
    assert!(profile.modern_knotweed);
    assert!(profile.wheel_may_be_played_near_gates);
    assert!(profile.rocks_are_unwheelable);
    assert!(profile.white_lotus_protected_from_basic_flowers);
    assert!(profile.wild_orchid_captures_any_flower);
}
