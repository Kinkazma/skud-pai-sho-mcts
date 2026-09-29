use std::str::FromStr;

use paisho_core::{
    Accent, AccentLoadout, AccentLoadoutError, BasicFlower, GameOutcome, Player, PointType,
    Position, StandardSetup, Tile, TileKind, BASIC_FLOWERS, NORTH_GATE, SOUTH_GATE,
    STANDARD_TILE_KINDS,
};

#[test]
fn standard_tile_codes_round_trip() {
    assert_eq!(STANDARD_TILE_KINDS.len(), TileKind::COUNT);
    for kind in STANDARD_TILE_KINDS {
        assert_eq!(TileKind::from_str(kind.code()), Ok(kind));
        assert_eq!(kind.to_string(), kind.code());
    }
}

#[test]
fn circle_of_harmony_and_clashes_are_exact() {
    for (left_index, left) in BASIC_FLOWERS.iter().enumerate() {
        for (right_index, right) in BASIC_FLOWERS.iter().enumerate() {
            let cyclic_distance = left_index.abs_diff(right_index);
            let expected_harmony = cyclic_distance == 1 || cyclic_distance == 5;
            assert_eq!(left.harmonizes_with(*right), expected_harmony);

            let expected_clash =
                left.color() != right.color() && left.movement() == right.movement();
            assert_eq!(left.clashes_with(*right), expected_clash);
        }
    }
}

#[test]
fn white_lotus_harmony_belongs_to_the_basic_flower_owner() {
    let lotus = Tile::new(Player::Host, TileKind::WhiteLotus);
    let guest_flower = Tile::new(Player::Guest, TileKind::Basic(BasicFlower::White4));
    assert_eq!(lotus.harmony_owner_with(guest_flower), Some(Player::Guest));
    assert_eq!(guest_flower.harmony_owner_with(lotus), Some(Player::Guest));
    assert_eq!(lotus.harmony_owner_with(lotus), None);
}

#[test]
fn loadout_requires_four_available_accents() {
    assert_eq!(AccentLoadout::balanced().count(Accent::Rock), 1);
    assert_eq!(
        AccentLoadout::new(2, 0, 0, 2).unwrap().count(Accent::Boat),
        2
    );
    assert_eq!(
        AccentLoadout::new(3, 0, 0, 1),
        Err(AccentLoadoutError::MoreThanTwoOfAKind)
    );
    assert_eq!(
        AccentLoadout::new(1, 1, 0, 0),
        Err(AccentLoadoutError::WrongTotal { total: 2 })
    );
}

#[test]
fn formal_standard_setup_places_matching_flowers_in_opposite_gates() {
    let starting_flower = BasicFlower::Red3;
    let position = Position::from_standard_setup(StandardSetup::balanced(starting_flower));
    let kind = TileKind::Basic(starting_flower);

    assert_eq!(position.board().occupied_count(), 2);
    assert_eq!(
        position.board().get(NORTH_GATE),
        Some(Tile::new(Player::Host, kind))
    );
    assert_eq!(
        position.board().get(SOUTH_GATE),
        Some(Tile::new(Player::Guest, kind))
    );
    assert_eq!(position.reserve(Player::Host).count(kind), 2);
    assert_eq!(position.reserve(Player::Guest).count(kind), 2);
    assert_eq!(position.reserve(Player::Host).total_count(), 23);
    assert_eq!(position.reserve(Player::Guest).total_count(), 23);
    assert_eq!(position.to_move(), Player::Guest);
    assert_eq!(position.completed_turns(), 0);
    assert_eq!(position.outcome(), GameOutcome::Ongoing);

    assert_eq!(paisho_core::point_type_at(NORTH_GATE), PointType::Gate);
    assert_eq!(paisho_core::point_type_at(SOUTH_GATE), PointType::Gate);
}
