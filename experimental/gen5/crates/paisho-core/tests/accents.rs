use paisho_core::{
    apply_accent, harmonies, is_drained, legal_accent_placements, Accent, AccentError,
    AccentPlacement, BasicFlower, Board, Coordinate, Player, Tile, TileKind, SOUTH_GATE,
};

fn at(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}

fn basic(owner: Player, flower: BasicFlower) -> Tile {
    Tile::new(owner, TileKind::Basic(flower))
}

#[test]
fn rock_and_knotweed_apply_their_persistent_line_effects() {
    let mut rock_board = Board::empty();
    rock_board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    rock_board
        .place(at(2, 0), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    assert_eq!(harmonies(&rock_board).len(), 1);
    apply_accent(
        &mut rock_board,
        Player::Host,
        Accent::Rock,
        AccentPlacement::At(at(6, 0)),
    )
    .unwrap();
    assert!(harmonies(&rock_board).is_empty());

    let mut knotweed_board = Board::empty();
    knotweed_board
        .place(at(0, 0), basic(Player::Guest, BasicFlower::White4))
        .unwrap();
    apply_accent(
        &mut knotweed_board,
        Player::Host,
        Accent::Knotweed,
        AccentPlacement::At(at(1, 1)),
    )
    .unwrap();
    assert!(is_drained(&knotweed_board, at(0, 0)));
}

#[test]
fn wheel_rotates_all_surrounding_tiles_clockwise_simultaneously() {
    let mut board = Board::empty();
    let top_tile = basic(Player::Guest, BasicFlower::Red3);
    let right_tile = Tile::new(Player::Host, TileKind::Orchid);
    board.place(at(0, 1), top_tile).unwrap();
    board.place(at(1, 0), right_tile).unwrap();

    apply_accent(
        &mut board,
        Player::Guest,
        Accent::Wheel,
        AccentPlacement::At(at(0, 0)),
    )
    .unwrap();

    assert_eq!(board.get(at(1, 1)), Some(top_tile));
    assert_eq!(board.get(at(1, -1)), Some(right_tile));
    assert_eq!(
        board.get(at(0, 0)),
        Some(Tile::new(Player::Guest, TileKind::Accent(Accent::Wheel)))
    );
}

#[test]
fn wheel_cannot_move_a_rock_gate_tile_or_flower_into_opposite_garden() {
    let mut rock_board = Board::empty();
    rock_board
        .place(
            at(0, 1),
            Tile::new(Player::Host, TileKind::Accent(Accent::Rock)),
        )
        .unwrap();
    assert_eq!(
        apply_accent(
            &mut rock_board,
            Player::Guest,
            Accent::Wheel,
            AccentPlacement::At(at(0, 0)),
        ),
        Err(AccentError::WheelCannotMoveRock)
    );

    let mut gate_board = Board::empty();
    gate_board
        .place(SOUTH_GATE, basic(Player::Host, BasicFlower::Red3))
        .unwrap();
    assert_eq!(
        apply_accent(
            &mut gate_board,
            Player::Guest,
            Accent::Wheel,
            AccentPlacement::At(at(1, -8)),
        ),
        Err(AccentError::WheelWouldMoveIntoOrOutOfGate)
    );

    let mut garden_board = Board::empty();
    garden_board
        .place(at(0, 5), basic(Player::Host, BasicFlower::White3))
        .unwrap();
    assert_eq!(
        apply_accent(
            &mut garden_board,
            Player::Guest,
            Accent::Wheel,
            AccentPlacement::At(at(0, 4)),
        ),
        Err(AccentError::OppositeGarden)
    );
}

#[test]
fn boat_removes_an_accent_or_replaces_and_moves_a_blooming_flower() {
    let mut removal_board = Board::empty();
    let rock = Tile::new(Player::Host, TileKind::Accent(Accent::Rock));
    removal_board.place(at(0, 0), rock).unwrap();
    let effect = apply_accent(
        &mut removal_board,
        Player::Guest,
        Accent::Boat,
        AccentPlacement::At(at(0, 0)),
    )
    .unwrap();
    assert_eq!(effect.removed, Some(rock));
    assert!(removal_board.is_empty(at(0, 0)));

    let mut movement_board = Board::empty();
    let flower = basic(Player::Host, BasicFlower::Red4);
    movement_board.place(at(0, 0), flower).unwrap();
    let effect = apply_accent(
        &mut movement_board,
        Player::Guest,
        Accent::Boat,
        AccentPlacement::BoatMove {
            flower: at(0, 0),
            destination: at(1, 0),
        },
    )
    .unwrap();
    assert_eq!(movement_board.get(at(1, 0)), Some(flower));
    assert_eq!(
        movement_board.get(at(0, 0)),
        Some(Tile::new(Player::Guest, TileKind::Accent(Accent::Boat)))
    );
    assert_eq!(effect.moved_flower, Some((at(0, 0), at(1, 0), flower)));
}

#[test]
fn boat_cannot_remove_a_blocker_if_that_would_expose_a_clash() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(
            at(0, 0),
            Tile::new(Player::Host, TileKind::Accent(Accent::Knotweed)),
        )
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Host, BasicFlower::White3))
        .unwrap();

    assert_eq!(
        apply_accent(
            &mut board,
            Player::Guest,
            Accent::Boat,
            AccentPlacement::At(at(0, 0)),
        ),
        Err(AccentError::CreatesClash)
    );
}

#[test]
fn placement_generator_returns_only_actions_that_apply() {
    let mut board = Board::empty();
    board
        .place(at(0, 0), basic(Player::Host, BasicFlower::Red3))
        .unwrap();

    for accent in [Accent::Rock, Accent::Wheel, Accent::Knotweed, Accent::Boat] {
        let placements = legal_accent_placements(&board, Player::Guest, accent);
        assert!(
            !placements.is_empty(),
            "no placement generated for {accent:?}"
        );
        for placement in placements {
            let mut copy = board.clone();
            assert!(apply_accent(&mut copy, Player::Guest, accent, placement).is_ok());
        }
    }
}
