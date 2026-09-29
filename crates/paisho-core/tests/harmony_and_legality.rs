use paisho_core::{
    apply_arrangement, can_capture, clashes, harmonies, has_clash, is_drained, is_trapped,
    legal_arrangement_destinations, validate_arrangement, Accent, ArrangementError, BasicFlower,
    Board, Coordinate, LineOrientation, Player, Tile, TileKind,
};

fn basic(owner: Player, flower: BasicFlower) -> Tile {
    Tile::new(owner, TileKind::Basic(flower))
}

fn at(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}

#[test]
fn unobstructed_compatible_flowers_form_one_harmony() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();

    let found = harmonies(&board);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].first, at(-2, 0));
    assert_eq!(found[0].second, at(2, 0));
    assert_eq!(found[0].owner, Player::Guest);
    assert_eq!(found[0].orientation, LineOrientation::Horizontal);
}

#[test]
fn tiles_and_gates_interrupt_lines() {
    let mut blocked = Board::empty();
    blocked
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    blocked
        .place(at(0, 0), basic(Player::Host, BasicFlower::White5))
        .unwrap();
    blocked
        .place(at(2, 0), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    assert!(harmonies(&blocked).is_empty());

    let mut across_gate = Board::empty();
    across_gate
        .place(at(-1, -8), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    across_gate
        .place(at(1, -8), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    assert!(harmonies(&across_gate).is_empty());
}

#[test]
fn rocks_cancel_their_entire_grid_line_and_knotweed_drains_neighbors() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    board
        .place(
            at(6, 0),
            Tile::new(Player::Host, TileKind::Accent(Accent::Rock)),
        )
        .unwrap();
    assert!(harmonies(&board).is_empty());

    board.remove(at(6, 0));
    board
        .place(
            at(-2, 1),
            Tile::new(Player::Host, TileKind::Accent(Accent::Knotweed)),
        )
        .unwrap();
    assert!(is_drained(&board, at(-2, 0)));
    assert!(harmonies(&board).is_empty());
}

#[test]
fn white_lotus_harmony_is_owned_by_the_basic_flower_player() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), Tile::new(Player::Host, TileKind::WhiteLotus))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Guest, BasicFlower::White5))
        .unwrap();
    assert_eq!(harmonies(&board)[0].owner, Player::Guest);
}

#[test]
fn visible_opposite_same_number_flowers_clash() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Host, BasicFlower::White3))
        .unwrap();
    let found = clashes(&board);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].orientation, LineOrientation::Horizontal);
    assert!(has_clash(&board));
}

#[test]
fn basic_flowers_respect_gardens_but_may_land_on_neutral_boundaries() {
    let mut board = Board::empty();
    let start = at(0, 4);
    board
        .place(start, basic(Player::Guest, BasicFlower::Red3))
        .unwrap();

    assert_eq!(
        validate_arrangement(&board, Player::Guest, start, at(-1, 5)),
        Err(ArrangementError::OppositeGarden)
    );
    assert!(validate_arrangement(&board, Player::Guest, start, at(-2, 5)).is_ok());
}

#[test]
fn clashing_flowers_capture_but_other_basic_flowers_do_not() {
    let attacker = basic(Player::Guest, BasicFlower::Red3);
    let clashing = basic(Player::Host, BasicFlower::White3);
    let neutral = basic(Player::Host, BasicFlower::White4);
    assert!(can_capture(&Board::empty(), attacker, clashing));
    assert!(!can_capture(&Board::empty(), attacker, neutral));
}

#[test]
fn orchid_traps_surrounding_opponent_flowers() {
    let mut board = Board::empty();
    let trapped = at(0, 0);
    board
        .place(trapped, basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    board
        .place(at(1, 1), Tile::new(Player::Host, TileKind::Orchid))
        .unwrap();

    assert!(is_trapped(&board, trapped));
    assert_eq!(
        validate_arrangement(&board, Player::Guest, trapped, at(0, 1)),
        Err(ArrangementError::TrappedByOrchid)
    );
}

#[test]
fn blooming_lotus_makes_its_orchid_raging_and_vulnerable() {
    let mut board = Board::empty();
    let guest_orchid = Tile::new(Player::Guest, TileKind::Orchid);
    let host_orchid = Tile::new(Player::Host, TileKind::Orchid);
    let host_flower = basic(Player::Host, BasicFlower::White4);
    let guest_flower = basic(Player::Guest, BasicFlower::Red5);

    assert!(!can_capture(&board, guest_orchid, host_flower));
    board
        .place(at(0, 3), Tile::new(Player::Guest, TileKind::WhiteLotus))
        .unwrap();
    assert!(can_capture(&board, guest_orchid, host_flower));

    assert!(!can_capture(&board, guest_flower, host_orchid));
    board
        .place(at(3, 0), Tile::new(Player::Host, TileKind::WhiteLotus))
        .unwrap();
    assert!(can_capture(&board, guest_flower, host_orchid));
}

#[test]
fn moving_a_blocker_may_not_expose_a_clash() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(0, 0), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Host, BasicFlower::White3))
        .unwrap();

    assert_eq!(
        validate_arrangement(&board, Player::Guest, at(0, 0), at(0, 1)),
        Err(ArrangementError::CreatesClash)
    );
    assert!(!legal_arrangement_destinations(&board, Player::Guest, at(0, 0)).contains(&at(0, 1)));
}

#[test]
fn moving_a_flower_may_not_create_a_clash_at_its_destination() {
    let mut board = Board::empty();
    board
        .place(at(0, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(2, 1), basic(Player::Host, BasicFlower::White3))
        .unwrap();

    assert_eq!(
        validate_arrangement(&board, Player::Guest, at(0, 0), at(0, 1)),
        Err(ArrangementError::CreatesClash)
    );
    assert!(!legal_arrangement_destinations(&board, Player::Guest, at(0, 0)).contains(&at(0, 1)));
}

#[test]
fn applying_an_arrangement_reports_capture_and_new_harmony() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(0, 1), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();

    let destinations = legal_arrangement_destinations(&board, Player::Guest, at(0, 1));
    assert!(destinations.contains(&at(2, 0)));

    let result = apply_arrangement(&mut board, Player::Guest, at(0, 1), at(2, 0)).unwrap();
    assert_eq!(result.distance, 3);
    assert_eq!(result.captured, None);
    assert!(result.formed_new_harmony);
}

#[test]
fn the_same_harmonious_tile_pair_does_not_create_another_bonus() {
    let mut board = Board::empty();
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();

    let result = apply_arrangement(&mut board, Player::Guest, at(-2, 0), at(-1, 0)).unwrap();

    assert!(!result.formed_new_harmony);
    assert_eq!(harmonies(&board).len(), 1);
}
