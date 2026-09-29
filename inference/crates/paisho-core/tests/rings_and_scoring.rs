use paisho_core::{
    harmony_ring_owners, midline_crossing_harmony_count, BasicFlower, Board, Coordinate, Player,
    Tile, TileKind,
};

fn at(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}

fn basic(owner: Player, flower: BasicFlower) -> Tile {
    Tile::new(owner, TileKind::Basic(flower))
}

fn alternating_square(owner: Player, left: i8, right: i8, bottom: i8, top: i8) -> Board {
    let mut board = Board::empty();
    for (coordinate, flower) in [
        (at(left, bottom), BasicFlower::Red3),
        (at(right, bottom), BasicFlower::Red4),
        (at(right, top), BasicFlower::Red3),
        (at(left, top), BasicFlower::Red4),
    ] {
        board.place(coordinate, basic(owner, flower)).unwrap();
    }
    board
}

#[test]
fn four_harmonies_around_the_centre_form_a_ring() {
    let board = alternating_square(Player::Guest, -2, 2, -2, 2);
    assert_eq!(harmony_ring_owners(&board), vec![Player::Guest]);
}

#[test]
fn a_closed_chain_away_from_the_centre_is_not_a_ring() {
    let board = alternating_square(Player::Guest, 1, 3, 1, 3);
    assert!(harmony_ring_owners(&board).is_empty());
}

#[test]
fn a_chain_touching_the_centre_is_not_a_ring() {
    let board = alternating_square(Player::Guest, -2, 2, 0, 2);
    assert!(harmony_ring_owners(&board).is_empty());
}

#[test]
fn incomplete_and_mixed_owner_chains_are_not_rings() {
    let mut board = alternating_square(Player::Guest, -2, 2, -2, 2);
    board.remove(at(-2, 2));
    assert!(harmony_ring_owners(&board).is_empty());

    board
        .place(at(-2, 2), basic(Player::Host, BasicFlower::Red4))
        .unwrap();
    assert!(harmony_ring_owners(&board).is_empty());
}

#[test]
fn midline_score_excludes_the_centre_and_tiles_on_a_midline() {
    let mut board = Board::empty();
    // Counts: horizontal line crosses x=0 away from y=0.
    board
        .place(at(-2, 4), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(2, 4), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();
    // Does not count: crosses the centre itself.
    board
        .place(at(-2, 0), basic(Player::Guest, BasicFlower::White4))
        .unwrap();
    board
        .place(at(2, 0), basic(Player::Guest, BasicFlower::White5))
        .unwrap();
    // Does not count: one endpoint lies on x=0.
    board
        .place(at(0, -2), basic(Player::Guest, BasicFlower::Red3))
        .unwrap();
    board
        .place(at(0, 2), basic(Player::Guest, BasicFlower::Red4))
        .unwrap();

    assert_eq!(midline_crossing_harmony_count(&board, Player::Guest), 1);
    assert_eq!(midline_crossing_harmony_count(&board, Player::Host), 0);
}
