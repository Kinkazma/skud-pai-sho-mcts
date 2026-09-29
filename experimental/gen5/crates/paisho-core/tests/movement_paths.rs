use paisho_core::{
    clear_path_distance, reachable_points, BasicFlower, Board, Coordinate, Player, Tile, TileKind,
    SOUTH_GATE,
};

fn flower(owner: Player) -> Tile {
    Tile::new(owner, TileKind::Basic(BasicFlower::Red5))
}

#[test]
fn paths_may_turn_but_may_not_cross_a_tile() {
    let mut board = Board::empty();
    let start = Coordinate::new(0, 0).unwrap();
    let blocker = Coordinate::new(1, 0).unwrap();
    let destination = Coordinate::new(2, 0).unwrap();
    board.place(start, flower(Player::Guest)).unwrap();
    board.place(blocker, flower(Player::Host)).unwrap();

    assert_eq!(clear_path_distance(&board, start, destination, 3), None);
    assert_eq!(clear_path_distance(&board, start, destination, 4), Some(4));
}

#[test]
fn an_occupied_destination_is_reachable_but_not_traversable() {
    let mut board = Board::empty();
    let start = Coordinate::new(0, 0).unwrap();
    let occupied = Coordinate::new(1, 0).unwrap();
    let beyond = Coordinate::new(2, 0).unwrap();
    board.place(start, flower(Player::Guest)).unwrap();
    board.place(occupied, flower(Player::Host)).unwrap();

    assert_eq!(clear_path_distance(&board, start, occupied, 1), Some(1));
    assert_eq!(clear_path_distance(&board, start, beyond, 2), None);
}

#[test]
fn an_empty_gate_can_be_crossed_as_an_intermediate_point() {
    let mut board = Board::empty();
    let start = Coordinate::new(1, -8).unwrap();
    let destination = Coordinate::new(-1, -8).unwrap();
    board.place(start, flower(Player::Guest)).unwrap();

    assert_eq!(clear_path_distance(&board, start, destination, 2), Some(2));
    assert!(reachable_points(&board, start, 2).contains(&SOUTH_GATE));

    board.place(SOUTH_GATE, flower(Player::Host)).unwrap();
    assert_eq!(clear_path_distance(&board, start, destination, 2), None);
}

#[test]
fn empty_sources_have_no_movement_graph() {
    let board = Board::empty();
    let start = Coordinate::new(0, 0).unwrap();
    assert_eq!(clear_path_distance(&board, start, start, 6), None);
    assert!(reachable_points(&board, start, 6).is_empty());
}
