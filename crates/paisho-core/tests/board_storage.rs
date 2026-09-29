use paisho_core::{BasicFlower, Board, BoardError, Coordinate, Player, Tile, TileKind};

#[test]
fn board_rejects_corners_and_occupied_points() {
    let mut board = Board::empty();
    let tile = Tile::new(Player::Guest, TileKind::Basic(BasicFlower::Red3));
    let corner = Coordinate::new(-8, 8).unwrap();
    let centre = Coordinate::new(0, 0).unwrap();

    assert_eq!(
        board.place(corner, tile),
        Err(BoardError::NonPlayable(corner))
    );
    assert_eq!(board.place(centre, tile), Ok(()));
    assert_eq!(board.place(centre, tile), Err(BoardError::Occupied(centre)));
    assert_eq!(board.get(centre), Some(tile));
    assert_eq!(board.remove(centre), Some(tile));
    assert!(board.is_empty(centre));
}
