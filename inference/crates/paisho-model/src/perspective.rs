use paisho_core::{Coordinate, Player};

/// Maps the native board to the V1 perspective in which the player to move
/// always owns the south opening side.
pub const fn canonical_coordinate_v1(perspective: Player, coordinate: Coordinate) -> Coordinate {
    match perspective {
        Player::Guest => coordinate,
        Player::Host => coordinate.rotate_180(),
    }
}

/// Reverses [`canonical_coordinate_v1`]. A half-turn is its own inverse.
pub const fn native_coordinate_v1(perspective: Player, coordinate: Coordinate) -> Coordinate {
    canonical_coordinate_v1(perspective, coordinate)
}
