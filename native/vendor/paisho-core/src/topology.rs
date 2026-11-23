use crate::{Coordinate, BOARD_SIZE};

/// Number of playable intersections on the standard Skud Pai Sho board.
pub const PLAYABLE_POINT_COUNT: usize = 249;

pub const NORTH_GATE: Coordinate = Coordinate::validated(0, 8);
pub const EAST_GATE: Coordinate = Coordinate::validated(8, 0);
pub const SOUTH_GATE: Coordinate = Coordinate::validated(0, -8);
pub const WEST_GATE: Coordinate = Coordinate::validated(-8, 0);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Region {
    Red,
    White,
    Neutral,
}

/// Exact combinations used by the reference board implementation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum PointType {
    NonPlayable,
    Gate,
    Neutral,
    Red,
    White,
    RedWhite,
    RedNeutral,
    WhiteNeutral,
    RedWhiteNeutral,
}

impl PointType {
    pub const fn is_playable(self) -> bool {
        !matches!(self, Self::NonPlayable)
    }

    pub const fn is_gate(self) -> bool {
        matches!(self, Self::Gate)
    }

    pub const fn belongs_to(self, region: Region) -> bool {
        match region {
            Region::Red => matches!(
                self,
                Self::Red | Self::RedWhite | Self::RedNeutral | Self::RedWhiteNeutral
            ),
            Region::White => matches!(
                self,
                Self::White | Self::RedWhite | Self::WhiteNeutral | Self::RedWhiteNeutral
            ),
            Region::Neutral => matches!(
                self,
                Self::Neutral | Self::RedNeutral | Self::WhiteNeutral | Self::RedWhiteNeutral
            ),
        }
    }
}

/// Returns the static type of one location in the 17 × 17 board envelope.
pub const fn point_type_at(coordinate: Coordinate) -> PointType {
    let x = coordinate.x();
    let y = coordinate.y();
    let distance = magnitude(x) + magnitude(y);

    if distance > 12 {
        return PointType::NonPlayable;
    }

    if (x == 0 && magnitude(y) == 8) || (y == 0 && magnitude(x) == 8) {
        return PointType::Gate;
    }

    if distance > 7 {
        return PointType::Neutral;
    }

    let neutral_boundary = distance == 7;
    if x == 0 || y == 0 {
        return if neutral_boundary {
            PointType::RedWhiteNeutral
        } else {
            PointType::RedWhite
        };
    }

    let is_red = (x > 0 && y > 0) || (x < 0 && y < 0);
    match (is_red, neutral_boundary) {
        (true, true) => PointType::RedNeutral,
        (true, false) => PointType::Red,
        (false, true) => PointType::WhiteNeutral,
        (false, false) => PointType::White,
    }
}

/// Iterates over the full dense envelope in site row-major order.
pub fn all_coordinates() -> impl Iterator<Item = Coordinate> {
    (0..BOARD_SIZE).flat_map(|row| {
        (0..BOARD_SIZE).map(move |column| {
            Coordinate::from_grid(row, column).expect("loop indices are inside the board envelope")
        })
    })
}

/// Iterates over all 249 playable intersections in site row-major order.
pub fn playable_coordinates() -> impl Iterator<Item = Coordinate> {
    all_coordinates().filter(|coordinate| point_type_at(*coordinate).is_playable())
}

pub fn orthogonal_neighbors(coordinate: Coordinate) -> impl Iterator<Item = Coordinate> {
    [(0, 1), (1, 0), (0, -1), (-1, 0)]
        .into_iter()
        .filter_map(move |(delta_x, delta_y)| coordinate.translated(delta_x, delta_y))
        .filter(|neighbor| point_type_at(*neighbor).is_playable())
}

pub fn surrounding_neighbors(coordinate: Coordinate) -> impl Iterator<Item = Coordinate> {
    [
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
    ]
    .into_iter()
    .filter_map(move |(delta_x, delta_y)| coordinate.translated(delta_x, delta_y))
    .filter(|neighbor| point_type_at(*neighbor).is_playable())
}

const fn magnitude(value: i8) -> i8 {
    if value < 0 {
        -value
    } else {
        value
    }
}
