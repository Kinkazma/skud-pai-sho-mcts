use core::fmt;

use crate::harmony::relocation_creates_clash;
use crate::{
    harmonies, has_clash, point_type_at, reachable_points, surrounding_neighbors, Accent, Board,
    Coordinate, FlowerColor, Player, PointType, Region, Tile, TileKind,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArrangementResult {
    pub moved: Tile,
    pub captured: Option<Tile>,
    pub distance: u8,
    pub formed_new_harmony: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArrangementError {
    EmptySource,
    NotOwnedByPlayer,
    AccentCannotMove,
    TrappedByOrchid,
    SamePoint,
    GateDestination,
    OppositeGarden,
    OccupiedByUncapturableTile,
    NoClearPath,
    CreatesClash,
}

pub fn validate_arrangement(
    board: &Board,
    player: Player,
    from: Coordinate,
    to: Coordinate,
) -> Result<u8, ArrangementError> {
    let moving = board.get(from).ok_or(ArrangementError::EmptySource)?;
    if moving.owner != player {
        return Err(ArrangementError::NotOwnedByPlayer);
    }
    let maximum = moving
        .kind
        .movement()
        .ok_or(ArrangementError::AccentCannotMove)?;
    if is_trapped(board, from) {
        return Err(ArrangementError::TrappedByOrchid);
    }
    if from == to {
        return Err(ArrangementError::SamePoint);
    }

    let destination_type = point_type_at(to);
    if destination_type.is_gate() {
        return Err(ArrangementError::GateDestination);
    }
    if !can_end_on(moving.kind, destination_type) {
        return Err(ArrangementError::OppositeGarden);
    }
    if let Some(target) = board.get(to) {
        if !can_capture(board, moving, target) {
            return Err(ArrangementError::OccupiedByUncapturableTile);
        }
    }

    let distance = crate::clear_path_distance(board, from, to, maximum)
        .ok_or(ArrangementError::NoClearPath)?;

    let clashes=if has_clash(board) {
        let mut resulting_board=board.clone();
        resulting_board.relocate(from,to);
        has_clash(&resulting_board)
    } else {
        relocation_creates_clash(board,from,to,moving)
    };
    if clashes {
        return Err(ArrangementError::CreatesClash);
    }

    Ok(distance)
}

pub fn legal_arrangement_destinations(
    board: &Board,
    player: Player,
    from: Coordinate,
) -> Vec<Coordinate> {
    legal_arrangement_destinations_with_clash_state(board, player, from, has_clash(board))
}

pub(crate) fn legal_arrangement_destinations_with_clash_state(
    board: &Board,
    player: Player,
    from: Coordinate,
    board_has_clash: bool,
) -> Vec<Coordinate> {
    let Some(moving) = board.get(from) else {
        return Vec::new();
    };
    if moving.owner != player || is_trapped(board, from) {
        return Vec::new();
    }
    let Some(maximum) = moving.kind.movement() else {
        return Vec::new();
    };

    reachable_points(board, from, maximum)
        .into_iter()
        .filter(|to| {
            let destination_type = point_type_at(*to);
            if destination_type.is_gate() || !can_end_on(moving.kind, destination_type) {
                return false;
            }
            if board
                .get(*to)
                .is_some_and(|target| !can_capture(board, moving, target))
            {
                return false;
            }
            if board_has_clash {
                let mut resulting_board = board.clone();
                resulting_board.relocate(from, *to);
                !has_clash(&resulting_board)
            } else {
                !relocation_creates_clash(board, from, *to, moving)
            }
        })
        .collect()
}

pub fn apply_arrangement(
    board: &mut Board,
    player: Player,
    from: Coordinate,
    to: Coordinate,
) -> Result<ArrangementResult, ArrangementError> {
    let distance = validate_arrangement(board, player, from, to)?;
    let old_harmonies = harmonies(board);
    let moved = board.get(from).expect("validated source remains occupied");
    let captured = board.relocate(from, to);
    let formed_new_harmony = harmonies(board).iter().any(|harmony| {
        harmony.owner == player
            && !harmony_existed_before_arrangement(harmony, &old_harmonies, from, to)
    });

    Ok(ArrangementResult {
        moved,
        captured,
        distance,
        formed_new_harmony,
    })
}

/// Compares tile pairs rather than coordinates. The moved tile changes from
/// `from` to `to`; any old Harmony involving a captured destination tile is
/// discarded instead of being mistaken for the arriving tile.
fn harmony_existed_before_arrangement(
    current: &crate::Harmony,
    previous: &[crate::Harmony],
    from: Coordinate,
    to: Coordinate,
) -> bool {
    previous.iter().any(|old| {
        if old.owner != current.owner || old.first == to || old.second == to {
            return false;
        }
        let old_first = if old.first == from { to } else { old.first };
        let old_second = if old.second == from { to } else { old.second };
        (current.first == old_first && current.second == old_second)
            || (current.first == old_second && current.second == old_first)
    })
}

pub fn is_trapped(board: &Board, coordinate: Coordinate) -> bool {
    let Some(tile) = board.get(coordinate) else {
        return false;
    };
    if !tile.kind.is_flower() || point_type_at(coordinate).is_gate() {
        return false;
    }

    surrounding_neighbors(coordinate).any(|neighbor| {
        board.get(neighbor).is_some_and(|other| {
            other.owner != tile.owner
                && other.kind == TileKind::Orchid
                && !point_type_at(neighbor).is_gate()
        })
    })
}

pub fn is_drained(board: &Board, coordinate: Coordinate) -> bool {
    let Some(tile) = board.get(coordinate) else {
        return false;
    };
    if point_type_at(coordinate).is_gate()
        || !matches!(tile.kind, TileKind::Basic(_) | TileKind::WhiteLotus)
    {
        return false;
    }

    surrounding_neighbors(coordinate).any(|neighbor| {
        board
            .get(neighbor)
            .is_some_and(|other| other.kind == TileKind::Accent(Accent::Knotweed))
    })
}

pub fn can_capture(board: &Board, attacker: Tile, target: Tile) -> bool {
    if attacker.owner == target.owner || target.kind.is_accent() {
        return false;
    }

    if attacker.kind == TileKind::Orchid
        && has_blooming_white_lotus(board, attacker.owner)
        && target.kind.is_flower()
    {
        return true;
    }

    if target.kind == TileKind::WhiteLotus {
        return false;
    }

    if target.kind == TileKind::Orchid
        && has_blooming_white_lotus(board, target.owner)
        && attacker.kind.is_flower()
    {
        return true;
    }

    attacker.clashes_with(target)
}

fn has_blooming_white_lotus(board: &Board, player: Player) -> bool {
    board.occupied().any(|(coordinate, tile)| {
        tile.owner == player
            && tile.kind == TileKind::WhiteLotus
            && !point_type_at(coordinate).is_gate()
    })
}

pub(crate) fn can_end_on(kind: TileKind, point_type: PointType) -> bool {
    if !point_type.is_playable() {
        return false;
    }
    let TileKind::Basic(flower) = kind else {
        return true;
    };
    let matching_region = match flower.color() {
        FlowerColor::Red => Region::Red,
        FlowerColor::White => Region::White,
    };
    point_type.belongs_to(Region::Neutral) || point_type.belongs_to(matching_region)
}

impl fmt::Display for ArrangementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptySource => "the source point is empty",
            Self::NotOwnedByPlayer => "the source tile belongs to the other player",
            Self::AccentCannotMove => "Accent Tiles cannot be Arranged",
            Self::TrappedByOrchid => "the Flower Tile is trapped by an opposing Orchid",
            Self::SamePoint => "an Arrangement must move at least one point",
            Self::GateDestination => "an Arrangement cannot end in a Gate",
            Self::OppositeGarden => "a Basic Flower cannot end inside the opposite Garden",
            Self::OccupiedByUncapturableTile => "the destination tile cannot be captured",
            Self::NoClearPath => "no clear path reaches the destination within the movement limit",
            Self::CreatesClash => "the Arrangement would leave a Clash on the board",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for ArrangementError {}
