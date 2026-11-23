use core::fmt;

use crate::{
    all_coordinates, has_clash, point_type_at, surrounding_neighbors, Accent, Board, Coordinate,
    Player, Tile, TileKind,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum AccentPlacement {
    At(Coordinate),
    BoatMove {
        flower: Coordinate,
        destination: Coordinate,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccentEffect {
    pub removed: Option<Tile>,
    pub moved_flower: Option<(Coordinate, Coordinate, Tile)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccentError {
    WrongPlacementShape,
    PointMustBeOpen,
    GateNotAllowed,
    NonPlayablePoint,
    WheelCannotMoveRock,
    WheelWouldMoveOffBoard,
    WheelWouldMoveIntoOrOutOfGate,
    OppositeGarden,
    BoatRequiresAccentOrBloomingFlower,
    BoatDestinationMustSurroundFlower,
    BoatDestinationMustBeOpen,
    CreatesClash,
}

pub fn legal_accent_placements(
    board: &Board,
    player: Player,
    accent: Accent,
) -> Vec<AccentPlacement> {
    let mut placements = Vec::new();
    match accent {
        Accent::Rock | Accent::Wheel | Accent::Knotweed => {
            for coordinate in all_coordinates() {
                let placement = AccentPlacement::At(coordinate);
                if validate_accent(board, player, accent, placement).is_ok() {
                    placements.push(placement);
                }
            }
        }
        Accent::Boat => {
            for (coordinate, target) in board.occupied() {
                if target.kind.is_accent() {
                    let placement = AccentPlacement::At(coordinate);
                    if validate_accent(board, player, accent, placement).is_ok() {
                        placements.push(placement);
                    }
                } else if target.kind.is_flower() && !point_type_at(coordinate).is_gate() {
                    for destination in surrounding_neighbors(coordinate) {
                        let placement = AccentPlacement::BoatMove {
                            flower: coordinate,
                            destination,
                        };
                        if validate_accent(board, player, accent, placement).is_ok() {
                            placements.push(placement);
                        }
                    }
                }
            }
        }
    }
    placements
}

pub fn apply_accent(
    board: &mut Board,
    player: Player,
    accent: Accent,
    placement: AccentPlacement,
) -> Result<AccentEffect, AccentError> {
    let (candidate, effect) = validate_accent(board, player, accent, placement)?;
    *board = candidate;
    Ok(effect)
}

fn validate_accent(
    board: &Board,
    player: Player,
    accent: Accent,
    placement: AccentPlacement,
) -> Result<(Board, AccentEffect), AccentError> {
    let mut candidate = board.clone();
    let effect = match (accent, placement) {
        (Accent::Rock, AccentPlacement::At(at)) | (Accent::Knotweed, AccentPlacement::At(at)) => {
            validate_open_accent_point(&candidate, at)?;
            candidate
                .place(at, Tile::new(player, TileKind::Accent(accent)))
                .expect("validated open playable point");
            AccentEffect {
                removed: None,
                moved_flower: None,
            }
        }
        (Accent::Wheel, AccentPlacement::At(at)) => apply_wheel(&mut candidate, player, at)?,
        (Accent::Boat, AccentPlacement::At(at)) => apply_boat_removal(&mut candidate, at)?,
        (
            Accent::Boat,
            AccentPlacement::BoatMove {
                flower,
                destination,
            },
        ) => apply_boat_move(&mut candidate, player, flower, destination)?,
        _ => return Err(AccentError::WrongPlacementShape),
    };

    if has_clash(&candidate) {
        return Err(AccentError::CreatesClash);
    }
    Ok((candidate, effect))
}

fn validate_open_accent_point(board: &Board, at: Coordinate) -> Result<(), AccentError> {
    let point_type = point_type_at(at);
    if !point_type.is_playable() {
        return Err(AccentError::NonPlayablePoint);
    }
    if point_type.is_gate() {
        return Err(AccentError::GateNotAllowed);
    }
    if !board.is_empty(at) {
        return Err(AccentError::PointMustBeOpen);
    }
    Ok(())
}

fn apply_wheel(
    board: &mut Board,
    player: Player,
    at: Coordinate,
) -> Result<AccentEffect, AccentError> {
    validate_open_accent_point(board, at)?;
    let mut rotations = Vec::new();

    // The Wheel advances the eight surrounding points by one place (45°),
    // matching the clockwise perimeter walk used by the reference game.
    const NEIGHBOR_DELTAS: [(i8, i8); 8] = [
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
    ];

    for (index, (delta_x, delta_y)) in NEIGHBOR_DELTAS.into_iter().enumerate() {
        let Some(source) = at.translated(delta_x, delta_y) else {
            continue;
        };
        let Some(tile) = board.get(source) else {
            continue;
        };
        if tile.kind == TileKind::Accent(Accent::Rock) {
            return Err(AccentError::WheelCannotMoveRock);
        }
        if point_type_at(source).is_gate() {
            return Err(AccentError::WheelWouldMoveIntoOrOutOfGate);
        }

        let target_delta = NEIGHBOR_DELTAS[(index + 1) % NEIGHBOR_DELTAS.len()];
        let Some(target) = at.translated(target_delta.0, target_delta.1) else {
            return Err(AccentError::WheelWouldMoveOffBoard);
        };
        let target_type = point_type_at(target);
        if !target_type.is_playable() {
            return Err(AccentError::WheelWouldMoveOffBoard);
        }
        if target_type.is_gate() {
            return Err(AccentError::WheelWouldMoveIntoOrOutOfGate);
        }
        if !crate::legality::can_end_on(tile.kind, target_type) {
            return Err(AccentError::OppositeGarden);
        }
        rotations.push((source, target, tile));
    }

    for (source, _, _) in &rotations {
        board.remove(*source);
    }
    board
        .place(at, Tile::new(player, TileKind::Accent(Accent::Wheel)))
        .expect("validated Wheel point remains open");
    for (_, target, tile) in rotations {
        board
            .place(target, tile)
            .expect("clockwise rotation is a one-to-one permutation");
    }

    Ok(AccentEffect {
        removed: None,
        moved_flower: None,
    })
}

fn apply_boat_removal(board: &mut Board, at: Coordinate) -> Result<AccentEffect, AccentError> {
    if point_type_at(at).is_gate() {
        return Err(AccentError::GateNotAllowed);
    }
    let target = board
        .get(at)
        .ok_or(AccentError::BoatRequiresAccentOrBloomingFlower)?;
    if !target.kind.is_accent() {
        return Err(AccentError::BoatRequiresAccentOrBloomingFlower);
    }
    board.remove(at);
    Ok(AccentEffect {
        removed: Some(target),
        moved_flower: None,
    })
}

fn apply_boat_move(
    board: &mut Board,
    player: Player,
    flower: Coordinate,
    destination: Coordinate,
) -> Result<AccentEffect, AccentError> {
    if point_type_at(flower).is_gate() {
        return Err(AccentError::BoatRequiresAccentOrBloomingFlower);
    }
    let moving = board
        .get(flower)
        .filter(|tile| tile.kind.is_flower())
        .ok_or(AccentError::BoatRequiresAccentOrBloomingFlower)?;
    if !surrounding_neighbors(flower).any(|neighbor| neighbor == destination) {
        return Err(AccentError::BoatDestinationMustSurroundFlower);
    }
    if !board.is_empty(destination) {
        return Err(AccentError::BoatDestinationMustBeOpen);
    }
    let destination_type = point_type_at(destination);
    if destination_type.is_gate() {
        return Err(AccentError::WheelWouldMoveIntoOrOutOfGate);
    }
    if !crate::legality::can_end_on(moving.kind, destination_type) {
        return Err(AccentError::OppositeGarden);
    }

    board.remove(flower);
    board
        .place(flower, Tile::new(player, TileKind::Accent(Accent::Boat)))
        .expect("the moved Flower leaves its point open");
    board
        .place(destination, moving)
        .expect("validated Boat destination remains open");
    Ok(AccentEffect {
        removed: None,
        moved_flower: Some((flower, destination, moving)),
    })
}

impl fmt::Display for AccentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::WrongPlacementShape => "that placement does not match the Accent Tile",
            Self::PointMustBeOpen => "this Accent Tile requires an open point",
            Self::GateNotAllowed => "Accent Tiles cannot be played in a Gate",
            Self::NonPlayablePoint => "Accent Tiles must be played on the board",
            Self::WheelCannotMoveRock => "a Wheel cannot move a Rock",
            Self::WheelWouldMoveOffBoard => "the Wheel would move a tile off the board",
            Self::WheelWouldMoveIntoOrOutOfGate => {
                "an Accent Tile cannot move a tile into or out of a Gate"
            }
            Self::OppositeGarden => {
                "the Accent effect would move a Basic Flower into the opposite Garden"
            }
            Self::BoatRequiresAccentOrBloomingFlower => {
                "a Boat must be played on an Accent Tile or Blooming Flower"
            }
            Self::BoatDestinationMustSurroundFlower => {
                "a Boat must move the Flower to a surrounding point"
            }
            Self::BoatDestinationMustBeOpen => "the Boat's Flower destination must be open",
            Self::CreatesClash => "the Accent action would leave a Clash on the board",
        };
        formatter.write_str(message)
    }
}

impl std::error::Error for AccentError {}
