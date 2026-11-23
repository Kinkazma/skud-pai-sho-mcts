//! Deterministic domain primitives for standard Skud Pai Sho.

mod accent;
mod board;
mod coordinate;
mod game;
mod harmony;
mod legality;
mod movement;
mod notation;
mod player;
mod record;
mod reserve;
mod ring;
mod rules;
mod setup;
mod tile;
mod topology;

pub use accent::{
    apply_accent, legal_accent_placements, AccentEffect, AccentError, AccentPlacement,
};
pub use board::{Board, BoardError, CELL_COUNT};
pub use coordinate::{Coordinate, CoordinateError, BOARD_RADIUS, BOARD_SIZE};
pub use game::{legal_actions, Action, ActionResult, ApplyError};
pub use harmony::{clashes, harmonies, visit_harmonies, has_clash, Clash, Harmony, LineOrientation};
pub use legality::{
    apply_arrangement, can_capture, is_drained, is_trapped, legal_arrangement_destinations,
    validate_arrangement, ArrangementError, ArrangementResult,
};
pub use movement::{clear_path_distance, reachable_points};
pub use notation::ActionNotationError;
pub use player::Player;
pub use record::{GameRecord, RecordParseError, ReplayError};
pub use reserve::{AccentLoadout, AccentLoadoutError, Reserve};
pub use ring::{
    harmony_crosses_midline, harmony_ring_owners, harmony_ring_owners_for_profile,
    midline_crossing_harmony_count,
};
pub use rules::{
    RuleProfile, RuleProfileId, RuleProfileIdError, LEGACY_RULE_PROFILE, STANDARD_RULE_PROFILE,
};
pub use setup::{GameOutcome, Position, StandardSetup, TurnPhase};
pub use tile::{
    Accent, BasicFlower, FlowerColor, SpecialFlower, Tile, TileCodeError, TileKind, ACCENTS,
    BASIC_FLOWERS, SPECIAL_FLOWERS, STANDARD_TILE_KINDS,
};
pub use topology::{
    all_coordinates, orthogonal_neighbors, playable_coordinates, point_type_at,
    surrounding_neighbors, PointType, Region, EAST_GATE, NORTH_GATE, PLAYABLE_POINT_COUNT,
    SOUTH_GATE, WEST_GATE,
};
