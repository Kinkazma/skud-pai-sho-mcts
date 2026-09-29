use core::fmt;

use paisho_core::{
    all_coordinates, legal_actions, point_type_at, GameOutcome, Player, Position, Region, TileKind,
    TurnPhase,
};

use crate::{
    canonical_coordinate_v1, tile_slot_v1, BOARD_CELL_COUNT_V1, BOARD_SIZE_V1, TILE_KINDS_V1,
    TILE_KIND_COUNT_V1,
};

pub const PLAYABLE_CHANNEL_V1: usize = 0;
pub const GATE_CHANNEL_V1: usize = 1;
pub const RED_REGION_CHANNEL_V1: usize = 2;
pub const WHITE_REGION_CHANNEL_V1: usize = 3;
pub const NEUTRAL_REGION_CHANNEL_V1: usize = 4;
pub const CURRENT_TILE_CHANNEL_START_V1: usize = 5;
pub const OPPONENT_TILE_CHANNEL_START_V1: usize =
    CURRENT_TILE_CHANNEL_START_V1 + TILE_KIND_COUNT_V1;
pub const SPATIAL_CHANNEL_COUNT_V1: usize = OPPONENT_TILE_CHANNEL_START_V1 + TILE_KIND_COUNT_V1;
pub const SPATIAL_VALUE_COUNT_V1: usize = BOARD_CELL_COUNT_V1 * SPATIAL_CHANNEL_COUNT_V1;

pub const CURRENT_RESERVE_START_V1: usize = 0;
pub const OPPONENT_RESERVE_START_V1: usize = CURRENT_RESERVE_START_V1 + TILE_KIND_COUNT_V1;
pub const MAIN_PHASE_FEATURE_V1: usize = OPPONENT_RESERVE_START_V1 + TILE_KIND_COUNT_V1;
pub const HARMONY_BONUS_PHASE_FEATURE_V1: usize = MAIN_PHASE_FEATURE_V1 + 1;
pub const GLOBAL_FEATURE_COUNT_V1: usize = HARMONY_BONUS_PHASE_FEATURE_V1 + 1;

/// Fixed V1 neural state. Spatial values use contiguous NHWC order:
/// `(row * 17 + column) * 29 + channel`.
#[derive(Clone, Debug, PartialEq)]
pub struct StateEncodingV1 {
    perspective: Player,
    spatial_nhwc: Vec<f32>,
    global: [f32; GLOBAL_FEATURE_COUNT_V1],
}

impl StateEncodingV1 {
    pub const fn perspective(&self) -> Player {
        self.perspective
    }

    pub fn spatial_nhwc(&self) -> &[f32] {
        &self.spatial_nhwc
    }

    pub const fn spatial_shape_nhwc(&self) -> [usize; 3] {
        [BOARD_SIZE_V1, BOARD_SIZE_V1, SPATIAL_CHANNEL_COUNT_V1]
    }

    pub const fn global(&self) -> &[f32; GLOBAL_FEATURE_COUNT_V1] {
        &self.global
    }

    pub fn spatial_value(
        &self,
        channel: usize,
        coordinate: paisho_core::Coordinate,
    ) -> Option<f32> {
        (channel < SPATIAL_CHANNEL_COUNT_V1)
            .then(|| self.spatial_nhwc[spatial_index(channel, coordinate)])
    }
}

pub const fn current_tile_channel_v1(kind: TileKind) -> usize {
    CURRENT_TILE_CHANNEL_START_V1 + tile_slot_v1(kind)
}

pub const fn opponent_tile_channel_v1(kind: TileKind) -> usize {
    OPPONENT_TILE_CHANNEL_START_V1 + tile_slot_v1(kind)
}

pub fn encode_state_v1(position: &Position) -> Result<StateEncodingV1, StateEncodingError> {
    if position.outcome() != GameOutcome::Ongoing {
        return Err(StateEncodingError::TerminalPosition);
    }
    if legal_actions(position).is_empty() {
        return Err(StateEncodingError::NoLegalAction);
    }

    Ok(encode_state_features_v1(position))
}

pub(crate) fn encode_state_features_v1(position: &Position) -> StateEncodingV1 {
    let perspective = position.to_move();
    let mut spatial_nhwc = vec![0.0; SPATIAL_VALUE_COUNT_V1];
    for native in all_coordinates() {
        let canonical = canonical_coordinate_v1(perspective, native);
        let point = point_type_at(native);
        set_spatial(
            &mut spatial_nhwc,
            PLAYABLE_CHANNEL_V1,
            canonical,
            binary(point.is_playable()),
        );
        set_spatial(
            &mut spatial_nhwc,
            GATE_CHANNEL_V1,
            canonical,
            binary(point.is_gate()),
        );
        set_spatial(
            &mut spatial_nhwc,
            RED_REGION_CHANNEL_V1,
            canonical,
            binary(point.belongs_to(Region::Red)),
        );
        set_spatial(
            &mut spatial_nhwc,
            WHITE_REGION_CHANNEL_V1,
            canonical,
            binary(point.belongs_to(Region::White)),
        );
        set_spatial(
            &mut spatial_nhwc,
            NEUTRAL_REGION_CHANNEL_V1,
            canonical,
            binary(point.belongs_to(Region::Neutral)),
        );

        if let Some(tile) = position.board().get(native) {
            let channel = if tile.owner == perspective {
                current_tile_channel_v1(tile.kind)
            } else {
                opponent_tile_channel_v1(tile.kind)
            };
            set_spatial(&mut spatial_nhwc, channel, canonical, 1.0);
        }
    }

    let mut global = [0.0; GLOBAL_FEATURE_COUNT_V1];
    encode_reserve(&mut global, CURRENT_RESERVE_START_V1, position, perspective);
    encode_reserve(
        &mut global,
        OPPONENT_RESERVE_START_V1,
        position,
        perspective.opponent(),
    );
    match position.phase() {
        TurnPhase::Main => global[MAIN_PHASE_FEATURE_V1] = 1.0,
        TurnPhase::HarmonyBonus => global[HARMONY_BONUS_PHASE_FEATURE_V1] = 1.0,
    }

    StateEncodingV1 {
        perspective,
        spatial_nhwc,
        global,
    }
}

fn encode_reserve(
    global: &mut [f32; GLOBAL_FEATURE_COUNT_V1],
    start: usize,
    position: &Position,
    player: Player,
) {
    for kind in TILE_KINDS_V1 {
        global[start + tile_slot_v1(kind)] = f32::from(position.reserve(player).count(kind))
            / f32::from(maximum_reserve_count(kind));
    }
}

const fn maximum_reserve_count(kind: TileKind) -> u8 {
    match kind {
        TileKind::Basic(_) => 3,
        TileKind::WhiteLotus | TileKind::Orchid => 1,
        TileKind::Accent(_) => 2,
    }
}

const fn binary(value: bool) -> f32 {
    if value {
        1.0
    } else {
        0.0
    }
}

fn set_spatial(
    spatial_nhwc: &mut [f32],
    channel: usize,
    coordinate: paisho_core::Coordinate,
    value: f32,
) {
    spatial_nhwc[spatial_index(channel, coordinate)] = value;
}

const fn spatial_index(channel: usize, coordinate: paisho_core::Coordinate) -> usize {
    coordinate.dense_index() * SPATIAL_CHANNEL_COUNT_V1 + channel
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateEncodingError {
    TerminalPosition,
    NoLegalAction,
}

impl fmt::Display for StateEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TerminalPosition => {
                formatter.write_str("StateEncodingV1 rejects terminal positions")
            }
            Self::NoLegalAction => {
                formatter.write_str("StateEncodingV1 requires at least one legal decision")
            }
        }
    }
}

impl std::error::Error for StateEncodingError {}
