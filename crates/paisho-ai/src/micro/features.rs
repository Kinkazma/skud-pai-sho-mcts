//! Gen4 input order is versioned. No action application or legal enumeration here.
use crate::CompactValueFeatures;
use paisho_core::{
    point_type_at, AccentPlacement, Action, Position, TileKind, TurnPhase, STANDARD_TILE_KINDS,
};

pub const MICRO_INPUTS: usize = 128;
pub const MICRO_ACTION_INPUTS: usize = 32;
pub const MICRO_FEATURE_SCHEMA: &str = "paisho-micro-state128-action32-v1";

/// Inputs 0..64: legacy features from the player-to-move perspective.
/// 64..88: own then opponent reserves, twelve standard tile kinds, /32.
/// 88..124: own/opponent × basic/special/accent × six zones, /32.
/// Zones: centre (Chebyshev <=2), rim (>=6), four intermediate quadrants.
/// 124..128: main phase, bonus phase, turns/(turns+100), occupied/64.
/// Seat-relative features, absolute board coordinates; terminal targets are exact
/// game outcomes, never a prediction from this representation.
pub fn micro_state_features(position: &Position) -> [f64; MICRO_INPUTS] {
    let mut x = [0.0; MICRO_INPUTS];
    let player = position.to_move();
    x[..64].copy_from_slice(CompactValueFeatures::extract(position, player).values());
    for (seat, owner) in [player, player.opponent()].into_iter().enumerate() {
        for (kind, tile) in STANDARD_TILE_KINDS.iter().enumerate() {
            x[64 + seat * 12 + kind] = f64::from(position.reserve(owner).count(*tile)) / 32.0;
        }
    }
    for (at, tile) in position.board().occupied() {
        let seat = usize::from(tile.owner != player);
        let family = match tile.kind {
            TileKind::Basic(_) => 0,
            TileKind::WhiteLotus | TileKind::Orchid => 1,
            TileKind::Accent(_) => 2,
        };
        let radius = at.x().abs().max(at.y().abs());
        let zone = if radius <= 2 {
            0
        } else if radius >= 6 {
            1
        } else {
            2 + usize::from(at.x() >= 0) + 2 * usize::from(at.y() >= 0)
        };
        x[88 + seat * 18 + family * 6 + zone] += 1.0 / 32.0;
    }
    x[124] = f64::from(position.phase() == TurnPhase::Main);
    x[125] = f64::from(position.phase() == TurnPhase::HarmonyBonus);
    let turns = f64::from(position.completed_turns());
    x[126] = turns / (turns + 100.0);
    x[127] = position.board().occupied_count() as f64 / 64.0;
    x
}

/// Six action types, twelve tile kinds, from/to coordinates, displacement,
/// distance/centrality and six local flags. All extracted before applying a move.
pub fn micro_action_features(position: &Position, action: Action) -> [f64; MICRO_ACTION_INPUTS] {
    let mut x = [0.0; MICRO_ACTION_INPUTS];
    let (category, kind, from, to, boat_move) = match action {
        Action::Plant { flower, gate } => {
            (0, Some(TileKind::Basic(flower)), None, Some(gate), false)
        }
        Action::Arrange { from, to } => (
            1,
            position.board().get(from).map(|t| t.kind),
            Some(from),
            Some(to),
            false,
        ),
        Action::SkipHarmonyBonus => (2, None, None, None, false),
        Action::PlayAccent { accent, placement } => match placement {
            AccentPlacement::At(at) => (3, Some(TileKind::Accent(accent)), None, Some(at), false),
            AccentPlacement::BoatMove {
                flower,
                destination,
            } => (
                3,
                Some(TileKind::Accent(accent)),
                Some(flower),
                Some(destination),
                true,
            ),
        },
        Action::PlantSpecial { flower, gate } => (4, Some(flower.kind()), None, Some(gate), false),
        Action::BonusPlantBasic { flower, gate } => {
            (5, Some(TileKind::Basic(flower)), None, Some(gate), false)
        }
    };
    x[category] = 1.0;
    if let Some(kind) = kind {
        x[6 + kind.index()] = 1.0;
    }
    if let Some(at) = from {
        x[18] = f64::from(at.x()) / 8.0;
        x[19] = f64::from(at.y()) / 8.0;
        x[26] = 1.0;
    }
    if let Some(at) = to {
        x[20] = f64::from(at.x()) / 8.0;
        x[21] = f64::from(at.y()) / 8.0;
        x[25] = 1.0 - f64::from(at.x().abs().max(at.y().abs())) / 8.0;
        x[30] = f64::from(point_type_at(at).is_gate());
        if let Some(tile) = position.board().get(at) {
            x[27] = f64::from(tile.owner == position.to_move());
            x[28] = f64::from(tile.owner != position.to_move());
            x[29] = f64::from(tile.kind.is_flower());
        }
    }
    if from.is_some() && to.is_some() {
        x[22] = (x[20] - x[18]) / 2.0;
        x[23] = (x[21] - x[19]) / 2.0;
        x[24] = (x[22].abs() + x[23].abs()) / 2.0;
    }
    x[31] = f64::from(boat_move);
    x
}
