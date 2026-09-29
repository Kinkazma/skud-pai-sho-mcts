//! Small, trainable CPU value function. Feature extraction performs no search
//! and does not enumerate legal actions. The initial model is the legacy
//! heuristic; additional structural coefficients start at zero.

use core::fmt;
use rayon::prelude::*;

use paisho_core::{
    harmony_crosses_midline, orthogonal_neighbors, point_type_at, surrounding_neighbors,
    visit_harmonies, Accent, Coordinate, GameOutcome, LineOrientation, Player, Position, TileKind,
    TurnPhase, BOARD_SIZE, STANDARD_TILE_KINDS,
};

use crate::{HeuristicWeights, MctsEvaluator};

pub const COMPACT_VALUE_SCHEMA_V1: &str = "paisho-compact-value-features-v1";
pub const COMPACT_FEATURE_COUNT: usize = 64;

/// Every feature is a player difference divided by the corresponding scale.
/// Tempo features are +1 for the player to move and -1 for the other player.
/// Names and scales are part of the persisted model schema, in this exact order.
pub const COMPACT_FEATURE_NAMES: [&str; COMPACT_FEATURE_COUNT] = [
    "harmonies",
    "midline_harmonies",
    "blooming_flowers",
    "flowers",
    "basic_reserve_progress",
    "board_r3",
    "board_r4",
    "board_r5",
    "board_w3",
    "board_w4",
    "board_w5",
    "board_lotus",
    "board_orchid",
    "board_rock",
    "board_wheel",
    "board_knotweed",
    "board_boat",
    "reserve_r3",
    "reserve_r4",
    "reserve_r5",
    "reserve_w3",
    "reserve_w4",
    "reserve_w5",
    "reserve_lotus",
    "reserve_orchid",
    "reserve_rock",
    "reserve_wheel",
    "reserve_knotweed",
    "reserve_boat",
    "blooming_r3",
    "blooming_r4",
    "blooming_r5",
    "blooming_w3",
    "blooming_w4",
    "blooming_w5",
    "blooming_lotus",
    "blooming_orchid",
    "main_tempo",
    "bonus_tempo",
    "harmony_vertices",
    "harmony_degree_at_least_two",
    "harmony_degree_at_least_three",
    "largest_harmony_component",
    "harmony_components",
    "harmony_cycle_rank",
    "horizontal_harmonies",
    "vertical_harmonies",
    "harmony_total_length",
    "flower_octant_coverage",
    "central_flowers",
    "middle_flowers",
    "outer_flowers",
    "axis_flowers",
    "flower_x_span",
    "flower_y_span",
    "flower_centrality",
    "harmony_octant_coverage",
    "central_harmony_vertices",
    "middle_harmony_vertices",
    "outer_harmony_vertices",
    "flower_empty_orthogonal_neighbors",
    "flower_friendly_orthogonal_contacts",
    "knotweed_suppressed_flowers",
    "basic_flower_diversity",
];

pub const COMPACT_FEATURE_SCALES: [f64; COMPACT_FEATURE_COUNT] = [
    128.0, 128.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
    32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 32.0,
    32.0, 32.0, 32.0, 32.0, 32.0, 32.0, 1.0, 1.0, 64.0, 64.0, 64.0, 64.0, 64.0, 128.0, 128.0,
    128.0, 2048.0, 8.0, 32.0, 32.0, 32.0, 32.0, 16.0, 16.0, 512.0, 8.0, 64.0, 64.0, 64.0, 128.0,
    128.0, 32.0, 8.0,
];

#[derive(Clone, Debug, PartialEq)]
pub struct CompactValueFeatures {
    values: [f64; COMPACT_FEATURE_COUNT],
    terminal_value: Option<f64>,
}

impl CompactValueFeatures {
    pub fn extract(position: &Position, perspective: Player) -> Self {
        let terminal_value = match position.outcome() {
            GameOutcome::Ongoing => None,
            GameOutcome::Win(winner) => Some(if winner == perspective { 1.0 } else { -1.0 }),
            GameOutcome::Draw => Some(0.0),
        };
        if terminal_value.is_some() {
            return Self {
                values: [0.0; COMPACT_FEATURE_COUNT],
                terminal_value,
            };
        }

        // Build the suppression mask once instead of probing eight neighbors
        // around every flower (most positions contain no Knotweed at all).
        let mut suppressed = [0u32; BOARD_SIZE];
        for (at, tile) in position.board().occupied() {
            if tile.kind == TileKind::Accent(Accent::Knotweed) {
                for near in surrounding_neighbors(at) {
                    suppressed[near.row()] |= 1 << near.column();
                }
            }
        }
        let mut counts = [[0.0; COMPACT_FEATURE_COUNT]; 2];
        let mut flower_octants = [0_u8; 2];
        let mut basic_kinds = [0_u8; 2];
        let mut minimum = [[8_i8; 2]; 2];
        let mut maximum = [[-8_i8; 2]; 2];
        for (coordinate, tile) in position.board().occupied() {
            let owner = tile.owner.index();
            let row = &mut counts[owner];
            row[5 + tile.kind.index()] += 1.0;
            if !tile.kind.is_flower() {
                continue;
            }
            row[3] += 1.0;
            if !point_type_at(coordinate).is_gate() {
                row[2] += 1.0;
                row[29 + tile.kind.index()] += 1.0;
            }
            if let TileKind::Basic(flower) = tile.kind {
                basic_kinds[owner] |= 1 << flower as u8;
            }
            flower_octants[owner] |= octant_mask(coordinate);
            row[49 + radial_band(coordinate)] += 1.0;
            if coordinate.x() == 0 || coordinate.y() == 0 {
                row[52] += 1.0;
            }
            minimum[owner][0] = minimum[owner][0].min(coordinate.x());
            minimum[owner][1] = minimum[owner][1].min(coordinate.y());
            maximum[owner][0] = maximum[owner][0].max(coordinate.x());
            maximum[owner][1] = maximum[owner][1].max(coordinate.y());
            row[55] += f64::from(16 - coordinate.x().abs() - coordinate.y().abs());
            for neighbor in orthogonal_neighbors(coordinate) {
                match position.board().get(neighbor) {
                    None => row[60] += 1.0,
                    Some(other) if other.owner == tile.owner && other.kind.is_flower() => {
                        row[61] += 1.0
                    }
                    Some(_) => {}
                }
            }
            if suppressed[coordinate.row()] & (1 << coordinate.column()) != 0 {
                row[62] += 1.0;
            }
        }
        for player in [Player::Host, Player::Guest] {
            let owner = player.index();
            let row = &mut counts[owner];
            // Opponent-minus-self reserves matches the legacy progress sign.
            row[4] = -f64::from(position.reserve(player).basic_count());
            for kind in STANDARD_TILE_KINDS {
                row[17 + kind.index()] = f64::from(position.reserve(player).count(kind));
            }
            row[48] = f64::from(flower_octants[owner].count_ones());
            row[63] = f64::from(basic_kinds[owner].count_ones());
            if row[3] > 0.0 {
                row[53] = f64::from(maximum[owner][0] - minimum[owner][0]);
                row[54] = f64::from(maximum[owner][1] - minimum[owner][1]);
            }
        }
        let tempo_slot = if position.phase() == TurnPhase::Main {
            37
        } else {
            38
        };
        counts[position.to_move().index()][tempo_slot] = 1.0;

        let mut graphs = [HarmonyGraph::new(), HarmonyGraph::new()];
        visit_harmonies(position.board(), |harmony| {
            let row = &mut counts[harmony.owner.index()];
            row[0] += 1.0;
            if harmony_crosses_midline(harmony) {
                row[1] += 1.0;
            }
            let orientation_slot = if harmony.orientation == LineOrientation::Horizontal {
                45
            } else {
                46
            };
            row[orientation_slot] += 1.0;
            row[47] += f64::from(
                (harmony.first.x() - harmony.second.x()).abs()
                    + (harmony.first.y() - harmony.second.y()).abs(),
            );
            graphs[harmony.owner.index()].add_edge(harmony.first, harmony.second);
        });
        for (graph, row) in graphs.iter_mut().zip(&mut counts) {
            graph.write_features(row);
        }

        let own = &counts[perspective.index()];
        let other = &counts[perspective.opponent().index()];
        let values = std::array::from_fn(|index| {
            (own[index] - other[index]) / COMPACT_FEATURE_SCALES[index]
        });
        debug_assert!(values
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 1.0));
        Self {
            values,
            terminal_value,
        }
    }

    /// Accepts a cached example only in the pinned normalized representation.
    pub fn from_values(
        values: [f64; COMPACT_FEATURE_COUNT],
        terminal_value: Option<f64>,
    ) -> Result<Self, CompactValueError> {
        if values
            .iter()
            .any(|value| !value.is_finite() || value.abs() > 1.0)
            || terminal_value.is_some_and(|value| ![-1.0, 0.0, 1.0].contains(&value))
        {
            return Err(CompactValueError::InvalidFeatures);
        }
        Ok(Self {
            values,
            terminal_value,
        })
    }

    pub const fn values(&self) -> &[f64; COMPACT_FEATURE_COUNT] {
        &self.values
    }
    pub const fn terminal_value(&self) -> Option<f64> {
        self.terminal_value
    }
}

/// A zero-bias linear value function followed by raw/(1+abs(raw)).
/// Its 64 f64 weights take 512 bytes; game features retain antisymmetry after
/// arbitrary learning updates. Model-file schema/provenance belongs to callers.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactValueModel {
    weights: [f64; COMPACT_FEATURE_COUNT],
}

impl Default for CompactValueModel {
    fn default() -> Self {
        Self::from_heuristic(HeuristicWeights::default()).expect("default heuristic is finite")
    }
}

impl CompactValueModel {
    pub fn from_heuristic(heuristic: HeuristicWeights) -> Result<Self, CompactValueError> {
        let mut weights = [0.0; COMPACT_FEATURE_COUNT];
        let legacy = [
            heuristic.harmony,
            heuristic.midline_harmony,
            heuristic.blooming_flower,
            heuristic.total_flower,
            heuristic.basic_reserve_progress,
        ];
        for (index, weight) in legacy.into_iter().enumerate() {
            weights[index] = f64::from(weight) * COMPACT_FEATURE_SCALES[index];
        }
        Self::from_weights(weights)
    }

    pub fn from_weights(weights: [f64; COMPACT_FEATURE_COUNT]) -> Result<Self, CompactValueError> {
        // Bounding the L1 sum, not only individual coefficients, also prevents
        // overflow for any valid normalized feature vector.
        if weights.iter().any(|weight| !weight.is_finite())
            || !weights
                .iter()
                .map(|weight| weight.abs())
                .sum::<f64>()
                .is_finite()
        {
            return Err(CompactValueError::InvalidWeights);
        }
        Ok(Self { weights })
    }

    pub const fn weights(&self) -> &[f64; COMPACT_FEATURE_COUNT] {
        &self.weights
    }

    pub fn evaluate(&self, position: &Position, perspective: Player) -> f32 {
        self.predict(&CompactValueFeatures::extract(position, perspective)) as f32
    }

    pub fn raw_value(&self, features: &CompactValueFeatures) -> f64 {
        self.weights
            .iter()
            .zip(features.values())
            .map(|(weight, value)| weight * value)
            .sum()
    }

    pub fn predict(&self, features: &CompactValueFeatures) -> f64 {
        features.terminal_value.unwrap_or_else(|| {
            let raw = self.raw_value(features);
            raw / (1.0 + raw.abs())
        })
    }

    /// Derivative of the prediction with respect to each coefficient. Terminal
    /// values are facts from the rules engine, so their derivatives are zero.
    pub fn gradient(&self, features: &CompactValueFeatures) -> [f64; COMPACT_FEATURE_COUNT] {
        if features.terminal_value.is_some() {
            return [0.0; COMPACT_FEATURE_COUNT];
        }
        let reciprocal = 1.0 / (1.0 + self.raw_value(features).abs());
        std::array::from_fn(|index| features.values[index] * reciprocal * reciprocal)
    }

    /// One SGD step on half squared error plus half `l2` times squared weight
    /// norm. Returns the prediction loss before the update, excluding L2.
    /// Validation/overflow errors leave the complete model unchanged.
    pub fn train_step(
        &mut self,
        features: &CompactValueFeatures,
        target: f64,
        learning_rate: f64,
        l2: f64,
    ) -> Result<f64, CompactValueError> {
        if !target.is_finite() || target.abs() > 1.0 {
            return Err(CompactValueError::InvalidTarget);
        }
        if !learning_rate.is_finite() || learning_rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
            return Err(CompactValueError::InvalidLearningRate);
        }
        let error = self.predict(features) - target;
        let loss = 0.5 * error * error;
        if features.terminal_value.is_some() {
            return Ok(loss);
        }
        let gradient = self.gradient(features);
        let next = std::array::from_fn(|index| {
            self.weights[index]
                - learning_rate * (error * gradient[index] + l2 * self.weights[index])
        });
        *self = Self::from_weights(next)?;
        Ok(loss)
    }
}

impl MctsEvaluator for CompactValueModel {
    fn ordering_matches_leaf(&self) -> bool {
        true
    }

    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        _weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        let evaluate =
            |position: &Position| CompactValueModel::evaluate(self, position, perspective);
        // Each position keeps its scalar operation order. Indexed parallel
        // collection preserves candidate order and deterministic search ties.
        Ok(
            if positions.len() >= crate::mcts::PARALLEL_ACTION_RANKING_THRESHOLD {
                positions.par_iter().map(evaluate).collect()
            } else {
                positions.iter().map(evaluate).collect()
            },
        )
    }

    fn evaluate_leaf(
        &self,
        position: &Position,
        perspective: Player,
        _weights: HeuristicWeights,
    ) -> Result<f32, String> {
        Ok(CompactValueModel::evaluate(self, position, perspective))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompactValueError {
    InvalidFeatures,
    InvalidWeights,
    InvalidTarget,
    InvalidLearningRate,
}

impl fmt::Display for CompactValueError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidFeatures => "compact value features must be finite and normalized, with an exact optional terminal value",
            Self::InvalidWeights => "compact value weights and their absolute sum must be finite",
            Self::InvalidTarget => "compact value target must be finite in [-1, 1]",
            Self::InvalidLearningRate => "compact value learning rate must be positive and finite; L2 must be nonnegative and finite",
        })
    }
}
impl std::error::Error for CompactValueError {}

fn radial_band(coordinate: Coordinate) -> usize {
    let squared = i16::from(coordinate.x()).pow(2) + i16::from(coordinate.y()).pow(2);
    if squared <= 9 {
        0
    } else if squared <= 36 {
        1
    } else {
        2
    }
}

/// Integer eight-sector coverage; the exact centre belongs to no sector.
fn octant_mask(coordinate: Coordinate) -> u8 {
    let x = coordinate.x();
    let y = coordinate.y();
    if x == 0 && y == 0 {
        return 0;
    }
    let octant = if x > 0 && y >= 0 {
        usize::from(y >= x)
    } else if y > 0 && x <= 0 {
        2 + usize::from(-x >= y)
    } else if x < 0 && y <= 0 {
        4 + usize::from(-y >= -x)
    } else {
        6 + usize::from(x >= -y)
    };
    1 << octant
}

const BOARD_CELLS: usize = BOARD_SIZE * BOARD_SIZE;

struct HarmonyGraph {
    parent: [u16; BOARD_CELLS],
    degree: [u8; BOARD_CELLS],
    active: Vec<Coordinate>,
    edges: usize,
}

impl HarmonyGraph {
    fn new() -> Self {
        Self {
            parent: [0; BOARD_CELLS],
            degree: [0; BOARD_CELLS],
            active: Vec::new(),
            edges: 0,
        }
    }

    fn root(&mut self, mut index: usize) -> usize {
        while usize::from(self.parent[index]) != index {
            self.parent[index] = self.parent[usize::from(self.parent[index])];
            index = usize::from(self.parent[index]);
        }
        index
    }

    fn add_edge(&mut self, first: Coordinate, second: Coordinate) {
        self.edges += 1;
        for coordinate in [first, second] {
            let index = coordinate.dense_index();
            if self.degree[index] == 0 {
                self.parent[index] = index as u16;
                self.active.push(coordinate);
            }
            self.degree[index] += 1;
        }
        let first_root = self.root(first.dense_index());
        let second_root = self.root(second.dense_index());
        self.parent[first_root] = second_root as u16;
    }

    fn write_features(&mut self, row: &mut [f64; COMPACT_FEATURE_COUNT]) {
        if self.active.is_empty() {
            return;
        }
        let mut component_sizes = [0_u16; BOARD_CELLS];
        let mut octants = 0_u8;
        for active_index in 0..self.active.len() {
            let coordinate = self.active[active_index];
            let index = coordinate.dense_index();
            row[39] += 1.0;
            if self.degree[index] >= 2 {
                row[40] += 1.0;
            }
            if self.degree[index] >= 3 {
                row[41] += 1.0;
            }
            let root = self.root(index);
            if component_sizes[root] == 0 {
                row[43] += 1.0;
            }
            component_sizes[root] += 1;
            row[42] = row[42].max(f64::from(component_sizes[root]));
            octants |= octant_mask(coordinate);
            row[57 + radial_band(coordinate)] += 1.0;
        }
        row[44] = self.edges as f64 - row[39] + row[43];
        row[56] = f64::from(octants.count_ones());
    }
}

#[cfg(test)]
mod sparse_graph_tests {
    use super::*;
    #[test]
    fn graph_features_match_independent_component_traversal() {
        let points: Vec<_> = paisho_core::playable_coordinates().take(24).collect();
        let mut rng = 73u64;
        for edges in 0..64 {
            let mut graph = HarmonyGraph::new();
            let mut adjacent = vec![Vec::new(); points.len()];
            for _ in 0..edges {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let a = (rng >> 32) as usize % points.len();
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let b = (rng >> 32) as usize % points.len();
                graph.add_edge(points[a], points[b]);
                adjacent[a].push(b);
                adjacent[b].push(a);
            }
            let mut seen = vec![false; points.len()];
            let mut sizes = vec![];
            for first in 0..points.len() {
                if seen[first] || adjacent[first].is_empty() {
                    continue;
                }
                let mut pending = vec![first];
                seen[first] = true;
                let mut size = 0;
                while let Some(a) = pending.pop() {
                    size += 1;
                    for &b in &adjacent[a] {
                        if !seen[b] {
                            seen[b] = true;
                            pending.push(b);
                        }
                    }
                }
                sizes.push(size);
            }
            let vertices = adjacent.iter().filter(|a| !a.is_empty()).count();
            let mut row = [0.; COMPACT_FEATURE_COUNT];
            graph.write_features(&mut row);
            assert_eq!(row[39], vertices as f64);
            assert_eq!(
                row[40],
                adjacent.iter().filter(|a| a.len() >= 2).count() as f64
            );
            assert_eq!(
                row[41],
                adjacent.iter().filter(|a| a.len() >= 3).count() as f64
            );
            assert_eq!(row[42], sizes.iter().copied().max().unwrap_or(0) as f64);
            assert_eq!(row[43], sizes.len() as f64);
            assert_eq!(row[44], edges as f64 - vertices as f64 + sizes.len() as f64);
        }
    }
}
