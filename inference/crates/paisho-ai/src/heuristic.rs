use paisho_core::{
    harmony_crosses_midline, point_type_at, visit_harmonies, Action, GameOutcome, Player, Position,
};

use crate::{Agent, AgentError, AgentTelemetry, StableRng};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeuristicWeights {
    pub harmony: f32,
    pub midline_harmony: f32,
    pub blooming_flower: f32,
    pub total_flower: f32,
    pub basic_reserve_progress: f32,
}

impl Default for HeuristicWeights {
    fn default() -> Self {
        Self {
            harmony: 0.80,
            midline_harmony: 0.35,
            blooming_flower: 0.08,
            total_flower: 0.04,
            basic_reserve_progress: 0.02,
        }
    }
}

impl HeuristicWeights {
    pub fn all_finite(self) -> bool {
        self.harmony.is_finite()
            && self.midline_harmony.is_finite()
            && self.blooming_flower.is_finite()
            && self.total_flower.is_finite()
            && self.basic_reserve_progress.is_finite()
    }
}

/// Evaluates from `perspective` in `[-1, 1]`; terminal values are exact.
pub fn evaluate_position(
    position: &Position,
    perspective: Player,
    weights: HeuristicWeights,
) -> f32 {
    match position.outcome() {
        GameOutcome::Win(winner) => {
            return if winner == perspective { 1.0 } else { -1.0 };
        }
        GameOutcome::Draw => return 0.0,
        GameOutcome::Ongoing => {}
    }

    let mut harmony_counts = [0_u16; 2];
    let mut midline_counts = [0_u16; 2];
    visit_harmonies(position.board(), |harmony| {
        harmony_counts[harmony.owner.index()] += 1;
        if harmony_crosses_midline(harmony) {
            midline_counts[harmony.owner.index()] += 1;
        }
    });
    let opponent = perspective.opponent();
    let harmony_difference = f64::from(harmony_counts[perspective.index()])
        - f64::from(harmony_counts[opponent.index()]);
    let midline_difference = f64::from(midline_counts[perspective.index()])
        - f64::from(midline_counts[opponent.index()]);

    let mut blooming_difference = 0.0_f64;
    let mut flower_difference = 0.0_f64;
    for (coordinate, tile) in position.board().occupied() {
        if !tile.kind.is_flower() {
            continue;
        }
        let sign = if tile.owner == perspective { 1.0 } else { -1.0 };
        flower_difference += sign;
        if !point_type_at(coordinate).is_gate() {
            blooming_difference += sign;
        }
    }

    let reserve_progress = f64::from(position.reserve(opponent).basic_count())
        - f64::from(position.reserve(perspective).basic_count());
    let raw = f64::from(weights.harmony) * harmony_difference
        + f64::from(weights.midline_harmony) * midline_difference
        + f64::from(weights.blooming_flower) * blooming_difference
        + f64::from(weights.total_flower) * flower_difference
        + f64::from(weights.basic_reserve_progress) * reserve_progress;
    (raw / (1.0 + raw.abs())) as f32
}

#[derive(Clone, Debug, PartialEq)]
pub struct GreedyAgent {
    rng: StableRng,
    weights: HeuristicWeights,
    decisions: usize,
    evaluated_actions: usize,
}

impl GreedyAgent {
    pub fn new(seed: u64, weights: HeuristicWeights) -> Self {
        Self {
            rng: StableRng::new(seed),
            weights,
            decisions: 0,
            evaluated_actions: 0,
        }
    }
}

impl Agent for GreedyAgent {
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        self.decisions += 1;
        self.evaluated_actions += legal_actions.len();
        let perspective = position.to_move();
        let mut best_score = f32::NEG_INFINITY;
        let mut best_indices = Vec::new();

        for (index, action) in legal_actions.iter().copied().enumerate() {
            let mut candidate = position.clone();
            if candidate.apply(action).is_err() {
                continue;
            }
            let score = evaluate_position(&candidate, perspective, self.weights);
            match score.total_cmp(&best_score) {
                core::cmp::Ordering::Greater => {
                    best_score = score;
                    best_indices.clear();
                    best_indices.push(index);
                }
                core::cmp::Ordering::Equal => best_indices.push(index),
                core::cmp::Ordering::Less => {}
            }
        }

        Ok(best_indices[self.rng.index(best_indices.len())])
    }

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry {
            decisions: self.decisions,
            simulations: 0,
            evaluated_actions: self.evaluated_actions,
            ..AgentTelemetry::default()
        }
    }

    fn reset_telemetry(&mut self) {
        self.decisions = 0;
        self.evaluated_actions = 0;
    }
}
