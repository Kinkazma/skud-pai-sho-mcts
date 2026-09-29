//! Persistent ~200k memory using the TTT-MLP block, read only at actual roots.
//! Supervised consolidation, not test-time/meta-learning. Every input is available
//! during both replay and inference without reconstructing successor positions.
use super::*;
mod backward;
mod forward;
mod gelu;
mod query;
mod scratch;
use scratch::Scratch;
mod weights;
pub(super) use weights::BackwardWeights;
#[cfg(test)]
mod tests;
pub(super) use forward::Cache;
pub const MICRO_NEURAL_MEMORY_START: usize = MICRO_DEEP_VALUE_PARAMETERS;
pub const MICRO_NEURAL_MEMORY_WEIGHTS: usize = 199_292;
pub const MICRO_NEURAL_MEMORY_PARAMETERS: usize =
    MICRO_NEURAL_MEMORY_START + MICRO_NEURAL_MEMORY_WEIGHTS;
pub const MICRO_NEURAL_MEMORY_MODEL_SCHEMA: &str = "paisho-micro-neural-memory130-520-v6";
const INPUT: usize = 483;
const WIDTH: usize = 130;
const EXPAND: usize = 520;
const W0: usize = 0;
const B0: usize = INPUT * WIDTH;
const W1: usize = B0 + WIDTH;
const B1: usize = W1 + WIDTH * EXPAND;
const W2: usize = B1 + EXPAND;
const B2: usize = W2 + EXPAND * WIDTH;
const GAMMA: usize = B2 + WIDTH;
const BETA: usize = GAMMA + WIDTH;
const OUT: usize = BETA + WIDTH;
const BOUT: usize = OUT + WIDTH * 2;

pub(super) struct SideGradient {
    pub value: f64,
    pub memory: [f64; 32],
    pub logits: Vec<f64>,
}
pub(super) fn inference(model: &MicroModel, state: &[f64], actions: &[[f64; 32]],
    memory: &[f64; 32], value: f64, logits: &[f64]) -> Cache {
    Cache::inference(&model.parameters[MICRO_NEURAL_MEMORY_START..], state, actions, memory, value, logits)
}
impl MicroModel {
    pub fn has_neural_memory(&self) -> bool {
        self.parameters.len() >= MICRO_NEURAL_MEMORY_PARAMETERS
    }
    /// Neutral, idempotent migration; every old parameter and the bank Arc survive.
    pub fn with_neural_memory(&self, seed: u64) -> Self {
        if self.has_neural_memory() {
            return self.clone();
        }
        let base = self.with_deep_value(seed);
        let mut w = base.parameters.as_ref().clone();
        w.resize(MICRO_NEURAL_MEMORY_PARAMETERS, 0.);
        let tail = &mut w[MICRO_NEURAL_MEMORY_START..];
        let mut rng = StableRng::new(seed);
        for (start, end, inputs, outputs) in [
            (W0, B0, INPUT, WIDTH),
            (W1, B1, WIDTH, EXPAND),
            (W2, B2, EXPAND, WIDTH),
        ] {
            let scale = (6. / (inputs + outputs) as f64).sqrt();
            for x in &mut tail[start..end] {
                *x = (rng.next_f64() * 2. - 1.) * scale;
            }
        }
        tail[GAMMA..BETA].fill(1.);
        let mut model = Self::from_parameters(w).expect("finite neural memory migration");
        model.memory = base.memory.clone();
        model
    }
    pub(super) fn neural_memory_active(&self) -> bool {
        self.has_neural_memory()
            && self.parameters
                [MICRO_NEURAL_MEMORY_START + OUT..MICRO_NEURAL_MEMORY_START + BOUT + 2]
                .iter()
                .any(|x| *x != 0.)
    }
    pub(super) fn neural_forward(
        &self,
        state: &[f64],
        actions: &[[f64; 32]],
        memory: &[f64; 32],
        value: f64,
        logits: &[f64],
    ) -> Cache {
        Cache::new(
            &self.parameters[MICRO_NEURAL_MEMORY_START..],
            state,
            actions,
            memory,
            value,
            logits,
        )
    }
    pub(super) fn neural_backward(&self, cache: &Cache, d: &[f64], g: &mut [f64]) -> SideGradient {
        cache.backward(
            &self.parameters[MICRO_NEURAL_MEMORY_START..],
            d,
            &mut g[MICRO_NEURAL_MEMORY_START..],
            self.neural_backward_weights.as_deref(),
        )
    }
    /// Auxiliary predictions are bounded estimates; never solver certificates or
    /// extra leaf evaluations. The existing value-policy coupling remains single.
    pub fn neural_action_values(
        &self,
        state: &[f64],
        actions: &[[f64; 32]],
        excluded: u64,
    ) -> Result<Option<Vec<f64>>, String> {
        if !self.has_neural_memory() {
            return Ok(None);
        }
        let emb = self.embed(state);
        let vector = self
            .memory_context(state, excluded)?
            .as_ref()
            .map(|c| self.memory_vector(c))
            .unwrap_or([0.; 32]);
        let cache = self.neural_forward(
            state,
            actions,
            &vector,
            emb.value,
            &Self::logits(&emb, actions),
        );
        Ok(Some(
            cache.output.chunks_exact(2).map(|y| y[1].tanh()).collect(),
        ))
    }
}
