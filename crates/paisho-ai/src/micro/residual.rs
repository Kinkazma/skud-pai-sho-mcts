//! Function-preserving upgrade: the old policy plus a small nonlinear action head.
//! Immutable weights are shared by cached embeddings, never copied per action.
use super::*;
use std::sync::Arc;
pub const MICRO_RESIDUAL_HIDDEN: usize = 16;
pub const MICRO_RESIDUAL_PARAMETERS: usize = 6274;
pub const MICRO_RESIDUAL_MODEL_SCHEMA: &str = "paisho-micro-value-policy-128-32-residual16-v2";
const A: usize = MICRO_PARAMETERS;
const H: usize = A + 16 * MICRO_ACTION_INPUTS;
const B: usize = H + 16 * MICRO_HIDDEN;
const O: usize = B + 16;
const OB: usize = O + 16;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct ResidualWeights {
    // Only the residual suffix; old weights remain owned by MicroModel.
    parameters: Vec<f64>,
    action_columns: [[f64; MICRO_RESIDUAL_HIDDEN]; MICRO_ACTION_INPUTS],
    pub active: bool,
}
#[derive(Clone, Debug)]
pub(super) struct ResidualEmbedding {
    pub weights: Arc<ResidualWeights>,
    context: [f64; MICRO_RESIDUAL_HIDDEN],
}
impl MicroModel {
    pub fn schema(&self) -> &'static str {
        if self.parameters.len() == MICRO_MEMORY_PARAMETERS {
            MICRO_MEMORY_MODEL_SCHEMA
        } else if self.residual.is_some() {
            MICRO_RESIDUAL_MODEL_SCHEMA
        } else {
            MICRO_MODEL_SCHEMA
        }
    }
    /// Preserve every old weight and every initial logit. Repeated upgrade is a no-op.
    pub fn with_residual_policy(&self, seed: u64) -> Self {
        if self.residual.is_some() {
            return self.clone();
        }
        let mut parameters = self.parameters.clone();
        parameters.resize(MICRO_RESIDUAL_PARAMETERS, 0.0);
        let mut rng = StableRng::new(seed);
        for w in &mut parameters[A..B] {
            *w = (rng.next_f64() * 2.0 - 1.0) * (6.0_f64 / 80.0).sqrt();
        }
        Self::from_parameters(parameters).expect("finite deterministic upgrade")
    }
}
impl ResidualWeights {
    pub fn from_parameters(w: &[f64]) -> Option<Self> {
        (w.len() >= MICRO_RESIDUAL_PARAMETERS).then(|| Self {
            parameters: w[A..MICRO_RESIDUAL_PARAMETERS].to_vec(),
            action_columns: std::array::from_fn(|k| {
                std::array::from_fn(|j| w[A + j * MICRO_ACTION_INPUTS + k])
            }),
            active: w[O..MICRO_RESIDUAL_PARAMETERS].iter().any(|w| *w != 0.0),
        })
    }
    fn w(&self, index: usize) -> f64 {
        self.parameters[index - A]
    }
    pub fn embed(self: &Arc<Self>, hidden: &[f64; MICRO_HIDDEN]) -> ResidualEmbedding {
        let mut context = [0.0; MICRO_RESIDUAL_HIDDEN];
        for (j, c) in context.iter_mut().enumerate() {
            *c = self.w(B + j)
                + hidden
                    .iter()
                    .enumerate()
                    .map(|(k, h)| h * self.w(H + j * MICRO_HIDDEN + k))
                    .sum::<f64>();
        }
        ResidualEmbedding {
            weights: self.clone(),
            context,
        }
    }
}
impl ResidualEmbedding {
    fn activations(&self, action: &[f64; MICRO_ACTION_INPUTS]) -> [f64; MICRO_RESIDUAL_HIDDEN] {
        // Action features are sparse. Accumulate the sixteen independent dots
        // together to skip zero inputs once and expose contiguous CPU SIMD.
        // Input order within each dot remains unchanged.
        let mut h = [0.0; MICRO_RESIDUAL_HIDDEN];
        for (value, column) in action.iter().zip(&self.weights.action_columns) {
            if *value != 0.0 {
                for (sum, weight) in h.iter_mut().zip(column) {
                    *sum += value * weight;
                }
            }
        }
        for (value, context) in h.iter_mut().zip(self.context) {
            *value = (context + *value).tanh();
        }
        h
    }
    pub fn logit(&self, action: &[f64; MICRO_ACTION_INPUTS]) -> f64 {
        self.weights.w(OB)
            + self
                .activations(action)
                .iter()
                .enumerate()
                .map(|(j, h)| h * self.weights.w(O + j))
                .sum::<f64>()
    }
    pub fn accumulate_gradient(
        &self,
        action: &[f64; MICRO_ACTION_INPUTS],
        hidden: &[f64; MICRO_HIDDEN],
        delta: f64,
        gradient: &mut [f64],
        dh: &mut [f64; MICRO_HIDDEN],
    ) {
        gradient[OB] += delta;
        for (j, u) in self.activations(action).iter().enumerate() {
            gradient[O + j] += delta * u;
            let d = delta * self.weights.w(O + j) * (1.0 - u * u);
            gradient[B + j] += d;
            for k in 0..MICRO_ACTION_INPUTS {
                gradient[A + j * MICRO_ACTION_INPUTS + k] += d * action[k];
            }
            for k in 0..MICRO_HIDDEN {
                gradient[H + j * MICRO_HIDDEN + k] += d * hidden[k];
                dh[k] += d * self.weights.w(H + j * MICRO_HIDDEN + k);
            }
        }
    }
}
