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
    local_columns: Vec<[f64; MICRO_RESIDUAL_HIDDEN]>,
}
#[derive(Clone, Debug)]
pub(super) struct ResidualEmbedding {
    pub weights: Arc<ResidualWeights>,
    context: [f64; MICRO_RESIDUAL_HIDDEN],
    board: Vec<f64>,
}
impl MicroModel {
    pub fn schema(&self) -> &'static str {
        if self.has_relational() {
            MICRO_RELATIONAL_MODEL_SCHEMA
        } else if self.has_neural_memory() {
            MICRO_NEURAL_MEMORY_MODEL_SCHEMA
        } else if self.has_deep_value() {
            MICRO_DEEP_VALUE_MODEL_SCHEMA
        } else if self.has_spatial() {
            MICRO_SPATIAL_MODEL_SCHEMA
        } else if self.parameters.len() == MICRO_MEMORY_PARAMETERS {
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
        let mut parameters = self.parameters.as_ref().clone();
        parameters.resize(MICRO_RESIDUAL_PARAMETERS, 0.0);
        let mut rng = StableRng::new(seed);
        for w in &mut parameters[A..B] {
            *w = (rng.next_f64() * 2.0 - 1.0) * (6.0_f64 / 80.0).sqrt();
        }
        Self::from_parameters(parameters).expect("finite deterministic upgrade")
    }
}
impl ResidualWeights {
    pub(super) fn placeholder(self: &Arc<Self>, board: &[f64]) -> ResidualEmbedding {
        ResidualEmbedding {weights:self.clone(),context:[0.;MICRO_RESIDUAL_HIDDEN],board:board.to_vec()}
    }
    pub fn from_parameters(w: &[f64]) -> Option<Self> {
        (w.len() >= MICRO_RESIDUAL_PARAMETERS).then(|| Self {
            parameters: w[A..MICRO_RESIDUAL_PARAMETERS].to_vec(),
            local_columns: w.get(MICRO_SPATIAL_LOCAL..MICRO_VALUE_TRUNK).map_or_else(Vec::new, |local| {
                (0..18).map(|k|std::array::from_fn(|j|local[j*18+k])).collect()
            }),
            action_columns: std::array::from_fn(|k| {
                std::array::from_fn(|j| w[A + j * MICRO_ACTION_INPUTS + k])
            }),
            active: w[O..MICRO_RESIDUAL_PARAMETERS].iter().any(|w| *w != 0.0),
        })
    }
    fn w(&self, index: usize) -> f64 {
        self.parameters[index - A]
    }
    pub fn embed(self: &Arc<Self>, hidden: &[f64; MICRO_HIDDEN], board: &[f64]) -> ResidualEmbedding {
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
            board: board.to_vec(),
        }
    }
}
impl ResidualEmbedding {
    pub fn bytes(&self)->usize {self.board.capacity()*8}
    pub(super) fn activations(&self, action: &[f64; MICRO_ACTION_INPUTS]) -> [f64; MICRO_RESIDUAL_HIDDEN] {
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
        if !self.weights.local_columns.is_empty() && self.board.len() == MICRO_BOARD_INPUTS {
            let local = micro_local_features(&self.board, action);
            // Sixteen independent dots expose SIMD across units. Keep every
            // product, the original input order and f64::sum's -0 identity.
            let mut dots = [-0.0; MICRO_RESIDUAL_HIDDEN];
            for (x, column) in local.iter().zip(&self.weights.local_columns) {
                for (sum, w) in dots.iter_mut().zip(column) { *sum += x*w; }
            }
            for (value, dot) in h.iter_mut().zip(dots) { *value += dot; }
        }
        for (value, context) in h.iter_mut().zip(self.context) {
            *value = (context + *value).tanh();
        }
        h
    }
    pub fn logit(&self, action: &[f64; MICRO_ACTION_INPUTS]) -> f64 {
        self.logit_from_activations(&self.activations(action))
    }
    pub(super) fn logit_from_activations(&self, activations: &[f64; MICRO_RESIDUAL_HIDDEN]) -> f64 {
        self.weights.w(OB)
            + activations
                .iter()
                .enumerate()
                .map(|(j, h)| h * self.weights.w(O + j))
                .sum::<f64>()
    }
    pub fn accumulate_gradient(
        &self,
        action: &[f64; MICRO_ACTION_INPUTS],
        activations: &[f64; MICRO_RESIDUAL_HIDDEN],
        hidden: &[f64; MICRO_HIDDEN],
        delta: f64,
        gradient: &mut [f64],
        dh: &mut [f64; MICRO_HIDDEN],
    ) {
        gradient[OB] += delta;
        let local = (!self.weights.local_columns.is_empty() && self.board.len() == MICRO_BOARD_INPUTS)
            .then(|| micro_local_features(&self.board, action));
        for (j, u) in activations.iter().enumerate() {
            gradient[O + j] += delta * u;
            let d = delta * self.weights.w(O + j) * (1.0 - u * u);
            gradient[B + j] += d;
            if let Some(local) = &local {
                for (k, x) in local.iter().enumerate() {
                    gradient[MICRO_SPATIAL_LOCAL + j * 18 + k] += d * x;
                }
            }
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

#[cfg(test)]
mod exact_tests {
    use super::*;
    #[test]
    fn transposed_local_activations_match_scalar_reference_bits() {
        let mut rng = StableRng::new(34853);
        let weights: Vec<_> = (0..MICRO_SPATIAL_PARAMETERS)
            .map(|_|(rng.next_f64()*2.-1.)*0.03).collect();
        let model = MicroModel::from_parameters(weights.clone()).unwrap();
        let hidden = std::array::from_fn(|_|rng.next_f64()*2.-1.);
        let board: Vec<_> = (0..MICRO_BOARD_INPUTS).map(|i| {
            if i%3==0 { -0. } else { (rng.index(25) as f64-12.)/12. }
        }).collect();
        let embedded = model.residual.as_ref().unwrap().embed(&hidden,&board);
        for _ in 0..128 {
            let mut action = std::array::from_fn(|_|rng.next_f64()*2.-1.);
            action[2] = if rng.index(2)==0 { 0. } else { 1. };
            action[26] = if rng.index(2)==0 { 0. } else { 1. };
            let local = micro_local_features(&board,&action);
            let expected: [f64;16] = std::array::from_fn(|j| {
                let mut dot = 0.;
                for (k,x) in action.iter().enumerate() {
                    if *x!=0. { dot += x*weights[A+j*MICRO_ACTION_INPUTS+k]; }
                }
                dot += local.iter().zip(&weights[MICRO_SPATIAL_LOCAL+j*18..MICRO_SPATIAL_LOCAL+(j+1)*18])
                    .map(|(x,w)|x*w).sum::<f64>();
                let context = weights[B+j] + hidden.iter().enumerate()
                    .map(|(k,h)|h*weights[H+j*MICRO_HIDDEN+k]).sum::<f64>();
                (context+dot).tanh()
            });
            assert!(embedded.activations(&action).iter().zip(expected)
                .all(|(a,b)|a.to_bits()==b.to_bits()));
        }
    }
}
