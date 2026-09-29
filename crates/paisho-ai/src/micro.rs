//! Small CPU value/policy model; independent schema from the frozen Gen1–3 agents.
mod features;
mod memory;
pub use memory::*;
mod residual;
mod search;
mod training;
use crate::StableRng;
pub use features::*;
pub use residual::{MICRO_RESIDUAL_HIDDEN, MICRO_RESIDUAL_MODEL_SCHEMA, MICRO_RESIDUAL_PARAMETERS};
pub use search::*;
pub use training::*;

pub const MICRO_HIDDEN: usize = 32;
pub const MICRO_PARAMETERS: usize = 5217;
pub const MICRO_MODEL_SCHEMA: &str = "paisho-micro-value-policy-128-32-v1";
const TRUNK_B: usize = 4096;
const VALUE_W: usize = 4128;
const VALUE_B: usize = 4160;
const POLICY_W: usize = 4161;
const POLICY_B: usize = 5185;

#[derive(Clone, Debug, PartialEq)]
pub struct MicroModel {
    parameters: Vec<f64>,
    residual: Option<std::sync::Arc<residual::ResidualWeights>>,
    memory: Option<std::sync::Arc<crate::SequenceBank>>,
}

/// Cached node inference. Policy context is computed once, each action costs a
/// 32-component dot product plus the optional nonlinear residual. Value is
/// ALWAYS from the position's player to move.
#[derive(Clone, Debug)]
pub struct MicroEmbedding {
    pub hidden: [f64; MICRO_HIDDEN],
    pub policy_context: [f64; MICRO_ACTION_INPUTS],
    pub value: f64,
    residual: Option<residual::ResidualEmbedding>,
}

impl MicroModel {
    pub fn seeded(seed: u64) -> Self {
        let mut rng = StableRng::new(seed);
        let mut parameters = vec![0.0; MICRO_PARAMETERS];
        for p in &mut parameters[..TRUNK_B] {
            *p = (rng.next_f64() * 2.0 - 1.0) * (6.0_f64 / 160.0).sqrt();
        }
        for p in &mut parameters[VALUE_W..VALUE_B] {
            *p = (rng.next_f64() * 2.0 - 1.0) * 0.1;
        }
        for p in &mut parameters[POLICY_W..POLICY_B] {
            *p = (rng.next_f64() * 2.0 - 1.0) * 0.1;
        }
        Self {
            parameters,
            residual: None,
            memory: None,
        }
    }
    pub fn from_parameters(parameters: Vec<f64>) -> Result<Self, String> {
        if ![MICRO_PARAMETERS, MICRO_RESIDUAL_PARAMETERS, MICRO_MEMORY_PARAMETERS].contains(&parameters.len())
            || parameters.iter().any(|x| !x.is_finite())
        {
            return Err("micro model requires 5217, 6274 or 6286 finite parameters".into());
        }
        let residual =
            residual::ResidualWeights::from_parameters(&parameters).map(std::sync::Arc::new);
        Ok(Self {
            parameters,
            residual,
            memory: None,
        })
    }
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    pub fn embed(&self, x: &[f64; MICRO_INPUTS]) -> MicroEmbedding {
        let w = &self.parameters;
        let mut h = [0.0; MICRO_HIDDEN];
        for (j, output) in h.iter_mut().enumerate() {
            *output = (w[TRUNK_B + j]
                + x.iter()
                    .zip(&w[j * MICRO_INPUTS..(j + 1) * MICRO_INPUTS])
                    .map(|(a, b)| a * b)
                    .sum::<f64>())
            .tanh();
        }
        let value = (w[VALUE_B]
            + h.iter()
                .zip(&w[VALUE_W..VALUE_B])
                .map(|(a, b)| a * b)
                .sum::<f64>())
        .tanh();
        let mut context = [0.0; MICRO_ACTION_INPUTS];
        for (k, output) in context.iter_mut().enumerate() {
            *output = w[POLICY_B + k]
                + h.iter()
                    .enumerate()
                    .map(|(j, h)| h * w[POLICY_W + k * MICRO_HIDDEN + j])
                    .sum::<f64>();
        }
        MicroEmbedding {
            hidden: h,
            policy_context: context,
            value,
            residual: self.residual.as_ref().map(|w| w.embed(&h)),
        }
    }
    pub fn logits(embedding: &MicroEmbedding, actions: &[[f64; MICRO_ACTION_INPUTS]]) -> Vec<f64> {
        actions.iter().map(|a| Self::logit(embedding, a)).collect()
    }
    pub fn logit(embedding: &MicroEmbedding, action: &[f64; MICRO_ACTION_INPUTS]) -> f64 {
        let base = action
            .iter()
            .zip(embedding.policy_context)
            .map(|(a, b)| a * b)
            .sum::<f64>();
        match &embedding.residual {
            Some(residual) if residual.weights.active => base + residual.logit(action),
            _ => base,
        }
    }
}

pub fn micro_softmax(logits: &[f64]) -> Result<Vec<f64>, String> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("policy requires finite legal logits".into());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let total: f64 = p.iter().sum();
    for x in &mut p {
        *x /= total;
    }
    Ok(p)
}
