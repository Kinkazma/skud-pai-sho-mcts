//! A bounded, learned reader consulted at real roots, not at every simulated leaf.
use super::*;
use crate::{SequenceBank, SequenceContext, SEQUENCE_CHANNELS};
use std::sync::Arc;
pub const MICRO_MEMORY_PARAMETERS: usize = MICRO_RESIDUAL_PARAMETERS + SEQUENCE_CHANNELS;
pub const MICRO_MEMORY_MODEL_SCHEMA: &str = "paisho-micro-value-policy-sequence-v3";
impl MicroModel {
    pub fn with_sequence_memory(&self, bank: Arc<SequenceBank>) -> Self {
        let mut model = self.with_residual_policy(0);
        model.parameters.resize(MICRO_MEMORY_PARAMETERS, 0.0);
        model.memory = Some(bank);
        model
    }
    pub fn sequence_memory(&self) -> Option<&Arc<SequenceBank>> {
        self.memory.as_ref()
    }
    pub fn memory_context(
        &self,
        state: &[f64; 128],
        excluded: u64,
    ) -> Option<Arc<SequenceContext>> {
        self.memory.as_ref().map(|b| b.context(state, excluded))
    }
    pub(super) fn memory_vector(&self, context: &SequenceContext) -> [f64; 32] {
        let mut vector = [0.; 32];
        for (pattern, w) in context
            .patterns
            .iter()
            .zip(&self.parameters[MICRO_RESIDUAL_PARAMETERS..])
        {
            for k in 0..32 {
                vector[k] += pattern[k] * w / 8.;
            }
        }
        vector
    }
    pub(super) fn memory_action(vector: &[f64; 32], action: &[f64; 32]) -> (f64, f64) {
        let t = (vector.iter().zip(action).map(|(a, b)| a * b).sum::<f64>() / 2.).tanh();
        (2. * t, 1. - t * t)
    }
    /// Preserve exact base priors with zero-initialized reader weights. Multiplying
    /// cached probabilities avoids recomputing the nonlinear base action head.
    pub fn memory_priors(
        &self,
        state: &[f64; 128],
        actions: &[[f64; 32]],
        base: &[f64],
        excluded: u64,
    ) -> Result<Vec<f64>, String> {
        if actions.len() != base.len() {
            return Err("memory action alignment mismatch".into());
        }
        let Some(context) = self.memory_context(state, excluded) else {
            return Ok(base.to_vec());
        };
        if self.parameters[MICRO_RESIDUAL_PARAMETERS..]
            .iter()
            .all(|w| *w == 0.)
        {
            return Ok(base.to_vec());
        }
        let vector = self.memory_vector(&context);
        let mut p: Vec<_> = actions
            .iter()
            .zip(base)
            .map(|(a, p)| p * Self::memory_action(&vector, a).0.exp())
            .collect();
        let total: f64 = p.iter().sum();
        if !total.is_finite() || total <= 0. {
            return Err("invalid memory probabilities".into());
        }
        for x in &mut p {
            *x /= total;
        }
        Ok(p)
    }
}
