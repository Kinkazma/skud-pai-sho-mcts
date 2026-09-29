//! A bounded, learned reader consulted at real roots, not at every simulated leaf.
use super::*;
use crate::{SequenceBank, SequenceContext, SEQUENCE_CHANNELS};
use std::sync::Arc;
pub const MICRO_MEMORY_PARAMETERS: usize = MICRO_RESIDUAL_PARAMETERS + SEQUENCE_CHANNELS;
pub const MICRO_MEMORY_MODEL_SCHEMA: &str = "paisho-micro-value-policy-sequence-v3";
impl MicroModel {
    pub fn with_sequence_memory(&self, bank: Arc<SequenceBank>) -> Self {
        let mut model = self.with_residual_policy(0);
        if model.parameters.len() < MICRO_MEMORY_PARAMETERS {
            Arc::make_mut(&mut model.parameters).resize(MICRO_MEMORY_PARAMETERS, 0.0);
        }
        model.memory = Some(bank);
        model
    }
    /// Attach the same immutable bank while retaining owned parameter storage.
    /// A newly updated model needs no second clone of its entire weight vector.
    pub fn with_sequence_memory_owned(mut self, bank: Arc<SequenceBank>) -> Self {
        if self.residual.is_none() { self=self.with_residual_policy(0); }
        if self.parameters.len()<MICRO_MEMORY_PARAMETERS { Arc::make_mut(&mut self.parameters).resize(MICRO_MEMORY_PARAMETERS,0.); }
        self.memory=Some(bank);
        self
    }
    pub fn sequence_memory(&self) -> Option<&Arc<SequenceBank>> {
        self.memory.as_ref()
    }
    pub fn memory_context(
        &self,
        state: &[f64],
        excluded: u64,
    ) -> Result<Option<Arc<SequenceContext>>, String> {
        if excluded == u64::MAX {
            return Ok(None);
        }
        self.memory
            .as_ref()
            .map(|b| b.try_context(state, excluded))
            .transpose()
    }
    pub(super) fn memory_vector(&self, context: &SequenceContext) -> [f64; 32] {
        let mut vector = [0.; 32];
        for (pattern, w) in context
            .patterns
            .iter()
            .zip(&self.parameters[MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS])
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
        state: &[f64],
        actions: &[[f64; 32]],
        base: &[f64],
        excluded: u64,
    ) -> Result<Vec<f64>, String> {
        self.memory_priors_cached(state,actions,base,excluded).map(|(p,_)|p)
    }
    /// Read a root's value and memory policy with one embedding and logit read.
    /// The neural reader receives the very same inputs as `memory_priors`;
    /// no sums, softmax operations, or memory exclusions are regrouped.
    pub fn policy_value_priors(
        &self, state: &[f64], actions: &[[f64; 32]], excluded: u64,
    ) -> Result<(f64, Vec<f64>), String> {
        let embedding = self.embed(state);
        let logits = Self::logits(&embedding, actions);
        let base = micro_softmax(&logits)?;
        let old = self.sequence_priors(state, actions, &base, excluded)?;
        if !self.neural_memory_active() || actions.is_empty() {
            return Ok((embedding.value, old));
        }
        let vector = self.memory_context(state, excluded)?.as_ref()
            .map(|c| self.memory_vector(c)).unwrap_or([0.; 32]);
        let cache = self.neural_forward(state, actions, &vector, embedding.value, &logits);
        let p = micro_softmax(&old.iter().zip(cache.output.chunks_exact(2))
            .map(|(p, y)| p.max(1e-300).ln() + y[0]).collect::<Vec<_>>())?;
        Ok((embedding.value, p))
    }
    pub(super) fn memory_priors_cached(
        &self, state:&[f64], actions:&[[f64;32]], base:&[f64], excluded:u64,
    )->Result<(Vec<f64>,Option<neural_memory::Cache>),String> {
        if actions.len() != base.len() {
            return Err("memory action alignment mismatch".into());
        }
        let old = self.sequence_priors(state, actions, base, excluded)?;
        if !self.neural_memory_active() || actions.is_empty() { return Ok((old,None)); }
        let embedding = self.embed(state);
        let vector = self.memory_context(state, excluded)?.as_ref().map(|c|self.memory_vector(c)).unwrap_or([0.;32]);
        let cache = self.neural_forward(state, actions, &vector, embedding.value, &Self::logits(&embedding, actions));
        let p=micro_softmax(&old.iter().zip(cache.output.chunks_exact(2)).map(|(p,y)|p.max(1e-300).ln()+y[0]).collect::<Vec<_>>())?;
        Ok((p,Some(cache)))
    }
    fn sequence_priors(&self,state:&[f64],actions:&[[f64;32]],base:&[f64],excluded:u64)->Result<Vec<f64>,String> {
        let Some(context) = self.memory_context(state, excluded)? else {
            return Ok(base.to_vec());
        };
        if self.parameters[MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS]
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
