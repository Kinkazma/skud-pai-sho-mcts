//! Gen3 lineage: exact compact value and retained UCT, plus a learned bounded
//! progressive action bias. A neutral policy preserves the Gen3.1 search.
mod guard;
use crate::*;
pub use guard::{gen32_tactical_guard, Gen32GuardReport};
use paisho_core::{Action, Player, Position};
use rayon::prelude::*;
use std::sync::Arc;
mod extension;
pub use extension::Gen3MemoryScope;
mod value_residual;
pub use value_residual::*;

#[derive(Clone, Debug)]
pub struct Gen32Model {
    pub value_residual: Option<Arc<Gen3ValueResidual>>,
    pub value_extra: Option<[f64; 64]>,
    pub memory_scope: Gen3MemoryScope,
    pub value: CompactValueModel,
    pub policy: MicroModel,
}
impl Gen32Model {
    pub fn from_gen31(value: CompactValueModel, seed: u64) -> Self {
        let mut parameters = MicroModel::seeded(seed).parameters().to_vec();
        // Keep the feature trunk; no random action preference at the bridge.
        parameters[4161..].fill(0.0);
        let policy = MicroModel::from_parameters(parameters)
            .unwrap()
            .with_residual_policy(seed);
        Self {
            value_residual: None,
            value,
            policy,
            value_extra: None,
            memory_scope: Gen3MemoryScope::Root,
        }
    }
    pub fn with_memory(mut self, bank: Arc<SequenceBank>) -> Self {
        self.policy = self.policy.with_sequence_memory(bank);
        self
    }
    pub fn train(&mut self, example: &MicroExample, rate: f64) -> Result<(), String> {
        example.validate()?;
        let (value, extra, residual) = self.trained_value(example, rate)?;
        // The policy operation is atomic; publish the small compact head only
        // after it succeeds, without cloning legal-action arrays or model twice.
        if example.policy_weight > 0.0 {
            self.policy.train_policy_step(example, rate)?;
        }
        self.value = value;
        self.value_extra = extra;
        self.value_residual = residual;
        Ok(())
    }
}
impl MctsEvaluator for Gen32Model {
    fn ordering_matches_leaf(&self) -> bool {
        true
    }
    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        if self.value_extra.is_none() && self.value_residual.is_none() {
            return MctsEvaluator::evaluate(&self.value, positions, perspective, weights);
        }
        let f = |p: &Position| self.value_at(p, perspective);
        Ok(if positions.len() >= 64 {
            positions.par_iter().map(f).collect()
        } else {
            positions.iter().map(f).collect()
        })
    }
    fn evaluate_leaf(
        &self,
        position: &Position,
        perspective: Player,
        _: HeuristicWeights,
    ) -> Result<f32, String> {
        Ok(self.value_at(position, perspective))
    }
    fn policy_bias(&self, p: &Position, a: &[Action]) -> Result<Option<Vec<f64>>, String> {
        self.bias(p, a, self.memory_scope.uses_memory(p))
    }
    fn root_policy_bias(&self, p: &Position, a: &[Action]) -> Result<Option<Vec<f64>>, String> {
        if self.policy.sequence_memory().is_none()
            || self.policy.parameters()[MICRO_RESIDUAL_PARAMETERS..]
                .iter()
                .all(|w| *w == 0.)
        {
            return Ok(None);
        }
        self.bias(p, a, true)
    }
}
impl Gen32Model {
    fn bias(
        &self,
        position: &Position,
        actions: &[Action],
        memory: bool,
    ) -> Result<Option<Vec<f64>>, String> {
        if actions.is_empty() {
            return Ok(None);
        }
        let state = micro_state_features(position);
        let embedding = self.policy.embed(&state);
        let score = |a: &Action| {
            let f = micro_action_features(position, *a);
            (f, MicroModel::logit(&embedding, &f))
        };
        let rows: Vec<_> = if actions.len() >= 64 {
            actions.par_iter().map(score).collect()
        } else {
            actions.iter().map(score).collect()
        };
        let (features, logits): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        let base = micro_softmax(&logits)?;
        let probabilities = if memory {
            self.policy.memory_priors(&state, &features, &base, 0)?
        } else {
            base
        };
        // Center log probabilities: uniform policy means exact zero even with
        // a varying legal-action count. Bounded influence decays with visits.
        let logs: Vec<_> = probabilities.iter().map(|p| p.max(1e-300).ln()).collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        if logs.iter().all(|p| *p == logs[0]) {
            return Ok(None);
        }
        Ok(Some(
            logs.iter().map(|p| ((p - mean) / 2.0).tanh()).collect(),
        ))
    }
}
