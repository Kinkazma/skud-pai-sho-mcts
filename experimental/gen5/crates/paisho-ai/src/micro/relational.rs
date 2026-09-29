//! V7 neutral action-consequence branch. Read at actual roots; value shares graph.
use super::*;
use std::sync::Arc;
mod backward;
#[cfg(test)]
mod tests;
pub(super) const H: usize = 64;
pub(super) const INPUT: usize = 497;
pub(super) const OUTPUT: usize = 21;
pub(super) const HW: usize = MICRO_GRAPH_PARAMETERS;
pub(super) const HB: usize = HW + INPUT * H;
pub(super) const OW: usize = HB + H;
pub(super) const OB: usize = OW + H * OUTPUT;
pub(super) const QUERY: usize = OB + OUTPUT;
pub(super) const VALUE: usize = QUERY + 32 * 16;
pub const MICRO_RELATIONAL_START: usize = MICRO_NEURAL_MEMORY_PARAMETERS;
pub const MICRO_RELATIONAL_WEIGHTS: usize = VALUE + 33;
pub const MICRO_RELATIONAL_PARAMETERS: usize = MICRO_RELATIONAL_START + MICRO_RELATIONAL_WEIGHTS;
pub const MICRO_RELATIONAL_MODEL_SCHEMA: &str = "paisho-micro-relational-consequences-v7";

/// Shared graph parameters affect value too; a value-frozen publication relay
/// must freeze them, as well as the direct graph-value readout.
pub fn micro_relational_value_parameter(index: usize) -> bool {
    (MICRO_RELATIONAL_START..MICRO_RELATIONAL_START + MICRO_GRAPH_PARAMETERS).contains(&index)
        || (MICRO_RELATIONAL_START + VALUE..MICRO_RELATIONAL_PARAMETERS).contains(&index)
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Weights {
    w: Arc<Vec<f64>>,
    encoder: MicroGraphEncoder,
    policy_active: bool,
    value_active: bool,
}
pub(super) struct PositionRead {
    graph: MicroPieceGraph,
    pool: [f64; 32],
    nodes: Vec<[f64; 16]>,
}
pub(super) struct Cache {
    position: PositionRead,
    hidden: Vec<[f64; H]>,
    context: Vec<[f64; 16]>,
    query: Vec<[f64; 16]>,
    attention: Vec<Vec<f64>>,
    pub output: Vec<[f64; OUTPUT]>,
}

impl Weights {
    pub(super) fn from_parameters(w: &[f64]) -> Option<Self> {
        (w.len() == MICRO_RELATIONAL_PARAMETERS).then(|| {
            let tail = &w[MICRO_RELATIONAL_START..];
            Self {
                w: Arc::new(tail.to_vec()),
                encoder: MicroGraphEncoder::from_parameters(tail[..HW].to_vec()).unwrap(),
                policy_active: tail[OB] != 0. || (0..H).any(|j| tail[OW + j * OUTPUT] != 0.),
                value_active: tail[VALUE..].iter().any(|x| *x != 0.),
            }
        })
    }
    fn position(&self, state: &[f64]) -> Result<PositionRead, String> {
        let graph = MicroPieceGraph::from_spatial_features(state)?;
        let (pool, nodes) = self
            .encoder
            .encode_all(&graph, MicroGraphPropagation::TwoHarmonyRounds);
        Ok(PositionRead { graph, pool, nodes })
    }
    pub(super) fn value(&self, state: &[f64]) -> f64 {
        if !self.value_active {
            return 0.;
        }
        let p = self
            .position(state)
            .expect("canonical relational board inputs");
        self.w[VALUE + 32]
            + p.pool
                .iter()
                .zip(&self.w[VALUE..VALUE + 32])
                .map(|(a, b)| a * b)
                .sum::<f64>()
    }
    pub(super) fn forward(&self, state: &[f64], actions: &[[f64; 32]]) -> Result<Cache, String> {
        self.forward_impl::<true>(state, actions)
    }
    /// Actual play needs the policy residual only. Keep exactly its summation
    /// order, but skip the twenty unused auxiliary heads and backward tapes.
    fn forward_impl<const TRAINING: bool>(
        &self,
        state: &[f64],
        actions: &[[f64; 32]],
    ) -> Result<Cache, String> {
        let p = self.position(state)?;
        let w = &self.w;
        let state_part: [f64; H] = std::array::from_fn(|j| {
            state
                .iter()
                .enumerate()
                .map(|(i, x)| x * w[HW + i * H + j])
                .sum()
        });
        let pool_part: [f64; H] = std::array::from_fn(|j| {
            p.pool
                .iter()
                .enumerate()
                .map(|(i, x)| x * w[HW + (449 + i) * H + j])
                .sum()
        });
        let mut result = Cache {
            position: p,
            hidden: vec![],
            context: vec![],
            query: vec![],
            attention: vec![],
            output: vec![],
        };
        for a in actions {
            let query: [f64; 16] = std::array::from_fn(|j| {
                a.iter()
                    .enumerate()
                    .map(|(i, x)| x * w[QUERY + i * 16 + j])
                    .sum()
            });
            let attention = if result.position.nodes.is_empty() {
                vec![]
            } else {
                micro_softmax(
                    &result
                        .position
                        .nodes
                        .iter()
                        .map(|h| query.iter().zip(h).map(|(a, b)| a * b).sum::<f64>() / 4.)
                        .collect::<Vec<_>>(),
                )?
            };
            let context: [f64; 16] = std::array::from_fn(|j| {
                attention
                    .iter()
                    .zip(&result.position.nodes)
                    .map(|(a, h)| a * h[j])
                    .sum()
            });
            let hidden: [f64; H] = std::array::from_fn(|j| {
                let action: f64 = a
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * w[HW + (417 + i) * H + j])
                    .sum();
                let c: f64 = context
                    .iter()
                    .enumerate()
                    .map(|(i, x)| x * w[HW + (481 + i) * H + j])
                    .sum();
                (state_part[j] + action + pool_part[j] + c + w[HB + j]).tanh()
            });
            result.output.push(std::array::from_fn(|k| {
                if TRAINING || k == 0 {
                    hidden
                        .iter()
                        .enumerate()
                        .map(|(j, h)| h * w[OW + j * OUTPUT + k])
                        .sum::<f64>()
                        + w[OB + k]
                } else {
                    0.
                }
            }));
            if TRAINING {
                result.hidden.push(hidden);
                result.context.push(context);
                result.query.push(query);
                result.attention.push(attention);
            }
        }
        Ok(result)
    }
}

impl MicroModel {
    pub fn has_relational(&self) -> bool {
        self.relational.is_some()
    }
    pub fn with_relational(&self, seed: u64) -> Self {
        if self.has_relational() {
            return self.clone();
        }
        let base = self.with_neural_memory(seed);
        let mut w = base.parameters.as_ref().clone();
        w.resize(MICRO_RELATIONAL_PARAMETERS, 0.);
        let tail = &mut w[MICRO_RELATIONAL_START..];
        tail[..HW].copy_from_slice(MicroGraphEncoder::seeded(seed).parameters());
        let mut rng = StableRng::new(seed);
        for (start, end, inputs, outputs) in [(HW, HB, INPUT, H), (QUERY, VALUE, 32, 16)] {
            let scale = (6. / (inputs + outputs) as f64).sqrt();
            for x in &mut tail[start..end] {
                *x = (2. * rng.next_f64() - 1.) * scale;
            }
        }
        let mut model = Self::from_parameters(w).expect("finite relational migration");
        model.memory = base.memory.clone();
        model
    }
    pub(super) fn relational_value(&self, state: &[f64]) -> f64 {
        self.relational.as_ref().map_or(0., |w| w.value(state))
    }
    pub(super) fn relational_priors(
        &self,
        state: &[f64],
        actions: &[[f64; 32]],
        base: Vec<f64>,
    ) -> Result<Vec<f64>, String> {
        let Some(w) = self.relational.as_ref().filter(|w| w.policy_active) else {
            return Ok(base);
        };
        if actions.is_empty() {
            return Ok(base);
        }
        let c = w.forward_impl::<false>(state, actions)?;
        micro_softmax(
            &base
                .iter()
                .zip(c.output)
                .map(|(p, y)| p.max(1e-300).ln() + y[0])
                .collect::<Vec<_>>(),
        )
    }
    /// Auxiliary predictions are estimates, never solver proofs.
    pub fn structured_predictions(
        &self,
        state: &[f64],
        actions: &[[f64; 32]],
    ) -> Result<Option<Vec<[f64; 20]>>, String> {
        self.relational
            .as_ref()
            .map(|w| {
                w.forward(state, actions).map(|c| {
                    c.output
                        .iter()
                        .map(|y| {
                            std::array::from_fn(|i| {
                                if i < 10 {
                                    y[i + 1]
                                } else {
                                    1. / (1. + (-y[i + 1]).exp())
                                }
                            })
                        })
                        .collect()
                })
            })
            .transpose()
    }
}
