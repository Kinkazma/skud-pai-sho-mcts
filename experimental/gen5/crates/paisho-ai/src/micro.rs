//! Small CPU value/policy model; independent schema from the frozen Gen1–3 agents.
mod features;
mod memory;
pub use memory::*;
mod residual;
mod search;
mod training;
mod spatial;
pub use spatial::*;
mod relations;
pub use relations::*;
mod piece_graph;
pub use piece_graph::*;
mod piece_graph_encoder;
pub use piece_graph_encoder::*;
mod structured;
pub use structured::*;
mod relational;
pub use relational::{MICRO_RELATIONAL_START,MICRO_RELATIONAL_WEIGHTS,MICRO_RELATIONAL_PARAMETERS,MICRO_RELATIONAL_MODEL_SCHEMA,micro_relational_value_parameter};
mod deep_value;
pub use deep_value::{MICRO_DEEP_VALUE_START, MICRO_DEEP_VALUE_PARAMETERS, MICRO_DEEP_VALUE_MODEL_SCHEMA};
mod neural_memory;
pub use neural_memory::{MICRO_NEURAL_MEMORY_START, MICRO_NEURAL_MEMORY_WEIGHTS, MICRO_NEURAL_MEMORY_PARAMETERS, MICRO_NEURAL_MEMORY_MODEL_SCHEMA};
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
    parameters: std::sync::Arc<Vec<f64>>,
    residual: Option<std::sync::Arc<residual::ResidualWeights>>,
    memory: Option<std::sync::Arc<crate::SequenceBank>>,
    deep_value_active: bool,
    deep_value_weights: Option<std::sync::Arc<deep_value::DeepValueWeights>>,
    neural_backward_weights: Option<std::sync::Arc<neural_memory::BackwardWeights>>,
    relational: Option<std::sync::Arc<relational::Weights>>,
}

/// Cached node inference. Policy context is computed once, each action costs a
/// 32-component dot product plus the optional nonlinear residual. Value is
/// ALWAYS from the position's player to move.
#[derive(Clone, Debug)]
pub struct MicroEmbedding {
    pub hidden: [f64; MICRO_HIDDEN],
    pub policy_context: [f64; MICRO_ACTION_INPUTS],
    pub value: f64,
    value_hidden: Option<[f64; MICRO_HIDDEN]>,
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
            parameters: std::sync::Arc::new(parameters),
            residual: None,
            memory: None,
            deep_value_active: false,
            deep_value_weights: None,
            neural_backward_weights: None,
            relational: None,
        }
    }
    pub fn from_parameters(parameters: Vec<f64>) -> Result<Self, String> {
        if ![MICRO_PARAMETERS, MICRO_RESIDUAL_PARAMETERS, MICRO_MEMORY_PARAMETERS, MICRO_SPATIAL_PARAMETERS, MICRO_DEEP_VALUE_PARAMETERS, MICRO_NEURAL_MEMORY_PARAMETERS,MICRO_RELATIONAL_PARAMETERS].contains(&parameters.len())
            || parameters.iter().any(|x| !x.is_finite())
        {
            return Err("micro model requires a supported finite parameter vector".into());
        }
        let residual =
            residual::ResidualWeights::from_parameters(&parameters).map(std::sync::Arc::new);
        Ok(Self {
            deep_value_active: deep_value::active(&parameters),
            deep_value_weights: deep_value::DeepValueWeights::from_parameters(&parameters).map(std::sync::Arc::new),
            neural_backward_weights: neural_memory::BackwardWeights::from_parameters(&parameters).map(std::sync::Arc::new),
            relational: relational::Weights::from_parameters(&parameters).map(std::sync::Arc::new),
            parameters: std::sync::Arc::new(parameters),
            residual,
            memory: None,
        })
    }
    /// Identity of an immutable snapshot, including its retrieval bank.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.parameters,&other.parameters)
            && match (&self.memory,&other.memory) {
                (None,None)=>true,
                (Some(a),Some(b))=>std::sync::Arc::ptr_eq(a,b),
                _=>false,
            }
    }
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    pub fn embed(&self, x: &[f64]) -> MicroEmbedding {
        self.embed_with_deep(x,None)
    }
    /// Diagnostic read of the policy used at internal search nodes. Reuse the
    /// exact search policy embedding without a value or root-reader forward.
    /// The existing residual16/spatial action head is part of this base policy.
    #[doc(hidden)]
    pub fn diagnostic_interior_policy_logits(&self, x: &[f64], actions: &[[f64; MICRO_ACTION_INPUTS]]) -> Vec<f64> {
        let placeholder=if self.has_spatial() {MicroEmbedding {
            hidden:[0.;MICRO_HIDDEN],policy_context:[0.;MICRO_ACTION_INPUTS],
            value:0.,value_hidden:None,residual:None,
        }} else {self.embed(x)};
        Self::logits(&self.embed_policy_for_search(x,&placeholder),actions)
    }
    /// Search-only placeholder. Keep the eager allocation/accounting shape, but
    /// defer policy arithmetic until that node's policy is actually requested.
    fn embed_value_for_search(&self, x: &[f64]) -> MicroEmbedding {
        if !self.has_spatial() { return self.embed(x); }
        let occupied = spatial::OccupiedFeatures::new(x);
        let value_hidden = self.value_hidden(x, &occupied);
        let mut raw = self.parameters[VALUE_B] + value_hidden.iter()
            .zip(&self.parameters[VALUE_W..VALUE_B]).map(|(a,b)|a*b).sum::<f64>();
        if self.has_deep_value() && self.deep_value_active {
            raw += self.deep_value_forward(x,&occupied,&mut deep_value::DeepValueActivations::default());
        }
        let relation=self.relational_value(x);if relation!=0. {raw+=relation;}
        MicroEmbedding {
            hidden:[0.;MICRO_HIDDEN],policy_context:[0.;MICRO_ACTION_INPUTS],
            value:raw.tanh(),value_hidden:Some(value_hidden),
            residual:self.residual.as_ref().map(|w|w.placeholder(x.get(MICRO_INPUTS..).unwrap_or(&[]))),
        }
    }
    /// Complete a search placeholder without reevaluating its independent value.
    fn embed_policy_for_search(&self, x: &[f64], value: &MicroEmbedding) -> MicroEmbedding {
        if !self.has_spatial() { return value.clone(); }
        let w=&self.parameters;
        let occupied=spatial::OccupiedFeatures::new(x);
        let mut h=[0.;MICRO_HIDDEN];
        for (j,output) in h.iter_mut().enumerate() {
            *output=(w[TRUNK_B+j]+x[..MICRO_INPUTS].iter()
                .zip(&w[j*MICRO_INPUTS..(j+1)*MICRO_INPUTS]).map(|(a,b)|a*b).sum::<f64>()
                +self.spatial_trunk(x,j,&occupied)).tanh();
        }
        let mut context=[0.;MICRO_ACTION_INPUTS];
        for (k,output) in context.iter_mut().enumerate() {
            *output=w[POLICY_B+k]+h.iter().enumerate().map(|(j,h)|h*w[POLICY_W+k*MICRO_HIDDEN+j]).sum::<f64>();
        }
        MicroEmbedding {hidden:h,policy_context:context,value:value.value,value_hidden:value.value_hidden,
            residual:self.residual.as_ref().map(|w|w.embed(&h,x.get(MICRO_INPUTS..).unwrap_or(&[])))}
    }
    /// Evaluate the independent value trunk without constructing a policy embedding.
    /// Legacy shared-trunk models retain their original path and arithmetic.
    pub fn value(&self, x: &[f64]) -> f64 {
        if !self.has_spatial() { return self.embed(x).value; }
        let occupied = spatial::OccupiedFeatures::new(x);
        let h = self.value_hidden(x, &occupied);
        let mut raw = self.parameters[VALUE_B] + h.iter()
            .zip(&self.parameters[VALUE_W..VALUE_B])
            .map(|(a,b)|a*b).sum::<f64>();
        if self.has_deep_value() && self.deep_value_active {
            raw += self.deep_value_forward(x,&occupied,&mut deep_value::DeepValueActivations::default());
        }
        let relation=self.relational_value(x);if relation!=0. {raw+=relation;}
        raw.tanh()
    }
    fn embed_with_deep(&self, x: &[f64], deep: Option<&mut deep_value::DeepValueActivations>) -> MicroEmbedding {
        let w = &self.parameters;
        let occupied = spatial::OccupiedFeatures::new(x);
        let mut h = [0.0; MICRO_HIDDEN];
        for (j, output) in h.iter_mut().enumerate() {
            *output = (w[TRUNK_B + j]
                + x[..MICRO_INPUTS].iter()
                    .zip(&w[j * MICRO_INPUTS..(j + 1) * MICRO_INPUTS])
                    .map(|(a, b)| a * b)
                    .sum::<f64>()
                + self.spatial_trunk(x, j, &occupied))
            .tanh();
        }
        let value_hidden = self.has_spatial().then(||self.value_hidden(x, &occupied));
        let mut raw_value = w[VALUE_B]
            + value_hidden.as_ref().unwrap_or(&h).iter()
                .zip(&w[VALUE_W..VALUE_B])
                .map(|(a, b)| a * b)
                .sum::<f64>();
        if self.has_deep_value() {
            if let Some(activations)=deep {
                let correction=self.deep_value_forward(x,&occupied,activations);
                if self.deep_value_active {raw_value+=correction;}
            } else if self.deep_value_active {
                raw_value+=self.deep_value_forward(x,&occupied,&mut deep_value::DeepValueActivations::default());
            }
        }
        let relation=self.relational_value(x);if relation!=0. {raw_value+=relation;}
        let value=raw_value.tanh();
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
            value_hidden,
            residual: self.residual.as_ref().map(|w| w.embed(&h, x.get(MICRO_INPUTS..).unwrap_or(&[]))),
        }
    }
    pub fn logits(embedding: &MicroEmbedding, actions: &[[f64; MICRO_ACTION_INPUTS]]) -> Vec<f64> {
        actions.iter().map(|a| Self::logit(embedding, a)).collect()
    }
    pub fn logit(embedding: &MicroEmbedding, action: &[f64; MICRO_ACTION_INPUTS]) -> f64 {
        let base = Self::base_logit(embedding, action);
        match &embedding.residual {
            Some(residual) if residual.weights.active => base + residual.logit(action),
            _ => base,
        }
    }
    fn base_logit(embedding: &MicroEmbedding, action: &[f64; MICRO_ACTION_INPUTS]) -> f64 {
        action.iter().zip(embedding.policy_context).map(|(a,b)| a*b).sum()
    }
}

pub fn micro_softmax(logits: &[f64]) -> Result<Vec<f64>, String> {
    micro_softmax_parts(logits).map(|(probabilities, _, _)| probabilities)
}
/// Forward probabilities and their normalizer share the same exponentials.
fn micro_softmax_parts(logits: &[f64]) -> Result<(Vec<f64>, f64, f64), String> {
    if logits.is_empty() || logits.iter().any(|x| !x.is_finite()) {
        return Err("policy requires finite legal logits".into());
    }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<_> = logits.iter().map(|x| (x - max).exp()).collect();
    let total: f64 = p.iter().sum();
    for x in &mut p {
        *x /= total;
    }
    Ok((p, max, total))
}
