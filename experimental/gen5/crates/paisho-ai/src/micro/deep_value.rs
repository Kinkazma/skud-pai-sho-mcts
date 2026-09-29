//! V5 adds a neutral 417→128→64→32→1 correction before the existing value tanh.
//! Activations live only on the stack/current learning example, never in MCTS nodes.
use super::*;
use super::spatial::OccupiedFeatures;
#[cfg(test)] mod tests;
#[cfg(test)] mod groups_bench;

pub const MICRO_DEEP_VALUE_START: usize = MICRO_SPATIAL_PARAMETERS;
const W1: usize = MICRO_DEEP_VALUE_START;
const B1: usize = W1 + MICRO_SPATIAL_INPUTS * 128;
const W2: usize = B1 + 128;
const B2: usize = W2 + 128 * 64;
const W3: usize = B2 + 64;
const B3: usize = W3 + 64 * 32;
const OUT: usize = B3 + 32;
const BOUT: usize = OUT + 32;
pub const MICRO_DEEP_VALUE_PARAMETERS: usize = BOUT + 1;
pub const MICRO_DEEP_VALUE_MODEL_SCHEMA: &str = "paisho-micro-spatial-deep-value128-64-32-v5";

const GROUP:usize=16;

/// Immutable groups of independent output neurons, contiguous for each input value.
/// The serialized parameter order stays unchanged; snapshots share these views.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct DeepValueWeights {
    first: Vec<[f64;GROUP]>,
    second: Vec<[f64;GROUP]>,
    third: Vec<[f64;GROUP]>,
}
impl DeepValueWeights {
    pub(super) fn from_parameters(w:&[f64])->Option<Self> {
        if w.len()<MICRO_DEEP_VALUE_PARAMETERS {return None;}
        let transpose=|start:usize,inputs:usize,outputs:usize| {
            (0..outputs).step_by(GROUP).flat_map(|j|(0..inputs).map(move |i| {
                std::array::from_fn(|lane|w[start+(j+lane)*inputs+i])
            })).collect()
        };
        Some(Self {first:transpose(W1,417,128),second:transpose(W2,128,64),third:transpose(W3,64,32)})
    }
}

pub(super) struct DeepValueActivations {
    first: [f64;128],
    second: [f64;64],
    third: [f64;32],
}
impl Default for DeepValueActivations {
    fn default()->Self {Self {first:[0.;128],second:[0.;64],third:[0.;32]}}
}
// Independent output neurons retain each scalar dot's input/addition order.
// This exposes parallel arithmetic without reassociating a floating-point sum.
#[inline]
fn dots<const N:usize>(x: &[f64], weights: &[[f64;N]]) -> [f64;N] {
    let mut sums = [-0.0;N];
    for (x, w) in x.iter().zip(weights) {
        for lane in 0..N { sums[lane] += x * w[lane]; }
    }
    sums
}
/// Zero products do not change a nonzero finite sum. Preserve the exceptional
/// zero/nonfinite case with the original full dot, including signed-zero bits.
fn sparse_dots<const N:usize>(x: &[f64], indices: &[u8], weights: &[[f64;N]]) -> [f64;N] {
    let mut sums = [-0.0;N];
    for &i in indices {
        let i = i as usize;
        for lane in 0..N { sums[lane] += x[i] * weights[i][lane]; }
    }
    for lane in 0..N {
        if sums[lane] == 0.0 || !sums[lane].is_finite() {
            sums[lane] = x.iter().zip(weights).map(|(v,w)|v*w[lane]).sum();
        }
    }
    sums
}
pub(super) fn active(parameters: &[f64])->bool {
    parameters.len()>=MICRO_DEEP_VALUE_PARAMETERS && parameters[OUT..MICRO_DEEP_VALUE_PARAMETERS].iter().any(|x|*x!=0.)
}
impl MicroModel {
    pub fn has_deep_value(&self)->bool {self.parameters.len()>=MICRO_DEEP_VALUE_PARAMETERS}

    /// Explicit, idempotent migration; preserve every existing weight and bank Arc.
    pub fn with_deep_value(&self, seed:u64)->Self {
        if self.has_deep_value() {return self.clone();}
        let base=self.with_spatial_policy();
        let mut weights=base.parameters.as_ref().clone();
        weights.resize(MICRO_DEEP_VALUE_PARAMETERS,0.);
        let mut rng=StableRng::new(seed);
        for (start,end,inputs,outputs) in [(W1,B1,417,128),(W2,B2,128,64),(W3,B3,64,32)] {
            let scale=(6.0/(inputs+outputs) as f64).sqrt();
            for w in &mut weights[start..end] {*w=(rng.next_f64()*2.-1.)*scale;}
        }
        let mut model=Self::from_parameters(weights).expect("finite deterministic deep-value migration");
        model.memory=base.memory.clone();model
    }

    pub(super) fn deep_value_forward(&self, state:&[f64], occupied:&OccupiedFeatures, a:&mut DeepValueActivations)->f64 {
        let w=&self.parameters;
        let packed=self.deep_value_weights.as_ref().expect("deep value weight views");
        let mut indices=[0u8;128];let mut occupied_dense=0;
        for (i,x) in state[..128].iter().enumerate() {
            if *x!=0.0 {indices[occupied_dense]=i as u8;occupied_dense+=1;}
        }
        for j in (0..128).step_by(GROUP) {
            let start=j/GROUP*MICRO_SPATIAL_INPUTS;
            let dense=sparse_dots(&state[..128], &indices[..occupied_dense], &packed.first[start..start+128]);
            let board=if state.len()==MICRO_SPATIAL_INPUTS {occupied.dots(state,&packed.first[start+128..start+417])} else {[0.;GROUP]};
            for lane in 0..GROUP {
                a.first[j+lane]=(w[B1+j+lane]+dense[lane]+board[lane]).tanh();
            }
        }
        for j in (0..64).step_by(GROUP) {
            let sums=dots(&a.first,&packed.second[j/GROUP*128..(j/GROUP+1)*128]);
            for lane in 0..GROUP {a.second[j+lane]=(w[B2+j+lane]+sums[lane]).tanh();}
        }
        for j in (0..32).step_by(GROUP) {
            let sums=dots(&a.second,&packed.third[j/GROUP*64..(j/GROUP+1)*64]);
            for lane in 0..GROUP {a.third[j+lane]=(w[B3+j+lane]+sums[lane]).tanh();}
        }
        w[BOUT]+a.third.iter().zip(&w[OUT..BOUT]).map(|(x,w)|x*w).sum::<f64>()
    }

    pub(super) fn deep_value_gradient(&self, state:&[f64], a:&DeepValueActivations, dv:f64, g:&mut[f64]) {
        let w=&self.parameters;
        g[BOUT]=dv;
        let mut d3=[0.;32];let mut d2=[0.;64];let mut d1=[0.;128];
        for j in 0..32 {
            g[OUT+j]=dv*a.third[j];
            d3[j]=dv*w[OUT+j]*(1.-a.third[j]*a.third[j]);
            g[B3+j]=d3[j];
            for i in 0..64 {g[W3+j*64+i]=d3[j]*a.second[i];d2[i]+=d3[j]*w[W3+j*64+i];}
        }
        for j in 0..64 {
            d2[j]*=1.-a.second[j]*a.second[j];g[B2+j]=d2[j];
            for i in 0..128 {g[W2+j*128+i]=d2[j]*a.first[i];d1[i]+=d2[j]*w[W2+j*128+i];}
        }
        for j in 0..128 {
            d1[j]*=1.-a.first[j]*a.first[j];g[B1+j]=d1[j];
            for (i,x) in state.iter().enumerate() {g[W1+j*417+i]=d1[j]*x;}
        }
    }
}
