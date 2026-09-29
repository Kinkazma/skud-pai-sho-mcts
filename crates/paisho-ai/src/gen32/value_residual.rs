//! Small contextual correction to the linear value, before its existing squash.
//! Weights are immutable and Arc-shared by game snapshots; no inference I/O.
use crate::StableRng;

pub const GEN3_VALUE_RESIDUAL_HIDDEN: usize = 16;
pub const GEN3_VALUE_RESIDUAL_PARAMETERS: usize = 2081;
pub const GEN3_VALUE_RESIDUAL_SCHEMA: &str = "paisho-gen3-value-residual16-memory-v1";
const B: usize = 128 * GEN3_VALUE_RESIDUAL_HIDDEN;
const O: usize = B + GEN3_VALUE_RESIDUAL_HIDDEN;
const OB: usize = O + GEN3_VALUE_RESIDUAL_HIDDEN;

#[derive(Clone, Debug, PartialEq)]
pub struct Gen3ValueResidual {
    parameters: Vec<f64>,
    active: bool,
}
impl Gen3ValueResidual {
    pub fn seeded(seed: u64) -> Self {
        let mut rng = StableRng::new(seed);
        let mut parameters = vec![0.; GEN3_VALUE_RESIDUAL_PARAMETERS];
        let scale = (6_f64 / (128 + GEN3_VALUE_RESIDUAL_HIDDEN) as f64).sqrt();
        for w in &mut parameters[..B] {
            *w = (2. * rng.next_f64() - 1.) * scale;
        }
        // Nonzero hidden biases allow even as well as odd contextual relations.
        for w in &mut parameters[B..O] {
            *w = (2. * rng.next_f64() - 1.) * 0.5;
        }
        Self::from_parameters(parameters).expect("finite seeded residual")
    }
    pub fn from_parameters(parameters: Vec<f64>) -> Result<Self, String> {
        if parameters.len() != GEN3_VALUE_RESIDUAL_PARAMETERS
            || parameters.iter().any(|w| !w.is_finite())
            || !parameters.iter().map(|w| w.abs()).sum::<f64>().is_finite()
        {
            return Err("invalid Gen3 value residual parameters".into());
        }
        let active = parameters[O..].iter().any(|w| *w != 0.);
        Ok(Self { parameters, active })
    }
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    pub fn active(&self) -> bool {
        self.active
    }
    fn hidden(&self, state: &[f64; 128]) -> [f64; GEN3_VALUE_RESIDUAL_HIDDEN] {
        std::array::from_fn(|j| {
            (self.parameters[B + j]
                + self.parameters[j * 128..(j + 1) * 128]
                    .iter()
                    .zip(state)
                    .map(|(w, x)| w * x)
                    .sum::<f64>())
            .tanh()
        })
    }
    pub fn raw(&self, state: &[f64; 128]) -> f64 {
        if !self.active {
            return 0.;
        }
        self.parameters[OB]
            + self
                .hidden(state)
                .iter()
                .zip(&self.parameters[O..OB])
                .map(|(h, w)| h * w)
                .sum::<f64>()
    }
    /// `step` is learning_rate * d(loss)/d(total pre-squash value).
    pub(super) fn trained(&self, state: &[f64; 128], step: f64) -> Result<Self, String> {
        let hidden = self.hidden(state);
        let mut next = self.parameters.clone();
        for (j, h) in hidden.into_iter().enumerate() {
            let d = step * self.parameters[O + j] * (1. - h * h);
            for (k, x) in state.iter().enumerate() {
                next[j * 128 + k] -= d * x;
            }
            next[B + j] -= d;
            next[O + j] -= step * h;
        }
        next[OB] -= step;
        Self::from_parameters(next)
    }
}
