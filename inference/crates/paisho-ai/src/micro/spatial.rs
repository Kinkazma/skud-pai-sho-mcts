//! V4 preserves the V1 input prefix and adds the exact occupied-square map.
//! Empty=0, own kinds=(index+1)/12, opponent kinds=-(index+1)/12.
//! No move application, action-sized storage, or extra search is needed.
use super::*;
use paisho_core::Position;

pub const MICRO_BOARD_INPUTS: usize = 17 * 17;
pub const MICRO_SPATIAL_INPUTS: usize = MICRO_INPUTS + MICRO_BOARD_INPUTS;
pub const MICRO_SPATIAL_LOCAL: usize = MICRO_MEMORY_PARAMETERS + MICRO_BOARD_INPUTS * MICRO_HIDDEN;
pub const MICRO_VALUE_TRUNK: usize = MICRO_SPATIAL_LOCAL + 18 * MICRO_RESIDUAL_HIDDEN;
pub const MICRO_VALUE_BOARD: usize = MICRO_VALUE_TRUNK + MICRO_HIDDEN * (MICRO_INPUTS + 1);
pub const MICRO_SPATIAL_PARAMETERS: usize = MICRO_VALUE_BOARD + MICRO_HIDDEN * MICRO_BOARD_INPUTS;
pub const MICRO_SPATIAL_MODEL_SCHEMA: &str = "paisho-micro-spatial-policy-sequence-v4";
pub const MICRO_SPATIAL_FEATURE_SCHEMA: &str = "paisho-micro-state417-action32-local18-v2";

/// Scan the board once for all 64 policy/value units. Retain the original cell
/// order and Iterator::sum arithmetic, including the empty-board identity.
pub(super) struct OccupiedFeatures {
    indices: [u16; MICRO_BOARD_INPUTS],
    len: usize,
}
impl OccupiedFeatures {
    pub fn new(state: &[f64]) -> Self {
        let mut out = Self { indices: [0; MICRO_BOARD_INPUTS], len: 0 };
        if state.len() == MICRO_SPATIAL_INPUTS {
            for (i, x) in state[MICRO_INPUTS..].iter().enumerate() {
                if *x != 0.0 {
                    out.indices[out.len] = i as u16;
                    out.len += 1;
                }
            }
        }
        out
    }
    pub(super) fn dot(&self, state: &[f64], weights: &[f64]) -> f64 {
        self.indices[..self.len].iter()
            .map(|&i| state[MICRO_INPUTS + i as usize] * weights[i as usize])
            .sum()
    }
    pub(super) fn dots<const N:usize>(&self, state: &[f64], weights: &[[f64;N]]) -> [f64;N] {
        let mut sums=[-0.0;N];
        for &i in &self.indices[..self.len] {
            let x=state[MICRO_INPUTS+i as usize];
            for lane in 0..N {sums[lane]+=x*weights[i as usize][lane];}
        }
        sums
    }
}

pub fn micro_spatial_state_features(position: &Position) -> Vec<f64> {
    let mut x = micro_state_features(position).to_vec();
    x.resize(MICRO_SPATIAL_INPUTS, 0.0);
    for (at, tile) in position.board().occupied() {
        let cell = (at.y() + 8) as usize * 17 + (at.x() + 8) as usize;
        let sign = if tile.owner == position.to_move() {
            1.0
        } else {
            -1.0
        };
        x[MICRO_INPUTS + cell] = sign * (tile.kind.index() + 1) as f64 / 12.0;
    }
    x
}

/// Two 3x3 neighborhoods give the nonlinear action head explicit local context.
/// Coordinates come from the existing action encoding; absent endpoints are zero.
pub(super) fn micro_local_features(board: &[f64], action: &[f64; 32]) -> [f64; 18] {
    let mut out = [0.0; 18];
    for (endpoint, (offset, present)) in [(18, action[26] != 0.0), (20, action[2] == 0.0)]
        .into_iter()
        .enumerate()
    {
        if !present {
            continue;
        }
        let cx = (action[offset] * 8.0).round() as i32 + 8;
        let cy = (action[offset + 1] * 8.0).round() as i32 + 8;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (x, y) = (cx + dx, cy + dy);
                if (0..17).contains(&x) && (0..17).contains(&y) {
                    out[endpoint * 9 + ((dy + 1) * 3 + dx + 1) as usize] =
                        board[y as usize * 17 + x as usize];
                }
            }
        }
    }
    out
}

impl MicroModel {
    pub fn has_spatial(&self) -> bool {
        [MICRO_SPATIAL_PARAMETERS,MICRO_DEEP_VALUE_PARAMETERS,MICRO_NEURAL_MEMORY_PARAMETERS].contains(&self.parameters.len())
    }
    pub fn feature_schema(&self) -> &'static str {
        if self.has_spatial() {
            MICRO_SPATIAL_FEATURE_SCHEMA
        } else {
            MICRO_FEATURE_SCHEMA
        }
    }
    /// All old weights, counters and reader coefficients remain bit-identical.
    pub fn with_spatial_policy(&self) -> Self {
        if self.has_spatial() {
            return self.clone();
        }
        let old = self.with_residual_policy(0);
        let trunk = old.parameters[..VALUE_W].to_vec();
        let mut parameters=old.parameters.as_ref().clone();
        parameters.resize(MICRO_SPATIAL_PARAMETERS, 0.0);
        parameters[MICRO_VALUE_TRUNK..MICRO_VALUE_BOARD].copy_from_slice(&trunk);
        let mut model = Self::from_parameters(parameters).expect("finite spatial migration");
        model.memory = self.memory.clone();
        model
    }
    pub fn state_features(&self, position: &Position) -> Vec<f64> {
        if self.has_spatial() {
            micro_spatial_state_features(position)
        } else {
            micro_state_features(position).to_vec()
        }
    }
    pub(super) fn value_hidden(&self, state: &[f64], occupied: &OccupiedFeatures) -> [f64; MICRO_HIDDEN] {
        std::array::from_fn(|j| {
            let start = MICRO_VALUE_TRUNK + j * MICRO_INPUTS;
            let base = self.parameters[MICRO_VALUE_TRUNK + TRUNK_B + j]
                + state[..MICRO_INPUTS]
                    .iter()
                    .zip(&self.parameters[start..start + MICRO_INPUTS])
                    .map(|(x, w)| x * w)
                    .sum::<f64>();
            let start = MICRO_VALUE_BOARD + j * MICRO_BOARD_INPUTS;
            let board = if state.len() == MICRO_SPATIAL_INPUTS {
                occupied.dot(state, &self.parameters[start..start + MICRO_BOARD_INPUTS])
            } else {
                0.0
            };
            (base + board).tanh()
        })
    }
    pub(super) fn value_trunk_gradient(
        &self,
        state: &[f64],
        hidden: &[f64; 32],
        dv: f64,
        g: &mut [f64],
    ) {
        for j in 0..MICRO_HIDDEN {
            let d = dv * self.parameters[VALUE_W + j] * (1.0 - hidden[j] * hidden[j]);
            g[MICRO_VALUE_TRUNK + TRUNK_B + j] = d;
            for i in 0..MICRO_INPUTS {
                g[MICRO_VALUE_TRUNK + j * MICRO_INPUTS + i] = d * state[i];
            }
            if state.len() == MICRO_SPATIAL_INPUTS {
                for i in 0..MICRO_BOARD_INPUTS {
                    g[MICRO_VALUE_BOARD + j * MICRO_BOARD_INPUTS + i] = d * state[MICRO_INPUTS + i];
                }
            }
        }
    }
    pub(super) fn spatial_trunk(&self, state: &[f64], unit: usize, occupied: &OccupiedFeatures) -> f64 {
        if !self.has_spatial() || state.len() != MICRO_SPATIAL_INPUTS {
            return 0.0;
        }
        let start = MICRO_MEMORY_PARAMETERS + unit * MICRO_BOARD_INPUTS;
        occupied.dot(state, &self.parameters[start..start + MICRO_BOARD_INPUTS])
    }
    pub(super) fn spatial_gradient(
        &self,
        state: &[f64],
        unit: usize,
        delta: f64,
        gradient: &mut [f64],
    ) {
        if !self.has_spatial() || state.len() != MICRO_SPATIAL_INPUTS {
            return;
        }
        let start = MICRO_MEMORY_PARAMETERS + unit * MICRO_BOARD_INPUTS;
        for (g, x) in gradient[start..start + MICRO_BOARD_INPUTS]
            .iter_mut()
            .zip(&state[MICRO_INPUTS..])
        {
            *g = delta * x;
        }
    }
}

#[cfg(test)]
mod exact_tests {
    use super::*;
    #[test]
    fn occupied_dots_preserve_dense_reference_bits_including_signed_zero() {
        let mut rng = StableRng::new(81357);
        for density in [0, 1, 4, 17, 289] {
            for _ in 0..16 {
                let mut state = vec![0.; MICRO_SPATIAL_INPUTS];
                let mut weights = [0.; MICRO_BOARD_INPUTS];
                for i in 0..MICRO_BOARD_INPUTS {
                    state[MICRO_INPUTS+i] = if rng.index(289) < density {
                        (rng.next_f64()*2.-1.) * 1e-6
                    } else if i%2==0 { -0. } else { 0. };
                    weights[i] = (rng.next_f64()*2.-1.) * 1e6;
                }
                let reference = state[MICRO_INPUTS..].iter().zip(weights)
                    .filter(|(x,_)| **x!=0.).map(|(x,w)| x*w).sum::<f64>();
                let actual = OccupiedFeatures::new(&state).dot(&state,&weights);
                assert_eq!(actual.to_bits(),reference.to_bits());
            }
        }
    }
}
