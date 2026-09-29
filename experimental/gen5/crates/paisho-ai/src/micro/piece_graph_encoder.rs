//! Differentiable piece/harmony encoder shared by the optional V7 policy and value.
use super::{MicroPieceGraph, MICRO_HARMONY_INPUTS, MICRO_PIECE_INPUTS};
use crate::StableRng;
use std::sync::Arc;

pub const MICRO_GRAPH_HIDDEN: usize = 16;
pub const MICRO_GRAPH_OUTPUTS: usize = 2 * MICRO_GRAPH_HIDDEN;
const H: usize = MICRO_GRAPH_HIDDEN;
const MESSAGE_INPUTS: usize = 2 * H + MICRO_HARMONY_INPUTS;
const NODE_B: usize = MICRO_PIECE_INPUTS * H;
const MESSAGE_W: usize = NODE_B + H;
const MESSAGE_B: usize = MESSAGE_W + MESSAGE_INPUTS * H;
const UPDATE_W: usize = MESSAGE_B + H;
const UPDATE_B: usize = UPDATE_W + 2 * H * H;
pub const MICRO_GRAPH_PARAMETERS: usize = UPDATE_B + H;
pub const MICRO_GRAPH_ENCODER_SCHEMA: &str = "paisho-micro-piece-mp16x2-v1";

#[derive(Clone, Copy, Debug)]
pub enum MicroGraphPropagation {
    IndependentPieces,
    TwoHarmonyRounds,
}

impl MicroGraphPropagation {
    fn rounds(self) -> usize {
        match self {
            Self::IndependentPieces => 0,
            Self::TwoHarmonyRounds => 2,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MicroGraphEncoder {
    weights: Arc<Vec<f64>>,
}

struct Round {
    messages: Vec<[f64; H]>,
    aggregate: Vec<[f64; H]>,
}
struct Forward {
    states: Vec<Vec<[f64; H]>>,
    rounds: Vec<Round>,
    degree: Vec<f64>,
    output: [f64; MICRO_GRAPH_OUTPUTS],
}

fn dense(input: &[f64], weights: &[f64], bias: &[f64]) -> [f64; H] {
    std::array::from_fn(|j| {
        (bias[j]
            + input
                .iter()
                .enumerate()
                .map(|(i, x)| x * weights[i * H + j])
                .sum::<f64>())
        .tanh()
    })
}
fn dense_gradient(
    input: &[f64],
    output: &[f64; H],
    dy: &[f64; H],
    weights: &[f64],
    gradient: &mut [f64],
    w: usize,
    b: usize,
) -> Vec<f64> {
    let mut dx = vec![0.; input.len()];
    for j in 0..H {
        let dz = dy[j] * (1. - output[j] * output[j]);
        gradient[b + j] += dz;
        for (i, &x) in input.iter().enumerate() {
            gradient[w + i * H + j] += x * dz;
            dx[i] += weights[w + i * H + j] * dz;
        }
    }
    dx
}

impl MicroGraphEncoder {
    pub fn seeded(seed: u64) -> Self {
        let mut rng = StableRng::new(seed);
        let mut weights = vec![0.; MICRO_GRAPH_PARAMETERS];
        for (start, inputs) in [
            (0, MICRO_PIECE_INPUTS),
            (MESSAGE_W, MESSAGE_INPUTS),
            (UPDATE_W, 2 * H),
        ] {
            let scale = (6. / (inputs + H) as f64).sqrt();
            for x in &mut weights[start..start + inputs * H] {
                *x = (2. * rng.next_f64() - 1.) * scale;
            }
        }
        Self {
            weights: Arc::new(weights),
        }
    }
    pub fn from_parameters(weights: Vec<f64>) -> Result<Self, String> {
        if weights.len() != MICRO_GRAPH_PARAMETERS || weights.iter().any(|x| !x.is_finite()) {
            return Err("piece graph encoder requires 1424 finite parameters".into());
        }
        Ok(Self {
            weights: Arc::new(weights),
        })
    }
    pub fn parameters(&self) -> &[f64] {
        &self.weights
    }

    fn forward<const BACKWARD:bool>(&self, graph: &MicroPieceGraph, mode: MicroGraphPropagation) -> Forward {
        let w = &self.weights;
        let mut states = vec![graph
            .nodes
            .iter()
            .map(|n| dense(&n.features, &w[..NODE_B], &w[NODE_B..MESSAGE_W]))
            .collect::<Vec<_>>()];
        let mut degree = vec![0.; graph.nodes.len()];
        for edge in &graph.messages {
            degree[edge.destination] += 1.;
        }
        for d in &mut degree {
            if *d < 1. {
                *d = 1.;
            }
        }
        let mut rounds = vec![];
        for _ in 0..mode.rounds() {
            let previous = states.last().unwrap();
            let mut messages = vec![];
            let mut aggregate = vec![[0.; H]; graph.nodes.len()];
            for edge in &graph.messages {
                let mut x = [0.; MESSAGE_INPUTS];
                x[..H].copy_from_slice(&previous[edge.source]);
                x[H..2 * H].copy_from_slice(&previous[edge.destination]);
                x[2 * H..].copy_from_slice(&edge.features);
                let m = dense(&x, &w[MESSAGE_W..MESSAGE_B], &w[MESSAGE_B..UPDATE_W]);
                for j in 0..H {
                    aggregate[edge.destination][j] += m[j] / degree[edge.destination];
                }
                if BACKWARD {messages.push(m);}
            }
            let next = previous
                .iter()
                .zip(&aggregate)
                .map(|(h, m)| {
                    let mut x = [0.; 2 * H];
                    x[..H].copy_from_slice(h);
                    x[H..].copy_from_slice(m);
                    dense(&x, &w[UPDATE_W..UPDATE_B], &w[UPDATE_B..])
                })
                .collect();
            if !BACKWARD {states.clear();}
            states.push(next);
            if BACKWARD {rounds.push(Round {
                messages,
                aggregate,
            });}
        }
        let mut output = [0.; MICRO_GRAPH_OUTPUTS];
        for (node, h) in graph.nodes.iter().zip(states.last().unwrap()) {
            let offset = usize::from(node.owner != graph.perspective) * H;
            for j in 0..H {
                output[offset + j] += h[j] / 8.;
            }
        }
        Forward {
            states,
            rounds,
            degree,
            output,
        }
    }
    pub fn encode(
        &self,
        graph: &MicroPieceGraph,
        mode: MicroGraphPropagation,
    ) -> [f64; MICRO_GRAPH_OUTPUTS] {
        self.forward::<false>(graph, mode).output
    }
    /// Share the same traversal between global and per-piece readers.
    pub fn encode_all(&self, graph: &MicroPieceGraph, mode: MicroGraphPropagation)
        -> ([f64; MICRO_GRAPH_OUTPUTS], Vec<[f64; H]>)
    {
        let mut f=self.forward::<false>(graph,mode);
        (f.output,f.states.pop().unwrap())
    }
    /// States of the individual pieces, for an action-conditioned reader.
    /// The graph is computed once per position, then shared by all legal actions.
    pub fn encode_nodes(
        &self,
        graph: &MicroPieceGraph,
        mode: MicroGraphPropagation,
    ) -> Vec<[f64; H]> {
        self.forward::<false>(graph, mode).states.pop().unwrap()
    }
    /// Backpropagate a downstream loss through the shared encoder. This does not
    /// update any weights or touch the production optimizer/checkpoint format.
    pub fn gradient(
        &self,
        graph: &MicroPieceGraph,
        mode: MicroGraphPropagation,
        output_gradient: &[f64; MICRO_GRAPH_OUTPUTS],
    ) -> Vec<f64> {
        self.gradient_with_nodes(
            graph,
            mode,
            output_gradient,
            &vec![[0.; H]; graph.nodes.len()],
        )
        .expect("matching graph node count")
    }
    pub fn gradient_with_nodes(
        &self,
        graph: &MicroPieceGraph,
        mode: MicroGraphPropagation,
        output_gradient: &[f64; MICRO_GRAPH_OUTPUTS],
        node_gradients: &[[f64; H]],
    ) -> Result<Vec<f64>, String> {
        if node_gradients.len() != graph.nodes.len() {
            return Err("one gradient is required per graph piece".into());
        }
        let f = self.forward::<true>(graph, mode);
        let mut gradient = vec![0.; MICRO_GRAPH_PARAMETERS];
        let mut dh: Vec<[f64; H]> = graph
            .nodes
            .iter()
            .map(|n| {
                let offset = usize::from(n.owner != graph.perspective) * H;
                std::array::from_fn(|j| output_gradient[offset + j] / 8.)
            })
            .collect();
        for (dy, extra) in dh.iter_mut().zip(node_gradients) {
            for j in 0..H {
                dy[j] += extra[j];
            }
        }
        for t in (0..mode.rounds()).rev() {
            let mut previous = vec![[0.; H]; graph.nodes.len()];
            let mut dm = vec![[0.; H]; graph.nodes.len()];
            for i in 0..graph.nodes.len() {
                let mut x = [0.; 2 * H];
                x[..H].copy_from_slice(&f.states[t][i]);
                x[H..].copy_from_slice(&f.rounds[t].aggregate[i]);
                let dx = dense_gradient(
                    &x,
                    &f.states[t + 1][i],
                    &dh[i],
                    &self.weights,
                    &mut gradient,
                    UPDATE_W,
                    UPDATE_B,
                );
                previous[i].copy_from_slice(&dx[..H]);
                dm[i].copy_from_slice(&dx[H..]);
            }
            for (k, edge) in graph.messages.iter().enumerate() {
                let mut x = [0.; MESSAGE_INPUTS];
                x[..H].copy_from_slice(&f.states[t][edge.source]);
                x[H..2 * H].copy_from_slice(&f.states[t][edge.destination]);
                x[2 * H..].copy_from_slice(&edge.features);
                let dy =
                    std::array::from_fn(|j| dm[edge.destination][j] / f.degree[edge.destination]);
                let dx = dense_gradient(
                    &x,
                    &f.rounds[t].messages[k],
                    &dy,
                    &self.weights,
                    &mut gradient,
                    MESSAGE_W,
                    MESSAGE_B,
                );
                for j in 0..H {
                    previous[edge.source][j] += dx[j];
                    previous[edge.destination][j] += dx[H + j];
                }
            }
            dh = previous;
        }
        for i in 0..graph.nodes.len() {
            dense_gradient(
                &graph.nodes[i].features,
                &f.states[0][i],
                &dh[i],
                &self.weights,
                &mut gradient,
                0,
                NODE_B,
            );
        }
        Ok(gradient)
    }
}

#[cfg(test)]
#[path = "piece_graph_tests.rs"]
mod tests;
