//! Root exploration and Gumbel policy improvement (Danihelka et al., 2022).
//! The completed-Q transform and visit schedule follow DeepMind mctx's
//! action_selection.py/qtransforms.py/seq_halving.py (Apache-2.0).
//! Adaptation: retained Q is reused, but sequential halving allocates NEW visits.
//! Copyright 2021 DeepMind Technologies Limited (reference algorithms).
//! Copyright 2026 Pai Sho contributors (Rust implementation and retention).
//! Licensed under Apache-2.0; see licenses/mctx-Apache-2.0.txt.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MicroSearchMode {
    Puct,
    Gumbel,
}
#[derive(Clone, Copy, Debug)]
pub struct MicroSearchOptions {
    pub mode: MicroSearchMode,
    pub seed: u64,
    /// PUCT-only root Dirichlet mixing; no mutation of cached network priors.
    pub dirichlet_fraction: f64,
    /// Total concentration, divided by the number of legal actions.
    pub dirichlet_total: f64,
    /// Gumbel sampling is disabled for deterministic evaluation when zero.
    pub gumbel_scale: f64,
    pub considered_actions: usize,
    /// PUCT training only: zero disables both forced playouts and target pruning.
    pub forced_playout_strength: f64,
    /// Propagate exact rule outcomes through the explored tree (MCTS-Solver).
    pub proof_search: bool,
}
impl Default for MicroSearchOptions {
    fn default() -> Self {
        Self {
            mode: MicroSearchMode::Puct,
            seed: 0,
            dirichlet_fraction: 0.0,
            dirichlet_total: 10.0,
            gumbel_scale: 1.0,
            considered_actions: 16,
            forced_playout_strength: 0.0,
            proof_search: false,
        }
    }
}
impl MicroSearchOptions {
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.dirichlet_fraction)
            || !self.dirichlet_total.is_finite()
            || self.dirichlet_total <= 0.0
            || !self.gumbel_scale.is_finite()
            || self.gumbel_scale < 0.0
            || self.considered_actions == 0
            || !self.forced_playout_strength.is_finite()
            || !(0.0..=16.0).contains(&self.forced_playout_strength)
            || (self.mode == MicroSearchMode::Gumbel && self.forced_playout_strength != 0.0)
            || (self.mode == MicroSearchMode::Gumbel && self.dirichlet_fraction != 0.0)
        {
            return Err("invalid micro exploration options".into());
        }
        Ok(())
    }
}
fn uniform(rng: &mut StableRng) -> f64 {
    rng.next_f64().max(f64::MIN_POSITIVE)
}
fn normal(rng: &mut StableRng) -> f64 {
    (-2.0 * uniform(rng).ln()).sqrt() * (std::f64::consts::TAU * rng.next_f64()).cos()
}
// Marsaglia-Tsang gamma sampler; shape augmentation handles alpha < 1.
fn gamma(alpha: f64, rng: &mut StableRng) -> f64 {
    if alpha < 1.0 {
        return gamma(alpha + 1.0, rng) * uniform(rng).powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = (9.0 * d).sqrt().recip();
    loop {
        let x = normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = uniform(rng);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}
pub(super) fn noisy_priors(priors: &[f64], o: MicroSearchOptions) -> Vec<f64> {
    if o.dirichlet_fraction == 0.0 {
        return priors.to_vec();
    }
    let mut rng = StableRng::new(o.seed);
    let noise: Vec<_> = priors
        .iter()
        .map(|_| gamma(o.dirichlet_total / priors.len() as f64, &mut rng))
        .collect();
    let sum: f64 = noise.iter().sum();
    priors
        .iter()
        .zip(noise)
        .map(|(p, n)| {
            (1.0 - o.dirichlet_fraction) * p
                + o.dirichlet_fraction
                    * if sum > 0.0 {
                        n / sum
                    } else {
                        1.0 / priors.len() as f64
                    }
        })
        .collect()
}
fn stats(node: &Node, count: usize) -> (Vec<usize>, Vec<f64>) {
    (0..count)
        .map(|i| {
            node.children
                .get(i)
                .and_then(Option::as_ref)
                .map_or((0, node.inference.value()), |c| {
                    let sign =
                        if c.inference.position.to_move() == node.inference.position.to_move() {
                            1.0
                        } else {
                            -1.0
                        };
                    (c.visits, sign * c.value_sum / c.visits.max(1) as f64)
                })
        })
        .unzip()
}
/// Value completion uses the prior-weighted mean of visited actions and raw V.
fn completed_logits(node: &Node, priors: &[f64]) -> Vec<f64> {
    let (visits, values) = stats(node, priors.len());
    let total: usize = visits.iter().sum();
    let mass: f64 = priors
        .iter()
        .zip(&visits)
        .filter(|(_, n)| **n > 0)
        .map(|(p, _)| p.max(f64::MIN_POSITIVE))
        .sum();
    let weighted: f64 = priors
        .iter()
        .zip(&visits)
        .zip(&values)
        .filter(|((_, n), _)| **n > 0)
        .map(|((p, _), q)| p.max(f64::MIN_POSITIVE) * q)
        .sum();
    let mixed = (node.inference.value() + total as f64 * weighted / mass.max(f64::MIN_POSITIVE))
        / (1 + total) as f64;
    let q: Vec<_> = values
        .iter()
        .zip(&visits)
        .map(|(q, n)| if *n == 0 { mixed } else { *q })
        .collect();
    let lo = q.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = q.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = (50 + visits.iter().max().copied().unwrap_or(0)) as f64 * 0.1;
    let cached = node
        .inference
        .policy()
        .expect("policy already materialized");
    let logs = cached.log_priors.get_or_init(|| {
        priors
            .iter()
            .map(|p| p.max(f64::MIN_POSITIVE).ln())
            .collect()
    });
    logs.iter()
        .zip(q)
        .map(|(log, q)| log + scale * (q - lo) / (hi - lo).max(1e-8))
        .collect()
}
pub(super) fn improved_policy(node: &Node, priors: &[f64]) -> Vec<f64> {
    micro_softmax(&completed_logits(node, priors)).expect("finite completed-Q logits")
}
pub(super) fn interior(node: &Node, priors: &[f64]) -> usize {
    let target = improved_policy(node, priors);
    let (visits, _) = stats(node, priors.len());
    let denominator = 1 + visits.iter().sum::<usize>();
    argmax((0..priors.len()).map(|i| (i, target[i] - visits[i] as f64 / denominator as f64)))
}
fn argmax(iter: impl Iterator<Item = (usize, f64)>) -> usize {
    iter.max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .unwrap()
        .0
}
fn schedule(count: usize, budget: usize) -> Vec<usize> {
    if count <= 1 {
        return (0..budget).collect();
    }
    let rounds = (count as f64).log2().ceil() as usize;
    let mut visits = vec![0; count];
    let mut live = count;
    let mut output = Vec::with_capacity(budget + count);
    while output.len() < budget {
        for _ in 0..(budget / (rounds * live)).max(1) {
            output.extend_from_slice(&visits[..live]);
            for v in &mut visits[..live] {
                *v += 1;
            }
        }
        live = (live / 2).max(2);
    }
    output.truncate(budget);
    output
}
pub(super) struct GumbelRoot {
    noise: Vec<f64>,
    schedule: Vec<usize>,
}
impl GumbelRoot {
    pub(super) fn new(priors: &[f64], budget: usize, o: MicroSearchOptions) -> Self {
        let mut rng = StableRng::new(o.seed);
        Self {
            noise: priors
                .iter()
                .map(|_| -(-uniform(&mut rng).ln()).ln() * o.gumbel_scale)
                .collect(),
            schedule: schedule(o.considered_actions.min(priors.len()).min(budget), budget),
        }
    }
    pub(super) fn select(
        &self,
        node: &Node,
        priors: &[f64],
        before: &[usize],
        step: Option<usize>,
    ) -> usize {
        let (visits, _) = stats(node, priors.len());
        let fresh: Vec<_> = visits.iter().zip(before).map(|(a, b)| a - b).collect();
        let wanted = step.map_or_else(|| *fresh.iter().max().unwrap(), |s| self.schedule[s]);
        let logits = completed_logits(node, priors);
        argmax(
            (0..priors.len())
                .filter(|&i| fresh[i] == wanted)
                .map(|i| (i, logits[i] + self.noise[i])),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn halving_has_expected_rounds_and_all_budgets() {
        assert_eq!(
            schedule(4, 16),
            vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        );
        for b in 1..=512 {
            for n in 1..=16 {
                assert_eq!(schedule(n.min(b), b).len(), b);
            }
        }
    }
    #[test]
    fn noise_is_seeded_normalized_and_never_mutates_prior() {
        let prior = vec![0.25; 4];
        let o = MicroSearchOptions {
            dirichlet_fraction: 0.25,
            ..Default::default()
        };
        let p = noisy_priors(&prior, o);
        assert_eq!(p, noisy_priors(&prior, o));
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(prior, vec![0.25; 4]);
        assert_ne!(prior, p);
    }
}
