//! Immutable transposed views for repeated backward reads. Serialized weights
//! retain their original order; snapshots share these derived arrays by Arc.
use super::*;
#[derive(Clone, Debug, PartialEq)]
pub(in crate::micro) struct BackwardWeights {
    pub(super) first: Vec<f64>,
    pub(super) second: Vec<f64>,
    pub(super) side: Vec<f64>,
}
impl BackwardWeights {
    pub(in crate::micro) fn from_parameters(parameters: &[f64]) -> Option<Self> {
        if !cfg!(all(target_os = "macos", target_arch = "aarch64"))
            || parameters.len() != MICRO_NEURAL_MEMORY_PARAMETERS
        {
            return None;
        }
        let w = &parameters[MICRO_NEURAL_MEMORY_START..];
        let transpose = |data: &[f64], rows: usize, cols: usize| {
            let mut out = vec![0.; data.len()];
            for i in 0..rows {
                for j in 0..cols {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
            out
        };
        Some(Self {
            first: transpose(&w[W1..B1], WIDTH, EXPAND),
            second: transpose(&w[W2..B2], EXPAND, WIDTH),
            side: transpose(&w[449 * WIDTH..B0], 34, WIDTH),
        })
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "explicit frozen cost probe; GEN5_GROUP_MODEL contains f64 weights"]
    fn measure_backward_views_with_frozen_weights() {
        let bytes = std::fs::read(std::env::var("GEN5_GROUP_MODEL").unwrap()).unwrap();
        let model = MicroModel::from_parameters(
            bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )
        .unwrap();
        let mut original = model.clone();
        original.neural_backward_weights = None;
        let mut rows = vec![];
        for text in [
            include_str!("../../../tests/fixtures/micro-alias-1-a.psr"),
            include_str!("../../../tests/fixtures/micro-alias-2-a.psr"),
            include_str!("../../../tests/fixtures/micro-alias-0-a.psr"),
        ] {
            let record: paisho_core::GameRecord = text.parse().unwrap();
            let mut p = record.initial_position();
            for (i, &action) in record.actions().iter().enumerate() {
                if i % 4 == 0 {
                    let actions = paisho_core::legal_actions(&p);
                    rows.push(MicroExample { policy_support: false,
                        state: model.state_features(&p),
                        actions: actions
                            .iter()
                            .map(|&a| micro_action_features(&p, a))
                            .collect(),
                        policy: vec![1. / actions.len() as f64; actions.len()],
                        value: 0.4,
                        policy_weight: 1.,
                        value_weight: 1.,
                        sequence_source: 0,
                        action_values: vec![Some(0.3); actions.len()],
                    });
                }
                p.apply(action).unwrap();
            }
        }
        for e in &rows {
            let (a, x) = original.loss_gradient(e).unwrap();
            let (b, y) = model.loss_gradient(e).unwrap();
            assert_eq!(
                (a.value.to_bits(), a.policy.to_bits()),
                (b.value.to_bits(), b.policy.to_bits())
            );
            assert!(x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        for cached in [false, true, true, false] {
            let m = if cached { &model } else { &original };
            let now = std::time::Instant::now();
            for _ in 0..16 {
                for e in &rows {
                    std::hint::black_box(m.loss_gradient(std::hint::black_box(e)).unwrap());
                }
            }
            println!("cached_views={cached} examples={} seconds={} all_gradient_bits_exact=true bank_not_attached=true",rows.len()*16,now.elapsed().as_secs_f64());
        }
    }
    #[test]
    fn backward_views_keep_all_gradient_bits() {
        let m = MicroModel::seeded(1591).with_neural_memory(1753);
        let mut w = m.parameters().to_vec();
        let mut rng = StableRng::new(38529);
        for v in &mut w[MICRO_NEURAL_MEMORY_START + OUT..] {
            *v = (rng.next_f64() - 0.5) * 0.2;
        }
        let m = MicroModel::from_parameters(w).unwrap();
        let w = &m.parameters()[MICRO_NEURAL_MEMORY_START..];
        for n in [3, 31, 32, 33, 65, 130, 185, 257, 512, 1025] {
            let state = (0..417).map(|_| rng.next_f64() - 0.5).collect::<Vec<_>>();
            let actions = (0..n)
                .map(|_| std::array::from_fn(|_| rng.next_f64() - 0.5))
                .collect::<Vec<_>>();
            let logits = (0..n).map(|_| rng.next_f64() - 0.5).collect::<Vec<_>>();
            let cache = m.neural_forward(&state, &actions, &[0.3; 32], -0.2, &logits);
            for d in [
                vec![0.; n * 2],
                vec![-0.; n * 2],
                (0..n * 2).map(|_| rng.next_f64() - 0.5).collect(),
            ] {
                let mut a = vec![0.; MICRO_NEURAL_MEMORY_WEIGHTS];
                let mut b = a.clone();
                let x = cache.backward(w, &d, &mut a, None);
                let y = cache.backward(w, &d, &mut b, m.neural_backward_weights.as_deref());
                assert!(
                    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "gradient rows {n}"
                );
                assert_eq!(x.value.to_bits(), y.value.to_bits());
                assert!(
                    x.memory
                        .iter()
                        .chain(&x.logits)
                        .zip(y.memory.iter().chain(&y.logits))
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "side rows {n}"
                );
            }
        }
    }
}
