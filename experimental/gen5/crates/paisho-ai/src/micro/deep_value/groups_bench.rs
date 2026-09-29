//! Explicit frozen native cost probe; no production grouping choice is changed.
use super::*;
struct Packed<const N: usize> {
    first: Vec<[f64; N]>,
    second: Vec<[f64; N]>,
    third: Vec<[f64; N]>,
}
impl<const N: usize> Packed<N> {
    fn new(w: &[f64]) -> Self {
        let pack = |start: usize, inputs: usize, outputs: usize| {
            (0..outputs)
                .step_by(N)
                .flat_map(|j| {
                    (0..inputs).map(move |i| {
                        std::array::from_fn(|lane| w[start + (j + lane) * inputs + i])
                    })
                })
                .collect()
        };
        Self {
            first: pack(W1, 417, 128),
            second: pack(W2, 128, 64),
            third: pack(W3, 64, 32),
        }
    }
    fn forward(
        &self,
        w: &[f64],
        state: &[f64],
        occupied: &OccupiedFeatures,
        a: &mut DeepValueActivations,
    ) -> f64 {
        let mut indices = [0u8; 128];
        let mut count = 0;
        for (i, x) in state[..128].iter().enumerate() {
            if *x != 0. {
                indices[count] = i as u8;
                count += 1;
            }
        }
        for j in (0..128).step_by(N) {
            let start = j / N * 417;
            let dense = sparse_dots(
                &state[..128],
                &indices[..count],
                &self.first[start..start + 128],
            );
            let board = occupied.dots(state, &self.first[start + 128..start + 417]);
            for lane in 0..N {
                a.first[j + lane] = (w[B1 + j + lane] + dense[lane] + board[lane]).tanh();
            }
        }
        for j in (0..64).step_by(N) {
            let sums = dots(&a.first, &self.second[j / N * 128..(j / N + 1) * 128]);
            for lane in 0..N {
                a.second[j + lane] = (w[B2 + j + lane] + sums[lane]).tanh();
            }
        }
        for j in (0..32).step_by(N) {
            let sums = dots(&a.second, &self.third[j / N * 64..(j / N + 1) * 64]);
            for lane in 0..N {
                a.third[j + lane] = (w[B3 + j + lane] + sums[lane]).tanh();
            }
        }
        w[BOUT]
            + a.third
                .iter()
                .zip(&w[OUT..BOUT])
                .map(|(x, w)| x * w)
                .sum::<f64>()
    }
}
fn measure<const N: usize>(m: &MicroModel, states: &[(Vec<f64>, OccupiedFeatures)]) -> f64 {
    let weights = Packed::<N>::new(m.parameters());
    let mut activations = DeepValueActivations::default();
    for (state, occupied) in states {
        let mut original = DeepValueActivations::default();
        let before = m.deep_value_forward(state, occupied, &mut original);
        let after = weights.forward(m.parameters(), state, occupied, &mut activations);
        assert_eq!(before.to_bits(), after.to_bits());
        assert!(original
            .first
            .iter()
            .chain(&original.second)
            .chain(&original.third)
            .zip(
                activations
                    .first
                    .iter()
                    .chain(&activations.second)
                    .chain(&activations.third)
            )
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    let now = std::time::Instant::now();
    for _ in 0..1024 {
        for (state, occupied) in states {
            std::hint::black_box(weights.forward(
                std::hint::black_box(m.parameters()),
                std::hint::black_box(state),
                occupied,
                &mut activations,
            ));
        }
    }
    now.elapsed().as_secs_f64()
}
#[test]
#[ignore = "explicit frozen model benchmark: GEN5_GROUP_MODEL points to f64 little-endian weights"]
fn compare_deep_group_widths_on_real_positions() {
    let path = std::env::var("GEN5_GROUP_MODEL").expect("GEN5_GROUP_MODEL");
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 8, 0);
    let m = MicroModel::from_parameters(
        bytes
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect(),
    )
    .unwrap();
    let mut states = vec![];
    for text in [
        include_str!("../../../tests/fixtures/micro-alias-1-a.psr"),
        include_str!("../../../tests/fixtures/micro-alias-2-a.psr"),
        include_str!("../../../tests/fixtures/micro-alias-0-a.psr"),
    ] {
        let r: paisho_core::GameRecord = text.parse().unwrap();
        let mut p = r.initial_position();
        for (i, &action) in r.actions().iter().enumerate() {
            if i % 4 == 0 {
                let state = m.state_features(&p);
                let occupied = OccupiedFeatures::new(&state);
                states.push((state, occupied));
            }
            p.apply(action).unwrap();
        }
    }
    for n in [8, 16, 32, 32, 16, 8] {
        let seconds = match n {
            8 => measure::<8>(&m, &states),
            16 => measure::<16>(&m, &states),
            32 => measure::<32>(&m, &states),
            _ => unreachable!(),
        };
        println!(
            "group={n} states={} evaluations={} seconds={seconds} all_224_activations_exact=true",
            states.len(),
            states.len() * 1024
        );
    }
}
