use paisho_ai::*;
use paisho_core::*;
use std::sync::Arc;

fn model() -> Gen32Model {
    Gen32Model::from_gen31(CompactValueModel::from_weights([0.; 64]).unwrap(), 31)
        .with_value128([0.; 64])
        .unwrap()
}
fn example(state: [f64; 128], value: f64) -> MicroExample {
    MicroExample {
        sequence_source: 0,
        state,
        actions: vec![],
        policy: vec![],
        value,
        policy_weight: 0.,
    }
}
#[test]
fn neutral_residual_preserves_record_values_search_and_shared_snapshots() {
    let mut old = Gen32Model::from_gen31(CompactValueModel::default(), 31)
        .with_value128([0.017; 64])
        .unwrap();
    let record: GameRecord = include_str!("fixtures/gen34-r5-capture.psr")
        .parse()
        .unwrap();
    let mut new = old.clone().with_value_residual(71);
    let frozen = new.clone();
    assert!(Arc::ptr_eq(
        new.value_residual.as_ref().unwrap(),
        frozen.value_residual.as_ref().unwrap()
    ));
    assert_eq!(
        new.clone().with_value_residual(999).value_residual,
        new.value_residual
    );
    let mut p = record.initial_position();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for (i, a) in record.actions().iter().enumerate() {
        for player in [Player::Host, Player::Guest] {
            assert_eq!(old.value_at(&p, player), new.value_at(&p, player));
        }
        if [5, 9, 13].contains(&i) {
            let c = MctsConfig {
                simulations: 32,
                ..Default::default()
            };
            let mut x = MctsSession::new(17, c, &old).unwrap();
            let mut y = MctsSession::new(17, c, &new).unwrap();
            let legal = legal_actions(&p);
            pool.install(|| {
                assert_eq!(
                    x.search_until(&p, &legal, None).unwrap(),
                    y.search_until(&p, &legal, None).unwrap()
                )
            });
        }
        p.apply(*a).unwrap();
    }
    assert_eq!(
        old.value_at(&p, Player::Host),
        new.value_at(&p, Player::Host)
    );
    let ex = example([0.1; 128], -0.7);
    old.train(&ex, 0.01).unwrap();
    new.train(&ex, 0.01).unwrap();
    assert_eq!(old.value.weights(), new.value.weights());
    assert_eq!(old.value_extra, new.value_extra);
    assert_eq!(old.policy.parameters(), new.policy.parameters());
    assert!(new.value_residual.as_ref().unwrap().active());
    assert!(!frozen.value_residual.as_ref().unwrap().active());
    assert!(!Arc::ptr_eq(
        new.value_residual.as_ref().unwrap(),
        frozen.value_residual.as_ref().unwrap()
    ));
}
#[test]
fn all_residual_gradients_match_finite_differences_and_errors_are_atomic() {
    let mut m = model().with_value_residual(7);
    let mut w = m.value_residual.as_ref().unwrap().parameters().to_vec();
    for (i, x) in w[2064..].iter_mut().enumerate() {
        *x = (i as f64 - 8.) * 0.023;
    }
    m.value_residual = Some(Arc::new(
        Gen3ValueResidual::from_parameters(w.clone()).unwrap(),
    ));
    let ex = example(
        std::array::from_fn(|i| ((i * 13 % 29) as f64 - 14.) / 20.),
        0.65,
    );
    let mut learned = m.clone();
    learned.train(&ex, 0.01).unwrap();
    let loss = |m: &Gen32Model| 0.5 * (m.predict_value_state(&ex.state) - ex.value).powi(2);
    for k in 0..GEN3_VALUE_RESIDUAL_PARAMETERS {
        let mut plus = m.clone();
        let mut minus = m.clone();
        let mut a = w.clone();
        let epsilon = 1e-6;
        a[k] += epsilon;
        plus.value_residual = Some(Arc::new(
            Gen3ValueResidual::from_parameters(a.clone()).unwrap(),
        ));
        a[k] -= 2. * epsilon;
        minus.value_residual = Some(Arc::new(Gen3ValueResidual::from_parameters(a).unwrap()));
        let expected = (loss(&plus) - loss(&minus)) / (2. * epsilon);
        let observed = (w[k] - learned.value_residual.as_ref().unwrap().parameters()[k]) / 0.01;
        assert!(
            (expected - observed).abs() < 1e-7,
            "residual gradient {k}: {expected} vs {observed}"
        );
    }
    let before = learned.clone();
    assert!(learned.train(&ex, f64::NAN).is_err());
    assert_eq!(before.value.weights(), learned.value.weights());
    assert_eq!(before.value_extra, learned.value_extra);
    assert_eq!(before.value_residual, learned.value_residual);
    assert_eq!(before.policy.parameters(), learned.policy.parameters());
    assert!(Gen3ValueResidual::from_parameters(vec![0.; 2080]).is_err());
    assert!(Gen3ValueResidual::from_parameters(vec![f64::NAN; 2081]).is_err());
    assert!(Gen3ValueResidual::from_parameters(vec![f64::MAX; 2081]).is_err());
}
#[test]
fn learns_context_dependent_sign_and_transfers_to_untrained_feature_magnitudes() {
    // Synthetic capacity test: not a legal-position or human-strength claim.
    // A linear head cannot separate the four product-sign classes simultaneously.
    for seed in [7, 71, 731] {
        let mut m = model().with_value_residual(seed);
        let mut rng = StableRng::new(seed + 1000);
        for _ in 0..200000 {
            let index = rng.index(4);
            let a = if index & 1 == 0 { -1. } else { 1. };
            let b = if index & 2 == 0 { -1. } else { 1. };
            let mut state = [0.; 128];
            state[4] = a;
            state[63] = b;
            m.train(&example(state, 0.6 * a * b), 0.01).unwrap();
        }
        for a in [-1., 1.] {
            for b in [-1., 1.] {
                let mut x = [0.; 128];
                x[4] = a;
                x[63] = b;
                eprintln!(
                    "fit seed={seed} a={a} b={b} v={}",
                    m.predict_value_state(&x)
                );
            }
        }
        for a in [-0.8, -0.6, 0.6, 0.8] {
            for b in [-0.8, -0.6, 0.6, 0.8] {
                let mut state = [0.; 128];
                state[4] = a;
                state[63] = b;
                let v = m.predict_value_state(&state);
                assert!(v * a * b > 0.05, "seed {seed}: {a},{b} -> {v}");
            }
        }
    }
}
