use paisho_ai::*;
use paisho_core::*;
use std::sync::Arc;
fn start() -> Position {
    Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3))
}
fn example() -> MicroExample {
    let p = start();
    let actions = legal_actions(&p);
    let mut policy = vec![0.0; actions.len()];
    policy[3] = 1.0;
    MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: 0,
        state: micro_state_features(&p).to_vec(),
        actions: actions
            .into_iter()
            .map(|a| micro_action_features(&p, a))
            .collect(),
        policy,
        value: 0.7,
        policy_weight: 0.8,
    }
}
#[test]
fn shared_trunk_value_and_policy_gradients_match_finite_differences() {
    let m = MicroModel::seeded(17);
    let ex = example();
    let (_, g) = m.loss_gradient(&ex).unwrap();
    for i in (0..MICRO_PARAMETERS)
        .step_by(13)
        .chain([0, 4095, 4096, 4127, 4128, 4160, 4161, 5184, 5185, 5216])
    {
        let mut plus = m.parameters().to_vec();
        let mut minus = plus.clone();
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        let a = MicroModel::from_parameters(plus)
            .unwrap()
            .loss_gradient(&ex)
            .unwrap()
            .0
            .total(ex.policy_weight);
        let b = MicroModel::from_parameters(minus)
            .unwrap()
            .loss_gradient(&ex)
            .unwrap()
            .0
            .total(ex.policy_weight);
        assert!(((a - b) / 2e-5 - g[i]).abs() < 1e-7, "parameter {i}");
    }
}
#[test]
fn joint_training_lowers_both_losses_and_rejects_invalid_update() {
    let mut m = MicroModel::seeded(17);
    let ex = example();
    let before = m.loss_gradient(&ex).unwrap().0;
    for _ in 0..200 {
        m.train_step(&ex, 0.05, 1e-5).unwrap();
    }
    let after = m.loss_gradient(&ex).unwrap().0;
    assert!(after.value < before.value);
    assert!(after.policy < before.policy);
    let saved = m.clone();
    let mut bad = ex;
    bad.policy[0] = f64::NAN;
    assert!(m.train_step(&bad, 0.1, 0.0).is_err());
    assert_eq!(saved, m);
}
#[test]
fn learned_policy_changes_visits_beyond_the_first_expansion() {
    let mut w = vec![0.0; MICRO_PARAMETERS];
    // Action-type bias: plant vs arrangement. Zero values isolate PUCT.
    w[5185] = 8.0;
    let planted = Arc::new(MicroModel::from_parameters(w.clone()).unwrap());
    w[5185] = -8.0;
    w[5186] = 8.0;
    let arranged = Arc::new(MicroModel::from_parameters(w).unwrap());
    let mut a = MicroMctsSession::new(planted);
    let mut b = MicroMctsSession::new(arranged);
    let a = a.search_until(&start(), 32, None).unwrap();
    let b = b.search_until(&start(), 32, None).unwrap();
    let count = |r: &MicroSearchReport| {
        r.actions
            .iter()
            .zip(&r.visits)
            .filter(|(a, _)| matches!(a, Action::Plant { .. }))
            .map(|(_, n)| n)
            .sum::<usize>()
    };
    assert!(count(&a) > count(&b) + 10);
    assert_eq!(a.visits.iter().sum::<usize>(), 32);
    assert!((a.priors.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}
#[test]
fn retained_tree_counts_new_visits_and_reuses_inference_without_weight_reload() {
    let model = Arc::new(MicroModel::seeded(9));
    let mut s = MicroMctsSession::new(model.clone());
    assert_eq!(Arc::strong_count(&model), 2);
    let p = start();
    let a = s.search_until(&p, 32, None).unwrap();
    let b = s.search_until(&p, 32, None).unwrap();
    assert_eq!(b.inherited_visits, 32);
    assert_eq!(b.new_visits.iter().sum::<usize>(), 32);
    assert_eq!(b.visits.iter().sum::<usize>(), 64);
    let action = b.actions[b.selected_index];
    let inherited = b.visits[b.selected_index];
    assert!(s.advance(action).unwrap());
    assert_eq!(s.retained_visits(), inherited);
    let mut next = p;
    next.apply(action).unwrap();
    let c = s.search_until(&next, 8, None).unwrap();
    assert_eq!(c.inherited_visits, inherited);
    assert_ne!(a.state, c.example(1.0, 0.0).unwrap().state);
}
#[test]
fn memory_limit_and_model_identity_prevent_stale_reuse() {
    let mut s = MicroMctsSession::new(Arc::new(MicroModel::seeded(1)));
    s.set_limits(0, 0);
    let r = s.search_until(&start(), 8, None).unwrap();
    assert!(r.memory_reset);
    assert_eq!(s.retained_visits(), 0);
    assert_eq!(s.retained_bytes(), 0);
    let mut other = MicroMctsSession::new(Arc::new(MicroModel::seeded(2)));
    let q = other.search_until(&start(), 8, None).unwrap();
    assert_eq!(q.inherited_visits, 0);
    assert_ne!(r.priors, q.priors);
}
#[test]
fn expired_deadline_does_not_create_fake_training_visits() {
    let mut s = MicroMctsSession::new(Arc::new(MicroModel::seeded(1)));
    let r = s
        .search_until(&start(), 8, Some(std::time::Instant::now()))
        .unwrap();
    assert_eq!(r.simulations, 0);
    assert!(r.example(1.0, 1.0).is_err());
}
#[test]
fn pool_size_keeps_exact_search_results() {
    let m = Arc::new(MicroModel::seeded(3));
    let p = start();
    let run = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .unwrap()
            .install(|| {
                MicroMctsSession::new(m.clone())
                    .search_until(&p, 32, None)
                    .unwrap()
            })
    };
    let a = run(1);
    let b = run(4);
    assert_eq!(a.visits, b.visits);
    assert_eq!(a.values, b.values);
    assert_eq!(a.priors, b.priors);
}

#[test]
fn dense_features_cover_all_gradient_blocks_and_parallel_updates_are_exact() {
    let mut ex = example();
    ex.state = (0..MICRO_INPUTS).map(|i| (i as f64 * 0.71).sin() * 0.5).collect();
    for (j, a) in ex.actions.iter_mut().enumerate() {
        *a = std::array::from_fn(|i| ((i + j * 32) as f64 * 0.13).cos() * 0.5);
    }
    let m = MicroModel::seeded(7);
    let (_, g) = m.loss_gradient(&ex).unwrap();
    for i in (0..MICRO_PARAMETERS).step_by(7) {
        let mut a = m.parameters().to_vec();
        let mut b = a.clone();
        a[i] += 1e-5;
        b[i] -= 1e-5;
        let loss = |w| {
            MicroModel::from_parameters(w)
                .unwrap()
                .loss_gradient(&ex)
                .unwrap()
                .0
                .total(ex.policy_weight)
        };
        assert!(((loss(a) - loss(b)) / 2e-5 - g[i]).abs() < 1e-7, "{i}");
    }
    let run = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut m = m.clone();
                m.train_batch(&[&ex, &example()], 0.1, 1e-5).unwrap();
                m
            })
    };
    assert_eq!(run(1), run(4));
}
#[test]
fn exact_position_revisit_reuses_inference_but_not_other_path_visits() {
    let mut s = MicroMctsSession::new(Arc::new(MicroModel::seeded(8)));
    let p = start();
    let a = s.search_until(&p, 8, None).unwrap();
    let mut next = p.clone();
    next.apply(a.actions[a.selected_index]).unwrap();
    s.search_until(&next, 8, None).unwrap();
    let again = s.search_until(&p, 1, None).unwrap();
    assert_eq!(again.inherited_visits, 0);
    assert!(again.inference_cache_hits >= 1);
}

#[test]
fn absolute_reserves_add_information_missing_from_legacy_differences() {
    let a = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let b = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red4));
    assert_eq!(
        CompactValueFeatures::extract(&a, a.to_move()).values(),
        CompactValueFeatures::extract(&b, b.to_move()).values()
    );
    assert_ne!(micro_state_features(&a), micro_state_features(&b));
    assert!(micro_state_features(&a).iter().all(|x| x.is_finite()));
}

#[test]
fn an_expired_search_cannot_relabel_inherited_visits_as_new_deep_targets() {
    let mut s = MicroMctsSession::new(Arc::new(MicroModel::seeded(1)));
    s.search_until(&start(), 32, None).unwrap();
    let r = s
        .search_until(&start(), 256, Some(std::time::Instant::now()))
        .unwrap();
    assert_eq!(r.inherited_visits, 32);
    assert_eq!(r.simulations, 0);
    assert!(r.example(1.0, 1.0).is_err());
}

#[test]
fn inline_minibatches_preserve_parallel_weights_exactly() {
    let mut parallel = MicroModel::seeded(12);
    let mut inline = parallel.clone();
    let a = example();
    let b = example();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for _ in 0..5 {
        pool.install(|| parallel.train_batch(&[&a, &b], 0.1, 1e-5))
            .unwrap();
        inline.train_batch_inline(&[&a, &b], 0.1, 1e-5).unwrap();
        assert_eq!(inline.parameters(), parallel.parameters());
    }
}
