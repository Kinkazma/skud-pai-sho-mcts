use paisho_ai::*;
use paisho_core::*;
fn example(model: &MicroModel) -> (Position, MicroExample) {
    let old: GameRecord = include_str!("fixtures/site_bot_v1_ring_finish.psr")
        .parse()
        .unwrap();
    let (r, _) = old
        .replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)
        .unwrap();
    let mut p = r.initial_position();
    for a in &r.actions()[..r.actions().len() - 1] {
        p.apply(*a).unwrap();
    }
    let legal = legal_actions(&p);
    let mut policy = vec![0.0; legal.len()];
    policy[legal
        .iter()
        .position(|a| a == r.actions().last().unwrap())
        .unwrap()] = 1.0;
    let ex = MicroExample { policy_support: false, action_values: vec![], 
        value_weight: 1.0, sequence_source: 0,
        state: model.state_features(&p),
        actions: legal
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect(),
        policy,
        value: 1.0,
        policy_weight: 1.0,
    };
    (p, ex)
}
#[test]
fn upgrade_preserves_all_old_weights_and_predictions_and_is_idempotent() {
    let seeded = MicroModel::seeded(43).with_residual_policy(18);
    let mut weights=seeded.parameters().to_vec();
    weights.resize(MICRO_MEMORY_PARAMETERS,0.0);
    // Include an active residual output and nonzero historical reader weights.
    for (i,w) in weights.iter_mut().enumerate() {*w+=(i as f64*0.91).sin()*0.03;}
    let old=MicroModel::from_parameters(weights).unwrap();
    let next = old.with_spatial_policy();
    let (p, ex) = example(&next);
    assert_eq!(next.parameters().len(), MICRO_SPATIAL_PARAMETERS);
    assert_eq!(
        &next.parameters()[..old.parameters().len()],
        old.parameters()
    );
    let a = old.embed(&micro_state_features(&p));
    let b = next.embed(&ex.state);
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    assert_eq!(
        MicroModel::logits(&a, &ex.actions),
        MicroModel::logits(&b, &ex.actions)
    );
    assert_eq!(next, next.with_spatial_policy());
}
#[test]
fn numerical_gradients_cover_global_geometry_local_actions_and_separate_value() {
    let initial = MicroModel::seeded(13)
        .with_residual_policy(9)
        .with_spatial_policy();
    let mut weights = initial.parameters().to_vec();
    for (i, w) in weights.iter_mut().enumerate() {
        *w += (i as f64 * 0.731).sin() * 0.03;
    }
    let model = MicroModel::from_parameters(weights.clone()).unwrap();
    let (_, ex) = example(&model);
    let (_, g) = model.loss_gradient(&ex).unwrap();
    // Select nonzero derivatives from every new block, avoiding a vacuous check.
    for (start, end) in [
        (0, 4096),
        (MICRO_MEMORY_PARAMETERS, MICRO_SPATIAL_LOCAL),
        (MICRO_SPATIAL_LOCAL, MICRO_VALUE_TRUNK),
        (MICRO_VALUE_TRUNK, MICRO_VALUE_BOARD),
        (MICRO_VALUE_BOARD, MICRO_SPATIAL_PARAMETERS),
    ] {
        let i = (start..end)
            .max_by(|a, b| g[*a].abs().total_cmp(&g[*b].abs()))
            .unwrap();
        assert!(g[i].abs() > 1e-10, "inactive gradient block {start}");
        let eps = 1e-5;
        let mut plus = weights.clone();
        plus[i] += eps;
        let mut minus = weights.clone();
        minus[i] -= eps;
        let a = MicroModel::from_parameters(plus)
            .unwrap()
            .loss_gradient(&ex)
            .unwrap()
            .0
            .total(1.0);
        let b = MicroModel::from_parameters(minus)
            .unwrap()
            .loss_gradient(&ex)
            .unwrap()
            .0
            .total(1.0);
        assert!(
            ((a - b) / (2.0 * eps) - g[i]).abs() < 2e-6,
            "gradient mismatch at {i}"
        );
    }
}
#[test]
fn value_training_cannot_erase_policy_and_policy_training_cannot_move_value() {
    let mut model = MicroModel::seeded(13).with_spatial_policy();
    let (_, ex) = example(&model);
    let before = model.embed(&ex.state);
    let logits = MicroModel::logits(&before, &ex.actions);
    let mut value = ex.clone();
    value.actions.clear();
    value.policy.clear();
    value.policy_weight = 0.0;
    model.train_step(&value, 0.02, 0.0).unwrap();
    assert_ne!(model.embed(&ex.state).value, before.value);
    assert_eq!(
        MicroModel::logits(&model.embed(&ex.state), &ex.actions),
        logits
    );
    let value = model.embed(&ex.state).value;
    model.train_policy_step(&ex, 0.02).unwrap();
    assert_eq!(model.embed(&ex.state).value.to_bits(), value.to_bits());
}
#[test]
fn batch_exposure_scaling_removes_singleton_weight_inflation() {
    let model = MicroModel::seeded(13).with_spatial_policy();
    let (_, ex) = example(&model);
    let mut full = model.clone();
    let mut small = model.clone();
    full.train_batch_inline(&vec![&ex; 64], 0.02, 1e-5).unwrap();
    small
        .train_batch_inline(&vec![&ex; 5], 0.02 * 5.0 / 64.0, 1e-5)
        .unwrap();
    for ((a, b), old) in full
        .parameters()
        .iter()
        .zip(small.parameters())
        .zip(model.parameters())
    {
        assert!(((b - old) * 64.0 / 5.0 - (a - old)).abs() < 2e-14);
    }
}

#[test]
fn streamed_spatial_batches_match_ordered_parallel_reduction_and_reject_atomically() {
    let seed = MicroModel::seeded(87).with_spatial_policy();
    let weights = seed.parameters().iter().enumerate()
        .map(|(i,w)| w + (i as f64 * 0.731).sin()*0.03).collect();
    let mut streamed = MicroModel::from_parameters(weights).unwrap();
    let mut parallel = streamed.clone();
    let (_, ex) = example(&streamed);
    let examples: Vec<_> = (0..64).map(|i| {
        let mut e = ex.clone();
        e.value = i as f64/32. - 1.;
        let shift = i % e.policy.len();
        e.policy.rotate_left(shift);
        e
    }).collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    for size in [1,5,64,2] {
        let batch: Vec<_> = examples[..size].iter().collect();
        let a = streamed.train_batch_inline(&batch,0.02,1e-5).unwrap();
        let b = pool.install(||parallel.train_batch(&batch,0.02,1e-5)).unwrap();
        assert_eq!(a.value.to_bits(),b.value.to_bits());
        assert_eq!(a.policy.to_bits(),b.policy.to_bits());
        assert!(streamed.parameters().iter().zip(parallel.parameters())
            .all(|(a,b)|a.to_bits()==b.to_bits()));
    }
    let before = streamed.clone();
    let mut bad = ex.clone(); bad.value = f64::NAN;
    for batch in [vec![&bad,&ex],vec![&ex,&bad]] {
        let a = streamed.train_batch_inline(&batch,0.02,1e-5).unwrap_err();
        let b = parallel.train_batch(&batch,0.02,1e-5).unwrap_err();
        assert_eq!(a,b);
        assert_eq!(streamed,before);
        assert_eq!(parallel,before);
    }
}
