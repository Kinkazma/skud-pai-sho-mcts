use paisho_ai::*;
use paisho_core::*;
use std::sync::Arc;
fn example() -> MicroExample {
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let actions: Vec<_> = legal_actions(&p)
        .iter()
        .map(|a| micro_action_features(&p, *a))
        .collect();
    let mut policy = vec![0.; actions.len()];
    policy[3] = 1.;
    MicroExample { policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: 0,
        state: micro_state_features(&p).to_vec(),
        actions,
        policy,
        value: 0.4,
        policy_weight: 0.7,
    }
}
#[test]
fn upgrade_preserves_logits_values_and_search_exactly() {
    let old = MicroModel::seeded(17);
    let new = old.with_residual_policy(29);
    let ex = example();
    assert_eq!(&new.parameters()[..MICRO_PARAMETERS], old.parameters());
    assert_eq!(new.parameters().len(), MICRO_RESIDUAL_PARAMETERS);
    assert_eq!(new, new.with_residual_policy(70));
    assert_eq!(
        MicroModel::logits(&old.embed(&ex.state), &ex.actions),
        MicroModel::logits(&new.embed(&ex.state), &ex.actions)
    );
    assert_eq!(old.embed(&ex.state).value, new.embed(&ex.state).value);
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let a = MicroMctsSession::new(Arc::new(old))
        .search_until(&p, 64, None)
        .unwrap();
    let b = MicroMctsSession::new(Arc::new(new))
        .search_until(&p, 64, None)
        .unwrap();
    assert_eq!(a.visits, b.visits);
    assert_eq!(a.priors, b.priors);
    assert_eq!(a.selected_index, b.selected_index);
}
#[test]
fn nonlinear_gradient_matches_finite_differences_in_every_parameter_block() {
    let mut w = MicroModel::seeded(17)
        .with_residual_policy(29)
        .parameters()
        .to_vec();
    for (i, p) in w[6257..].iter_mut().enumerate() {
        *p = 0.1 * (i as f64 - 7.);
    }
    let m = MicroModel::from_parameters(w).unwrap();
    let ex = example();
    let (_, g) = m.loss_gradient(&ex).unwrap();
    for i in (0..MICRO_RESIDUAL_PARAMETERS)
        .step_by(11)
        .chain([5216, 5217, 5728, 5729, 6240, 6241, 6256, 6257, 6272, 6273])
    {
        let mut plus = m.parameters().to_vec();
        let mut minus = plus.clone();
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        let loss = |w| {
            MicroModel::from_parameters(w)
                .unwrap()
                .loss_gradient(&ex)
                .unwrap()
                .0
                .total(ex.policy_weight)
        };
        assert!(
            ((loss(plus) - loss(minus)) / 2e-5 - g[i]).abs() < 1e-7,
            "parameter {i}"
        );
    }
}
#[test]
fn nonlinear_policy_learns_a_convex_interior_winner() {
    let mut actions = vec![[0.; MICRO_ACTION_INPUTS]; 3];
    actions[0][0] = -1.;
    actions[1][0] = 0.;
    actions[2][0] = 1.;
    // A linear score can never prefer the midpoint to both endpoints.
    let ex = MicroExample { policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: 0,
        state: vec![0.2; MICRO_INPUTS],
        actions,
        policy: vec![0., 1., 0.],
        value: 0.,
        policy_weight: 1.,
    };
    let mut m = MicroModel::seeded(17).with_residual_policy(29);
    for _ in 0..2000 {
        m.train_step(&ex, 0.05, 0.).unwrap();
    }
    let logits = MicroModel::logits(&m.embed(&ex.state), &ex.actions);
    assert!(
        logits[1] > logits[0] + 1. && logits[1] > logits[2] + 1.,
        "{logits:?}"
    );
}
#[test]
fn residual_batch_paths_agree_and_old_embeddings_remain_immutable() {
    let mut a = MicroModel::seeded(17).with_residual_policy(29);
    let mut b = a.clone();
    let ex = example();
    let before = a.embed(&ex.state);
    let logits = MicroModel::logits(&before, &ex.actions);
    for _ in 0..3 {
        a.train_batch(&[&ex, &ex], 0.05, 1e-5).unwrap();
        b.train_batch_inline(&[&ex, &ex], 0.05, 1e-5).unwrap();
    }
    assert_eq!(a, b);
    assert_eq!(logits, MicroModel::logits(&before, &ex.actions));
    assert_ne!(logits, MicroModel::logits(&a.embed(&ex.state), &ex.actions));
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let r = MicroMctsSession::new(Arc::new(a.clone()))
        .search_until(&p, 16, None)
        .unwrap();
    assert_eq!(
        r.priors,
        micro_softmax(&MicroModel::logits(&a.embed(&ex.state), &ex.actions)).unwrap()
    );
    let mut bad = ex;
    bad.value = f64::NAN;
    let saved = a.clone();
    assert!(a.train_step(&bad, 0.05, 0.).is_err());
    assert_eq!(a, saved);
}
