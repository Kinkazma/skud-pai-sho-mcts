use paisho_ai::*;
use paisho_core::*;
#[test]
fn neutral_bridge_preserves_compact_values_search_and_retained_branches() {
    let value = CompactValueModel::default();
    let hybrid = Gen32Model::from_gen31(value.clone(), 31);
    let mut p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let config = MctsConfig {
        simulations: 32,
        ..Default::default()
    };
    let mut old = MctsSession::new(91, config, &value).unwrap();
    let mut new = MctsSession::new(91, config, &hybrid).unwrap();
    for _ in 0..12 {
        let legal = legal_actions(&p);
        if legal.is_empty() || p.outcome() != GameOutcome::Ongoing {
            break;
        }
        assert_eq!(
            hybrid
                .evaluate_leaf(&p, p.to_move(), Default::default())
                .unwrap(),
            value.evaluate(&p, p.to_move())
        );
        let a = old.search_until(&p, &legal, None).unwrap();
        let b = new.search_until(&p, &legal, None).unwrap();
        assert_eq!(a.selected_index, b.selected_index);
        for (a, b) in a.actions.iter().zip(&b.actions) {
            assert_eq!(a.visits, b.visits);
            assert_eq!(a.value_sum, b.value_sum);
        }
        let chosen = legal[a.selected_index];
        assert_eq!(old.advance(chosen), new.advance(chosen));
        p.apply(chosen).unwrap();
    }
}
#[test]
fn compact_sgd_is_exact_while_policy_learns_and_invalid_steps_are_atomic() {
    let mut old = CompactValueModel::default();
    let mut model = Gen32Model::from_gen31(old.clone(), 32);
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let legal = legal_actions(&p);
    let mut target = vec![0.; legal.len()];
    target[0] = 1.;
    let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
        value_weight: 1.0, sequence_source: 0,
        state: micro_state_features(&p).to_vec(),
        actions: legal
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect(),
        policy: target,
        value: -0.7,
        policy_weight: 1.,
    };
    let before = model.policy.parameters().to_vec();
    for _ in 0..8 {
        old.train_step(
            &CompactValueFeatures::extract(&p, p.to_move()),
            ex.value,
            0.01,
            0.,
        )
        .unwrap();
        model.train(&ex, 0.01).unwrap();
    }
    assert_eq!(model.value.weights(), old.weights());
    assert_ne!(model.policy.parameters(), before);
    assert!(model.policy_bias(&p, &legal).unwrap().is_some());
    let value = model.value.weights().to_vec();
    let policy = model.policy.parameters().to_vec();
    assert!(model.train(&ex, f64::NAN).is_err());
    assert_eq!(model.value.weights().as_slice(), value);
    assert_eq!(model.policy.parameters(), policy);
}
#[test]
fn value_only_targets_do_not_change_the_policy() {
    let mut model = Gen32Model::from_gen31(CompactValueModel::default(), 1);
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
        value_weight: 1.0, sequence_source: 0,
        state: micro_state_features(&p).to_vec(),
        actions: vec![],
        policy: vec![],
        value: -1.,
        policy_weight: 0.,
    };
    let before = model.policy.parameters().to_vec();
    model.train(&ex, 0.01).unwrap();
    assert_eq!(model.policy.parameters(), before);
}
