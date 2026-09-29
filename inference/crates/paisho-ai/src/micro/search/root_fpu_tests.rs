//! Diagnostic opt-in contract, not a claim about learned-value calibration.
use super::*;
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn record() -> paisho_core::GameRecord {
    include_str!("../../../tests/fixtures/root-fpu-positive-arity.psr")
        .parse()
        .unwrap()
}
fn constant(value: f64) -> Arc<MicroModel> {
    let mut w = vec![0.; MICRO_DEEP_VALUE_PARAMETERS];
    w[MICRO_DEEP_VALUE_PARAMETERS - 1] = value.atanh();
    Arc::new(MicroModel::from_parameters(w).unwrap())
}
fn options(proof_search: bool) -> MicroSearchOptions {
    MicroSearchOptions {
        proof_search,
        seed: 37,
        gumbel_scale: 0.,
        ..Default::default()
    }
}
fn configured(model: Arc<MicroModel>, active: bool, beta: f64) -> MicroMctsSession {
    let mut s = MicroMctsSession::new(model);
    s.set_root_value_strength(beta).unwrap();
    s.set_root_successor_fpu(active);
    s
}
fn expected_q(model: &MicroModel, p: &Position, action: Action) -> f64 {
    let mut next = p.clone();
    next.apply(action).unwrap();
    match next.outcome() {
        GameOutcome::Win(w) => {
            if w == p.to_move() {
                1.
            } else {
                -1.
            }
        }
        GameOutcome::Draw => 0.,
        GameOutcome::Ongoing => {
            model.embed(&model.state_features(&next)).value
                * if next.to_move() == p.to_move() {
                    1.
                } else {
                    -1.
                }
        }
    }
}
#[test]
fn diagnostic_disabled_and_beta_zero_preserve_native_reports() {
    let p = record().replay().unwrap();
    let model = constant(0.4);
    let mut a = MicroMctsSession::new(model.clone());
    a.set_root_value_strength(16.).unwrap();
    let mut b = configured(model.clone(), false, 16.);
    for budget in [8, 32] {
        let x = a
            .search_with_options(&p, budget, None, options(false))
            .unwrap();
        let y = b
            .search_with_options(&p, budget, None, options(false))
            .unwrap();
        successor_reuse_tests::equal(&x, &y);
        assert_eq!(x.retained_bytes, y.retained_bytes);
        assert!(b.root_successor_fpu_values().is_none());
    }
    let mut a = configured(model.clone(), false, 0.);
    let mut b = configured(model, true, 0.);
    let x = a.search_with_options(&p, 8, None, options(false)).unwrap();
    let y = b.search_with_options(&p, 8, None, options(false)).unwrap();
    successor_reuse_tests::equal(&x, &y);
    assert_eq!(x.retained_bytes, y.retained_bytes);
    assert!(b.root_successor_fpu_values().is_none());
}
#[test]
fn cached_values_have_bonus_perspective_and_survive_only_the_same_root() {
    let r = record();
    let bonus = r.replay().unwrap();
    let mut before = r.initial_position();
    for a in &r.actions()[..r.actions().len() - 1] {
        before.apply(*a).unwrap();
    }
    assert_eq!(before.to_move(), bonus.to_move());
    let model = constant(0.4);
    for p in [before, bonus] {
        let mut s = configured(model.clone(), true, 16.);
        let first = s.search_with_options(&p, 1, None, options(false)).unwrap();
        let q = s.root_successor_fpu_values().unwrap().to_vec();
        assert_eq!(q.len(), first.actions.len());
        for (a, v) in first.actions.iter().zip(&q) {
            assert_eq!(v.to_bits(), expected_q(&model, &p, *a).to_bits());
        }
        let mut reference = configured(model.clone(), false, 16.);
        let old = reference
            .search_with_options(&p, 1, None, options(false))
            .unwrap();
        assert_eq!(old.inference_evaluations, first.inference_evaluations);
        assert_eq!(bits(&old.priors), bits(&first.priors));
        s.set_root_successor_fpu(true); // An unchanged opt-in does not destroy reuse.
        let second = s.search_with_options(&p, 1, None, options(false)).unwrap();
        assert_eq!(second.inherited_visits, 1);
        assert_eq!(bits(&q), bits(s.root_successor_fpu_values().unwrap()));
        let action = *first
            .actions
            .iter()
            .find(|a| {
                let mut n = p.clone();
                n.apply(**a).unwrap();
                n.outcome() == GameOutcome::Ongoing
            })
            .unwrap();
        let mut next = p.clone();
        next.apply(action).unwrap();
        s.advance(action).unwrap();
        assert!(s.root_successor_fpu_values().is_none());
        let third = s
            .search_with_options(&next, 1, None, options(false))
            .unwrap();
        for (a, v) in third
            .actions
            .iter()
            .zip(s.root_successor_fpu_values().unwrap())
        {
            assert_eq!(v.to_bits(), expected_q(&model, &next, *a).to_bits());
        }
        s.set_root_successor_fpu(false);
        assert_eq!(s.retained_visits(), 0);
        assert!(s.root_successor_fpu_values().is_none());
    }
}
#[test]
fn immediate_proof_priority_does_not_use_estimated_fpu() {
    let p = record().replay().unwrap();
    let model = constant(0.6);
    let mut a = configured(model.clone(), false, 16.);
    let mut b = configured(model, true, 16.);
    let x = a.search_with_options(&p, 8, None, options(true)).unwrap();
    let y = b.search_with_options(&p, 8, None, options(true)).unwrap();
    successor_reuse_tests::equal(&x, &y);
    assert_eq!(x.retained_bytes, y.retained_bytes);
    assert_eq!(y.proven_value, Some(1));
    assert_eq!(y.simulations, 0);
    assert!(b.root_successor_fpu_values().is_none());
    b.certificate(10000).unwrap().verify(&p).unwrap();
}
fn injected(value: f64, active: bool) -> (Position, MicroMctsSession) {
    let p = record().replay().unwrap();
    let model = constant(value);
    let actions: Vec<_> = legal_actions(&p)
        .into_iter()
        .filter(|a| {
            let mut n = p.clone();
            n.apply(*a).unwrap();
            n.outcome() == GameOutcome::Ongoing && n.to_move() != p.to_move()
        })
        .collect();
    assert!(
        actions.len() > 512,
        "controlled arity must leave unvisited alternatives"
    );
    let mut s = configured(model.clone(), active, 16.);
    s.maximum_depth = 1;
    let inf = s.cache.get(p.clone(), &model);
    let mut priors = vec![0.1 / (actions.len() - 1) as f64; actions.len()];
    priors[0] = 0.9;
    let policy = Arc::new(Policy {
        features: Arc::new(
            actions
                .iter()
                .map(|a| micro_action_features(&p, *a))
                .collect(),
        ),
        actions,
        priors,
        log_priors: OnceLock::new(),
    });
    assert!(inf.policy.set(Ok(policy.clone())).is_ok());
    let mut node = Node::new(inf);
    node.memory_policy = Some(policy.clone());
    s.root = Some(node);
    if active {
        s.root_successor_values = Some(
            policy
                .actions
                .iter()
                .map(|a| expected_q(&model, &p, *a))
                .collect(),
        );
    }
    (p, s)
}
#[test]
fn flat_zero_and_positive_player_values_expose_first_play_inconsistency() {
    for value in [0., 0.4, 0.6] {
        let (p, mut old) = injected(value, false);
        let (_, mut new) = injected(value, true);
        let a = old
            .search_with_options(&p, 512, None, options(false))
            .unwrap();
        let b = new
            .search_with_options(&p, 512, None, options(false))
            .unwrap();
        assert_eq!(a.priors[0], 0.9);
        assert_eq!(b.priors[0], 0.9);
        assert!(b.policy_target[0] > 0.85);
        if value == 0. {
            successor_reuse_tests::equal(&a, &b);
        } else {
            assert!(a.policy_target[0] < 0.2);
            let expected = 1.5 * (512f64).sqrt() * 0.9 / (2. * value);
            assert!((a.visits[0] as f64 - expected).abs() < 3.);
        }
        println!("constant-player-value={value} legal={} parent-FPU={} successor-FPU={} n_old={} n_new={}",
            a.actions.len(),a.policy_target[0],b.policy_target[0],a.visits[0],b.visits[0]);
    }
    // Constant positive estimates from alternating players violate the Bellman
    // equation. This isolates a possible estimator mismatch, not its frequency.
}
#[test]
fn first_selection_uses_per_action_q_only_when_enabled() {
    let (p, mut old) = injected(0.4, false);
    let (_, mut new) = injected(0.4, true);
    let q = new.root_successor_values.as_mut().unwrap();
    q.fill(-0.9);
    q[1] = 0.9;
    let a = old
        .search_with_options(&p, 1, None, options(false))
        .unwrap();
    let b = new
        .search_with_options(&p, 1, None, options(false))
        .unwrap();
    assert_eq!(a.new_visits[0], 1);
    assert_eq!(b.new_visits[1], 1);
    assert!(b.proven_value.is_none());
    assert!(b.proven_action_values.iter().all(Option::is_none));
}
#[test]
fn importing_a_verified_proof_invalidates_only_the_old_root_fpu_cache() {
    let p = record().replay().unwrap();
    let model = constant(0.4);
    let mut prover = configured(model.clone(), false, 16.);
    prover
        .search_with_options(&p, 8, None, options(true))
        .unwrap();
    let cert = prover.certificate(10000).unwrap();
    let mut s = configured(model, true, 16.);
    s.search_with_options(&p, 1, None, options(false)).unwrap();
    assert!(s.root_successor_fpu_values().is_some());
    s.install_certificate(&p, &cert).unwrap();
    assert!(s.root_successor_fpu_values().is_none());
    let r = s.search_with_options(&p, 8, None, options(true)).unwrap();
    assert_eq!(r.proven_value, Some(1));
    assert_eq!(r.simulations, 0);
    assert!(s.root_successor_fpu_values().is_none());
    s.certificate(10000).unwrap().verify(&p).unwrap();
}

#[test]
fn end_of_search_memory_eviction_clears_successor_values_before_next_root() {
    let p = record().replay().unwrap();
    let model = constant(0.4);
    let mut session = configured(model.clone(), true, 16.);
    // Set the limit while empty: no preceding tree exists to evict in set_limits.
    assert_eq!(session.retained_bytes(), 0);
    session.set_limits(65536, 1);
    assert!(session.root.is_none());
    assert!(session.root_successor_fpu_values().is_none());
    let evicted = session
        .search_with_options(&p, 1, None, options(false))
        .unwrap();
    assert_eq!(evicted.inherited_visits, 0);
    assert_eq!(evicted.simulations, 1);
    assert!(evicted.retained_bytes > 1);
    assert!(evicted.memory_reset);
    assert!(session.root.is_none());
    assert_eq!(session.cache.count, 0);
    assert_eq!(session.retained_visits(), 0);
    assert!(session.root_successor_fpu_values().is_none());
    assert_eq!(session.retained_bytes(), 0);

    session.set_limits(65536, 512 * 1024 * 1024);
    let fresh = session
        .search_with_options(&p, 1, None, options(false))
        .unwrap();
    assert_eq!(fresh.inherited_visits, 0);
    assert!(!fresh.memory_reset);
    let q = session.root_successor_fpu_values().unwrap();
    assert_eq!(q.len(), fresh.actions.len());
    for (a, v) in fresh.actions.iter().zip(q) {
        assert_eq!(v.to_bits(), expected_q(&model, &p, *a).to_bits());
    }
    assert_eq!(evicted.selected_index, fresh.selected_index);
    assert_eq!(evicted.visits, fresh.visits);
    assert_eq!(bits(&evicted.priors), bits(&fresh.priors));
    assert_eq!(bits(&evicted.values), bits(&fresh.values));
}
