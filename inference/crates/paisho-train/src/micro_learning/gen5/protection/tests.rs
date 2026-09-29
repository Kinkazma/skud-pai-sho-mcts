use super::*;

fn value_example(value: f64) -> Arc<MicroExample> {
    Arc::new(MicroExample { policy_support: false,
        state: vec![0.; 128], actions: vec![], policy: vec![], action_values: vec![],
        policy_weight: 0., value_weight: 1., value, sequence_source: 0,
    })
}
fn zero_model() -> MicroModel {
    MicroModel::from_parameters(vec![0.; MicroModel::seeded(1).parameters().len()]).unwrap()
}
fn same_bits(a: &MicroModel, b: &MicroModel) {
    assert!(a.parameters().iter().zip(b.parameters()).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn finite_rhs_aims_inside_the_nonnegative_acceptance_interval() {
    let limits = [0., 5e-12, 0.1, 1e-10];
    let current = [0., 5e-12, 0.2, 1e-10];
    let rhs = repair_rhs(&current, &limits);
    for i in [0, 1, 3] { assert!(rhs[i] < 0., "satisfied class must retain slack"); }
    assert!((rhs[2] - (0.1 - 0.5 * FINITE_LOSS_TOLERANCE)).abs() < 1e-16);
    assert_eq!(violation_merit(&limits, &limits, &[], &[], false), 0.);
    let inside = limits.map(|v| v + 0.75 * FINITE_LOSS_TOLERANCE);
    assert_eq!(violation_merit(&inside, &limits, &[], &[], false), 0.);
}

#[test]
fn interior_target_repairs_the_recorded_one_bit_boundary_excess() {
    let before: f64 = 0.023118107457977623;
    let after: f64 = 0.023118108457977626;
    let boundary = before + FINITE_LOSS_TOLERANCE;
    assert!(after > boundary);
    assert!(after - boundary < 4e-18);
    let rhs = repair_rhs(&[after, 0., 0., 0.], &[before, 0., 0., 0.]);
    assert!(rhs[0] > 4.99e-10);
    let correction = projection::affine(&[0.], &[vec![1.]], &[rhs[0]]).unwrap()[0];
    assert!(after - correction < boundary);
    assert!(after - correction >= 0.);
    // The same outer tolerance remains enforced; it has not been enlarged.
    assert!(violation_merit(&[after, 0., 0., 0.], &[before, 0., 0., 0.], &[], &[], false) > 0.);
}

#[test]
fn native_squared_value_loss_reaches_the_interior_without_relaxing_acceptance() {
    let mut parameters = zero_model().parameters().to_vec();
    parameters[4160] = (2. * 0.023118107457977623_f64).sqrt().atanh();
    let anchor = MicroModel::from_parameters(parameters.clone()).unwrap();
    let rows = vec![value_example(0.)];
    let mut p = Protection::new(&anchor, rows.clone()).unwrap();
    parameters[4160] = (anchor.value(&rows[0].state) + 1e-8).atanh();
    let mut candidate = MicroModel::from_parameters(parameters).unwrap();
    assert!(losses(&candidate, &rows, None).unwrap().0[2] > p.anchor_losses[2] + FINITE_LOSS_TOLERANCE);
    p.consolidate(&mut candidate).unwrap();
    assert_eq!(p.last["accepted"], true);
    let previous = p.last["reference_before"][2].as_f64().unwrap();
    assert!(losses(&candidate, &rows, None).unwrap().0[2] < previous + FINITE_LOSS_TOLERANCE);
    assert!(p.last["iterations"].as_u64().unwrap() <= 2);
}

#[test]
fn failed_weight_transaction_preserves_anchor_data_and_resume() {
    let anchor = zero_model();
    let refs = vec![value_example(0.)];
    let fresh = vec![value_example(1.)];
    let mut model = anchor.clone();
    let mut p = Protection::new(&anchor, refs.clone()).unwrap();
    p.observe(&fresh);
    let initial_gradients = p.gradients.clone();
    // These labels deliberately conflict at the identical input: retaining 5%
    // of the fresh improvement cannot satisfy the established zero-value proof.
    for expected_updates in 1..=2 {
        p.train_shared(&mut model, &fresh, 0.1, 0.).unwrap();
        assert!(model.value(&fresh[0].state) > 0.09);
        p.consolidate(&mut model).unwrap();
        assert_eq!(p.last["accepted"], false);
        assert_eq!(p.last["rolled_back"], true);
        assert_eq!(p.last["fresh_after_measured"], false);
        assert!(p.last["fresh_after"].is_null());
        assert_eq!(p.updates, expected_updates);
        assert_eq!(p.fresh.len(), 1);
        same_bits(&model, &anchor);
        same_bits(&p.anchor, &anchor);
        assert_eq!(p.gradients, initial_gradients);
    }
    let directory = std::env::temp_dir().join(format!("paisho-protection-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    fs::create_dir(&directory).unwrap();
    p.checkpoint(&directory).unwrap();
    let mut restored = Protection::new(&model, refs).unwrap();
    restored.restore(&p.progress()).unwrap();
    assert_eq!(restored.updates, 2);
    assert_eq!(restored.checks, 2);
    assert_eq!(restored.fresh.len(), 1);
    same_bits(&restored.anchor, &anchor);
    let mut resumed = model.clone();
    p.train_shared(&mut model, &fresh, 0.001, 1e-6).unwrap();
    restored.train_shared(&mut resumed, &fresh, 0.001, 1e-6).unwrap();
    same_bits(&model, &resumed);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn accepted_anchor_rebuild_keeps_presentations_and_recent_data() {
    let model = zero_model();
    let rows = vec![value_example(1.)];
    let mut p = Protection::new(&model, rows.clone()).unwrap();
    p.observe(&rows);
    let mut accepted = model.clone();
    // Along the proof gradient there is no conflict and its loss decreases.
    p.train_shared(&mut accepted, &rows, 0.01, 0.).unwrap();
    p.consolidate(&mut accepted).unwrap();
    assert_eq!(p.last["accepted"], true);
    assert_eq!(p.last["fresh_after_measured"], true);
    assert!(p.last["fresh_after"].is_number());
    assert!(p.anchor_losses[3] < 0.5);
    let counters = (p.updates, p.checks, p.accepted, p.fresh.len(), p.reference_evaluations);
    p.adopt_accepted(&accepted).unwrap();
    assert_eq!((p.updates, p.checks, p.accepted, p.fresh.len(), p.reference_evaluations), counters);
    same_bits(&p.anchor, &accepted);
}

#[test]
fn v3_confidence_is_not_a_finite_ratchet_when_decisions_and_values_are_retained() {
    let mut anchor = zero_model();
    let mut proof = (*value_example(1.)).clone();
    proof.actions = vec![[0.;32], [1.;32]];
    proof.policy = vec![1., 0.];
    proof.policy_weight = 1.; proof.value_weight = 0.;
    for _ in 0..8 { anchor.train_batch_inline(&[&proof], 0.01, 0.).unwrap(); }
    assert_eq!(best(&prior(&anchor, &proof).unwrap()), 0);
    let mut other = proof.clone(); other.policy = vec![0., 1.];
    let mut candidate = anchor.clone();
    candidate.train_batch_inline(&[&other], 0.001, 0.).unwrap();
    assert_eq!(best(&prior(&candidate, &proof).unwrap()), 0);
    let rows = vec![Arc::new(proof)];
    let mut p = Protection::new(&anchor, rows.clone()).unwrap();
    assert!(losses(&candidate, &rows, None).unwrap().0[0] > p.anchor_losses[0] + FINITE_LOSS_TOLERANCE);
    p.enable_loop_v3();
    p.observe(&[Arc::new(other)]);
    let expected = candidate.clone();
    p.consolidate(&mut candidate).unwrap();
    assert_eq!(p.last["accepted"], true);
    assert_eq!(p.last["iterations"], 0);
    same_bits(&candidate, &expected);
}

#[test]
fn diagnostic_policy_halfspace_can_remove_a_safe_improvement_on_another_input() {
    let mut parameters = MicroModel::seeded(91).parameters().to_vec();
    // Isolate policy learning: both states have a constant value, including
    // after policy-only updates of this legacy model's shared hidden layer.
    parameters[4128..4161].fill(0.);
    let mut anchor = MicroModel::from_parameters(parameters).unwrap();
    let mut proof = (*value_example(1.)).clone();
    proof.actions = vec![[0.; 32], [1.; 32]];
    proof.policy = vec![1., 0.];
    proof.policy_weight = 1.; proof.value_weight = 0.;
    for _ in 0..8 { anchor.train_batch_inline(&[&proof], 0.01, 0.).unwrap(); }
    let before = prior(&anchor, &proof).unwrap();
    assert_eq!(best(&before), 0);
    let mut fresh = proof.clone();
    fresh.state[0] = 1.;
    fresh.policy = vec![0., 1.];
    let fresh = Arc::new(fresh);
    let rows = vec![Arc::new(proof)];
    let mut outcomes = Vec::new();
    for policy_constraint in [true, false] {
        let mut protection = Protection::new(&anchor, rows.clone()).unwrap();
        protection.enable_loop_v3();
        protection.diagnostic_set_policy_gradient_constraint(policy_constraint);
        let mut model = anchor.clone();
        protection.train_shared(&mut model, &[fresh.clone()], 0.0005, 0.).unwrap();
        let after = prior(&model, &rows[0]).unwrap();
        assert_eq!(best(&after), 0, "old winning action remains first");
        let kl = before.iter().zip(&after).map(|(p,q)|p*(p.ln()-q.ln())).sum::<f64>();
        assert!(kl < 0.001);
        assert_eq!(model.value(&rows[0].state), anchor.value(&rows[0].state));
        outcomes.push(model.loss_loop_v3(&fresh).unwrap().total(1.));
    }
    let original = anchor.loss_loop_v3(&fresh).unwrap().total(1.);
    assert!(outcomes[1] < original);
    assert!(outcomes[1] + 1e-8 < outcomes[0], "the old confidence halfspace removed useful gradient");
}
