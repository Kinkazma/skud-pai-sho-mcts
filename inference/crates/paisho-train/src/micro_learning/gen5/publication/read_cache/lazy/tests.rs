use super::*;

fn model() -> MicroModel {
    let model = MicroModel::seeded(23).with_deep_value(27);
    let mut weights = model.parameters().to_vec();
    for (i, w) in weights.iter_mut().enumerate() {
        if branches::value_parameter(i) {
            *w = 0.;
        }
    }
    MicroModel::from_parameters(weights).unwrap()
}
fn values(
    model: &MicroModel,
    q: &[Option<f64>],
    terminal: &[bool],
    bounded: bool,
) -> Arc<ValueRow> {
    let position = Position::from_standard_setup(paisho_core::StandardSetup::balanced(
        paisho_core::BASIC_FLOWERS[0],
    ));
    let example = MicroExample {
        state: vec![0.; MICRO_SPATIAL_INPUTS],
        actions: vec![[0.; 32]; q.len()],
        policy: vec![1. / q.len() as f64; q.len()],
        value: 1.,
        policy_weight: 1.,
        value_weight: 0.,
        action_values: vec![],
        sequence_source: 0,
        policy_support: false,
    };
    let witness = Arc::new(Witness {
        position,
        valid: terminal.to_vec(),
        example: Arc::new(example),
        successors: Default::default(),
    });
    witness
        .successors
        .set(
            q.iter()
                .zip(terminal)
                .map(|(q, t)| {
                    if *t {
                        (q.unwrap(), vec![])
                    } else {
                        (1., vec![1.; MICRO_SPATIAL_INPUTS])
                    }
                })
                .collect(),
        )
        .unwrap();
    let row = Arc::new(ValueRow::new(
        witness,
        model,
        bounded,
        Arc::new(AtomicUsize::new(0)),
    ));
    for (i, value) in q.iter().enumerate() {
        if value.is_some() {
            row.values.lock().unwrap()[i] = *value;
        }
    }
    row
}
fn logp(p: &[f64]) -> Vec<f64> {
    p.iter().map(|p| p.max(1e-300).ln()).collect()
}
fn same_bits(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn terminal_bound_skips_reads_and_complete_logits_remain_exact() {
    let m = model();
    let q = values(&m, &[Some(1.), None, None], &[true, false, false], true);
    let logits = CoupledLogits::pending(logp(&[0.9, 0.05, 0.05]), 16., q.clone(), true);
    assert_eq!(logits.winner(&[true, false, false]), 0);
    assert!(logits.zero_hinge(&[true, false, false], 1e-6));
    assert_eq!(q.loaded(), 1);
    let expected = vec![
        add_value(0.9f64.ln(), 16., 1.),
        add_value(0.05f64.ln(), 16., 0.),
        add_value(0.05f64.ln(), 16., 0.),
    ];
    same_bits(&logits, &expected);
    assert_eq!(q.loaded(), 3);
}

#[test]
fn changed_policy_fills_previously_unread_cells_without_stale_fabricated_values() {
    let m = model();
    let q = values(&m, &[Some(1.), None, None], &[true, false, false], true);
    let a = CoupledLogits::pending(logp(&[0.9, 0.05, 0.05]), 16., q.clone(), true);
    assert_eq!(a.winner(&[true, false, false]), 0);
    assert_eq!(q.loaded(), 1);
    let b = CoupledLogits::pending(logp(&[0.01, 0.98, 0.01]), 16., q.clone(), true);
    assert_eq!(b.winner(&[true, false, false]), 0);
    assert_eq!(q.loaded(), 2);
    let c = CoupledLogits::pending(logp(&[0.01, 0.01, 0.98]), 16., q.clone(), true);
    assert_eq!(c.winner(&[true, false, false]), 0);
    assert_eq!(q.loaded(), 3);
    for logits in [&a, &b, &c] {
        assert_eq!(logits.winner(&[true, false, false]), best_full(logits));
    }
}

#[test]
fn floored_priors_and_lower_index_ties_use_original_total_order() {
    let m = model();
    let p = logp(&[0., 1e-320, 1e-300]);
    assert_eq!(p[0].to_bits(), p[2].to_bits());
    let q = values(
        &m,
        &[Some(1.), Some(1.), Some(1.)],
        &[false, false, true],
        true,
    );
    let logits = CoupledLogits::pending(p, 16., q, true);
    assert_eq!(logits.winner(&[false, false, true]), 0);
    assert_eq!(logits.winner(&[false, false, true]), best_full(&logits));
    assert!(!better(0, -0., 1, 0.));
    assert!(better(1, 0., 0, -0.));
    assert!(better(0, 1., 1, 1.));
}

#[test]
fn zero_negative_and_nonfinite_beta_preserve_full_evaluation() {
    let m = model();
    for beta in [0., -1., 17., f64::INFINITY, f64::NAN] {
        let q = values(&m, &[Some(1.), None], &[true, false], true);
        let logits = CoupledLogits::pending(logp(&[0.9, 0.1]), beta, q.clone(), true);
        assert_eq!(logits.winner(&[true, false]), best_full(&logits));
        assert_eq!(q.loaded(), 2);
        assert!(!logits.zero_hinge(&[true, false], 1e-6));
    }
}

#[test]
fn finite_but_overflowing_value_parameters_are_not_hidden_by_pruning() {
    let mut w = model().parameters().to_vec();
    for x in &mut w[MICRO_VALUE_TRUNK..MICRO_VALUE_TRUNK + MICRO_INPUTS] {
        *x = f64::MAX;
    }
    for x in &mut w[MICRO_VALUE_BOARD..MICRO_VALUE_BOARD + MICRO_BOARD_INPUTS] {
        *x = -f64::MAX;
    }
    let m = MicroModel::from_parameters(w).unwrap();
    assert!(!bounded_weights(&m));
    let expected = m.value(&vec![1.; MICRO_SPATIAL_INPUTS]);
    assert!(expected.is_nan());
    let q = values(&m, &[Some(1.), None], &[true, false], bounded_weights(&m));
    let logits = CoupledLogits::pending(logp(&[0.99, 0.01]), 16., q.clone(), true);
    let old = vec![
        add_value(0.99f64.ln(), 16., 1.),
        add_value(0.01f64.ln(), 16., expected),
    ];
    assert_eq!(logits.winner(&[true, false]), best_full(&old));
    same_bits(&logits, &old);
    assert_eq!(q.loaded(), 2);
    assert!(!logits.zero_hinge(&[true, false], 1e-6));
}

#[test]
fn input_domain_and_hinge_boundary_are_conservative() {
    assert!(bounded_inputs(&[(1., vec![1e100; MICRO_SPATIAL_INPUTS])]));
    for x in [1e101, f64::INFINITY, f64::NAN] {
        assert!(!bounded_inputs(&[(1., vec![x; MICRO_SPATIAL_INPUTS])]));
    }
    assert!(!bounded_inputs(&[(1., vec![0.; 5])]));
    let m = model();
    let q = values(&m, &[Some(1.), None], &[true, false], true);
    let logits = CoupledLogits::pending(vec![-1., -2.], 16., q, true);
    // upper_bad=14, lower_good=15: at margin1, retain the original expression.
    assert!(!logits.zero_hinge(&[true, false], 1.));
    assert!(logits.zero_hinge(&[true, false], 0.5));
    assert!(!logits.zero_hinge(&[true, false], f64::NAN));
}

#[test]
fn cache_key_reuses_policy_variants_but_invalidates_changed_value_weights() {
    let m = model();
    let row = values(&m, &[Some(1.), None], &[true, false], true);
    let rows = vec![row.witness.clone()];
    let reads = Reads::new(true);
    let a = reads.evaluate(&rows, &m, 16., None).unwrap();
    assert!(a.coupled[0]);
    let first = reads.values.lock().unwrap()[0].1.clone();
    let mut w = m.parameters().to_vec();
    w[0] += 0.125;
    let policy = MicroModel::from_parameters(w).unwrap();
    reads.evaluate(&rows, &policy, 16., None).unwrap();
    assert_eq!(reads.values.lock().unwrap().len(), 1);
    assert!(Arc::ptr_eq(&first, &reads.values.lock().unwrap()[0].1));
    let mut w = m.parameters().to_vec();
    w[MICRO_VALUE_TRUNK] += 0.125;
    let value = MicroModel::from_parameters(w).unwrap();
    reads.evaluate(&rows, &value, 16., None).unwrap();
    assert_eq!(reads.values.lock().unwrap().len(), 2);
    assert!(!Arc::ptr_eq(&first, &reads.values.lock().unwrap()[1].1));
}


#[test]
fn scheduling_metadata_never_dereferences_pending_logits() {
    let m = model();
    let q = values(&m, &[Some(1.), None, None], &[true, false, false], true);
    let logits = CoupledLogits::pending(logp(&[0.01, 0.98, 0.01]), 16., q.clone(), true);
    assert_eq!(logits.action_count(), 3);
    assert!(!logits.is_materialized());
    assert_eq!(q.loaded(), 1);
    assert_eq!(q.forward_reads.load(Ordering::Relaxed), 0);
    logits.materialize();
    assert!(logits.is_materialized());
    assert_eq!(logits.action_count(), 3);
    assert_eq!(q.forward_reads.load(Ordering::Relaxed), 2);
    logits.materialize();
    assert_eq!(q.forward_reads.load(Ordering::Relaxed), 2);
    let complete = CoupledLogits::from(vec![-0., 1.]);
    assert!(complete.is_materialized());
    assert_eq!(complete.action_count(), 2);
    same_bits(&complete, &[-0., 1.]);
}

#[test]
fn eager_backend_does_not_consume_prefill_selection() {
    for reads in [Reads::original(), Reads::new(false)] {
        reads.prefill(std::iter::from_fn(|| -> Option<&CoupledLogits> {
            panic!("eager prefill must not inspect the iterator")
        }), None);
    }
}

#[test]
fn selected_prefill_matches_serial_bits_and_keeps_zero_hinges_cold() {
    let m = model();
    let valid = [true, false, false];
    let priors = [[0.9, 0.05, 0.05], [0.01, 0.98, 0.01], [0.02, 0.03, 0.95]];
    let serial_values: Vec<_> = priors.iter().map(|_| {
        values(&m, &[Some(1.), None, None], &valid, true)
    }).collect();
    let parallel_values: Vec<_> = priors.iter().map(|_| {
        values(&m, &[Some(1.), None, None], &valid, true)
    }).collect();
    let serial: Vec<_> = priors.iter().zip(&serial_values).map(|(p, q)| {
        CoupledLogits::pending(logp(p), 16., q.clone(), true)
    }).collect();
    let parallel: Vec<_> = priors.iter().zip(&parallel_values).map(|(p, q)| {
        CoupledLogits::pending(logp(p), 16., q.clone(), true)
    }).collect();
    let pool = cpu::Ordered::new(&[Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap(),
    )]);
    let reads = Reads::new(true);
    for (logits, pool) in [(&serial, None), (&parallel, Some(&pool))] {
        reads.prefill(logits.iter().filter(|v| !v.zero_hinge(&valid, 1e-6)), pool);
        assert!(!logits[0].is_materialized());
        assert!(logits[1].is_materialized());
        assert!(logits[2].is_materialized());
    }
    for rows in [&serial_values, &parallel_values] {
        assert_eq!(rows[0].loaded(), 1);
        assert_eq!(rows[0].forward_reads.load(Ordering::Relaxed), 0);
        for row in &rows[1..] {
            assert_eq!(row.loaded(), 3);
            assert_eq!(row.forward_reads.load(Ordering::Relaxed), 2);
        }
    }
    for i in 1..3 {
        same_bits(&serial[i], &parallel[i]);
    }
    reads.prefill(parallel[1..].iter(), Some(&pool));
    for row in &parallel_values[1..] {
        assert_eq!(row.forward_reads.load(Ordering::Relaxed), 2);
    }
}
