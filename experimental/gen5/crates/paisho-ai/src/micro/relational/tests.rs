use super::*;
use paisho_core::*;
fn position() -> Position {
    let r: GameRecord = include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr")
        .parse()
        .unwrap();
    let mut p = r.initial_position();
    for &a in &r.actions()[..r.actions().len() - 1] {
        p.apply(a).unwrap();
    }
    p
}
fn example(m: &MicroModel) -> MicroExample {
    let p = position();
    let actions: Vec<_> = legal_actions(&p).into_iter().take(4).collect();
    let before = MicroRelations::extract(&p, p.to_move());
    let structured = actions
        .iter()
        .map(|&a| {
            let mut q = p.clone();
            q.apply(a).unwrap();
            Some(MicroStructuredTarget::from_successor(
                &p,
                a,
                &q,
                &before,
                &micro_immediate_threat(&q, p.to_move(), 0).unwrap(),
            ))
        })
        .collect();
    MicroExample {
        structured,
        policy_support: false,
        action_values: vec![None; actions.len()],
        sequence_source: u64::MAX,
        value_weight: 1.,
        state: m.state_features(&p),
        actions: actions
            .iter()
            .map(|&a| micro_action_features(&p, a))
            .collect(),
        policy: vec![0., 1., 0., 0.],
        value: 0.7,
        policy_weight: 1.,
    }
}
#[test]
fn neutral_migration_keeps_values_priors_and_old_gradients() {
    let old = MicroModel::seeded(11).with_neural_memory(19);
    let new = old.with_relational(23);
    assert_eq!(new.schema(), MICRO_RELATIONAL_MODEL_SCHEMA);
    assert_eq!(new.parameters.len(), MICRO_RELATIONAL_PARAMETERS);
    assert!(new.parameters[..old.parameters.len()]
        .iter()
        .zip(old.parameters())
        .all(|(a, b)| a.to_bits() == b.to_bits()));
    assert_eq!(new, new.with_relational(999));
    let mut ex = example(&new);
    ex.structured.clear();
    for model in [&old, &new] {
        let (v, p) = model
            .policy_value_priors(&ex.state, &ex.actions, u64::MAX)
            .unwrap();
        let (base_v, base) = old
            .policy_value_priors(&ex.state, &ex.actions, u64::MAX)
            .unwrap();
        assert_eq!(v.to_bits(), base_v.to_bits());
        assert_eq!(model.value(&ex.state).to_bits(), base_v.to_bits());
        assert!(p.iter().zip(base).all(|(a, b)| a.to_bits() == b.to_bits()));
        let compact = model
            .memory_priors_compact(
                &ex.state,
                &ex.actions,
                &micro_softmax(&MicroModel::logits(&model.embed(&ex.state), &ex.actions)).unwrap(),
                u64::MAX,
            )
            .unwrap();
        assert_eq!(p, compact);
    }
    let (a, ga) = old.loss_gradient(&ex).unwrap();
    let (b, gb) = new.loss_gradient(&ex).unwrap();
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    assert_eq!(a.policy.to_bits(), b.policy.to_bits());
    assert!(ga.iter().zip(gb).all(|(a, b)| a.to_bits() == b.to_bits()));
}
#[test]
fn full_joint_gradient_includes_policy_auxiliary_and_shared_value() {
    let m = MicroModel::seeded(11).with_relational(23);
    let mut w = m.parameters().to_vec();
    let mut rng = StableRng::new(7);
    for x in &mut w[MICRO_RELATIONAL_START + OW..MICRO_RELATIONAL_START + QUERY] {
        *x = (rng.next_f64() - 0.5) * 0.1;
    }
    for x in &mut w[MICRO_RELATIONAL_START + VALUE..] {
        *x = (rng.next_f64() - 0.5) * 0.1;
    }
    for x in &mut w[MICRO_NEURAL_MEMORY_PARAMETERS - 262..MICRO_NEURAL_MEMORY_PARAMETERS] {
        *x = (rng.next_f64() - 0.5) * 0.02;
    }
    let model = MicroModel::from_parameters(w.clone()).unwrap();
    let ex = example(&model);
    let (_, g) = model.loss_gradient(&ex).unwrap();
    let eps = 1e-5;
    let mut indices = vec![VALUE_W, VALUE_B];
    for (start, end) in [
        (0, HW),
        (HW, HB),
        (HB, OW),
        (OW, OB),
        (OB, QUERY),
        (QUERY, VALUE),
        (VALUE, MICRO_RELATIONAL_WEIGHTS),
    ] {
        for i in [start, (start + end) / 2, end - 1] {
            indices.push(MICRO_RELATIONAL_START + i);
        }
    }
    for i in indices {
        let original = w[i];
        w[i] = original + eps;
        let hi = MicroModel::from_parameters(w.clone())
            .unwrap()
            .loss(&ex)
            .unwrap()
            .total(ex.policy_weight);
        w[i] = original - eps;
        let lo = MicroModel::from_parameters(w.clone())
            .unwrap()
            .loss(&ex)
            .unwrap()
            .total(ex.policy_weight);
        w[i] = original;
        assert!(
            ((hi - lo) / (2. * eps) - g[i]).abs() < 3e-7,
            "parameter {i}: numeric {}, analytic {}",
            (hi - lo) / (2. * eps),
            g[i]
        );
    }
    let frozen: Vec<_> = w
        .iter()
        .enumerate()
        .filter(|(i, _)| micro_relational_value_parameter(*i))
        .map(|(_, w)| w.to_bits())
        .collect();
    assert_eq!(frozen.len(), MICRO_GRAPH_PARAMETERS + 33);
}
#[test]
fn spatial_graph_reconstruction_keeps_borrowed_lotus_and_isolated_pieces() {
    let r: GameRecord = include_str!(
        "../../../../paisho-train/examples/gen5_structured_corpus/fixtures/cross_owner_lotus.psr"
    )
    .parse()
    .unwrap();
    let p = r.replay().unwrap();
    let x = micro_spatial_state_features(&p);
    let a = MicroPieceGraph::extract(&p, p.to_move());
    let b = MicroPieceGraph::from_spatial_features(&x).unwrap();
    assert_eq!(
        a.nodes.iter().map(|n| n.features).collect::<Vec<_>>(),
        b.nodes.iter().map(|n| n.features).collect::<Vec<_>>()
    );
    let enc = MicroGraphEncoder::seeded(7);
    let u = enc.encode(&a, MicroGraphPropagation::TwoHarmonyRounds);
    let v = enc.encode(&b, MicroGraphPropagation::TwoHarmonyRounds);
    assert!(u.iter().zip(v).all(|(a, b)| (a - b).abs() < 1e-12));
    assert!(MicroPieceGraph::from_spatial_features(&vec![0.; 128]).is_err());
}
#[test]
fn auxiliary_teaching_survives_zero_value_and_policy_weights_but_not_policy_only_mode() {
    let mut model = MicroModel::seeded(7).with_relational(19);
    let mut ex = example(&model);
    ex.policy_weight = 0.;
    ex.value_weight = 0.;
    ex.action_values.clear();
    let before = model.loss(&ex).unwrap().value;
    let original = model.clone();
    model.train_policy_step(&ex, 0.01).unwrap();
    assert_eq!(model, original);
    for _ in 0..16 {
        model.train_step(&ex, 0.01, 0.).unwrap();
    }
    assert!(model.loss(&ex).unwrap().value < before);
    assert_ne!(
        model.parameters()[MICRO_RELATIONAL_START..],
        original.parameters()[MICRO_RELATIONAL_START..]
    );
    assert_eq!(
        model.parameters()[..MICRO_RELATIONAL_START],
        original.parameters()[..MICRO_RELATIONAL_START]
    );
}
#[test]
fn exact_consequences_keep_proved_threat_when_reanalysis_is_incomplete() {
    let model = MicroModel::seeded(1).with_relational(2);
    let ex = example(&model);
    let mut known = ex.structured[0].clone().unwrap();
    known.events[5] = Some(true);
    known.threat = MicroThreatEvidence::Present {
        reply: "skip-bonus".into(),
    };
    let mut unknown = known.clone();
    unknown.events[5] = None;
    unknown.threat = MicroThreatEvidence::Unknown;
    assert_eq!(known.merge_evidence(&unknown).unwrap(), known);
    let mut opposite = known.clone();
    opposite.events[5] = Some(false);
    opposite.threat = MicroThreatEvidence::Absent { examined: 1 };
    assert!(known.merge_evidence(&opposite).is_err());
    unknown.counts[0] += 1.;
    assert!(known.merge_evidence(&unknown).is_err());
}
#[test]
fn policy_only_inference_matches_all_heads_bit_for_bit() {
    let p = position();
    let actions: Vec<_> = legal_actions(&p)
        .iter()
        .map(|&a| micro_action_features(&p, a))
        .collect();
    for seed in [7, 19, 37] {
        let base = MicroModel::seeded(seed).with_relational(seed + 1);
        let mut w = base.parameters().to_vec();
        let mut rng = StableRng::new(seed + 2);
        for x in &mut w[MICRO_RELATIONAL_START + OW..MICRO_RELATIONAL_START + QUERY] {
            *x = (rng.next_f64() - 0.5) * 0.1;
        }
        let m = MicroModel::from_parameters(w).unwrap();
        let state = m.state_features(&p);
        let reader = m.relational.as_ref().unwrap();
        let full = reader.forward(&state, &actions).unwrap();
        let lean = reader.forward_impl::<false>(&state, &actions).unwrap();
        assert!(lean.hidden.is_empty() && lean.attention.is_empty());
        assert_eq!(
            full.output
                .iter()
                .map(|v| v[0].to_bits())
                .collect::<Vec<_>>(),
            lean.output
                .iter()
                .map(|v| v[0].to_bits())
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn batch_balance_preserves_rare_positive_budget_and_matches_derivatives() {
    let model = MicroModel::seeded(5).with_relational(7);
    let mut negative = example(&model);
    negative.policy_weight = 0.;
    negative.value_weight = 0.;
    for target in negative.structured.iter_mut().flatten() {
        target.events = [None; 10];
        target.events[0] = Some(false);
        target.threat = MicroThreatEvidence::Unknown;
    }
    let mut positive = negative.clone();
    positive.structured[0].as_mut().unwrap().events[0] = Some(true);
    let rows = [positive, negative.clone(), negative.clone(), negative];
    let balance = MicroStructuredBatchBalance::new(rows.iter());
    let mut unknown = rows[0].clone();
    unknown.structured.clear();
    let with_unknown =
        MicroStructuredBatchBalance::new(rows.iter().chain(std::iter::once(&unknown)));
    for j in 0..10 {
        for value in [false, true] {
            assert_eq!(balance.weight(j, value), with_unknown.weight(j, value));
        }
    }
    let cache = model
        .relational
        .as_ref()
        .unwrap()
        .forward(&rows[0].state, &rows[0].actions)
        .unwrap();
    let mut old = 0.;
    let mut positive_sum = 0.;
    let mut negative_sum = 0.;
    for row in &rows {
        let (_, local) = backward::auxiliary(&cache, &row.structured, 1., None);
        old += local.iter().map(|d| d[11]).sum::<f64>() / rows.len() as f64;
        let (_, balanced) = backward::auxiliary(&cache, &row.structured, 1., Some(&balance));
        for (d, t) in balanced.iter().zip(&row.structured) {
            if t.as_ref().unwrap().events[0] == Some(true) {
                positive_sum += d[11] / rows.len() as f64;
            } else {
                negative_sum += d[11] / rows.len() as f64;
            }
        }
    }
    assert!(
        (old - 0.075).abs() < 1e-14,
        "the old average suppresses the rare event"
    );
    assert!((positive_sum + 0.05).abs() < 1e-14);
    assert!((negative_sum - 0.05).abs() < 1e-14);
    let mut weights = model.parameters().to_vec();
    let index = MICRO_RELATIONAL_START + OB + 11;
    weights[index] = 0.3;
    let model = MicroModel::from_parameters(weights.clone()).unwrap();
    let analytic: f64 = rows
        .iter()
        .map(|e| {
            model
                .loss_gradient_structured_batch_reusing(e, &balance, false, Vec::new())
                .unwrap()
                .1[index]
                / 4.
        })
        .sum();
    let mut loss = |offset: f64| {
        weights[index] = 0.3 + offset;
        let m = MicroModel::from_parameters(weights.clone()).unwrap();
        rows.iter()
            .map(|e| m.loss_structured_batch(e, &balance).unwrap().total(0.) / 4.)
            .sum::<f64>()
    };
    assert!(((loss(1e-5) - loss(-1e-5)) / 2e-5 - analytic).abs() < 1e-8);
    let mut serial = model.clone();
    let mut parallel = model;
    let batch = rows.iter().collect::<Vec<_>>();
    serial.train_batch_inline(&batch, 0.01, 0.).unwrap();
    parallel.train_batch(&batch, 0.01, 0.).unwrap();
    assert_eq!(serial.parameters(), parallel.parameters());
}
