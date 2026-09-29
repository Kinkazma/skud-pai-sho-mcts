use paisho_ai::*;
use paisho_core::*;
use std::sync::Arc;

fn aliases() -> Vec<(Vec<f64>, Vec<f64>)> {
    [
        (
            include_str!("fixtures/micro-alias-0-a.psr"),
            include_str!("fixtures/micro-alias-0-b.psr"),
        ),
        (
            include_str!("fixtures/micro-alias-1-a.psr"),
            include_str!("fixtures/micro-alias-1-b.psr"),
        ),
        (
            include_str!("fixtures/micro-alias-2-a.psr"),
            include_str!("fixtures/micro-alias-2-b.psr"),
        ),
    ]
    .iter()
    .map(|(a, b)| {
        let state = |s: &str| {
            micro_spatial_state_features(&s.parse::<GameRecord>().unwrap().replay().unwrap())
        };
        (state(a), state(b))
    })
    .collect()
}
fn entry(state: &[f64], source: u64, game: u32) -> SequenceEntry {
    SequenceEntry {
        key: sequence_key(state[..128].try_into().unwrap()),
        patterns: [[game as i8; 32]; 4],
        source,
        game,
        decision: 1,
        end_decision: 20,
        outcome: (game % 3) as i8 - 1,
        phase: u8::from(state[125] > 0.5),
    }
}
#[test]
fn real_aliases_have_distinct_neighbors_and_cache_entries_with_exact_source_exclusion() {
    for (a, b) in aliases() {
        assert_eq!(&a[..128], &b[..128]);
        let ga = SequenceGeometry::from_state(&a).unwrap();
        let gb = SequenceGeometry::from_state(&b).unwrap();
        assert_ne!(ga, gb);
        assert!(ga.distance(&gb) > 0.);
        assert_eq!(ga.distance(&ga), 0.);
        let rows: Vec<_> = (0..20)
            .map(|i| entry(if i % 2 == 0 { &a } else { &b }, i as u64 + 1, i))
            .collect();
        let geometry = (0..20).map(|i| if i % 2 == 0 { ga } else { gb }).collect();
        let bank = SequenceBank::build_spatial(rows, geometry, 20, 0, 4);
        let ca = bank.context(&a, 1);
        let cb = bank.context(&b, 1);
        assert!(!Arc::ptr_eq(&ca, &cb));
        assert!(Arc::ptr_eq(&ca, &bank.context(&a, 1)));
        assert_eq!(ca.neighbors.len(), 8);
        assert_eq!(cb.neighbors.len(), 8);
        for (context, geometry) in [(&ca, ga), (&cb, gb)] {
            let sources: std::collections::HashSet<_> = context
                .neighbors
                .iter()
                .map(|i| bank.entries[*i].source)
                .collect();
            assert_eq!(sources.len(), 8);
            assert!(!sources.contains(&1));
            assert!(context
                .neighbors
                .iter()
                .all(|i| bank.geometry(*i) == Some(&geometry)));
        }
        let mut bytes = vec![];
        bank.write_to(&mut bytes).unwrap();
        assert_eq!(&bytes[..8], b"PSSEQ002");
        let restored = SequenceBank::read_from(&mut &bytes[..], bank.spec.clone()).unwrap();
        assert_eq!(ca.neighbors, restored.context(&a, 1).neighbors);
        assert_eq!(cb.neighbors, restored.context(&b, 1).neighbors);
        let mut output = vec![];
        restored.write_to(&mut output).unwrap();
        assert_eq!(bytes, output);
        // Geometry follows every row through packing and deserialization.
        for (i, e) in restored.entries.iter().enumerate() {
            assert_eq!(
                restored.geometry(i),
                Some(if e.game % 2 == 0 { &ga } else { &gb })
            );
        }
        // Padding beyond square 288 is forbidden in the first geometry plane.
        bytes[24 + 278 + 32 + 7] |= 128;
        assert!(SequenceBank::read_from(&mut &bytes[..], bank.spec.clone()).is_err());
    }
}
#[test]
fn typed_geometry_is_categorical_and_requires_the_full_board() {
    let mut a = vec![0.; 417];
    let mut b = a.clone();
    a[128] = 1. / 12.;
    b[128] = 12. / 12.;
    let ga = SequenceGeometry::from_state(&a).unwrap();
    let gb = SequenceGeometry::from_state(&b).unwrap();
    assert_eq!(ga.distance(&gb), 1.);
    b[128] = -1. / 12.;
    assert_eq!(ga.distance(&SequenceGeometry::from_state(&b).unwrap()), 1.);
    assert!(SequenceGeometry::from_state(&a[..128]).is_err());
    a[128] = 0.01;
    assert!(SequenceGeometry::from_state(&a).is_err());
}
#[test]
fn spatial_reranking_matches_full_scan_and_retains_phase_and_source_filters() {
    let mut state = aliases()[0].0.clone();
    state[125] = 0.;
    let mut rows = vec![];
    let mut maps = vec![];
    for i in 0..32 {
        let mut s = state.clone();
        s[128 + i] = ((i % 12) + 1) as f64 / 12.;
        if i % 7 == 0 {
            s[125] = 1.;
        }
        rows.push(entry(&s, (i / 2 + 1) as u64, i as u32));
        maps.push(SequenceGeometry::from_state(&s).unwrap());
    }
    let bank = SequenceBank::build_spatial(rows, maps, 32, 0, 4);
    let key = sequence_key(state[..128].try_into().unwrap());
    let g = SequenceGeometry::from_state(&state).unwrap();
    let (actual, _) = bank.nearest_spatial(&key, &g, 0, 2, 4, 8);
    let mut all: Vec<_> = bank
        .entries
        .iter()
        .enumerate()
        .filter(|(_, e)| e.phase == 0 && e.source != 2)
        .map(|(i, e)| {
            (
                sequence_distance(&key, &e.key) + 0.5 * g.distance(bank.geometry(i).unwrap()),
                i,
            )
        })
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut seen = std::collections::HashSet::new();
    all.retain(|(_, i)| seen.insert(bank.entries[*i].source));
    all.truncate(8);
    assert_eq!(actual, all);
}

#[test]
fn spatial_reader_training_gradient_and_live_root_use_the_same_geometry() {
    let state = aliases()[0].0.clone();
    let geometry = SequenceGeometry::from_state(&state).unwrap();
    let rows = (0..12).map(|i| entry(&state, i as u64 + 1, i)).collect();
    let bank = Arc::new(SequenceBank::build_spatial(
        rows,
        vec![geometry; 12],
        12,
        0,
        2,
    ));
    let mut w = MicroModel::seeded(3)
        .with_spatial_policy()
        .parameters()
        .to_vec();
    for x in &mut w[MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS] {
        *x = 0.15;
    }
    let make = |w| {
        MicroModel::from_parameters(w)
            .unwrap()
            .with_sequence_memory(bank.clone())
    };
    let model = make(w);
    let p = include_str!("fixtures/micro-alias-0-a.psr")
        .parse::<GameRecord>()
        .unwrap()
        .replay()
        .unwrap();
    let actions: Vec<_> = legal_actions(&p)
        .iter()
        .map(|a| micro_action_features(&p, *a))
        .collect();
    let mut policy = vec![0.; actions.len()];
    policy[0] = 1.;
    let ex = MicroExample { policy_support: false, action_values: vec![], 
        state: state.clone(),
        actions,
        policy,
        value: 0.3,
        policy_weight: 1.,
        value_weight: 1.0, sequence_source: 1,
    };
    let (_, gradient) = model.loss_gradient(&ex).unwrap();
    for i in [
        MICRO_RESIDUAL_PARAMETERS,
        MICRO_RESIDUAL_PARAMETERS + 5,
        MICRO_MEMORY_PARAMETERS - 1,
    ] {
        let mut plus = model.parameters().to_vec();
        let mut minus = plus.clone();
        plus[i] += 1e-5;
        minus[i] -= 1e-5;
        let f = |w| make(w).loss_gradient(&ex).unwrap().0.total(1.);
        assert!(((f(plus) - f(minus)) / 2e-5 - gradient[i]).abs() < 1e-7);
    }
    let c = model.memory_context(&state, 1).unwrap().unwrap();
    assert!(model.memory_context(&state[..128], 1).is_err());
    let mut incomplete = ex.clone();
    incomplete.state.truncate(128);
    assert!(model.loss_gradient(&incomplete).is_err());
    assert!(c.neighbors.iter().all(|i| bank.entries[*i].source != 1));
    let base = micro_softmax(&MicroModel::logits(&model.embed(&state), &ex.actions)).unwrap();
    let expected = model.memory_priors(&state, &ex.actions, &base, 0).unwrap();
    let report = MicroMctsSession::new(Arc::new(model))
        .search_until(&p, 8, None)
        .unwrap();
    assert_eq!(report.priors, expected);
}
