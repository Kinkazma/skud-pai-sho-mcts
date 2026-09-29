use paisho_ai::*;
use paisho_core::*;
use std::sync::Arc;
fn bank(state: &[f64; 128]) -> Arc<SequenceBank> {
    let mut entries = vec![];
    for i in 0..12 {
        let mut patterns = [[0; 32]; 4];
        patterns[i % 4][i % 32] = 127;
        entries.push(SequenceEntry {
            key: sequence_key(state),
            patterns,
            source: (i + 1) as u64,
            game: i as u32,
            decision: 0,
            end_decision: 20,
            outcome: (i % 3) as i8 - 1,
            phase: 0,
        });
    }
    Arc::new(SequenceBank::build(entries, 12, 2, 4))
}
#[test]
fn rare_win_loses_protection_when_common_or_regularly_used() {
    let policy = RarityPolicy {
        max_source_games: 3,
        max_recent_uses: 8,
    };
    let mut e = MotifEvidence {
        human: false,
        associated_win: true,
        source_games: 1,
        recent_uses: 0,
    };
    assert_eq!(policy.protection(e), MemoryProtection::RareWin);
    e.recent_uses = 9;
    assert_eq!(policy.protection(e), MemoryProtection::Ordinary);
    e.recent_uses = 0;
    e.source_games = 4;
    assert_eq!(policy.protection(e), MemoryProtection::Ordinary);
    e.human = true;
    assert_eq!(policy.protection(e), MemoryProtection::Human);
    e.human = false;
    e.source_games = 1;
    e.associated_win = false;
    assert_eq!(policy.protection(e), MemoryProtection::Ordinary);
}
#[test]
fn retrieval_excludes_entire_source_and_codec_roundtrips() {
    let state = [0.2; 128];
    let b = bank(&state);
    let c = b.context(&state, 1);
    assert_eq!(c.neighbors.len(), 8);
    assert!(c.neighbors.iter().all(|i| b.entries[*i].source != 1));
    assert!(Arc::ptr_eq(&c, &b.context(&state, 1)));
    let mut bytes = vec![];
    b.write_to(&mut bytes).unwrap();
    let restored = SequenceBank::read_from(&mut &bytes[..], b.spec.clone()).unwrap();
    assert_eq!(restored.games, 12);
    assert_eq!(restored.entries.len(), 12);
    bytes.push(0);
    assert!(SequenceBank::read_from(&mut &bytes[..], b.spec.clone()).is_err());
}
#[test]
fn reader_gradient_and_zero_head_search_identity() {
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let state = micro_state_features(&p);
    let b = bank(&state);
    let old = MicroModel::seeded(3).with_residual_policy(4);
    let mut model = old.with_sequence_memory(b.clone());
    let actions: Vec<_> = legal_actions(&p)
        .iter()
        .map(|a| micro_action_features(&p, *a))
        .collect();
    let mut target = vec![0.; actions.len()];
    target[0] = 1.;
    let ex = MicroExample {
        sequence_source: 1,
        state,
        actions,
        policy: target,
        value: 0.3,
        policy_weight: 1.,
    };
    let a = MicroMctsSession::new(Arc::new(old))
        .search_until(&p, 32, None)
        .unwrap();
    let a2 = MicroMctsSession::new(Arc::new(model.clone()))
        .search_until(&p, 32, None)
        .unwrap();
    assert_eq!(a.visits, a2.visits);
    assert_eq!(a.priors, a2.priors);
    let mut w = model.parameters().to_vec();
    for x in &mut w[MICRO_RESIDUAL_PARAMETERS..] {
        *x = 0.15;
    }
    model = MicroModel::from_parameters(w)
        .unwrap()
        .with_sequence_memory(b.clone());
    let (_, g) = model.loss_gradient(&ex).unwrap();
    for i in MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS {
        let mut plus = model.parameters().to_vec();
        plus[i] += 1e-5;
        let mut minus = model.parameters().to_vec();
        minus[i] -= 1e-5;
        let f = |w| {
            MicroModel::from_parameters(w)
                .unwrap()
                .with_sequence_memory(b.clone())
                .loss_gradient(&ex)
                .unwrap()
                .0
                .total(1.)
        };
        let numerical = (f(plus) - f(minus)) / 2e-5;
        assert!((numerical - g[i]).abs() < 1e-7, "gradient {i}");
    }
    let base = micro_softmax(&MicroModel::logits(&model.embed(&ex.state), &ex.actions)).unwrap();
    let adjusted = model
        .memory_priors(&ex.state, &ex.actions, &base, 0)
        .unwrap();
    assert!(adjusted
        .iter()
        .zip(&base)
        .any(|(a, b)| (a - b).abs() > 1e-12));
    let root = MicroMctsSession::new(Arc::new(model.clone()))
        .search_until(&p, 32, None)
        .unwrap();
    for (a, b) in root.priors.iter().zip(adjusted) {
        assert!((a - b).abs() < 1e-12);
    }
    model.train_step(&ex, 0.01, 0.).unwrap();
    assert!(Arc::ptr_eq(model.sequence_memory().unwrap(), &b));
    model.train_batch(&[&ex], 0.01, 0.).unwrap();
    assert!(Arc::ptr_eq(model.sequence_memory().unwrap(), &b));
}

#[test]
fn repeated_retrieval_counts_cached_uses_and_expires_old_usage() {
    let mut state = [0.2; 128];
    state[125] = 0.;
    let b = bank(&state);
    let c = b.context(&state, 0);
    for _ in 0..9 {
        b.context(&state, 0);
    }
    assert!(b.recent_usage().iter().all(|u| u.uses == 10));
    state[125] = 1.;
    for _ in 0..16384 {
        b.context(&state, 0);
    }
    assert!(b.recent_usage().is_empty());
    assert!(!c.neighbors.is_empty());
}
