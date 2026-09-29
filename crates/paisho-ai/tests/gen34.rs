use paisho_ai::*;
use paisho_core::*;
fn position() -> Position {
    Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3))
}
#[test]
fn zero_extension_preserves_retained_search_exactly() {
    let old = Gen32Model::from_gen31(CompactValueModel::default(), 31);
    let new = old.clone().with_value128([0.; 64]).unwrap();
    let c = MctsConfig {
        simulations: 32,
        ..Default::default()
    };
    let mut a = MctsSession::new(81, c, &old).unwrap();
    let mut b = MctsSession::new(81, c, &new).unwrap();
    let mut p = position();
    for _ in 0..16 {
        if p.outcome() != GameOutcome::Ongoing {
            break;
        }
        for player in [Player::Host, Player::Guest] {
            assert_eq!(old.value_at(&p, player), new.value_at(&p, player));
        }
        let legal = legal_actions(&p);
        let x = a.search_until(&p, &legal, None).unwrap();
        let y = b.search_until(&p, &legal, None).unwrap();
        assert_eq!(x.selected_index, y.selected_index);
        for (x, y) in x.actions.iter().zip(&y.actions) {
            assert_eq!((x.visits, x.value_sum), (y.visits, y.value_sum));
        }
        let action = legal[x.selected_index];
        assert_eq!(a.advance(action), b.advance(action));
        p.apply(action).unwrap();
    }
}
#[test]
fn all_coordinates_have_correct_gradient_and_invalid_update_is_atomic() {
    let model = Gen32Model::from_gen31(CompactValueModel::default(), 1)
        .with_value128([0.01; 64])
        .unwrap();
    let ex = MicroExample {
        sequence_source: 0,
        state: std::array::from_fn(|i| (i as f64 + 1.) / 200.),
        actions: vec![],
        policy: vec![],
        value: -0.7,
        policy_weight: 0.,
    };
    let mut learned = model.clone();
    learned.train(&ex, 0.01).unwrap();
    for k in 0..128 {
        let mut plus = model.clone();
        let mut minus = model.clone();
        let epsilon = 1e-6;
        if k < 64 {
            let mut a = *model.value.weights();
            a[k] += epsilon;
            plus.value = CompactValueModel::from_weights(a).unwrap();
            a[k] -= 2. * epsilon;
            minus.value = CompactValueModel::from_weights(a).unwrap();
        } else {
            plus.value_extra.as_mut().unwrap()[k - 64] += epsilon;
            minus.value_extra.as_mut().unwrap()[k - 64] -= epsilon;
        }
        let loss = |m: &Gen32Model| 0.5 * (m.predict_value_state(&ex.state) - ex.value).powi(2);
        let gradient = (loss(&plus) - loss(&minus)) / (2. * epsilon);
        let observed = if k < 64 {
            (model.value.weights()[k] - learned.value.weights()[k]) / 0.01
        } else {
            (model.value_extra.unwrap()[k - 64] - learned.value_extra.unwrap()[k - 64]) / 0.01
        };
        assert!(
            (gradient - observed).abs() < 1e-7,
            "{k}: {gradient} != {observed}"
        );
    }
    let old = learned.clone();
    assert!(learned.train(&ex, f64::NAN).is_err());
    assert_eq!(old.value.weights(), learned.value.weights());
    assert_eq!(old.value_extra, learned.value_extra);
    assert_eq!(old.policy.parameters(), learned.policy.parameters());
    let p = position();
    assert_eq!(
        learned.value_at(&p, Player::Host),
        -learned.value_at(&p, Player::Guest)
    );
}
#[test]
fn bonus_context_is_phase_filtered_source_excluded_and_cached() {
    let mut s = micro_state_features(&position());
    s[124] = 0.;
    s[125] = 1.;
    let entries = (1..=10)
        .map(|i| SequenceEntry {
            key: sequence_key(&s),
            patterns: [[20; 32]; 4],
            source: i,
            game: i as u32,
            decision: 2,
            end_decision: 3,
            outcome: 1,
            phase: 1,
        })
        .collect();
    let bank = SequenceBank::build(entries, 10, 0, 2);
    let ctx = bank.context(&s, 1);
    assert_eq!(ctx.neighbors.len(), 8);
    assert!(ctx.neighbors.iter().all(|&i| bank.entries[i].source != 1));
    assert!(std::sync::Arc::ptr_eq(&ctx, &bank.context(&s, 1)));
    s[125] = 0.;
    s[124] = 1.;
    assert!(bank.context(&s, 0).neighbors.is_empty());
}

#[test]
fn internal_bonus_recall_uses_learned_reader_and_legacy_scope_stays_root_only() {
    let record: GameRecord = include_str!("fixtures/gen34_bonus.psr").parse().unwrap();
    let p = record.replay().unwrap();
    assert_eq!(p.phase(), TurnPhase::HarmonyBonus);
    let state = micro_state_features(&p);
    let actions = legal_actions(&p);
    let mut patterns = [[0; 32]; 4];
    patterns[0] =
        micro_action_features(&p, actions[0]).map(|x| (127. * x.clamp(-1., 1.)).round() as i8);
    let bank = std::sync::Arc::new(SequenceBank::build(
        (1..=8)
            .map(|i| SequenceEntry {
                key: sequence_key(&state),
                patterns,
                source: i,
                game: i as u32,
                decision: 0,
                end_decision: 1,
                outcome: 1,
                phase: 1,
            })
            .collect(),
        8,
        0,
        2,
    ));
    let mut model =
        Gen32Model::from_gen31(CompactValueModel::default(), 1).with_memory(bank.clone());
    let mut parameters = model.policy.parameters().to_vec();
    parameters[MICRO_RESIDUAL_PARAMETERS..].fill(1.);
    model.policy = MicroModel::from_parameters(parameters)
        .unwrap()
        .with_sequence_memory(bank.clone());
    let plain = model.policy_bias(&p, &actions).unwrap();
    assert_eq!(bank.telemetry()[0], 0);
    let root = model.root_policy_bias(&p, &actions).unwrap();
    assert_eq!(bank.telemetry()[0], 1);
    assert_ne!(plain, root);
    model.memory_scope = Gen3MemoryScope::BonusNodes;
    assert_eq!(model.policy_bias(&p, &actions).unwrap(), root);
    assert_eq!(bank.telemetry()[0], 2);
    let main = position();
    model.policy_bias(&main, &legal_actions(&main)).unwrap();
    assert_eq!(bank.telemetry()[0], 2);
    model.memory_scope = Gen3MemoryScope::AllNodes;
    model.policy_bias(&main, &legal_actions(&main)).unwrap();
    assert_eq!(bank.telemetry()[0], 3);
}

#[test]
fn terminal_outcomes_override_extended_value_for_both_seats() {
    let r: GameRecord = include_str!("fixtures/site_bot_v1_ring_finish.psr")
        .parse()
        .unwrap();
    let p = r.replay().unwrap();
    let GameOutcome::Win(winner) = p.outcome() else {
        panic!("fixture must end in win")
    };
    let model = Gen32Model::from_gen31(CompactValueModel::default(), 7)
        .with_value128([100.; 64])
        .unwrap();
    assert_eq!(model.value_at(&p, winner), 1.);
    assert_eq!(model.value_at(&p, winner.opponent()), -1.);
}
