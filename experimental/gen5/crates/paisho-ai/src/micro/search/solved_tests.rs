use super::*;

#[test]
fn draw_floor_does_not_invent_proofs_or_discard_a_promising_unknown() {
    let p = [Some(0), None, None, Some(-1)];
    let visits = [0, 500, 12, 1000];
    assert_eq!(
        proofs::decision_allowed(&p, None, &[0., -0.2, -0.1, -1.], &visits),
        [true, false, false, false]
    );
    assert_eq!(
        proofs::decision_allowed(&p, None, &[0., -0.2, 0.1, -1.], &visits),
        [false, false, true, false]
    );
    // An unvisited optimistic prior is not evidence for rejecting a guaranteed draw.
    assert_eq!(
        proofs::decision_allowed(&p, None, &[0., -0.2, 0.9, -1.], &[0, 500, 0, 1000]),
        [true, false, false, false]
    );
    let mut proof = proofs::Proof::new(GameOutcome::Ongoing);
    proof.child_solved(GameOutcome::Draw, Player::Host, 2);
    assert_eq!(proof.bound(-0.8), 0.0);
    assert_eq!(proof.bound(0.2), 0.2);
    assert_eq!(proof.outcome, None);
    assert_eq!(
        proofs::decision_allowed(&[Some(0), Some(1)], Some(1), &[0., 1.], &[1000, 0]),
        [false, true]
    );
}

#[test]
fn real_bonus_draw_stops_consuming_puct_forced_and_gumbel_visits() {
    let record: paisho_core::GameRecord =
        include_str!("../../../tests/fixtures/micro_proved_draw_bonus.psr")
            .parse()
            .unwrap();
    assert_eq!(record.replay().unwrap().outcome(), GameOutcome::Draw);
    let mut p = record.initial_position();
    for a in &record.actions()[..record.actions().len() - 1] {
        p.apply(*a).unwrap();
    }
    assert_eq!(p.phase(), paisho_core::TurnPhase::HarmonyBonus);
    let legal = legal_actions(&p);
    let draws: Vec<_> = legal
        .iter()
        .map(|a| {
            let mut next = p.clone();
            next.apply(*a).unwrap();
            next.outcome() == GameOutcome::Draw
        })
        .collect();
    assert!(draws.contains(&true) && draws.contains(&false));
    for (mode, strength) in [
        (MicroSearchMode::Puct, 0.),
        (MicroSearchMode::Puct, 2.),
        (MicroSearchMode::Gumbel, 0.),
    ] {
        let mut session =
            MicroMctsSession::new(Arc::new(MicroModel::seeded(3).with_spatial_policy()));
        let opts = MicroSearchOptions {
            mode,
            proof_search: true,
            forced_playout_strength: strength,
            ..Default::default()
        };
        for _ in 0..2 {
            let r = session.search_with_options(&p, 512, None, opts).unwrap();
            assert_eq!(r.actions, legal);
            assert_eq!(r.new_visits.iter().sum::<usize>(), r.simulations);
            for (i, draw) in draws.iter().enumerate() {
                if *draw {
                    assert_eq!(r.visits[i], 0);
                    assert_eq!(r.proven_action_values[i], Some(0));
                }
            }
            let chosen = r.selected_index;
            if r.proven_value.is_none() {
                assert!(draws[chosen] || (r.visits[chosen] > 0 && r.values[chosen] > 0.));
                for (i, mass) in r.policy_target.iter().enumerate() {
                    if *mass > 0. {
                        assert!(draws[i] || r.values[i] > 0.);
                    }
                }
            }
            assert!((r.policy_target.iter().sum::<f64>() - 1.).abs() < 1e-12);
            if !r.pruned_visits.is_empty() {
                let kept: Vec<_> = r
                    .visits
                    .iter()
                    .zip(&r.pruned_visits)
                    .map(|(raw, pruned)| raw - pruned)
                    .collect();
                let total: usize = kept.iter().sum();
                if total > 0 {
                    for (mass, count) in r.policy_target.iter().zip(kept) {
                        assert!((*mass - count as f64 / total as f64).abs() < 1e-12);
                    }
                } else {
                    assert_eq!(r.policy_target[chosen], 1.);
                    assert_eq!(r.proven_action_values[chosen], Some(0));
                }
            }
            r.example(0., 1.).unwrap();
        }
    }
}

#[test]
fn draw_bound_is_backed_up_at_the_actual_chooser_including_depth_cutoff() {
    let model = MicroModel::seeded(7).with_spatial_policy();
    let mut session = MicroMctsSession::new(Arc::new(model.clone()));
    let record: paisho_core::GameRecord =
        include_str!("../../../tests/fixtures/micro_proved_draw_bonus.psr")
            .parse()
            .unwrap();
    let p = record.initial_position();
    let inference = session.cache.get(p.clone(), &model);
    let mut node = Node::new(inference);
    node.proof.child_solved(GameOutcome::Draw, p.to_move(), 2);
    let expected = node.inference.value().max(0.);
    let value = simulate_mode(
        &mut node,
        &mut session.cache,
        &model,
        1.5,
        0,
        0,
        None,
        None,
        MicroSearchMode::Puct,
        0.,
        true,
    )
    .unwrap();
    assert_eq!(value, expected);
    assert_eq!(node.proof.outcome, None);
}
