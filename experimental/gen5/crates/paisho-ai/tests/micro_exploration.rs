use paisho_ai::*;
use paisho_core::*;
use std::{sync::Arc, time::Instant};
fn position() -> Position {
    GameRecord::with_rules(
        StandardSetup::balanced(BASIC_FLOWERS[0]),
        RuleProfileId::SkudPaiShoGen5V1,
    )
    .initial_position()
}
#[test]
fn forced_playouts_preserve_budget_retained_statistics_and_correct_only_targets() {
    let p = position();
    let mut session = MicroMctsSession::new(Arc::new(MicroModel::seeded(5)));
    let options = MicroSearchOptions {
        dirichlet_fraction: 0.25,
        seed: 79,
        forced_playout_strength: 2.0,
        ..Default::default()
    };
    let mut forced = 0;
    let mut pruned = 0;
    let mut previous = vec![];
    for budget in [64, 256] {
        let r = session
            .search_with_options(&p, budget, None, options)
            .unwrap();
        assert_eq!(r.new_visits.iter().sum::<usize>(), budget);
        assert!(r.new_forced_visits.iter().sum::<usize>() <= budget);
        assert_eq!(r.pruned_visits[r.selected_index], 0);
        assert_eq!(r.new_forced_visits.len(), r.actions.len());
        if !previous.is_empty() {
            for ((total, fresh), old) in r.visits.iter().zip(&r.new_visits).zip(&previous) {
                assert_eq!(total - fresh, *old);
            }
        }
        let retained: usize = r
            .visits
            .iter()
            .zip(&r.pruned_visits)
            .map(|(n, p)| n - p)
            .sum();
        for ((n, p), target) in r.visits.iter().zip(&r.pruned_visits).zip(&r.policy_target) {
            assert!(p <= n);
            assert!((*target - (n - p) as f64 / retained as f64).abs() < 1e-12);
        }
        r.example(0.0, 1.0).unwrap();
        forced += r.new_forced_visits.iter().sum::<usize>();
        pruned += r.pruned_visits.iter().sum::<usize>();
        previous = r.visits;
    }
    assert!(forced > 0, "the test must exercise forced search");
    assert!(pruned > 0, "the test must exercise corrected targets");
    let expired = session
        .search_with_options(&p, 64, Some(Instant::now()), options)
        .unwrap();
    assert_eq!(expired.simulations, 0);
    assert_eq!(expired.new_forced_visits.iter().sum::<usize>(), 0);
    assert_eq!(expired.visits, previous);
}

#[test]
fn disabled_forcing_keeps_puct_exact_and_gumbel_rejects_combination() {
    let p = position();
    let model = Arc::new(MicroModel::seeded(9));
    let mut a = MicroMctsSession::new(model.clone());
    let mut b = MicroMctsSession::new(model);
    let x = a.search_until(&p, 64, None).unwrap();
    let y = b
        .search_with_options(
            &p,
            64,
            None,
            MicroSearchOptions {
                forced_playout_strength: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        (x.visits, x.values, x.selected_index),
        (y.visits, y.values, y.selected_index)
    );
    assert!(y.pruned_visits.is_empty());
    assert!(MicroSearchOptions {
        mode: MicroSearchMode::Gumbel,
        forced_playout_strength: 2.0,
        ..Default::default()
    }
    .validate()
    .is_err());
}
#[test]
fn modes_allocate_exact_new_visits_retain_trees_and_make_aligned_targets() {
    let model = Arc::new(MicroModel::seeded(5));
    for mode in [MicroSearchMode::Puct, MicroSearchMode::Gumbel] {
        let mut p = position();
        let mut session = MicroMctsSession::new(model.clone());
        let options = MicroSearchOptions {
            mode,
            dirichlet_fraction: if mode == MicroSearchMode::Puct {
                0.25
            } else {
                0.0
            },
            seed: 79,
            ..Default::default()
        };
        for budget in [1, 8, 32, 64, 128, 256, 512] {
            let r = session
                .search_with_options(&p, budget, None, options)
                .unwrap();
            assert_eq!(r.simulations, budget);
            assert_eq!(r.new_visits.iter().sum::<usize>(), budget);
            assert!((r.policy_target.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            assert!(r.policy_target.iter().all(|p| p.is_finite() && *p >= 0.0));
            r.example(0.0, 1.0).unwrap();
            let action = r.actions[r.selected_index];
            p.apply(action).unwrap();
            session.advance(action).unwrap();
            if p.outcome() != GameOutcome::Ongoing {
                break;
            }
        }
    }
}
#[test]
fn seeded_exploration_is_reproducible_and_keeps_cached_priors_exact() {
    let model = Arc::new(MicroModel::seeded(4));
    let p = position();
    let mut noisy = MicroMctsSession::new(model.clone());
    let mut same = MicroMctsSession::new(model.clone());
    let mut plain = MicroMctsSession::new(model);
    let options = MicroSearchOptions {
        dirichlet_fraction: 0.25,
        seed: 46,
        ..Default::default()
    };
    let a = noisy.search_with_options(&p, 64, None, options).unwrap();
    let b = same.search_with_options(&p, 64, None, options).unwrap();
    let c = plain.search_until(&p, 64, None).unwrap();
    assert_eq!(a.visits, b.visits);
    assert_eq!(a.priors, c.priors);
    assert_ne!(a.search_priors, a.priors);
    let d = noisy.search_until(&p, 1, None).unwrap();
    assert_eq!(d.search_priors, c.priors);
}
#[test]
fn gumbel_deadline_and_fresh_halving_on_reused_root() {
    let model = Arc::new(MicroModel::seeded(5));
    let p = position();
    let mut s = MicroMctsSession::new(model);
    let o = MicroSearchOptions {
        mode: MicroSearchMode::Gumbel,
        ..Default::default()
    };
    s.search_with_options(&p, 64, None, o).unwrap();
    let r = s.search_with_options(&p, 32, None, o).unwrap();
    assert_eq!(r.inherited_visits, 64);
    assert_eq!(r.new_visits.iter().sum::<usize>(), 32);
    let r = s
        .search_with_options(&p, 512, Some(Instant::now()), o)
        .unwrap();
    assert_eq!(r.simulations, 0);
    assert_eq!(r.new_visits.iter().sum::<usize>(), 0);
    assert!(r.example(0.0, 1.0).is_err());
}
