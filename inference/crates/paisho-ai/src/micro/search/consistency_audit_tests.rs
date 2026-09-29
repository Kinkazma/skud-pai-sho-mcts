//! Controlled prior/target audit. Only the root prior is injected; native search,
//! rules, selection, forced playouts and pruning run unchanged. No training.
use super::*;

#[test]
fn flat_values_preserve_a_strong_prior_without_root_noise() {
    let position = Position::from_standard_setup_with_rules(
        paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3),
        paisho_core::RuleProfileId::SkudPaiShoGen5V1,
    );
    let model = Arc::new(MicroModel::from_parameters(vec![0.; MICRO_DEEP_VALUE_PARAMETERS]).unwrap());
    for budget in [32, 64, 128, 256, 512] {
        for (mode, noise, strength) in [
            (MicroSearchMode::Puct, 0., 0.),
            (MicroSearchMode::Puct, 0., 2.),
            (MicroSearchMode::Puct, 0.25, 2.),
            (MicroSearchMode::Gumbel, 0., 0.),
        ] {
            for seed in [11, 29, 47] {
                let mut session = MicroMctsSession::new(model.clone());
                let inference = session.cache.get(position.clone(), &model);
                let base = inference.policy(&model).unwrap();
                let count = base.actions.len();
                let mut priors = vec![0.1 / (count - 1) as f64; count];
                priors[0] = 0.9;
                let mut root = Node::new(inference);
                root.memory_policy = Some(Arc::new(Policy {
                    actions: base.actions.clone(), features: base.features.clone(),
                    priors, log_priors: OnceLock::new(),
                }));
                session.root = Some(root);
                let options = MicroSearchOptions { mode, seed, dirichlet_fraction: noise,
                    forced_playout_strength: strength, proof_search: true,
                    ..Default::default() };
                // Repeat on the same root to cover retained visits as well.
                for retained in [false, true] {
                    let r = session.search_with_options(&position, budget, None, options).unwrap();
                    assert!(r.values.iter().all(|v| *v == 0.));
                    assert!(r.proven_action_values.iter().all(Option::is_none));
                    assert_eq!(r.priors[0], 0.9);
                    assert_eq!(r.simulations, budget);
                    if noise == 0. && mode == MicroSearchMode::Puct {
                        assert!(r.policy_target[0] > 0.85, "prior dilution: {:?}", r.policy_target);
                    }
                    if mode == MicroSearchMode::Gumbel {
                        assert!((r.policy_target[0] - 0.9).abs() < 1e-12);
                    }
                    println!("CONSISTENCY budget={budget} mode={mode:?} noise={noise} forced={strength} seed={seed} retained={retained} legal={count} visited={} prior={} search_prior={} raw={} target={} pruned={}",
                        r.visits.iter().filter(|n| **n > 0).count(), r.priors[0], r.search_priors[0],
                        r.visits[0] as f64 / r.visits.iter().sum::<usize>() as f64,
                        r.policy_target[0], r.pruned_visits.iter().sum::<usize>());
                }
            }
        }
    }
}
