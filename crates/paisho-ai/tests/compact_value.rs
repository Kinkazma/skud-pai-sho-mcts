use paisho_ai::{
    evaluate_position, CompactValueFeatures, CompactValueModel, HeuristicWeights, MctsEvaluator,
    COMPACT_FEATURE_COUNT, COMPACT_FEATURE_NAMES, COMPACT_FEATURE_SCALES,
};
use paisho_core::{GameOutcome, GameRecord, Player, Position};

const RECORDS: [&str; 6] = [
    include_str!("fixtures/site_bot_v1_ring_finish.psr"),
    include_str!("fixtures/site_bot_v1_reserve_finish.psr"),
    include_str!("fixtures/tactical_guest_forced_three.psr"),
    include_str!("fixtures/tactical_host_forced_three.psr"),
    include_str!("fixtures/tactical_guest_wheel_finish.psr"),
    include_str!("fixtures/tactical_host_wheel_finish.psr"),
];

fn positions() -> Vec<Position> {
    let mut result = Vec::new();
    for text in RECORDS {
        let record: GameRecord = text.parse().unwrap();
        let mut position = record.initial_position();
        result.push(position.clone());
        for action in record.actions() {
            position.apply(*action).unwrap();
            result.push(position.clone());
        }
    }
    result
}

#[test]
fn learned_candidate_batches_keep_scalar_bits_and_order_across_pools() {
    let model =
        CompactValueModel::from_weights(std::array::from_fn(|i| (i as f64 * 0.37).sin() * 0.8))
            .unwrap();
    let real = positions();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for count in [0, 1, 63, 64, 65, 256] {
            let batch: Vec<_> = (0..count).map(|i| real[i % real.len()].clone()).collect();
            for player in [Player::Host, Player::Guest] {
                let expected: Vec<_> = batch
                    .iter()
                    .map(|p| model.evaluate(p, player).to_bits())
                    .collect();
                let actual = pool.install(|| {
                    MctsEvaluator::evaluate(&model, &batch, player, HeuristicWeights::default())
                        .unwrap()
                });
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected
                );
            }
        }
    }
}

#[test]
fn initial_model_matches_legacy_exactly_on_real_trajectories() {
    let settings = [
        HeuristicWeights::default(),
        HeuristicWeights {
            harmony: -0.43,
            midline_harmony: 0.72,
            blooming_flower: -0.17,
            total_flower: 0.06,
            basic_reserve_progress: 0.41,
        },
        HeuristicWeights {
            harmony: f32::MAX,
            midline_harmony: f32::MAX,
            blooming_flower: f32::MAX,
            total_flower: f32::MAX,
            basic_reserve_progress: f32::MAX,
        },
    ];
    for weights in settings {
        let model = CompactValueModel::from_heuristic(weights).unwrap();
        for position in positions() {
            for player in [Player::Host, Player::Guest] {
                assert_eq!(
                    model.evaluate(&position, player).to_bits(),
                    evaluate_position(&position, player, weights).to_bits()
                );
            }
        }
    }
}

#[test]
fn all_features_are_normalized_and_antisymmetric_after_learning() {
    let weights = std::array::from_fn(|index| (index as f64 * 1.23).sin());
    let model = CompactValueModel::from_weights(weights).unwrap();
    for position in positions() {
        let host = CompactValueFeatures::extract(&position, Player::Host);
        let guest = CompactValueFeatures::extract(&position, Player::Guest);
        for (own, opponent) in host.values().iter().zip(guest.values()) {
            assert!(own.is_finite() && own.abs() <= 1.0);
            assert_eq!(*own, -*opponent);
        }
        assert_eq!(model.predict(&host), -model.predict(&guest));
    }
    let names = COMPACT_FEATURE_NAMES
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(names.len(), COMPACT_FEATURE_COUNT);
    assert!(COMPACT_FEATURE_SCALES
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
}

#[test]
fn terminal_values_cannot_be_changed_by_training_or_extreme_weights() {
    let mut model = CompactValueModel::from_weights([1e200; COMPACT_FEATURE_COUNT]).unwrap();
    let before = model.clone();
    let terminal = positions()
        .into_iter()
        .find(|position| matches!(position.outcome(), GameOutcome::Win(_)))
        .unwrap();
    for player in [Player::Host, Player::Guest] {
        let features = CompactValueFeatures::extract(&terminal, player);
        let expected = if terminal.outcome() == GameOutcome::Win(player) {
            1.0
        } else {
            -1.0
        };
        assert_eq!(model.predict(&features), expected);
        model.train_step(&features, -expected, 1.0, 0.5).unwrap();
        assert_eq!(model.gradient(&features), [0.0; COMPACT_FEATURE_COUNT]);
        assert_eq!(model, before);
    }
    let draw = CompactValueFeatures::from_values([0.5; COMPACT_FEATURE_COUNT], Some(0.0)).unwrap();
    assert_eq!(model.predict(&draw), 0.0);
    model.train_step(&draw, 1.0, 1.0, 0.5).unwrap();
    assert_eq!(model, before);
}

#[test]
fn gradient_matches_finite_differences_including_zero() {
    let features = CompactValueFeatures::from_values(
        std::array::from_fn(|index| (index as f64 * 0.123).cos()),
        None,
    )
    .unwrap();
    for magnitude in [0.0, -0.2, 0.7] {
        let weights = [magnitude; COMPACT_FEATURE_COUNT];
        let model = CompactValueModel::from_weights(weights).unwrap();
        let gradient = model.gradient(&features);
        for index in [0, 4, 13, 42, 63] {
            let mut plus = weights;
            let mut minus = weights;
            plus[index] += 1e-6;
            minus[index] -= 1e-6;
            let finite = (CompactValueModel::from_weights(plus)
                .unwrap()
                .predict(&features)
                - CompactValueModel::from_weights(minus)
                    .unwrap()
                    .predict(&features))
                / 2e-6;
            assert!(
                (gradient[index] - finite).abs() < 2e-6,
                "feature {index}: {} != {finite}",
                gradient[index]
            );
        }
    }
}

#[test]
fn sgd_learns_a_feature_relation_on_held_out_values() {
    // A known relationship on an additional structural feature, not one of the
    // five legacy terms; test on feature values absent from the training grid.
    fn example(value: f64) -> CompactValueFeatures {
        let mut values = [0.0; COMPACT_FEATURE_COUNT];
        values[42] = value;
        CompactValueFeatures::from_values(values, None).unwrap()
    }
    fn target(value: f64) -> f64 {
        2.0 * value / (1.0 + (2.0 * value).abs())
    }
    let mut model = CompactValueModel::from_weights([0.0; COMPACT_FEATURE_COUNT]).unwrap();
    for _ in 0..400 {
        for value in [-0.9, -0.4, 0.2, 0.7] {
            model
                .train_step(&example(value), target(value), 0.3, 0.0)
                .unwrap();
        }
    }
    for value in [-0.75, -0.1, 0.35, 0.95] {
        assert!((model.predict(&example(value)) - target(value)).abs() < 0.002);
    }
    assert!((model.weights()[42] - 2.0).abs() < 0.02);
}

#[test]
fn invalid_inputs_and_overflow_leave_weights_unchanged() {
    assert!(CompactValueModel::from_weights([f64::NAN; COMPACT_FEATURE_COUNT]).is_err());
    assert!(CompactValueModel::from_weights([f64::MAX; COMPACT_FEATURE_COUNT]).is_err());
    assert!(CompactValueFeatures::from_values([1.01; COMPACT_FEATURE_COUNT], None).is_err());
    assert!(CompactValueFeatures::from_values([0.0; COMPACT_FEATURE_COUNT], Some(0.5)).is_err());
    let features = CompactValueFeatures::from_values([1.0; COMPACT_FEATURE_COUNT], None).unwrap();
    let mut model = CompactValueModel::default();
    let before = model.clone();
    for (target, rate, l2) in [
        (f64::NAN, 0.1, 0.0),
        (1.1, 0.1, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.1, -1.0),
        (0.0, f64::MAX, f64::MAX),
    ] {
        assert!(model.train_step(&features, target, rate, l2).is_err());
        assert_eq!(model, before);
    }
}

#[test]
fn mcts_adapter_preserves_order_and_uses_the_model_for_leaves() {
    let positions = positions();
    let model = CompactValueModel::from_weights([0.01; COMPACT_FEATURE_COUNT]).unwrap();
    for player in [Player::Host, Player::Guest] {
        let batch =
            MctsEvaluator::evaluate(&model, &positions, player, HeuristicWeights::default())
                .unwrap();
        for (position, score) in positions.iter().zip(batch) {
            assert_eq!(score, model.evaluate(position, player));
            assert_eq!(
                score,
                model
                    .evaluate_leaf(position, player, HeuristicWeights::default())
                    .unwrap()
            );
        }
    }
}

#[test]
fn suppression_feature_matches_direct_neighbor_queries_on_real_positions() {
    use paisho_core::{surrounding_neighbors, Accent, TileKind};
    for p in positions()
        .into_iter()
        .filter(|p| p.outcome() == GameOutcome::Ongoing)
    {
        let mut counts = [0f64; 2];
        for (at, tile) in p.board().occupied() {
            if tile.kind.is_flower()
                && surrounding_neighbors(at).any(|n| {
                    p.board()
                        .get(n)
                        .is_some_and(|t| t.kind == TileKind::Accent(Accent::Knotweed))
                })
            {
                counts[tile.owner.index()] += 1.;
            }
        }
        let features = CompactValueFeatures::extract(&p, Player::Host);
        assert_eq!(
            features.values()[62],
            (counts[0] - counts[1]) / COMPACT_FEATURE_SCALES[62]
        );
    }
}
