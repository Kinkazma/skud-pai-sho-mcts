use super::*;

#[test]
fn measurement_cadences_are_optional_positive_and_independent() {
    let base = [
        "--campaign-dir",
        "/tmp/paisho-test",
        "--target-generation",
        "20",
    ];
    let defaults = options::Options::parse_from(base.into_iter().map(str::to_owned)).unwrap();
    assert_eq!(
        (defaults.promotion_every, defaults.evaluation_every),
        (1, 1)
    );
    let configured = options::Options::parse_from(
        base.into_iter()
            .chain(["--promotion-every", "5", "--evaluation-every", "10"])
            .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(
        (configured.promotion_every, configured.evaluation_every),
        (5, 10)
    );
    for flag in ["--promotion-every", "--evaluation-every"] {
        assert!(options::Options::parse_from(
            base.into_iter().chain([flag, "0"]).map(str::to_owned)
        )
        .is_err());
    }
}

#[test]
fn child_output_requires_one_exact_key() {
    assert_eq!(required_value("alpha=1\nbeta=2\n", "alpha").unwrap(), "1");
    assert!(required_value("beta=2\n", "alpha").is_err());
    assert!(required_value("alpha=1\nalpha=2\n", "alpha").is_err());
}

#[test]
fn generation_seeds_are_reproducible_and_separated() {
    let seed = generation_seed(17, 3, 5);
    assert_eq!(seed, generation_seed(17, 3, 5));
    assert_ne!(seed, generation_seed(17, 4, 5));
    assert_ne!(seed, generation_seed(17, 3, 6));
}

#[test]
fn promotion_conclusions_remain_distinct() {
    for (name, expected) in [
        (
            "PromoteCandidate",
            PromotionCampaignConclusion::PromoteCandidate,
        ),
        (
            "RejectCandidate",
            PromotionCampaignConclusion::RejectCandidate,
        ),
        (
            "InconclusiveMaximumEligiblePairs",
            PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs,
        ),
        (
            "InconclusiveMaximumAttemptedPairs",
            PromotionCampaignConclusion::InconclusiveMaximumAttemptedPairs,
        ),
    ] {
        assert_eq!(parse_conclusion(name).unwrap(), expected);
    }
    assert!(parse_conclusion("promote").is_err());
}

#[test]
fn class_arguments_preserve_capacity_order() {
    let classes = [
        GenerationInferenceClassV1 {
            legal_action_capacity: 64,
            batch_size: 8,
        },
        GenerationInferenceClassV1 {
            legal_action_capacity: 1024,
            batch_size: 1,
        },
    ];
    assert_eq!(class_argument(&classes), "64:8,1024:1");
}

#[test]
fn command_options_use_measured_actor_defaults() {
    let options = options::Options::parse_from(
        [
            "--campaign-dir",
            "/tmp/paisho-test-campaign",
            "--target-generation",
            "1",
            "--workers",
            "10",
        ]
        .into_iter()
        .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(options.actors_per_round, 80);
    assert_eq!(
        options.classes,
        vec![
            options::ClassShape {
                capacity: 64,
                batch_size: 8,
            },
            options::ClassShape {
                capacity: 128,
                batch_size: 4,
            },
            options::ClassShape {
                capacity: 1_024,
                batch_size: 4,
            },
        ]
    );
}

#[test]
fn command_options_allow_neutral_starts_to_be_disabled() {
    let options = options::Options::parse_from(
        [
            "--campaign-dir",
            "/tmp/paisho-test-campaign",
            "--target-generation",
            "1",
            "--start-horizon",
            "0",
            "--promotion-start-horizon",
            "0",
            "--promotion-sampling-temperature",
            "0",
            "--actor-games",
            "2",
            "--actors",
            "2",
            "--promotion-max-attempted",
            "2",
            "--promotion-max-eligible",
            "1",
        ]
        .into_iter()
        .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(options.start_horizon, None);
    assert_eq!(options.promotion_start_horizon, None);
    assert_eq!(options.promotion_sampling_temperature, None);
    assert_eq!(options.target_generation, 1);
}

#[test]
fn command_options_keep_promotion_starts_separate_from_actor_starts() {
    let options = options::Options::parse_from(
        [
            "--campaign-dir",
            "/tmp/paisho-test-campaign",
            "--target-generation",
            "1",
            "--start-horizon",
            "32",
            "--promotion-start-horizon",
            "64",
            "--promotion-start-seed",
            "71",
        ]
        .into_iter()
        .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(options.start_horizon, Some(32));
    assert_eq!(options.promotion_start_horizon, Some(64));
    assert_eq!(options.promotion_start_seed, 71);
    assert_eq!(options.promotion_sampling_temperature, Some(1.0));
    assert_eq!(options.promotion_sampling_uniform_mix, 0.05);
}

#[test]
fn command_options_reject_mcts_above_the_curriculum_ceiling() {
    let result = options::Options::parse_from(
        [
            "--campaign-dir",
            "/tmp/paisho-test-campaign",
            "--target-generation",
            "1",
            "--opponent",
            "mcts:513",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    assert!(result.is_err());
}

#[test]
fn adaptive_curriculum_has_an_independent_two_sided_elo_window() {
    let options = options::Options::parse_from(
        [
            "--campaign-dir",
            "/tmp/paisho-test-campaign",
            "--target-generation",
            "4",
            "--curriculum-dir",
            "/tmp/paisho-test-curriculum",
            "--opponent",
            "mcts:32",
            "--curriculum-lower-elo",
            "-75",
            "--curriculum-center-elo",
            "5",
            "--curriculum-upper-elo",
            "90",
            "--curriculum-start-horizon",
            "0",
            "--curriculum-sampling-temperature",
            "0",
            "--curriculum-max-attempted",
            "20",
            "--curriculum-max-eligible",
            "10",
        ]
        .into_iter()
        .map(str::to_owned),
    )
    .unwrap();
    assert_eq!(
        options.curriculum_directory,
        Some(std::path::PathBuf::from("/tmp/paisho-test-curriculum"))
    );
    assert_eq!(options.curriculum_lower_elo, -75.0);
    assert_eq!(options.curriculum_center_elo, 5.0);
    assert_eq!(options.curriculum_upper_elo, 90.0);
    assert_eq!(options.curriculum_start_horizon, None);
    assert_eq!(options.curriculum_sampling_temperature, None);
}

#[test]
fn adaptive_curriculum_accepts_only_reference_ladder_tiers() {
    let result = options::Options::parse_from(
        [
            "--campaign-dir",
            "/tmp/paisho-test-campaign",
            "--target-generation",
            "1",
            "--curriculum-dir",
            "/tmp/paisho-test-curriculum",
            "--opponent",
            "mcts:16",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    assert!(result.is_err());
}
