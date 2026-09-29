use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static TEMPORARY_DIRECTORY_COUNTER: AtomicU64 = AtomicU64::new(0);

fn options(opponent: Opponent) -> Options {
    Options {
        output_directory: PathBuf::new(),
        service: PathBuf::new(),
        checkpoint: None,
        preset: NetworkPreset::Pure,
        optimization: OptimizationLevel::Level1,
        classes: vec![ClassShape {
            capacity: 1_024,
            batch_size: 1,
        }],
        wide_lanes: 1,
        workers: 4,
        actors: 4,
        target_games: 12,
        maximum_attempts: 48,
        shard_index: 7,
        first_game_id: 500,
        decision_soft_limit: 2_048,
        maximum_batch_wait: Duration::from_millis(1),
        model_seed: 17,
        actor_seed: 91,
        selection: CandidateSelection::Sample,
        temperature: 1.0,
        uniform_mix: 0.05,
        neutral_start: None,
        opponent,
    }
}

#[test]
fn distinct_opponent_tasks_reverse_seats_on_identical_setups() {
    let options = options(Opponent::Random);
    let tasks = build_tasks(&options, 0, 12).unwrap();

    for (pair_index, pair) in tasks.chunks_exact(2).enumerate() {
        assert_eq!(pair[0].game_id + 1, pair[1].game_id);
        assert_eq!(pair[0].setup, pair[1].setup);
        assert_eq!(pair[0].setup.starting_flower, BASIC_FLOWERS[pair_index]);
        assert!(pair[0].starting_actions.is_empty());
        assert!(pair[0].neutral_start.is_none());
    }
}

#[test]
fn paired_tasks_share_one_neutral_random_prefix() {
    let mut options = options(Opponent::Random);
    options.target_games = 2;
    options.neutral_start =
        Some(NeutralStartConfigurationV1::new(20_260_905, 16, 4_096, 16).unwrap());
    let tasks = build_tasks(&options, 0, 2).unwrap();

    assert_eq!(tasks.len(), 2);
    assert_eq!(tasks[0].setup, tasks[1].setup);
    assert_eq!(tasks[0].starting_actions, tasks[1].starting_actions);
    assert!(!tasks[0].starting_actions.is_empty());
    assert_eq!(tasks[0].neutral_start, tasks[1].neutral_start);
    assert!(tasks[0]
        .neutral_start
        .is_some_and(|start| start.source_remaining_decisions().abs_diff(16) <= 1));

    let first_source_seed = tasks[0].neutral_start.unwrap().source_seed();
    options.first_game_id += 100;
    let later_tasks = build_tasks(&options, 0, 2).unwrap();
    assert_ne!(
        first_source_seed,
        later_tasks[0].neutral_start.unwrap().source_seed()
    );
}

#[test]
fn incomplete_paired_results_are_excluded_as_a_pair() {
    let mut run = CampaignRun::default();
    absorb_matches(
        &mut run,
        vec![
            Err(ReplayMatchError::DecisionLimit {
                game_id: 10,
                decisions: 20,
            }),
            Err(ReplayMatchError::DecisionLimit {
                game_id: 11,
                decisions: 20,
            }),
        ],
        Opponent::Site,
    );

    assert!(run.retained.is_empty());
    assert_eq!(run.interrupted, 2);
    assert_eq!(run.excluded_pairs, 1);
}

#[test]
fn terminal_games_without_a_network_decision_are_excluded_not_fatal() {
    let mut run = CampaignRun::default();
    absorb_matches(
        &mut run,
        vec![
            Err(ReplayMatchError::Replay(
                ReplayValidationError::NoTrainingDecision,
            )),
            Err(ReplayMatchError::Replay(
                ReplayValidationError::NoTrainingDecision,
            )),
        ],
        Opponent::Random,
    );

    assert!(run.retained.is_empty());
    assert_eq!(run.no_training_decision, 2);
    assert_eq!(run.excluded_pairs, 1);
    assert_eq!(run.failed, 0);
    assert!(run.abort_reason.is_none());
}

#[test]
fn publication_attempt_is_reserved_atomically() {
    let counter = TEMPORARY_DIRECTORY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let parent = env::temp_dir().join(format!(
        "paisho-actor-reservation-{}-{counter}",
        std::process::id()
    ));
    fs::create_dir(&parent).unwrap();

    let first = reserve_publication_directory(&parent, 3).unwrap();
    let second = reserve_publication_directory(&parent, 3).unwrap();
    assert_eq!(first.attempt, 0);
    assert_eq!(second.attempt, 1);
    assert!(first.partial_directory.exists());
    assert!(second.partial_directory.exists());

    let final_directory = first.publish().unwrap();
    assert!(final_directory.exists());
    drop(second);
    fs::remove_dir_all(parent).unwrap();
}

#[test]
fn network_descriptor_distinguishes_training_sampling_from_argmax_evaluation() {
    let mut options = options(Opponent::Random);
    let sampled = network_descriptor(&options, None, [1; 32], [2; 32]);
    assert!(sampled.contains("selection=sample;temperature=1;uniform-mix=0.05"));

    options.selection = CandidateSelection::Argmax;
    let argmax = network_descriptor(&options, None, [1; 32], [2; 32]);
    assert!(argmax.contains("selection=argmax"));
    assert!(!argmax.contains("temperature="));
    assert_ne!(digest_descriptor(&sampled), digest_descriptor(&argmax));
}
