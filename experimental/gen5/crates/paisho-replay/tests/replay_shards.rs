use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use paisho_core::{legal_actions, GameRecord, Player};
use paisho_model::encode_action_v1;
use paisho_replay::{
    PolicyEntryV1, PolicyTargetKindV1, PolicyTargetV1, ReplayCodecV1Error, ReplayDatasetV1,
    ReplayDatasetV1Error, ReplayDecisionV1, ReplayDigestV1, ReplayGameV1, ReplaySamplerStateV1,
    ReplaySamplerV1, ReplaySamplerV1Error, ReplayShardReferenceV1, ReplayShardV1,
    ReplayShardV1Error, ReplaySnapshotV1, ReplaySnapshotV1Error, ReplayTerminalPpoExampleError,
    ReplayValidationError,
};

const TERMINAL_RING: &str =
    include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
static NEXT_TEMP_DIRECTORY: AtomicU64 = AtomicU64::new(0);

fn temporary_directory() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let sequence = NEXT_TEMP_DIRECTORY.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "paisho-replay-test-{}-{nonce}-{sequence}",
        std::process::id()
    ))
}

fn replay_game(indices: &[usize]) -> ReplayGameV1 {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let producer = ReplayDigestV1::from_bytes([7; 32]);
    let mut position = record.initial_position();
    let mut decisions = Vec::new();
    for (index, action) in record.actions().iter().copied().enumerate() {
        if indices.contains(&index) {
            let encoded = encode_action_v1(action, position.to_move()).unwrap();
            decisions.push(ReplayDecisionV1::new(
                index,
                PolicyTargetV1::one_hot(producer, encoded).unwrap(),
            ));
        }
        position.apply(action).unwrap();
    }
    ReplayGameV1::new(
        42,
        ReplayDigestV1::from_bytes([1; 32]),
        ReplayDigestV1::from_bytes([2; 32]),
        record,
        decisions,
    )
    .unwrap()
}

fn replay_game_with_targets(
    game_id: u64,
    targets: &[(usize, PolicyTargetKindV1, ReplayDigestV1)],
) -> ReplayGameV1 {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let mut position = record.initial_position();
    let mut decisions = Vec::new();
    let mut host_agent = None;
    let mut guest_agent = None;
    for (index, action) in record.actions().iter().copied().enumerate() {
        if let Some((_, kind, producer)) = targets.iter().find(|target| target.0 == index) {
            if *kind == PolicyTargetKindV1::Behavior {
                let acting_agent = match position.to_move() {
                    Player::Host => &mut host_agent,
                    Player::Guest => &mut guest_agent,
                };
                if let Some(previous) = acting_agent {
                    assert_eq!(*previous, *producer);
                } else {
                    *acting_agent = Some(*producer);
                }
            }
            let encoded = encode_action_v1(action, position.to_move()).unwrap();
            decisions.push(ReplayDecisionV1::new(
                index,
                PolicyTargetV1::new(
                    *kind,
                    *producer,
                    vec![PolicyEntryV1::new(encoded, 1.0).unwrap()],
                )
                .unwrap(),
            ));
        }
        position.apply(action).unwrap();
    }
    ReplayGameV1::new(
        game_id,
        host_agent.unwrap_or_else(|| ReplayDigestV1::from_bytes([1; 32])),
        guest_agent.unwrap_or_else(|| ReplayDigestV1::from_bytes([2; 32])),
        record,
        decisions,
    )
    .unwrap()
}

fn behavior_game_with_value(
    game_id: u64,
    behavior_value: f32,
) -> Result<ReplayGameV1, ReplayValidationError> {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let producer = ReplayDigestV1::from_bytes([23; 32]);
    let position = record.initial_position();
    let action = record.actions()[0];
    let encoded = encode_action_v1(action, position.to_move()).unwrap();
    let policy = PolicyTargetV1::new(
        PolicyTargetKindV1::Behavior,
        producer,
        vec![PolicyEntryV1::new(encoded, 1.0).unwrap()],
    )
    .unwrap();
    ReplayGameV1::new(
        game_id,
        producer,
        producer,
        record,
        vec![ReplayDecisionV1::new(0, policy).with_behavior_value(behavior_value)],
    )
}

#[test]
fn terminal_bonus_decisions_keep_the_real_players_value_perspective() {
    let game = replay_game(&[4, 5]);
    let examples = game.materialize_training_examples().unwrap();

    assert_eq!(examples.len(), 2);
    assert_eq!(examples[0].perspective(), examples[1].perspective());
    assert_eq!(examples[0].value_target(), examples[1].value_target());
    for example in &examples {
        let played_index = example
            .inference()
            .legal_actions()
            .iter()
            .position(|&action| action == example.played_action())
            .unwrap();
        assert_eq!(example.policy_target()[played_index], 1.0);
        assert_eq!(example.policy_target().iter().sum::<f32>(), 1.0);
    }
    assert_eq!(game.materialize_training_example(0).unwrap(), examples[0]);
    assert_eq!(game.materialize_training_example(1).unwrap(), examples[1]);
    assert!(matches!(
        game.materialize_training_example(2),
        Err(ReplayValidationError::SelectedDecisionOutOfRange {
            selected_decision: 2,
            decision_count: 2,
        })
    ));
}

#[test]
fn behavior_examples_use_winner_and_loser_once_from_their_own_perspectives() {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let producer = ReplayDigestV1::from_bytes([21; 32]);
    let mut position = record.initial_position();
    let mut first_host = None;
    let mut first_guest = None;
    for (index, &action) in record.actions().iter().enumerate() {
        match position.to_move() {
            paisho_core::Player::Host => first_host.get_or_insert(index),
            paisho_core::Player::Guest => first_guest.get_or_insert(index),
        };
        position.apply(action).unwrap();
    }
    let game = replay_game_with_targets(
        43,
        &[
            (first_host.unwrap(), PolicyTargetKindV1::Behavior, producer),
            (first_guest.unwrap(), PolicyTargetKindV1::Behavior, producer),
        ],
    );
    let examples = game.materialize_training_examples().unwrap();

    assert_eq!(examples.len(), 2);
    assert_ne!(examples[0].perspective(), examples[1].perspective());
    assert_eq!(
        examples
            .iter()
            .map(|example| example.terminal_return())
            .sum::<f32>(),
        0.0
    );
    assert!(examples
        .iter()
        .any(|example| example.terminal_return() == 1.0));
    assert!(examples
        .iter()
        .any(|example| example.terminal_return() == -1.0));
    for example in examples {
        assert_eq!(example.played_behavior_probability(), Some(1.0));
        assert_eq!(
            example.inference().legal_actions()[example.played_action_index()],
            example.played_action()
        );
        let terminal = example.to_terminal_ppo_example(0.25).unwrap();
        assert_eq!(
            terminal.played_action_index(),
            example.played_action_index()
        );
        assert_eq!(terminal.behavior_probability(), 1.0);
        assert_eq!(terminal.terminal_value(), example.value_class());
        assert_eq!(terminal.advantage(), example.terminal_return() - 0.25);
    }
}

#[test]
fn frozen_behavior_value_round_trips_and_marks_a_complete_dataset() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let shard = ReplayShardV1::new(18, vec![behavior_game_with_value(45, 0.375).unwrap()]).unwrap();
    let shard_name = "shard-0018.psrbuf";
    shard.write_new(&directory.join(shard_name)).unwrap();
    let restored = ReplayShardV1::read(&directory.join(shard_name)).unwrap();
    assert_eq!(
        restored.games()[0].decisions()[0].behavior_value(),
        Some(0.375)
    );
    assert_eq!(
        restored.training_examples().unwrap()[0].behavior_value(),
        Some(0.375)
    );

    let snapshot = ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
        shard_name, &restored,
    )
    .unwrap()])
    .unwrap();
    let dataset = ReplayDatasetV1::from_snapshot_for_behavior(
        &snapshot,
        &directory,
        ReplayDigestV1::from_bytes([23; 32]),
    )
    .unwrap();
    assert!(dataset.has_complete_behavior_values());
    fs::remove_file(directory.join(shard_name)).unwrap();
    assert!(ReplayDatasetV1::from_snapshot_for_behavior(
        &snapshot,
        &directory,
        ReplayDigestV1::from_bytes([23; 32]),
    )
    .is_err()); // Disk recovery still checks storage; RAM needs no file.
    let resident = ReplayDatasetV1::from_shard_for_behavior(
        shard,
        shard_name,
        ReplayDigestV1::from_bytes([23; 32]),
    )
    .unwrap();
    assert_eq!(resident.snapshot_digest(), dataset.snapshot_digest());
    assert_eq!(
        resident.materialize_all().unwrap(),
        dataset.materialize_all().unwrap()
    );
    let mut disk_sampler = ReplaySamplerV1::new(&dataset, 73);
    let mut ram_sampler = ReplaySamplerV1::new(&resident, 73);
    assert_eq!(
        disk_sampler.prepare_batch(8).unwrap().examples(),
        ram_sampler.prepare_batch(8).unwrap().examples()
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn behavior_values_are_bounded_and_reserved_for_behavior_policies() {
    assert!(matches!(
        behavior_game_with_value(46, f32::NAN),
        Err(ReplayValidationError::InvalidBehaviorValue {
            decision_index: 0,
            value,
        }) if value.is_nan()
    ));

    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let producer = ReplayDigestV1::from_bytes([24; 32]);
    let position = record.initial_position();
    let encoded = encode_action_v1(record.actions()[0], position.to_move()).unwrap();
    let decision = ReplayDecisionV1::new(0, PolicyTargetV1::one_hot(producer, encoded).unwrap())
        .with_behavior_value(0.0);
    assert!(matches!(
        ReplayGameV1::new(47, producer, producer, record, vec![decision]),
        Err(ReplayValidationError::BehaviorValueWithoutBehaviorPolicy {
            decision_index: 0,
            policy_kind: PolicyTargetKindV1::PlayedAction,
        })
    ));
}

#[test]
fn terminal_ppo_conversion_rejects_non_behavior_policy() {
    let game = replay_game(&[5]);
    let example = game.materialize_training_examples().unwrap().remove(0);

    assert!(matches!(
        example.to_terminal_ppo_example(0.0),
        Err(ReplayTerminalPpoExampleError::NotBehaviorPolicy(
            PolicyTargetKindV1::PlayedAction
        ))
    ));
}

#[test]
fn policy_gradient_dataset_selects_only_the_requested_behavior_producer() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let selected = ReplayDigestV1::from_bytes([31; 32]);
    let other = ReplayDigestV1::from_bytes([32; 32]);
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let mut position = record.initial_position();
    let mut first_host = None;
    let mut first_guest = None;
    for (index, &action) in record.actions().iter().enumerate() {
        match position.to_move() {
            Player::Host => first_host.get_or_insert(index),
            Player::Guest => first_guest.get_or_insert(index),
        };
        position.apply(action).unwrap();
    }
    let first_host = first_host.unwrap();
    let first_guest = first_guest.unwrap();
    let mcts_decision = (0..record.actions().len())
        .find(|&index| index != first_host && index != first_guest)
        .unwrap();
    let shard = ReplayShardV1::new(
        19,
        vec![replay_game_with_targets(
            44,
            &[
                (first_host, PolicyTargetKindV1::Behavior, selected),
                (mcts_decision, PolicyTargetKindV1::MctsVisit, other),
                (first_guest, PolicyTargetKindV1::Behavior, other),
            ],
        )],
    )
    .unwrap();
    let shard_name = "shard-0019.psrbuf";
    shard.write_new(&directory.join(shard_name)).unwrap();
    let snapshot =
        ReplaySnapshotV1::new(vec![
            ReplayShardReferenceV1::from_shard(shard_name, &shard).unwrap()
        ])
        .unwrap();
    let dataset =
        ReplayDatasetV1::from_snapshot_for_behavior(&snapshot, &directory, selected).unwrap();

    assert_eq!(dataset.len(), 1);
    let resident = ReplayDatasetV1::from_shard_for_behavior(shard, shard_name, selected).unwrap();
    assert_eq!(
        resident.materialize_all().unwrap(),
        dataset.materialize_all().unwrap()
    );
    assert!(!dataset.has_complete_behavior_values());
    let mut sampler = ReplaySamplerV1::new(&dataset, 7);
    let batch = sampler.prepare_batch(1).unwrap();
    assert_eq!(
        batch.examples()[0].policy_kind(),
        PolicyTargetKindV1::Behavior
    );
    assert_eq!(batch.examples()[0].policy_producer(), selected);
    assert!(batch.examples()[0].played_behavior_probability().unwrap() > 0.0);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn shard_round_trip_is_atomic_canonical_and_checksum_verified() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let destination = directory.join("shard-0007.psrbuf");
    let shard = ReplayShardV1::new(7, vec![replay_game(&[5, 4])]).unwrap();

    let digest = shard.write_new(&destination).unwrap();
    assert_eq!(digest, shard.digest().unwrap());
    assert_eq!(ReplayShardV1::read(&destination).unwrap(), shard);
    assert!(matches!(
        shard.write_new(&destination),
        Err(ReplayShardV1Error::DestinationExists(path)) if path == destination
    ));

    let mut corrupted = fs::read(&destination).unwrap();
    corrupted[20] ^= 1;
    let corrupted_path = directory.join("corrupted.psrbuf");
    fs::write(&corrupted_path, corrupted).unwrap();
    assert!(matches!(
        ReplayShardV1::read(&corrupted_path),
        Err(ReplayShardV1Error::Codec(
            ReplayCodecV1Error::ChecksumMismatch
        ))
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn incomplete_games_and_nonlegal_policy_addresses_are_rejected() {
    let complete = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let producer = ReplayDigestV1::from_bytes([9; 32]);
    let opening = complete.initial_position();
    let first = complete.actions()[0];
    let first_encoded = encode_action_v1(first, opening.to_move()).unwrap();
    let decision =
        ReplayDecisionV1::new(0, PolicyTargetV1::one_hot(producer, first_encoded).unwrap());
    let mut incomplete = GameRecord::with_rules(complete.setup(), complete.rules());
    incomplete.push(first);
    assert!(matches!(
        ReplayGameV1::new(
            1,
            ReplayDigestV1::from_bytes([1; 32]),
            ReplayDigestV1::from_bytes([2; 32]),
            incomplete,
            vec![decision.clone()],
        ),
        Err(ReplayValidationError::NonTerminalGame(1))
    ));

    let other_action = encode_action_v1(complete.actions()[1], opening.to_move()).unwrap();
    let wrong_policy = ReplayDecisionV1::new(
        0,
        PolicyTargetV1::new(
            PolicyTargetKindV1::Behavior,
            producer,
            vec![PolicyEntryV1::new(other_action, 1.0).unwrap()],
        )
        .unwrap(),
    );
    assert!(matches!(
        ReplayGameV1::new(
            2,
            ReplayDigestV1::from_bytes([1; 32]),
            ReplayDigestV1::from_bytes([2; 32]),
            complete,
            vec![wrong_policy],
        ),
        Err(ReplayValidationError::PolicyActionNotLegal {
            decision_index: 0,
            ..
        })
    ));
}

#[test]
fn played_action_target_must_name_the_action_that_was_played() {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let opening = record.initial_position();
    let played = record.actions()[0];
    let other_legal = legal_actions(&opening)
        .into_iter()
        .find(|&action| action != played)
        .unwrap();
    let target = PolicyTargetV1::one_hot(
        ReplayDigestV1::from_bytes([8; 32]),
        encode_action_v1(other_legal, opening.to_move()).unwrap(),
    )
    .unwrap();

    assert!(matches!(
        ReplayGameV1::new(
            3,
            ReplayDigestV1::from_bytes([1; 32]),
            ReplayDigestV1::from_bytes([2; 32]),
            record,
            vec![ReplayDecisionV1::new(0, target)],
        ),
        Err(ReplayValidationError::PlayedActionTargetMismatch {
            decision_index: 0,
            ..
        })
    ));
}

#[test]
fn behavior_policy_must_assign_positive_probability_to_the_played_action() {
    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let opening = record.initial_position();
    let played = record.actions()[0];
    let other_legal = legal_actions(&opening)
        .into_iter()
        .find(|&action| action != played)
        .unwrap();
    let target = PolicyTargetV1::new(
        PolicyTargetKindV1::Behavior,
        ReplayDigestV1::from_bytes([18; 32]),
        vec![PolicyEntryV1::new(
            encode_action_v1(other_legal, opening.to_move()).unwrap(),
            1.0,
        )
        .unwrap()],
    )
    .unwrap();

    assert!(matches!(
        ReplayGameV1::new(
            4,
            ReplayDigestV1::from_bytes([1; 32]),
            ReplayDigestV1::from_bytes([2; 32]),
            record,
            vec![ReplayDecisionV1::new(0, target)],
        ),
        Err(ReplayValidationError::BehaviorExcludesPlayedAction {
            decision_index: 0,
            ..
        })
    ));

    let record = TERMINAL_RING.parse::<GameRecord>().unwrap();
    let opening = record.initial_position();
    let played = record.actions()[0];
    let producer = ReplayDigestV1::from_bytes([18; 32]);
    let target = PolicyTargetV1::new(
        PolicyTargetKindV1::Behavior,
        producer,
        vec![
            PolicyEntryV1::new(encode_action_v1(played, opening.to_move()).unwrap(), 1.0).unwrap(),
        ],
    )
    .unwrap();

    assert!(matches!(
        ReplayGameV1::new(
            5,
            ReplayDigestV1::from_bytes([1; 32]),
            ReplayDigestV1::from_bytes([2; 32]),
            record,
            vec![ReplayDecisionV1::new(0, target)],
        ),
        Err(ReplayValidationError::BehaviorProducerMismatch {
            decision_index: 0,
            producer: actual_producer,
            ..
        }) if actual_producer == producer
    ));
}

#[test]
fn snapshot_binds_the_exact_ordered_shards_and_verifies_the_directory() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let shard = ReplayShardV1::new(11, vec![replay_game(&[4, 5])]).unwrap();
    let shard_name = "shard-0011.psrbuf";
    shard.write_new(&directory.join(shard_name)).unwrap();
    let reference = ReplayShardReferenceV1::from_shard(shard_name, &shard).unwrap();
    let snapshot = ReplaySnapshotV1::new(vec![reference]).unwrap();
    let snapshot_path = directory.join("snapshot-0001.psrsnap");

    let digest = snapshot.write_new(&snapshot_path).unwrap();
    let restored = ReplaySnapshotV1::read(&snapshot_path).unwrap();
    assert_eq!(restored, snapshot);
    assert_eq!(restored.digest(), digest);
    let verification = restored.verify_directory(&directory).unwrap();
    assert_eq!(verification.shard_count, 1);
    assert_eq!(verification.game_count, 1);
    assert_eq!(verification.example_count, 2);
    assert_eq!(verification.digest, digest);
    assert_eq!(
        restored.verify_directory_integrity(&directory).unwrap(),
        verification
    );

    let shard_path = directory.join(shard_name);
    let original_shard = fs::read(&shard_path).unwrap();
    let mut damaged_shard = original_shard.clone();
    damaged_shard[20] ^= 1;
    fs::write(&shard_path, damaged_shard).unwrap();
    assert!(matches!(
        restored.verify_directory_integrity(&directory),
        Err(ReplaySnapshotV1Error::InvalidShard { .. })
    ));
    fs::write(&shard_path, original_shard).unwrap();

    let mut damaged_snapshot = fs::read(&snapshot_path).unwrap();
    let digit = damaged_snapshot
        .iter()
        .position(|byte| *byte == b'1')
        .unwrap();
    damaged_snapshot[digit] = b'2';
    let damaged_path = directory.join("damaged.psrsnap");
    fs::write(&damaged_path, damaged_snapshot).unwrap();
    assert!(matches!(
        ReplaySnapshotV1::read(&damaged_path),
        Err(ReplaySnapshotV1Error::ChecksumMismatch) | Err(ReplaySnapshotV1Error::InvalidHeader(_))
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn snapshot_verification_rejects_duplicate_game_ids_across_shards() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let first = ReplayShardV1::new(1, vec![replay_game(&[4])]).unwrap();
    let second = ReplayShardV1::new(2, vec![replay_game(&[5])]).unwrap();
    let first_name = "shard-0001.psrbuf";
    let second_name = "shard-0002.psrbuf";
    first.write_new(&directory.join(first_name)).unwrap();
    second.write_new(&directory.join(second_name)).unwrap();
    let snapshot = ReplaySnapshotV1::new(vec![
        ReplayShardReferenceV1::from_shard(first_name, &first).unwrap(),
        ReplayShardReferenceV1::from_shard(second_name, &second).unwrap(),
    ])
    .unwrap();

    assert!(matches!(
        snapshot.verify_directory(&directory),
        Err(ReplaySnapshotV1Error::DuplicateGameId { game_id: 42, .. })
    ));
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn concurrent_publication_never_overwrites_an_immutable_shard() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let destination = directory.join("contended.psrbuf");
    let first = ReplayShardV1::new(20, vec![replay_game(&[4])]).unwrap();
    let second = ReplayShardV1::new(21, vec![replay_game(&[5])]).unwrap();

    let outcomes = std::thread::scope(|scope| {
        let left = scope.spawn(|| first.write_new(&destination));
        let right = scope.spawn(|| second.write_new(&destination));
        [left.join().unwrap(), right.join().unwrap()]
    });
    assert_eq!(outcomes.iter().filter(|result| result.is_ok()).count(), 1);
    let published = ReplayShardV1::read(&destination).unwrap();
    assert!(published == first || published == second);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sampler_resume_is_exact_across_epoch_boundaries() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let shard = ReplayShardV1::new(31, vec![replay_game(&[0, 1, 4, 5])]).unwrap();
    let shard_name = "shard-0031.psrbuf";
    shard.write_new(&directory.join(shard_name)).unwrap();
    let snapshot =
        ReplaySnapshotV1::new(vec![
            ReplayShardReferenceV1::from_shard(shard_name, &shard).unwrap()
        ])
        .unwrap();
    let dataset = ReplayDatasetV1::from_snapshot(&snapshot, &directory).unwrap();
    assert_eq!(dataset.len(), 4);
    assert_eq!(
        dataset
            .materialize_all()
            .unwrap()
            .iter()
            .map(|example| example.decision_index())
            .collect::<Vec<_>>(),
        vec![0, 1, 4, 5]
    );

    let mut uninterrupted = ReplaySamplerV1::new(&dataset, 0x1234_5678);
    let expected_batch = uninterrupted.prepare_batch(10).unwrap();
    assert_eq!(uninterrupted.state().next_replay_index(), 0);
    uninterrupted.commit_batch(&expected_batch).unwrap();
    let expected = expected_batch.into_examples();

    dataset.preload().unwrap();
    assert_eq!(dataset.materialize_all().unwrap().len(), dataset.len());
    let mut preloaded = ReplaySamplerV1::new(&dataset, 0x1234_5678);
    let preloaded_batch = preloaded.prepare_batch(10).unwrap();
    assert_eq!(preloaded_batch.examples(), expected);

    let mut first_run = ReplaySamplerV1::new(&dataset, 0x1234_5678);
    let first_batch = first_run.prepare_batch(3).unwrap();
    first_run.commit_batch(&first_batch).unwrap();
    let mut resumed_examples = first_batch.into_examples();
    let state = first_run.state();
    let mut resumed = ReplaySamplerV1::resume(&dataset, state).unwrap();
    let second_batch = resumed.prepare_batch(7).unwrap();
    assert_eq!(second_batch.start_replay_index(), 3);
    assert_eq!(second_batch.next_replay_index(), 10);
    assert_eq!(resumed.state().next_replay_index(), 3);
    resumed.commit_batch(&second_batch).unwrap();
    assert!(matches!(
        resumed.commit_batch(&second_batch),
        Err(ReplaySamplerV1Error::PreparedBatchStateMismatch { .. })
    ));
    resumed_examples.extend(second_batch.into_examples());
    assert_eq!(resumed_examples, expected);

    let first_epoch = expected[..dataset.len()]
        .iter()
        .map(|example| example.decision_index())
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(first_epoch.len(), dataset.len());
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn sampler_refuses_another_snapshot_and_zero_sized_batches_do_not_advance() {
    let directory = temporary_directory();
    fs::create_dir_all(&directory).unwrap();
    let shard = ReplayShardV1::new(32, vec![replay_game(&[4, 5])]).unwrap();
    let shard_name = "shard-0032.psrbuf";
    shard.write_new(&directory.join(shard_name)).unwrap();
    let snapshot =
        ReplaySnapshotV1::new(vec![
            ReplayShardReferenceV1::from_shard(shard_name, &shard).unwrap()
        ])
        .unwrap();
    let dataset = ReplayDatasetV1::from_snapshot(&snapshot, &directory).unwrap();
    let wrong_state = ReplaySamplerStateV1::new(ReplayDigestV1::from_bytes([0xff; 32]), 1, 7);
    assert!(matches!(
        ReplaySamplerV1::resume(&dataset, wrong_state),
        Err(ReplaySamplerV1Error::SnapshotMismatch { .. })
    ));

    let mut sampler = ReplaySamplerV1::new(&dataset, 1);
    assert!(matches!(
        sampler.prepare_batch(0),
        Err(ReplaySamplerV1Error::ZeroBatchSize)
    ));
    assert_eq!(sampler.state().next_replay_index(), 0);

    let empty = ReplaySnapshotV1::new(Vec::new()).unwrap();
    assert!(matches!(
        ReplayDatasetV1::from_snapshot(&empty, &directory),
        Err(ReplayDatasetV1Error::Empty)
    ));
    fs::remove_dir_all(directory).unwrap();
}
