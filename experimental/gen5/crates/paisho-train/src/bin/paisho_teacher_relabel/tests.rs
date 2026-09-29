use super::*;
use paisho_core::GameRecord;
use paisho_replay::ReplayDatasetV1;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "paisho-teacher-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn source(directory: &Path, producer_byte: u8, kind: PolicyTargetKindV1) -> ReplayGameV1 {
    fs::create_dir(directory).unwrap();
    let record: GameRecord = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr"
    ))
    .parse()
    .unwrap();
    let producer = ReplayDigestV1::from_bytes([producer_byte; 32]);
    let mut position = record.initial_position();
    let mut decisions = Vec::new();
    // Keep the fixture cheap while covering both player perspectives.
    for (index, &action) in record.actions().iter().take(2).enumerate() {
        let policy = PolicyTargetV1::new(
            kind,
            producer,
            vec![
                PolicyEntryV1::new(encode_action_v1(action, position.to_move()).unwrap(), 1.0)
                    .unwrap(),
            ],
        )
        .unwrap();
        let mut decision = ReplayDecisionV1::new(index, policy);
        if kind == PolicyTargetKindV1::Behavior {
            decision = decision.with_behavior_value(0.125);
        }
        decisions.push(decision);
        position.apply(action).unwrap();
    }
    let game = ReplayGameV1::new(42, producer, producer, record, decisions).unwrap();
    let shard = ReplayShardV1::new(0, vec![game.clone()]).unwrap();
    shard.write_new(&directory.join("source.psrshard")).unwrap();
    ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
        "source.psrshard",
        &shard,
    )
    .unwrap()])
    .unwrap()
    .write_new(&directory.join("snapshot.psrsnap"))
    .unwrap();
    game
}

#[test]
fn arguments_require_explicit_bound_seed_and_source() {
    let parse = |text: &str| Options::parse(text.split_whitespace().map(str::to_owned));
    assert!(parse("--source a --positions 0 --seed 2 --output b").is_err());
    assert!(parse("--source a --positions 2 --output b").is_err());
    assert!(parse("--positions 2 --seed 2 --output b").is_err());
    assert!(parse("--source a --positions 2 --seed 2 --output").is_err());
    let options = parse("--source a --source b --positions 2 --seed 0 --output c").unwrap();
    assert_eq!(
        options.sources,
        vec![PathBuf::from("a"), PathBuf::from("b")]
    );
    assert_eq!(options.seed, 0);
    assert_eq!(options.simulations, 8);
    assert_eq!(
        options.workers,
        std::thread::available_parallelism().map_or(1, |count| count.get())
    );
    assert_eq!(
        parse("--source a --positions 2 --seed 0 --output c --workers 2")
            .unwrap()
            .workers,
        2
    );
    assert!(parse("--source a --positions 2 --seed 0 --output c --workers 0").is_err());
    assert!(parse("--source a --positions 2 --seed 0 --output c --workers invalid").is_err());
}

#[test]
fn bounded_sample_matches_full_sort_and_depends_on_seed() {
    let candidates = |seed| {
        (0..30).map(move |index| Selection {
            rank: position_hash(
                b"PAISHO-TEACHER-SAMPLE-V1\0",
                seed,
                ReplayDigestV1::from_bytes([3; 32]),
                42,
                index,
            ),
            shard: 0,
            game: 42,
            decision: index,
        })
    };
    let mut expected: Vec<_> = candidates(7).collect();
    expected.sort();
    expected.truncate(5);
    let mut heap = BinaryHeap::new();
    for candidate in candidates(7).collect::<Vec<_>>().into_iter().rev() {
        retain_sample(&mut heap, candidate, 5);
        assert!(heap.len() <= 5);
    }
    assert_eq!(heap.into_sorted_vec(), expected);
    let mut different: Vec<_> = candidates(8).collect();
    different.sort();
    assert_ne!(
        expected.iter().map(|x| x.decision).collect::<Vec<_>>(),
        different[..5]
            .iter()
            .map(|x| x.decision)
            .collect::<Vec<_>>()
    );
}

#[test]
fn worker_counts_preserve_corpus_provenance_and_terminal_values() {
    let temporary = TemporaryDirectory::new();
    let first = source(&temporary.0.join("first"), 1, PolicyTargetKindV1::Behavior);
    let second = source(&temporary.0.join("second"), 2, PolicyTargetKindV1::Behavior);
    let mut options = Options {
        sources: vec![
            temporary.0.join("first"),
            temporary.0.join("second/snapshot.psrsnap"),
            temporary.0.join("first"),
        ],
        positions: 10,
        seed: 81,
        workers: 1,
        simulations: 8,
        output: temporary.0.join("output"),
    };
    run_with_workers(&options).unwrap();
    let snapshot = ReplaySnapshotV1::read(&options.output.join("snapshot.psrsnap")).unwrap();
    assert_eq!(
        snapshot
            .verify_directory(&options.output)
            .unwrap()
            .example_count,
        4
    );
    assert_eq!(snapshot.game_count().unwrap(), 2); // Same source game ID, different generations.
    let dataset = ReplayDatasetV1::from_snapshot(&snapshot, &options.output).unwrap();
    let examples = dataset.materialize_all().unwrap();
    assert_ne!(examples[0].perspective(), examples[1].perspective());
    for (example, original) in examples.iter().zip(
        first
            .materialize_training_examples()
            .unwrap()
            .iter()
            .chain(second.materialize_training_examples().unwrap().iter()),
    ) {
        assert_eq!(example.value_target(), original.value_target());
        assert_eq!(example.inference(), original.inference());
        assert_eq!(example.played_action(), original.played_action());
        assert_eq!(example.policy_kind(), PolicyTargetKindV1::MctsVisit);
        assert!(example.behavior_value().is_none());
        assert!((example.policy_target().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(example
            .policy_target()
            .iter()
            .all(|p| (p * 8.0).fract() == 0.0));
        example.to_training_example().unwrap();
    }
    let mut provenance: Value =
        serde_json::from_slice(&fs::read(options.output.join("provenance.json")).unwrap()).unwrap();
    assert_eq!(
        provenance["teacher"]["protocol"],
        "paisho-offline-mcts8-teacher-v1"
    );
    assert_eq!(provenance["eligible_positions"], 4);
    assert_eq!(provenance["positions"].as_array().unwrap().len(), 4);
    assert_eq!(provenance["snapshot_sha256"], snapshot.digest().to_string());
    assert!(ReplayDatasetV1::from_snapshot_for_behavior(
        &snapshot,
        &options.output,
        first.host_agent()
    )
    .is_err());
    assert!(run_with_workers(&options)
        .unwrap_err()
        .to_string()
        .contains("already exists"));
    options.output = temporary.0.join("repeat");
    options.workers = 2;
    run_with_workers(&options).unwrap();
    let repeated = ReplaySnapshotV1::read(&options.output.join("snapshot.psrsnap")).unwrap();
    assert_eq!(snapshot, repeated);
    for shard in snapshot.shards() {
        assert_eq!(
            fs::read(temporary.0.join("output").join(shard.file_name())).unwrap(),
            fs::read(options.output.join(shard.file_name())).unwrap()
        );
    }
    let mut repeated_provenance: Value =
        serde_json::from_slice(&fs::read(options.output.join("provenance.json")).unwrap()).unwrap();
    assert_eq!(
        provenance.as_object_mut().unwrap().remove("cpu_threads"),
        Some(json!(1))
    );
    assert_eq!(
        repeated_provenance
            .as_object_mut()
            .unwrap()
            .remove("cpu_threads"),
        Some(json!(2))
    );
    assert_eq!(provenance, repeated_provenance);
}

#[test]
fn non_behavior_and_corrupted_sources_do_not_publish_outputs() {
    let temporary = TemporaryDirectory::new();
    source(&temporary.0.join("teacher"), 1, PolicyTargetKindV1::Teacher);
    let options = Options {
        sources: vec![temporary.0.join("teacher")],
        positions: 1,
        seed: 0,
        workers: 1,
        simulations: 8,
        output: temporary.0.join("output"),
    };
    assert!(run_with_workers(&options)
        .unwrap_err()
        .to_string()
        .contains("no recorded Behavior"));
    assert!(!options.output.exists());
    let path = temporary.0.join("teacher/source.psrshard");
    let mut bytes = fs::read(&path).unwrap();
    bytes[20] ^= 1;
    fs::write(path, bytes).unwrap();
    assert!(run_with_workers(&options).is_err());
    assert!(!options.output.exists());
}

#[test]
fn simulation_budget_accepts_only_current_bounded_teachers() {
    for budget in [8, 32] {
        let args = format!("--source a --positions 2 --seed 0 --output b --simulations {budget}");
        assert_eq!(
            Options::parse(args.split_whitespace().map(str::to_owned))
                .unwrap()
                .simulations,
            budget
        );
    }
    for budget in [0, 16, 128, 512] {
        let args = format!("--source a --positions 2 --seed 0 --output b --simulations {budget}");
        assert!(Options::parse(args.split_whitespace().map(str::to_owned)).is_err());
    }
}

#[test]
fn mcts32_publishes_legal_visit_targets_with_distinct_provenance() {
    let temporary = TemporaryDirectory::new();
    let original = source(&temporary.0.join("source"), 1, PolicyTargetKindV1::Behavior);
    let options = Options {
        sources: vec![temporary.0.join("source")],
        positions: 2,
        seed: 81,
        workers: 1,
        simulations: 32,
        output: temporary.0.join("output"),
    };
    run_with_workers(&options).unwrap();
    let snapshot = ReplaySnapshotV1::read(&options.output.join("snapshot.psrsnap")).unwrap();
    snapshot.verify_directory(&options.output).unwrap();
    let dataset = ReplayDatasetV1::from_snapshot(&snapshot, &options.output).unwrap();
    let examples = dataset.materialize_all().unwrap();
    for (example, original) in examples
        .iter()
        .zip(original.materialize_training_examples().unwrap())
    {
        assert_eq!(example.inference(), original.inference());
        assert_eq!(example.value_target(), original.value_target());
        assert_eq!(example.policy_kind(), PolicyTargetKindV1::MctsVisit);
        assert!((example.policy_target().iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(example
            .policy_target()
            .iter()
            .all(|p| (p * 32.0).fract() == 0.0));
        example.to_training_example().unwrap();
    }
    let provenance: Value =
        serde_json::from_slice(&fs::read(options.output.join("provenance.json")).unwrap()).unwrap();
    assert_eq!(
        provenance["teacher"]["protocol"],
        "paisho-offline-mcts32-teacher-v1"
    );
    assert!(provenance["teacher"]["configuration"]
        .as_str()
        .unwrap()
        .contains("simulations: 32"));
    let position = original.record().initial_position();
    let legal = legal_actions(&position);
    let report = MctsAgent::new(81, teacher_config(32))
        .unwrap()
        .search(&position, &legal);
    assert_eq!(report.actions.iter().map(|a| a.visits).sum::<usize>(), 32);
    assert!(report.actions.iter().all(|a| legal.contains(&a.action)));
}
