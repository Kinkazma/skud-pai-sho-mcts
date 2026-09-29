use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use paisho_mpsgraph_client::{NetworkPreset, OptimizationLevel};
use paisho_replay::ReplayDigestV1;
use paisho_train::{
    discover_latest_commit, run_learner, LearnerCommit, LearnerCommitError, LearnerConfiguration,
    LearnerError, LearnerIdentityV1, LearnerObjectiveConfiguration, LearnerObjectiveV1,
};
use sha2::{Digest, Sha256};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[test]
fn latest_commit_ignores_a_checkpoint_without_a_manifest() {
    let temporary = TemporaryDirectory::new();
    let identity = identity();
    let first_digest = write_checkpoint(
        &temporary.path.join(checkpoint_name(0, 2, 0)),
        b"committed checkpoint",
    );
    let first = LearnerCommit::new(
        identity,
        0,
        2,
        16,
        3,
        checkpoint_name(0, 2, 0),
        first_digest,
    )
    .unwrap();
    let manifest = first.write_idempotently(&temporary.path).unwrap();
    assert_eq!(LearnerCommit::read(&manifest).unwrap(), first);

    write_checkpoint(
        &temporary.path.join(checkpoint_name(0, 3, 0)),
        b"orphan checkpoint",
    );
    assert_eq!(
        discover_latest_commit(&temporary.path, identity).unwrap(),
        Some(first)
    );
}

#[test]
fn commit_publication_is_idempotent_only_for_identical_contents() {
    let temporary = TemporaryDirectory::new();
    let identity = identity();
    let checkpoint = temporary.path.join(checkpoint_name(0, 2, 0));
    let digest = write_checkpoint(&checkpoint, b"same checkpoint");
    let commit =
        LearnerCommit::new(identity, 0, 2, 16, 3, checkpoint_name(0, 2, 0), digest).unwrap();
    commit.write_idempotently(&temporary.path).unwrap();
    commit.write_idempotently(&temporary.path).unwrap();

    let conflicting =
        LearnerCommit::new(identity, 0, 2, 16, 4, checkpoint_name(0, 2, 0), digest).unwrap();
    assert!(matches!(
        conflicting.write_idempotently(&temporary.path),
        Err(LearnerCommitError::Io(error))
            if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
}

#[test]
fn latest_commit_rejects_identity_and_checkpoint_corruption() {
    let temporary = TemporaryDirectory::new();
    let identity = identity();
    let checkpoint = temporary.path.join(checkpoint_name(0, 1, 0));
    let digest = write_checkpoint(&checkpoint, b"valid checkpoint");
    LearnerCommit::new(identity, 0, 1, 8, 2, checkpoint_name(0, 1, 0), digest)
        .unwrap()
        .write_idempotently(&temporary.path)
        .unwrap();

    let mut other = identity;
    other.sampler_seed += 1;
    assert!(matches!(
        discover_latest_commit(&temporary.path, other),
        Err(LearnerCommitError::IdentityMismatch(_))
    ));

    let mut bytes = fs::read(&checkpoint).unwrap();
    bytes[0] ^= 1;
    fs::write(&checkpoint, bytes).unwrap();
    assert!(matches!(
        discover_latest_commit(&temporary.path, identity),
        Err(LearnerCommitError::Checkpoint(_))
    ));
}

#[test]
fn commit_rejects_a_cursor_that_does_not_match_fixed_batches() {
    assert!(matches!(
        LearnerCommit::new(identity(), 0, 2, 15, 3, checkpoint_name(0, 2, 0), [0; 32]),
        Err(LearnerCommitError::ReplayIndexMismatch {
            expected: 16,
            actual: 15
        })
    ));
}

#[test]
fn commit_v3_round_trip_preserves_generation_start_and_local_replay_progress() {
    let temporary = TemporaryDirectory::new();
    let mut identity = identity();
    identity.generation = 4;
    let checkpoint_name = checkpoint_name(4, 103, 0);
    let digest = write_checkpoint(
        &temporary.path.join(&checkpoint_name),
        b"generation child checkpoint",
    );
    let commit = LearnerCommit::new(identity, 100, 103, 24, 7, checkpoint_name, digest).unwrap();
    let path = commit.write_idempotently(&temporary.path).unwrap();
    let restored = LearnerCommit::read(&path).unwrap();
    assert_eq!(restored, commit);
    assert_eq!(restored.format_version(), 3);
    assert_eq!(restored.starting_training_step(), 100);
    assert_eq!(restored.training_step(), 103);
    assert_eq!(restored.next_replay_index(), 24);
    let text = fs::read_to_string(path).unwrap();
    assert!(text.starts_with("PAISHO-LEARNER-COMMIT\t3\n"));
    assert!(text.contains("objective\tsupervised-policy-value-v1\n"));
    assert!(text.contains("starting-training-step\t100\n"));
}

#[test]
fn commit_rejects_a_global_step_before_its_generation_origin() {
    let mut identity = identity();
    identity.generation = 4;
    assert!(matches!(
        LearnerCommit::new(
            identity,
            101,
            100,
            0,
            1,
            checkpoint_name(4, 100, 0),
            [0; 32]
        ),
        Err(LearnerCommitError::TrainingStepPrecedesGeneration {
            starting: 101,
            actual: 100
        })
    ));
}

#[test]
fn discovery_selects_the_highest_committed_step_and_accepts_attempt_suffixes() {
    let temporary = TemporaryDirectory::new();
    let identity = identity();
    let mut first_manifest = None;
    for (step, attempt) in [(1, 0), (2, 3)] {
        let checkpoint_name = checkpoint_name(0, step, attempt);
        let digest = write_checkpoint(
            &temporary.path.join(&checkpoint_name),
            format!("checkpoint {step}/{attempt}").as_bytes(),
        );
        let manifest = LearnerCommit::new(
            identity,
            0,
            step,
            step * 8,
            step + 10,
            checkpoint_name,
            digest,
        )
        .unwrap()
        .write_idempotently(&temporary.path)
        .unwrap();
        if step == 1 {
            first_manifest = Some(manifest);
        }
    }
    let first_manifest = first_manifest.unwrap();
    let text = fs::read_to_string(&first_manifest).unwrap();
    fs::write(
        &first_manifest,
        text.replace("next-request-id\t11", "next-request-id\t12"),
    )
    .unwrap();
    let latest = discover_latest_commit(&temporary.path, identity)
        .unwrap()
        .unwrap();
    assert_eq!(latest.training_step(), 2);
    assert_eq!(latest.checkpoint_file(), checkpoint_name(0, 2, 3));
}

#[test]
fn manifest_checksum_detects_progress_tampering() {
    let temporary = TemporaryDirectory::new();
    let checkpoint_name = checkpoint_name(0, 1, 0);
    let digest = write_checkpoint(
        &temporary.path.join(&checkpoint_name),
        b"checkpoint for manifest tampering",
    );
    let manifest = LearnerCommit::new(identity(), 0, 1, 8, 2, checkpoint_name, digest)
        .unwrap()
        .write_idempotently(&temporary.path)
        .unwrap();
    let text = fs::read_to_string(&manifest).unwrap();
    fs::write(
        &manifest,
        text.replace("next-request-id\t2", "next-request-id\t3"),
    )
    .unwrap();
    assert!(matches!(
        LearnerCommit::read(&manifest),
        Err(LearnerCommitError::ChecksumMismatch)
    ));
    assert!(matches!(
        discover_latest_commit(&temporary.path, identity()),
        Err(LearnerCommitError::InvalidCommit { .. })
    ));
}

#[cfg(unix)]
#[test]
fn non_utf8_run_directory_is_rejected_before_external_inputs_are_opened() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let temporary = TemporaryDirectory::new();
    let run_directory = temporary
        .path
        .join(OsString::from_vec(b"learner-\xff".to_vec()));
    let configuration = LearnerConfiguration {
        service_executable: temporary.path.join("missing-service"),
        replay_snapshot: temporary.path.join("missing-snapshot"),
        replay_directory: temporary.path.clone(),
        run_directory,
        initial_checkpoint: None,
        network_preset: NetworkPreset::Micro,
        optimization: OptimizationLevel::Level1,
        batch_size: 8,
        legal_action_capacity: 1024,
        model_seed: 17,
        sampler_seed: 23,
        generation: 0,
        learning_rate: 1.0e-4,
        objective: LearnerObjectiveConfiguration::SupervisedPolicyValue,
        target_training_step: 1,
        checkpoint_interval: 1,
    };
    let result = run_learner(&configuration);
    assert!(
        matches!(
            result,
            Err(LearnerError::NonUtf8Path(_) | LearnerError::Io(_))
        ),
        "unexpected result: {result:?}"
    );
}

fn identity() -> LearnerIdentityV1 {
    LearnerIdentityV1 {
        replay_snapshot: ReplayDigestV1::from_bytes([0x11; 32]),
        network_preset: NetworkPreset::Micro,
        optimization: OptimizationLevel::Level1,
        batch_size: 8,
        legal_action_capacity: 1024,
        model_seed: 17,
        sampler_seed: 23,
        generation: 0,
        learning_rate: 1.0e-4,
        objective: LearnerObjectiveV1::SupervisedPolicyValue,
    }
}

fn checkpoint_name(generation: u64, step: u64, attempt: u64) -> String {
    format!("checkpoint-g{generation:020}-s{step:020}-a{attempt:020}.psckpt")
}

fn write_checkpoint(path: &Path, content: &[u8]) -> [u8; 32] {
    let digest: [u8; 32] = Sha256::digest(content).into();
    let mut bytes = content.to_vec();
    bytes.extend_from_slice(&digest);
    fs::write(path, bytes).unwrap();
    digest
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Self {
        let sequence = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "paisho-train-commit-test-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
