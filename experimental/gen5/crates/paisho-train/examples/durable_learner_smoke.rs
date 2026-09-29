use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use paisho_core::{GameRecord, };
use paisho_model::{
    encode_action_v1, CheckpointRandomStateV1, CheckpointRequestV1, InferenceRequestV1,
    TrainingRequestV1, TrainingWireError,
};
use paisho_mpsgraph_client::{
    default_service_path, read_checkpoint_metadata, MpsGraphClientError, MpsGraphProcess,
    NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_replay::{
    PolicyTargetV1, ReplayDatasetV1, ReplayDecisionV1, ReplayDigestV1, ReplayGameV1,
    ReplaySamplerStateV1, ReplaySamplerV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1,
};
use paisho_train::{
    discover_latest_commit, run_learner, LearnerConfiguration, LearnerIdentityV1,
    LearnerObjectiveConfiguration, LearnerObjectiveV1, LearnerOriginV1,
};
use sha2::{Digest, Sha256};

const TERMINAL_RING: &str =
    include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
const LEARNING_RATE: f32 = 1.0e-4;
const SAMPLER_SEED: u64 = 0xC0FF_EE11;

fn main() -> Result<(), Box<dyn Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let service = arguments
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| default_service_path(&repository));
    let preset = match arguments.get(1).map(String::as_str).unwrap_or("micro") {
        "micro" => NetworkPreset::Micro,
        "pure" => NetworkPreset::Pure,
        value => return Err(format!("invalid network preset {value}; use micro or pure").into()),
    };
    let batch_size = arguments
        .get(2)
        .map_or(Ok(8_usize), |value| value.parse())?;

    let temporary = TemporaryDirectory::new()?;
    let snapshot_path = write_replay_fixture(&temporary.path, "g0")?;
    let run_directory = temporary.path.join("learner");
    let mut configuration = LearnerConfiguration {
        service_executable: service,
        replay_snapshot: snapshot_path,
        replay_directory: temporary.path.clone(),
        run_directory,
        initial_checkpoint: None,
        network_preset: preset,
        optimization: OptimizationLevel::Level1,
        batch_size,
        legal_action_capacity: 1024,
        model_seed: 17,
        sampler_seed: SAMPLER_SEED,
        generation: 0,
        learning_rate: LEARNING_RATE,
        objective: LearnerObjectiveConfiguration::SupervisedPolicyValue,
        target_training_step: 1,
        checkpoint_interval: 1,
    };

    let started = Instant::now();
    let first = run_learner(&configuration)?;
    if first.resumed || first.initial_training_step != 0 || first.completed_training_step != 1 {
        return Err("fresh learner run reported the wrong progress".into());
    }
    let orphan_digest = publish_step_two_without_manifest(&configuration)?;
    configuration.target_training_step = 2;
    let resumed = run_learner(&configuration)?;
    if !resumed.resumed
        || resumed.initial_training_step != 1
        || resumed.completed_training_step != 2
    {
        return Err("resumed learner run reported the wrong progress".into());
    }
    let snapshot = ReplaySnapshotV1::read(&configuration.replay_snapshot)?;
    let recovered = discover_latest_commit(
        &configuration.run_directory,
        learner_identity(&configuration, snapshot.digest()),
    )?
    .ok_or("learner did not publish the recovered commit")?;
    let orphan_path = configuration.run_directory.join(checkpoint_name(0, 2, 0));
    if !orphan_path.is_file()
        || recovered.checkpoint_path(&configuration.run_directory) == orphan_path
    {
        return Err("orphan checkpoint was not kept separate from the recovered commit".into());
    }
    let completed = run_learner(&configuration)?;
    if completed.training_steps_this_run != 0
        || completed.latest_checkpoint != resumed.latest_checkpoint
    {
        return Err("completed target was not a stable no-op".into());
    }

    let invalid_parent = temporary.path.join("metadata-only-parent.psckpt");
    write_metadata_only_checkpoint(&resumed.latest_checkpoint, &invalid_parent)?;
    let mut recovered_child = configuration.clone();
    recovered_child.run_directory = temporary.path.join("invalid-parent-recovery");
    recovered_child.initial_checkpoint = Some(invalid_parent);
    recovered_child.generation = 2;
    recovered_child.target_training_step = resumed.completed_training_step + 1;
    if run_learner(&recovered_child).is_ok() {
        return Err("metadata-only parent unexpectedly trained".into());
    }
    if LearnerOriginV1::path(&recovered_child.run_directory).exists() {
        return Err("failed parent validation sealed a learner origin".into());
    }
    recovered_child.initial_checkpoint = Some(resumed.latest_checkpoint.clone());
    let recovered_parent_run = run_learner(&recovered_child)?;
    if !recovered_parent_run.started_from_parent_checkpoint
        || recovered_parent_run.completed_training_step != resumed.completed_training_step + 1
    {
        return Err("same directory did not recover with the valid parent".into());
    }

    let child_snapshot = write_replay_fixture(&temporary.path, "g1")?;
    let mut child = configuration.clone();
    child.replay_snapshot = child_snapshot;
    child.run_directory = temporary.path.join("learner-g1");
    child.initial_checkpoint = Some(resumed.latest_checkpoint.clone());
    child.sampler_seed = SAMPLER_SEED.wrapping_add(1);
    child.generation = 1;
    child.learning_rate = LEARNING_RATE / 2.0;
    child.target_training_step = resumed.completed_training_step + 1;
    verify_new_generation_boundary(&child)?;
    let first_child = run_learner(&child)?;
    if first_child.resumed
        || !first_child.has_parent_checkpoint
        || !first_child.started_from_parent_checkpoint
        || first_child.generation_start_training_step != resumed.completed_training_step
        || first_child.initial_training_step != resumed.completed_training_step
        || first_child.completed_training_step != resumed.completed_training_step + 1
        || first_child.completed_replay_index != batch_size as u64
    {
        return Err("first child-generation run reported the wrong progress".into());
    }
    let first_child_metadata = read_checkpoint_metadata(&first_child.latest_checkpoint)?;
    let child_snapshot_digest = ReplaySnapshotV1::read(&child.replay_snapshot)?.digest();
    if first_child_metadata.generation() != child.generation
        || first_child_metadata.training_step() != first_child.completed_training_step
        || first_child_metadata.replay_index() != batch_size as u64
        || first_child_metadata.replay_snapshot_sha256() != *child_snapshot_digest.as_bytes()
        || first_child_metadata.learning_rate() != child.learning_rate
    {
        return Err("child checkpoint metadata did not bind the new generation".into());
    }

    child.initial_checkpoint = None;
    child.target_training_step += 1;
    let resumed_child = run_learner(&child)?;
    if !resumed_child.resumed
        || !resumed_child.has_parent_checkpoint
        || resumed_child.started_from_parent_checkpoint
        || resumed_child.generation_start_training_step != resumed.completed_training_step
        || resumed_child.initial_training_step != first_child.completed_training_step
        || resumed_child.completed_training_step != first_child.completed_training_step + 1
        || resumed_child.completed_replay_index != (batch_size as u64) * 2
    {
        return Err("child-generation resume reported the wrong progress".into());
    }

    println!("network_preset={preset:?}");
    println!("fresh_completed_step={}", first.completed_training_step);
    println!("orphan_checkpoint_sha256={}", hex_digest(orphan_digest));
    println!(
        "recovered_checkpoint_sha256={}",
        hex_digest(recovered.checkpoint_sha256())
    );
    println!("orphan_checkpoint_ignored_and_replayed=true");
    println!("resumed_from_step={}", resumed.initial_training_step);
    println!("resumed_completed_step={}", resumed.completed_training_step);
    println!("completed_replay_index={}", resumed.completed_replay_index);
    println!("completed_target_no_op=true");
    println!("invalid_parent_did_not_seal_origin=true");
    println!("same_directory_recovered_with_valid_parent=true");
    println!("generation_rollover_inference_identical=true");
    println!("generation_rollover_requires_replay_index_zero=true");
    println!(
        "child_generation_start_step={}",
        first_child.generation_start_training_step
    );
    println!(
        "child_generation_completed_step={}",
        resumed_child.completed_training_step
    );
    println!(
        "child_generation_replay_index={}",
        resumed_child.completed_replay_index
    );
    println!("child_generation_resume_without_parent_path=true");
    println!(
        "latest_checkpoint={}",
        resumed_child.latest_checkpoint.display()
    );
    println!("elapsed_seconds={:.6}", started.elapsed().as_secs_f64());
    Ok(())
}

fn write_metadata_only_checkpoint(
    source: &std::path::Path,
    destination: &std::path::Path,
) -> Result<(), Box<dyn Error>> {
    const MAGIC: &[u8] = b"PAISHO-CKPT-V2\n";

    let bytes = fs::read(source)?;
    if bytes.len() < MAGIC.len() + 4 + 32 || &bytes[..MAGIC.len()] != MAGIC {
        return Err("source checkpoint has no V2 metadata header".into());
    }
    let length_offset = MAGIC.len();
    let metadata_length = u32::from_le_bytes(
        bytes[length_offset..length_offset + 4]
            .try_into()
            .expect("four-byte metadata length"),
    ) as usize;
    let metadata_end = length_offset
        .checked_add(4)
        .and_then(|offset| offset.checked_add(metadata_length))
        .ok_or("checkpoint metadata boundary overflow")?;
    if metadata_end > bytes.len() - 32 {
        return Err("source checkpoint metadata is truncated".into());
    }
    let mut malformed = bytes[..metadata_end].to_vec();
    let digest: [u8; 32] = Sha256::digest(&malformed).into();
    malformed.extend_from_slice(&digest);
    fs::write(destination, malformed)?;
    Ok(())
}

fn verify_new_generation_boundary(
    configuration: &LearnerConfiguration,
) -> Result<(), Box<dyn Error>> {
    let parent = configuration
        .initial_checkpoint
        .as_ref()
        .ok_or("generation boundary check requires a parent")?;
    let parent_metadata = read_checkpoint_metadata(parent)?;
    let snapshot = ReplaySnapshotV1::read(&configuration.replay_snapshot)?;
    let dataset = ReplayDatasetV1::from_snapshot(&snapshot, &configuration.replay_directory)?;
    let mut sampler = ReplaySamplerV1::resume(
        &dataset,
        ReplaySamplerStateV1::new(dataset.snapshot_digest(), configuration.sampler_seed, 0),
    )?;
    let sampled = sampler.prepare_batch(configuration.batch_size)?;
    let examples = sampled
        .examples()
        .iter()
        .map(|example| example.to_training_example())
        .collect::<Result<Vec<_>, _>>()?;
    let service_configuration = ServiceConfiguration {
        executable: configuration.service_executable.clone(),
        preset: configuration.network_preset,
        batch_size: configuration.batch_size,
        legal_action_capacity: configuration.legal_action_capacity,
        inference_slots: 1,
        optimization: configuration.optimization,
        seed: configuration.model_seed,
        checkpoint: Some(parent.clone()),
    };
    let inference = InferenceRequestV1::from_examples(
        90,
        examples
            .iter()
            .map(|example| example.inference().clone())
            .collect(),
        configuration.legal_action_capacity,
    )?;
    let mut ordinary = MpsGraphProcess::launch(service_configuration.clone())?;
    let ordinary_output = ordinary.infer(&inference)?;
    let ordinary_status = ordinary.shutdown()?;
    if !ordinary_status.success() {
        return Err(format!("ordinary comparison service exited with {ordinary_status}").into());
    }
    let mut rollover = MpsGraphProcess::launch_new_generation(service_configuration)?;
    let rollover_output = rollover.infer(&inference)?;
    if rollover_output != ordinary_output {
        return Err("new-generation launch changed restored inference outputs".into());
    }

    let rejected = TrainingRequestV1::new(
        91,
        parent_metadata.training_step(),
        configuration.learning_rate,
        configuration.legal_action_capacity,
        *dataset.snapshot_digest().as_bytes(),
        configuration.batch_size as u64,
        examples.clone(),
    )?;
    if !matches!(
        rollover.train(&rejected),
        Err(MpsGraphClientError::TrainingWire(TrainingWireError::Service(message)))
            if message.contains("replay index")
    ) {
        return Err("new generation accepted a nonzero first replay index".into());
    }
    let accepted = TrainingRequestV1::new(
        92,
        parent_metadata.training_step(),
        configuration.learning_rate,
        configuration.legal_action_capacity,
        *dataset.snapshot_digest().as_bytes(),
        0,
        examples,
    )?;
    let response = rollover.train(&accepted)?;
    if response.completed_training_step() != parent_metadata.training_step() + 1
        || response.completed_replay_index() != configuration.batch_size as u64
    {
        return Err("valid first child batch did not preserve global/local progress".into());
    }
    let rollover_status = rollover.shutdown()?;
    if !rollover_status.success() {
        return Err(
            format!("new-generation comparison service exited with {rollover_status}").into(),
        );
    }
    Ok(())
}

fn publish_step_two_without_manifest(
    configuration: &LearnerConfiguration,
) -> Result<[u8; 32], Box<dyn Error>> {
    let snapshot = ReplaySnapshotV1::read(&configuration.replay_snapshot)?;
    let dataset = ReplayDatasetV1::from_snapshot(&snapshot, &configuration.replay_directory)?;
    let identity = learner_identity(configuration, dataset.snapshot_digest());
    let committed = discover_latest_commit(&configuration.run_directory, identity)?
        .ok_or("missing first learner commit")?;
    let mut sampler = ReplaySamplerV1::resume(
        &dataset,
        ReplaySamplerStateV1::new(
            dataset.snapshot_digest(),
            configuration.sampler_seed,
            committed.next_replay_index(),
        ),
    )?;
    let mut process = MpsGraphProcess::launch(ServiceConfiguration {
        executable: configuration.service_executable.clone(),
        preset: configuration.network_preset,
        batch_size: configuration.batch_size,
        legal_action_capacity: configuration.legal_action_capacity,
        inference_slots: 1,
        optimization: configuration.optimization,
        seed: configuration.model_seed,
        checkpoint: Some(committed.checkpoint_path(&configuration.run_directory)),
    })?;
    let sampled = sampler.prepare_batch(configuration.batch_size)?;
    let examples = sampled
        .examples()
        .iter()
        .map(|example| example.to_training_example())
        .collect::<Result<Vec<_>, _>>()?;
    let training_request_id = committed.next_request_id();
    let training = TrainingRequestV1::new(
        training_request_id,
        committed.training_step(),
        configuration.learning_rate,
        configuration.legal_action_capacity,
        *dataset.snapshot_digest().as_bytes(),
        sampled.start_replay_index(),
        examples,
    )?;
    let response = process.train(&training)?;
    sampler.commit_batch(&sampled)?;
    let checkpoint_request_id = training_request_id
        .checked_add(1)
        .ok_or("request id overflow")?;
    let following_request_id = checkpoint_request_id
        .checked_add(1)
        .ok_or("request id overflow")?;
    let checkpoint_name = checkpoint_name(
        configuration.generation,
        response.completed_training_step(),
        0,
    );
    let checkpoint_path = configuration.run_directory.join(checkpoint_name);
    let destination = checkpoint_path
        .to_str()
        .ok_or("checkpoint path is not UTF-8")?;
    let checkpoint = CheckpointRequestV1::new(
        checkpoint_request_id,
        response.completed_training_step(),
        *dataset.snapshot_digest().as_bytes(),
        sampler.state().next_replay_index(),
        configuration.generation,
        configuration.learning_rate,
        vec![
            CheckpointRandomStateV1::new("learner-request-id", following_request_id)?,
            CheckpointRandomStateV1::new("replay-sampler-seed", configuration.sampler_seed)?,
        ],
        destination,
    )?;
    let published = process.publish_checkpoint(&checkpoint)?;
    let status = process.shutdown()?;
    if !status.success() {
        return Err(format!("orphan-producing service exited with {status}").into());
    }
    Ok(published.content_sha256())
}

fn learner_identity(
    configuration: &LearnerConfiguration,
    replay_snapshot: ReplayDigestV1,
) -> LearnerIdentityV1 {
    LearnerIdentityV1 {
        replay_snapshot,
        network_preset: configuration.network_preset,
        optimization: configuration.optimization,
        batch_size: configuration.batch_size,
        legal_action_capacity: configuration.legal_action_capacity,
        model_seed: configuration.model_seed,
        sampler_seed: configuration.sampler_seed,
        generation: configuration.generation,
        learning_rate: configuration.learning_rate,
        objective: LearnerObjectiveV1::SupervisedPolicyValue,
    }
}

fn checkpoint_name(generation: u64, step: u64, attempt: u64) -> String {
    format!("checkpoint-g{generation:020}-s{step:020}-a{attempt:020}.psckpt")
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(encoded, "{byte:02x}").unwrap();
    }
    encoded
}

fn write_replay_fixture(
    directory: &std::path::Path,
    suffix: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let record = TERMINAL_RING.parse::<GameRecord>()?;
    let producer = ReplayDigestV1::from_bytes([0x33; 32]);
    let mut position = record.initial_position();
    let mut decisions = Vec::with_capacity(record.actions().len());
    for (index, &action) in record.actions().iter().enumerate() {
        let encoded = encode_action_v1(action, position.to_move())?;
        decisions.push(ReplayDecisionV1::new(
            index,
            PolicyTargetV1::one_hot(producer, encoded)?,
        ));
        position.apply(action)?;
    }
    let game = ReplayGameV1::new(
        1,
        ReplayDigestV1::from_bytes([0x11; 32]),
        ReplayDigestV1::from_bytes([0x22; 32]),
        record,
        decisions,
    )?;
    let shard = ReplayShardV1::new(0, vec![game])?;
    let shard_name = format!("shard-{suffix}.psrbuf");
    shard.write_new(&directory.join(&shard_name))?;
    let snapshot = ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
        &shard_name,
        &shard,
    )?])?;
    let snapshot_path = directory.join(format!("snapshot-{suffix}.psrsnap"));
    snapshot.write_new(&snapshot_path)?;
    Ok(snapshot_path)
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Result<Self, std::io::Error> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "paisho-durable-learner-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
