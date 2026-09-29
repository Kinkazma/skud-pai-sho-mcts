use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use paisho_core::{GameRecord, };
use paisho_model::{
    encode_action_v1, CheckpointRandomStateV1, CheckpointRequestV1, InferenceRequestV1,
    TrainingRequestV1,
};
use paisho_mpsgraph_client::{
    default_service_path, MpsGraphProcess, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_replay::{
    PolicyTargetV1, ReplayDatasetV1, ReplayDecisionV1, ReplayDigestV1, ReplayGameV1,
    ReplaySamplerV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1,
};

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
    let steps = parse_or(&arguments, 1, 4_usize, "steps")?;
    let batch_size = parse_or(&arguments, 2, 8_usize, "batch size")?;
    let legal_action_capacity = parse_or(&arguments, 3, 1024_usize, "action capacity")?;
    let preset = match arguments.get(4).map(String::as_str).unwrap_or("micro") {
        "micro" => NetworkPreset::Micro,
        "pure" => NetworkPreset::Pure,
        value => return Err(format!("invalid network preset {value}; use micro or pure").into()),
    };
    if steps == 0 || batch_size == 0 || legal_action_capacity == 0 {
        return Err("steps, batch size and action capacity must be positive".into());
    }

    let temporary = TemporaryDirectory::new()?;
    let (snapshot, snapshot_path) = write_replay_fixture(&temporary.path)?;
    let restored_snapshot = ReplaySnapshotV1::read(&snapshot_path)?;
    if restored_snapshot != snapshot {
        return Err("restored replay snapshot differs from the published snapshot".into());
    }
    let dataset = ReplayDatasetV1::from_snapshot(&restored_snapshot, &temporary.path)?;
    let mut sampler = ReplaySamplerV1::new(&dataset, SAMPLER_SEED);
    let mut process = MpsGraphProcess::launch(ServiceConfiguration {
        executable: service.clone(),
        preset,
        batch_size,
        legal_action_capacity,
        inference_slots: 1,
        optimization: OptimizationLevel::Level1,
        seed: 17,
        checkpoint: None,
    })?;

    let started = Instant::now();
    let mut expected_step = 0_u64;
    for request_id in 0..steps {
        let sampled = sampler.prepare_batch(batch_size)?;
        let examples = sampled
            .examples()
            .iter()
            .map(|example| example.to_training_example())
            .collect::<Result<Vec<_>, _>>()?;
        let request = TrainingRequestV1::new(
            request_id as u64,
            expected_step,
            LEARNING_RATE,
            legal_action_capacity,
            *sampled.snapshot_digest().as_bytes(),
            sampled.start_replay_index(),
            examples,
        )?;
        let response = process.train(&request)?;
        if response.completed_replay_index() != sampled.next_replay_index() {
            return Err("training response advanced to the wrong replay index".into());
        }
        sampler.commit_batch(&sampled)?;
        expected_step = response.completed_training_step();
    }

    let first_checkpoint_path = temporary.path.join("generation-0000-step-first.psckpt");
    let first_checkpoint = checkpoint_request(
        steps as u64 + 100,
        expected_step,
        &dataset,
        sampler.state().next_replay_index(),
        &first_checkpoint_path,
    )?;
    let first_checkpoint_response = process.publish_checkpoint(&first_checkpoint)?;
    let status = process.shutdown()?;
    if !status.success() {
        return Err(format!("first MPSGraph service exited with {status}").into());
    }

    let mut resumed_sampler = ReplaySamplerV1::resume(&dataset, sampler.state())?;
    let mut resumed_process = MpsGraphProcess::launch(ServiceConfiguration {
        executable: service,
        preset,
        batch_size,
        legal_action_capacity,
        inference_slots: 1,
        optimization: OptimizationLevel::Level1,
        seed: 999,
        checkpoint: Some(first_checkpoint_path.clone()),
    })?;
    let repeated_checkpoint_response = resumed_process.publish_checkpoint(&first_checkpoint)?;
    if repeated_checkpoint_response != first_checkpoint_response {
        return Err("replayed checkpoint publication changed its acknowledgement".into());
    }

    let resumed_batch = resumed_sampler.prepare_batch(batch_size)?;
    let last_inference_examples = resumed_batch
        .examples()
        .iter()
        .map(|example| example.inference().clone())
        .collect::<Vec<_>>();
    let resumed_examples = resumed_batch
        .examples()
        .iter()
        .map(|example| example.to_training_example())
        .collect::<Result<Vec<_>, _>>()?;
    let resumed_training_request = TrainingRequestV1::new(
        steps as u64 + 101,
        expected_step,
        LEARNING_RATE,
        legal_action_capacity,
        *resumed_batch.snapshot_digest().as_bytes(),
        resumed_batch.start_replay_index(),
        resumed_examples,
    )?;
    let resumed_training_response = resumed_process.train(&resumed_training_request)?;
    expected_step = resumed_training_response.completed_training_step();
    let last_total_loss = resumed_training_response.total_loss();

    let resumed_checkpoint_path = temporary.path.join("generation-0000-step-resumed.psckpt");
    let resumed_checkpoint = checkpoint_request(
        steps as u64 + 102,
        expected_step,
        &dataset,
        resumed_batch.next_replay_index(),
        &resumed_checkpoint_path,
    )?;
    let resumed_checkpoint_response = resumed_process.publish_checkpoint(&resumed_checkpoint)?;
    resumed_sampler.commit_batch(&resumed_batch)?;

    let inference = InferenceRequestV1::from_examples(
        steps as u64 + 103,
        last_inference_examples,
        legal_action_capacity,
    )?;
    let inference_response = resumed_process.infer(&inference)?;
    let status = resumed_process.shutdown()?;
    if !status.success() {
        return Err(format!("resumed MPSGraph service exited with {status}").into());
    }

    let elapsed = started.elapsed().as_secs_f64();
    println!("snapshot_sha256={}", dataset.snapshot_digest());
    println!("network_preset={preset:?}");
    println!("examples={}", dataset.len());
    println!("training_steps={expected_step}");
    println!(
        "next_replay_index={}",
        resumed_sampler.state().next_replay_index()
    );
    println!("last_total_loss={last_total_loss:.6}");
    println!(
        "first_checkpoint_sha256={}",
        hex_digest(first_checkpoint_response.content_sha256())
    );
    println!(
        "resumed_checkpoint_sha256={}",
        hex_digest(resumed_checkpoint_response.content_sha256())
    );
    println!("checkpoint_retry_idempotent=true");
    println!("restart_completed=true");
    println!(
        "post_training_inference_rows={}",
        inference_response.outputs().len()
    );
    println!("elapsed_seconds={elapsed:.6}");
    Ok(())
}

fn checkpoint_request(
    request_id: u64,
    expected_step: u64,
    dataset: &ReplayDatasetV1,
    replay_index: u64,
    destination: &std::path::Path,
) -> Result<CheckpointRequestV1, Box<dyn Error>> {
    let destination = destination
        .to_str()
        .ok_or("temporary checkpoint path is not UTF-8")?;
    Ok(CheckpointRequestV1::new(
        request_id,
        expected_step,
        *dataset.snapshot_digest().as_bytes(),
        replay_index,
        0,
        LEARNING_RATE,
        vec![
            CheckpointRandomStateV1::new("learner-request-id", request_id + 1)?,
            CheckpointRandomStateV1::new("replay-sampler-seed", SAMPLER_SEED)?,
        ],
        destination,
    )?)
}

fn hex_digest(digest: [u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn write_replay_fixture(
    directory: &std::path::Path,
) -> Result<(ReplaySnapshotV1, PathBuf), Box<dyn Error>> {
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
    let shard_name = "shard-0000.psrbuf";
    shard.write_new(&directory.join(shard_name))?;
    let snapshot = ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
        shard_name, &shard,
    )?])?;
    let snapshot_path = directory.join("snapshot.psrsnap");
    snapshot.write_new(&snapshot_path)?;
    Ok((snapshot, snapshot_path))
}

fn parse_or<T>(
    arguments: &[String],
    index: usize,
    default: T,
    name: &'static str,
) -> Result<T, Box<dyn Error>>
where
    T: core::str::FromStr,
    T::Err: Error + 'static,
{
    arguments
        .get(index)
        .map_or(Ok(default), |value| value.parse().map_err(Into::into))
        .map_err(|source: Box<dyn Error>| format!("invalid {name}: {source}").into())
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
            "paisho-replay-training-{}-{nonce}",
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
