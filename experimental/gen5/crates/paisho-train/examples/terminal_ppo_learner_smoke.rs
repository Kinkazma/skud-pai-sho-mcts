use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use paisho_core::{GameRecord, Player, };
use paisho_model::{
    encode_action_v1, InferenceExampleV1, InferenceRequestV1, TerminalPpoParametersV1,
};
use paisho_mpsgraph_client::{
    default_service_path, MpsGraphProcess, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_replay::{
    PolicyEntryV1, PolicyTargetKindV1, PolicyTargetV1, ReplayDecisionV1, ReplayDigestV1,
    ReplayGameV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1,
};
use paisho_train::{
    run_learner, LearnerConfiguration, LearnerObjectiveConfiguration, LearnerObjectiveV1,
    LearnerOriginV1,
};

const TERMINAL_RING: &str =
    include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
const MODEL_SEED: u64 = 20_260_905;
const SAMPLER_SEED: u64 = 0xC0FF_EE22;
const LEARNING_RATE: f32 = 1.0e-4;
const POLICY_TEMPERATURE: f32 = 0.8;
const UNIFORM_MIX: f32 = 0.05;
const ACTION_CAPACITY: usize = 1_024;

fn main() -> Result<(), Box<dyn Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let service = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_service_path(&repository));
    let temporary = TemporaryDirectory::new()?;
    let producer = ReplayDigestV1::from_bytes([0x42; 32]);
    let snapshot_path = write_behavior_replay(&temporary.path, &service, "g0", producer, None)?;
    let parameters =
        TerminalPpoParametersV1::with_behavior(POLICY_TEMPERATURE, UNIFORM_MIX, 0.2, 0.5, 0.01)?;
    let run_directory = temporary.path.join("learner");
    let mut configuration = LearnerConfiguration {
        service_executable: service,
        replay_snapshot: snapshot_path,
        replay_directory: temporary.path.clone(),
        run_directory: run_directory.clone(),
        initial_checkpoint: None,
        network_preset: NetworkPreset::Micro,
        optimization: OptimizationLevel::Level1,
        batch_size: 2,
        legal_action_capacity: ACTION_CAPACITY,
        model_seed: MODEL_SEED,
        sampler_seed: SAMPLER_SEED,
        generation: 0,
        learning_rate: LEARNING_RATE,
        objective: LearnerObjectiveConfiguration::terminal_ppo(producer, None, parameters),
        target_training_step: 1,
        checkpoint_interval: 1,
    };

    let first = run_learner(&configuration)?;
    if first.resumed || first.completed_training_step != 1 || first.completed_replay_index != 2 {
        return Err("fresh terminal PPO learner reported incorrect progress".into());
    }
    let origin = LearnerOriginV1::read(&LearnerOriginV1::path(&run_directory))?;
    let objective = origin
        .identity()
        .objective
        .terminal_ppo()
        .ok_or("learner origin did not retain the terminal PPO objective")?;
    if objective.behavior_producer() != producer
        || objective.actor_checkpoint_sha256().is_some()
        || objective.parameters() != parameters
    {
        return Err("learner origin changed the terminal PPO identity".into());
    }

    configuration.target_training_step = 2;
    let resumed = run_learner(&configuration)?;
    if !resumed.resumed
        || resumed.initial_training_step != 1
        || resumed.completed_training_step != 2
        || resumed.completed_replay_index != 4
    {
        return Err("resumed terminal PPO learner reported incorrect progress".into());
    }

    let child_producer = ReplayDigestV1::from_bytes([0x43; 32]);
    let child_snapshot = write_behavior_replay(
        &temporary.path,
        &configuration.service_executable,
        "g1",
        child_producer,
        Some(&resumed.latest_checkpoint),
    )?;
    let mut child = configuration.clone();
    child.replay_snapshot = child_snapshot;
    child.run_directory = temporary.path.join("learner-g1");
    child.initial_checkpoint = Some(resumed.latest_checkpoint.clone());
    child.generation = 1;
    child.objective = LearnerObjectiveConfiguration::terminal_ppo(
        child_producer,
        Some(resumed.latest_checkpoint.clone()),
        parameters,
    );
    child.target_training_step = 3;
    let first_child = run_learner(&child)?;
    if first_child.resumed
        || !first_child.started_from_parent_checkpoint
        || first_child.generation_start_training_step != 2
        || first_child.completed_training_step != 3
        || first_child.completed_replay_index != 2
    {
        return Err("child terminal PPO generation reported incorrect progress".into());
    }
    child.initial_checkpoint = None;
    child.target_training_step = 4;
    let resumed_child = run_learner(&child)?;
    if !resumed_child.resumed
        || resumed_child.started_from_parent_checkpoint
        || resumed_child.initial_training_step != 3
        || resumed_child.completed_training_step != 4
        || resumed_child.completed_replay_index != 4
    {
        return Err("resumed child terminal PPO generation reported incorrect progress".into());
    }

    println!(
        "objective={}",
        LearnerObjectiveV1::TerminalPpo(objective).identifier()
    );
    println!("behavior_producer={producer}");
    println!("fresh_completed_step={}", first.completed_training_step);
    println!("resumed_completed_step={}", resumed.completed_training_step);
    println!("completed_replay_index={}", resumed.completed_replay_index);
    println!("manifest_retains_actor_and_ppo_parameters=true");
    println!("frozen_actor_baseline_used=true");
    println!("exact_resume=true");
    println!("child_generation_actor_matches_parent=true");
    println!("child_generation_resume=true");
    println!(
        "latest_checkpoint={}",
        resumed_child.latest_checkpoint.display()
    );
    Ok(())
}

fn write_behavior_replay(
    directory: &Path,
    service: &Path,
    suffix: &str,
    producer: ReplayDigestV1,
    checkpoint: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error>> {
    let record = TERMINAL_RING.parse::<GameRecord>()?;
    let mut position = record.initial_position();
    let mut selected = Vec::with_capacity(2);
    let mut host_selected = false;
    let mut guest_selected = false;
    for (decision_index, &action) in record.actions().iter().enumerate() {
        let perspective = position.to_move();
        let needs_example = match perspective {
            Player::Host => !host_selected,
            Player::Guest => !guest_selected,
        };
        if needs_example {
            let inference = InferenceExampleV1::from_position(&position)?;
            let played = encode_action_v1(action, perspective)?;
            let played_index = inference
                .legal_actions()
                .iter()
                .position(|candidate| *candidate == played)
                .ok_or("recorded action was absent from its regenerated legal actions")?;
            selected.push((decision_index, inference, played_index));
            match perspective {
                Player::Host => host_selected = true,
                Player::Guest => guest_selected = true,
            }
        }
        position.apply(action)?;
        if host_selected && guest_selected {
            break;
        }
    }
    if selected.len() != 2 {
        return Err("terminal fixture did not expose decisions for both players".into());
    }

    let inference_request = InferenceRequestV1::from_examples(
        1,
        selected
            .iter()
            .map(|(_, inference, _)| inference.clone())
            .collect(),
        ACTION_CAPACITY,
    )?;
    let mut actor = MpsGraphProcess::launch(ServiceConfiguration {
        executable: service.to_owned(),
        preset: NetworkPreset::Micro,
        batch_size: 2,
        legal_action_capacity: ACTION_CAPACITY,
        inference_slots: 1,
        optimization: OptimizationLevel::Level1,
        seed: MODEL_SEED,
        checkpoint: checkpoint.map(Path::to_owned),
    })?;
    let outputs = actor.infer(&inference_request)?.into_outputs();
    let status = actor.shutdown()?;
    if !status.success() {
        return Err(format!("behavior actor service exited with {status}").into());
    }

    let decisions = selected
        .into_iter()
        .zip(outputs)
        .map(|((decision_index, inference, played_index), output)| {
            let behavior = transformed_behavior_policy(
                output.policy_probabilities(),
                POLICY_TEMPERATURE,
                UNIFORM_MIX,
            );
            if behavior.get(played_index).copied().unwrap_or(0.0) <= 0.0 {
                return Err("behavior policy excludes the played action".into());
            }
            let entries = inference
                .legal_actions()
                .iter()
                .copied()
                .zip(behavior)
                .map(|(action, probability)| PolicyEntryV1::new(action, probability))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ReplayDecisionV1::new(
                decision_index,
                PolicyTargetV1::new(PolicyTargetKindV1::Behavior, producer, entries)?,
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let game = ReplayGameV1::new(1, producer, producer, record, decisions)?;
    let shard = ReplayShardV1::new(0, vec![game])?;
    let shard_name = format!("terminal-ppo-behavior-{suffix}.psrbuf");
    shard.write_new(&directory.join(&shard_name))?;
    let snapshot = ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
        &shard_name,
        &shard,
    )?])?;
    let snapshot_path = directory.join(format!("terminal-ppo-{suffix}.psrsnap"));
    snapshot.write_new(&snapshot_path)?;
    Ok(snapshot_path)
}

fn transformed_behavior_policy(network: &[f32], temperature: f32, uniform_mix: f32) -> Vec<f32> {
    let inverse_temperature = 1.0 / f64::from(temperature);
    let maximum_logit = network
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| f64::from(value).ln() * inverse_temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut probabilities = network
        .iter()
        .map(|&value| {
            if value > 0.0 {
                (f64::from(value).ln() * inverse_temperature - maximum_logit).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let total = probabilities.iter().sum::<f64>();
    let retained = 1.0 - f64::from(uniform_mix);
    let uniform = 1.0 / probabilities.len() as f64;
    for probability in &mut probabilities {
        *probability = retained * (*probability / total) + f64::from(uniform_mix) * uniform;
    }
    probabilities
        .into_iter()
        .map(|value| value as f32)
        .collect()
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
            "paisho-terminal-ppo-learner-{}-{nonce}",
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
