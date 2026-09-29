//! Explicit GPU smoke: two PPO cycles in one service, without checkpoint I/O.
use std::error::Error;
use std::path::PathBuf;

use paisho_core::{BasicFlower, Position, StandardSetup};
use paisho_model::{
    InferenceExampleV1, InferenceRequestV1, TerminalPpoExampleV1, TerminalPpoParametersV1,
    TerminalPpoRequestV1, ValueClassV1,
};
use paisho_mpsgraph_client::{
    default_service_path, MpsGraphClientError, MpsGraphProcess, NetworkPreset, OptimizationLevel,
    ServiceConfiguration, TrainingCycleProgressV1, TrainingCycleRandomStateV1,
    TrainingCycleRequestV1, TrainingCycleSchedulerV1,
};

fn main() -> Result<(), Box<dyn Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let service = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_service_path(&repository));
    let capacity = 1024;
    let snapshot_a = [0x5a; 32];
    let snapshot_b = [0x6b; 32];
    let rate = 1.0e-3;
    let inference = InferenceExampleV1::from_position(&Position::from_standard_setup(
        StandardSetup::balanced(BasicFlower::Red3),
    ))?;
    let mut process = MpsGraphProcess::launch(ServiceConfiguration {
        executable: service,
        preset: NetworkPreset::Micro,
        batch_size: 1,
        legal_action_capacity: capacity,
        inference_slots: 1,
        optimization: OptimizationLevel::Level1,
        seed: 20_260_905,
        checkpoint: None,
    })?;
    let pid = process.process_id().ok_or("service has no PID")?;
    let before = process.infer(&InferenceRequestV1::from_examples(
        1,
        vec![inference.clone()],
        capacity,
    )?)?;
    let output = &before.outputs()[0];
    let values = output.value_probabilities();
    let example = TerminalPpoExampleV1::new(
        inference,
        0,
        output.policy_probabilities()[0],
        ValueClassV1::Win,
        values[ValueClassV1::Win.index()] - values[ValueClassV1::Loss.index()],
    )?;
    // Legacy PST V2 establishes the initial snapshot binding without PSG1 or disk I/O.
    let first = process.train_terminal_ppo(&TerminalPpoRequestV1::new(
        2,
        0,
        rate,
        TerminalPpoParametersV1::new(0.2, 0.0, 0.0)?,
        capacity,
        snapshot_a,
        0,
        vec![example.clone()],
    )?)?;
    if first.completed_training_step() != 1 || first.completed_replay_index() != 1 {
        return Err("first PPO step did not complete at step 1 / replay index 1".into());
    }
    let mut transition = TrainingCycleRequestV1 {
        request_id: 3,
        expected_training_step: 1,
        previous_snapshot_sha256: Some("00".repeat(32)),
        next_progress: TrainingCycleProgressV1 {
            generation: 2,
            replay_index: 0,
            replay_snapshot_sha256: "6b".repeat(32),
            scheduler: TrainingCycleSchedulerV1 {
                learning_rate: rate,
                completed_steps: 1,
            },
            random_states: vec![TrainingCycleRandomStateV1 {
                name: "sampler".into(),
                state: 42,
            }],
        },
    };
    match process.begin_new_generation(&transition) {
        Err(MpsGraphClientError::TrainingCycleWire(message))
            if message.contains("service rejected training cycle: previousSnapshotMismatch") => {}
        Err(error) => {
            return Err(
                format!("wrong previous snapshot failed for an unexpected reason: {error}").into(),
            )
        }
        Ok(_) => return Err("wrong previous snapshot was accepted".into()),
    }
    transition.request_id = 4;
    transition.previous_snapshot_sha256 = Some("5a".repeat(32));
    let acknowledged = process.begin_new_generation(&transition)?;
    if acknowledged.training_step != 1 || acknowledged.progress != transition.next_progress {
        return Err("transition did not preserve step and acknowledge the new progress".into());
    }
    let second = process.train_terminal_ppo(&TerminalPpoRequestV1::new(
        5,
        1,
        rate,
        TerminalPpoParametersV1::new(0.2, 0.0, 0.0)?,
        capacity,
        snapshot_b,
        0,
        vec![example],
    )?)?;
    if second.completed_training_step() != 2 || second.completed_replay_index() != 1 {
        return Err("second PPO cycle did not complete at step 2 / replay index 1".into());
    }
    if process.process_id() != Some(pid) {
        return Err("service PID changed across training cycles".into());
    }
    let status = process.shutdown()?;
    if !status.success() {
        return Err(format!("MPSGraph service exited with {status}").into());
    }
    println!("service_pid={pid}");
    println!("wrong_previous_snapshot_rejected=true");
    println!("cycle_2_generation=2 training_step=2 completed_replay_index=1");
    println!("same_pid=true checkpoint_created=false");
    Ok(())
}
