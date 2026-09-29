use std::error::Error;
use std::path::PathBuf;

use paisho_core::{BasicFlower, Position, StandardSetup};
use paisho_model::{
    InferenceExampleV1, InferenceRequestV1, TerminalPpoExampleV1, TerminalPpoParametersV1,
    TerminalPpoRequestV1, ValueClassV1,
};
use paisho_mpsgraph_client::{
    default_service_path, MpsGraphProcess, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};

fn main() -> Result<(), Box<dyn Error>> {
    let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let service = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_service_path(&repository));
    let capacity = 1024;
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

    let before_request = InferenceRequestV1::from_examples(1, vec![inference.clone()], capacity)?;
    let before = process.infer(&before_request)?;
    let before_output = &before.outputs()[0];
    let before_probability = before_output.policy_probabilities()[0];
    let value = before_output.value_probabilities();
    let actor_value = value[ValueClassV1::Win.index()] - value[ValueClassV1::Loss.index()];

    let example = TerminalPpoExampleV1::new(
        inference.clone(),
        0,
        before_probability,
        ValueClassV1::Win,
        actor_value,
    )?;
    let training = TerminalPpoRequestV1::new(
        2,
        0,
        1.0e-3,
        TerminalPpoParametersV1::new(0.2, 0.0, 0.0)?,
        capacity,
        [0x5a; 32],
        0,
        vec![example],
    )?;
    let trained = process.train_terminal_ppo(&training)?;

    let after_request = InferenceRequestV1::from_examples(3, vec![inference], capacity)?;
    let after = process.infer(&after_request)?;
    let after_probability = after.outputs()[0].policy_probabilities()[0];
    if after_probability <= before_probability {
        return Err(format!(
            "positive terminal advantage did not increase the played probability: \
             {before_probability} -> {after_probability}"
        )
        .into());
    }
    let status = process.shutdown()?;
    if !status.success() {
        return Err(format!("MPSGraph service exited with {status}").into());
    }

    println!("objective=ppo-terminal-v1");
    println!("training_step={}", trained.completed_training_step());
    println!(
        "completed_replay_index={}",
        trained.completed_replay_index()
    );
    println!("policy_loss={:.6}", trained.policy_loss());
    println!("value_loss={:.6}", trained.value_loss());
    println!("entropy={:.6}", trained.entropy());
    println!("mean_advantage={:.6}", trained.mean_advantage());
    println!("played_probability_before={before_probability:.9}");
    println!("played_probability_after={after_probability:.9}");
    println!("rust_swift_round_trip=true");
    Ok(())
}
