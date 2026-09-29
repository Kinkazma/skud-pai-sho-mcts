//! Export real legal positions and MPSGraph predictions for the Core ML study.
use std::{collections::BTreeMap, error::Error, fs::OpenOptions, path::PathBuf};

use paisho_core::{legal_actions, BasicFlower, GameOutcome, Position, StandardSetup};
use paisho_model::{InferenceRequestV1, NO_COORDINATE_V1, NO_TILE_V1};
use paisho_mpsgraph_client::{
    default_service_path, MpsGraphProcess, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use serde_json::{json, Value};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: coreml_parity_fixture CHECKPOINT NEW_OUTPUT.json".into());
    }
    let batch = 4;
    let capacity = 1024;
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    let mut process = MpsGraphProcess::launch(ServiceConfiguration {
        executable: default_service_path(&std::env::current_dir()?),
        preset: NetworkPreset::Pure,
        batch_size: batch,
        legal_action_capacity: capacity,
        inference_slots: 1,
        optimization: OptimizationLevel::Level1,
        seed: 17,
        checkpoint: Some(PathBuf::from(&args[0])),
    })?;
    let setup = StandardSetup::balanced(BasicFlower::Red3);
    let mut position = Position::from_standard_setup(setup);
    let mut positions = Vec::new();
    let mut families = [0usize; 7];
    // Spaced samples include developed positions, both seats and bonus phases.
    for decision in 0usize..4000 {
        let actions = legal_actions(&position);
        if position.outcome() != GameOutcome::Ongoing || actions.is_empty() {
            position = Position::from_standard_setup(setup);
            continue;
        }
        if decision % 7 == 0 && actions.len() <= capacity {
            positions.push(position.clone());
        }
        position.apply(actions[(decision.wrapping_mul(37) + 11) % actions.len()])?;
        if positions.len() == 64 {
            break;
        }
    }
    if positions.len() != 64 {
        return Err("could not collect 64 eligible positions".into());
    }
    let mut batches = Vec::new();
    for (id, positions) in positions.chunks_exact(batch).enumerate() {
        let request = InferenceRequestV1::from_positions(id as u64, positions, capacity)?;
        let mut inputs: BTreeMap<&str, Vec<Value>> = [
            "spatial",
            "global_features",
            "family_indices",
            "tile_indices",
            "tile_presence",
            "destination_indices",
            "destination_presence",
            "pair_indices",
            "pair_presence",
            "legal_mask",
        ]
        .into_iter()
        .map(|name| (name, Vec::new()))
        .collect();
        for example in request.examples() {
            inputs
                .get_mut("spatial")
                .unwrap()
                .extend(example.state().spatial_nhwc().iter().map(|x| json!(x)));
            inputs
                .get_mut("global_features")
                .unwrap()
                .extend(example.state().global().iter().map(|x| json!(x)));
            for index in 0..capacity {
                let legal = example.legal_actions().get(index);
                let [family, tile, source, destination] = legal
                    .map_or([0, NO_TILE_V1, NO_COORDINATE_V1, NO_COORDINATE_V1], |x| {
                        x.slots()
                    });
                let has_tile = tile != NO_TILE_V1;
                let has_destination = destination != NO_COORDINATE_V1;
                let has_pair = source != NO_COORDINATE_V1 && has_destination;
                if legal.is_some() {
                    families[usize::from(family)] += 1;
                }
                let values = [
                    ("family_indices", u32::from(family)),
                    ("tile_indices", if has_tile { u32::from(tile) } else { 0 }),
                    ("tile_presence", u32::from(has_tile)),
                    (
                        "destination_indices",
                        if has_destination {
                            u32::from(destination)
                        } else {
                            0
                        },
                    ),
                    ("destination_presence", u32::from(has_destination)),
                    (
                        "pair_indices",
                        if has_pair {
                            u32::from(source) * 289 + u32::from(destination)
                        } else {
                            0
                        },
                    ),
                    ("pair_presence", u32::from(has_pair)),
                    ("legal_mask", u32::from(legal.is_some())),
                ];
                for (name, value) in values {
                    inputs.get_mut(name).unwrap().push(json!(value));
                }
            }
        }
        let response = process.infer(&request)?;
        let mut policy = Vec::with_capacity(batch * capacity);
        let mut value = Vec::with_capacity(batch * 3);
        for prediction in response.outputs() {
            policy.extend_from_slice(prediction.policy_probabilities());
            policy.resize(
                policy.len() + capacity - prediction.policy_probabilities().len(),
                0.0,
            );
            value.extend_from_slice(prediction.value_probabilities());
        }
        batches.push(
            json!({"inputs":inputs,"policy_probabilities":policy,"value_probabilities":value}),
        );
    }
    process.shutdown()?;
    serde_json::to_writer(
        output,
        &json!({"batch":batch,"capacity":capacity,
        "checkpoint":args[0],"legal_family_counts":families,"batches":batches}),
    )?;
    println!("positions=64 legal_family_counts={families:?}");
    Ok(())
}
