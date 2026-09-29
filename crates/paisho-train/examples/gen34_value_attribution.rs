//! Read-only exact value attribution on a recorded game. No search or training.
use paisho_ai::{micro_state_features, COMPACT_FEATURE_NAMES};
use paisho_core::{GameRecord, Player, STANDARD_TILE_KINDS};
use paisho_train::gen32::Artifact;
use serde_json::json;
use std::fs;

fn names() -> Vec<String> {
    let mut result: Vec<_> = COMPACT_FEATURE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect();
    for seat in ["own", "opponent"] {
        for kind in STANDARD_TILE_KINDS {
            result.push(format!("absolute_{seat}_reserve_{kind:?}"));
        }
    }
    for seat in ["own", "opponent"] {
        for family in ["basic", "special", "accent"] {
            for zone in [
                "centre",
                "rim",
                "negative_xy",
                "positive_x",
                "positive_y",
                "positive_xy",
            ] {
                result.push(format!("absolute_{seat}_{family}_{zone}"));
            }
        }
    }
    result.extend(["phase_main", "phase_bonus", "turn_fraction", "occupancy"].map(str::to_string));
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: MODEL HUMAN.json OUTPUT.json".into());
    }
    let artifact = Artifact::load(std::path::Path::new(&args[1]))?;
    let model = artifact.model()?;
    let input: serde_json::Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let record: GameRecord = input["moves"].as_str().ok_or("missing PSR")?.parse()?;
    let mut weights = model.value.weights().to_vec();
    weights.extend(model.value_extra.unwrap_or([0.; 64]));
    let labels = names();
    assert_eq!(labels.len(), 128);
    let mut position = record.initial_position();
    let mut rows = vec![];
    let mut captured_positions = vec![];
    for (i, action) in record.actions().iter().copied().enumerate() {
        let result = position.apply(action)?;
        if ![6, 7, 10, 11, 14, 15].contains(&(i + 1)) {
            continue;
        }
        if [7, 11, 15].contains(&(i + 1)) {
            captured_positions.push(position.clone());
        }
        let state = micro_state_features(&position);
        let sign = if position.to_move() == Player::Host {
            1.
        } else {
            -1.
        };
        let contributions: Vec<f64> = weights
            .iter()
            .zip(state)
            .map(|(w, x)| sign * w * x)
            .collect();
        let raw: f64 = contributions.iter().sum();
        let reconstructed = raw / (1. + raw.abs());
        let actual = model.value_at(&position, Player::Host) as f64;
        assert!(
            (actual - reconstructed).abs() < 1e-6,
            "attribution must match production value"
        );
        let opposite = model.value_at(&position, Player::Guest) as f64;
        assert_eq!(actual, -opposite);
        let features: Vec<_> = (0..128).map(|j| json!({"index":j,"name":labels[j],"feature":state[j],"weight":weights[j],"host_raw_contribution":contributions[j]})).collect();
        rows.push(json!({"decision":i+1,"action":action.to_string(),"chooser":format!("{:?}",position.to_move()),"phase":format!("{:?}",position.phase()),"captured":format!("{:?}",result.captured),"host_raw":raw,"host_value":actual,"compact_raw":contributions[..64].iter().sum::<f64>(),"extra_raw":contributions[64..].iter().sum::<f64>(),"features":features}));
    }
    let repeated_boards: Vec<_> = captured_positions.windows(2).map(|pair| {
        let before = &pair[0]; let after = &pair[1];
        json!({"same_board":before.board()==after.board(),"same_chooser":before.to_move()==after.to_move(),"same_phase":before.phase()==after.phase(),"before_turns":before.completed_turns(),"after_turns":after.completed_turns(),"before_value":model.value_at(before,Player::Host),"after_value":model.value_at(after,Player::Host)})
    }).collect();
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(
            &json!({"model_updates":artifact.updates,"perspective":"Host","no_search":true,"rows":rows,"successive_captures":repeated_boards}),
        )?,
    )?;
    Ok(())
}
