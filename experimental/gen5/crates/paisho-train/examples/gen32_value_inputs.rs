//! Source-disjoint value diagnostic inputs. No weights are installed by this tool.
use paisho_ai::micro_state_features;
use paisho_core::{GameRecord, RuleProfileId};
use paisho_train::compact_learning::load_dataset;
use sha2::{Digest, Sha256};
use std::{fs, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let data = load_dataset(std::path::Path::new(&a[1]))?;
    let mut counts = [0; 2];
    let mut rows = vec![];
    let mut extraction = 0.;
    for game in &data.games {
        let split = usize::from(game.held_out);
        if counts[split] >= if split == 0 { 192 } else { 64 } {
            continue;
        }
        let raw = fs::read(&game.originals[0].path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            game.originals[0].sha256
        );
        let record: GameRecord = std::str::from_utf8(&raw)?.parse()?;
        let mut p = GameRecord::with_rules(record.setup(), RuleProfileId::SkudPaiSho2022V2)
            .initial_position();
        for decision in 0..record.actions().len() {
            if let Some(ex) = game.examples.iter().find(|e| e.decision_index == decision) {
                if decision % 3 == 0 {
                    let began = Instant::now();
                    let x = micro_state_features(&p);
                    extraction += began.elapsed().as_secs_f64();
                    rows.push(serde_json::json!({"source":game.game_sha256,"held_out":game.held_out,"decision":decision,"x":x.as_slice(),"y":ex.target}));
                }
            }
            p.apply(record.actions()[decision])?;
        }
        counts[split] += 1;
    }
    fs::write(
        &a[2],
        serde_json::to_vec(
            &serde_json::json!({"games":counts,"feature_seconds":extraction,"rows":rows}),
        )?,
    )?;
    Ok(())
}
