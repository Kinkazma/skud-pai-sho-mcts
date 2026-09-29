//! Explicit frozen comparison using the existing production comparison runner.
use paisho_ai::{Agent, MicroSearchOptions, RandomAgent};
use paisho_core::{legal_actions, GameRecord, Player, RuleProfileId};
use paisho_train::micro_learning::{compare_micro_profile, MicroArtifact, MicroCompareOptions};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--identity" {
        let artifact = MicroArtifact::load(Path::new(&args[2]))?;
        println!("{}", artifact.identity());
        return Ok(());
    }
    if args.len() == 3 && args[1] == "--verify-random" {
        let directory = Path::new(&args[2]);
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("report.json"))?)?;
        let plan: serde_json::Value =
            serde_json::from_slice(&std::fs::read(directory.join("plan.json"))?)?;
        assert_eq!(plan["reference_kind"], "uniform-random");
        let mut verified = 0;
        for game in report["games"].as_array().ok_or("games")? {
            let id = game["id"].as_u64().ok_or("id")?;
            let record: GameRecord =
                std::fs::read_to_string(directory.join(format!("game-{id:04}.psr")))?.parse()?;
            let mut position = record.initial_position();
            let candidate = if game["candidate_seat"] == "H" {
                Player::Host
            } else {
                Player::Guest
            };
            let mut random = RandomAgent::new(
                plan["options"]["seed"]
                    .as_u64()
                    .ok_or("seed")?
                    .wrapping_add(id / 2),
            );
            for &action in record.actions() {
                if position.to_move() != candidate {
                    let actions = legal_actions(&position);
                    let index = random.select_action(&position, &actions)?;
                    assert_eq!(action, actions[index]);
                    verified += 1;
                }
                position.apply(action)?;
            }
        }
        println!("{{\"random_decisions_verified\":{verified}}}");
        return Ok(());
    }
    if args.len() != 7 {
        return Err("MODEL REFERENCE NEW_OUTPUT_DIRECTORY PAIRS V5_BUDGET REFERENCE_BUDGET".into());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let options = MicroCompareOptions {
        pairs: args[4].parse()?,
        workers: 4,
        varied_setups: true,
        simulations: args[5].parse()?,
        move_ms: 0,
        game_seconds: 600.,
        seconds: 1200.,
        decisions: 800,
        seed: 79137,
    };
    compare_micro_profile(
        Path::new(&args[1]),
        Path::new(&args[2]),
        Path::new(&args[3]),
        options,
        &pool,
        RuleProfileId::SkudPaiShoGen5V1,
        MicroSearchOptions {
            proof_search: true,
            ..Default::default()
        },
        MicroSearchOptions {
            proof_search: true,
            ..Default::default()
        },
        args[6].parse()?,
    )
}
