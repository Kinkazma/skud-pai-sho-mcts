//! Same terminal-horizon source-generation protocol for Random and MCTS-8.
use paisho_ai::{Agent, MctsAgent, MctsConfig, StableRng};
use paisho_core::{
    legal_actions, BasicFlower, GameOutcome, GameRecord, Position, StandardSetup, TurnPhase,
};
use rayon::prelude::*;
use serde_json::json;
use std::{
    cmp::Reverse,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 && args.len() != 5 {
        return Err(
            "usage: start_source_bench random|mcts8|mcts32 GAMES WORKERS NEW_OUTPUT [FIRST_ID]"
                .into(),
        );
    }
    let mode = &args[0];
    if mode != "random" && mode != "mcts8" && mode != "mcts32" {
        return Err("unknown mode".into());
    }
    let games: usize = args[1].parse()?;
    let workers: usize = args[2].parse()?;
    if games == 0 || workers == 0 {
        return Err("positive counts required".into());
    }
    let output = PathBuf::from(&args[3]);
    fs::create_dir(&output)?;
    let first_id: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(0);
    let config = MctsConfig {
        simulations: if mode == "mcts32" { 32 } else { 8 },
        ..MctsConfig::default()
    };
    fs::write(
        output.join("plan.json"),
        serde_json::to_vec_pretty(&json!({
            "mode": mode, "games": games, "workers": workers, "source_limit": 16384,
            "remaining_horizon": 64, "seed": 70332026, "mcts_config": format!("{config:?}"),
            "scope": "1000 source attempts, same terminal-horizon rule; unfinished sources excluded, no silent replacements"
        }))?,
    )?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let completed = AtomicUsize::new(0);
    let started = Instant::now();
    let rows: Vec<_> = pool.install(|| (0..games).into_par_iter().map(|offset| {
                let id = first_id + offset;
        let begin = Instant::now();
        let mut seed_rng = StableRng::new(70332026 ^ (id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let seed = seed_rng.next_u64();
        let mut rng = StableRng::new(seed);
        let mut agent = MctsAgent::new(seed, config).unwrap();
        let setup = StandardSetup::balanced(BasicFlower::Red3);
        let mut position = Position::from_standard_setup(setup);
        let mut record = GameRecord::new(setup);
        let mut boundaries = Vec::new();
        while position.outcome() == GameOutcome::Ongoing {
            if record.actions().len() >= 16384 && position.phase() == TurnPhase::Main { break; }
            let actions = legal_actions(&position);
            let index = if mode == "random" { rng.index(actions.len()) }
                else { agent.select_action(&position, &actions).unwrap() };
            position.apply(actions[index]).unwrap();
            record.push(actions[index]);
            if position.outcome() == GameOutcome::Ongoing && position.phase() == TurnPhase::Main {
                boundaries.push(record.actions().len());
            }
        }
        let decisions = record.actions().len();
        let terminal = position.outcome() != GameOutcome::Ongoing;
        let cut = if terminal { boundaries.into_iter().min_by_key(|&b| ((decisions-b).abs_diff(64), Reverse(decisions-b), b)) } else { None };
        let mut prefix = GameRecord::new(setup);
        if let Some(cut) = cut { for &a in &record.actions()[..cut] { prefix.push(a); } }
        let generation_seconds = begin.elapsed().as_secs_f64();
        let row = json!({"id":id,"seed":seed,"terminal":terminal,"outcome":format!("{:?}",position.outcome()),
            "decisions":decisions,"prefix_decisions":cut,"remaining":cut.map(|b|decisions-b),
            "generation_seconds":generation_seconds,"simulations":agent.telemetry().simulations});
        fs::write(output.join(format!("game-{id:04}.psr")), record.to_string()).unwrap();
        if cut.is_some() { fs::write(output.join(format!("prefix-{id:04}.psr")),prefix.to_string()).unwrap(); }
        fs::write(output.join(format!("game-{id:04}.json")),serde_json::to_vec(&row).unwrap()).unwrap();
        let count = completed.fetch_add(1,Ordering::Relaxed)+1;
        if count % 20 == 0 || count == games { eprintln!("mode={mode} completed={count}/{games} elapsed_seconds={:.3}",started.elapsed().as_secs_f64()); }
        row
    }).collect());
    let seconds = started.elapsed().as_secs_f64();
    let valid = rows
        .iter()
        .filter(|r| r["prefix_decisions"].is_u64())
        .count();
    let decisions: u64 = rows.iter().map(|r| r["decisions"].as_u64().unwrap()).sum();
    let summary = json!({"mode":mode,"attempts":games,"valid_starts":valid,"unfinished_or_unusable":games-valid,
        "wall_seconds":seconds,"decisions":decisions,"mean_decisions":decisions as f64/games as f64,
        "games_per_second":games as f64/seconds,"rows":rows});
    fs::write(
        output.join("result.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!(
        "mode={mode} attempts={games} valid={valid} seconds={seconds:.3} decisions={decisions}"
    );
    Ok(())
}
