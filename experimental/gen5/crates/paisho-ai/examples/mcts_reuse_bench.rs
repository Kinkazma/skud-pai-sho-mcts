//! Read-only matched real-position ABBA. Run via benchmark_with_training_paused.py.
//! CPU heuristic evaluator; no training, no claim about complete-game strength.
use paisho_ai::{CpuMctsEvaluator, MctsAgent, MctsConfig, MctsSession};
use paisho_core::{legal_actions, GameOutcome, GameRecord, };
use std::{error::Error, fs, time::Instant};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: mcts_reuse_bench INPUT.psr".into());
    }
    let record: GameRecord = fs::read_to_string(&args[0])?.parse()?;
    let mut position = record.initial_position();
    let mut panel = Vec::new();
    for &action in record.actions().iter().take(32) {
        if position.outcome() != GameOutcome::Ongoing {
            break;
        }
        panel.push((position.clone(), action));
        position.apply(action)?;
    }
    if panel.is_empty() {
        return Err("PSR has no ongoing decisions".into());
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    pool.install(|| -> Result<(), String> {
        println!("budget,pass,mode,positions,seconds,evaluated_actions,inherited_visits,reused_positions,reused_values");
        for budget in [32, 64, 128, 256, 512] {
            let config = MctsConfig { simulations: budget, ..MctsConfig::default() };
            // Warm up all paths; do not include this search in measurements.
            MctsSession::new(11, config, &CpuMctsEvaluator)?.search_until(&panel[0].0, &legal_actions(&panel[0].0), None)?;
            for (pass, mode) in ["legacy", "cache", "retained", "retained", "cache", "legacy"].into_iter().enumerate() {
                let mut session = MctsSession::new(81, config, &CpuMctsEvaluator)?;
                let mut total = [0usize; 4];
                let start = Instant::now();
                for (index, (position, action)) in panel.iter().enumerate() {
                    let actions = legal_actions(position);
                    let report = if mode == "legacy" {
                        MctsAgent::new(81 + index as u64, config).unwrap().search_with_evaluator(position, &actions, &CpuMctsEvaluator)?
                    } else if mode == "cache" {
                        let mut fresh = MctsSession::new(81 + index as u64, config, &CpuMctsEvaluator)?;
                        let report = fresh.search_until(position, &actions, None)?;
                        let reuse = fresh.reuse_statistics();
                        total[2] += reuse.reused_candidate_positions;
                        total[3] += reuse.reused_leaf_values;
                        report
                    } else {
                        let report = session.search_until(position, &actions, None)?;
                        let reuse = session.reuse_statistics();
                        total[1] += reuse.inherited_root_visits;
                        total[2] += reuse.reused_candidate_positions;
                        total[3] += reuse.reused_leaf_values;
                        session.advance(*action);
                        report
                    };
                    total[0] += report.evaluated_actions;
                }
                println!("{budget},{pass},{mode},{},{:.9},{},{},{},{}", panel.len(), start.elapsed().as_secs_f64(), total[0], total[1], total[2], total[3]);
            }
        }
        Ok(())
    })?;
    Ok(())
}
