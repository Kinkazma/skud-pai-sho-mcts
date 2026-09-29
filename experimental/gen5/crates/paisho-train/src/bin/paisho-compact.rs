//! Small CPU MCTS model tools. Performance commands must be invoked through
//! tools/benchmark_with_training_paused.py around a stable training campaign.

use paisho_train::compact_learning::{run_cost, run_offline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("prepare" | "train") => run_offline(&args),
        Some("cost") => run_cost(&args[1..]),
        Some("repetitions") => paisho_train::compact_selfplay::analyze_repetitions(&args[1..]),
        Some("selfplay") => paisho_train::compact_selfplay::run(&args[1..]),
        Some("compare-suite") => paisho_train::compact_compare::run_suite(&args[1..]),
        Some("compare") => paisho_train::compact_compare::run(&args[1..]),
        _ => {
            Err("usage: paisho-compact prepare|train|cost|selfplay|compare|repetitions --name value ...".into())
        }
    }
}
