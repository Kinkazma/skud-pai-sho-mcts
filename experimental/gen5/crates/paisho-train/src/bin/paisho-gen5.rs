//! Explicit, bounded Gen5 run or frozen comparison; no automatic restart.
use paisho_train::micro_learning::*;
use std::{fs, path::Path};
fn main() {
    // Before workers: actor pools own the CPU budget, without nested BLAS pools.
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || !matches!(args[0].as_str(), "run" | "compare") {
        return Err("usage: paisho-gen5 run|compare CONFIG.json".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    if args[0] == "run" {
        let options: gen5::Options = serde_json::from_value(value)?;
        if let Some(parent) = options.output.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        paisho_platform::training_time::enable(&options.output.with_extension("pause-clock.bin"))?;
        return gen5::run(options);
    }
    let model = value["model"].as_str().ok_or("model missing")?;
    let reference = value["reference"].as_str().ok_or("reference missing")?;
    let output = value["output"].as_str().ok_or("output missing")?;
    let num = |key: &str, default: u64| value[key].as_u64().unwrap_or(default);
    let threads = num("threads", 10) as usize;
    let budget = num("budget", 64) as usize;
    let reference_budget = num("reference_budget", budget as u64) as usize;
    if threads == 0
        || budget == 0
        || budget > 2048
        || reference_budget == 0
        || reference_budget > 2048
    {
        return Err("Gen5 comparison budget must be 1..2048, threads positive".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mode = |key: &str| -> Result<paisho_ai::MicroSearchOptions, Box<dyn std::error::Error>> {
        let o = gen5::Options {
            mode: value[key].as_str().unwrap_or("puct").into(),
            proof_search: value[if key == "mode" {
                "proof_search"
            } else {
                "reference_proof_search"
            }]
            .as_bool()
            .unwrap_or(true),
            ..Default::default()
        };
        Ok(o.search(0, false)?)
    };
    compare_micro_profile(
        Path::new(model),
        Path::new(reference),
        Path::new(output),
        MicroCompareOptions {
            pairs: num("pairs", 4) as usize,
            workers: num("workers", threads as u64) as usize,
            varied_setups: true,
            simulations: budget,
            move_ms: num("move_ms", 0),
            game_seconds: value["game_seconds"].as_f64().unwrap_or(30.0),
            seconds: value["seconds"]
                .as_f64()
                .ok_or("explicit seconds required")?,
            decisions: num("decisions", 600) as usize,
            seed: num("seed", 97500),
        },
        &pool,
        gen5::RULES,
        mode("mode")?,
        mode("reference_mode")?,
        reference_budget,
    )
}
