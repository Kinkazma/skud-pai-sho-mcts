//! Gen4 CPU entry point; all runs use fresh output directories and explicit bounds.
use paisho_train::micro_learning::*;
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
fn main() {
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let command = args
        .first()
        .ok_or("expected human, selfplay, compare or upgrade-policy")?;
    let mut f = BTreeMap::new();
    for pair in args[1..].chunks(2) {
        if pair.len() != 2
            || !pair[0].starts_with("--")
            || f.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("expected distinct --flag value options".into());
        }
    }
    let output = PathBuf::from(f.remove("--output").ok_or("missing --output")?);
    let threads = f
        .remove("--threads")
        .map(str::parse::<usize>)
        .transpose()?
        .unwrap_or(std::thread::available_parallelism()?.get());
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?,
    );
    match command.as_str() {
        "upgrade-neural-memory" => {
            let path=PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            let seed:u64=f.remove("--seed").unwrap_or("9478").parse()?;
            if !f.is_empty(){return Err("unknown upgrade-neural-memory option".into());}
            let parent=MicroArtifact::load(&path)?;
            let model=parent.model()?.with_neural_memory(seed);
            MicroArtifact::new(&model,parent.updates,serde_json::json!({"operation":"neutral TTT-MLP memory consolidation upgrade",
                "architecture":[483,130,520,130,2],"additional_parameters":199292,"seed":seed,
                "inputs":"state417/action32/reader32/current-value/base-action-logit",
                "outputs":"policy-logit-residual/auxiliary-action-value","parent":parent.identity(),
                "parent_provenance":parent.provenance,"existing_parameters_preserved":true})).save(&output)
        }
        "upgrade-deep-value" => {
            let path=PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            let seed:u64=f.remove("--seed").unwrap_or("9473").parse()?;
            if !f.is_empty() {return Err("unknown upgrade-deep-value option".into());}
            let parent=MicroArtifact::load(&path)?;
            let model=parent.model()?.with_deep_value(seed);
            MicroArtifact::new(&model,parent.updates,serde_json::json!({"operation":"zero-output deep value residual upgrade",
                "architecture":[417,128,64,32,1],"seed":seed,"parent":parent.identity(),"parent_provenance":parent.provenance,
                "existing_parameters_preserved":true})).save(&output)
        }
        "upgrade-memory" => {
            let path = PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            let memory = PathBuf::from(f.remove("--memory").ok_or("missing --memory")?).canonicalize()?;
            if !f.is_empty() {return Err("unknown upgrade-memory option".into());}
            use sha2::{Digest,Sha256};
            let spec = paisho_ai::SequenceMemorySpec{path:memory.to_string_lossy().into(),sha256:format!("{:x}",Sha256::digest(std::fs::read(&memory)?))};
            let bank = load_sequence_memory(&spec)?;
            let parent = MicroArtifact::load(&path)?;
            let model = parent.model()?.with_sequence_memory(bank);
            MicroArtifact::new(&model,parent.updates,serde_json::json!({"operation":"sequence bank attachment","existing_parameters_preserved":true,"missing_reader_initialized_to_zero":parent.parameters.len()<paisho_ai::MICRO_MEMORY_PARAMETERS,"spatial_retrieval":model.sequence_memory().unwrap().has_spatial(),"parent":parent.identity(),"parent_provenance":parent.provenance})).save(&output)
        }
        "upgrade-spatial" => {
            let path = PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            if !f.is_empty() {return Err("unknown upgrade-spatial option".into());}
            let parent = MicroArtifact::load(&path)?;
            let model = parent.model()?.with_spatial_policy();
            MicroArtifact::new(&model, parent.updates, serde_json::json!({"operation":"zero-connection spatial upgrade", "parent":parent.identity(), "parent_provenance":parent.provenance})).save(&output)
        }
        "upgrade-policy" => {
            let path = PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            let seed: u64 = f.remove("--seed").unwrap_or("17").parse()?;
            if !f.is_empty() {
                return Err("unknown upgrade-policy option".into());
            }
            let parent = MicroArtifact::load(&path)?;
            let model = parent.model()?.with_residual_policy(seed);
            MicroArtifact::new(
                &model,
                parent.updates,
                serde_json::json!({
                    "operation": "zero-output residual policy upgrade", "parent": parent.identity(),
                    "parent_provenance": parent.provenance, "seed": seed,
                }),
            )
            .save(&output)
        }
        "human" => {
            let dataset = PathBuf::from(f.remove("--dataset").ok_or("missing --dataset")?);
            let parent = f.remove("--model").map(PathBuf::from);
            let o = MicroFitOptions {
                epochs: f.remove("--epochs").unwrap_or("40").parse()?,
                patience: f.remove("--patience").unwrap_or("8").parse()?,
                batch: f.remove("--batch").unwrap_or("64").parse()?,
                rate: f.remove("--rate").unwrap_or("0.05").parse()?,
                l2: f.remove("--l2").unwrap_or("0.00001").parse()?,
                seed: f.remove("--seed").unwrap_or("4").parse()?,
                max_games: f
                    .remove("--max-games")
                    .unwrap_or("18446744073709551615")
                    .parse()?,
            };
            if !f.is_empty() {
                return Err("unknown human option".into());
            }
            pool.install(|| {
                fit_micro_human(&dataset, parent.as_deref(), &output, o).map_err(|e| e.to_string())
            })
            .map_err(Into::into)
        }
        "compare" => {
            let model = PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            let reference = PathBuf::from(f.remove("--reference").ok_or("missing --reference")?);
            let o = MicroCompareOptions {
                varied_setups: f.remove("--varied-setups").unwrap_or("false").parse()?,
                workers: f
                    .remove("--workers")
                    .map(str::parse)
                    .transpose()?
                    .unwrap_or(threads),
                pairs: f.remove("--pairs").unwrap_or("8").parse()?,
                simulations: f.remove("--simulations").unwrap_or("64").parse()?,
                move_ms: f.remove("--move-ms").unwrap_or("0").parse()?,
                game_seconds: f.remove("--game-seconds").unwrap_or("30").parse()?,
                seconds: f
                    .remove("--seconds")
                    .ok_or("comparison requires --seconds")?
                    .parse()?,
                decisions: f.remove("--decisions").unwrap_or("600").parse()?,
                seed: f.remove("--seed").unwrap_or("47").parse()?,
            };
            if !f.is_empty() {
                return Err("unknown compare option".into());
            }
            compare_micro(&model, &reference, &output, o, &pool)
        }
        "selfplay" => {
            let model = PathBuf::from(f.remove("--model").ok_or("missing --model")?);
            let o = MicroSelfplayOptions {
                history_interval: f.remove("--history-interval").unwrap_or("0").parse()?,
                games: f.remove("--games").unwrap_or("100000").parse()?,
                seconds: f
                    .remove("--seconds")
                    .ok_or("selfplay requires explicit --seconds")?
                    .parse()?,
                game_seconds: f.remove("--game-seconds").unwrap_or("30").parse()?,
                workers: f
                    .remove("--workers")
                    .map(str::parse)
                    .transpose()?
                    .unwrap_or(threads),
                decision_limit: f.remove("--decision-limit").unwrap_or("600").parse()?,
                low_budget: f.remove("--low-budget").unwrap_or("64").parse()?,
                high_budget: f.remove("--high-budget").unwrap_or("256").parse()?,
                high_fraction: f.remove("--high-fraction").unwrap_or("0.2").parse()?,
                seed: f.remove("--seed").unwrap_or("4").parse()?,
                replay_capacity: f.remove("--replay-capacity").unwrap_or("4096").parse()?,
                replay_ratio: f.remove("--replay-ratio").unwrap_or("4").parse()?,
                rate: f.remove("--rate").unwrap_or("0.02").parse()?,
                replay_input: f.remove("--replay").map(PathBuf::from),
            };
            if !f.is_empty() {
                return Err("unknown selfplay option".into());
            }
            run_micro_selfplay(&model, &output, o, pool)
        }
        _ => Err("expected human, selfplay, compare or upgrade-policy".into()),
    }
}
