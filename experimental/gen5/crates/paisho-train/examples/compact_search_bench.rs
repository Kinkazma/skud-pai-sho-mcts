//! Frozen real-position search cost and exact action-statistics fingerprints.
//! Run with benchmark_with_training_paused.py; no training or game collection.
use paisho_ai::{CompactValueModel, HeuristicWeights, MctsAgent, MctsConfig, MctsEvaluator};
use paisho_core::{legal_actions, GameOutcome, GameRecord, Player, Position};
use paisho_train::compact_learning::load_model;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

struct TimedEvaluator<'a> {
    model: &'a CompactValueModel,
    ranking_ns: AtomicU64,
    leaf_ns: AtomicU64,
    ranking_positions: AtomicU64,
    leaves: AtomicU64,
}
impl MctsEvaluator for TimedEvaluator<'_> {
    fn evaluate(
        &self,
        positions: &[Position],
        player: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        let start = Instant::now();
        let result = MctsEvaluator::evaluate(self.model, positions, player, weights);
        self.ranking_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.ranking_positions
            .fetch_add(positions.len() as u64, Ordering::Relaxed);
        result
    }
    fn evaluate_leaf(
        &self,
        position: &Position,
        player: Player,
        weights: HeuristicWeights,
    ) -> Result<f32, String> {
        let start = Instant::now();
        let result = MctsEvaluator::evaluate_leaf(self.model, position, player, weights);
        self.leaf_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.leaves.fetch_add(1, Ordering::Relaxed);
        result
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("usage: compact_search_bench MODEL PSR_DIRECTORY OUTPUT REPEATS".into());
    }
    let model_path = PathBuf::from(&args[0]);
    let directory = PathBuf::from(&args[1]);
    let output = PathBuf::from(&args[2]);
    let repeats: usize = args[3].parse()?;
    if output.exists() || !(1..=20).contains(&repeats) {
        return Err("new output and 1..=20 repeats required".into());
    }
    let model = load_model(&model_path)?.model()?;
    let mut files: Vec<_> = fs::read_dir(directory)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "psr"))
        .collect();
    files.sort();
    files.truncate(4);
    if files.len() != 4 {
        return Err("four PSRs required".into());
    }
    let mut panel = Vec::new();
    for file in files {
        let raw = fs::read(&file)?;
        let record: GameRecord = std::str::from_utf8(&raw)?.parse()?;
        let mut wanted: Vec<_> = [1, 3, 5, 7, 9]
            .into_iter()
            .map(|n| record.actions().len() * n / 10)
            .collect();
        wanted.sort();
        wanted.dedup();
        let mut position = record.initial_position();
        for (decision, action) in record.actions().iter().enumerate() {
            if wanted.contains(&decision) && position.outcome() == GameOutcome::Ongoing {
                panel.push((position.clone(),json!({"file":file,"sha256":format!("{:x}",Sha256::digest(&raw)),"decision":decision})));
            }
            position.apply(*action)?;
        }
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let results: Result<Vec<Value>,String>=pool.install(|| {
        let mut rows=Vec::new();
        // One complete warmup per budget; no warmup timings enter the report.
        for budget in [128,256,512] {
            let config=MctsConfig {simulations:budget,..MctsConfig::default()};
            let p=&panel[0].0;
            MctsAgent::new(123,config).unwrap().search_with_evaluator(p,&legal_actions(p),&model)?;
        }
        for repeat in 0..repeats {
            for budget in [128,256,512] {
                for (index,(position,identity)) in panel.iter().enumerate() {
                    let evaluator=TimedEvaluator {model:&model,ranking_ns:AtomicU64::new(0),leaf_ns:AtomicU64::new(0),ranking_positions:AtomicU64::new(0),leaves:AtomicU64::new(0)};
                    let config=MctsConfig {simulations:budget,..MctsConfig::default()};
                    let seed=850_512 + index as u64*10_000 + repeat as u64;
                    let mut agent=MctsAgent::new(seed,config).unwrap();
                    let start=Instant::now();
                    let actions=legal_actions(position);
                    let report=agent.search_with_evaluator(position,&actions,&evaluator)?;
                    let seconds=start.elapsed().as_secs_f64();
                    let mut hash=Sha256::new();
                    hash.update((report.selected_index as u64).to_le_bytes());
                    hash.update((report.simulations as u64).to_le_bytes());
                    for a in &report.actions {
                        hash.update(format!("{:?}",a.action));
                        hash.update((a.visits as u64).to_le_bytes());
                        hash.update(a.value_sum.to_bits().to_le_bytes());
                    }
                    rows.push(json!({"budget":budget,"repeat":repeat,"panel_index":index,"seed":seed,"seconds":seconds,
                        "search_fingerprint":format!("{:x}",hash.finalize()),"legal_actions":actions.len(),
                        "evaluated_actions":report.evaluated_actions,"expanded_nodes":report.expanded_nodes,
                        "generated_nodes":report.generated_nodes,"generated_actions":report.generated_actions,
                        "maximum_depth":report.maximum_depth,
                        "ranking_value_seconds":evaluator.ranking_ns.load(Ordering::Relaxed) as f64/1e9,
                        "leaf_value_seconds":evaluator.leaf_ns.load(Ordering::Relaxed) as f64/1e9,
                        "ranking_positions":evaluator.ranking_positions.load(Ordering::Relaxed),
                        "leaf_evaluations":evaluator.leaves.load(Ordering::Relaxed),"position":identity}));
                }
            }
        }
        Ok(rows)
    });
    let rows = results?;
    let report = json!({"schema":"compact-search-cost-v1","source_sha256":env!("PAISHO_BUILD_SOURCE_SHA256"),
        "binary_sha256":format!("{:x}",Sha256::digest(fs::read(std::env::current_exe()?)?)),
        "model_sha256":format!("{:x}",Sha256::digest(fs::read(model_path)?)),
        "scope":"single-worker real-position searches including root legality; evaluator clocks included, no complete-game throughput or strength claim",
        "positions":panel.len(),"repeats":repeats,"rows":rows});
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    println!(
        "{}",
        json!({"searches":report["rows"].as_array().unwrap().len(),"positions":panel.len()})
    );
    Ok(())
}
