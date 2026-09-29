//! Frozen-position sequential CPU cost comparison, including extraction of all
//! compact features. This is evaluator latency evidence, not MCTS strength or
//! complete-game throughput. PSR replay and input validation precede timing.

use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use paisho_ai::{evaluate_position, CompactValueModel, HeuristicWeights, StableRng};
use paisho_core::{GameRecord, Position};
use serde::Serialize;

use super::{invalid, save_json_new, sha256, CompactDataset, ModelArtifact, Result};

#[derive(Debug, Serialize)]
struct CostOptions {
    dataset: PathBuf,
    output: PathBuf,
    model: Option<PathBuf>,
    max_positions: usize,
    repeats: usize,
    max_seconds: f64,
    seed: u64,
}

impl CostOptions {
    fn parse(args: &[String]) -> Result<Self> {
        let mut flags = BTreeMap::new();
        for pair in args.chunks(2) {
            if pair.len() != 2
                || !pair[0].starts_with("--")
                || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err(invalid("cost expects distinct --name value options"));
            }
        }
        let options = Self {
            dataset: flags
                .remove("--dataset")
                .ok_or_else(|| invalid("cost requires --dataset"))?
                .into(),
            output: flags
                .remove("--output")
                .ok_or_else(|| invalid("cost requires --output"))?
                .into(),
            model: flags.remove("--model").map(PathBuf::from),
            max_positions: flags.remove("--max-positions").unwrap_or("128").parse()?,
            repeats: flags.remove("--repeats").unwrap_or("100").parse()?,
            max_seconds: flags.remove("--max-seconds").unwrap_or("10").parse()?,
            seed: flags.remove("--seed").unwrap_or("1").parse()?,
        };
        if !flags.is_empty() {
            return Err(invalid("unknown cost option"));
        }
        if !(1..=1024).contains(&options.max_positions)
            || !(1..=100_000).contains(&options.repeats)
            || !options.max_seconds.is_finite()
            || !(0.001..=60.0).contains(&options.max_seconds)
        {
            return Err(invalid(
                "cost limits: positions 1..=1024; repeats 1..=100000; max-seconds 0.001..=60",
            ));
        }
        Ok(options)
    }
}

#[derive(Serialize)]
struct PanelIdentity {
    game_sha256: String,
    decision_index: usize,
    original_path: PathBuf,
    original_sha256: String,
}

#[derive(Serialize)]
struct Arm {
    evaluator: &'static str,
    completed_repeats: usize,
    calls: usize,
    seconds: f64,
    seconds_per_call: Option<f64>,
    checksum: f64,
    completed: bool,
}

/// Runs bounded ABBA evaluator measurements. The caller must arrange the
/// repository's paused-training benchmark wrapper; this command cannot infer
/// which controller the user is benchmarking against.
pub fn run_cost(args: &[String]) -> Result<()> {
    let options = CostOptions::parse(args)?;
    if options.output.exists() {
        return Err(invalid("cost output already exists"));
    }
    let dataset_bytes = fs::read(&options.dataset)?;
    let dataset: CompactDataset = serde_json::from_slice(&dataset_bytes)?;
    dataset.validate()?;
    let (artifact, model_identity) = match &options.model {
        Some(path) => {
            let bytes = fs::read(path)?;
            (
                serde_json::from_slice::<ModelArtifact>(&bytes)?,
                serde_json::json!({"path":path.canonicalize()?,"sha256":sha256(&bytes)}),
            )
        }
        None => (
            ModelArtifact::legacy(),
            serde_json::json!({"initialization":"exact-legacy-heuristic-v1"}),
        ),
    };
    let model = artifact.model()?;
    let (positions, panel) = reconstruct_panel(&dataset, options.max_positions, options.seed)?;
    let binary = std::env::current_exe()?;
    let binary_sha256 = sha256(&fs::read(&binary)?);
    let legacy = HeuristicWeights::default();
    let maximum_initial_difference = positions
        .iter()
        .map(|position| {
            f64::from(
                (evaluate_position(position, position.to_move(), legacy)
                    - model.evaluate(position, position.to_move()))
                .abs(),
            )
        })
        .fold(0.0_f64, f64::max);
    if options.model.is_none() && maximum_initial_difference != 0.0 {
        return Err(invalid(
            "initial compact evaluator failed exact legacy parity on the selected panel",
        ));
    }
    // Warm each evaluator on exactly the same real positions before timing.
    for position in &positions {
        black_box(evaluate_position(
            black_box(position),
            position.to_move(),
            legacy,
        ));
        black_box(model.evaluate(black_box(position), position.to_move()));
    }
    let started = Instant::now();
    let budget = Duration::from_secs_f64(options.max_seconds);
    let mut arms = Vec::new();
    for compact in [false, true, true, false] {
        arms.push(measure(
            &positions,
            &model,
            compact,
            options.repeats,
            started,
            budget,
        ));
    }
    let timed_wall_seconds = started.elapsed().as_secs_f64();
    let complete = arms.iter().all(|arm| arm.completed);
    let ratio =
        complete.then(|| (arms[1].seconds + arms[2].seconds) / (arms[0].seconds + arms[3].seconds));
    let report = serde_json::json!({
        "schema":"paisho-compact-evaluator-cost-v1",
        "rules":dataset.rules,
        "scope":"sequential evaluator latency, includes feature extraction; excludes replay, MCTS and game throughput",
        "options":options,
        "binary":{"path":binary,"sha256":binary_sha256},
        "dataset":{"path":options.dataset.canonicalize()?,"sha256":sha256(&dataset_bytes)},
        "model":model_identity,
        "panel":panel,
        "positions":positions.len(),
        "maximum_panel_value_difference_from_legacy":maximum_initial_difference,
        "arms":arms,"complete_matched_work":complete,"compact_over_legacy_time":ratio,
        "timed_wall_seconds":timed_wall_seconds,
        "time_limit_check":"between complete panel passes; at most one pass may overrun the global time cap"
    });
    save_json_new(&options.output, &report)?;
    println!(
        "{}",
        serde_json::json!({"output":options.output,"positions":positions.len(),"complete_matched_work":complete,"compact_over_legacy_time":ratio})
    );
    Ok(())
}

fn reconstruct_panel(
    dataset: &CompactDataset,
    maximum: usize,
    seed: u64,
) -> Result<(Vec<Position>, Vec<PanelIdentity>)> {
    let mut candidates: Vec<_> = dataset
        .games
        .iter()
        .enumerate()
        .flat_map(|(game_index, game)| {
            game.examples
                .iter()
                .map(move |example| (game_index, example.decision_index))
        })
        .collect();
    let mut rng = StableRng::new(seed);
    for index in (1..candidates.len()).rev() {
        let other = rng.index(index + 1);
        candidates.swap(index, other);
    }
    candidates.truncate(maximum);
    candidates.sort_unstable();
    if candidates.is_empty() {
        return Err(invalid("cost dataset contains no positions"));
    }
    let mut wanted: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (game, decision) in candidates {
        wanted.entry(game).or_default().push(decision);
    }
    let mut positions = Vec::new();
    let mut panel = Vec::new();
    for (game_index, decisions) in wanted {
        let game = &dataset.games[game_index];
        let original = &game.originals[0];
        let bytes = fs::read(&original.path)?;
        if sha256(&bytes) != original.sha256 {
            return Err(invalid("cost original PSR byte hash mismatch"));
        }
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        if sha256(record.to_string().as_bytes()) != game.game_sha256
            || record.actions().len() != game.decisions
            || record.rules().as_str() != dataset.rules
        {
            return Err(invalid(
                "cost original PSR canonical identity or decision count mismatch",
            ));
        }
        let mut remaining = decisions.into_iter().peekable();
        let mut position = record.initial_position();
        for (decision_index, action) in record.actions().iter().enumerate() {
            if remaining.peek() == Some(&decision_index) {
                remaining.next();
                positions.push(position.clone());
                panel.push(PanelIdentity {
                    game_sha256: game.game_sha256.clone(),
                    decision_index,
                    original_path: original.path.clone(),
                    original_sha256: original.sha256.clone(),
                });
            }
            position.apply(*action)?;
        }
    }
    Ok((positions, panel))
}

fn measure(
    positions: &[Position],
    model: &CompactValueModel,
    compact: bool,
    repeats: usize,
    global_started: Instant,
    budget: Duration,
) -> Arm {
    let started = Instant::now();
    let mut completed_repeats = 0;
    let mut checksum = 0.0;
    for _ in 0..repeats {
        if global_started.elapsed() >= budget {
            break;
        }
        for position in positions {
            let score = if compact {
                model.evaluate(black_box(position), position.to_move())
            } else {
                evaluate_position(
                    black_box(position),
                    position.to_move(),
                    HeuristicWeights::default(),
                )
            };
            checksum += f64::from(black_box(score));
        }
        completed_repeats += 1;
    }
    let seconds = started.elapsed().as_secs_f64();
    let calls = completed_repeats * positions.len();
    Arm {
        evaluator: if compact { "compact" } else { "legacy" },
        completed_repeats,
        calls,
        seconds,
        seconds_per_call: (calls > 0).then(|| seconds / calls as f64),
        checksum,
        completed: completed_repeats == repeats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn args(extra: &[&str]) -> Vec<String> {
        ["--dataset", "data.json", "--output", "cost.json"]
            .into_iter()
            .chain(extra.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn cost_parser_retains_bounded_explicit_options() {
        let defaults = CostOptions::parse(&args(&[])).unwrap();
        assert_eq!(defaults.max_positions, 128);
        assert_eq!(defaults.repeats, 100);
        assert_eq!(defaults.max_seconds, 10.0);
        let parsed = CostOptions::parse(&args(&[
            "--max-positions",
            "16",
            "--repeats",
            "2",
            "--max-seconds",
            "1.5",
            "--model",
            "candidate.json",
            "--seed",
            "12",
        ]))
        .unwrap();
        assert_eq!(parsed.max_positions, 16);
        assert_eq!(parsed.repeats, 2);
        assert_eq!(parsed.max_seconds, 1.5);
        assert_eq!(parsed.seed, 12);
        assert_eq!(parsed.model, Some("candidate.json".into()));
    }

    #[test]
    fn cost_parser_rejects_ignored_or_unbounded_options() {
        for extra in [
            vec!["--repeats", "0"],
            vec!["--max-positions", "1025"],
            vec!["--max-seconds", "NaN"],
            vec!["--max-seconds", "61"],
            vec!["--unknown", "1"],
            vec!["--seed"],
            vec!["--output", "duplicate.json"],
        ] {
            assert!(CostOptions::parse(&args(&extra)).is_err(), "{extra:?}");
        }
    }
}
