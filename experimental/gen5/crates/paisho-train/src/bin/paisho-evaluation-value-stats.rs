use std::collections::HashSet;
use std::error::Error;
use std::fmt::Write;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use paisho_core::{GameRecord, Player};
use paisho_mpsgraph_client::{
    default_service_path, read_checkpoint_metadata, CapacityInferenceBroker,
    InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_train::{
    materialize_candidate_value_examples_v1, summarize_value_predictions_v1,
    CandidateValueExampleV1, EvaluationCampaignArchive, ValuePredictionObservationV1,
    ValuePredictionSummaryV1,
};
use rayon::prelude::*;
use serde::Deserialize;

type BoxError = Box<dyn Error + Send + Sync>;

struct Options {
    evaluation_directory: PathBuf,
    checkpoint: PathBuf,
    service: PathBuf,
    workers: usize,
    maximum_batch_wait: Duration,
    model_seed: u64,
    maximum_examples: Option<usize>,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, BoxError> {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut evaluation_directory = None;
        let mut checkpoint = None;
        let mut service = default_service_path(&repository);
        let mut workers = std::thread::available_parallelism()?.get();
        let mut maximum_batch_wait = Duration::from_millis(5);
        let mut model_seed = 17;
        let mut maximum_examples = None;
        let arguments = arguments.collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            if flag == "--help" || flag == "-h" {
                print_usage();
                std::process::exit(0);
            }
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--evaluation-dir" => evaluation_directory = Some(PathBuf::from(value)),
                "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
                "--service" => service = PathBuf::from(value),
                "--workers" => workers = positive_usize(value, flag)?,
                "--wait-us" => maximum_batch_wait = Duration::from_micros(value.parse()?),
                "--model-seed" => model_seed = value.parse()?,
                "--examples" => maximum_examples = Some(positive_usize(value, flag)?),
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        Ok(Self {
            evaluation_directory: evaluation_directory.ok_or("missing --evaluation-dir PATH")?,
            checkpoint: checkpoint.ok_or("missing --checkpoint PATH")?,
            service,
            workers,
            maximum_batch_wait,
            model_seed,
            maximum_examples,
        })
    }
}

#[derive(Deserialize)]
struct BatchFile {
    format: String,
    games: Vec<ArchivedGame>,
}

#[derive(Deserialize)]
struct ArchivedGame {
    candidate_is_host: bool,
    termination: Option<String>,
    record: Option<String>,
    error: Option<String>,
    host_telemetry: Option<ArchivedTelemetry>,
    guest_telemetry: Option<ArchivedTelemetry>,
}

#[derive(Clone, Copy, Deserialize)]
struct ArchivedTelemetry {
    decisions: usize,
}

struct LoadedExamples {
    attempted_games: usize,
    terminal_games: usize,
    skipped_games: usize,
    examples: Vec<CandidateValueExampleV1>,
}

fn main() -> Result<(), BoxError> {
    let options = Options::parse(std::env::args().skip(1))?;
    let archive = EvaluationCampaignArchive::open_existing(&options.evaluation_directory)?;
    let analysis = archive.analysis()?;
    let identity = archive.identity();
    let preset = parse_preset(&identity.preset)?;
    let optimization = parse_level(identity.optimization_level)?;
    let mut loaded = load_examples(&options.evaluation_directory)?;
    if let Some(maximum) = options.maximum_examples {
        loaded.examples.truncate(maximum);
    }
    if loaded.examples.is_empty() {
        return Err("evaluation archive contains no terminal candidate decision".into());
    }
    let checkpoint = read_checkpoint_metadata(&options.checkpoint)?;
    if checkpoint.network_preset() != preset {
        return Err("checkpoint and evaluation network presets differ".into());
    }
    let services = identity
        .inference_classes
        .iter()
        .map(|class| ServiceConfiguration {
            executable: options.service.clone(),
            preset,
            batch_size: class.batch_size,
            legal_action_capacity: class.legal_action_capacity,
            inference_slots: 1,
            optimization,
            seed: options.model_seed,
            checkpoint: Some(options.checkpoint.clone()),
        })
        .collect();
    let broker = CapacityInferenceBroker::launch(
        services,
        InferenceBrokerConfiguration {
            prepare_ahead: false,
            maximum_batch_wait: options.maximum_batch_wait,
            maximum_in_flight_batches: 1,
        },
    )?;
    let client = broker.client()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let observed_workers = Arc::new(Mutex::new(HashSet::new()));
    let started = Instant::now();
    let observations = pool
        .install(|| {
            loaded
                .examples
                .par_iter()
                .map(|example| -> Result<ValuePredictionObservationV1, String> {
                    observed_workers
                        .lock()
                        .map_err(|_| "worker observation lock was poisoned".to_owned())?
                        .insert(rayon::current_thread_index().expect("inside configured pool"));
                    let output = client
                        .infer_encoded(example.inference().clone())
                        .map_err(|source| source.to_string())?;
                    ValuePredictionObservationV1::new(
                        example.target(),
                        *output.value_probabilities(),
                    )
                    .map_err(|source| source.to_string())
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .map_err(|message| -> BoxError { message.into() })?;
    let elapsed = started.elapsed();
    let observed_worker_count = observed_workers
        .lock()
        .map_err(|_| "worker observation lock was poisoned")?
        .len();
    drop(client);
    let telemetry = broker.shutdown()?;
    if observed_worker_count != options.workers && observations.len() >= options.workers {
        return Err(format!(
            "only {observed_worker_count}/{} CPU workers processed examples",
            options.workers
        )
        .into());
    }

    println!(
        "evaluation_directory={}",
        options.evaluation_directory.display()
    );
    println!(
        "archive_candidate_checkpoint={}",
        identity.candidate_checkpoint_sha256
    );
    println!(
        "measured_checkpoint={}",
        hex_digest(checkpoint.content_sha256())
    );
    println!(
        "checkpoint_is_archive_candidate={}",
        identity.candidate_checkpoint_sha256 == hex_digest(checkpoint.content_sha256())
    );
    println!("attempted_pairs={}", analysis.attempted_pairs);
    println!("eligible_pairs={}", analysis.eligible_pairs);
    println!("attempted_games={}", loaded.attempted_games);
    println!("terminal_games={}", loaded.terminal_games);
    println!("skipped_games={}", loaded.skipped_games);
    println!("candidate_examples={}", observations.len());
    println!(
        "workers_observed={observed_worker_count}/{}",
        options.workers
    );
    println!("inference_positions={}", telemetry.requested_positions());
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!(
        "positions_per_second={:.3}",
        observations.len() as f64 / elapsed.as_secs_f64()
    );
    print_value_summary(summarize_value_predictions_v1(&observations)?);
    Ok(())
}

fn load_examples(root: &Path) -> Result<LoadedExamples, BoxError> {
    let paths = batch_json_paths(root)?;
    let mut attempted_games = 0;
    let mut terminal_games = 0;
    let mut skipped_games = 0;
    let mut examples = Vec::new();
    for path in paths {
        let batch: BatchFile = serde_json::from_slice(&fs::read(path)?)?;
        if batch.format != "PAISHO-EVALUATION-BATCH-1" {
            return Err(format!("unexpected evaluation batch format {}", batch.format).into());
        }
        for game in batch.games {
            attempted_games += 1;
            if game.error.is_some()
                || !matches!(
                    game.termination.as_deref(),
                    Some("host-win" | "guest-win" | "draw")
                )
            {
                skipped_games += 1;
                continue;
            }
            let record = game
                .record
                .ok_or("terminal evaluation game has no record")?
                .parse::<GameRecord>()?;
            let host = game
                .host_telemetry
                .ok_or("terminal evaluation game has no host telemetry")?;
            let guest = game
                .guest_telemetry
                .ok_or("terminal evaluation game has no guest telemetry")?;
            let continuation = host
                .decisions
                .checked_add(guest.decisions)
                .ok_or("evaluation continuation decision count overflow")?;
            let candidate = if game.candidate_is_host {
                Player::Host
            } else {
                Player::Guest
            };
            examples.extend(materialize_candidate_value_examples_v1(
                &record,
                candidate,
                continuation,
            )?);
            terminal_games += 1;
        }
    }
    Ok(LoadedExamples {
        attempted_games,
        terminal_games,
        skipped_games,
        examples,
    })
}

fn batch_json_paths(root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut paths = Vec::new();
    for entry in fs::read_dir(root.join("batches"))? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().starts_with('.') {
            continue;
        }
        paths.push(entry.path().join("batch.json"));
    }
    paths.sort();
    Ok(paths)
}

fn print_value_summary(summary: ValuePredictionSummaryV1) {
    println!(
        "value_target_examples=win:{},draw:{},loss:{}",
        summary.classes[0].examples, summary.classes[1].examples, summary.classes[2].examples
    );
    println!(
        "value_mean_probabilities=win:{:.9},draw:{:.9},loss:{:.9}",
        summary.mean_probabilities[0], summary.mean_probabilities[1], summary.mean_probabilities[2]
    );
    println!("value_cross_entropy={:.9}", summary.cross_entropy);
    println!(
        "value_empirical_prior_cross_entropy={:.9}",
        summary.empirical_prior_cross_entropy
    );
    println!(
        "value_cross_entropy_excess={:.9}",
        summary.cross_entropy_excess
    );
    println!("value_brier_score={:.9}", summary.brier_score);
    println!(
        "value_empirical_prior_brier_score={:.9}",
        summary.empirical_prior_brier_score
    );
    println!("value_brier_score_excess={:.9}", summary.brier_score_excess);
    println!("value_top_one_accuracy={:.9}", summary.top_one_accuracy);
    println!(
        "value_majority_class_accuracy={:.9}",
        summary.majority_class_accuracy
    );
    println!(
        "value_accuracy_over_majority={:.9}",
        summary.accuracy_over_majority
    );
    println!(
        "value_mean_signed_prediction={:.9}",
        summary.mean_signed_prediction
    );
    println!("value_mean_signed_target={:.9}", summary.mean_signed_target);
    println!(
        "value_mean_terminal_residual={:.9}",
        summary.mean_terminal_residual
    );
    println!(
        "value_mean_absolute_terminal_residual={:.9}",
        summary.mean_absolute_terminal_residual
    );
    for (name, class) in ["win", "draw", "loss"].into_iter().zip(summary.classes) {
        print_optional(
            &format!("value_{name}_mean_target_probability"),
            class.mean_target_probability,
        );
        print_optional(
            &format!("value_{name}_mean_signed_prediction"),
            class.mean_signed_prediction,
        );
    }
}

fn print_optional(name: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{name}={value:.9}"),
        None => println!("{name}=none"),
    }
}

fn parse_preset(value: &str) -> Result<NetworkPreset, BoxError> {
    match value {
        "micro" => Ok(NetworkPreset::Micro),
        "pure" => Ok(NetworkPreset::Pure),
        _ => Err(format!("invalid network preset {value}").into()),
    }
}

fn parse_level(value: u8) -> Result<OptimizationLevel, BoxError> {
    match value {
        0 => Ok(OptimizationLevel::Level0),
        1 => Ok(OptimizationLevel::Level1),
        _ => Err(format!("invalid optimization level {value}").into()),
    }
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, BoxError> {
    let value = value.parse::<usize>()?;
    if value == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(value)
    }
}

fn hex_digest(bytes: [u8; 32]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
            hex
        })
}

fn print_usage() {
    println!(
        "usage: paisho-evaluation-value-stats --evaluation-dir PATH \
         --checkpoint PATH [--service PATH] [--workers N] [--wait-us N] \
         [--model-seed N] [--examples N]"
    );
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn hidden_partial_batches_are_not_diagnostic_inputs() {
        let root = std::env::temp_dir().join(format!(
            "paisho-evaluation-value-stats-{}-{}",
            std::process::id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        let batches = root.join("batches");
        let published = batches.join("batch-00000000000000000000");
        let partial = batches.join(".batch-00000000000000000001.partial-test");
        fs::create_dir_all(&published).unwrap();
        fs::create_dir_all(partial).unwrap();

        assert_eq!(
            batch_json_paths(&root).unwrap(),
            vec![published.join("batch.json")]
        );

        fs::remove_dir_all(root).unwrap();
    }
}
