use std::collections::HashSet;
use std::error::Error;
use std::fmt::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use paisho_mpsgraph_client::{
    default_service_path, read_checkpoint_metadata, CapacityInferenceBroker,
    InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_replay::{
    ReplayDatasetV1, ReplayDigestV1, ReplaySamplerStateV1, ReplaySamplerV1, ReplaySnapshotV1,
};
use paisho_train::{
    observe_played_policy_update_v1, observe_sampling_policy_v1, summarize_policy_observations_v1,
    summarize_policy_updates_v1, summarize_value_predictions_v1, DiagnosticRangeV1,
    PolicyDistributionObservationV1, PolicyUpdateClassSummaryV1, PolicyUpdateObservationV1,
    PolicyUpdateSummaryV1, SamplingProfileV1, ValuePredictionObservationV1,
    ValuePredictionSummaryV1,
};
use rayon::prelude::*;

type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Copy, Debug)]
struct ClassShape {
    capacity: usize,
    batch_size: usize,
}

struct Options {
    snapshot: PathBuf,
    replay_directory: PathBuf,
    behavior_producer: ReplayDigestV1,
    service: PathBuf,
    checkpoint: PathBuf,
    preset: NetworkPreset,
    optimization: OptimizationLevel,
    classes: Vec<ClassShape>,
    workers: usize,
    maximum_batch_wait: Duration,
    model_seed: u64,
    sampler_seed: u64,
    start_replay_index: u64,
    examples: Option<usize>,
    profiles: Vec<SamplingProfileV1>,
    ppo_clip: f64,
}

struct NetworkDiagnosticRow {
    policies: Vec<PolicyDistributionObservationV1>,
    updates: Vec<PolicyUpdateObservationV1>,
    value: ValuePredictionObservationV1,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, BoxError> {
        let available_workers = std::thread::available_parallelism()?.get();
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut snapshot = None;
        let mut replay_directory = None;
        let mut behavior_producer = None;
        let mut service = default_service_path(&repository);
        let mut checkpoint = None;
        let mut preset = NetworkPreset::Pure;
        let mut optimization = OptimizationLevel::Level1;
        let mut classes = vec![
            ClassShape {
                capacity: 64,
                batch_size: 8,
            },
            ClassShape {
                capacity: 128,
                batch_size: 4,
            },
            ClassShape {
                capacity: 1_024,
                batch_size: 4,
            },
        ];
        let mut workers = available_workers;
        let mut maximum_batch_wait = Duration::from_millis(5);
        let mut model_seed = 17;
        let mut sampler_seed = 0;
        let mut start_replay_index = 0;
        let mut examples = None;
        let mut profiles = vec![SamplingProfileV1::new(1.0, 0.05)?];
        let mut ppo_clip = 0.2;

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
                "--snapshot" => snapshot = Some(PathBuf::from(value)),
                "--replay-dir" => replay_directory = Some(PathBuf::from(value)),
                "--behavior-producer" => behavior_producer = Some(value.parse()?),
                "--service" => service = PathBuf::from(value),
                "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
                "--preset" => preset = parse_preset(value)?,
                "--level" => optimization = parse_level(value)?,
                "--classes" => classes = parse_classes(value)?,
                "--workers" => workers = positive_usize(value, flag)?,
                "--wait-us" => maximum_batch_wait = Duration::from_micros(value.parse()?),
                "--model-seed" => model_seed = value.parse()?,
                "--sampler-seed" => sampler_seed = value.parse()?,
                "--start-index" => start_replay_index = value.parse()?,
                "--examples" => examples = Some(positive_usize(value, flag)?),
                "--profiles" => profiles = parse_profiles(value)?,
                "--ppo-clip" => ppo_clip = value.parse()?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        Ok(Self {
            snapshot: snapshot.ok_or("missing --snapshot PATH")?,
            replay_directory: replay_directory.ok_or("missing --replay-dir DIR")?,
            behavior_producer: behavior_producer.ok_or("missing --behavior-producer SHA256")?,
            service,
            checkpoint: checkpoint.ok_or("missing --checkpoint PATH")?,
            preset,
            optimization,
            classes,
            workers,
            maximum_batch_wait,
            model_seed,
            sampler_seed,
            start_replay_index,
            examples,
            profiles,
            ppo_clip,
        })
    }
}

fn main() -> Result<(), BoxError> {
    let options = Options::parse(std::env::args().skip(1))?;
    let snapshot = ReplaySnapshotV1::read(&options.snapshot)?;
    let dataset = ReplayDatasetV1::from_snapshot_for_behavior(
        &snapshot,
        &options.replay_directory,
        options.behavior_producer,
    )?;
    let example_count = options.examples.unwrap_or(dataset.len());
    let start_replay_index = usize::try_from(options.start_replay_index)
        .map_err(|_| "start replay index does not fit usize")?;
    if start_replay_index > dataset.len()
        || example_count > dataset.len().saturating_sub(start_replay_index)
    {
        return Err(format!(
            "requested {example_count} examples from index {start_replay_index}, but the verified dataset contains only {}",
            dataset.len(),
        )
        .into());
    }
    let mut sampler = ReplaySamplerV1::resume(
        &dataset,
        ReplaySamplerStateV1::new(
            dataset.snapshot_digest(),
            options.sampler_seed,
            options.start_replay_index,
        ),
    )?;
    let sampled = sampler.prepare_batch(example_count)?;
    let checkpoint = read_checkpoint_metadata(&options.checkpoint)?;
    if checkpoint.network_preset() != options.preset {
        return Err("checkpoint and requested network preset differ".into());
    }

    let services = options
        .classes
        .iter()
        .map(|class| ServiceConfiguration {
            executable: options.service.clone(),
            preset: options.preset,
            batch_size: class.batch_size,
            legal_action_capacity: class.capacity,
            inference_slots: 1,
            optimization: options.optimization,
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
    let rows = pool
        .install(|| {
            sampled
                .examples()
                .par_iter()
                .map(|example| -> Result<NetworkDiagnosticRow, String> {
                    observed_workers
                        .lock()
                        .map_err(|_| "worker observation lock was poisoned".to_owned())?
                        .insert(rayon::current_thread_index().expect("inside the configured pool"));
                    let output = client
                        .infer_encoded(example.inference().clone())
                        .map_err(|source| source.to_string())?;
                    let policies = options
                        .profiles
                        .iter()
                        .map(|&profile| {
                            observe_sampling_policy_v1(output.policy_probabilities(), profile)
                                .map_err(|source| source.to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let behavior_probability =
                        example.played_behavior_probability().ok_or_else(|| {
                            "selected replay example is not a behavior policy".to_owned()
                        })?;
                    let updates = options
                        .profiles
                        .iter()
                        .map(|&profile| {
                            observe_played_policy_update_v1(
                                output.policy_probabilities(),
                                example.played_action_index(),
                                behavior_probability,
                                example.value_class(),
                                profile,
                            )
                            .map_err(|source| source.to_string())
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let value = ValuePredictionObservationV1::new(
                        example.value_class(),
                        *output.value_probabilities(),
                    )
                    .map_err(|source| source.to_string())?;
                    Ok(NetworkDiagnosticRow {
                        policies,
                        updates,
                        value,
                    })
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
    if observed_worker_count != options.workers && example_count >= options.workers {
        return Err(format!(
            "only {observed_worker_count}/{} CPU workers processed examples",
            options.workers
        )
        .into());
    }

    println!("snapshot_sha256={}", snapshot.digest());
    println!("behavior_producer={}", options.behavior_producer);
    println!(
        "checkpoint_sha256={}",
        hex_digest(checkpoint.content_sha256())
    );
    println!("dataset_examples={}", dataset.len());
    println!("start_replay_index={}", options.start_replay_index);
    println!("measured_examples={example_count}");
    println!("workers_configured={}", options.workers);
    println!("workers_observed={observed_worker_count}");
    println!("inference_positions={}", telemetry.requested_positions());
    println!(
        "inference_executed_positions={}",
        telemetry.executed_positions()
    );
    println!("elapsed_seconds={:.6}", elapsed.as_secs_f64());
    println!(
        "positions_per_second={:.3}",
        example_count as f64 / elapsed.as_secs_f64()
    );
    for (profile_index, &profile) in options.profiles.iter().enumerate() {
        let observations = rows
            .iter()
            .map(|row| row.policies[profile_index])
            .collect::<Vec<_>>();
        let summary = summarize_policy_observations_v1(&observations)?;
        print_summary(profile, summary);
        let updates = rows
            .iter()
            .map(|row| row.updates[profile_index])
            .collect::<Vec<_>>();
        print_policy_update_summary(summarize_policy_updates_v1(&updates, options.ppo_clip)?);
    }
    let value_observations = rows.iter().map(|row| row.value).collect::<Vec<_>>();
    print_value_summary(summarize_value_predictions_v1(&value_observations)?);
    Ok(())
}

fn print_policy_update_summary(summary: PolicyUpdateSummaryV1) {
    println!("policy_update_examples={}", summary.examples);
    println!("policy_update_forced_examples={}", summary.forced_examples);
    print_policy_update_class("overall", summary.overall);
    for (name, class) in ["win", "draw", "loss"].into_iter().zip(summary.classes) {
        print_policy_update_class(name, class);
    }
}

fn print_policy_update_class(name: &str, summary: PolicyUpdateClassSummaryV1) {
    println!("policy_update_{name}_examples={}", summary.examples);
    println!(
        "policy_update_{name}_forced_examples={}",
        summary.forced_examples
    );
    for (metric, value) in [
        (
            "mean_behavior_probability",
            summary.mean_behavior_probability,
        ),
        (
            "mean_candidate_probability",
            summary.mean_candidate_probability,
        ),
        ("mean_importance_ratio", summary.mean_importance_ratio),
        (
            "mean_log_importance_ratio",
            summary.mean_log_importance_ratio,
        ),
        ("increased_fraction", summary.increased_fraction),
        ("unchanged_fraction", summary.unchanged_fraction),
        ("decreased_fraction", summary.decreased_fraction),
        ("below_clip_fraction", summary.below_clip_fraction),
        ("above_clip_fraction", summary.above_clip_fraction),
    ] {
        print_optional_value(&format!("policy_update_{name}_{metric}"), value);
    }
}

fn print_summary(profile: SamplingProfileV1, summary: paisho_train::PolicyDistributionSummaryV1) {
    println!(
        "profile=temperature:{:.6},uniform_mix:{:.6}",
        profile.temperature(),
        profile.uniform_mix()
    );
    println!("examples={}", summary.examples);
    println!("forced_examples={}", summary.forced_examples);
    println!(
        "forced_fraction={:.9}",
        summary.forced_examples as f64 / summary.examples as f64
    );
    print_range("legal_actions", summary.legal_actions);
    print_range("entropy", summary.entropy);
    print_range("normalized_entropy", summary.normalized_entropy);
    print_range("effective_actions", summary.effective_actions);
    print_range("maximum_probability", summary.maximum_probability);
    print_range("collision_probability", summary.collision_probability);
}

fn print_range(name: &str, range: DiagnosticRangeV1) {
    println!(
        "{name}=min:{:.9},mean:{:.9},max:{:.9}",
        range.minimum, range.mean, range.maximum
    );
}

fn print_value_summary(summary: ValuePredictionSummaryV1) {
    println!("value_examples={}", summary.examples);
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
        print_optional_value(
            &format!("value_{name}_mean_target_probability"),
            class.mean_target_probability,
        );
        print_optional_value(
            &format!("value_{name}_mean_signed_prediction"),
            class.mean_signed_prediction,
        );
    }
}

fn print_optional_value(name: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{name}={value:.9}"),
        None => println!("{name}=none"),
    }
}

fn parse_preset(value: &str) -> Result<NetworkPreset, BoxError> {
    match value {
        "micro" => Ok(NetworkPreset::Micro),
        "pure" => Ok(NetworkPreset::Pure),
        _ => Err(format!("invalid preset {value}").into()),
    }
}

fn parse_level(value: &str) -> Result<OptimizationLevel, BoxError> {
    match value {
        "0" => Ok(OptimizationLevel::Level0),
        "1" => Ok(OptimizationLevel::Level1),
        _ => Err(format!("invalid optimization level {value}").into()),
    }
}

fn parse_classes(value: &str) -> Result<Vec<ClassShape>, BoxError> {
    let classes = value
        .split(',')
        .map(|part| -> Result<ClassShape, BoxError> {
            let (capacity, batch_size) = part
                .split_once(':')
                .ok_or_else(|| format!("invalid class {part}; expected CAPACITY:BATCH"))?;
            Ok(ClassShape {
                capacity: positive_usize(capacity, "class capacity")?,
                batch_size: positive_usize(batch_size, "class batch")?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if classes.is_empty() {
        return Err("--classes must contain at least one class".into());
    }
    for pair in classes.windows(2) {
        if pair[0].capacity >= pair[1].capacity {
            return Err("class capacities must be strictly increasing".into());
        }
    }
    Ok(classes)
}

fn parse_profiles(value: &str) -> Result<Vec<SamplingProfileV1>, BoxError> {
    let profiles = value
        .split(',')
        .map(|part| -> Result<SamplingProfileV1, BoxError> {
            let (temperature, uniform_mix) = part
                .split_once(':')
                .ok_or_else(|| format!("invalid profile {part}; expected TEMPERATURE:MIX"))?;
            Ok(SamplingProfileV1::new(
                temperature.parse()?,
                uniform_mix.parse()?,
            )?)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if profiles.is_empty() {
        Err("--profiles must contain at least one profile".into())
    } else {
        Ok(profiles)
    }
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, BoxError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|source| format!("invalid {flag}: {source}"))?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
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
        "usage: paisho-policy-stats --snapshot PATH --replay-dir DIR \
         --behavior-producer SHA256 --checkpoint PATH [options]\n\
         --service PATH --preset pure|micro --level 0|1 --model-seed N\n\
         --classes 64:8,128:4,1024:4 --workers N --wait-us N\n\
         --sampler-seed N --start-index N --examples N \
         --profiles TEMPERATURE:MIX,... --ppo-clip EPSILON"
    );
}
