use std::error::Error;
use std::path::PathBuf;

use paisho_model::TerminalPpoParametersV1;
use paisho_mpsgraph_client::{default_service_path, NetworkPreset, OptimizationLevel};
use paisho_replay::ReplayDigestV1;
use paisho_train::{run_learner, LearnerConfiguration, LearnerObjectiveConfiguration};
use paisho_train::{ScalarMetricSummaryV1, TerminalPpoMetricsSummaryV1};

const DEFAULT_LEARNING_RATE: f32 = 1.0e-4;
const DEFAULT_MODEL_SEED: u64 = 17;
const DEFAULT_SAMPLER_SEED: u64 = 0xC0FF_EE11;
const DEFAULT_POLICY_TEMPERATURE: f32 = 1.0;
const DEFAULT_UNIFORM_MIX: f32 = 0.05;
const DEFAULT_PPO_CLIP: f32 = 0.2;
const DEFAULT_PPO_VALUE_WEIGHT: f32 = 0.5;
const DEFAULT_PPO_ENTROPY_WEIGHT: f32 = 0.01;

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse(std::env::args().skip(1))?;
    let report = run_learner(&arguments.configuration)?;
    println!("resumed={}", report.resumed);
    println!("has_parent_checkpoint={}", report.has_parent_checkpoint);
    println!(
        "started_from_parent_checkpoint={}",
        report.started_from_parent_checkpoint
    );
    println!(
        "generation_start_training_step={}",
        report.generation_start_training_step
    );
    println!("initial_training_step={}", report.initial_training_step);
    println!("completed_training_step={}", report.completed_training_step);
    println!("completed_replay_index={}", report.completed_replay_index);
    println!("training_steps_this_run={}", report.training_steps_this_run);
    println!("examples_this_run={}", report.examples_this_run);
    println!("checkpoints_this_run={}", report.checkpoints_this_run);
    println!(
        "checkpoint_seconds={:.6}",
        report.checkpoint_elapsed.as_secs_f64()
    );
    println!(
        "replay_preload_seconds={:.6}",
        report.replay_preload_elapsed.as_secs_f64()
    );
    match report.recorded_behavior_values_complete {
        Some(value) => println!("recorded_behavior_values_complete={value}"),
        None => println!("recorded_behavior_values_complete=not-applicable"),
    }
    if let Some(metrics) = report.terminal_ppo_metrics_this_run {
        print_terminal_ppo_metrics(metrics);
    }
    println!("latest_checkpoint={}", report.latest_checkpoint.display());
    println!("elapsed_seconds={:.6}", report.elapsed.as_secs_f64());
    if report.elapsed.as_secs_f64() > 0.0 {
        println!(
            "examples_per_second={:.3}",
            report.examples_this_run as f64 / report.elapsed.as_secs_f64()
        );
    }
    Ok(())
}

fn print_terminal_ppo_metrics(metrics: TerminalPpoMetricsSummaryV1) {
    println!("ppo_metrics_batches={}", metrics.batches);
    println!(
        "ppo_metrics_first_training_step={}",
        metrics.first_training_step
    );
    println!(
        "ppo_metrics_last_training_step={}",
        metrics.last_training_step
    );
    print_metric("ppo_policy_loss", metrics.policy_loss);
    print_metric("ppo_value_loss", metrics.value_loss);
    print_metric("ppo_entropy", metrics.entropy);
    print_metric("ppo_total_loss", metrics.total_loss);
    print_metric("ppo_mean_advantage", metrics.mean_advantage);
    print_metric("ppo_mean_importance_ratio", metrics.mean_importance_ratio);
    print_metric(
        "ppo_mean_squared_ratio_deviation",
        metrics.mean_squared_ratio_deviation,
    );
}

fn print_metric(name: &str, metric: ScalarMetricSummaryV1) {
    println!("{name}_minimum={:.9}", metric.minimum);
    println!("{name}_mean={:.9}", metric.mean);
    println!("{name}_maximum={:.9}", metric.maximum);
    println!("{name}_last={:.9}", metric.last);
}

struct Arguments {
    configuration: LearnerConfiguration,
}

impl Arguments {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut service = default_service_path(&repository);
        let mut snapshot = None;
        let mut replay_directory = None;
        let mut run_directory = None;
        let mut initial_checkpoint = None;
        let mut network_preset = NetworkPreset::Pure;
        let mut optimization = OptimizationLevel::Level1;
        let mut batch_size = 64_usize;
        let mut legal_action_capacity = 1024_usize;
        let mut model_seed = DEFAULT_MODEL_SEED;
        let mut sampler_seed = DEFAULT_SAMPLER_SEED;
        let mut generation = 0_u64;
        let mut learning_rate = DEFAULT_LEARNING_RATE;
        let mut objective_name = "supervised".to_owned();
        let mut behavior_producer = None;
        let mut actor_checkpoint = None;
        let mut policy_temperature = DEFAULT_POLICY_TEMPERATURE;
        let mut uniform_mix = DEFAULT_UNIFORM_MIX;
        let mut ppo_clip = DEFAULT_PPO_CLIP;
        let mut ppo_value_weight = DEFAULT_PPO_VALUE_WEIGHT;
        let mut ppo_entropy_weight = DEFAULT_PPO_ENTROPY_WEIGHT;
        let mut target_training_step = None;
        let mut checkpoint_interval = 100_u64;
        let mut arguments = arguments.peekable();
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value after {flag}"))?;
            match flag.as_str() {
                "--service" => service = PathBuf::from(value),
                "--snapshot" => snapshot = Some(PathBuf::from(value)),
                "--replay-dir" => replay_directory = Some(PathBuf::from(value)),
                "--run-dir" => run_directory = Some(PathBuf::from(value)),
                "--initial-checkpoint" => initial_checkpoint = Some(PathBuf::from(value)),
                "--preset" => {
                    network_preset = match value.as_str() {
                        "micro" => NetworkPreset::Micro,
                        "pure" => NetworkPreset::Pure,
                        _ => {
                            return Err(format!("invalid preset {value}; use micro or pure").into())
                        }
                    }
                }
                "--level" => {
                    optimization = match value.as_str() {
                        "0" => OptimizationLevel::Level0,
                        "1" => OptimizationLevel::Level1,
                        _ => return Err(format!("invalid level {value}; use 0 or 1").into()),
                    }
                }
                "--batch" => batch_size = parse(&value, "batch size")?,
                "--actions" => legal_action_capacity = parse(&value, "action capacity")?,
                "--model-seed" => model_seed = parse(&value, "model seed")?,
                "--sampler-seed" => sampler_seed = parse(&value, "sampler seed")?,
                "--generation" => generation = parse(&value, "generation")?,
                "--learning-rate" => learning_rate = parse(&value, "learning rate")?,
                "--objective" => objective_name = value,
                "--behavior-producer" => behavior_producer = Some(value.parse::<ReplayDigestV1>()?),
                "--actor-checkpoint" => actor_checkpoint = Some(PathBuf::from(value)),
                "--policy-temperature" => policy_temperature = parse(&value, "policy temperature")?,
                "--uniform-mix" => uniform_mix = parse(&value, "uniform mix")?,
                "--ppo-clip" => ppo_clip = parse(&value, "PPO clip")?,
                "--ppo-value-weight" => ppo_value_weight = parse(&value, "PPO value weight")?,
                "--ppo-entropy-weight" => ppo_entropy_weight = parse(&value, "PPO entropy weight")?,
                "--target-step" => target_training_step = Some(parse(&value, "target step")?),
                "--checkpoint-every" => checkpoint_interval = parse(&value, "checkpoint interval")?,
                _ => return Err(format!("unknown argument {flag}").into()),
            }
        }
        let objective = match objective_name.as_str() {
            "supervised" => {
                if behavior_producer.is_some() || actor_checkpoint.is_some() {
                    return Err(
                        "--behavior-producer and --actor-checkpoint require --objective terminal-ppo"
                            .into(),
                    );
                }
                LearnerObjectiveConfiguration::SupervisedPolicyValue
            }
            "terminal-ppo" => LearnerObjectiveConfiguration::terminal_ppo(
                behavior_producer.ok_or("terminal PPO requires --behavior-producer SHA256")?,
                actor_checkpoint,
                TerminalPpoParametersV1::with_behavior(
                    policy_temperature,
                    uniform_mix,
                    ppo_clip,
                    ppo_value_weight,
                    ppo_entropy_weight,
                )?,
            ),
            value => {
                return Err(
                    format!("invalid objective {value}; use supervised or terminal-ppo").into(),
                )
            }
        };
        Ok(Self {
            configuration: LearnerConfiguration {
                service_executable: service,
                replay_snapshot: snapshot.ok_or("missing --snapshot PATH")?,
                replay_directory: replay_directory.ok_or("missing --replay-dir DIR")?,
                run_directory: run_directory.ok_or("missing --run-dir DIR")?,
                initial_checkpoint,
                network_preset,
                optimization,
                batch_size,
                legal_action_capacity,
                model_seed,
                sampler_seed,
                generation,
                learning_rate,
                objective,
                target_training_step: target_training_step.ok_or("missing --target-step N")?,
                checkpoint_interval,
            },
        })
    }
}

fn parse<T>(text: &str, name: &'static str) -> Result<T, Box<dyn Error>>
where
    T: core::str::FromStr,
    T::Err: Error + 'static,
{
    text.parse()
        .map_err(|source| format!("invalid {name}: {source}").into())
}
