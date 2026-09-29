use std::env;
use std::path::PathBuf;
use std::time::Duration;

use paisho_mpsgraph_client::{default_service_path, NetworkPreset, OptimizationLevel};
use paisho_train::EvaluationOpponentV1;

use super::BoxError;

#[derive(Clone, Copy, Debug)]
pub struct ClassShape {
    pub capacity: usize,
    pub batch_size: usize,
}

pub struct Options {
    pub output_directory: PathBuf,
    pub service: PathBuf,
    pub candidate_checkpoint: PathBuf,
    pub opponent: EvaluationOpponentV1,
    pub preset: NetworkPreset,
    pub optimization: OptimizationLevel,
    pub classes: Vec<ClassShape>,
    pub workers: usize,
    pub pairs_per_batch: usize,
    pub maximum_attempted_pairs: u64,
    pub maximum_eligible_pairs: u64,
    pub first_pair_id: u64,
    pub decision_soft_limit: usize,
    pub maximum_batch_wait: Duration,
    pub model_seed: u64,
    pub neutral_start_horizon: Option<usize>,
    pub neutral_start_seed: u64,
    pub neutral_start_source_limit: usize,
    pub neutral_start_attempts: usize,
    pub sampling_temperature: Option<f32>,
    pub sampling_uniform_mix: f32,
    pub elo0: f64,
    pub elo1: f64,
    pub lower_elo: Option<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl Options {
    pub fn parse() -> Result<Self, BoxError> {
        let available_workers = std::thread::available_parallelism()?.get();
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut output_directory = None;
        let mut service = default_service_path(&repository);
        let mut candidate_checkpoint = None;
        let mut opponent = None;
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
        let mut pairs_per_batch = None;
        let mut maximum_attempted_pairs = None;
        let mut maximum_eligible_pairs = None;
        let mut first_pair_id = 0;
        let mut decision_soft_limit = 2_048;
        let mut maximum_batch_wait = Duration::from_millis(5);
        let mut model_seed = 17;
        let mut neutral_start_horizon = Some(64);
        let mut neutral_start_seed = 0x4556_414c_5541_5445;
        let mut neutral_start_source_limit = 16_384;
        let mut neutral_start_attempts = 16;
        let mut sampling_temperature = Some(1.0);
        let mut sampling_uniform_mix = 0.05;
        let mut elo0 = 0.0;
        let mut elo1 = 100.0;
        let mut lower_elo = None;
        let mut alpha = 0.05;
        let mut beta = 0.05;

        let arguments = env::args().skip(1).collect::<Vec<_>>();
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
                "--output-dir" => output_directory = Some(PathBuf::from(value)),
                "--service" => service = PathBuf::from(value),
                "--candidate-checkpoint" => candidate_checkpoint = Some(PathBuf::from(value)),
                "--opponent" => opponent = Some(EvaluationOpponentV1::parse(value)?),
                "--preset" => {
                    preset = match value.as_str() {
                        "micro" => NetworkPreset::Micro,
                        "pure" => NetworkPreset::Pure,
                        _ => return Err(format!("invalid preset {value}").into()),
                    }
                }
                "--level" => {
                    optimization = match value.as_str() {
                        "0" => OptimizationLevel::Level0,
                        "1" => OptimizationLevel::Level1,
                        _ => return Err(format!("invalid optimization level {value}").into()),
                    }
                }
                "--classes" => classes = parse_classes(value)?,
                "--workers" => workers = positive_usize(value, flag)?,
                "--pairs-per-batch" => pairs_per_batch = Some(positive_usize(value, flag)?),
                "--max-attempted-pairs" => {
                    maximum_attempted_pairs = Some(positive_u64(value, flag)?)
                }
                "--max-eligible-pairs" => maximum_eligible_pairs = Some(positive_u64(value, flag)?),
                "--first-pair-id" => first_pair_id = value.parse()?,
                "--decision-limit" => decision_soft_limit = positive_usize(value, flag)?,
                "--wait-us" => maximum_batch_wait = Duration::from_micros(value.parse()?),
                "--model-seed" => model_seed = value.parse()?,
                "--start-horizon" => {
                    let value = value.parse::<usize>()?;
                    neutral_start_horizon = (value != 0).then_some(value);
                }
                "--start-seed" => neutral_start_seed = value.parse()?,
                "--start-source-limit" => neutral_start_source_limit = positive_usize(value, flag)?,
                "--start-source-attempts" => neutral_start_attempts = positive_usize(value, flag)?,
                "--sampling-temperature" => {
                    let value = value.parse::<f32>()?;
                    sampling_temperature = (value != 0.0).then_some(value);
                }
                "--sampling-uniform-mix" => sampling_uniform_mix = value.parse()?,
                "--elo0" => elo0 = value.parse()?,
                "--elo1" => elo1 = value.parse()?,
                "--lower-elo" => lower_elo = Some(value.parse()?),
                "--alpha" => alpha = value.parse()?,
                "--beta" => beta = value.parse()?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        let maximum_eligible_pairs =
            maximum_eligible_pairs.ok_or("missing --max-eligible-pairs N")?;
        let maximum_attempted_pairs = maximum_attempted_pairs.unwrap_or(
            maximum_eligible_pairs
                .checked_mul(2)
                .ok_or("pair budget overflow")?,
        );
        classes.sort_by_key(|class| class.capacity);
        Ok(Self {
            output_directory: output_directory.ok_or("missing --output-dir DIR")?,
            service,
            candidate_checkpoint: candidate_checkpoint
                .ok_or("missing --candidate-checkpoint PATH")?,
            opponent: opponent.ok_or("missing --opponent random|site|mcts:N")?,
            preset,
            optimization,
            classes,
            workers,
            pairs_per_batch: pairs_per_batch.unwrap_or(workers),
            maximum_attempted_pairs,
            maximum_eligible_pairs,
            first_pair_id,
            decision_soft_limit,
            maximum_batch_wait,
            model_seed,
            neutral_start_horizon,
            neutral_start_seed,
            neutral_start_source_limit,
            neutral_start_attempts,
            sampling_temperature,
            sampling_uniform_mix,
            elo0,
            elo1,
            lower_elo,
            alpha,
            beta,
        })
    }
}

fn print_usage() {
    println!(
        "usage: cargo run --release -p paisho-train --bin paisho-evaluate -- \\
         --output-dir DIR --candidate-checkpoint PATH \\
         --opponent random|site|mcts:N --max-eligible-pairs N [options]\n\\
         --max-attempted-pairs N --pairs-per-batch N --workers N\n\\
         --preset pure|micro --level 0|1 --classes 64:8,128:4,1024:4\n\\
         --lower-elo F --elo0 F --elo1 F --alpha F --beta F --decision-limit N\n\\
         --start-horizon N (0 disables) --start-seed N\n\\
         --start-source-limit N --start-source-attempts N\n\\
         --sampling-temperature F (0 selects argmax) --sampling-uniform-mix F\n\\
         --first-pair-id N --model-seed N --wait-us N --service PATH"
    );
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

fn positive_u64(value: &str, flag: &str) -> Result<u64, BoxError> {
    let parsed = value
        .parse::<u64>()
        .map_err(|source| format!("invalid {flag}: {source}"))?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
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
        Err("--classes must contain at least one class".into())
    } else {
        Ok(classes)
    }
}
