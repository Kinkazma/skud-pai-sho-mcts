use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use paisho_core::{legal_actions, BasicFlower, GameOutcome, Position, StandardSetup};
use paisho_model::InferenceRequestV1;
use paisho_mpsgraph_client::{
    default_service_path, MpsGraphProcess, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};

struct Options {
    service: PathBuf,
    checkpoint: Option<PathBuf>,
    preset: NetworkPreset,
    optimization: OptimizationLevel,
    batch_size: usize,
    warmup: usize,
    iterations: usize,
    seed: u64,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error>> {
        let repository = env::current_dir()?;
        let mut options = Self {
            service: default_service_path(&repository),
            checkpoint: None,
            preset: NetworkPreset::Pure,
            optimization: OptimizationLevel::Level1,
            batch_size: 8,
            warmup: 2,
            iterations: 10,
            seed: 20_260_903,
        };
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
                "--service" => options.service = PathBuf::from(value),
                "--checkpoint" => options.checkpoint = Some(PathBuf::from(value)),
                "--preset" => {
                    options.preset = match value.as_str() {
                        "micro" => NetworkPreset::Micro,
                        "pure" => NetworkPreset::Pure,
                        _ => return Err(format!("invalid preset {value}").into()),
                    }
                }
                "--level" => {
                    options.optimization = match value.as_str() {
                        "0" => OptimizationLevel::Level0,
                        "1" => OptimizationLevel::Level1,
                        _ => return Err(format!("invalid level {value}").into()),
                    }
                }
                "--batch" => options.batch_size = positive_usize(value, flag)?,
                "--warmup" => options.warmup = value.parse()?,
                "--iterations" => options.iterations = positive_usize(value, flag)?,
                "--seed" => options.seed = value.parse()?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        Ok(options)
    }
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, Box<dyn Error>> {
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
    }
}

fn print_usage() {
    println!(
        "usage: cargo run -p paisho-mpsgraph-client --example bridge_smoke --release -- [options]\n\
         --service PATH --checkpoint PATH --preset micro|pure --level 0|1 --batch N \
         --warmup N --iterations N --seed N"
    );
}

fn representative_positions(count: usize) -> Vec<Position> {
    let setup = StandardSetup::balanced(BasicFlower::Red3);
    let mut position = Position::from_standard_setup(setup);
    let mut positions = Vec::with_capacity(count);
    let mut decision = 0usize;
    while positions.len() < count {
        let actions = legal_actions(&position);
        if position.outcome() != GameOutcome::Ongoing || actions.is_empty() {
            position = Position::from_standard_setup(setup);
            continue;
        }
        positions.push(position.clone());
        let selected = (decision.wrapping_mul(37).wrapping_add(11)) % actions.len();
        position.apply(actions[selected]).unwrap();
        decision += 1;
    }
    positions
}

fn percentile(durations: &[Duration], fraction: f64) -> Duration {
    let mut sorted = durations.to_vec();
    sorted.sort_unstable();
    let index = ((sorted.len() as f64 * fraction).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn resident_kib(process_id: u32) -> Option<u64> {
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &process_id.to_string()])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then_some(())
        .and_then(|()| String::from_utf8(output.stdout).ok())?
        .trim()
        .parse()
        .ok()
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse()?;
    let positions = representative_positions(options.batch_size);
    let maximum_legal_actions = positions
        .iter()
        .map(|position| legal_actions(position).len())
        .max()
        .unwrap();
    let legal_action_capacity = maximum_legal_actions.next_power_of_two();
    let configuration = ServiceConfiguration {
        executable: options.service,
        preset: options.preset,
        batch_size: options.batch_size,
        legal_action_capacity,
        inference_slots: 1,
        optimization: options.optimization,
        seed: options.seed,
        checkpoint: options.checkpoint,
    };
    let mut process = MpsGraphProcess::launch(configuration.clone())?;
    let total_iterations = options.warmup + options.iterations;
    let mut encoding_measurements = Vec::with_capacity(options.iterations);
    let mut service_measurements = Vec::with_capacity(options.iterations);
    let mut total_measurements = Vec::with_capacity(options.iterations);
    let mut resident_after_warmup = None;
    for iteration in 0..total_iterations {
        let total_start = Instant::now();
        let encoding_start = Instant::now();
        let request = InferenceRequestV1::from_positions(
            iteration as u64,
            &positions,
            legal_action_capacity,
        )?;
        let encoding_elapsed = encoding_start.elapsed();
        let service_start = Instant::now();
        let response = process.infer(&request)?;
        let service_elapsed = service_start.elapsed();
        let total_elapsed = total_start.elapsed();
        assert_eq!(response.outputs().len(), positions.len());
        if iteration + 1 == options.warmup {
            resident_after_warmup = process.process_id().and_then(resident_kib);
        }
        if iteration >= options.warmup {
            encoding_measurements.push(encoding_elapsed);
            service_measurements.push(service_elapsed);
            total_measurements.push(total_elapsed);
        }
    }
    let resident_after_measurement = process.process_id().and_then(resident_kib);
    let status = process.shutdown()?;
    if !status.success() {
        return Err(format!("MPSGraph service exited with {status}").into());
    }

    let encoding_median = percentile(&encoding_measurements, 0.5);
    let service_median = percentile(&service_measurements, 0.5);
    let total_median = percentile(&total_measurements, 0.5);
    let total_p95 = percentile(&total_measurements, 0.95);
    let positions_per_second = options.batch_size as f64 / total_median.as_secs_f64();
    println!(
        "bridge=rust-process-mpsgraph-v1 preset={:?} level={:?} batch={} actions={} \
         warmup={} iterations={} encode_median_ms={:.3} service_median_ms={:.3} \
         total_median_ms={:.3} total_p95_ms={:.3} positions_per_second={:.1}",
        options.preset,
        options.optimization,
        options.batch_size,
        legal_action_capacity,
        options.warmup,
        options.iterations,
        encoding_median.as_secs_f64() * 1_000.0,
        service_median.as_secs_f64() * 1_000.0,
        total_median.as_secs_f64() * 1_000.0,
        total_p95.as_secs_f64() * 1_000.0,
        positions_per_second
    );
    if let (Some(start), Some(end)) = (resident_after_warmup, resident_after_measurement) {
        println!(
            "service_rss_start_mib={:.1} service_rss_end_mib={:.1} service_rss_delta_mib={:.1}",
            start as f64 / 1_024.0,
            end as f64 / 1_024.0,
            (end as i64 - start as i64) as f64 / 1_024.0
        );
    }
    Ok(())
}
