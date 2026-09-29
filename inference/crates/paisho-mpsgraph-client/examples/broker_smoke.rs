use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use paisho_core::{legal_actions, BasicFlower, GameOutcome, Position, StandardSetup};
use paisho_mpsgraph_client::{
    default_service_path, InferenceBroker, InferenceBrokerConfiguration, NetworkPreset,
    OptimizationLevel, ServiceConfiguration,
};
use rayon::prelude::*;

struct Options {
    service: PathBuf,
    checkpoint: Option<PathBuf>,
    preset: NetworkPreset,
    optimization: OptimizationLevel,
    actor_workers: Option<usize>,
    batch_size: usize,
    warmup_jobs: Option<usize>,
    jobs: usize,
    maximum_batch_wait: Duration,
    maximum_in_flight_batches: usize,
    seed: u64,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut options = Self {
            service: default_service_path(&env::current_dir()?),
            checkpoint: None,
            preset: NetworkPreset::Pure,
            optimization: OptimizationLevel::Level1,
            actor_workers: None,
            batch_size: 8,
            warmup_jobs: None,
            jobs: 400,
            maximum_batch_wait: Duration::from_millis(5),
            maximum_in_flight_batches: 1,
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
                "--actor-workers" => {
                    options.actor_workers = Some(positive_usize(value, flag)?);
                }
                "--batch" => options.batch_size = positive_usize(value, flag)?,
                "--warmup-jobs" => options.warmup_jobs = Some(positive_usize(value, flag)?),
                "--jobs" => options.jobs = positive_usize(value, flag)?,
                "--wait-us" => {
                    options.maximum_batch_wait = Duration::from_micros(value.parse()?);
                }
                "--inflight" => {
                    options.maximum_in_flight_batches = positive_usize(value, flag)?;
                }
                "--seed" => options.seed = value.parse()?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        Ok(options)
    }
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let parsed = value.parse::<usize>()?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
    }
}

fn print_usage() {
    println!(
        "usage: cargo run -p paisho-mpsgraph-client --example broker_smoke --release -- [options]\n\
         --service PATH --checkpoint PATH --preset micro|pure --level 0|1 --actor-workers N --batch N \
         --warmup-jobs N --jobs N --wait-us N --inflight N --seed N"
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

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let options = Options::parse()?;
    let available_workers = std::thread::available_parallelism()?.get();
    let configured_workers = options.actor_workers.unwrap_or(available_workers);
    let positions = representative_positions(configured_workers.max(options.batch_size));
    let maximum_legal_actions = positions
        .iter()
        .map(|position| legal_actions(position).len())
        .max()
        .unwrap();
    let legal_action_capacity = maximum_legal_actions
        .checked_next_power_of_two()
        .ok_or("legal action capacity overflow")?;
    let service = ServiceConfiguration {
        executable: options.service,
        preset: options.preset,
        batch_size: options.batch_size,
        legal_action_capacity,
        inference_slots: options.maximum_in_flight_batches,
        optimization: options.optimization,
        seed: options.seed,
        checkpoint: options.checkpoint,
    };
    let broker = InferenceBroker::launch(
        service,
        InferenceBrokerConfiguration {
            prepare_ahead: false,
            maximum_batch_wait: options.maximum_batch_wait,
            maximum_in_flight_batches: options.maximum_in_flight_batches,
        },
    )?;
    let client = broker.client()?;
    let observed_workers = Arc::new(Mutex::new(HashSet::new()));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(configured_workers)
        .build()?;

    let warmup_jobs = options
        .warmup_jobs
        .unwrap_or(options.batch_size.max(configured_workers) * 2);
    pool.install(|| {
        (0..warmup_jobs)
            .into_par_iter()
            .try_for_each(|job| client.infer(&positions[job % positions.len()]).map(|_| ()))
    })?;

    let started = Instant::now();
    pool.install(|| {
        (0..options.jobs)
            .into_par_iter()
            .map(|job| {
                let worker = rayon::current_thread_index().unwrap();
                observed_workers.lock().unwrap().insert(worker);
                let output = client.infer(&positions[job % positions.len()])?;
                Ok::<usize, Box<dyn Error + Send + Sync>>(output.policy_probabilities().len())
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let elapsed = started.elapsed();
    drop(client);
    let telemetry = broker.shutdown()?;
    let observed_workers = observed_workers.lock().unwrap().len();
    let positions_per_second = options.jobs as f64 / elapsed.as_secs_f64();

    println!(
        "broker=rust-actors-mpsgraph-v1 preset={:?} level={:?} workers_available={} \
         workers_configured={} workers_observed={} batch={} actions={} warmup_jobs={} jobs={} wait_us={} \
         inflight={} maximum_observed_inflight={} batches={} full_batches={} padded_positions={} \
         elapsed_s={:.3} positions_per_second={:.1}",
        options.preset,
        options.optimization,
        available_workers,
        configured_workers,
        observed_workers,
        options.batch_size,
        legal_action_capacity,
        warmup_jobs,
        options.jobs,
        options.maximum_batch_wait.as_micros(),
        options.maximum_in_flight_batches,
        telemetry.maximum_observed_in_flight_batches,
        telemetry.batches,
        telemetry.full_batches,
        telemetry.padded_positions,
        elapsed.as_secs_f64(),
        positions_per_second
    );
    if observed_workers != configured_workers {
        return Err(format!(
            "only {observed_workers}/{configured_workers} actor workers submitted inference"
        )
        .into());
    }
    if telemetry.requested_positions != (warmup_jobs + options.jobs) as u64 {
        return Err("broker telemetry lost requested positions".into());
    }
    Ok(())
}
