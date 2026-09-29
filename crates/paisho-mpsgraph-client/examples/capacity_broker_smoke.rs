use std::collections::{BTreeMap, HashSet};
use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use paisho_core::{legal_actions, GameOutcome, Position, StandardSetup, BASIC_FLOWERS};
use paisho_mpsgraph_client::{
    default_service_path, CapacityClassConfiguration, CapacityInferenceBroker,
    CapacityInferenceBrokerClient, InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel,
    ServiceConfiguration,
};
use rayon::prelude::*;

struct Options {
    service: PathBuf,
    checkpoint: Option<PathBuf>,
    preset: NetworkPreset,
    optimization: OptimizationLevel,
    worker_count: Option<usize>,
    actor_count: Option<usize>,
    classes: Vec<ClassShape>,
    wide_lanes: usize,
    warmup_decisions: usize,
    measured_decisions: usize,
    maximum_batch_wait: Duration,
    maximum_in_flight_batches: usize,
    prepare_ahead: bool,
    seed: u64,
}

#[derive(Clone, Copy, Debug)]
struct ClassShape {
    capacity: usize,
    batch_size: usize,
}

impl Options {
    fn parse() -> Result<Self, Box<dyn Error + Send + Sync>> {
        let mut options = Self {
            service: default_service_path(&env::current_dir()?),
            checkpoint: None,
            preset: NetworkPreset::Pure,
            optimization: OptimizationLevel::Level1,
            worker_count: None,
            actor_count: None,
            classes: vec![
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
            ],
            wide_lanes: 1,
            warmup_decisions: 4,
            measured_decisions: 20,
            maximum_batch_wait: Duration::from_millis(5),
            maximum_in_flight_batches: 1,
            prepare_ahead: false,
            seed: 20_260_905,
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
                    };
                }
                "--level" => {
                    options.optimization = match value.as_str() {
                        "0" => OptimizationLevel::Level0,
                        "1" => OptimizationLevel::Level1,
                        _ => return Err(format!("invalid level {value}").into()),
                    };
                }
                "--workers" => options.worker_count = Some(positive_usize(value, flag)?),
                "--actors" => options.actor_count = Some(positive_usize(value, flag)?),
                "--classes" => options.classes = parse_classes(value)?,
                "--wide-lanes" => options.wide_lanes = positive_usize(value, flag)?,
                "--warmup-decisions" => {
                    options.warmup_decisions = nonnegative_usize(value, flag)?;
                }
                "--measured-decisions" => {
                    options.measured_decisions = positive_usize(value, flag)?;
                }
                "--wait-us" => {
                    options.maximum_batch_wait = Duration::from_micros(value.parse()?);
                }
                "--inflight" => {
                    options.maximum_in_flight_batches = positive_usize(value, flag)?;
                }
                "--prefetch" => options.prepare_ahead = value.parse()?,
                "--seed" => options.seed = value.parse()?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        Ok(options)
    }
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
    let parsed = nonnegative_usize(value, flag)?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
    }
}

fn nonnegative_usize(value: &str, flag: &str) -> Result<usize, Box<dyn Error + Send + Sync>> {
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag}: {error}").into())
}

fn parse_classes(value: &str) -> Result<Vec<ClassShape>, Box<dyn Error + Send + Sync>> {
    let classes = value
        .split(',')
        .map(|part| -> Result<ClassShape, Box<dyn Error + Send + Sync>> {
            let (capacity, batch_size) = part
                .split_once(':')
                .ok_or_else(|| format!("invalid class {part}; expected CAPACITY:BATCH"))?;
            Ok(ClassShape {
                capacity: positive_usize(capacity, "--classes capacity")?,
                batch_size: positive_usize(batch_size, "--classes batch")?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if classes.is_empty() {
        Err("--classes must contain at least one value".into())
    } else {
        Ok(classes)
    }
}

fn print_usage() {
    println!(
        "usage: cargo run -p paisho-mpsgraph-client --example capacity_broker_smoke \
         --release -- [options]\n\
         --service PATH --checkpoint PATH --preset micro|pure --level 0|1 \
         --workers N --actors N --classes 64:8,128:4,1024:4 --wide-lanes N \
         --warmup-decisions N --measured-decisions N --wait-us N --inflight N \
         --prefetch true|false --seed N"
    );
}

struct ActorState {
    actor: usize,
    games_started: usize,
    position: Position,
    random_state: u64,
}

impl ActorState {
    fn new(actor: usize, seed: u64) -> Self {
        Self {
            actor,
            games_started: 1,
            position: new_position(actor, 0),
            random_state: seed ^ (actor as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        }
    }

    fn begin_next_game(&mut self) {
        self.position = new_position(self.actor, self.games_started);
        self.games_started += 1;
    }

    fn sample_policy(&mut self, probabilities: &[f32]) -> Option<usize> {
        self.random_state = self.random_state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.random_state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^= value >> 31;
        let threshold = ((value >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))) as f32;
        let mut cumulative = 0.0;
        for (index, &probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if threshold < cumulative {
                return Some(index);
            }
        }
        probabilities.len().checked_sub(1)
    }
}

fn new_position(actor: usize, game: usize) -> Position {
    let flower = BASIC_FLOWERS[(actor + game) % BASIC_FLOWERS.len()];
    Position::from_standard_setup(StandardSetup::balanced(flower))
}

#[derive(Default)]
struct PhaseStatistics {
    decisions: usize,
    completed_games: usize,
    maximum_legal_actions: usize,
    routed_positions: BTreeMap<usize, usize>,
}

fn advance_actor(
    actor: &mut ActorState,
    client: &CapacityInferenceBrokerClient,
    decisions: usize,
) -> Result<PhaseStatistics, Box<dyn Error + Send + Sync>> {
    let mut statistics = PhaseStatistics::default();
    while statistics.decisions < decisions {
        if actor.position.outcome() != GameOutcome::Ongoing {
            statistics.completed_games += 1;
            actor.begin_next_game();
        }
        let actions = legal_actions(&actor.position);
        if actions.is_empty() {
            return Err("ongoing position has no legal action".into());
        }
        let capacity = client
            .capacity_for(actions.len())
            .ok_or_else(|| format!("no capacity class fits {} legal actions", actions.len()))?;
        let output = client.infer(&actor.position)?;
        if output.policy_probabilities().len() != actions.len() {
            return Err("inference policy length does not match legal actions".into());
        }
        let selected = actor
            .sample_policy(output.policy_probabilities())
            .ok_or("empty inference policy")?;
        actor.position.apply(actions[selected])?;
        if actor.position.outcome() != GameOutcome::Ongoing {
            statistics.completed_games += 1;
            actor.begin_next_game();
        }

        statistics.decisions += 1;
        statistics.maximum_legal_actions = statistics.maximum_legal_actions.max(actions.len());
        *statistics.routed_positions.entry(capacity).or_default() += 1;
    }
    Ok(statistics)
}

fn aggregate(statistics: Vec<PhaseStatistics>) -> PhaseStatistics {
    let mut total = PhaseStatistics::default();
    for statistics in statistics {
        total.decisions += statistics.decisions;
        total.completed_games += statistics.completed_games;
        total.maximum_legal_actions = total
            .maximum_legal_actions
            .max(statistics.maximum_legal_actions);
        for (capacity, count) in statistics.routed_positions {
            *total.routed_positions.entry(capacity).or_default() += count;
        }
    }
    total
}

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let options = Options::parse()?;
    let available_workers = std::thread::available_parallelism()?.get();
    let configured_workers = options.worker_count.unwrap_or(available_workers);
    let actor_count = options
        .actor_count
        .unwrap_or(configured_workers.saturating_mul(8));
    let wide_capacity = options
        .classes
        .iter()
        .map(|class| class.capacity)
        .max()
        .expect("validated classes are non-empty");
    let services = options
        .classes
        .iter()
        .map(|class| CapacityClassConfiguration {
            service: ServiceConfiguration {
                executable: options.service.clone(),
                checkpoint: options.checkpoint.clone(),
                preset: options.preset,
                batch_size: class.batch_size,
                legal_action_capacity: class.capacity,
                inference_slots: options.maximum_in_flight_batches,
                optimization: options.optimization,
                seed: options.seed,
            },
            lanes: if class.capacity == wide_capacity {
                options.wide_lanes
            } else {
                1
            },
        })
        .collect();
    let broker = CapacityInferenceBroker::launch_with_lanes(
        services,
        InferenceBrokerConfiguration {
            prepare_ahead: options.prepare_ahead,
            maximum_batch_wait: options.maximum_batch_wait,
            maximum_in_flight_batches: options.maximum_in_flight_batches,
        },
    )?;
    let client = broker.client()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(configured_workers)
        .build()?;
    let actor_seed = options.seed.wrapping_add(0xa076_1d64_78bd_642f);
    let mut actors = (0..actor_count)
        .map(|actor| ActorState::new(actor, actor_seed))
        .collect::<Vec<_>>();

    let warmup = pool.install(|| {
        actors
            .par_iter_mut()
            .map(|actor| advance_actor(actor, &client, options.warmup_decisions))
            .collect::<Result<Vec<_>, _>>()
    })?;
    let observed_workers = Arc::new(Mutex::new(HashSet::new()));
    let started = Instant::now();
    let measured = pool.install(|| {
        actors
            .par_iter_mut()
            .map(|actor| {
                observed_workers
                    .lock()
                    .unwrap()
                    .insert(rayon::current_thread_index().unwrap());
                advance_actor(actor, &client, options.measured_decisions)
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let elapsed = started.elapsed();
    let warmup = aggregate(warmup);
    let measured = aggregate(measured);
    let observed_workers = observed_workers.lock().unwrap().len();
    drop(client);
    let telemetry = broker.shutdown()?;

    println!(
        "broker=capacity-game-load-v1 preset={:?} level={:?} workers_available={} \
         workers_configured={} workers_observed={} actors={} classes={:?} wide_lanes={} inflight={} prefetch={} \
         warmup_decisions={} measured_decisions={} completed_games={} max_legal_actions={} \
         elapsed_s={:.3} positions_per_second={:.1}",
        options.preset,
        options.optimization,
        available_workers,
        configured_workers,
        observed_workers,
        actor_count,
        options
            .classes
            .iter()
            .map(|class| (class.capacity, class.batch_size))
            .collect::<Vec<_>>(),
        options.wide_lanes,
        options.maximum_in_flight_batches,
        options.prepare_ahead,
        warmup.decisions,
        measured.decisions,
        measured.completed_games,
        measured.maximum_legal_actions,
        elapsed.as_secs_f64(),
        measured.decisions as f64 / elapsed.as_secs_f64()
    );
    for class in &telemetry.classes {
        println!(
            "capacity={} lanes={} routed_measured={} batches={} full_batches={} requested={} \
             executed={} padded={} maximum_batch={}",
            class.legal_action_capacity,
            class.lanes.len(),
            measured
                .routed_positions
                .get(&class.legal_action_capacity)
                .copied()
                .unwrap_or(0),
            class.broker.batches,
            class.broker.full_batches,
            class.broker.requested_positions,
            class.broker.executed_positions,
            class.broker.padded_positions,
            class.broker.maximum_observed_batch
        );
        for (lane_index, lane) in class.lanes.iter().enumerate() {
            println!(
                "capacity={} lane={} batches={} full_batches={} requested={} executed={} \
                 padded={} maximum_batch={}",
                class.legal_action_capacity,
                lane_index,
                lane.batches,
                lane.full_batches,
                lane.requested_positions,
                lane.executed_positions,
                lane.padded_positions,
                lane.maximum_observed_batch
            );
        }
    }
    if observed_workers != configured_workers {
        return Err(format!(
            "only {observed_workers}/{configured_workers} CPU workers advanced actors"
        )
        .into());
    }
    if telemetry.requested_positions() != (warmup.decisions + measured.decisions) as u64 {
        return Err("capacity broker telemetry lost requested positions".into());
    }
    Ok(())
}
