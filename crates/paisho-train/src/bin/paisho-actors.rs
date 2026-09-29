use std::env;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use paisho_ai::{
    AgentTelemetry, HeuristicWeights, MctsAgent, MctsConfig, NetworkPolicy, PureNetworkAgent,
    RandomAgent, SiteBotV1, StableRng, EXHAUSTIVE_ACTION_RANKING, SITE_BOT_V1_SOURCE_COMMIT,
};
use paisho_core::{Action, Player, Position, StandardSetup, BASIC_FLOWERS};
use paisho_mpsgraph_client::{
    default_service_path, verify_checkpoint_file, CapacityClassConfiguration,
    CapacityInferenceBroker, CapacityInferenceBrokerClient, CapacityInferenceTelemetry,
    InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_replay::{
    ReplayDigestV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1, ReplayValidationError,
};
use paisho_train::{
    play_replay_parallel, MctsReplayAgent, NeutralStartConfigurationV1, RecordedNetworkAgent,
    ReplayActorAgent, ReplayActorChoice, ReplayMatchConfiguration, ReplayMatchError,
    ReplayMatchResult, ReplayMatchTask, UnrecordedReplayAgent, NEUTRAL_START_POLICY_V1,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

#[path = "paisho_actors/results.rs"]
mod results;

use results::CandidateResults;

type BoxError = Box<dyn Error + Send + Sync>;
const BUILD_SOURCE_REVISION: &str = env!("PAISHO_BUILD_GIT_REVISION");
const BUILD_SOURCE_DIRTY: &str = env!("PAISHO_BUILD_GIT_DIRTY");
const BUILD_SOURCE_SHA256: &str = env!("PAISHO_BUILD_SOURCE_SHA256");

#[derive(Clone, Copy, Debug)]
struct ClassShape {
    capacity: usize,
    batch_size: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Opponent {
    SelfPlay,
    Random,
    Site,
    Mcts { simulations: usize },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CandidateSelection {
    Sample,
    Argmax,
}

impl CandidateSelection {
    const fn policy(self, temperature: f32, uniform_mix: f32) -> NetworkPolicy {
        match self {
            Self::Sample => NetworkPolicy::Sample {
                temperature,
                uniform_mix,
            },
            Self::Argmax => NetworkPolicy::Argmax,
        }
    }
}

impl Opponent {
    const fn requires_paired_seats(self) -> bool {
        !matches!(self, Self::SelfPlay)
    }
}

struct Options {
    output_directory: PathBuf,
    service: PathBuf,
    checkpoint: Option<PathBuf>,
    preset: NetworkPreset,
    optimization: OptimizationLevel,
    classes: Vec<ClassShape>,
    wide_lanes: usize,
    workers: usize,
    actors: usize,
    target_games: usize,
    maximum_attempts: usize,
    shard_index: u64,
    first_game_id: u64,
    decision_soft_limit: usize,
    maximum_batch_wait: Duration,
    model_seed: u64,
    actor_seed: u64,
    selection: CandidateSelection,
    temperature: f32,
    uniform_mix: f32,
    neutral_start: Option<NeutralStartConfigurationV1>,
    opponent: Opponent,
}

impl Options {
    fn parse() -> Result<Self, BoxError> {
        let available_workers = std::thread::available_parallelism()?.get();
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut output_directory = None;
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
        let mut wide_lanes = 1;
        let mut workers = available_workers;
        let mut actors = None;
        let mut target_games = None;
        let mut maximum_attempts = None;
        let mut shard_index = 0;
        let mut first_game_id = 0;
        let mut decision_soft_limit = 2_048;
        let mut maximum_batch_wait = Duration::from_millis(5);
        let mut model_seed = 17;
        let mut actor_seed = 0x5041_4953_484f_2026;
        let mut selection = CandidateSelection::Sample;
        let mut temperature = 1.0;
        let mut uniform_mix = 0.05;
        let mut start_horizon = None;
        let mut start_seed = 0x4e45_5554_5241_4c31;
        let mut start_source_limit = 16_384;
        let mut start_source_attempts = 16;
        let mut opponent = Opponent::Random;

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
                "--checkpoint" => checkpoint = Some(PathBuf::from(value)),
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
                "--wide-lanes" => wide_lanes = positive_usize(value, flag)?,
                "--workers" => workers = positive_usize(value, flag)?,
                "--actors" => actors = Some(positive_usize(value, flag)?),
                "--target-games" => target_games = Some(positive_usize(value, flag)?),
                "--max-attempts" => maximum_attempts = Some(positive_usize(value, flag)?),
                "--shard-index" => shard_index = value.parse()?,
                "--first-game-id" => first_game_id = value.parse()?,
                "--decision-limit" => decision_soft_limit = positive_usize(value, flag)?,
                "--wait-us" => maximum_batch_wait = Duration::from_micros(value.parse()?),
                "--model-seed" => model_seed = value.parse()?,
                "--actor-seed" => actor_seed = value.parse()?,
                "--selection" => {
                    selection = match value.as_str() {
                        "sample" => CandidateSelection::Sample,
                        "argmax" => CandidateSelection::Argmax,
                        _ => return Err(format!("invalid selection {value}").into()),
                    }
                }
                "--temperature" => temperature = value.parse()?,
                "--uniform-mix" => uniform_mix = value.parse()?,
                "--start-horizon" => start_horizon = Some(positive_usize(value, flag)?),
                "--start-seed" => start_seed = value.parse()?,
                "--start-source-limit" => start_source_limit = positive_usize(value, flag)?,
                "--start-source-attempts" => start_source_attempts = positive_usize(value, flag)?,
                "--opponent" => opponent = parse_opponent(value)?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }

        let target_games = target_games.ok_or("missing --target-games N")?;
        let actors = actors.unwrap_or_else(|| workers.saturating_mul(8));
        let maximum_attempts =
            maximum_attempts.unwrap_or_else(|| target_games.saturating_mul(4).max(actors));
        if maximum_attempts < target_games {
            return Err("--max-attempts cannot be smaller than --target-games".into());
        }
        if opponent.requires_paired_seats() {
            if target_games % 2 != 0 {
                return Err("--target-games must be even against a distinct opponent".into());
            }
            if actors < 2 || actors % 2 != 0 {
                return Err(
                    "--actors must be even and at least 2 against a distinct opponent".into(),
                );
            }
        }
        selection.policy(temperature, uniform_mix).validate()?;
        let neutral_start = start_horizon
            .map(|horizon| {
                NeutralStartConfigurationV1::new(
                    start_seed,
                    horizon,
                    start_source_limit,
                    start_source_attempts,
                )
            })
            .transpose()?;
        Ok(Self {
            output_directory: output_directory.ok_or("missing --output-dir DIR")?,
            service,
            checkpoint,
            preset,
            optimization,
            classes,
            wide_lanes,
            workers,
            actors,
            target_games,
            maximum_attempts,
            shard_index,
            first_game_id,
            decision_soft_limit,
            maximum_batch_wait,
            model_seed,
            actor_seed,
            selection,
            temperature,
            uniform_mix,
            neutral_start,
            opponent,
        })
    }
}

fn print_usage() {
    println!(
        "usage: cargo run --release -p paisho-train --bin paisho-actors -- \
         --output-dir DIR --target-games N [options]\n\
         --checkpoint PATH --preset pure|micro --level 0|1\n\
         --opponent self|random|site|mcts:N --workers N --actors N\n\
         --classes 64:8,128:4,1024:4 --wide-lanes N \
         --decision-limit N --max-attempts N\n\
         --shard-index N --first-game-id N --selection sample|argmax\n\
         --temperature F --uniform-mix F\n\
         --start-horizon N --start-seed N --start-source-limit N\n\
         --start-source-attempts N\n\
         --model-seed N --actor-seed N --wait-us N --service PATH"
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

fn parse_opponent(value: &str) -> Result<Opponent, BoxError> {
    match value {
        "self" => Ok(Opponent::SelfPlay),
        "random" => Ok(Opponent::Random),
        "site" => Ok(Opponent::Site),
        _ => {
            let simulations = value
                .strip_prefix("mcts:")
                .ok_or_else(|| format!("invalid opponent {value}"))?;
            Ok(Opponent::Mcts {
                simulations: positive_usize(simulations, "MCTS simulations")?,
            })
        }
    }
}

enum CampaignAgent {
    Network(RecordedNetworkAgent<CapacityInferenceBrokerClient>),
    Random(UnrecordedReplayAgent<RandomAgent>),
    Site(UnrecordedReplayAgent<SiteBotV1>),
    Mcts(Box<MctsReplayAgent>),
}

impl ReplayActorAgent for CampaignAgent {
    fn digest(&self) -> ReplayDigestV1 {
        match self {
            Self::Network(agent) => agent.digest(),
            Self::Random(agent) => agent.digest(),
            Self::Site(agent) => agent.digest(),
            Self::Mcts(agent) => agent.digest(),
        }
    }

    fn select_for_replay(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<ReplayActorChoice, paisho_ai::AgentError> {
        match self {
            Self::Network(agent) => agent.select_for_replay(position, legal_actions),
            Self::Random(agent) => agent.select_for_replay(position, legal_actions),
            Self::Site(agent) => agent.select_for_replay(position, legal_actions),
            Self::Mcts(agent) => agent.select_for_replay(position, legal_actions),
        }
    }

    fn telemetry(&self) -> AgentTelemetry {
        match self {
            Self::Network(agent) => agent.telemetry(),
            Self::Random(agent) => agent.telemetry(),
            Self::Site(agent) => agent.telemetry(),
            Self::Mcts(agent) => agent.telemetry(),
        }
    }

    fn reset_telemetry(&mut self) {
        match self {
            Self::Network(agent) => agent.reset_telemetry(),
            Self::Random(agent) => agent.reset_telemetry(),
            Self::Site(agent) => agent.reset_telemetry(),
            Self::Mcts(agent) => agent.reset_telemetry(),
        }
    }
}

struct AgentFactory {
    client: CapacityInferenceBrokerClient,
    candidate_digest: ReplayDigestV1,
    opponent_digest: ReplayDigestV1,
    opponent: Opponent,
    first_game_id: u64,
    actor_seed: u64,
    policy: NetworkPolicy,
}

impl AgentFactory {
    fn make(&self, task: &ReplayMatchTask, player: Player) -> CampaignAgent {
        let candidate_is_host = (task.game_id - self.first_game_id) % 2 == 0;
        let is_candidate = matches!(self.opponent, Opponent::SelfPlay)
            || candidate_is_host == (player == Player::Host);
        if is_candidate {
            return self.network(task.game_id, player);
        }
        let seed = agent_seed(self.actor_seed, task.game_id, player, 1);
        match self.opponent {
            Opponent::SelfPlay => unreachable!("self-play makes both seats candidates"),
            Opponent::Random => CampaignAgent::Random(UnrecordedReplayAgent::new(
                self.opponent_digest,
                RandomAgent::new(seed),
            )),
            Opponent::Site => CampaignAgent::Site(UnrecordedReplayAgent::new(
                self.opponent_digest,
                SiteBotV1::new(seed),
            )),
            Opponent::Mcts { simulations } => CampaignAgent::Mcts(Box::new(MctsReplayAgent::new(
                self.opponent_digest,
                MctsAgent::new(seed, mcts_configuration(simulations))
                    .expect("campaign MCTS configuration is valid"),
            ))),
        }
    }

    fn network(&self, game_id: u64, player: Player) -> CampaignAgent {
        let seed = agent_seed(self.actor_seed, game_id, player, 0);
        let network = PureNetworkAgent::new(self.client.clone(), seed, self.policy)
            .expect("validated campaign policy remains valid");
        CampaignAgent::Network(RecordedNetworkAgent::new(self.candidate_digest, network))
    }
}

fn agent_seed(base: u64, game_id: u64, player: Player, family: u64) -> u64 {
    let side = match player {
        Player::Host => 0x484f_5354,
        Player::Guest => 0x0047_5545_5354,
    };
    let mut rng = StableRng::new(
        base ^ game_id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ side ^ family.rotate_left(29),
    );
    rng.next_u64()
}

fn mcts_configuration(simulations: usize) -> MctsConfig {
    MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 96,
        action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    }
}

#[derive(Default)]
struct CampaignRun {
    retained: Vec<ReplayMatchResult>,
    attempts: usize,
    completed: usize,
    interrupted: usize,
    no_training_decision: usize,
    failed: usize,
    excluded_pairs: usize,
    maximum_workers_observed: usize,
    abort_reason: Option<String>,
}

impl CampaignRun {
    fn abort(&mut self, reason: impl Into<String>) {
        if self.abort_reason.is_none() {
            self.abort_reason = Some(reason.into());
        }
    }

    fn target_reached(&self, target_games: usize) -> bool {
        self.retained.len() == target_games
    }
}

struct PublicationReservation {
    attempt: u64,
    partial_directory: PathBuf,
    final_directory: PathBuf,
    published: bool,
}

impl PublicationReservation {
    fn publish(mut self) -> io::Result<PathBuf> {
        sync_tree(&self.partial_directory)?;
        fs::rename(&self.partial_directory, &self.final_directory)?;
        self.published = true;
        sync_directory(
            self.final_directory
                .parent()
                .unwrap_or_else(|| Path::new(".")),
        )?;
        Ok(self.final_directory.clone())
    }
}

impl Drop for PublicationReservation {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.partial_directory);
        }
    }
}

struct PublishedActorRun {
    directory: PathBuf,
    snapshot_path: PathBuf,
    shard_digest: ReplayDigestV1,
    snapshot_digest: ReplayDigestV1,
    game_count: usize,
    example_count: u64,
    candidate_results: Option<CandidateResults>,
}

fn main() -> Result<(), BoxError> {
    let options = Options::parse()?;
    fs::create_dir_all(&options.output_directory)?;
    let checkpoint_digest = options
        .checkpoint
        .as_deref()
        .map(verify_checkpoint_file)
        .transpose()?;
    let service_digest = sha256_file(&options.service)?;
    let actor_executable_digest = sha256_file(&env::current_exe()?)?;
    let reservation =
        reserve_publication_directory(&options.output_directory, options.shard_index)?;
    let candidate_descriptor = network_descriptor(
        &options,
        checkpoint_digest,
        service_digest,
        actor_executable_digest,
    );
    let candidate_digest = digest_descriptor(&candidate_descriptor);
    let opponent_descriptor = opponent_descriptor(
        options.opponent,
        &candidate_descriptor,
        actor_executable_digest,
    );
    let opponent_digest = digest_descriptor(&opponent_descriptor);
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
                preset: options.preset,
                batch_size: class.batch_size,
                legal_action_capacity: class.capacity,
                inference_slots: 1,
                optimization: options.optimization,
                seed: options.model_seed,
                checkpoint: options.checkpoint.clone(),
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
            prepare_ahead: false,
            maximum_batch_wait: options.maximum_batch_wait,
            maximum_in_flight_batches: 1,
        },
    )?;
    let factory = AgentFactory {
        client: broker.client()?,
        candidate_digest,
        opponent_digest,
        opponent: options.opponent,
        first_game_id: options.first_game_id,
        actor_seed: options.actor_seed,
        policy: options
            .selection
            .policy(options.temperature, options.uniform_mix),
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let started = Instant::now();
    let mut run = run_campaign(&options, &pool, &factory);

    drop(factory);
    let broker_telemetry = match broker.shutdown() {
        Ok(telemetry) => telemetry,
        Err(source) => {
            run.abort(format!("MPSGraph broker shutdown failed: {source}"));
            CapacityInferenceTelemetry::default()
        }
    };
    if !run.target_reached(options.target_games) && run.abort_reason.is_none() {
        run.abort(format!(
            "only {} paired terminal games were retained after {} attempts",
            run.retained.len(),
            run.attempts
        ));
    }
    if run.retained.is_empty() {
        return Err(run
            .abort_reason
            .clone()
            .unwrap_or_else(|| "actor campaign produced no training game".to_owned())
            .into());
    }

    let elapsed = started.elapsed();
    let published = publish_actor_run(
        reservation,
        &options,
        &candidate_descriptor,
        candidate_digest,
        &opponent_descriptor,
        opponent_digest,
        &run,
        elapsed,
        &broker_telemetry,
    )?;
    println!("run_directory={}", published.directory.display());
    println!("snapshot={}", published.snapshot_path.display());
    println!("snapshot_sha256={}", published.snapshot_digest);
    println!("shard_sha256={}", published.shard_digest);
    println!("candidate={candidate_digest}");
    println!("opponent={opponent_digest}");
    println!("terminal_games={}", published.game_count);
    println!("training_examples={}", published.example_count);
    println!("attempts={}", run.attempts);
    println!("interrupted={}", run.interrupted);
    println!("no_training_decision={}", run.no_training_decision);
    println!("excluded_pairs={}", run.excluded_pairs);
    if let Some(results) = published.candidate_results {
        println!("candidate_wins={}", results.wins);
        println!("candidate_draws={}", results.draws);
        println!("candidate_losses={}", results.losses);
        println!("candidate_half_points={}", results.half_points());
        println!(
            "candidate_pentanomial={},{},{},{},{}",
            results.pentanomial[0],
            results.pentanomial[1],
            results.pentanomial[2],
            results.pentanomial[3],
            results.pentanomial[4]
        );
        let comparison = results.paired_comparison();
        println!("candidate_favorable_pairs={}", comparison.favorable());
        println!("candidate_tied_pairs={}", comparison.tied());
        println!("candidate_unfavorable_pairs={}", comparison.unfavorable());
        println!(
            "candidate_paired_sign_test_p={:.12}",
            comparison.exact_two_sided_sign_test_p_value()
        );
    }
    println!(
        "workers_observed={}/{}",
        run.maximum_workers_observed, options.workers
    );
    println!(
        "inference_positions={} elapsed_seconds={:.3}",
        broker_telemetry.requested_positions(),
        elapsed.as_secs_f64()
    );

    if let Some(reason) = run.abort_reason {
        return Err(format!(
            "{reason}; valid completed games were published at {}",
            published.directory.display()
        )
        .into());
    }
    Ok(())
}

fn run_campaign(
    options: &Options,
    pool: &rayon::ThreadPool,
    factory: &AgentFactory,
) -> CampaignRun {
    let mut run = CampaignRun {
        retained: Vec::with_capacity(options.target_games),
        ..CampaignRun::default()
    };
    while run.retained.len() < options.target_games
        && run.attempts < options.maximum_attempts
        && run.abort_reason.is_none()
    {
        let round_size = actor_round_size(options, &run);
        if round_size == 0 {
            break;
        }
        let tasks = match pool.install(|| build_tasks(options, run.attempts, round_size)) {
            Ok(tasks) => tasks,
            Err(source) => {
                run.abort(source.to_string());
                break;
            }
        };
        let batch = pool.install(|| {
            play_replay_parallel(
                &tasks,
                ReplayMatchConfiguration {
                    decision_soft_limit: options.decision_soft_limit,
                },
                |task| factory.make(task, Player::Host),
                |task| factory.make(task, Player::Guest),
            )
        });
        run.maximum_workers_observed = run.maximum_workers_observed.max(batch.workers);
        run.attempts += round_size;
        absorb_matches(&mut run, batch.matches, options.opponent);
        if round_size >= options.workers.saturating_mul(2) && batch.workers != options.workers {
            run.abort(format!(
                "actor round used only {}/{} configured CPU workers",
                batch.workers, options.workers
            ));
        }
    }
    run
}

fn actor_round_size(options: &Options, run: &CampaignRun) -> usize {
    let mut size = options
        .actors
        .min(options.maximum_attempts.saturating_sub(run.attempts))
        .min(options.target_games.saturating_sub(run.retained.len()));
    if options.opponent.requires_paired_seats() {
        size -= size % 2;
    }
    size
}

fn build_tasks(
    options: &Options,
    first_attempt: usize,
    count: usize,
) -> Result<Vec<ReplayMatchTask>, BoxError> {
    if options.opponent.requires_paired_seats() && (first_attempt % 2 != 0 || count % 2 != 0) {
        return Err("paired actor rounds must begin and end on pair boundaries".into());
    }
    let divisor = if options.opponent.requires_paired_seats() {
        2
    } else {
        1
    };
    let first_slot = first_attempt / divisor;
    let slot_count = count / divisor;
    let starts = options
        .neutral_start
        .map(|configuration| {
            (0..slot_count)
                .into_par_iter()
                .map(|offset| -> Result<_, BoxError> {
                    let slot = first_slot
                        .checked_add(offset)
                        .ok_or("neutral-start slot overflow")?;
                    let first_slot_attempt = slot
                        .checked_mul(divisor)
                        .ok_or("neutral-start attempt overflow")?;
                    let ordinal = options
                        .first_game_id
                        .checked_add(
                            u64::try_from(first_slot_attempt)
                                .map_err(|_| "neutral-start attempt does not fit u64")?,
                        )
                        .ok_or("neutral-start global game id overflow")?;
                    let setup = StandardSetup::balanced(BASIC_FLOWERS[slot % BASIC_FLOWERS.len()]);
                    configuration.generate(setup, ordinal).map_err(Into::into)
                })
                .collect::<Result<Vec<_>, BoxError>>()
        })
        .transpose()?;

    (0..count)
        .map(|offset| {
            let attempt = first_attempt
                .checked_add(offset)
                .ok_or("actor attempt counter overflow")?;
            let attempt_u64 =
                u64::try_from(attempt).map_err(|_| "actor attempt does not fit u64")?;
            let game_id = options
                .first_game_id
                .checked_add(attempt_u64)
                .ok_or("game id overflow")?;
            let slot = attempt / divisor;
            match &starts {
                Some(starts) => {
                    let start = starts
                        .get(slot - first_slot)
                        .ok_or("neutral-start schedule lost a generated slot")?;
                    Ok(ReplayMatchTask::from_neutral(game_id, start))
                }
                None => Ok(ReplayMatchTask::standard(
                    game_id,
                    StandardSetup::balanced(BASIC_FLOWERS[slot % BASIC_FLOWERS.len()]),
                )),
            }
        })
        .collect()
}

fn absorb_matches(
    run: &mut CampaignRun,
    matches: Vec<Result<ReplayMatchResult, ReplayMatchError>>,
    opponent: Opponent,
) {
    if !opponent.requires_paired_seats() {
        for result in matches {
            if let Some(game) = classify_match(run, result) {
                run.retained.push(game);
            }
        }
        return;
    }

    let mut matches = matches.into_iter();
    while let Some(first) = matches.next() {
        let second = matches
            .next()
            .expect("paired actor batches always contain an even number of games");
        let mut completed_pair = Vec::with_capacity(2);
        if let Some(game) = classify_match(run, first) {
            completed_pair.push(game);
        }
        if let Some(game) = classify_match(run, second) {
            completed_pair.push(game);
        }
        if completed_pair.len() == 2 {
            run.retained.extend(completed_pair);
        } else {
            run.excluded_pairs += 1;
        }
    }
}

fn classify_match(
    run: &mut CampaignRun,
    result: Result<ReplayMatchResult, ReplayMatchError>,
) -> Option<ReplayMatchResult> {
    match result {
        Ok(result) => {
            run.completed += 1;
            Some(result)
        }
        Err(ReplayMatchError::DecisionLimit { .. }) => {
            run.interrupted += 1;
            None
        }
        Err(ReplayMatchError::Replay(ReplayValidationError::NoTrainingDecision)) => {
            run.no_training_decision += 1;
            None
        }
        Err(source) => {
            run.failed += 1;
            run.abort(format!("actor match failed: {source}"));
            None
        }
    }
}

fn network_descriptor(
    options: &Options,
    checkpoint_digest: Option<[u8; 32]>,
    service_digest: [u8; 32],
    actor_executable_digest: [u8; 32],
) -> String {
    let model = checkpoint_digest.map_or_else(
        || format!("initial-seed:{}", options.model_seed),
        |digest| format!("checkpoint-sha256:{}", hex_digest(digest)),
    );
    let selection = match options.selection {
        CandidateSelection::Sample => format!(
            "selection=sample;temperature={};uniform-mix={}",
            options.temperature, options.uniform_mix
        ),
        CandidateSelection::Argmax => "selection=argmax".to_owned(),
    };
    format!(
        "family=pure-network;rules=skud-pai-sho-2022-03-14;encoding=v1;backend=mpsgraph;\
         preset={:?};optimization={:?};model={model};service-sha256={};{selection};\
         seed-policy=game-seat-splitmix64-v1;\
         source-sha256={BUILD_SOURCE_SHA256};actor-binary-sha256={};\
         implementation-revision={BUILD_SOURCE_REVISION};implementation-dirty={BUILD_SOURCE_DIRTY}",
        options.preset,
        options.optimization,
        hex_digest(service_digest),
        hex_digest(actor_executable_digest),
    )
}

fn opponent_descriptor(
    opponent: Opponent,
    candidate: &str,
    actor_executable_digest: [u8; 32],
) -> String {
    let implementation = format!(
        "source-sha256={BUILD_SOURCE_SHA256};actor-binary-sha256={};implementation-revision={BUILD_SOURCE_REVISION};implementation-dirty={BUILD_SOURCE_DIRTY}",
        hex_digest(actor_executable_digest)
    );
    match opponent {
        Opponent::SelfPlay => candidate.to_owned(),
        Opponent::Random => format!(
            "family=random;rules=skud-pai-sho-2022-03-14;seed-policy=game-seat-splitmix64-v1;{implementation}"
        ),
        Opponent::Site => format!(
            "family=site-bot-v1;rules=skud-pai-sho-2022-03-14;upstream={SITE_BOT_V1_SOURCE_COMMIT};seed-policy=game-seat-splitmix64-v1;{implementation}"
        ),
        Opponent::Mcts { simulations } => format!(
            "family=heuristic-mcts;rules=skud-pai-sho-2022-03-14;simulations={simulations};trees=1;depth=96;ranking=exhaustive;widening=1/1;rollout=0;exploration=sqrt2;seed-policy=game-seat-splitmix64-v1;{implementation}"
        ),
    }
}

fn digest_descriptor(descriptor: &str) -> ReplayDigestV1 {
    digest_bytes(descriptor.as_bytes())
}

fn reserve_publication_directory(
    parent: &Path,
    shard_index: u64,
) -> Result<PublicationReservation, BoxError> {
    for attempt in 0..=u64::MAX {
        let name = format!("actor-run-{shard_index:020}-a{attempt:020}");
        let final_directory = parent.join(&name);
        if final_directory.exists() {
            continue;
        }
        let partial_directory = parent.join(format!(".{name}.partial"));
        match fs::create_dir(&partial_directory) {
            Ok(()) => {
                if final_directory.exists() {
                    fs::remove_dir(&partial_directory)?;
                    continue;
                }
                return Ok(PublicationReservation {
                    attempt,
                    partial_directory,
                    final_directory,
                    published: false,
                });
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(source.into()),
        }
    }
    Err("actor publication attempt overflow".into())
}

#[allow(clippy::too_many_arguments)]
fn publish_actor_run(
    reservation: PublicationReservation,
    options: &Options,
    candidate_descriptor: &str,
    candidate_digest: ReplayDigestV1,
    opponent_descriptor: &str,
    opponent_digest: ReplayDigestV1,
    run: &CampaignRun,
    elapsed: Duration,
    broker_telemetry: &CapacityInferenceTelemetry,
) -> Result<PublishedActorRun, BoxError> {
    let mut retained = run.retained.clone();
    retained.sort_by_key(|result| result.game.game_id());
    validate_retained_games(&retained, options, candidate_digest, opponent_digest)?;
    let candidate_results = options
        .opponent
        .requires_paired_seats()
        .then(|| CandidateResults::from_paired_matches(&retained, candidate_digest))
        .transpose()
        .map_err(|message| -> BoxError { message.into() })?;
    let games = retained.iter().map(|result| result.game.clone()).collect();
    let shard = ReplayShardV1::new(options.shard_index, games)?;
    let shard_name = "shard.psrbuf";
    let snapshot_name = "snapshot.psrsnap";
    let telemetry_name = "telemetry.tsv";
    let starts_name = "starts.tsv";
    let metadata_name = "actors.txt";
    let shard_digest = shard.write_new(&reservation.partial_directory.join(shard_name))?;
    let snapshot = ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
        shard_name, &shard,
    )?])?;
    let snapshot_digest = snapshot.write_new(&reservation.partial_directory.join(snapshot_name))?;
    let verification = snapshot.verify_directory(&reservation.partial_directory)?;
    let telemetry_text = match_telemetry_text(&retained);
    let telemetry_digest = digest_bytes(telemetry_text.as_bytes());
    write_new_synced(
        &reservation.partial_directory.join(telemetry_name),
        telemetry_text.as_bytes(),
    )?;
    let starts_text = start_telemetry_text(&retained);
    let starts_digest = digest_bytes(starts_text.as_bytes());
    write_new_synced(
        &reservation.partial_directory.join(starts_name),
        starts_text.as_bytes(),
    )?;
    let metadata = actor_metadata(
        options,
        reservation.attempt,
        candidate_descriptor,
        candidate_digest,
        opponent_descriptor,
        opponent_digest,
        run,
        candidate_results.as_ref(),
        verification.example_count,
        shard_name,
        shard_digest,
        snapshot_name,
        snapshot_digest,
        telemetry_name,
        telemetry_digest,
        starts_name,
        starts_digest,
        elapsed,
        broker_telemetry,
    );
    write_new_synced(
        &reservation.partial_directory.join(metadata_name),
        metadata.as_bytes(),
    )?;
    write_manifest(
        &reservation.partial_directory,
        &[
            metadata_name,
            shard_name,
            snapshot_name,
            starts_name,
            telemetry_name,
        ],
    )?;
    let directory = reservation.publish()?;
    Ok(PublishedActorRun {
        snapshot_path: directory.join(snapshot_name),
        directory,
        shard_digest,
        snapshot_digest,
        game_count: shard.games().len(),
        example_count: verification.example_count,
        candidate_results,
    })
}

fn validate_retained_games(
    retained: &[ReplayMatchResult],
    options: &Options,
    candidate: ReplayDigestV1,
    opponent: ReplayDigestV1,
) -> Result<(), BoxError> {
    if retained.len() > options.target_games {
        return Err("actor campaign retained more games than requested".into());
    }
    for result in retained {
        match result.neutral_start {
            Some(start) if start.prefix_decisions() == result.starting_decisions => {}
            Some(_) => {
                return Err("retained actor game disagrees with its neutral-start prefix".into())
            }
            None if result.starting_decisions == 0 => {}
            None => return Err("retained actor game has an unidentified starting prefix".into()),
        }
    }
    if !options.opponent.requires_paired_seats() {
        return Ok(());
    }
    if retained.len() % 2 != 0 {
        return Err("paired actor campaign retained an unpaired game".into());
    }
    for pair in retained.chunks_exact(2) {
        let first = &pair[0].game;
        let second = &pair[1].game;
        if first.game_id().checked_add(1) != Some(second.game_id())
            || first.record().setup() != second.record().setup()
            || pair[0].starting_decisions != pair[1].starting_decisions
            || pair[0].neutral_start != pair[1].neutral_start
        {
            return Err("retained actor games do not form a reversed-seat setup pair".into());
        }
        let prefix_decisions = pair[0].starting_decisions;
        if first.record().actions()[..prefix_decisions]
            != second.record().actions()[..prefix_decisions]
        {
            return Err("retained actor pair does not share the same starting prefix".into());
        }
        if first.host_agent() != candidate
            || first.guest_agent() != opponent
            || second.host_agent() != opponent
            || second.guest_agent() != candidate
        {
            return Err("retained actor pair does not reverse candidate and opponent seats".into());
        }
    }
    Ok(())
}

fn start_telemetry_text(matches: &[ReplayMatchResult]) -> String {
    let mut text = String::from(
        "game_id\tmode\tprefix_decisions\tsource_seed\tsource_attempt\tsource_decisions\tsource_remaining_decisions\n",
    );
    use core::fmt::Write as _;
    for result in matches {
        match result.neutral_start {
            Some(start) => writeln!(
                text,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                result.game.game_id(),
                NEUTRAL_START_POLICY_V1,
                result.starting_decisions,
                start.source_seed(),
                start.source_attempt(),
                start.source_decisions(),
                start.source_remaining_decisions(),
            ),
            None => writeln!(
                text,
                "{}\tstandard\t0\tNONE\tNONE\tNONE\tNONE",
                result.game.game_id()
            ),
        }
        .expect("writing start telemetry to a String cannot fail");
    }
    text
}

fn match_telemetry_text(matches: &[ReplayMatchResult]) -> String {
    let mut text = String::from(
        "game_id\tplayer\tagent\tdecisions\tsimulations\tevaluated_actions\texpanded_nodes\tgenerated_nodes\tgenerated_actions\tmaximum_search_depth\tmaximum_search_trees\tmaximum_search_workers\tmaximum_search_worker_capacity\tmaximum_action_ranking_workers\tmaximum_action_ranking_worker_capacity\trollout_steps\n",
    );
    for result in matches {
        append_agent_telemetry(
            &mut text,
            result.game.game_id(),
            "HOST",
            result.game.host_agent(),
            result.host_telemetry,
        );
        append_agent_telemetry(
            &mut text,
            result.game.game_id(),
            "GUEST",
            result.game.guest_agent(),
            result.guest_telemetry,
        );
    }
    text
}

fn append_agent_telemetry(
    text: &mut String,
    game_id: u64,
    player: &str,
    agent: ReplayDigestV1,
    telemetry: AgentTelemetry,
) {
    use core::fmt::Write as _;
    writeln!(
        text,
        "{game_id}\t{player}\t{agent}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        telemetry.decisions,
        telemetry.simulations,
        telemetry.evaluated_actions,
        telemetry.expanded_nodes,
        telemetry.generated_nodes,
        telemetry.generated_actions,
        telemetry.maximum_search_depth,
        telemetry.maximum_search_trees,
        telemetry.maximum_search_workers,
        telemetry.maximum_search_worker_capacity,
        telemetry.maximum_action_ranking_workers,
        telemetry.maximum_action_ranking_worker_capacity,
        telemetry.rollout_steps,
    )
    .expect("writing telemetry to a String cannot fail");
}

#[allow(clippy::too_many_arguments)]
fn actor_metadata(
    options: &Options,
    publication_attempt: u64,
    candidate_descriptor: &str,
    candidate_digest: ReplayDigestV1,
    opponent_descriptor: &str,
    opponent_digest: ReplayDigestV1,
    run: &CampaignRun,
    candidate_results: Option<&CandidateResults>,
    examples: u64,
    shard_name: &str,
    shard_digest: ReplayDigestV1,
    snapshot_name: &str,
    snapshot_digest: ReplayDigestV1,
    telemetry_name: &str,
    telemetry_digest: ReplayDigestV1,
    starts_name: &str,
    starts_digest: ReplayDigestV1,
    elapsed: Duration,
    telemetry: &CapacityInferenceTelemetry,
) -> String {
    let mut text = String::new();
    use core::fmt::Write as _;
    writeln!(text, "PAISHO-ACTOR-RUN\t4").unwrap();
    writeln!(text, "publication-attempt\t{publication_attempt}").unwrap();
    writeln!(
        text,
        "candidate\t{candidate_digest}\t{candidate_descriptor}"
    )
    .unwrap();
    writeln!(text, "opponent\t{opponent_digest}\t{opponent_descriptor}").unwrap();
    writeln!(text, "actor-seed\t{}", options.actor_seed).unwrap();
    match options.neutral_start {
        Some(start) => {
            writeln!(text, "start-policy\t{NEUTRAL_START_POLICY_V1}").unwrap();
            writeln!(
                text,
                "start-target-remaining-decisions\t{}",
                start.target_remaining_decisions()
            )
            .unwrap();
            writeln!(text, "start-seed\t{}", start.base_seed()).unwrap();
            writeln!(
                text,
                "start-source-decision-limit\t{}",
                start.source_decision_soft_limit()
            )
            .unwrap();
            writeln!(
                text,
                "start-source-attempts\t{}",
                start.maximum_source_attempts()
            )
            .unwrap();
        }
        None => {
            writeln!(text, "start-policy\tstandard").unwrap();
            writeln!(text, "start-target-remaining-decisions\tNONE").unwrap();
            writeln!(text, "start-seed\tNONE").unwrap();
            writeln!(text, "start-source-decision-limit\tNONE").unwrap();
            writeln!(text, "start-source-attempts\tNONE").unwrap();
        }
    }
    writeln!(text, "first-game-id\t{}", options.first_game_id).unwrap();
    writeln!(text, "decision-soft-limit\t{}", options.decision_soft_limit).unwrap();
    writeln!(text, "target-games\t{}", options.target_games).unwrap();
    writeln!(text, "maximum-attempts\t{}", options.maximum_attempts).unwrap();
    writeln!(
        text,
        "target-reached\t{}",
        run.target_reached(options.target_games)
    )
    .unwrap();
    writeln!(
        text,
        "abort-reason\t{}",
        run.abort_reason.as_deref().map_or("NONE", sanitize)
    )
    .unwrap();
    writeln!(text, "attempts\t{}", run.attempts).unwrap();
    writeln!(text, "completed\t{}", run.completed).unwrap();
    writeln!(text, "retained-games\t{}", run.retained.len()).unwrap();
    writeln!(text, "interrupted\t{}", run.interrupted).unwrap();
    writeln!(text, "no-training-decision\t{}", run.no_training_decision).unwrap();
    writeln!(text, "failed\t{}", run.failed).unwrap();
    writeln!(text, "excluded-pairs\t{}", run.excluded_pairs).unwrap();
    match candidate_results {
        Some(results) => {
            writeln!(text, "evaluation-scope\tcurriculum-terminal-non-elo").unwrap();
            writeln!(text, "candidate-wins\t{}", results.wins).unwrap();
            writeln!(text, "candidate-draws\t{}", results.draws).unwrap();
            writeln!(text, "candidate-losses\t{}", results.losses).unwrap();
            writeln!(text, "candidate-pairs\t{}", results.pairs).unwrap();
            writeln!(text, "candidate-half-points\t{}", results.half_points()).unwrap();
            writeln!(
                text,
                "candidate-pentanomial\t{},{},{},{},{}",
                results.pentanomial[0],
                results.pentanomial[1],
                results.pentanomial[2],
                results.pentanomial[3],
                results.pentanomial[4]
            )
            .unwrap();
            let comparison = results.paired_comparison();
            writeln!(
                text,
                "candidate-favorable-pairs\t{}",
                comparison.favorable()
            )
            .unwrap();
            writeln!(text, "candidate-tied-pairs\t{}", comparison.tied()).unwrap();
            writeln!(
                text,
                "candidate-unfavorable-pairs\t{}",
                comparison.unfavorable()
            )
            .unwrap();
            writeln!(
                text,
                "candidate-paired-sign-test-p\t{:.12}",
                comparison.exact_two_sided_sign_test_p_value()
            )
            .unwrap();
        }
        None => writeln!(text, "evaluation-scope\tself-play-not-applicable").unwrap(),
    }
    writeln!(text, "workers-configured\t{}", options.workers).unwrap();
    writeln!(text, "workers-observed\t{}", run.maximum_workers_observed).unwrap();
    writeln!(text, "actors-per-round\t{}", options.actors).unwrap();
    writeln!(text, "wide-class-lanes\t{}", options.wide_lanes).unwrap();
    writeln!(text, "training-examples\t{examples}").unwrap();
    writeln!(text, "shard\t{shard_name}\t{shard_digest}").unwrap();
    writeln!(text, "snapshot\t{snapshot_name}\t{snapshot_digest}").unwrap();
    writeln!(text, "telemetry\t{telemetry_name}\t{telemetry_digest}").unwrap();
    writeln!(text, "starts\t{starts_name}\t{starts_digest}").unwrap();
    writeln!(
        text,
        "inference-requested\t{}",
        telemetry.requested_positions()
    )
    .unwrap();
    writeln!(
        text,
        "inference-executed\t{}",
        telemetry.executed_positions()
    )
    .unwrap();
    for class in &telemetry.classes {
        writeln!(
            text,
            "capacity\t{}\t{}\t{}\t{}\t{}\t{}",
            class.legal_action_capacity,
            class.batch_size,
            class.broker.batches,
            class.broker.requested_positions,
            class.broker.executed_positions,
            class.broker.padded_positions
        )
        .unwrap();
        for (lane_index, lane) in class.lanes.iter().enumerate() {
            writeln!(
                text,
                "capacity-lane\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                class.legal_action_capacity,
                lane_index,
                class.batch_size,
                lane.batches,
                lane.full_batches,
                lane.requested_positions,
                lane.executed_positions,
                lane.padded_positions
            )
            .unwrap();
        }
    }
    writeln!(text, "elapsed-seconds\t{:.6}", elapsed.as_secs_f64()).unwrap();
    let checksum = digest_bytes(text.as_bytes());
    writeln!(text, "sha256\t{checksum}").unwrap();
    text
}

fn sanitize(value: &str) -> &str {
    if value.contains(['\t', '\n', '\r']) {
        "ABORT_REASON_CONTAINS_CONTROL_CHARACTERS"
    } else {
        value
    }
}

fn write_manifest(directory: &Path, file_names: &[&str]) -> Result<(), BoxError> {
    let mut names = file_names.to_vec();
    names.sort_unstable();
    let mut manifest = String::new();
    use core::fmt::Write as _;
    for name in names {
        writeln!(
            manifest,
            "{}  {name}",
            hex_digest(sha256_file(&directory.join(name))?)
        )
        .expect("writing a manifest to a String cannot fail");
    }
    let path = directory.join("MANIFEST.sha256");
    write_new_synced(&path, manifest.as_bytes())?;
    if fs::read_to_string(path)? != manifest {
        return Err("actor manifest did not round-trip after publication".into());
    }
    Ok(())
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), BoxError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<[u8; 32], BoxError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn digest_bytes(bytes: &[u8]) -> ReplayDigestV1 {
    ReplayDigestV1::from_bytes(Sha256::digest(bytes).into())
}

fn sync_tree(directory: &Path) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            sync_tree(&entry.path())?;
        } else {
            File::open(entry.path())?.sync_all()?;
        }
    }
    sync_directory(directory)
}

fn sync_directory(directory: &Path) -> io::Result<()> {
    File::open(directory)?.sync_all()
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing to a String cannot fail");
    }
    text
}

#[cfg(test)]
#[path = "paisho_actors/tests.rs"]
mod tests;
