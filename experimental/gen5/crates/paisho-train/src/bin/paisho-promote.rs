use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use paisho_ai::{MatchConfig, NetworkPolicy, PureNetworkAgent, StableRng};
use paisho_core::Player;
use paisho_mpsgraph_client::{
    default_service_path, read_checkpoint_metadata, CapacityInferenceBroker,
    CapacityInferenceBrokerClient, CapacityInferenceTelemetry, InferenceBrokerConfiguration,
    NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_rating::evaluate_promotion_sprt;
use paisho_train::{
    build_promotion_schedule_with_neutral_starts, run_promotion_schedule, PairExclusionReason,
    PromotionCampaignArchive, PromotionCampaignConclusion, PromotionCampaignProgress,
    PromotionInferenceClassV1, PromotionNeutralStartV1, PromotionRunIdentityV1,
    PromotionSamplingPolicyV1, PromotionScheduledGame,
};
use sha2::{Digest, Sha256};

type BoxError = Box<dyn Error + Send + Sync>;
const BUILD_SOURCE_REVISION: &str = env!("PAISHO_BUILD_GIT_REVISION");
const BUILD_SOURCE_DIRTY: &str = env!("PAISHO_BUILD_GIT_DIRTY");
const BUILD_SOURCE_SHA256: &str = env!("PAISHO_BUILD_SOURCE_SHA256");

#[derive(Clone, Copy, Debug)]
struct ClassShape {
    capacity: usize,
    batch_size: usize,
}

struct Options {
    output_directory: PathBuf,
    service: PathBuf,
    candidate_checkpoint: PathBuf,
    champion_checkpoint: PathBuf,
    preset: NetworkPreset,
    optimization: OptimizationLevel,
    classes: Vec<ClassShape>,
    workers: usize,
    pairs_per_batch: usize,
    maximum_attempted_pairs: u64,
    maximum_eligible_pairs: u64,
    first_pair_id: u64,
    decision_soft_limit: usize,
    maximum_batch_wait: Duration,
    model_seed: u64,
    neutral_start_horizon: Option<usize>,
    neutral_start_seed: u64,
    neutral_start_source_limit: usize,
    neutral_start_attempts: usize,
    sampling_temperature: Option<f32>,
    sampling_uniform_mix: f32,
    elo0: f64,
    elo1: f64,
    alpha: f64,
    beta: f64,
}

impl Options {
    fn parse() -> Result<Self, BoxError> {
        let available_workers = std::thread::available_parallelism()?.get();
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut output_directory = None;
        let mut service = default_service_path(&repository);
        let mut candidate_checkpoint = None;
        let mut champion_checkpoint = None;
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
        let mut neutral_start_horizon = None;
        let mut neutral_start_seed = 0x5052_4f4d_4f54_4553;
        let mut neutral_start_source_limit = 16_384;
        let mut neutral_start_attempts = 16;
        let mut sampling_temperature = None;
        let mut sampling_uniform_mix = 0.05;
        let mut elo0 = 0.0;
        let mut elo1 = 10.0;
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
                "--champion-checkpoint" => champion_checkpoint = Some(PathBuf::from(value)),
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
            champion_checkpoint: champion_checkpoint.ok_or("missing --champion-checkpoint PATH")?,
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
            alpha,
            beta,
        })
    }
}

fn print_usage() {
    println!(
        "usage: cargo run --release -p paisho-train --bin paisho-promote -- \
         --output-dir DIR --candidate-checkpoint PATH --champion-checkpoint PATH \
         --max-eligible-pairs N [options]\n\
         --max-attempted-pairs N --pairs-per-batch N --workers N\n\
         --preset pure|micro --level 0|1 --classes 64:8,128:4,1024:4\n\
         --elo0 F --elo1 F --alpha F --beta F --decision-limit N\n\
         --start-horizon N (0 disables) --start-seed N\n\
         --start-source-limit N --start-source-attempts N\n\
         --sampling-temperature F (0 disables) --sampling-uniform-mix F\n\
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

fn main() -> Result<(), BoxError> {
    let options = Options::parse()?;
    let candidate_metadata = read_checkpoint_metadata(&options.candidate_checkpoint)?;
    let champion_metadata = read_checkpoint_metadata(&options.champion_checkpoint)?;
    if candidate_metadata.network_preset() != options.preset
        || champion_metadata.network_preset() != options.preset
    {
        return Err("both checkpoints must use the selected network preset".into());
    }
    let identity = run_identity(
        &options,
        candidate_metadata.content_sha256(),
        champion_metadata.content_sha256(),
    )?;
    let archive = PromotionCampaignArchive::open_or_create(&options.output_directory, identity)?;
    let progress = archive.load_progress()?;
    if progress.conclusion(archive.identity())?.is_some() {
        let conclusion = archive.publish_conclusion(&progress)?;
        print_final(&archive, &progress, conclusion)?;
        return Ok(());
    }

    let candidate_broker = launch_broker(&options, &options.candidate_checkpoint)?;
    let champion_broker = match launch_broker(&options, &options.champion_checkpoint) {
        Ok(broker) => broker,
        Err(source) => {
            let _ = candidate_broker.shutdown();
            return Err(source);
        }
    };
    let candidate_client = candidate_broker.client()?;
    let champion_client = champion_broker.client()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let campaign = run_campaign(
        &archive,
        &pool,
        candidate_client.clone(),
        champion_client.clone(),
    );
    drop(candidate_client);
    drop(champion_client);
    let candidate_telemetry = candidate_broker.shutdown();
    let champion_telemetry = champion_broker.shutdown();
    let (progress, conclusion) = campaign?;
    let candidate_telemetry = candidate_telemetry?;
    let champion_telemetry = champion_telemetry?;
    print_broker_telemetry("candidate", &candidate_telemetry);
    print_broker_telemetry("champion", &champion_telemetry);
    print_final(&archive, &progress, conclusion)?;
    Ok(())
}

fn run_identity(
    options: &Options,
    candidate_checkpoint_digest: [u8; 32],
    champion_checkpoint_digest: [u8; 32],
) -> Result<PromotionRunIdentityV1, BoxError> {
    let maximum_batch_wait_microseconds = u64::try_from(options.maximum_batch_wait.as_micros())
        .map_err(|_| "maximum batch wait does not fit u64 microseconds")?;
    let identity = PromotionRunIdentityV1 {
        source_sha256: BUILD_SOURCE_SHA256.to_owned(),
        source_revision: BUILD_SOURCE_REVISION.to_owned(),
        source_dirty: BUILD_SOURCE_DIRTY == "true",
        service_sha256: sha256_file(&options.service)?,
        candidate_checkpoint_sha256: hex_digest(candidate_checkpoint_digest),
        champion_checkpoint_sha256: hex_digest(champion_checkpoint_digest),
        preset: preset_name(options.preset).to_owned(),
        optimization_level: optimization_level(options.optimization),
        inference_classes: options
            .classes
            .iter()
            .map(|class| PromotionInferenceClassV1 {
                legal_action_capacity: class.capacity,
                batch_size: class.batch_size,
            })
            .collect(),
        workers: options.workers,
        pairs_per_batch: options.pairs_per_batch,
        maximum_attempted_pairs: options.maximum_attempted_pairs,
        maximum_eligible_pairs: options.maximum_eligible_pairs,
        first_pair_id: options.first_pair_id,
        decision_soft_limit: options.decision_soft_limit,
        maximum_batch_wait_microseconds,
        model_seed: options.model_seed,
        neutral_start: options
            .neutral_start_horizon
            .map(|target_remaining_decisions| PromotionNeutralStartV1 {
                target_remaining_decisions,
                seed: options.neutral_start_seed,
                source_decision_limit: options.neutral_start_source_limit,
                maximum_source_attempts: options.neutral_start_attempts,
            }),
        sampling_policy: options
            .sampling_temperature
            .map(|temperature| {
                PromotionSamplingPolicyV1::new(temperature, options.sampling_uniform_mix)
            })
            .transpose()?,
        elo0: options.elo0,
        elo1: options.elo1,
        alpha: options.alpha,
        beta: options.beta,
    };
    identity.validate()?;
    Ok(identity)
}

fn launch_broker(
    options: &Options,
    checkpoint: &Path,
) -> Result<CapacityInferenceBroker, BoxError> {
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
            checkpoint: Some(checkpoint.to_owned()),
        })
        .collect();
    Ok(CapacityInferenceBroker::launch(
        services,
        InferenceBrokerConfiguration {
            prepare_ahead: false,
            maximum_batch_wait: options.maximum_batch_wait,
            maximum_in_flight_batches: 1,
        },
    )?)
}

fn run_campaign(
    archive: &PromotionCampaignArchive,
    pool: &rayon::ThreadPool,
    candidate: CapacityInferenceBrokerClient,
    champion: CapacityInferenceBrokerClient,
) -> Result<(PromotionCampaignProgress, PromotionCampaignConclusion), BoxError> {
    let identity = archive.identity();
    let network_policy = identity.network_policy()?;
    let mut progress = archive.load_progress()?;
    loop {
        if progress.conclusion(identity)?.is_some() {
            let conclusion = archive.publish_conclusion(&progress)?;
            return Ok((progress, conclusion));
        }
        let pair_count = next_batch_pair_count(identity, &progress)?;
        let first_pair_id = progress.next_pair_id(identity)?;
        let neutral_start = identity.neutral_start_configuration()?;
        let schedule = pool.install(|| {
            build_promotion_schedule_with_neutral_starts(
                identity.first_pair_id,
                first_pair_id,
                pair_count,
                neutral_start,
            )
        })?;
        let execution = pool.install(|| {
            run_promotion_schedule(
                &schedule,
                MatchConfig {
                    decision_soft_limit: identity.decision_soft_limit,
                },
                |scheduled, player| {
                    network_agent(
                        candidate.clone(),
                        identity.model_seed,
                        scheduled,
                        player,
                        0,
                        network_policy,
                    )
                },
                |scheduled, player| {
                    network_agent(
                        champion.clone(),
                        identity.model_seed,
                        scheduled,
                        player,
                        1,
                        network_policy,
                    )
                },
            )
        });
        let match_error = execution
            .pair_evaluations()?
            .iter()
            .any(|pair| pair.exclusion == Some(PairExclusionReason::MatchError));
        progress = archive.publish_next_batch(&progress, &execution)?;
        print_progress(identity, &progress)?;
        if match_error {
            return Err(format!(
                "promotion batch {} contains an agent or engine failure; committed evidence was preserved at {}",
                progress.batches - 1,
                archive.root().display()
            )
            .into());
        }
    }
}

fn next_batch_pair_count(
    identity: &PromotionRunIdentityV1,
    progress: &PromotionCampaignProgress,
) -> Result<usize, BoxError> {
    let attempted_remaining = identity
        .maximum_attempted_pairs
        .checked_sub(progress.attempted_pairs)
        .ok_or("attempted promotion pairs exceed the configured maximum")?;
    let eligible_remaining = identity
        .maximum_eligible_pairs
        .checked_sub(progress.eligible_pairs)
        .ok_or("eligible promotion pairs exceed the configured maximum")?;
    let configured = u64::try_from(identity.pairs_per_batch)
        .map_err(|_| "configured pair batch does not fit u64")?;
    let count = configured.min(attempted_remaining).min(eligible_remaining);
    if count == 0 {
        return Err("promotion campaign has no remaining pair slot before a conclusion".into());
    }
    usize::try_from(count).map_err(|_| "next promotion batch does not fit usize".into())
}

fn network_agent(
    client: CapacityInferenceBrokerClient,
    base_seed: u64,
    scheduled: &PromotionScheduledGame,
    player: Player,
    family: u64,
    policy: NetworkPolicy,
) -> PureNetworkAgent<CapacityInferenceBrokerClient> {
    PureNetworkAgent::new(
        client,
        network_seed(base_seed, scheduled.pair_id, player, family),
        policy,
    )
    .expect("the promotion identity already validated its network policy")
}

fn network_seed(base: u64, pair_id: u64, player: Player, family: u64) -> u64 {
    let side = match player {
        Player::Host => 0x484f_5354,
        Player::Guest => 0x0047_5545_5354,
    };
    let mut rng = StableRng::new(
        base ^ pair_id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ side ^ family.rotate_left(29),
    );
    rng.next_u64()
}

fn print_progress(
    identity: &PromotionRunIdentityV1,
    progress: &PromotionCampaignProgress,
) -> Result<(), BoxError> {
    let report = evaluate_promotion_sprt(progress.pentanomial, identity.sprt_configuration()?)?;
    println!(
        "batch={} attempted_pairs={} eligible_pairs={} excluded_pairs={} pentanomial={:?}",
        progress.batches,
        progress.attempted_pairs,
        progress.eligible_pairs,
        progress.excluded_pairs,
        progress.pentanomial.bins(),
    );
    println!(
        "llr={:.9} bounds=[{:.9},{:.9}] decision={:?} workers_observed={}/{}",
        report.log_likelihood_ratio,
        report.lower_bound,
        report.upper_bound,
        report.decision,
        progress.maximum_workers_observed,
        identity.workers,
    );
    Ok(())
}

fn print_final(
    archive: &PromotionCampaignArchive,
    progress: &PromotionCampaignProgress,
    conclusion: PromotionCampaignConclusion,
) -> Result<(), BoxError> {
    print_progress(archive.identity(), progress)?;
    println!("conclusion={conclusion:?}");
    println!("archive={}", archive.root().display());
    println!(
        "inference_positions=candidate:{} champion:{}",
        progress.candidate_inference_positions, progress.champion_inference_positions
    );
    Ok(())
}

fn print_broker_telemetry(name: &str, telemetry: &CapacityInferenceTelemetry) {
    println!(
        "{name}_broker=requested:{} executed:{}",
        telemetry.requested_positions(),
        telemetry.executed_positions()
    );
    for class in &telemetry.classes {
        println!(
            "{name}_capacity={} batch={} batches={} requested={} executed={} padded={}",
            class.legal_action_capacity,
            class.batch_size,
            class.broker.batches,
            class.broker.requested_positions,
            class.broker.executed_positions,
            class.broker.padded_positions,
        );
    }
}

fn preset_name(preset: NetworkPreset) -> &'static str {
    match preset {
        NetworkPreset::Micro => "micro",
        NetworkPreset::Pure => "pure",
    }
}

const fn optimization_level(level: OptimizationLevel) -> u8 {
    match level {
        OptimizationLevel::Level0 => 0,
        OptimizationLevel::Level1 => 1,
    }
}

fn sha256_file(path: &Path) -> Result<String, BoxError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1_024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(hasher.finalize().into()))
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing a digest to String cannot fail");
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_size_stops_at_each_remaining_budget() {
        let options = Options {
            output_directory: PathBuf::new(),
            service: PathBuf::new(),
            candidate_checkpoint: PathBuf::new(),
            champion_checkpoint: PathBuf::new(),
            preset: NetworkPreset::Pure,
            optimization: OptimizationLevel::Level1,
            classes: Vec::new(),
            workers: 10,
            pairs_per_batch: 10,
            maximum_attempted_pairs: 30,
            maximum_eligible_pairs: 20,
            first_pair_id: 0,
            decision_soft_limit: 2_048,
            maximum_batch_wait: Duration::ZERO,
            model_seed: 17,
            neutral_start_horizon: None,
            neutral_start_seed: 0,
            neutral_start_source_limit: 1,
            neutral_start_attempts: 1,
            sampling_temperature: None,
            sampling_uniform_mix: 0.05,
            elo0: 0.0,
            elo1: 10.0,
            alpha: 0.05,
            beta: 0.05,
        };
        let identity = PromotionRunIdentityV1 {
            source_sha256: "11".repeat(32),
            source_revision: "test".to_owned(),
            source_dirty: false,
            service_sha256: "22".repeat(32),
            candidate_checkpoint_sha256: "33".repeat(32),
            champion_checkpoint_sha256: "44".repeat(32),
            preset: "pure".to_owned(),
            optimization_level: 1,
            inference_classes: vec![PromotionInferenceClassV1 {
                legal_action_capacity: 1,
                batch_size: 1,
            }],
            workers: options.workers,
            pairs_per_batch: options.pairs_per_batch,
            maximum_attempted_pairs: options.maximum_attempted_pairs,
            maximum_eligible_pairs: options.maximum_eligible_pairs,
            first_pair_id: options.first_pair_id,
            decision_soft_limit: options.decision_soft_limit,
            maximum_batch_wait_microseconds: 0,
            model_seed: options.model_seed,
            neutral_start: None,
            sampling_policy: None,
            elo0: options.elo0,
            elo1: options.elo1,
            alpha: options.alpha,
            beta: options.beta,
        };
        let mut progress = PromotionCampaignProgress {
            attempted_pairs: 24,
            eligible_pairs: 15,
            ..PromotionCampaignProgress::default()
        };
        assert_eq!(next_batch_pair_count(&identity, &progress).unwrap(), 5);
        progress.attempted_pairs = 29;
        assert_eq!(next_batch_pair_count(&identity, &progress).unwrap(), 1);
    }

    #[test]
    fn network_seeds_are_reproducible_and_separate_model_families() {
        let candidate = network_seed(17, 42, Player::Host, 0);
        assert_eq!(candidate, network_seed(17, 42, Player::Host, 0));
        assert_ne!(candidate, network_seed(17, 42, Player::Guest, 0));
        assert_ne!(candidate, network_seed(17, 42, Player::Host, 1));
    }
}
