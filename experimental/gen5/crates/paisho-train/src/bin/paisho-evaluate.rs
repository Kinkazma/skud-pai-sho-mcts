use std::env;
use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use paisho_ai::{MatchConfig, NetworkPolicy, PureNetworkAgent, StableRng};
use paisho_core::Player;
use paisho_mpsgraph_client::{
    read_checkpoint_metadata, CapacityInferenceBroker, CapacityInferenceBrokerClient,
    CapacityInferenceTelemetry, InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel,
    ServiceConfiguration,
};
use paisho_rating::evaluate_promotion_sprt;
use paisho_train::{
    build_promotion_schedule_with_neutral_starts, run_promotion_schedule, ContextualEloPointV1,
    EvaluationAnalysisV1, EvaluationCampaignArchive, EvaluationInferenceClassV1,
    EvaluationProgress, EvaluationRunIdentityV1, PromotionNeutralStartV1,
    PromotionSamplingPolicyV1, PromotionScheduledGame,
};
use sha2::{Digest, Sha256};

type BoxError = Box<dyn Error + Send + Sync>;

#[path = "paisho_evaluate/options.rs"]
mod options;

use options::Options;

const BUILD_SOURCE_REVISION: &str = env!("PAISHO_BUILD_GIT_REVISION");
const BUILD_SOURCE_DIRTY: &str = env!("PAISHO_BUILD_GIT_DIRTY");
const BUILD_SOURCE_SHA256: &str = env!("PAISHO_BUILD_SOURCE_SHA256");

fn main() -> Result<(), BoxError> {
    if let Some(directory) = verify_directory()? {
        let archive = EvaluationCampaignArchive::open_existing(&directory)?;
        let progress = archive.load_progress()?;
        let analysis = archive.analysis()?;
        print_final(&archive, &progress, &analysis)?;
        println!("verified_evaluation_archive={}", directory.display());
        return Ok(());
    }
    let options = Options::parse()?;
    let checkpoint_metadata = read_checkpoint_metadata(&options.candidate_checkpoint)?;
    if checkpoint_metadata.network_preset() != options.preset {
        return Err("candidate checkpoint does not use the selected network preset".into());
    }
    let identity = run_identity(&options, checkpoint_metadata.content_sha256())?;
    let archive = EvaluationCampaignArchive::open_or_create(&options.output_directory, identity)?;
    let progress = archive.load_progress()?;
    if progress.conclusion(archive.identity())?.is_some() {
        let analysis = archive.publish_conclusion(&progress)?;
        print_final(&archive, &progress, &analysis)?;
        return Ok(());
    }

    let broker = launch_broker(&options)?;
    let client = broker.client()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let campaign = run_campaign(&archive, &pool, client.clone());
    drop(client);
    let telemetry = broker.shutdown();
    let (progress, analysis) = campaign?;
    let telemetry = telemetry?;
    print_broker_telemetry(&telemetry);
    print_final(&archive, &progress, &analysis)?;
    Ok(())
}

fn verify_directory() -> Result<Option<PathBuf>, BoxError> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .map_or(true, |argument| argument != "--verify")
    {
        return Ok(None);
    }
    if arguments.len() != 2 {
        return Err("usage: paisho-evaluate --verify EVALUATION-DIRECTORY".into());
    }
    Ok(Some(PathBuf::from(&arguments[1])))
}

fn run_identity(
    options: &Options,
    checkpoint_digest: [u8; 32],
) -> Result<EvaluationRunIdentityV1, BoxError> {
    let maximum_batch_wait_microseconds = u64::try_from(options.maximum_batch_wait.as_micros())
        .map_err(|_| "maximum batch wait does not fit u64 microseconds")?;
    let opponent_descriptor = options.opponent.descriptor(BUILD_SOURCE_SHA256);
    let identity = EvaluationRunIdentityV1 {
        source_sha256: BUILD_SOURCE_SHA256.to_owned(),
        source_revision: BUILD_SOURCE_REVISION.to_owned(),
        source_dirty: BUILD_SOURCE_DIRTY == "true",
        service_sha256: sha256_file(&options.service)?,
        candidate_checkpoint_sha256: hex_digest(checkpoint_digest),
        opponent: options.opponent,
        opponent_sha256: paisho_train::sha256_text(&opponent_descriptor),
        opponent_descriptor,
        preset: preset_name(options.preset).to_owned(),
        optimization_level: optimization_level(options.optimization),
        inference_classes: options
            .classes
            .iter()
            .map(|class| EvaluationInferenceClassV1 {
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
        lower_elo: options.lower_elo,
        alpha: options.alpha,
        beta: options.beta,
    };
    identity.validate()?;
    Ok(identity)
}

fn launch_broker(options: &Options) -> Result<CapacityInferenceBroker, BoxError> {
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
            checkpoint: Some(options.candidate_checkpoint.clone()),
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
    archive: &EvaluationCampaignArchive,
    pool: &rayon::ThreadPool,
    candidate: CapacityInferenceBrokerClient,
) -> Result<(EvaluationProgress, EvaluationAnalysisV1), BoxError> {
    let identity = archive.identity();
    let network_policy = identity.network_policy()?;
    let mut progress = archive.load_progress()?;
    loop {
        if progress.conclusion(identity)?.is_some() {
            let analysis = archive.publish_conclusion(&progress)?;
            return Ok((progress, analysis));
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
                        network_policy,
                    )
                },
                |scheduled, player| {
                    identity.opponent.make(agent_seed(
                        identity.model_seed,
                        scheduled.pair_id,
                        player,
                        1,
                    ))
                },
            )
        });
        let has_match_error = execution.games.iter().any(|game| game.result.is_err());
        progress = archive.publish_next_batch(&progress, &execution)?;
        print_progress(identity, &progress)?;
        if has_match_error {
            return Err(format!(
                "evaluation batch {} contains an agent or engine failure; committed evidence was preserved at {}",
                progress.batches - 1,
                archive.root().display()
            )
            .into());
        }
    }
}

fn next_batch_pair_count(
    identity: &EvaluationRunIdentityV1,
    progress: &EvaluationProgress,
) -> Result<usize, BoxError> {
    let attempted_remaining = identity
        .maximum_attempted_pairs
        .checked_sub(progress.attempted_pairs)
        .ok_or("attempted evaluation pairs exceed the configured maximum")?;
    let eligible_remaining = identity
        .maximum_eligible_pairs
        .checked_sub(progress.eligible_pairs)
        .ok_or("eligible evaluation pairs exceed the configured maximum")?;
    let configured = u64::try_from(identity.pairs_per_batch)
        .map_err(|_| "configured pair batch does not fit u64")?;
    let count = configured.min(attempted_remaining).min(eligible_remaining);
    if count == 0 {
        return Err("evaluation campaign has no remaining pair slot before a conclusion".into());
    }
    usize::try_from(count).map_err(|_| "next evaluation batch does not fit usize".into())
}

fn network_agent(
    client: CapacityInferenceBrokerClient,
    base_seed: u64,
    scheduled: &PromotionScheduledGame,
    player: Player,
    policy: NetworkPolicy,
) -> PureNetworkAgent<CapacityInferenceBrokerClient> {
    PureNetworkAgent::new(
        client,
        agent_seed(base_seed, scheduled.pair_id, player, 0),
        policy,
    )
    .expect("the evaluation identity already validated its network policy")
}

fn agent_seed(base: u64, pair_id: u64, player: Player, family: u64) -> u64 {
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
    identity: &EvaluationRunIdentityV1,
    progress: &EvaluationProgress,
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
        "contextual_score={:?} llr={:.9} bounds=[{:.9},{:.9}] decision={:?} workers_observed={}/{}",
        report.empirical_score,
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
    archive: &EvaluationCampaignArchive,
    progress: &EvaluationProgress,
    analysis: &EvaluationAnalysisV1,
) -> Result<(), BoxError> {
    print_progress(archive.identity(), progress)?;
    println!("conclusion={:?}", analysis.conclusion);
    println!(
        "candidate_wdl={}/{}/{} score={:?}",
        analysis.candidate_wins,
        analysis.draws,
        analysis.candidate_losses,
        analysis.empirical_score
    );
    match analysis.contextual_elo_point {
        ContextualEloPointV1::NoRatedGames => println!("contextual_elo=unavailable"),
        ContextualEloPointV1::NegativeInfinity => println!("contextual_elo=-infinity"),
        ContextualEloPointV1::Finite { elo } => println!("contextual_elo={elo:.3}"),
        ContextualEloPointV1::PositiveInfinity => println!("contextual_elo=+infinity"),
    }
    if let Some(mle) = &analysis.mle {
        let gap = mle.candidate_minus_opponent;
        println!(
            "davidson_gap={:.3} model_ci95=[{:.3},{:.3}]",
            gap.estimate, gap.model.interval_95_lower, gap.model.interval_95_upper
        );
        if let Some(cluster) = gap.paired_cluster {
            println!(
                "paired_cluster_ci95=[{:.3},{:.3}]",
                cluster.interval_95_lower, cluster.interval_95_upper
            );
        } else {
            println!("paired_cluster_ci95=unavailable");
        }
    } else if let Some(error) = &analysis.mle_error {
        println!("davidson_gap=unavailable ({error})");
    }
    if let Some(lower) = analysis.lower_window {
        println!(
            "lower_window=[{:.3},{:.3}] llr={:.9} bounds=[{:.9},{:.9}]",
            lower.lower_elo,
            lower.center_elo,
            lower.log_likelihood_ratio,
            lower.lower_bound,
            lower.upper_bound
        );
    }
    println!(
        "opponent={} candidate_inference_positions={} archive={}",
        archive.identity().opponent.display_label(),
        progress.candidate_inference_positions,
        archive.root().display()
    );
    Ok(())
}

fn print_broker_telemetry(telemetry: &CapacityInferenceTelemetry) {
    println!(
        "candidate_broker=requested:{} executed:{}",
        telemetry.requested_positions(),
        telemetry.executed_positions()
    );
    for class in &telemetry.classes {
        println!(
            "candidate_capacity={} batch={} batches={} requested={} executed={} padded={}",
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
    fn seed_streams_are_reproducible_and_separate_agent_families() {
        let candidate = agent_seed(17, 42, Player::Host, 0);
        assert_eq!(candidate, agent_seed(17, 42, Player::Host, 0));
        assert_ne!(candidate, agent_seed(17, 42, Player::Guest, 0));
        assert_ne!(candidate, agent_seed(17, 42, Player::Host, 1));
    }

    #[test]
    fn batch_size_stops_at_both_remaining_budgets() {
        let source_sha256 = "11".repeat(32);
        let opponent = paisho_train::EvaluationOpponentV1::Random;
        let opponent_descriptor = opponent.descriptor(&source_sha256);
        let identity = EvaluationRunIdentityV1 {
            source_sha256,
            source_revision: "test".to_owned(),
            source_dirty: false,
            service_sha256: "22".repeat(32),
            candidate_checkpoint_sha256: "33".repeat(32),
            opponent,
            opponent_sha256: paisho_train::sha256_text(&opponent_descriptor),
            opponent_descriptor,
            preset: "pure".to_owned(),
            optimization_level: 1,
            inference_classes: vec![EvaluationInferenceClassV1 {
                legal_action_capacity: 1,
                batch_size: 1,
            }],
            workers: 10,
            pairs_per_batch: 10,
            maximum_attempted_pairs: 30,
            maximum_eligible_pairs: 20,
            first_pair_id: 0,
            decision_soft_limit: 1,
            maximum_batch_wait_microseconds: 0,
            model_seed: 17,
            neutral_start: None,
            sampling_policy: None,
            elo0: 0.0,
            elo1: 10.0,
            lower_elo: None,
            alpha: 0.05,
            beta: 0.05,
        };
        let mut progress = EvaluationProgress {
            attempted_pairs: 24,
            eligible_pairs: 15,
            ..EvaluationProgress::default()
        };
        assert_eq!(next_batch_pair_count(&identity, &progress).unwrap(), 5);
        progress.attempted_pairs = 29;
        assert_eq!(next_batch_pair_count(&identity, &progress).unwrap(), 1);
    }
}
