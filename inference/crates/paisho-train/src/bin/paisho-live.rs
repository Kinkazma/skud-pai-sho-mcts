//! Fresh games + PPO with persistent learner and actor services; durable block boundaries.
#[path = "paisho_live/assessment.rs"]
mod assessment;
#[path = "paisho_live/journal.rs"]
mod journal;
#[path = "paisho_generations/options.rs"]
#[allow(dead_code)] // Shared parser also exposes the legacy binary's entry point.
mod options;
#[path = "paisho_live/results.rs"]
mod results;

use journal::Block;
use paisho_ai::{NetworkPolicy, StableRng};
use paisho_model::TerminalPpoParametersV1;
use paisho_mpsgraph_client::{
    read_checkpoint_metadata, CapacityInferenceBroker, InferenceBrokerConfiguration,
    ServiceConfiguration,
};
use paisho_replay::{
    ReplayDatasetV1, ReplayDigestV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1,
};
use paisho_train::{
    live_actors::{
        collect_live_games, collect_live_weighted_games, LiveActorsConfiguration,
        LiveActorsOpponent,
    },
    live_learner::{hex, LiveLearner},
    CurriculumTierV1, NeutralStartConfigurationV1,
};
use serde_json::json;
use std::{
    error::Error,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};
type BoxError = Box<dyn Error + Send + Sync>;
const GENERATION_IDENTIFIER_STRIDE: u64 = 1_000_000_000;

// Scheduling override, recorded per attempt rather than rewriting the campaign plan.
fn take_actor_round_size(arguments: &mut Vec<String>) -> Result<Option<usize>, BoxError> {
    let Some(index) = arguments.iter().position(|a| a == "--actor-round-size") else {
        return Ok(None);
    };
    let value: usize = arguments
        .get(index + 1)
        .ok_or("missing --actor-round-size")?
        .parse()?;
    if value == 0 || value % 2 != 0 {
        return Err("actor round size must be positive and even for paired opponents".into());
    }
    arguments.drain(index..index + 2);
    Ok(Some(value))
}

fn take_count_override(arguments: &mut Vec<String>, flag: &str) -> Result<Option<usize>, BoxError> {
    let Some(index) = arguments.iter().position(|a| a == flag) else {
        return Ok(None);
    };
    let value = arguments
        .get(index + 1)
        .ok_or("missing count option")?
        .parse()?;
    arguments.drain(index..index + 2);
    Ok(Some(value))
}

fn take_positive_override(
    arguments: &mut Vec<String>,
    flag: &str,
) -> Result<Option<usize>, BoxError> {
    let Some(index) = arguments.iter().position(|a| a == flag) else {
        return Ok(None);
    };
    let value: usize = arguments
        .get(index + 1)
        .ok_or("missing scheduling value")?
        .parse()?;
    if value == 0 {
        return Err("scheduling value must be positive".into());
    }
    arguments.drain(index..index + 2);
    Ok(Some(value))
}

#[cfg(test)]
mod scheduling_tests {
    use super::{take_actor_round_size, take_count_override, take_positive_override};

    #[test]
    fn explicit_external_quota_accepts_pure_self_play_and_rejects_bad_values() {
        let mut args = vec![
            "--external-games".into(),
            "0".into(),
            "--actor-games".into(),
            "256".into(),
        ];
        assert_eq!(
            take_count_override(&mut args, "--external-games").unwrap(),
            Some(0)
        );
        assert_eq!(args, ["--actor-games", "256"]);
        for value in ["-1", "bad"] {
            assert!(take_count_override(
                &mut vec!["--external-games".into(), value.into()],
                "--external-games"
            )
            .is_err());
        }
        assert!(
            take_count_override(&mut vec!["--external-games".into()], "--external-games").is_err()
        );
    }

    #[test]
    fn actor_execution_overrides_leave_learning_options_intact() {
        let mut args = vec![
            "--workers".into(),
            "20".into(),
            "--actor-workers".into(),
            "80".into(),
            "--actor-in-flight".into(),
            "2".into(),
        ];
        assert_eq!(
            take_positive_override(&mut args, "--actor-workers").unwrap(),
            Some(80)
        );
        assert_eq!(
            take_positive_override(&mut args, "--actor-in-flight").unwrap(),
            Some(2)
        );
        assert_eq!(args, ["--workers", "20"]);
        for value in ["0", "-1", "x"] {
            assert!(take_positive_override(
                &mut vec!["--actor-workers".into(), value.into()],
                "--actor-workers"
            )
            .is_err());
        }
    }

    #[test]
    fn scheduling_override_preserves_campaign_options() {
        let mut args = vec![
            "--actors".into(),
            "80".into(),
            "--actor-round-size".into(),
            "512".into(),
        ];
        assert_eq!(take_actor_round_size(&mut args).unwrap(), Some(512));
        assert_eq!(args, ["--actors", "80"]);
        assert_eq!(take_actor_round_size(&mut args).unwrap(), None);
    }

    #[test]
    fn scheduling_override_rejects_invalid_paired_sizes() {
        for value in ["0", "3", "bad"] {
            assert!(
                take_actor_round_size(&mut vec!["--actor-round-size".into(), value.into()])
                    .is_err()
            );
        }
        assert!(take_actor_round_size(&mut vec!["--actor-round-size".into()]).is_err());
    }
}

fn seed(base: u64, generation: u64, domain: u64) -> u64 {
    StableRng::new(base ^ generation.wrapping_mul(0x9e3779b97f4a7c15) ^ domain).next_u64()
}

fn main() -> Result<(), BoxError> {
    let mut arguments = std::env::args().skip(1).collect::<Vec<_>>();
    let external_games = take_count_override(&mut arguments, "--external-games")?;
    let mut mixed_self_play = false;
    let mut win_duration_reward = false;
    let mut site_min_win_rate = false;
    for (flag, output) in [
        ("--mixed-self-play", &mut mixed_self_play),
        ("--win-duration-reward", &mut win_duration_reward),
        ("--site-min-win-rate-70", &mut site_min_win_rate),
    ] {
        if let Some(index) = arguments.iter().position(|a| a == flag) {
            *output = arguments
                .get(index + 1)
                .ok_or("missing boolean option")?
                .parse()?;
            arguments.drain(index..index + 2);
        }
    }
    let mut initial_generation = None;
    let mut hold_random = false;
    for flag in ["--initial-generation", "--hold-random"] {
        if let Some(index) = arguments.iter().position(|a| a == flag) {
            let value = arguments
                .get(index + 1)
                .ok_or("missing program option value")?;
            if flag == "--initial-generation" {
                initial_generation = Some(value.parse::<u64>()?);
            } else {
                hold_random = value.parse::<bool>()?;
            }
            arguments.drain(index..index + 2);
        }
    }
    let mut initial_champion = None;
    if let Some(index) = arguments
        .iter()
        .position(|a| a == "--initial-champion-checkpoint")
    {
        initial_champion = Some(fs::canonicalize(
            arguments
                .get(index + 1)
                .ok_or("missing initial champion checkpoint")?,
        )?);
        arguments.drain(index..index + 2);
    }
    let mut durable_every = 5u64;
    let mut champion_only = false;
    let mut continue_inconclusive = false;
    let mut mcts_ceiling = 512usize;
    for flag in ["--continue-inconclusive", "--mcts-ceiling"] {
        if let Some(index) = arguments.iter().position(|a| a == flag) {
            let value = arguments
                .get(index + 1)
                .ok_or("missing selection option value")?;
            if flag == "--continue-inconclusive" {
                continue_inconclusive = value.parse()?;
            } else {
                mcts_ceiling = value.parse()?;
            }
            arguments.drain(index..index + 2);
        }
    }
    if ![32, 512].contains(&mcts_ceiling) {
        return Err("MCTS ceiling must be 32 or legacy 512".into());
    }
    if let Some(index) = arguments.iter().position(|a| a == "--champion-only") {
        champion_only = arguments
            .get(index + 1)
            .ok_or("missing --champion-only boolean")?
            .parse::<bool>()?;
        arguments.drain(index..index + 2);
    }
    let mut disk_replay = false;
    if let Some(index) = arguments.iter().position(|a| a == "--disk-replay") {
        disk_replay = arguments
            .get(index + 1)
            .ok_or("missing --disk-replay boolean")?
            .parse::<bool>()?;
        arguments.drain(index..index + 2);
    }
    // Benchmark-only control: same loop/PSW transport, additional durable checkpoints
    // and service reconstruction between cycles. Default campaign semantics stay intact.
    let mut restart_services_every_cycle = false;
    if let Some(index) = arguments
        .iter()
        .position(|a| a == "--restart-services-every-cycle")
    {
        restart_services_every_cycle = arguments
            .get(index + 1)
            .ok_or("missing --restart-services-every-cycle boolean")?
            .parse::<bool>()?;
        arguments.drain(index..index + 2);
    }
    if let Some(index) = arguments.iter().position(|a| a == "--durable-every") {
        durable_every = arguments
            .get(index + 1)
            .ok_or("missing --durable-every value")?
            .parse()?;
        arguments.drain(index..index + 2);
    }
    if durable_every == 0 {
        return Err("--durable-every must be positive".into());
    }
    for (flag, value) in [
        ("--workers", "20"),
        ("--actors", "80"),
        ("--promotion-every", "10"),
        ("--evaluation-every", "10"),
    ] {
        if !arguments.iter().any(|a| a == flag) {
            arguments.extend([flag.into(), value.into()]);
        }
    }
    if continue_inconclusive && restart_services_every_cycle {
        return Err("learner continuation is not supported by the restart benchmark".into());
    }
    let actor_round_size = take_actor_round_size(&mut arguments)?;
    let actor_workers = take_positive_override(&mut arguments, "--actor-workers")?;
    let actor_in_flight = take_positive_override(&mut arguments, "--actor-in-flight")?.unwrap_or(1);
    let mut options = options::Options::parse_from(arguments.into_iter())?;
    fs::create_dir_all(&options.campaign_directory)?;
    options.campaign_directory = fs::canonicalize(&options.campaign_directory)?;
    options.initial_checkpoint = Some(fs::canonicalize(
        options
            .initial_checkpoint
            .as_ref()
            .ok_or("live campaign requires --initial-checkpoint")?,
    )?);
    let root = &options.campaign_directory;
    let mut identity = serde_json::to_value(&options)?;
    identity
        .as_object_mut()
        .unwrap()
        .remove("target_generation");
    let initial = options.initial_checkpoint.as_ref().unwrap();
    let metadata = read_checkpoint_metadata(initial)?;
    let mut plan = json!({"format":"paisho-live-campaign-v1","durable_every":durable_every,
        "initial_sha256":hex(metadata.content_sha256()),"options":identity});
    if restart_services_every_cycle {
        plan["benchmark_restart_services_every_cycle"] = json!(true);
    }
    if champion_only {
        if options.evaluation_every % options.promotion_every != 0 {
            return Err("champion-only evaluation must coincide with promotion".into());
        }
        plan["selection_policy"] = json!("champion-only-v1");
    }
    if continue_inconclusive {
        if !champion_only {
            return Err("learner continuation requires champion-only selection".into());
        }
        plan["learner_continuation"] = json!("inconclusive-v1");
    }
    if mcts_ceiling == 32 {
        plan["mcts_ceiling"] = json!(32);
        if matches!(
            CurriculumTierV1::parse(&options.opponent)?,
            CurriculumTierV1::Mcts128 | CurriculumTierV1::Mcts512
        ) {
            return Err("initial opponent exceeds MCTS ceiling".into());
        }
    }
    if let Some(champion) = &initial_champion {
        let champion_metadata = read_checkpoint_metadata(champion)?;
        plan["initial_champion"] =
            json!({"path": champion, "sha256": hex(champion_metadata.content_sha256())});
    }
    if let Some(generation) = initial_generation {
        if generation < metadata.generation() {
            return Err("initial generation precedes learner checkpoint".into());
        }
        plan["initial_generation"] = json!(generation);
    }
    if hold_random {
        if options.opponent != "random" {
            return Err("hold-random requires initial Random opponent".into());
        }
        plan["hold_random"] = json!(true);
    }
    if external_games.is_some() && !mixed_self_play {
        return Err("--external-games requires --mixed-self-play true".into());
    }
    if mixed_self_play {
        if let Some(external) = external_games {
            if external > options.actor_target_games || external % 2 != 0 {
                return Err(
                    "external game quota must be even and no larger than actor games".into(),
                );
            }
            plan["mixed_self_play"] =
                json!({"protocol":"retained-quota-shared-round-v3", "external_games": external});
        } else {
            if options.actor_target_games % 4 != 0 || options.actor_maximum_attempts % 4 != 0 {
                return Err(
                    "mixed collection requires game and attempt counts divisible by four".into(),
                );
            }
            plan["mixed_self_play"] = json!("half-retained-games-shared-round-v2");
        }
    }
    if win_duration_reward {
        plan["reward"] = json!("win-duration-0.9-plus-0.1x256-over-256-plus-remaining-v1");
    }
    if site_min_win_rate {
        plan["site_min_win_rate"] = json!(0.7);
    }
    journal::plan(root, plan)?;
    let latest = journal::latest(root)?;
    let resumed = latest.is_some();
    let mut block = latest.unwrap_or(Block {
        generation: initial_generation.unwrap_or(metadata.generation()),
        checkpoint: initial.clone(),
        checkpoint_sha256: hex(metadata.content_sha256()),
        training_step: metadata.training_step(),
        champion: initial_champion.clone().unwrap_or_else(|| initial.clone()),
        tier: CurriculumTierV1::parse(&options.opponent)?,
        games: 0,
        examples: 0,
        attempt_directory: PathBuf::new(),
    });
    if options.target_generation < block.generation {
        return Err("target precedes the durable generation".into());
    }
    // Only blocks created by this live campaign have scheduled measurements to settle.
    let mut assessed = if resumed {
        assessment::settle(
            root,
            &options,
            &block,
            champion_only,
            mcts_ceiling,
            hold_random,
            site_min_win_rate,
        )?
    } else {
        assessment::Assessment {
            generation: block.generation,
            candidate_sha256: block.checkpoint_sha256.clone(),
            selected: block.checkpoint.clone(),
            champion: block.champion.clone(),
            tier: block.tier,
            evaluation: None,
            promotion: None,
        }
    };
    if block.generation == options.target_generation {
        println!(
            "live_resumed=true durable_generation={} work_needed=false",
            block.generation
        );
        return Ok(());
    }
    let configuration =
        |checkpoint: PathBuf, batch_size, legal_action_capacity| ServiceConfiguration {
            executable: options.service.clone(),
            preset: options.network_preset,
            batch_size,
            legal_action_capacity,
            inference_slots: 1,
            optimization: options.optimization,
            seed: options.model_seed,
            checkpoint: Some(checkpoint),
        };
    let learner_checkpoint = if resumed {
        assessment::learner_checkpoint(&assessed, &block, continue_inconclusive)
    } else {
        initial.clone()
    };
    let mut learner = LiveLearner::launch(
        configuration(
            learner_checkpoint,
            options.learner_batch_size,
            options.learner_action_capacity,
        ),
        options.learning_rate,
    )?;
    let mut actors = CapacityInferenceBroker::launch(
        options
            .classes
            .iter()
            .map(|c| ServiceConfiguration {
                inference_slots: actor_in_flight,
                ..configuration(assessed.selected.clone(), c.batch_size, c.capacity)
            })
            .collect(),
        InferenceBrokerConfiguration {
            maximum_batch_wait: Duration::from_micros(options.maximum_batch_wait_microseconds),
            maximum_in_flight_batches: actor_in_flight,
            ..Default::default()
        },
    )?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?;
    let actor_workers = actor_workers.unwrap_or(options.workers);
    let separate_actor_pool = (actor_workers != options.workers)
        .then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(actor_workers)
                .build()
        })
        .transpose()?;
    let actor_pool = separate_actor_pool.as_ref().unwrap_or(&pool);
    let attempt = root
        .join("attempts")
        .join(format!("attempt-{}", journal::unique_id()));
    fs::create_dir_all(&attempt)?;
    let actor_round_size = actor_round_size.unwrap_or(options.actors_per_round);
    journal::write_new(
        &attempt.join("runtime.json"),
        &json!({
            "actor_round_size": actor_round_size,
            "actor_workers": actor_workers, "actor_in_flight": actor_in_flight,
            "workers": options.workers,
            "source_sha256": env!("PAISHO_BUILD_SOURCE_SHA256"),
        }),
    )?;
    let mut games = block.games;
    let mut examples = block.examples;
    let parameters = TerminalPpoParametersV1::with_behavior(
        options.policy_temperature,
        options.uniform_mix,
        options.ppo_clip,
        options.ppo_value_weight,
        options.ppo_entropy_weight,
    )?;
    let first_generation = block
        .generation
        .checked_add(1)
        .ok_or("generation overflow")?;
    for generation in first_generation..=options.target_generation {
        let started = Instant::now();
        let mut timing = serde_json::Map::new();
        let directory = attempt.join(format!("generation-{generation:020}"));
        fs::create_dir(&directory)?;
        journal::status(
            root,
            &json!({"format":"paisho-live-status-v1","phase":"actors","generation":generation,
            "completed_generation":generation-1,"durable_generation":block.generation,"games":games,"examples":examples,
            "tier":assessed.tier,"learner_pid":learner.process.process_id(),"last_evaluation":assessed.evaluation}),
        )?;
        let timer = Instant::now();
        let request_id = learner.next_request_id()?;
        let weights = std::sync::Arc::new(learner.process.export_weights(request_id)?);
        let producer = ReplayDigestV1::from_bytes(weights.content_sha256());
        actors.update_services(move |process| {
            process.import_weights(request_id, &weights).map(|_| ())
        })?;
        timing.insert(
            "weights_seconds".into(),
            json!(timer.elapsed().as_secs_f64()),
        );
        let neutral_start = options
            .start_horizon
            .map(|h| {
                NeutralStartConfigurationV1::new(
                    seed(options.start_seed, generation, 1),
                    h,
                    options.start_source_decision_limit,
                    options.start_source_attempts,
                )
            })
            .transpose()?;
        let opponent = match assessed.tier {
            CurriculumTierV1::Random => LiveActorsOpponent::Random,
            CurriculumTierV1::SiteBotV1 => LiveActorsOpponent::Site,
            CurriculumTierV1::SelfPlay => LiveActorsOpponent::SelfPlay,
            tier => LiveActorsOpponent::Mcts {
                simulations: match tier {
                    CurriculumTierV1::Mcts8 => 8,
                    CurriculumTierV1::Mcts32 => 32,
                    CurriculumTierV1::Mcts128 => 128,
                    _ => 512,
                },
            },
        };
        let timer = Instant::now();
        let actor_configuration = LiveActorsConfiguration {
            actors: actor_round_size,
            target_games: options.actor_target_games,
            maximum_attempts: options.actor_maximum_attempts,
            first_game_id: generation
                .checked_mul(1_000_000_000)
                .ok_or("game id overflow")?,
            decision_soft_limit: options.actor_decision_soft_limit,
            actor_seed: seed(options.actor_seed, generation, 2),
            policy: NetworkPolicy::Sample {
                temperature: options.policy_temperature,
                uniform_mix: options.uniform_mix,
            },
            neutral_start,
            opponent,
        };
        let evaluator = actors.client()?;
        let report = if mixed_self_play {
            collect_live_weighted_games(
                &actor_configuration,
                external_games.unwrap_or(options.actor_target_games / 2),
                actor_pool,
                evaluator.clone(),
                producer,
            )?
        } else {
            collect_live_games(
                &actor_configuration,
                actor_pool,
                evaluator.clone(),
                producer,
            )?
        };
        timing.insert(
            "collection_seconds".into(),
            json!(timer.elapsed().as_secs_f64()),
        );
        timing.insert(
            "start_preparation_seconds".into(),
            json!(report.start_preparation_seconds),
        );
        timing.insert("match_seconds".into(), json!(report.match_seconds));
        let timer = Instant::now();
        let reached = report.target_reached;
        let abort = report.abort_reason.clone();
        let collection_results = results::summarize(&report, producer, generation, assessed.tier);
        journal::write_new(
            &directory.join("collection.json"),
            &json!({"attempts":report.attempts,"target_reached":reached,
            "abort_reason":abort,"workers":report.maximum_workers_observed,"producer":producer.to_string(),
            "results":collection_results,
            "interrupted":report.interrupted,"excluded_pairs":report.excluded_pairs,
            "retained_neural_decisions":report.retained.iter().map(|r| r.game.decisions().iter().filter(|d| d.behavior_value().is_some()).count()).sum::<usize>()}),
        )?;
        let replay_games = report.into_games();
        if replay_games.is_empty() {
            return Err(format!("no terminal replay games: {abort:?}").into());
        }
        let game_count = replay_games.len() as u64;
        let shard = ReplayShardV1::new(generation, replay_games)?;
        shard.write_new(&directory.join("shard.psrbuf"))?;
        let snapshot = ReplaySnapshotV1::new(vec![ReplayShardReferenceV1::from_shard(
            "shard.psrbuf",
            &shard,
        )?])?;
        snapshot.write_new(&directory.join("snapshot.psrsnap"))?;
        timing.insert(
            "replay_publication_seconds".into(),
            json!(timer.elapsed().as_secs_f64()),
        );
        if !reached {
            return Err(
                format!("actor target not reached; partial replays preserved: {abort:?}").into(),
            );
        }
        // Illustrative only: never choose training examples or block PPO on an export.
        // A previously published game remains a real game from this collection number
        // even when volatile cycles are recomputed after an interruption.
        let timer = Instant::now();
        let highlights = root.join("highlights");
        if !highlights
            .join(format!("generation-{generation:020}"))
            .join("best-game.psr")
            .exists()
        {
            let export_started = Instant::now();
            match paisho_train::export_highlighted_game(
                generation,
                producer,
                shard.games(),
                &highlights,
            ) {
                Ok(export) => println!(
                    "highlight_psr={:?} seconds={:.3}",
                    export.psr_path,
                    export_started.elapsed().as_secs_f64()
                ),
                Err(error) => eprintln!("highlight_warning={error}"),
            }
        }
        timing.insert(
            "highlight_seconds".into(),
            json!(timer.elapsed().as_secs_f64()),
        );
        let timer = Instant::now();
        let dataset = if disk_replay {
            ReplayDatasetV1::from_snapshot_for_behavior(&snapshot, &directory, producer)?
        } else {
            ReplayDatasetV1::from_shard_for_behavior(shard, "shard.psrbuf", producer)?
        };
        timing.insert("disk_replay".into(), json!(disk_replay));
        timing.insert(
            "dataset_open_seconds".into(),
            json!(timer.elapsed().as_secs_f64()),
        );
        journal::status(
            root,
            &json!({"format":"paisho-live-status-v1","phase":"learner","generation":generation,
            "completed_generation":generation-1,"durable_generation":block.generation,"games":games+game_count,"examples":examples,
            "tier":assessed.tier,"learner_pid":learner.process.process_id(),"last_evaluation":assessed.evaluation,
            "collection_results":collection_results}),
        )?;
        let timer = Instant::now();
        let metrics = pool.install(|| {
            learner.win_duration_reward = win_duration_reward;
            learner.train_cycle(
                &dataset,
                generation,
                seed(options.sampler_seed, generation, 3),
                options.training_steps_per_generation,
                parameters,
            )
        })?;
        timing.insert(
            "learner_total_seconds".into(),
            json!(timer.elapsed().as_secs_f64()),
        );
        games = games.checked_add(game_count).ok_or("game count overflow")?;
        examples = examples
            .checked_add(metrics.examples)
            .ok_or("example count overflow")?;
        journal::write_new(&directory.join("learning.json"), &metrics)?;
        let durable = restart_services_every_cycle
            || generation % durable_every == 0
            || assessment::due(generation, &options)
            || generation == options.target_generation;
        if durable {
            let timer = Instant::now();
            let checkpoint = directory.join("checkpoint.psckpt");
            let checkpoint_sha256 = learner.checkpoint(&checkpoint)?;
            block = Block {
                generation,
                checkpoint,
                checkpoint_sha256,
                training_step: learner.step,
                champion: assessed.champion.clone(),
                tier: assessed.tier,
                games,
                examples,
                attempt_directory: attempt.clone(),
            };
            journal::commit(root, &block)?;
            timing.insert(
                "checkpoint_seconds".into(),
                json!(timer.elapsed().as_secs_f64()),
            );
            journal::status(
                root,
                &json!({"format":"paisho-live-status-v1","phase":"assessment","generation":generation,
                "completed_generation":generation,"durable_generation":generation,"games":games,"examples":examples,"tier":assessed.tier}),
            )?;
            let previous_evaluation = assessed.evaluation.clone();
            let timer = Instant::now();
            assessed = assessment::settle(
                root,
                &options,
                &block,
                champion_only,
                mcts_ceiling,
                hold_random,
                site_min_win_rate,
            )?;
            timing.insert(
                "assessment_seconds".into(),
                json!(timer.elapsed().as_secs_f64()),
            );
            if assessed.evaluation.is_none() {
                assessed.evaluation = previous_evaluation;
            }
            if assessed.selected != block.checkpoint
                && !restart_services_every_cycle
                && !(continue_inconclusive && assessment::inconclusive(&assessed))
            {
                learner = LiveLearner::launch(
                    configuration(
                        assessed.selected.clone(),
                        options.learner_batch_size,
                        options.learner_action_capacity,
                    ),
                    options.learning_rate,
                )?;
            }
        }
        if restart_services_every_cycle && generation < options.target_generation {
            actors.shutdown()?;
            if !learner.process.shutdown()?.success() {
                return Err("benchmark learner failed during restart shutdown".into());
            }
            learner = LiveLearner::launch(
                configuration(
                    assessed.selected.clone(),
                    options.learner_batch_size,
                    options.learner_action_capacity,
                ),
                options.learning_rate,
            )?;
            actors = CapacityInferenceBroker::launch(
                options
                    .classes
                    .iter()
                    .map(|c| ServiceConfiguration {
                        inference_slots: actor_in_flight,
                        ..configuration(assessed.selected.clone(), c.batch_size, c.capacity)
                    })
                    .collect(),
                InferenceBrokerConfiguration {
                    maximum_batch_wait: Duration::from_micros(
                        options.maximum_batch_wait_microseconds,
                    ),
                    maximum_in_flight_batches: actor_in_flight,
                    ..Default::default()
                },
            )?;
        }
        timing.insert(
            "cycle_seconds".into(),
            json!(started.elapsed().as_secs_f64()),
        );
        journal::write_new(&directory.join("timing.json"), &timing)?;
        journal::status(
            root,
            &json!({"format":"paisho-live-status-v1","phase":"cycle-complete","generation":generation,
            "completed_generation":generation,"durable_generation":block.generation,"games":games,"examples":examples,
            "tier":assessed.tier,"learner_pid":learner.process.process_id(),"last_evaluation":assessed.evaluation,
            "checkpoint":block.checkpoint,"cycle_seconds":started.elapsed().as_secs_f64(),
            "collection_results":collection_results}),
        )?;
        println!("generation={generation} stage=complete durable={durable} durable_generation={} games={games} examples={examples} learner_pid={:?} seconds={:.3}",block.generation,learner.process.process_id(),started.elapsed().as_secs_f64());
    }
    actors.shutdown()?;
    if !learner.process.shutdown()?.success() {
        return Err("learner service failed during shutdown".into());
    }
    println!("live_complete=true durable_generation={}", block.generation);
    Ok(())
}
