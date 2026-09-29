#[path = "paisho_generations/options.rs"]
mod options;

use std::error::Error;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use paisho_train::{
    open_or_initialize_generation_campaign, ActorGenerationPlanV1, ActorGenerationStageV1,
    CampaignIdentityV1, CheckpointReferenceV1, CurriculumCampaignArchive, CurriculumTierV1,
    EvaluationCampaignArchive, ExternalFileReferenceV1, GenerationBootstrapConfiguration,
    GenerationCampaignArchive, GenerationInferenceClassV1, GenerationOutcomeV1, GenerationPlanV1,
    LearnerGenerationPlanV1, LearnerGenerationStageV1, NeutralStartGenerationPlanV1,
    PromotionCampaignArchive, PromotionCampaignConclusion, PromotionGenerationPlanV1,
    PromotionGenerationStageV1, PromotionSamplingPolicyV1,
};

use crate::options::Options;

type BoxError = Box<dyn Error + Send + Sync>;
const BUILD_SOURCE_REVISION: &str = env!("PAISHO_BUILD_GIT_REVISION");
const BUILD_SOURCE_DIRTY: &str = env!("PAISHO_BUILD_GIT_DIRTY");
const BUILD_SOURCE_SHA256: &str = env!("PAISHO_BUILD_SOURCE_SHA256");
const GENERATION_IDENTIFIER_STRIDE: u64 = 1_000_000_000;

fn main() -> Result<(), BoxError> {
    if let Some(directory) = verify_curriculum_directory()? {
        let curriculum = CurriculumCampaignArchive::open_existing(&directory)?;
        curriculum.verify_all_evidence()?;
        print_curriculum_state(&curriculum)?;
        println!("verified_curriculum_archive={}", directory.display());
        return Ok(());
    }
    let options = Options::parse()?;
    std::fs::create_dir_all(&options.campaign_directory)?;
    let campaign_directory = std::fs::canonicalize(&options.campaign_directory)?;
    let existing_identity = campaign_directory.join("campaign.json").is_file()
        || campaign_directory.join("genesis/genesis.json").is_file();
    let bootstrap_service = if existing_identity {
        None
    } else {
        Some(ExternalFileReferenceV1::from_file(&options.service)?)
    };
    let archive = open_or_initialize_generation_campaign(&GenerationBootstrapConfiguration {
        campaign_directory,
        service_executable: bootstrap_service
            .as_ref()
            .map(ExternalFileReferenceV1::path)
            .unwrap_or_else(|| options.service.clone()),
        initial_checkpoint: options.initial_checkpoint.clone(),
        network_preset: options.network_preset,
        optimization: options.optimization,
        batch_size: options.learner_batch_size,
        legal_action_capacity: options.learner_action_capacity,
        model_seed: options.model_seed,
        learning_rate: options.learning_rate,
        source_revision: BUILD_SOURCE_REVISION.to_owned(),
        source_dirty: BUILD_SOURCE_DIRTY == "true",
        source_sha256: BUILD_SOURCE_SHA256.to_owned(),
        service_sha256: bootstrap_service
            .as_ref()
            .map(|service| service.sha256.clone())
            .unwrap_or_else(|| "00".repeat(32)),
    })?;
    let curriculum = options
        .curriculum_directory
        .as_ref()
        .map(|directory| {
            CurriculumCampaignArchive::open_or_create(
                directory,
                &archive,
                CurriculumTierV1::parse(&options.opponent)?,
            )
        })
        .transpose()?;
    run_to_target(&archive, curriculum.as_ref(), &options)?;
    Ok(())
}

fn verify_curriculum_directory() -> Result<Option<PathBuf>, BoxError> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .map_or(true, |argument| argument != "--verify-curriculum")
    {
        return Ok(None);
    }
    if arguments.len() != 2 {
        return Err("usage: paisho-generations --verify-curriculum DIRECTORY".into());
    }
    Ok(Some(PathBuf::from(&arguments[1])))
}

fn run_to_target(
    archive: &GenerationCampaignArchive,
    curriculum: Option<&CurriculumCampaignArchive>,
    options: &Options,
) -> Result<(), BoxError> {
    loop {
        let mut chain = archive.load_chain()?;
        if let Some(curriculum) = curriculum {
            settle_completed_curriculum_decisions(archive, curriculum, options, &chain)?;
            chain = archive.load_chain()?;
        }
        if chain.next_generation > options.target_generation {
            if let Some(curriculum) = curriculum {
                print_curriculum_state(curriculum)?;
            }
            print_roles(archive, &chain.roles, chain.completed_generations)?;
            return Ok(());
        }
        let generation = chain.next_generation;
        let opponent = match curriculum {
            Some(curriculum) => curriculum.load_state()?.current_tier.training_opponent(),
            None => &options.opponent,
        };
        let plan = if chain.in_progress_generation == Some(generation) {
            let plan = archive.read_plan(generation)?;
            if plan.actor.opponent != opponent {
                return Err(format!(
                    "generation {generation} was sealed for opponent {}, while the curriculum requires {opponent}",
                    plan.actor.opponent
                )
                .into());
            }
            println!(
                "generation={generation} stage=planned resumed=true opponent={}",
                plan.actor.opponent
            );
            plan
        } else {
            let plan = build_plan(
                archive.identity(),
                options,
                ExternalFileReferenceV1::from_file(&options.service)?,
                generation,
                chain.roles.champion,
                chain.roles.training,
                opponent,
            )?;
            archive.establish_plan(&plan)?;
            println!(
                "generation={generation} stage=planned resumed=false opponent={}",
                plan.actor.opponent
            );
            plan
        };
        execute_generation(archive, &plan)?;
    }
}

fn build_plan(
    campaign: &CampaignIdentityV1,
    options: &Options,
    service: ExternalFileReferenceV1,
    generation: u64,
    parent_champion: CheckpointReferenceV1,
    training_parent: CheckpointReferenceV1,
    opponent: &str,
) -> Result<GenerationPlanV1, BoxError> {
    if generation > options.target_generation {
        return Err("generation plan exceeds the requested target".into());
    }
    let identifier_base = generation
        .checked_mul(GENERATION_IDENTIFIER_STRIDE)
        .ok_or("generation identifier range overflows")?;
    let target_training_step = training_parent
        .training_step
        .checked_add(options.training_steps_per_generation)
        .ok_or("generation training target overflows")?;
    let plan = GenerationPlanV1 {
        generation,
        promotion_due: generation % options.promotion_every == 0,
        curriculum_evaluation_due: generation % options.evaluation_every == 0,
        parent_champion,
        training_parent: Some(training_parent),
        source_revision: BUILD_SOURCE_REVISION.to_owned(),
        source_dirty: BUILD_SOURCE_DIRTY == "true",
        source_sha256: BUILD_SOURCE_SHA256.to_owned(),
        service,
        actor_executable: ExternalFileReferenceV1::from_file(&options.actor_executable)?,
        learner_executable: ExternalFileReferenceV1::from_file(&options.learner_executable)?,
        promotion_executable: ExternalFileReferenceV1::from_file(&options.promotion_executable)?,
        network_preset: campaign.network_preset.clone(),
        optimization_level: optimization_level(options.optimization),
        inference_classes: options
            .classes
            .iter()
            .map(|class| GenerationInferenceClassV1 {
                legal_action_capacity: class.capacity,
                batch_size: class.batch_size,
            })
            .collect(),
        workers: options.workers,
        maximum_batch_wait_microseconds: options.maximum_batch_wait_microseconds,
        model_seed: options.model_seed,
        actor: ActorGenerationPlanV1 {
            opponent: opponent.to_owned(),
            target_games: options.actor_target_games,
            maximum_attempts: options.actor_maximum_attempts,
            actors_per_round: options.actors_per_round,
            shard_index: generation,
            first_game_id: identifier_base,
            decision_soft_limit: options.actor_decision_soft_limit,
            actor_seed: generation_seed(options.actor_seed, generation, 0x0041_4354_4f52),
            policy_temperature_bits: options.policy_temperature.to_bits(),
            uniform_mix_bits: options.uniform_mix.to_bits(),
            neutral_start: options.start_horizon.map(|target_remaining_decisions| {
                NeutralStartGenerationPlanV1 {
                    target_remaining_decisions,
                    seed: generation_seed(options.start_seed, generation, 0x0053_5441_5254),
                    source_decision_limit: options.start_source_decision_limit,
                    maximum_source_attempts: options.start_source_attempts,
                }
            }),
        },
        learner: LearnerGenerationPlanV1 {
            batch_size: options.learner_batch_size,
            legal_action_capacity: options.learner_action_capacity,
            sampler_seed: generation_seed(options.sampler_seed, generation, 0x5341_4d50_4c45),
            learning_rate_bits: options.learning_rate.to_bits(),
            target_training_step,
            checkpoint_interval: options.checkpoint_interval,
            ppo_clip_bits: options.ppo_clip.to_bits(),
            ppo_value_weight_bits: options.ppo_value_weight.to_bits(),
            ppo_entropy_weight_bits: options.ppo_entropy_weight.to_bits(),
        },
        promotion: PromotionGenerationPlanV1 {
            pairs_per_batch: options.promotion_pairs_per_batch,
            maximum_attempted_pairs: options.promotion_maximum_attempted_pairs,
            maximum_eligible_pairs: options.promotion_maximum_eligible_pairs,
            first_pair_id: identifier_base,
            decision_soft_limit: options.promotion_decision_soft_limit,
            neutral_start: options
                .promotion_start_horizon
                .map(|target_remaining_decisions| NeutralStartGenerationPlanV1 {
                    target_remaining_decisions,
                    seed: generation_seed(
                        options.promotion_start_seed,
                        generation,
                        0x5052_4f4d_4f54_4553,
                    ),
                    source_decision_limit: options.promotion_start_source_decision_limit,
                    maximum_source_attempts: options.promotion_start_source_attempts,
                }),
            sampling_policy: options
                .promotion_sampling_temperature
                .map(|temperature| {
                    PromotionSamplingPolicyV1::new(
                        temperature,
                        options.promotion_sampling_uniform_mix,
                    )
                })
                .transpose()?,
            elo0_bits: options.elo0.to_bits(),
            elo1_bits: options.elo1.to_bits(),
            alpha_bits: options.alpha.to_bits(),
            beta_bits: options.beta.to_bits(),
        },
    };
    plan.validate()?;
    Ok(plan)
}

fn execute_generation(
    archive: &GenerationCampaignArchive,
    plan: &GenerationPlanV1,
) -> Result<(), BoxError> {
    plan.parent_champion
        .verify(archive.root(), &plan.network_preset)?;
    plan.training_parent()
        .verify(archive.root(), &plan.network_preset)?;
    let actor = match archive.read_actor_stage(plan.generation)? {
        Some(stage) => {
            archive.verify_actor_stage_integrity(plan.generation, &stage)?;
            println!("generation={} stage=actors resumed=true", plan.generation);
            stage
        }
        None => {
            plan.service.verify()?;
            plan.actor_executable.verify()?;
            let stage = run_actors(archive, plan)?;
            archive.publish_actor_stage(plan.generation, &stage)?;
            println!("generation={} stage=actors committed=true", plan.generation);
            stage
        }
    };
    let learner = match archive.read_learner_stage(plan.generation)? {
        Some(stage) => {
            archive.verify_learner_stage(plan.generation, &stage)?;
            println!("generation={} stage=learner resumed=true", plan.generation);
            stage
        }
        None => {
            plan.service.verify()?;
            plan.learner_executable.verify()?;
            let stage = run_learner(archive, plan, &actor)?;
            archive.publish_learner_stage(plan.generation, &stage)?;
            println!(
                "generation={} stage=learner committed=true",
                plan.generation
            );
            stage
        }
    };
    let promotion = if !plan.promotion_due {
        println!(
            "generation={} stage=promotion deferred=true",
            plan.generation
        );
        None
    } else {
        Some(match archive.read_promotion_stage(plan.generation)? {
            Some(stage) => {
                archive.verify_promotion_stage_integrity(plan.generation, &stage)?;
                println!(
                    "generation={} stage=promotion resumed=true conclusion={:?}",
                    plan.generation, stage.conclusion
                );
                stage
            }
            None => {
                plan.service.verify()?;
                plan.promotion_executable.verify()?;
                let stage = run_promotion(archive, plan, &learner)?;
                archive.publish_promotion_stage(plan.generation, &stage)?;
                println!(
                    "generation={} stage=promotion committed=true conclusion={:?}",
                    plan.generation, stage.conclusion
                );
                stage
            }
        })
    };
    let conclusion = promotion.as_ref().map(|stage| stage.conclusion);
    let champion_after = if conclusion == Some(PromotionCampaignConclusion::PromoteCandidate) {
        learner.checkpoint.clone()
    } else {
        plan.parent_champion.clone()
    };
    let outcome = GenerationOutcomeV1 {
        generation: plan.generation,
        parent_champion: plan.parent_champion.clone(),
        candidate: learner.checkpoint,
        promotion: conclusion,
        champion_after,
    };
    archive.publish_outcome(&outcome)?;
    println!(
        "generation={} stage=complete conclusion={:?} champion_sha256={}",
        plan.generation, outcome.promotion, outcome.champion_after.sha256
    );
    Ok(())
}

fn settle_completed_curriculum_decisions(
    archive: &GenerationCampaignArchive,
    curriculum: &CurriculumCampaignArchive,
    options: &Options,
    chain: &paisho_train::GenerationCampaignChainV1,
) -> Result<(), BoxError> {
    let completed_through = chain
        .in_progress_generation
        .unwrap_or(chain.next_generation)
        .checked_sub(1)
        .ok_or("generation chain has no completed baseline")?;
    loop {
        let state = curriculum.load_state()?;
        if state.last_decided_generation >= completed_through {
            return Ok(());
        }
        let generation = state
            .last_decided_generation
            .checked_add(1)
            .ok_or("curriculum generation overflows")?;
        let outcome = archive
            .read_outcome(generation)?
            .ok_or("curriculum cannot assess an unfinished generation")?;
        let checkpoint = if outcome.promotion == Some(PromotionCampaignConclusion::RejectCandidate)
        {
            outcome.champion_after
        } else {
            outcome.candidate
        };
        if state.current_tier == CurriculumTierV1::SelfPlay {
            let decision = curriculum.publish_self_play_decision(generation, checkpoint)?;
            print_curriculum_decision(&decision);
            continue;
        }
        if !archive.read_plan(generation)?.curriculum_evaluation_due {
            let decision =
                curriculum.publish_deferred_evaluation_decision(generation, checkpoint)?;
            print_curriculum_decision(&decision);
            continue;
        }
        let protocol_sha256 = curriculum_evaluation_protocol_sha256(
            archive,
            options,
            generation,
            state.current_tier,
            &checkpoint,
        )?;
        let evaluation_directory = curriculum.evaluation_directory(
            generation,
            state.current_tier,
            &checkpoint.sha256,
            &protocol_sha256,
        )?;
        run_curriculum_evaluation(
            archive,
            options,
            generation,
            state.current_tier,
            &checkpoint,
            &evaluation_directory,
        )?;
        let evaluation = EvaluationCampaignArchive::open_existing(&evaluation_directory)?;
        let decision =
            curriculum.publish_evaluation_decision(generation, checkpoint, &evaluation)?;
        print_curriculum_decision(&decision);
    }
}

fn curriculum_evaluation_protocol_sha256(
    archive: &GenerationCampaignArchive,
    options: &Options,
    generation: u64,
    tier: CurriculumTierV1,
    checkpoint: &CheckpointReferenceV1,
) -> Result<String, BoxError> {
    let plan = archive.read_plan(generation)?;
    let executable = ExternalFileReferenceV1::from_file(&options.evaluation_executable)?;
    let start_seed = generation_seed(
        options.curriculum_start_seed,
        generation,
        0x4355_5252_5354_4152,
    );
    let model_seed = generation_seed(plan.model_seed, generation, 0x4556_414c_5541_5445);
    let first_pair_id = generation
        .checked_mul(GENERATION_IDENTIFIER_STRIDE)
        .and_then(|value| value.checked_add(GENERATION_IDENTIFIER_STRIDE / 2))
        .ok_or("curriculum evaluation identifier range overflows")?;
    let sampling = options
        .curriculum_sampling_temperature
        .map(|temperature| {
            format!(
                "sample:{:08x}:{:08x}",
                temperature.to_bits(),
                options.curriculum_sampling_uniform_mix.to_bits()
            )
        })
        .unwrap_or_else(|| "argmax".to_owned());
    let start = options
        .curriculum_start_horizon
        .map(|horizon| {
            format!(
                "neutral:{horizon}:{start_seed}:{}:{}",
                options.curriculum_start_source_decision_limit,
                options.curriculum_start_source_attempts
            )
        })
        .unwrap_or_else(|| "standard".to_owned());
    let descriptor = format!(
        "generation={generation};evaluation_executable={};service={};checkpoint={};opponent={};preset={};level={};classes={};workers={};pairs_per_batch={};max_attempted={};max_eligible={};first_pair_id={first_pair_id};decision_limit={};wait_us={};model_seed={model_seed};start={start};policy={sampling};elo={:016x}:{:016x}:{:016x};errors={:016x}:{:016x}",
        executable.sha256,
        plan.service.sha256,
        checkpoint.sha256,
        tier.training_opponent(),
        plan.network_preset,
        plan.optimization_level,
        class_argument(&plan.inference_classes),
        plan.workers,
        options.curriculum_pairs_per_batch,
        options.curriculum_maximum_attempted_pairs,
        options.curriculum_maximum_eligible_pairs,
        options.curriculum_decision_soft_limit,
        plan.maximum_batch_wait_microseconds,
        options.curriculum_lower_elo.to_bits(),
        options.curriculum_center_elo.to_bits(),
        options.curriculum_upper_elo.to_bits(),
        options.curriculum_alpha.to_bits(),
        options.curriculum_beta.to_bits(),
    );
    Ok(paisho_train::sha256_text(&descriptor))
}

fn run_curriculum_evaluation(
    archive: &GenerationCampaignArchive,
    options: &Options,
    generation: u64,
    tier: CurriculumTierV1,
    checkpoint: &CheckpointReferenceV1,
    output_directory: &Path,
) -> Result<(), BoxError> {
    let plan = archive.read_plan(generation)?;
    plan.service.verify()?;
    let identifier_base = generation
        .checked_mul(GENERATION_IDENTIFIER_STRIDE)
        .and_then(|value| value.checked_add(GENERATION_IDENTIFIER_STRIDE / 2))
        .ok_or("curriculum evaluation identifier range overflows")?;
    let mut command = Command::new(&options.evaluation_executable);
    command
        .arg("--output-dir")
        .arg(output_directory)
        .arg("--service")
        .arg(plan.service.path())
        .arg("--candidate-checkpoint")
        .arg(checkpoint.path(archive.root())?)
        .arg("--opponent")
        .arg(tier.training_opponent())
        .arg("--preset")
        .arg(&plan.network_preset)
        .arg("--level")
        .arg(plan.optimization_level.to_string())
        .arg("--classes")
        .arg(class_argument(&plan.inference_classes))
        .arg("--workers")
        .arg(plan.workers.to_string())
        .arg("--pairs-per-batch")
        .arg(options.curriculum_pairs_per_batch.to_string())
        .arg("--max-attempted-pairs")
        .arg(options.curriculum_maximum_attempted_pairs.to_string())
        .arg("--max-eligible-pairs")
        .arg(options.curriculum_maximum_eligible_pairs.to_string())
        .arg("--first-pair-id")
        .arg(identifier_base.to_string())
        .arg("--decision-limit")
        .arg(options.curriculum_decision_soft_limit.to_string())
        .arg("--wait-us")
        .arg(plan.maximum_batch_wait_microseconds.to_string())
        .arg("--model-seed")
        .arg(generation_seed(plan.model_seed, generation, 0x4556_414c_5541_5445).to_string())
        .arg("--sampling-temperature")
        .arg(
            options
                .curriculum_sampling_temperature
                .unwrap_or(0.0)
                .to_string(),
        )
        .arg("--sampling-uniform-mix")
        .arg(options.curriculum_sampling_uniform_mix.to_string())
        .arg("--lower-elo")
        .arg(options.curriculum_lower_elo.to_string())
        .arg("--elo0")
        .arg(options.curriculum_center_elo.to_string())
        .arg("--elo1")
        .arg(options.curriculum_upper_elo.to_string())
        .arg("--alpha")
        .arg(options.curriculum_alpha.to_string())
        .arg("--beta")
        .arg(options.curriculum_beta.to_string());
    if let Some(horizon) = options.curriculum_start_horizon {
        command
            .arg("--start-horizon")
            .arg(horizon.to_string())
            .arg("--start-seed")
            .arg(
                generation_seed(
                    options.curriculum_start_seed,
                    generation,
                    0x4355_5252_5354_4152,
                )
                .to_string(),
            )
            .arg("--start-source-limit")
            .arg(options.curriculum_start_source_decision_limit.to_string())
            .arg("--start-source-attempts")
            .arg(options.curriculum_start_source_attempts.to_string());
    } else {
        command.arg("--start-horizon").arg("0");
    }
    run_command(command, "curriculum evaluation")?;
    Ok(())
}

fn print_curriculum_decision(decision: &paisho_train::CurriculumDecisionV1) {
    println!(
        "curriculum_generation={} tier_before={} action={:?} tier_after={} reason={:?}",
        decision.generation,
        decision.tier_before.label(),
        decision.assessment.action,
        decision.assessment.tier_after.label(),
        decision.assessment.reason,
    );
}

fn print_curriculum_state(curriculum: &CurriculumCampaignArchive) -> Result<(), BoxError> {
    let state = curriculum.load_state()?;
    println!("curriculum={}", curriculum.root().display());
    println!("curriculum_decisions={}", state.decisions);
    println!(
        "curriculum_last_generation={}",
        state.last_decided_generation
    );
    println!("curriculum_tier={}", state.current_tier.label());
    Ok(())
}

fn run_actors(
    archive: &GenerationCampaignArchive,
    plan: &GenerationPlanV1,
) -> Result<ActorGenerationStageV1, BoxError> {
    let generation_directory = archive.generation_directory(plan.generation);
    let output_directory = generation_directory.join("actors");
    let checkpoint = plan.training_parent().path(archive.root())?;
    let mut command = Command::new(plan.actor_executable.path());
    command
        .arg("--output-dir")
        .arg(&output_directory)
        .arg("--service")
        .arg(plan.service.path())
        .arg("--checkpoint")
        .arg(checkpoint)
        .arg("--preset")
        .arg(&plan.network_preset)
        .arg("--level")
        .arg(plan.optimization_level.to_string())
        .arg("--classes")
        .arg(class_argument(&plan.inference_classes))
        .arg("--workers")
        .arg(plan.workers.to_string())
        .arg("--actors")
        .arg(plan.actor.actors_per_round.to_string())
        .arg("--target-games")
        .arg(plan.actor.target_games.to_string())
        .arg("--max-attempts")
        .arg(plan.actor.maximum_attempts.to_string())
        .arg("--shard-index")
        .arg(plan.actor.shard_index.to_string())
        .arg("--first-game-id")
        .arg(plan.actor.first_game_id.to_string())
        .arg("--decision-limit")
        .arg(plan.actor.decision_soft_limit.to_string())
        .arg("--wait-us")
        .arg(plan.maximum_batch_wait_microseconds.to_string())
        .arg("--model-seed")
        .arg(plan.model_seed.to_string())
        .arg("--actor-seed")
        .arg(plan.actor.actor_seed.to_string())
        .arg("--temperature")
        .arg(f32::from_bits(plan.actor.policy_temperature_bits).to_string())
        .arg("--uniform-mix")
        .arg(f32::from_bits(plan.actor.uniform_mix_bits).to_string())
        .arg("--opponent")
        .arg(&plan.actor.opponent);
    if let Some(start) = &plan.actor.neutral_start {
        command
            .arg("--start-horizon")
            .arg(start.target_remaining_decisions.to_string())
            .arg("--start-seed")
            .arg(start.seed.to_string())
            .arg("--start-source-limit")
            .arg(start.source_decision_limit.to_string())
            .arg("--start-source-attempts")
            .arg(start.maximum_source_attempts.to_string());
    }
    let output = run_command(command, "actors")?;
    let run_directory = required_path(&output, "run_directory")?;
    let snapshot = required_path(&output, "snapshot")?;
    let stage = ActorGenerationStageV1 {
        run_directory: archive.artifact_relative_path(&run_directory)?,
        snapshot: archive.artifact_relative_path(&snapshot)?,
        snapshot_sha256: required_value(&output, "snapshot_sha256")?.to_owned(),
        behavior_producer: required_value(&output, "candidate")?.to_owned(),
        terminal_games: parse_value(&output, "terminal_games")?,
        training_examples: parse_value(&output, "training_examples")?,
    };
    Ok(stage)
}

fn run_learner(
    archive: &GenerationCampaignArchive,
    plan: &GenerationPlanV1,
    actor: &ActorGenerationStageV1,
) -> Result<LearnerGenerationStageV1, BoxError> {
    let generation_directory = archive.generation_directory(plan.generation);
    let replay_directory = archive.root().join(&actor.run_directory);
    let snapshot = archive.root().join(&actor.snapshot);
    let parent = plan.training_parent().path(archive.root())?;
    let mut command = Command::new(plan.learner_executable.path());
    command
        .arg("--service")
        .arg(plan.service.path())
        .arg("--snapshot")
        .arg(snapshot)
        .arg("--replay-dir")
        .arg(replay_directory)
        .arg("--run-dir")
        .arg(generation_directory.join("learner"))
        .arg("--initial-checkpoint")
        .arg(&parent)
        .arg("--preset")
        .arg(&plan.network_preset)
        .arg("--level")
        .arg(plan.optimization_level.to_string())
        .arg("--batch")
        .arg(plan.learner.batch_size.to_string())
        .arg("--actions")
        .arg(plan.learner.legal_action_capacity.to_string())
        .arg("--model-seed")
        .arg(plan.model_seed.to_string())
        .arg("--sampler-seed")
        .arg(plan.learner.sampler_seed.to_string())
        .arg("--generation")
        .arg(plan.generation.to_string())
        .arg("--learning-rate")
        .arg(f32::from_bits(plan.learner.learning_rate_bits).to_string())
        .arg("--objective")
        .arg("terminal-ppo")
        .arg("--behavior-producer")
        .arg(&actor.behavior_producer)
        .arg("--actor-checkpoint")
        .arg(parent)
        .arg("--policy-temperature")
        .arg(f32::from_bits(plan.actor.policy_temperature_bits).to_string())
        .arg("--uniform-mix")
        .arg(f32::from_bits(plan.actor.uniform_mix_bits).to_string())
        .arg("--ppo-clip")
        .arg(f32::from_bits(plan.learner.ppo_clip_bits).to_string())
        .arg("--ppo-value-weight")
        .arg(f32::from_bits(plan.learner.ppo_value_weight_bits).to_string())
        .arg("--ppo-entropy-weight")
        .arg(f32::from_bits(plan.learner.ppo_entropy_weight_bits).to_string())
        .arg("--target-step")
        .arg(plan.learner.target_training_step.to_string())
        .arg("--checkpoint-every")
        .arg(plan.learner.checkpoint_interval.to_string());
    let output = run_command(command, "learner")?;
    let checkpoint_path = required_path(&output, "latest_checkpoint")?;
    let stage = LearnerGenerationStageV1 {
        checkpoint: CheckpointReferenceV1::from_checkpoint(archive.root(), &checkpoint_path)?,
        completed_replay_index: parse_value(&output, "completed_replay_index")?,
    };
    Ok(stage)
}

fn run_promotion(
    archive: &GenerationCampaignArchive,
    plan: &GenerationPlanV1,
    learner: &LearnerGenerationStageV1,
) -> Result<PromotionGenerationStageV1, BoxError> {
    let promotion_directory = archive
        .generation_directory(plan.generation)
        .join("promotion");
    let mut command = Command::new(plan.promotion_executable.path());
    command
        .arg("--output-dir")
        .arg(&promotion_directory)
        .arg("--service")
        .arg(plan.service.path())
        .arg("--candidate-checkpoint")
        .arg(learner.checkpoint.path(archive.root())?)
        .arg("--champion-checkpoint")
        .arg(plan.parent_champion.path(archive.root())?)
        .arg("--preset")
        .arg(&plan.network_preset)
        .arg("--level")
        .arg(plan.optimization_level.to_string())
        .arg("--classes")
        .arg(class_argument(&plan.inference_classes))
        .arg("--workers")
        .arg(plan.workers.to_string())
        .arg("--pairs-per-batch")
        .arg(plan.promotion.pairs_per_batch.to_string())
        .arg("--max-attempted-pairs")
        .arg(plan.promotion.maximum_attempted_pairs.to_string())
        .arg("--max-eligible-pairs")
        .arg(plan.promotion.maximum_eligible_pairs.to_string())
        .arg("--first-pair-id")
        .arg(plan.promotion.first_pair_id.to_string())
        .arg("--decision-limit")
        .arg(plan.promotion.decision_soft_limit.to_string())
        .arg("--wait-us")
        .arg(plan.maximum_batch_wait_microseconds.to_string())
        .arg("--model-seed")
        .arg(plan.model_seed.to_string())
        .arg("--elo0")
        .arg(f64::from_bits(plan.promotion.elo0_bits).to_string())
        .arg("--elo1")
        .arg(f64::from_bits(plan.promotion.elo1_bits).to_string())
        .arg("--alpha")
        .arg(f64::from_bits(plan.promotion.alpha_bits).to_string())
        .arg("--beta")
        .arg(f64::from_bits(plan.promotion.beta_bits).to_string());
    if let Some(start) = &plan.promotion.neutral_start {
        command
            .arg("--start-horizon")
            .arg(start.target_remaining_decisions.to_string())
            .arg("--start-seed")
            .arg(start.seed.to_string())
            .arg("--start-source-limit")
            .arg(start.source_decision_limit.to_string())
            .arg("--start-source-attempts")
            .arg(start.maximum_source_attempts.to_string());
    }
    if let Some(policy) = plan.promotion.sampling_policy {
        command
            .arg("--sampling-temperature")
            .arg(f32::from_bits(policy.temperature_bits).to_string())
            .arg("--sampling-uniform-mix")
            .arg(f32::from_bits(policy.uniform_mix_bits).to_string());
    }
    let output = run_command(command, "promotion")?;
    let conclusion = parse_conclusion(required_value(&output, "conclusion")?)?;
    let archive_path = required_path(&output, "archive")?;
    let promotion = PromotionCampaignArchive::open_existing(&archive_path)?;
    let stage = PromotionGenerationStageV1 {
        archive_directory: archive.artifact_relative_path(&archive_path)?,
        conclusion,
        promotion_source_sha256: promotion.identity().source_sha256.clone(),
    };
    Ok(stage)
}

fn run_command(mut command: Command, label: &str) -> Result<String, BoxError> {
    command.stdin(Stdio::null()).stdout(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or("child process has no stdout pipe")?;
    let mut collected = String::new();
    for line in BufReader::new(stdout).lines() {
        let line = line?;
        println!("[{label}] {line}");
        collected.push_str(&line);
        collected.push('\n');
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(format!("{label} process exited with {status}").into());
    }
    Ok(collected)
}

fn required_value<'output>(output: &'output str, key: &str) -> Result<&'output str, BoxError> {
    let prefix = format!("{key}=");
    let mut values = output.lines().filter_map(|line| line.strip_prefix(&prefix));
    let value = values
        .next()
        .ok_or_else(|| format!("child output is missing {key}"))?;
    if values.next().is_some() || value.is_empty() {
        return Err(format!("child output has an ambiguous {key}").into());
    }
    Ok(value)
}

fn required_path(output: &str, key: &str) -> Result<PathBuf, BoxError> {
    Ok(PathBuf::from(required_value(output, key)?))
}

fn parse_value<T>(output: &str, key: &str) -> Result<T, BoxError>
where
    T: core::str::FromStr,
    T::Err: Error + Send + Sync + 'static,
{
    required_value(output, key)?
        .parse()
        .map_err(|source| Box::new(source) as BoxError)
}

fn parse_conclusion(value: &str) -> Result<PromotionCampaignConclusion, BoxError> {
    match value {
        "PromoteCandidate" => Ok(PromotionCampaignConclusion::PromoteCandidate),
        "RejectCandidate" => Ok(PromotionCampaignConclusion::RejectCandidate),
        "InconclusiveMaximumEligiblePairs" => {
            Ok(PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs)
        }
        "InconclusiveMaximumAttemptedPairs" => {
            Ok(PromotionCampaignConclusion::InconclusiveMaximumAttemptedPairs)
        }
        _ => Err(format!("unknown promotion conclusion {value}").into()),
    }
}

fn class_argument(classes: &[GenerationInferenceClassV1]) -> String {
    classes
        .iter()
        .map(|class| format!("{}:{}", class.legal_action_capacity, class.batch_size))
        .collect::<Vec<_>>()
        .join(",")
}

fn generation_seed(base: u64, generation: u64, family: u64) -> u64 {
    let mut value = base ^ generation.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ family.rotate_left(17);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn print_roles(
    archive: &GenerationCampaignArchive,
    roles: &paisho_train::GenerationRolesV1,
    completed_generations: u64,
) -> Result<(), BoxError> {
    println!("campaign={}", archive.root().display());
    println!("completed_generations={completed_generations}");
    println!("champion_sha256={}", roles.champion.sha256);
    println!(
        "champion_checkpoint={}",
        roles.champion.path(archive.root())?.display()
    );
    println!("training_sha256={}", roles.training.sha256);
    println!(
        "training_checkpoint={}",
        roles.training.path(archive.root())?.display()
    );
    match &roles.latest {
        Some(latest) => {
            println!("latest_sha256={}", latest.sha256);
            println!(
                "latest_checkpoint={}",
                latest.path(archive.root())?.display()
            );
        }
        None => println!("latest_checkpoint=NONE"),
    }
    println!("best_sha256={}", roles.best.sha256);
    println!("milestone_checkpoint=NONE");
    Ok(())
}

const fn optimization_level(level: paisho_mpsgraph_client::OptimizationLevel) -> u8 {
    match level {
        paisho_mpsgraph_client::OptimizationLevel::Level0 => 0,
        paisho_mpsgraph_client::OptimizationLevel::Level1 => 1,
    }
}

#[cfg(test)]
#[path = "paisho_generations/tests.rs"]
mod tests;
