use std::path::Path;

use paisho_mpsgraph_client::CheckpointMetadataV2;
use paisho_replay::ReplayDigestV1;

use crate::commit::commit_file_name;
use crate::{
    LearnerCommit, LearnerIdentityV1, LearnerObjectiveV1, LearnerOriginV1, ParentCheckpointV1,
    TerminalPpoLearnerObjectiveV1,
};

use super::{
    hex_digest, invalid, optimization_level, ActorGenerationStageV1, GenerationArchiveError,
    GenerationPlanV1, LearnerGenerationStageV1,
};

pub(super) fn validate_learner_stage_artifacts(
    campaign_root: &Path,
    generation: u64,
    plan: &GenerationPlanV1,
    actor: &ActorGenerationStageV1,
    stage: &LearnerGenerationStageV1,
    metadata: &CheckpointMetadataV2,
) -> Result<(), GenerationArchiveError> {
    validate_checkpoint_metadata(generation, plan, actor, stage, metadata)?;

    let replay_snapshot = parse_digest(&actor.snapshot_sha256, "actor snapshot")?;
    let behavior_producer = parse_digest(&actor.behavior_producer, "behavior producer")?;
    let training_parent = plan.training_parent();
    let parent_digest = parse_digest(&training_parent.sha256, "training parent checkpoint")?;
    let identity = LearnerIdentityV1 {
        replay_snapshot,
        network_preset: metadata.network_preset(),
        optimization: optimization_level(plan.optimization_level)?,
        batch_size: plan.learner.batch_size,
        legal_action_capacity: plan.learner.legal_action_capacity,
        model_seed: plan.model_seed,
        sampler_seed: plan.learner.sampler_seed,
        generation,
        learning_rate: f32::from_bits(plan.learner.learning_rate_bits),
        objective: LearnerObjectiveV1::TerminalPpo(TerminalPpoLearnerObjectiveV1::new(
            behavior_producer,
            Some(parent_digest),
            plan.terminal_ppo_parameters()?,
        )),
    };
    let expected_parent = ParentCheckpointV1::new(
        *parent_digest.as_bytes(),
        training_parent.generation,
        training_parent.training_step,
    );
    let checkpoint_path = stage.checkpoint.path(campaign_root)?;
    let learner_directory = checkpoint_path
        .parent()
        .ok_or_else(|| invalid("learner checkpoint has no parent directory"))?;
    let origin = LearnerOriginV1::read(&LearnerOriginV1::path(learner_directory))
        .map_err(|source| invalid(format!("learner origin verification failed: {source}")))?;
    if origin.identity() != identity || origin.parent() != Some(expected_parent) {
        return Err(invalid(
            "learner origin disagrees with its generation plan or training parent",
        ));
    }

    let commit_path = learner_directory.join(commit_file_name(plan.learner.target_training_step));
    let commit = LearnerCommit::read(&commit_path)
        .map_err(|source| invalid(format!("learner commit verification failed: {source}")))?;
    if commit.identity() != identity
        || commit.starting_training_step() != training_parent.training_step
        || commit.training_step() != plan.learner.target_training_step
        || commit.next_replay_index() != stage.completed_replay_index
        || commit.checkpoint_path(learner_directory) != checkpoint_path
        || commit.checkpoint_sha256()
            != *parse_digest(&stage.checkpoint.sha256, "candidate")?.as_bytes()
    {
        return Err(invalid(
            "learner commit disagrees with its generation stage or training parent",
        ));
    }
    Ok(())
}

pub(super) fn validate_checkpoint_metadata(
    generation: u64,
    plan: &GenerationPlanV1,
    actor: &ActorGenerationStageV1,
    stage: &LearnerGenerationStageV1,
    metadata: &CheckpointMetadataV2,
) -> Result<(), GenerationArchiveError> {
    if stage.checkpoint.generation != generation
        || stage.checkpoint.training_step != plan.learner.target_training_step
        || metadata.replay_index() != stage.completed_replay_index
        || hex_digest(metadata.replay_snapshot_sha256()) != actor.snapshot_sha256
        || metadata.optimization() != optimization_level(plan.optimization_level)?
        || metadata.learning_rate().to_bits() != plan.learner.learning_rate_bits
    {
        return Err(invalid(
            "learner stage checkpoint disagrees with its plan or replay corpus",
        ));
    }
    Ok(())
}

fn parse_digest(
    value: &str,
    field: &'static str,
) -> Result<ReplayDigestV1, GenerationArchiveError> {
    value
        .parse()
        .map_err(|source| invalid(format!("{field} digest is invalid: {source}")))
}
