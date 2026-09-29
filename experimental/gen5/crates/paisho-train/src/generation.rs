use core::fmt;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use paisho_core::RuleProfileId;
use paisho_model::TerminalPpoParametersV1;
use paisho_mpsgraph_client::{
    read_checkpoint_metadata, CheckpointMetadataError, CheckpointMetadataV2, NetworkPreset,
    OptimizationLevel,
};
use paisho_rating::{PromotionSprtConfig, PromotionSprtError};
use paisho_replay::{ReplayDigestV1, ReplayDigestV1Error, ReplaySnapshotV1, ReplaySnapshotV1Error};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    atomic_file, PromotionArchiveError, PromotionCampaignArchive, PromotionCampaignConclusion,
    PromotionNeutralStartV1, PromotionSamplingPolicyV1,
};

mod verification;

use verification::validate_learner_stage_artifacts;

const CAMPAIGN_DOCUMENT: &str = "paisho-generation-campaign-v1";
const PLAN_DOCUMENT: &str = "paisho-generation-plan-v1";
const ACTOR_STAGE_DOCUMENT: &str = "paisho-generation-actor-stage-v1";
const LEARNER_STAGE_DOCUMENT: &str = "paisho-generation-learner-stage-v1";
const PROMOTION_STAGE_DOCUMENT: &str = "paisho-generation-promotion-stage-v1";
const OUTCOME_DOCUMENT: &str = "paisho-generation-outcome-v1";
const CAMPAIGN_FILE: &str = "campaign.json";
const GENERATIONS_DIRECTORY: &str = "generations";
const PLAN_FILE: &str = "plan.json";
const ACTOR_STAGE_FILE: &str = "actor-stage.json";
const LEARNER_STAGE_FILE: &str = "learner-stage.json";
const PROMOTION_STAGE_FILE: &str = "promotion-stage.json";
const OUTCOME_FILE: &str = "outcome.json";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CheckpointReferenceV1 {
    pub relative_path: String,
    pub sha256: String,
    pub generation: u64,
    pub training_step: u64,
}

impl CheckpointReferenceV1 {
    pub fn from_checkpoint(
        campaign_root: &Path,
        checkpoint: &Path,
    ) -> Result<Self, GenerationArchiveError> {
        let campaign_root = fs::canonicalize(campaign_root)?;
        let checkpoint = fs::canonicalize(checkpoint)?;
        let relative_path = relative_path(&campaign_root, &checkpoint)?;
        let metadata = read_checkpoint_metadata(&checkpoint)?;
        Ok(Self {
            relative_path,
            sha256: hex_digest(metadata.content_sha256()),
            generation: metadata.generation(),
            training_step: metadata.training_step(),
        })
    }

    pub fn path(&self, campaign_root: &Path) -> Result<PathBuf, GenerationArchiveError> {
        validate_relative_path(&self.relative_path)?;
        Ok(campaign_root.join(&self.relative_path))
    }

    pub fn verify(
        &self,
        campaign_root: &Path,
        expected_preset: &str,
    ) -> Result<(), GenerationArchiveError> {
        self.verified_metadata(campaign_root, expected_preset)?;
        Ok(())
    }

    pub fn verified_metadata(
        &self,
        campaign_root: &Path,
        expected_preset: &str,
    ) -> Result<CheckpointMetadataV2, GenerationArchiveError> {
        validate_digest(&self.sha256, "checkpoint sha256")?;
        let path = self.path(campaign_root)?;
        let metadata = read_checkpoint_metadata(&path)?;
        if hex_digest(metadata.content_sha256()) != self.sha256 {
            return Err(invalid(format!(
                "checkpoint {} has a different content digest",
                path.display()
            )));
        }
        if metadata.generation() != self.generation
            || metadata.training_step() != self.training_step
        {
            return Err(invalid(format!(
                "checkpoint {} metadata disagrees with its generation reference",
                path.display()
            )));
        }
        if preset_name(metadata.network_preset()) != expected_preset {
            return Err(invalid(format!(
                "checkpoint {} uses a different network preset",
                path.display()
            )));
        }
        Ok(metadata)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExternalFileReferenceV1 {
    pub absolute_path: String,
    pub sha256: String,
}

impl ExternalFileReferenceV1 {
    pub fn from_file(path: &Path) -> Result<Self, GenerationArchiveError> {
        let path = fs::canonicalize(path)?;
        if !path.is_file() {
            return Err(invalid(format!("{} is not a file", path.display())));
        }
        let absolute_path = path
            .to_str()
            .ok_or_else(|| invalid(format!("{} is not valid UTF-8", path.display())))?
            .to_owned();
        Ok(Self {
            absolute_path,
            sha256: sha256_file(&path)?,
        })
    }

    pub fn path(&self) -> PathBuf {
        PathBuf::from(&self.absolute_path)
    }

    pub fn verify(&self) -> Result<(), GenerationArchiveError> {
        validate_digest(&self.sha256, "external file sha256")?;
        let path = self.path();
        if !path.is_absolute() || !path.is_file() {
            return Err(invalid(format!(
                "external file {} is unavailable",
                path.display()
            )));
        }
        let actual = sha256_file(&path)?;
        if actual != self.sha256 {
            return Err(invalid(format!(
                "external file {} changed after the generation plan was sealed",
                path.display()
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CampaignIdentityV1 {
    pub rules: String,
    pub network_preset: String,
    pub genesis_checkpoint: CheckpointReferenceV1,
    pub genesis_kind: String,
    pub genesis_source_revision: String,
    pub genesis_source_dirty: bool,
    pub genesis_source_sha256: String,
    pub genesis_service_sha256: String,
    pub genesis_model_seed: u64,
    pub genesis_learning_rate_bits: u32,
}

impl CampaignIdentityV1 {
    pub fn validate(&self) -> Result<(), GenerationArchiveError> {
        if self.rules != RuleProfileId::SkudPaiSho2022.as_str() {
            return Err(invalid(
                "generation campaign uses an unsupported rule profile",
            ));
        }
        validate_preset(&self.network_preset)?;
        if !matches!(self.genesis_kind.as_str(), "generated" | "imported") {
            return Err(invalid("generation campaign has an invalid genesis kind"));
        }
        validate_digest(&self.genesis_checkpoint.sha256, "genesis checkpoint sha256")?;
        validate_digest(&self.genesis_source_sha256, "genesis source sha256")?;
        validate_digest(&self.genesis_service_sha256, "genesis service sha256")?;
        positive_f32(self.genesis_learning_rate_bits, "genesis learning rate")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GenerationInferenceClassV1 {
    pub legal_action_capacity: usize,
    pub batch_size: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct NeutralStartGenerationPlanV1 {
    pub target_remaining_decisions: usize,
    pub seed: u64,
    pub source_decision_limit: usize,
    pub maximum_source_attempts: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ActorGenerationPlanV1 {
    pub opponent: String,
    pub target_games: usize,
    pub maximum_attempts: usize,
    pub actors_per_round: usize,
    pub shard_index: u64,
    pub first_game_id: u64,
    pub decision_soft_limit: usize,
    pub actor_seed: u64,
    pub policy_temperature_bits: u32,
    pub uniform_mix_bits: u32,
    pub neutral_start: Option<NeutralStartGenerationPlanV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LearnerGenerationPlanV1 {
    pub batch_size: usize,
    pub legal_action_capacity: usize,
    pub sampler_seed: u64,
    pub learning_rate_bits: u32,
    pub target_training_step: u64,
    pub checkpoint_interval: u64,
    pub ppo_clip_bits: u32,
    pub ppo_value_weight_bits: u32,
    pub ppo_entropy_weight_bits: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromotionGenerationPlanV1 {
    pub pairs_per_batch: usize,
    pub maximum_attempted_pairs: u64,
    pub maximum_eligible_pairs: u64,
    pub first_pair_id: u64,
    pub decision_soft_limit: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neutral_start: Option<NeutralStartGenerationPlanV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_policy: Option<PromotionSamplingPolicyV1>,
    pub elo0_bits: u64,
    pub elo1_bits: u64,
    pub alpha_bits: u64,
    pub beta_bits: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GenerationPlanV1 {
    pub generation: u64,
    pub parent_champion: CheckpointReferenceV1,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_parent: Option<CheckpointReferenceV1>,
    pub source_revision: String,
    pub source_dirty: bool,
    pub source_sha256: String,
    pub service: ExternalFileReferenceV1,
    pub actor_executable: ExternalFileReferenceV1,
    pub learner_executable: ExternalFileReferenceV1,
    pub promotion_executable: ExternalFileReferenceV1,
    pub network_preset: String,
    pub optimization_level: u8,
    pub inference_classes: Vec<GenerationInferenceClassV1>,
    pub workers: usize,
    pub maximum_batch_wait_microseconds: u64,
    pub model_seed: u64,
    pub actor: ActorGenerationPlanV1,
    pub learner: LearnerGenerationPlanV1,
    pub promotion: PromotionGenerationPlanV1,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub promotion_due: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub curriculum_evaluation_due: bool,
}

impl GenerationPlanV1 {
    pub fn training_parent(&self) -> &CheckpointReferenceV1 {
        self.training_parent
            .as_ref()
            .unwrap_or(&self.parent_champion)
    }

    pub fn validate(&self) -> Result<(), GenerationArchiveError> {
        if self.generation <= self.parent_champion.generation
            || self.generation <= self.training_parent().generation
        {
            return Err(invalid(
                "generation must be newer than its champion and training parent",
            ));
        }
        validate_digest(&self.source_sha256, "generation source sha256")?;
        validate_preset(&self.network_preset)?;
        if self.optimization_level > 1 {
            return Err(invalid("optimization level must be 0 or 1"));
        }
        if self.workers == 0 || self.inference_classes.is_empty() {
            return Err(invalid("workers and inference classes must be nonzero"));
        }
        let mut previous_capacity = 0;
        for class in &self.inference_classes {
            if class.legal_action_capacity == 0
                || class.batch_size == 0
                || class.legal_action_capacity <= previous_capacity
            {
                return Err(invalid(
                    "inference classes must have positive, strictly increasing capacities",
                ));
            }
            previous_capacity = class.legal_action_capacity;
        }
        self.validate_actor()?;
        self.validate_learner()?;
        self.validate_promotion()?;
        Ok(())
    }

    pub fn verify_execution_files(&self) -> Result<(), GenerationArchiveError> {
        self.service.verify()?;
        self.actor_executable.verify()?;
        self.learner_executable.verify()?;
        self.promotion_executable.verify()?;
        Ok(())
    }

    fn validate_actor(&self) -> Result<(), GenerationArchiveError> {
        let paired = validate_opponent(&self.actor.opponent)?;
        if self.actor.target_games == 0
            || self.actor.maximum_attempts < self.actor.target_games
            || self.actor.actors_per_round == 0
            || self.actor.decision_soft_limit == 0
        {
            return Err(invalid("actor counts and limits are inconsistent"));
        }
        if paired
            && (self.actor.target_games % 2 != 0
                || self.actor.actors_per_round < 2
                || self.actor.actors_per_round % 2 != 0)
        {
            return Err(invalid(
                "distinct-opponent actor work must preserve complete pairs",
            ));
        }
        let maximum_attempts = u64::try_from(self.actor.maximum_attempts)
            .map_err(|_| invalid("actor attempt count does not fit u64"))?;
        self.actor
            .first_game_id
            .checked_add(maximum_attempts)
            .ok_or_else(|| invalid("actor game identifier range overflows"))?;
        let temperature = positive_f32(
            self.actor.policy_temperature_bits,
            "actor policy temperature",
        )?;
        let uniform_mix = finite_f32(self.actor.uniform_mix_bits, "actor uniform mix")?;
        if !(0.0..=1.0).contains(&uniform_mix) || temperature <= 0.0 {
            return Err(invalid(
                "actor policy parameters are outside their valid range",
            ));
        }
        if let Some(start) = &self.actor.neutral_start {
            validate_neutral_start(start)?;
        }
        Ok(())
    }

    fn validate_learner(&self) -> Result<(), GenerationArchiveError> {
        if self.learner.batch_size == 0
            || self.learner.legal_action_capacity == 0
            || self.learner.checkpoint_interval == 0
            || self.learner.target_training_step <= self.training_parent().training_step
        {
            return Err(invalid("learner shape, interval or target step is invalid"));
        }
        self.terminal_ppo_parameters()?;
        positive_f32(self.learner.learning_rate_bits, "learner learning rate")?;
        Ok(())
    }

    fn terminal_ppo_parameters(&self) -> Result<TerminalPpoParametersV1, GenerationArchiveError> {
        TerminalPpoParametersV1::with_behavior(
            positive_f32(
                self.actor.policy_temperature_bits,
                "actor policy temperature",
            )?,
            finite_f32(self.actor.uniform_mix_bits, "actor uniform mix")?,
            positive_f32(self.learner.ppo_clip_bits, "PPO clip")?,
            finite_f32(self.learner.ppo_value_weight_bits, "PPO value weight")?,
            finite_f32(self.learner.ppo_entropy_weight_bits, "PPO entropy weight")?,
        )
        .map_err(|source| invalid(format!("invalid terminal PPO parameters: {source}")))
    }

    fn validate_promotion(&self) -> Result<(), GenerationArchiveError> {
        if self.promotion.pairs_per_batch == 0
            || self.promotion.maximum_attempted_pairs == 0
            || self.promotion.maximum_eligible_pairs == 0
            || self.promotion.maximum_attempted_pairs < self.promotion.maximum_eligible_pairs
            || self.promotion.decision_soft_limit == 0
        {
            return Err(invalid(
                "promotion pair budgets or decision limit are invalid",
            ));
        }
        self.promotion
            .first_pair_id
            .checked_add(self.promotion.maximum_attempted_pairs)
            .ok_or_else(|| invalid("promotion pair identifier range overflows"))?;
        if let Some(start) = &self.promotion.neutral_start {
            validate_neutral_start(start)?;
        }
        if let Some(policy) = self.promotion.sampling_policy {
            policy.network_policy()?;
        }
        PromotionSprtConfig::new(
            f64::from_bits(self.promotion.elo0_bits),
            f64::from_bits(self.promotion.elo1_bits),
            f64::from_bits(self.promotion.alpha_bits),
            f64::from_bits(self.promotion.beta_bits),
        )?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ActorGenerationStageV1 {
    pub run_directory: String,
    pub snapshot: String,
    pub snapshot_sha256: String,
    pub behavior_producer: String,
    pub terminal_games: u64,
    pub training_examples: u64,
}

impl ActorGenerationStageV1 {
    pub fn verify(&self, campaign_root: &Path) -> Result<(), GenerationArchiveError> {
        self.verify_with_replay(campaign_root, true)
    }

    fn verify_integrity(&self, campaign_root: &Path) -> Result<(), GenerationArchiveError> {
        self.verify_with_replay(campaign_root, false)
    }

    fn verify_with_replay(
        &self,
        campaign_root: &Path,
        replay_games: bool,
    ) -> Result<(), GenerationArchiveError> {
        validate_relative_path(&self.run_directory)?;
        validate_relative_path(&self.snapshot)?;
        validate_digest(&self.snapshot_sha256, "actor snapshot sha256")?;
        self.behavior_producer
            .parse::<ReplayDigestV1>()
            .map_err(GenerationArchiveError::ReplayDigest)?;
        let run_directory = campaign_root.join(&self.run_directory);
        let snapshot_path = campaign_root.join(&self.snapshot);
        if !run_directory.is_dir() || !snapshot_path.starts_with(&run_directory) {
            return Err(invalid(
                "actor stage paths do not identify one published run",
            ));
        }
        let snapshot = ReplaySnapshotV1::read(&snapshot_path)?;
        let verification = if replay_games {
            snapshot.verify_directory(&run_directory)?
        } else {
            snapshot.verify_directory_integrity(&run_directory)?
        };
        if verification.digest.to_string() != self.snapshot_sha256
            || verification.game_count != self.terminal_games
            || verification.example_count != self.training_examples
        {
            return Err(invalid(
                "actor stage disagrees with its verified replay snapshot",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LearnerGenerationStageV1 {
    pub checkpoint: CheckpointReferenceV1,
    pub completed_replay_index: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromotionGenerationStageV1 {
    pub archive_directory: String,
    pub conclusion: PromotionCampaignConclusion,
    pub promotion_source_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GenerationOutcomeV1 {
    pub generation: u64,
    pub parent_champion: CheckpointReferenceV1,
    pub candidate: CheckpointReferenceV1,
    pub promotion: Option<PromotionCampaignConclusion>,
    pub champion_after: CheckpointReferenceV1,
}

impl GenerationOutcomeV1 {
    fn validate(
        &self,
        plan: &GenerationPlanV1,
        learner: &LearnerGenerationStageV1,
        promotion: Option<&PromotionGenerationStageV1>,
    ) -> Result<(), GenerationArchiveError> {
        let promotion_conclusion = promotion.map(|stage| stage.conclusion);
        if self.generation != plan.generation
            || self.parent_champion != plan.parent_champion
            || self.candidate != learner.checkpoint
            || plan.promotion_due != promotion.is_some()
            || self.promotion != promotion_conclusion
        {
            return Err(invalid(
                "generation outcome disagrees with its sealed stages",
            ));
        }
        let expected_champion =
            if self.promotion == Some(PromotionCampaignConclusion::PromoteCandidate) {
                &self.candidate
            } else {
                &self.parent_champion
            };
        if &self.champion_after != expected_champion {
            return Err(invalid("generation outcome assigns the wrong champion"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationRolesV1 {
    pub latest: Option<CheckpointReferenceV1>,
    pub candidate: Option<CheckpointReferenceV1>,
    pub champion: CheckpointReferenceV1,
    pub training: CheckpointReferenceV1,
    pub best: CheckpointReferenceV1,
    pub milestone: Option<CheckpointReferenceV1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationCampaignChainV1 {
    pub completed_generations: u64,
    pub next_generation: u64,
    pub in_progress_generation: Option<u64>,
    pub roles: GenerationRolesV1,
}

pub struct GenerationCampaignArchive {
    root: PathBuf,
    identity: CampaignIdentityV1,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum ArtifactVerification {
    None,
    Integrity,
    Semantic,
}

impl GenerationCampaignArchive {
    pub fn open_or_create(
        root: impl AsRef<Path>,
        identity: CampaignIdentityV1,
    ) -> Result<Self, GenerationArchiveError> {
        identity.validate()?;
        let root = root.as_ref().to_owned();
        fs::create_dir_all(root.join(GENERATIONS_DIRECTORY))?;
        write_document(&root.join(CAMPAIGN_FILE), CAMPAIGN_DOCUMENT, &identity)?;
        let archive = Self { root, identity };
        archive
            .identity
            .genesis_checkpoint
            .verify(&archive.root, &archive.identity.network_preset)?;
        archive.load_chain_with_artifact_verification(ArtifactVerification::Integrity)?;
        Ok(archive)
    }

    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self, GenerationArchiveError> {
        let root = root.as_ref().to_owned();
        let identity: CampaignIdentityV1 =
            read_document(&root.join(CAMPAIGN_FILE), CAMPAIGN_DOCUMENT)?;
        identity.validate()?;
        identity
            .genesis_checkpoint
            .verify(&root, &identity.network_preset)?;
        let archive = Self { root, identity };
        archive.load_chain_with_artifact_verification(ArtifactVerification::Integrity)?;
        Ok(archive)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn identity(&self) -> &CampaignIdentityV1 {
        &self.identity
    }

    pub fn generation_directory(&self, generation: u64) -> PathBuf {
        self.root
            .join(GENERATIONS_DIRECTORY)
            .join(generation_name(generation))
    }

    pub fn artifact_relative_path(&self, path: &Path) -> Result<String, GenerationArchiveError> {
        let root = fs::canonicalize(&self.root)?;
        let path = fs::canonicalize(path)?;
        relative_path(&root, &path)
    }

    pub fn establish_plan(
        &self,
        plan: &GenerationPlanV1,
    ) -> Result<PathBuf, GenerationArchiveError> {
        plan.validate()?;
        if plan.training_parent.is_none() {
            return Err(invalid(
                "new generation plans must name their training parent explicitly",
            ));
        }
        if plan.network_preset != self.identity.network_preset {
            return Err(invalid(
                "generation plan changes the campaign network preset",
            ));
        }
        let directory = self.generation_directory(plan.generation);
        fs::create_dir_all(&directory)?;
        let path = directory.join(PLAN_FILE);
        write_document(&path, PLAN_DOCUMENT, plan)?;
        Ok(path)
    }

    pub fn read_plan(&self, generation: u64) -> Result<GenerationPlanV1, GenerationArchiveError> {
        let plan: GenerationPlanV1 = read_document(
            &self.generation_directory(generation).join(PLAN_FILE),
            PLAN_DOCUMENT,
        )?;
        plan.validate()?;
        if plan.generation != generation {
            return Err(invalid("generation directory and plan number disagree"));
        }
        Ok(plan)
    }

    pub fn read_actor_stage(
        &self,
        generation: u64,
    ) -> Result<Option<ActorGenerationStageV1>, GenerationArchiveError> {
        read_optional_document(
            &self.generation_directory(generation).join(ACTOR_STAGE_FILE),
            ACTOR_STAGE_DOCUMENT,
        )
    }

    pub fn publish_actor_stage(
        &self,
        generation: u64,
        stage: &ActorGenerationStageV1,
    ) -> Result<PathBuf, GenerationArchiveError> {
        self.verify_actor_stage_integrity(generation, stage)?;
        let path = self.generation_directory(generation).join(ACTOR_STAGE_FILE);
        write_document(&path, ACTOR_STAGE_DOCUMENT, stage)?;
        Ok(path)
    }

    pub fn verify_actor_stage(
        &self,
        generation: u64,
        stage: &ActorGenerationStageV1,
    ) -> Result<(), GenerationArchiveError> {
        self.verify_actor_stage_with_record_replay(generation, stage, true)
    }

    pub fn verify_actor_stage_integrity(
        &self,
        generation: u64,
        stage: &ActorGenerationStageV1,
    ) -> Result<(), GenerationArchiveError> {
        self.verify_actor_stage_with_record_replay(generation, stage, false)
    }

    fn verify_actor_stage_with_record_replay(
        &self,
        generation: u64,
        stage: &ActorGenerationStageV1,
        replay_records: bool,
    ) -> Result<(), GenerationArchiveError> {
        let plan = self.read_plan(generation)?;
        if replay_records {
            stage.verify(&self.root)?;
        } else {
            stage.verify_integrity(&self.root)?;
        }
        if stage.terminal_games != plan.actor.target_games as u64 || stage.training_examples == 0 {
            return Err(invalid(
                "actor stage did not produce the planned terminal training corpus",
            ));
        }
        Ok(())
    }

    pub fn read_learner_stage(
        &self,
        generation: u64,
    ) -> Result<Option<LearnerGenerationStageV1>, GenerationArchiveError> {
        read_optional_document(
            &self
                .generation_directory(generation)
                .join(LEARNER_STAGE_FILE),
            LEARNER_STAGE_DOCUMENT,
        )
    }

    pub fn publish_learner_stage(
        &self,
        generation: u64,
        stage: &LearnerGenerationStageV1,
    ) -> Result<PathBuf, GenerationArchiveError> {
        self.verify_learner_stage(generation, stage)?;
        let path = self
            .generation_directory(generation)
            .join(LEARNER_STAGE_FILE);
        write_document(&path, LEARNER_STAGE_DOCUMENT, stage)?;
        Ok(path)
    }

    pub fn verify_learner_stage(
        &self,
        generation: u64,
        stage: &LearnerGenerationStageV1,
    ) -> Result<(), GenerationArchiveError> {
        let plan = self.read_plan(generation)?;
        let actor = self
            .read_actor_stage(generation)?
            .ok_or_else(|| invalid("learner stage exists without an actor stage"))?;
        self.verify_actor_stage_integrity(generation, &actor)?;
        let metadata = stage
            .checkpoint
            .verified_metadata(&self.root, &self.identity.network_preset)?;
        validate_learner_stage_artifacts(&self.root, generation, &plan, &actor, stage, &metadata)
    }

    pub fn read_promotion_stage(
        &self,
        generation: u64,
    ) -> Result<Option<PromotionGenerationStageV1>, GenerationArchiveError> {
        read_optional_document(
            &self
                .generation_directory(generation)
                .join(PROMOTION_STAGE_FILE),
            PROMOTION_STAGE_DOCUMENT,
        )
    }

    pub fn verify_promotion_stage(
        &self,
        generation: u64,
        stage: &PromotionGenerationStageV1,
    ) -> Result<(), GenerationArchiveError> {
        self.verify_promotion_stage_with_record_replay(generation, stage, true)
    }

    pub fn verify_promotion_stage_integrity(
        &self,
        generation: u64,
        stage: &PromotionGenerationStageV1,
    ) -> Result<(), GenerationArchiveError> {
        self.verify_promotion_stage_with_record_replay(generation, stage, false)
    }

    fn verify_promotion_stage_with_record_replay(
        &self,
        generation: u64,
        stage: &PromotionGenerationStageV1,
        replay_records: bool,
    ) -> Result<(), GenerationArchiveError> {
        validate_relative_path(&stage.archive_directory)?;
        validate_digest(&stage.promotion_source_sha256, "promotion source sha256")?;
        let plan = self.read_plan(generation)?;
        if !plan.promotion_due {
            return Err(invalid(
                "promotion stage exists for a generation whose plan defers promotion",
            ));
        }
        let learner = self
            .read_learner_stage(generation)?
            .ok_or_else(|| invalid("promotion stage exists without a learner stage"))?;
        let archive_path = self.root.join(&stage.archive_directory);
        let (promotion, progress) = if replay_records {
            let promotion = PromotionCampaignArchive::open_existing(archive_path)?;
            let progress = promotion.load_progress()?;
            (promotion, progress)
        } else {
            PromotionCampaignArchive::open_existing_integrity(archive_path)?
        };
        let identity = promotion.identity();
        if identity.candidate_checkpoint_sha256 != learner.checkpoint.sha256
            || identity.champion_checkpoint_sha256 != plan.parent_champion.sha256
            || identity.service_sha256 != plan.service.sha256
            || identity.preset != plan.network_preset
            || identity.optimization_level != plan.optimization_level
            || identity.workers != plan.workers
            || identity.pairs_per_batch != plan.promotion.pairs_per_batch
            || identity.maximum_attempted_pairs != plan.promotion.maximum_attempted_pairs
            || identity.maximum_eligible_pairs != plan.promotion.maximum_eligible_pairs
            || identity.first_pair_id != plan.promotion.first_pair_id
            || identity.decision_soft_limit != plan.promotion.decision_soft_limit
            || identity.neutral_start != promotion_neutral_start(&plan.promotion)
            || identity.sampling_policy != plan.promotion.sampling_policy
            || identity.maximum_batch_wait_microseconds != plan.maximum_batch_wait_microseconds
            || identity.model_seed != plan.model_seed
            || identity.elo0.to_bits() != plan.promotion.elo0_bits
            || identity.elo1.to_bits() != plan.promotion.elo1_bits
            || identity.alpha.to_bits() != plan.promotion.alpha_bits
            || identity.beta.to_bits() != plan.promotion.beta_bits
            || identity.source_sha256 != stage.promotion_source_sha256
            || !same_inference_classes(identity, &plan.inference_classes)
        {
            return Err(invalid(
                "promotion archive disagrees with its generation plan",
            ));
        }
        let conclusion = progress
            .conclusion(identity)?
            .ok_or_else(|| invalid("promotion stage points to an unfinished campaign"))?;
        if conclusion != stage.conclusion {
            return Err(invalid(
                "promotion stage conclusion changed after publication",
            ));
        }
        promotion.require_published_conclusion(&progress, conclusion)?;
        Ok(())
    }

    pub fn publish_promotion_stage(
        &self,
        generation: u64,
        stage: &PromotionGenerationStageV1,
    ) -> Result<PathBuf, GenerationArchiveError> {
        self.verify_promotion_stage_integrity(generation, stage)?;
        let path = self
            .generation_directory(generation)
            .join(PROMOTION_STAGE_FILE);
        write_document(&path, PROMOTION_STAGE_DOCUMENT, stage)?;
        Ok(path)
    }

    pub fn read_outcome(
        &self,
        generation: u64,
    ) -> Result<Option<GenerationOutcomeV1>, GenerationArchiveError> {
        read_optional_document(
            &self.generation_directory(generation).join(OUTCOME_FILE),
            OUTCOME_DOCUMENT,
        )
    }

    pub fn publish_outcome(
        &self,
        outcome: &GenerationOutcomeV1,
    ) -> Result<PathBuf, GenerationArchiveError> {
        let plan = self.read_plan(outcome.generation)?;
        let learner = self
            .read_learner_stage(outcome.generation)?
            .ok_or_else(|| invalid("generation outcome requires a learner stage"))?;
        let promotion = self.read_promotion_stage(outcome.generation)?;
        outcome.validate(&plan, &learner, promotion.as_ref())?;
        let path = self
            .generation_directory(outcome.generation)
            .join(OUTCOME_FILE);
        write_document(&path, OUTCOME_DOCUMENT, outcome)?;
        Ok(path)
    }

    pub fn load_chain(&self) -> Result<GenerationCampaignChainV1, GenerationArchiveError> {
        self.load_chain_with_artifact_verification(ArtifactVerification::None)
    }

    pub fn verify_all_artifacts(&self) -> Result<(), GenerationArchiveError> {
        self.load_chain_with_artifact_verification(ArtifactVerification::Semantic)?;
        Ok(())
    }

    fn load_chain_with_artifact_verification(
        &self,
        verification: ArtifactVerification,
    ) -> Result<GenerationCampaignChainV1, GenerationArchiveError> {
        let directory = self.root.join(GENERATIONS_DIRECTORY);
        if !directory.is_dir() {
            return Err(invalid(
                "generation campaign is missing its generations directory",
            ));
        }
        let mut generations = Vec::new();
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let generation = parse_generation_name(&name)
                .ok_or_else(|| invalid(format!("unexpected generation entry {name}")))?;
            if !entry.file_type()?.is_dir() {
                return Err(invalid(format!(
                    "generation entry {name} is not a directory"
                )));
            }
            generations.push((generation, entry.path()));
        }
        generations.sort_unstable_by_key(|(generation, _)| *generation);
        let mut champion = self.identity.genesis_checkpoint.clone();
        let mut training = champion.clone();
        let mut latest = None;
        let mut in_progress = None;
        let mut expected = champion
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("genesis generation has no successor"))?;
        let mut completed = 0_u64;
        let generation_count = generations.len();
        for (index, (generation, generation_directory)) in generations.into_iter().enumerate() {
            if generation != expected || in_progress.is_some() {
                return Err(invalid(
                    "generation directories are not one contiguous chain",
                ));
            }
            let plan_path = generation_directory.join(PLAN_FILE);
            if !plan_path.exists() {
                if index + 1 == generation_count
                    && recoverable_unpublished_generation(&generation_directory)?
                {
                    continue;
                }
                return Err(invalid(format!(
                    "generation {generation} is missing its published plan"
                )));
            }
            let plan = self.read_plan(generation)?;
            let training_lineage_matches = plan
                .training_parent
                .as_ref()
                .map_or(true, |parent| parent == &training);
            if plan.parent_champion != champion
                || !training_lineage_matches
                || plan.network_preset != self.identity.network_preset
            {
                return Err(invalid(
                    "generation plan does not descend from the current champion and training lineage",
                ));
            }
            training = plan.training_parent().clone();
            let actor = self.read_actor_stage(generation)?;
            let learner = self.read_learner_stage(generation)?;
            let promotion = self.read_promotion_stage(generation)?;
            let outcome = self.read_outcome(generation)?;
            if learner.is_some() && actor.is_none()
                || promotion.is_some() && learner.is_none()
                || !plan.promotion_due && promotion.is_some()
                || outcome.is_some() && learner.is_none()
                || outcome.is_some() && plan.promotion_due != promotion.is_some()
            {
                return Err(invalid("generation stages are not in causal order"));
            }
            if verification != ArtifactVerification::None {
                plan.parent_champion
                    .verify(&self.root, &self.identity.network_preset)?;
                if plan.training_parent() != &plan.parent_champion {
                    plan.training_parent()
                        .verify(&self.root, &self.identity.network_preset)?;
                }
                if let Some(actor) = &actor {
                    if verification == ArtifactVerification::Semantic {
                        actor.verify(&self.root)?;
                    } else {
                        actor.verify_integrity(&self.root)?;
                    }
                    if actor.terminal_games != plan.actor.target_games as u64
                        || actor.training_examples == 0
                    {
                        return Err(invalid(
                            "actor stage did not produce the planned terminal training corpus",
                        ));
                    }
                }
                if let Some(learner) = &learner {
                    let actor = actor
                        .as_ref()
                        .expect("stage order guarantees the actor dependency");
                    let metadata = learner
                        .checkpoint
                        .verified_metadata(&self.root, &self.identity.network_preset)?;
                    validate_learner_stage_artifacts(
                        &self.root, generation, &plan, actor, learner, &metadata,
                    )?;
                }
                if let Some(promotion) = &promotion {
                    if verification == ArtifactVerification::Semantic {
                        self.verify_promotion_stage(generation, promotion)?;
                    } else {
                        self.verify_promotion_stage_integrity(generation, promotion)?;
                    }
                }
            }
            if let Some(outcome) = outcome {
                outcome.validate(
                    &plan,
                    learner.as_ref().expect("validated learner dependency"),
                    promotion.as_ref(),
                )?;
                latest = Some(outcome.candidate.clone());
                champion = outcome.champion_after;
                training =
                    if outcome.promotion == Some(PromotionCampaignConclusion::RejectCandidate) {
                        champion.clone()
                    } else {
                        outcome.candidate
                    };
                completed = completed
                    .checked_add(1)
                    .ok_or_else(|| invalid("completed generation count overflows"))?;
                expected = expected
                    .checked_add(1)
                    .ok_or_else(|| invalid("generation number overflows"))?;
            } else {
                in_progress = Some(generation);
            }
        }
        let candidate = match in_progress {
            Some(generation) => self
                .read_learner_stage(generation)?
                .map(|stage| stage.checkpoint),
            None => latest.clone(),
        };
        Ok(GenerationCampaignChainV1 {
            completed_generations: completed,
            next_generation: in_progress.unwrap_or(expected),
            in_progress_generation: in_progress,
            roles: GenerationRolesV1 {
                latest,
                candidate,
                champion: champion.clone(),
                training,
                best: champion,
                milestone: None,
            },
        })
    }
}

const fn default_true() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

fn same_inference_classes(
    identity: &crate::PromotionRunIdentityV1,
    planned: &[GenerationInferenceClassV1],
) -> bool {
    identity.inference_classes.len() == planned.len()
        && identity
            .inference_classes
            .iter()
            .zip(planned)
            .all(|(actual, expected)| {
                actual.legal_action_capacity == expected.legal_action_capacity
                    && actual.batch_size == expected.batch_size
            })
}

fn validate_neutral_start(
    start: &NeutralStartGenerationPlanV1,
) -> Result<(), GenerationArchiveError> {
    if start.target_remaining_decisions == 0
        || start.source_decision_limit == 0
        || start.maximum_source_attempts == 0
    {
        Err(invalid("neutral-start controls must be positive"))
    } else {
        Ok(())
    }
}

fn promotion_neutral_start(plan: &PromotionGenerationPlanV1) -> Option<PromotionNeutralStartV1> {
    plan.neutral_start
        .as_ref()
        .map(|start| PromotionNeutralStartV1 {
            target_remaining_decisions: start.target_remaining_decisions,
            seed: start.seed,
            source_decision_limit: start.source_decision_limit,
            maximum_source_attempts: start.maximum_source_attempts,
        })
}

fn validate_opponent(opponent: &str) -> Result<bool, GenerationArchiveError> {
    match opponent {
        "self" => Ok(false),
        "random" | "site" => Ok(true),
        _ => {
            let simulations = opponent
                .strip_prefix("mcts:")
                .ok_or_else(|| invalid(format!("unsupported actor opponent {opponent}")))?
                .parse::<usize>()
                .map_err(|_| invalid(format!("invalid MCTS opponent {opponent}")))?;
            if simulations == 0 || simulations > 512 {
                return Err(invalid("curriculum MCTS budget must be between 1 and 512"));
            }
            Ok(true)
        }
    }
}

fn validate_preset(preset: &str) -> Result<(), GenerationArchiveError> {
    if matches!(preset, "micro" | "pure") {
        Ok(())
    } else {
        Err(invalid(format!("unsupported network preset {preset}")))
    }
}

fn optimization_level(level: u8) -> Result<OptimizationLevel, GenerationArchiveError> {
    match level {
        0 => Ok(OptimizationLevel::Level0),
        1 => Ok(OptimizationLevel::Level1),
        _ => Err(invalid("optimization level must be 0 or 1")),
    }
}

fn preset_name(preset: NetworkPreset) -> &'static str {
    match preset {
        NetworkPreset::Micro => "micro",
        NetworkPreset::Pure => "pure",
    }
}

fn recoverable_unpublished_generation(directory: &Path) -> Result<bool, GenerationArchiveError> {
    for entry in fs::read_dir(directory)? {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if !name.starts_with(".plan.json.partial-") {
            return Ok(false);
        }
    }
    Ok(true)
}

fn finite_f32(bits: u32, field: &'static str) -> Result<f32, GenerationArchiveError> {
    let value = f32::from_bits(bits);
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid(format!("{field} must be finite")))
    }
}

fn positive_f32(bits: u32, field: &'static str) -> Result<f32, GenerationArchiveError> {
    let value = finite_f32(bits, field)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(invalid(format!("{field} must be positive")))
    }
}

#[derive(Deserialize, Serialize)]
struct SealedDocument<T> {
    format: String,
    payload: T,
    sha256: String,
}

fn write_document<T: Serialize>(
    destination: &Path,
    format: &str,
    payload: &T,
) -> Result<(), GenerationArchiveError> {
    let bytes = encode_document(format, payload)?;
    atomic_file::write_idempotently(destination, &bytes)?;
    Ok(())
}

pub(crate) fn write_campaign_identity_to_genesis(
    directory: &Path,
    identity: &CampaignIdentityV1,
) -> Result<(), GenerationArchiveError> {
    write_document(&directory.join("genesis.json"), CAMPAIGN_DOCUMENT, identity)
}

pub(crate) fn read_campaign_identity_from_genesis(
    directory: &Path,
) -> Result<CampaignIdentityV1, GenerationArchiveError> {
    read_document(&directory.join("genesis.json"), CAMPAIGN_DOCUMENT)
}

fn read_optional_document<T: DeserializeOwned + Serialize>(
    source: &Path,
    format: &str,
) -> Result<Option<T>, GenerationArchiveError> {
    match fs::read(source) {
        Ok(bytes) => read_document_bytes(&bytes, format).map(Some),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn read_document<T: DeserializeOwned + Serialize>(
    source: &Path,
    format: &str,
) -> Result<T, GenerationArchiveError> {
    let bytes = fs::read(source)?;
    read_document_bytes(&bytes, format)
}

fn read_document_bytes<T: DeserializeOwned + Serialize>(
    bytes: &[u8],
    expected_format: &str,
) -> Result<T, GenerationArchiveError> {
    let document: SealedDocument<T> = serde_json::from_slice(bytes)?;
    if document.format != expected_format {
        return Err(invalid(format!(
            "sealed document has format {}; expected {expected_format}",
            document.format
        )));
    }
    validate_digest(&document.sha256, "sealed document sha256")?;
    let canonical = encode_document(expected_format, &document.payload)?;
    if canonical != bytes {
        return Err(invalid(
            "sealed document checksum or canonical encoding is invalid",
        ));
    }
    Ok(document.payload)
}

fn encode_document<T: Serialize>(
    format: &str,
    payload: &T,
) -> Result<Vec<u8>, GenerationArchiveError> {
    let payload_bytes = serde_json::to_vec(payload)?;
    let mut hasher = Sha256::new();
    hasher.update(format.as_bytes());
    hasher.update([0]);
    hasher.update(&payload_bytes);
    let document = SealedDocument {
        format: format.to_owned(),
        payload,
        sha256: hex_digest(hasher.finalize().into()),
    };
    let mut bytes = serde_json::to_vec_pretty(&document)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn relative_path(root: &Path, path: &Path) -> Result<String, GenerationArchiveError> {
    let relative = path.strip_prefix(root).map_err(|_| {
        invalid(format!(
            "artifact {} is outside campaign {}",
            path.display(),
            root.display()
        ))
    })?;
    let text = relative
        .to_str()
        .ok_or_else(|| invalid(format!("{} is not valid UTF-8", relative.display())))?
        .to_owned();
    validate_relative_path(&text)?;
    Ok(text)
}

fn validate_relative_path(path: &str) -> Result<(), GenerationArchiveError> {
    let mut components = Path::new(path).components();
    let mut count = 0_usize;
    for component in &mut components {
        if !matches!(component, Component::Normal(_)) {
            return Err(invalid(format!("unsafe campaign-relative path {path}")));
        }
        count += 1;
    }
    if count == 0 || path.chars().any(char::is_control) {
        return Err(invalid(format!("invalid campaign-relative path {path}")));
    }
    Ok(())
}

fn validate_digest(digest: &str, field: &'static str) -> Result<(), GenerationArchiveError> {
    if digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(invalid(format!(
            "{field} is not a lowercase SHA-256 digest"
        )))
    }
}

fn sha256_file(path: &Path) -> Result<String, GenerationArchiveError> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    io::copy(&mut file, &mut hasher)?;
    Ok(hex_digest(hasher.finalize().into()))
}

fn hex_digest(digest: [u8; 32]) -> String {
    use core::fmt::Write as _;

    let mut text = String::with_capacity(64);
    for byte in digest {
        write!(text, "{byte:02x}").unwrap();
    }
    text
}

pub(crate) fn hex_digest_for_internal(digest: [u8; 32]) -> String {
    hex_digest(digest)
}

fn generation_name(generation: u64) -> String {
    format!("generation-{generation:020}")
}

fn parse_generation_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("generation-")?;
    if digits.len() != 20 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn invalid(message: impl Into<String>) -> GenerationArchiveError {
    GenerationArchiveError::Invalid(message.into())
}

#[derive(Debug)]
pub enum GenerationArchiveError {
    Io(io::Error),
    Json(serde_json::Error),
    Checkpoint(CheckpointMetadataError),
    ReplaySnapshot(ReplaySnapshotV1Error),
    ReplayDigest(ReplayDigestV1Error),
    Promotion(PromotionArchiveError),
    PromotionSprt(PromotionSprtError),
    Invalid(String),
}

impl fmt::Display for GenerationArchiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => write!(formatter, "generation archive I/O failed: {source}"),
            Self::Json(source) => write!(formatter, "generation archive JSON failed: {source}"),
            Self::Checkpoint(source) => {
                write!(
                    formatter,
                    "generation checkpoint verification failed: {source}"
                )
            }
            Self::ReplaySnapshot(source) => {
                write!(formatter, "generation replay verification failed: {source}")
            }
            Self::ReplayDigest(source) => {
                write!(formatter, "generation replay digest is invalid: {source}")
            }
            Self::Promotion(source) => {
                write!(
                    formatter,
                    "generation promotion verification failed: {source}"
                )
            }
            Self::PromotionSprt(source) => {
                write!(
                    formatter,
                    "generation promotion configuration is invalid: {source}"
                )
            }
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for GenerationArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::Checkpoint(source) => Some(source),
            Self::ReplaySnapshot(source) => Some(source),
            Self::ReplayDigest(source) => Some(source),
            Self::Promotion(source) => Some(source),
            Self::PromotionSprt(source) => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

impl From<io::Error> for GenerationArchiveError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for GenerationArchiveError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

impl From<CheckpointMetadataError> for GenerationArchiveError {
    fn from(source: CheckpointMetadataError) -> Self {
        Self::Checkpoint(source)
    }
}

impl From<ReplaySnapshotV1Error> for GenerationArchiveError {
    fn from(source: ReplaySnapshotV1Error) -> Self {
        Self::ReplaySnapshot(source)
    }
}

impl From<PromotionArchiveError> for GenerationArchiveError {
    fn from(source: PromotionArchiveError) -> Self {
        Self::Promotion(source)
    }
}

impl From<PromotionSprtError> for GenerationArchiveError {
    fn from(source: PromotionSprtError) -> Self {
        Self::PromotionSprt(source)
    }
}

#[cfg(test)]
#[path = "generation/tests.rs"]
mod tests;
