use super::*;
use paisho_core::RuleProfileId;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

#[test]
fn deferred_generation_advances_training_without_claiming_a_promotion_or_elo() {
    let temporary = TemporaryDirectory::new();
    let curriculum_directory = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let archive = GenerationCampaignArchive::open_or_create(
        &temporary.path,
        campaign_identity(genesis.clone()),
    )
    .unwrap();
    let curriculum = crate::CurriculumCampaignArchive::open_or_create(
        &curriculum_directory.path,
        &archive,
        crate::CurriculumTierV1::Random,
    )
    .unwrap();
    let mut first = plan(1, genesis.clone(), 1);
    first.promotion_due = false;
    first.curriculum_evaluation_due = false;
    archive.establish_plan(&first).unwrap();
    let candidate = write_checkpoint(
        &temporary.path,
        &archive.generation_directory(1).join("candidate.psckpt"),
        1,
        1,
        8,
    );
    publish_synthetic_stages(
        &archive,
        &first,
        candidate.clone(),
        PromotionCampaignConclusion::PromoteCandidate,
    );
    std::fs::remove_file(archive.generation_directory(1).join(PROMOTION_STAGE_FILE)).unwrap();
    let outcome = GenerationOutcomeV1 {
        generation: 1,
        parent_champion: genesis.clone(),
        candidate: candidate.clone(),
        promotion: None,
        champion_after: genesis.clone(),
    };
    let mut false_promotion = outcome.clone();
    false_promotion.promotion = Some(PromotionCampaignConclusion::PromoteCandidate);
    assert!(archive.publish_outcome(&false_promotion).is_err());
    archive.publish_outcome(&outcome).unwrap();
    archive.publish_outcome(&outcome).unwrap();
    let decision = curriculum
        .publish_deferred_evaluation_decision(1, candidate.clone())
        .unwrap();
    assert!(matches!(
        decision.evidence,
        crate::CurriculumEvidenceV1::DeferredEvaluation
    ));
    assert_eq!(
        curriculum.load_state().unwrap().current_tier,
        crate::CurriculumTierV1::Random
    );
    let chain = archive.load_chain().unwrap();
    assert_eq!(chain.next_generation, 2);
    assert_eq!(chain.roles.training, candidate.clone());
    assert_eq!(chain.roles.champion, genesis.clone());
    let mut second = plan(2, genesis, 2);
    second.training_parent = Some(candidate);
    archive.establish_plan(&second).unwrap();
    assert_eq!(
        archive.load_chain().unwrap().in_progress_generation,
        Some(2)
    );
}

#[test]
fn default_schedule_preserves_legacy_sealed_plan_encoding() {
    let reference = CheckpointReferenceV1 {
        relative_path: "genesis.psckpt".into(),
        sha256: "11".repeat(32),
        generation: 0,
        training_step: 0,
    };
    let original = plan(1, reference, 1);
    let bytes = encode_document(PLAN_DOCUMENT, &original).unwrap();
    let text = std::str::from_utf8(&bytes).unwrap();
    assert!(!text.contains("promotion_due"));
    assert!(!text.contains("curriculum_evaluation_due"));
    let restored: GenerationPlanV1 = read_document_bytes(&bytes, PLAN_DOCUMENT).unwrap();
    assert_eq!(restored, original);
    assert_eq!(encode_document(PLAN_DOCUMENT, &restored).unwrap(), bytes);
}

#[test]
fn generation_names_are_fixed_width_and_round_trip() {
    for generation in [0, 1, 42, u64::MAX] {
        let name = generation_name(generation);
        assert_eq!(parse_generation_name(&name), Some(generation));
    }
    assert_eq!(parse_generation_name("generation-1"), None);
    assert_eq!(parse_generation_name("other-00000000000000000001"), None);
}

#[test]
fn sealed_document_rejects_content_tampering() {
    let payload = NeutralStartGenerationPlanV1 {
        target_remaining_decisions: 64,
        seed: 17,
        source_decision_limit: 1_024,
        maximum_source_attempts: 4,
    };
    let bytes = encode_document("test-document", &payload).unwrap();
    let decoded: NeutralStartGenerationPlanV1 =
        read_document_bytes(&bytes, "test-document").unwrap();
    assert_eq!(decoded, payload);

    let mut damaged = bytes;
    let position = damaged
        .windows(2)
        .position(|window| window == b"64")
        .expect("fixture contains its horizon");
    damaged[position] = b'6';
    damaged[position + 1] = b'5';
    assert!(
        read_document_bytes::<NeutralStartGenerationPlanV1>(&damaged, "test-document").is_err()
    );
}

#[test]
fn curriculum_opponents_keep_the_mcts_ceiling() {
    assert!(!validate_opponent("self").unwrap());
    assert!(validate_opponent("random").unwrap());
    assert!(validate_opponent("site").unwrap());
    assert!(validate_opponent("mcts:512").unwrap());
    assert!(validate_opponent("mcts:513").is_err());
    assert!(validate_opponent("mcts:0").is_err());
    assert!(validate_opponent("teacher").is_err());
}

#[test]
fn append_only_outcomes_derive_latest_and_champion_roles() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let campaign = campaign_identity(genesis.clone());
    let archive = GenerationCampaignArchive::open_or_create(&temporary.path, campaign).unwrap();

    let first_plan = plan(1, genesis.clone(), 1);
    archive.establish_plan(&first_plan).unwrap();
    let first_candidate = write_checkpoint(
        &temporary.path,
        &archive.generation_directory(1).join("candidate.psckpt"),
        1,
        1,
        8,
    );
    publish_synthetic_stages(
        &archive,
        &first_plan,
        first_candidate.clone(),
        PromotionCampaignConclusion::PromoteCandidate,
    );
    archive
        .publish_outcome(&GenerationOutcomeV1 {
            generation: 1,
            parent_champion: genesis,
            candidate: first_candidate.clone(),
            promotion: Some(PromotionCampaignConclusion::PromoteCandidate),
            champion_after: first_candidate.clone(),
        })
        .unwrap();

    let second_plan = plan(2, first_candidate.clone(), 2);
    archive.establish_plan(&second_plan).unwrap();
    let second_candidate = write_checkpoint(
        &temporary.path,
        &archive.generation_directory(2).join("candidate.psckpt"),
        2,
        2,
        8,
    );
    publish_synthetic_stages(
        &archive,
        &second_plan,
        second_candidate.clone(),
        PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs,
    );
    archive
        .publish_outcome(&GenerationOutcomeV1 {
            generation: 2,
            parent_champion: first_candidate.clone(),
            candidate: second_candidate.clone(),
            promotion: Some(PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs),
            champion_after: first_candidate.clone(),
        })
        .unwrap();

    let chain = archive.load_chain().unwrap();
    assert_eq!(chain.completed_generations, 2);
    assert_eq!(chain.next_generation, 3);
    assert_eq!(chain.in_progress_generation, None);
    assert_eq!(chain.roles.latest, Some(second_candidate.clone()));
    assert_eq!(chain.roles.candidate, Some(second_candidate.clone()));
    assert_eq!(chain.roles.champion, first_candidate.clone());
    assert_eq!(chain.roles.training, chain.roles.latest.clone().unwrap());
    assert_eq!(chain.roles.best, first_candidate.clone());
    assert_eq!(chain.roles.milestone, None);

    let mut third_plan = plan(3, first_candidate.clone(), 3);
    third_plan.training_parent = Some(second_candidate);
    archive.establish_plan(&third_plan).unwrap();
    let third_candidate = write_checkpoint(
        &temporary.path,
        &archive.generation_directory(3).join("candidate.psckpt"),
        3,
        3,
        8,
    );
    publish_synthetic_stages(
        &archive,
        &third_plan,
        third_candidate.clone(),
        PromotionCampaignConclusion::RejectCandidate,
    );
    archive
        .publish_outcome(&GenerationOutcomeV1 {
            generation: 3,
            parent_champion: first_candidate.clone(),
            candidate: third_candidate.clone(),
            promotion: Some(PromotionCampaignConclusion::RejectCandidate),
            champion_after: first_candidate.clone(),
        })
        .unwrap();

    let chain = archive.load_chain().unwrap();
    assert_eq!(chain.completed_generations, 3);
    assert_eq!(chain.roles.latest, Some(third_candidate));
    assert_eq!(chain.roles.champion, first_candidate.clone());
    assert_eq!(chain.roles.training, first_candidate);
}

#[test]
fn a_generation_cannot_silently_reparent_after_its_plan_is_sealed() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let archive = GenerationCampaignArchive::open_or_create(
        &temporary.path,
        campaign_identity(genesis.clone()),
    )
    .unwrap();
    let first = plan(1, genesis.clone(), 1);
    archive.establish_plan(&first).unwrap();

    let mut changed = first;
    changed.parent_champion.sha256 = "aa".repeat(32);
    assert!(archive.establish_plan(&changed).is_err());
}

#[test]
fn legacy_plan_after_an_inconclusive_generation_keeps_its_historical_parent() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let archive = GenerationCampaignArchive::open_or_create(
        &temporary.path,
        campaign_identity(genesis.clone()),
    )
    .unwrap();
    let first_plan = plan(1, genesis.clone(), 1);
    archive.establish_plan(&first_plan).unwrap();
    let first_candidate = write_checkpoint(
        &temporary.path,
        &archive.generation_directory(1).join("candidate.psckpt"),
        1,
        1,
        8,
    );
    publish_synthetic_stages(
        &archive,
        &first_plan,
        first_candidate.clone(),
        PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs,
    );
    archive
        .publish_outcome(&GenerationOutcomeV1 {
            generation: 1,
            parent_champion: genesis.clone(),
            candidate: first_candidate,
            promotion: Some(PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs),
            champion_after: genesis.clone(),
        })
        .unwrap();

    let legacy_second_plan = plan(2, genesis.clone(), 2);
    let mut legacy_second_plan = legacy_second_plan;
    legacy_second_plan.training_parent = None;
    assert!(archive.establish_plan(&legacy_second_plan).is_err());
    let directory = archive.generation_directory(2);
    std::fs::create_dir_all(&directory).unwrap();
    write_document(
        &directory.join(PLAN_FILE),
        PLAN_DOCUMENT,
        &legacy_second_plan,
    )
    .unwrap();
    assert_eq!(archive.read_plan(2).unwrap().training_parent, None);

    let chain = archive.load_chain().unwrap();
    assert_eq!(chain.in_progress_generation, Some(2));
    assert_eq!(chain.roles.training, genesis);
}

#[test]
fn an_unpublished_terminal_generation_directory_is_recoverable() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let archive = GenerationCampaignArchive::open_or_create(
        &temporary.path,
        campaign_identity(genesis.clone()),
    )
    .unwrap();
    let directory = archive.generation_directory(1);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join(".plan.json.partial-interrupted"), b"partial").unwrap();

    let chain = archive.load_chain().unwrap();
    assert_eq!(chain.next_generation, 1);
    assert_eq!(chain.in_progress_generation, None);

    let expected = plan(1, genesis, 1);
    archive.establish_plan(&expected).unwrap();
    assert_eq!(archive.read_plan(1).unwrap(), expected);
}

#[test]
fn reopening_a_campaign_verifies_referenced_stage_artifacts() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let archive = GenerationCampaignArchive::open_or_create(
        &temporary.path,
        campaign_identity(genesis.clone()),
    )
    .unwrap();
    let generation_plan = plan(1, genesis, 1);
    archive.establish_plan(&generation_plan).unwrap();
    write_document(
        &archive.generation_directory(1).join(ACTOR_STAGE_FILE),
        ACTOR_STAGE_DOCUMENT,
        &ActorGenerationStageV1 {
            run_directory: "generations/generation-00000000000000000001/actors/missing".to_owned(),
            snapshot: "generations/generation-00000000000000000001/actors/missing/snapshot.psrsnap"
                .to_owned(),
            snapshot_sha256: "55".repeat(32),
            behavior_producer: "66".repeat(32),
            terminal_games: 2,
            training_examples: 1,
        },
    )
    .unwrap();

    let error = GenerationCampaignArchive::open_existing(&temporary.path)
        .err()
        .expect("missing replay artifacts must reject campaign reopening");
    assert!(error
        .to_string()
        .contains("actor stage paths do not identify one published run"));
}

#[test]
fn learner_metadata_is_bound_to_the_actor_corpus_and_training_plan() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let generation_plan = plan(1, genesis, 1);
    let snapshot_digest = "55".repeat(32);
    let candidate = write_checkpoint_with_snapshot(
        &temporary.path,
        &temporary.path.join("candidate.psckpt"),
        1,
        1,
        8,
        &snapshot_digest,
    );
    let metadata = read_checkpoint_metadata(&temporary.path.join("candidate.psckpt")).unwrap();
    let mut actor = ActorGenerationStageV1 {
        run_directory: "actors/run".to_owned(),
        snapshot: "actors/run/snapshot.psrsnap".to_owned(),
        snapshot_sha256: snapshot_digest,
        behavior_producer: "66".repeat(32),
        terminal_games: 2,
        training_examples: 8,
    };
    let stage = LearnerGenerationStageV1 {
        checkpoint: candidate,
        completed_replay_index: 8,
    };

    verification::validate_checkpoint_metadata(1, &generation_plan, &actor, &stage, &metadata)
        .unwrap();

    actor.snapshot_sha256 = "77".repeat(32);
    assert!(verification::validate_checkpoint_metadata(
        1,
        &generation_plan,
        &actor,
        &stage,
        &metadata,
    )
    .is_err());
    actor.snapshot_sha256 = "55".repeat(32);
    let mut wrong_optimization = generation_plan.clone();
    wrong_optimization.optimization_level = 0;
    assert!(verification::validate_checkpoint_metadata(
        1,
        &wrong_optimization,
        &actor,
        &stage,
        &metadata
    )
    .is_err());
    let mut wrong_rate = generation_plan;
    wrong_rate.learner.learning_rate_bits = 2.0e-4_f32.to_bits();
    assert!(
        verification::validate_checkpoint_metadata(1, &wrong_rate, &actor, &stage, &metadata,)
            .is_err()
    );
}

#[test]
fn learner_origin_and_commit_are_bound_to_the_training_parent() {
    let temporary = TemporaryDirectory::new();
    let genesis = write_checkpoint(
        &temporary.path,
        &temporary.path.join("genesis.psckpt"),
        0,
        0,
        0,
    );
    let generation_plan = plan(1, genesis.clone(), 1);
    let actor = ActorGenerationStageV1 {
        run_directory: "actors/run".to_owned(),
        snapshot: "actors/run/snapshot.psrsnap".to_owned(),
        snapshot_sha256: "55".repeat(32),
        behavior_producer: "66".repeat(32),
        terminal_games: 2,
        training_examples: 8,
    };
    let learner_directory = temporary.path.join("learner");
    std::fs::create_dir(&learner_directory).unwrap();
    let checkpoint_file = crate::commit::checkpoint_file_name(1, 1, 0);
    let candidate = write_checkpoint_with_snapshot(
        &temporary.path,
        &learner_directory.join(&checkpoint_file),
        1,
        1,
        8,
        &actor.snapshot_sha256,
    );
    let metadata = read_checkpoint_metadata(&learner_directory.join(&checkpoint_file)).unwrap();
    let replay_snapshot: ReplayDigestV1 = actor.snapshot_sha256.parse().unwrap();
    let behavior_producer: ReplayDigestV1 = actor.behavior_producer.parse().unwrap();
    let parent_digest: ReplayDigestV1 = genesis.sha256.parse().unwrap();
    let identity = crate::LearnerIdentityV1 {
        replay_snapshot,
        network_preset: metadata.network_preset(),
        optimization: metadata.optimization(),
        batch_size: generation_plan.learner.batch_size,
        legal_action_capacity: generation_plan.learner.legal_action_capacity,
        model_seed: generation_plan.model_seed,
        sampler_seed: generation_plan.learner.sampler_seed,
        generation: 1,
        learning_rate: f32::from_bits(generation_plan.learner.learning_rate_bits),
        objective: crate::LearnerObjectiveV1::TerminalPpo(
            crate::TerminalPpoLearnerObjectiveV1::new(
                behavior_producer,
                Some(parent_digest),
                generation_plan.terminal_ppo_parameters().unwrap(),
            ),
        ),
    };
    let parent = crate::ParentCheckpointV1::new(*parent_digest.as_bytes(), 0, 0);
    crate::LearnerOriginV1::new(identity, Some(parent))
        .unwrap()
        .establish(&learner_directory)
        .unwrap();
    let candidate_digest: ReplayDigestV1 = candidate.sha256.parse().unwrap();
    crate::LearnerCommit::new(
        identity,
        0,
        1,
        8,
        1,
        checkpoint_file,
        *candidate_digest.as_bytes(),
    )
    .unwrap()
    .write_idempotently(&learner_directory)
    .unwrap();
    let stage = LearnerGenerationStageV1 {
        checkpoint: candidate,
        completed_replay_index: 8,
    };

    validate_learner_stage_artifacts(
        &temporary.path,
        1,
        &generation_plan,
        &actor,
        &stage,
        &metadata,
    )
    .unwrap();

    std::fs::remove_file(crate::LearnerOriginV1::path(&learner_directory)).unwrap();
    crate::LearnerOriginV1::new(
        identity,
        Some(crate::ParentCheckpointV1::new([0xaa; 32], 0, 0)),
    )
    .unwrap()
    .establish(&learner_directory)
    .unwrap();
    assert!(validate_learner_stage_artifacts(
        &temporary.path,
        1,
        &generation_plan,
        &actor,
        &stage,
        &metadata,
    )
    .is_err());
}

fn campaign_identity(genesis_checkpoint: CheckpointReferenceV1) -> CampaignIdentityV1 {
    CampaignIdentityV1 {
        rules: RuleProfileId::SkudPaiSho2022.as_str().to_owned(),
        network_preset: "micro".to_owned(),
        genesis_checkpoint,
        genesis_kind: "imported".to_owned(),
        genesis_source_revision: "test".to_owned(),
        genesis_source_dirty: false,
        genesis_source_sha256: "11".repeat(32),
        genesis_service_sha256: "22".repeat(32),
        genesis_model_seed: 17,
        genesis_learning_rate_bits: 1.0e-4_f32.to_bits(),
    }
}

fn plan(
    generation: u64,
    parent_champion: CheckpointReferenceV1,
    target_training_step: u64,
) -> GenerationPlanV1 {
    let external = ExternalFileReferenceV1 {
        absolute_path: "/test/tool".to_owned(),
        sha256: "33".repeat(32),
    };
    GenerationPlanV1 {
        generation,
        training_parent: Some(parent_champion.clone()),
        parent_champion,
        source_revision: "test".to_owned(),
        source_dirty: false,
        source_sha256: "44".repeat(32),
        service: external.clone(),
        actor_executable: external.clone(),
        learner_executable: external.clone(),
        promotion_executable: external,
        network_preset: "micro".to_owned(),
        optimization_level: 1,
        inference_classes: vec![GenerationInferenceClassV1 {
            legal_action_capacity: 1_024,
            batch_size: 1,
        }],
        workers: 2,
        maximum_batch_wait_microseconds: 5_000,
        model_seed: 17,
        actor: ActorGenerationPlanV1 {
            opponent: "random".to_owned(),
            target_games: 2,
            maximum_attempts: 4,
            actors_per_round: 2,
            shard_index: generation,
            first_game_id: generation * 1_000,
            decision_soft_limit: 32,
            actor_seed: generation,
            policy_temperature_bits: 1.0_f32.to_bits(),
            uniform_mix_bits: 0.05_f32.to_bits(),
            neutral_start: None,
        },
        learner: LearnerGenerationPlanV1 {
            batch_size: 8,
            legal_action_capacity: 1_024,
            sampler_seed: generation,
            learning_rate_bits: 1.0e-4_f32.to_bits(),
            target_training_step,
            checkpoint_interval: 1,
            ppo_clip_bits: 0.2_f32.to_bits(),
            ppo_value_weight_bits: 0.5_f32.to_bits(),
            ppo_entropy_weight_bits: 0.01_f32.to_bits(),
        },
        promotion: PromotionGenerationPlanV1 {
            pairs_per_batch: 1,
            maximum_attempted_pairs: 2,
            maximum_eligible_pairs: 1,
            first_pair_id: generation * 1_000,
            decision_soft_limit: 32,
            neutral_start: None,
            sampling_policy: None,
            elo0_bits: 0.0_f64.to_bits(),
            elo1_bits: 10.0_f64.to_bits(),
            alpha_bits: 0.05_f64.to_bits(),
            beta_bits: 0.05_f64.to_bits(),
        },
        promotion_due: true,
        curriculum_evaluation_due: true,
    }
}

fn publish_synthetic_stages(
    archive: &GenerationCampaignArchive,
    plan: &GenerationPlanV1,
    candidate: CheckpointReferenceV1,
    conclusion: PromotionCampaignConclusion,
) {
    let directory = archive.generation_directory(plan.generation);
    write_document(
        &directory.join(ACTOR_STAGE_FILE),
        ACTOR_STAGE_DOCUMENT,
        &ActorGenerationStageV1 {
            run_directory: format!(
                "generations/{}/actors/run",
                generation_name(plan.generation)
            ),
            snapshot: format!(
                "generations/{}/actors/run/snapshot.psrsnap",
                generation_name(plan.generation)
            ),
            snapshot_sha256: "55".repeat(32),
            behavior_producer: "66".repeat(32),
            terminal_games: 2,
            training_examples: 1,
        },
    )
    .unwrap();
    write_document(
        &directory.join(LEARNER_STAGE_FILE),
        LEARNER_STAGE_DOCUMENT,
        &LearnerGenerationStageV1 {
            checkpoint: candidate,
            completed_replay_index: 8,
        },
    )
    .unwrap();
    write_document(
        &directory.join(PROMOTION_STAGE_FILE),
        PROMOTION_STAGE_DOCUMENT,
        &PromotionGenerationStageV1 {
            archive_directory: format!(
                "generations/{}/promotion",
                generation_name(plan.generation)
            ),
            conclusion,
            promotion_source_sha256: "77".repeat(32),
        },
    )
    .unwrap();
}

fn write_checkpoint(
    campaign_root: &Path,
    path: &Path,
    generation: u64,
    training_step: u64,
    replay_index: u64,
) -> CheckpointReferenceV1 {
    write_checkpoint_with_snapshot(
        campaign_root,
        path,
        generation,
        training_step,
        replay_index,
        &"00".repeat(32),
    )
}

fn write_checkpoint_with_snapshot(
    campaign_root: &Path,
    path: &Path,
    generation: u64,
    training_step: u64,
    replay_index: u64,
    replay_snapshot_sha256: &str,
) -> CheckpointReferenceV1 {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let metadata = serde_json::json!({
        "formatVersion": 2,
        "tensorSchema": "paisho-neural-encoding-v1",
        "ruleProfile": "skud-pai-sho-2022-03-14",
        "configuration": {
            "trunkChannels": 20,
            "residualBlocks": 3,
            "policyEmbeddingChannels": 8,
            "valueHiddenChannels": 32,
            "normalizationEpsilon": 1.0e-5
        },
        "optimization": "level1",
        "trainingStep": training_step,
        "progress": {
            "generation": generation,
            "replayIndex": replay_index,
            "replaySnapshotSHA256": replay_snapshot_sha256,
            "scheduler": {
                "learningRate": 1.0e-4,
                "completedSteps": training_step
            }
        }
    });
    let metadata = serde_json::to_vec(&metadata).unwrap();
    let mut bytes = b"PAISHO-CKPT-V2\n".to_vec();
    bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&metadata);
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&digest);
    std::fs::write(path, bytes).unwrap();
    CheckpointReferenceV1::from_checkpoint(campaign_root, path).unwrap()
}

struct TemporaryDirectory {
    path: PathBuf,
}

impl TemporaryDirectory {
    fn new() -> Self {
        let ordinal = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "paisho-generation-{}-{ordinal}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        Self { path }
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
