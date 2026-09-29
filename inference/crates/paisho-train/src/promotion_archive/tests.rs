use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use paisho_ai::{Agent, AgentError, AgentTelemetry, MatchConfig, MatchResult, MatchTermination};
use paisho_core::{legal_actions, Action, GameRecord, Player, Position};

use super::*;
use crate::{
    build_promotion_schedule, build_promotion_schedule_with_neutral_starts, run_promotion_schedule,
    PlayedPromotionGame,
};

static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

fn temporary_root(name: &str) -> PathBuf {
    let ordinal = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "paisho-promotion-{name}-{}-{ordinal}",
        std::process::id()
    ))
}

fn identity(maximum_attempted_pairs: u64, maximum_eligible_pairs: u64) -> PromotionRunIdentityV1 {
    PromotionRunIdentityV1 {
        source_sha256: "11".repeat(32),
        source_revision: "test-revision".to_owned(),
        source_dirty: false,
        service_sha256: "22".repeat(32),
        candidate_checkpoint_sha256: "33".repeat(32),
        champion_checkpoint_sha256: "44".repeat(32),
        preset: "pure".to_owned(),
        optimization_level: 1,
        inference_classes: vec![PromotionInferenceClassV1 {
            legal_action_capacity: 1_024,
            batch_size: 1,
        }],
        workers: 1,
        pairs_per_batch: 1,
        maximum_attempted_pairs,
        maximum_eligible_pairs,
        first_pair_id: 0,
        decision_soft_limit: 2_048,
        maximum_batch_wait_microseconds: 5_000,
        model_seed: 17,
        neutral_start: None,
        sampling_policy: None,
        elo0: 0.0,
        elo1: 50.0,
        alpha: 0.49,
        beta: 0.49,
    }
}

fn terminal_result(
    scheduled: crate::PromotionScheduledGame,
    record_text: &str,
) -> PlayedPromotionGame {
    let record: GameRecord = record_text.parse().unwrap();
    assert_eq!(record.setup(), scheduled.task.setup);
    let final_position = record.replay().unwrap();
    let (host_decisions, guest_decisions) = record_decision_counts(&record);
    let task_id = scheduled.task.id;
    PlayedPromotionGame {
        scheduled,
        result: Ok(MatchResult {
            task_id,
            termination: MatchTermination::Rules(final_position.outcome()),
            final_position,
            record,
            host_telemetry: AgentTelemetry {
                decisions: host_decisions,
                ..AgentTelemetry::default()
            },
            guest_telemetry: AgentTelemetry {
                decisions: guest_decisions,
                ..AgentTelemetry::default()
            },
        }),
    }
}

fn record_decision_counts(record: &GameRecord) -> (usize, usize) {
    let mut position = record.initial_position();
    let mut counts = [0_usize; 2];
    for action in record.actions() {
        let player = position.to_move();
        position.apply(*action).unwrap();
        counts[player.index()] += 1;
    }
    (counts[Player::Host.index()], counts[Player::Guest.index()])
}

#[test]
fn terminal_batch_resumes_and_publishes_an_idempotent_decision() {
    const HOST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002004-high-host.psr"
    );
    const GUEST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002004-high-guest.psr"
    );
    let root = temporary_root("terminal");
    let mut identity = identity(4, 4);
    identity.workers = 2; // Scheduling need not occupy every available worker.
    let archive = PromotionCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
    let schedule = build_promotion_schedule(0, 0, 1).unwrap();
    let execution = PromotionBatchExecution {
        games: vec![
            terminal_result(schedule[0].clone(), HOST_WIN),
            terminal_result(schedule[1].clone(), GUEST_WIN),
        ],
        elapsed_seconds: 1.25,
        observed_match_workers: 1,
        worker_capacity: 2,
    };
    let progress = archive
        .publish_next_batch(&PromotionCampaignProgress::default(), &execution)
        .unwrap();
    assert_eq!(progress.attempted_pairs, 1);
    assert_eq!(progress.eligible_pairs, 1);
    assert_eq!(archive.load_progress_integrity().unwrap(), progress);
    assert_eq!(progress.pentanomial.bins(), [0, 0, 0, 0, 1]);
    assert_eq!(
        progress.conclusion(&identity).unwrap(),
        Some(PromotionCampaignConclusion::PromoteCandidate)
    );
    assert_eq!(
        archive.publish_conclusion(&progress).unwrap(),
        PromotionCampaignConclusion::PromoteCandidate
    );
    assert_eq!(
        archive.publish_conclusion(&progress).unwrap(),
        PromotionCampaignConclusion::PromoteCandidate
    );
    archive
        .require_published_conclusion(&progress, PromotionCampaignConclusion::PromoteCandidate)
        .unwrap();
    fs::remove_dir_all(root.join(DECISION_DIRECTORY)).unwrap();
    assert_eq!(archive.load_progress().unwrap(), progress);
    assert!(archive
        .require_published_conclusion(&progress, PromotionCampaignConclusion::PromoteCandidate)
        .is_err());
    archive.publish_conclusion(&progress).unwrap();

    let reopened = PromotionCampaignArchive::open_or_create(&root, identity).unwrap();
    assert_eq!(reopened.load_progress().unwrap(), progress);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn unfinished_test_resumes_at_the_next_pair_without_replaying_its_first_batch() {
    const R3_HOST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002004-high-host.psr"
    );
    const R3_GUEST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002004-high-guest.psr"
    );
    const R4_HOST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002005-high-host.psr"
    );
    const R4_GUEST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002005-high-guest.psr"
    );
    let root = temporary_root("resume-next-pair");
    let mut identity = identity(2, 2);
    identity.elo1 = 10.0;
    identity.alpha = 0.05;
    identity.beta = 0.05;
    let archive = PromotionCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
    let first_schedule = build_promotion_schedule(0, 0, 1).unwrap();
    let first = PromotionBatchExecution {
        games: vec![
            terminal_result(first_schedule[0].clone(), R3_HOST_WIN),
            terminal_result(first_schedule[1].clone(), R3_GUEST_WIN),
        ],
        elapsed_seconds: 1.0,
        observed_match_workers: 1,
        worker_capacity: 1,
    };
    let first_progress = archive
        .publish_next_batch(&PromotionCampaignProgress::default(), &first)
        .unwrap();
    assert_eq!(first_progress.next_pair_id(&identity).unwrap(), 1);
    assert_eq!(first_progress.conclusion(&identity).unwrap(), None);

    let reopened = PromotionCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
    let second_schedule = build_promotion_schedule(0, 1, 1).unwrap();
    let second = PromotionBatchExecution {
        games: vec![
            terminal_result(second_schedule[0].clone(), R4_HOST_WIN),
            terminal_result(second_schedule[1].clone(), R4_GUEST_WIN),
        ],
        elapsed_seconds: 1.0,
        observed_match_workers: 1,
        worker_capacity: 1,
    };
    let final_progress = reopened
        .publish_next_batch(&first_progress, &second)
        .unwrap();
    assert_eq!(final_progress.batches, 2);
    assert_eq!(final_progress.attempted_pairs, 2);
    assert_eq!(final_progress.eligible_pairs, 2);
    assert_eq!(final_progress.pentanomial.bins(), [0, 0, 0, 0, 2]);
    assert_eq!(
        final_progress.conclusion(&identity).unwrap(),
        Some(PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs)
    );
    fs::remove_dir_all(root).unwrap();
}

#[derive(Default)]
struct FirstLegalAgent {
    decisions: usize,
}

#[test]
fn integrity_reopen_accepts_a_published_batch_larger_than_the_scheduling_hint() {
    let root = temporary_root("large-published-batch");
    let mut identity = identity(2, 2);
    identity.decision_soft_limit = 1;
    let archive = PromotionCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
    let schedule = build_promotion_schedule(0, 0, 2).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let execution = pool.install(|| {
        run_promotion_schedule(
            &schedule,
            MatchConfig {
                decision_soft_limit: 1,
            },
            |_, _| FirstLegalAgent::default(),
            |_, _| FirstLegalAgent::default(),
        )
    });
    let progress = archive
        .publish_next_batch(&PromotionCampaignProgress::default(), &execution)
        .unwrap();
    assert_eq!(progress.attempted_pairs, 2);
    assert_eq!(identity.pairs_per_batch, 1);
    let (_, reopened) = PromotionCampaignArchive::open_existing_integrity(&root).unwrap();
    assert_eq!(reopened, progress);
    fs::remove_dir_all(root).unwrap();
}

impl Agent for FirstLegalAgent {
    fn select_action(
        &mut self,
        _position: &Position,
        actions: &[Action],
    ) -> Result<usize, AgentError> {
        assert!(!actions.is_empty());
        self.decisions += 1;
        Ok(0)
    }

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry {
            decisions: self.decisions,
            ..AgentTelemetry::default()
        }
    }

    fn reset_telemetry(&mut self) {
        self.decisions = 0;
    }
}

#[test]
fn interrupted_pair_is_excluded_and_record_corruption_blocks_resume() {
    let root = temporary_root("interrupted");
    let mut identity = identity(1, 1);
    identity.decision_soft_limit = 1;
    identity.alpha = 0.05;
    identity.beta = 0.05;
    let archive = PromotionCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
    let schedule = build_promotion_schedule(0, 0, 1).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let execution = pool.install(|| {
        run_promotion_schedule(
            &schedule,
            MatchConfig {
                decision_soft_limit: 1,
            },
            |_, _| FirstLegalAgent::default(),
            |_, _| FirstLegalAgent::default(),
        )
    });
    for game in &execution.games {
        let result = game.result.as_ref().unwrap();
        assert_eq!(result.termination, MatchTermination::DecisionLimit);
        assert!(!legal_actions(&result.final_position).is_empty());
    }
    let progress = archive
        .publish_next_batch(&PromotionCampaignProgress::default(), &execution)
        .unwrap();
    assert_eq!(progress.eligible_pairs, 0);
    assert_eq!(progress.excluded_pairs, 1);
    assert_eq!(
        progress.conclusion(&identity).unwrap(),
        Some(PromotionCampaignConclusion::InconclusiveMaximumAttemptedPairs)
    );
    archive.publish_conclusion(&progress).unwrap();

    let record =
        root.join("batches/batch-00000000000000000000/records/game-00000000000000000000.psr");
    fs::write(record, "damaged\n").unwrap();
    assert!(archive.load_progress().is_err());
    assert!(PromotionCampaignArchive::open_existing_integrity(&root).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn neutral_start_prefixes_survive_semantic_archive_reopen() {
    let root = temporary_root("neutral-prefix");
    let mut identity = identity(1, 1);
    identity.decision_soft_limit = 1;
    identity.neutral_start = Some(PromotionNeutralStartV1 {
        target_remaining_decisions: 32,
        seed: 71,
        source_decision_limit: 4_096,
        maximum_source_attempts: 16,
    });
    let archive = PromotionCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
    let schedule = build_promotion_schedule_with_neutral_starts(
        0,
        0,
        1,
        identity.neutral_start_configuration().unwrap(),
    )
    .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let execution = pool.install(|| {
        run_promotion_schedule(
            &schedule,
            MatchConfig {
                decision_soft_limit: 1,
            },
            |_, _| FirstLegalAgent::default(),
            |_, _| FirstLegalAgent::default(),
        )
    });
    for game in &execution.games {
        let result = game.result.as_ref().unwrap();
        assert!(result
            .record
            .actions()
            .starts_with(game.scheduled.starting_record.actions()));
        assert!(result.record.actions().len() > game.scheduled.starting_record.actions().len());
        assert_eq!(
            result.host_telemetry.decisions + result.guest_telemetry.decisions,
            result.record.actions().len() - game.scheduled.starting_record.actions().len()
        );
    }
    let expected = execution.summary().unwrap();
    let progress = archive
        .publish_next_batch(&PromotionCampaignProgress::default(), &execution)
        .unwrap();
    assert_eq!(progress.attempted_pairs, 1);
    assert_eq!(progress.eligible_pairs, expected.eligible_pairs as u64);
    assert_eq!(progress.excluded_pairs, expected.excluded_pairs as u64);
    assert_eq!(archive.load_progress().unwrap(), progress);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn identity_rejects_unsorted_classes_and_budget_mismatch() {
    let mut value = identity(2, 1);
    value.inference_classes = vec![
        PromotionInferenceClassV1 {
            legal_action_capacity: 128,
            batch_size: 4,
        },
        PromotionInferenceClassV1 {
            legal_action_capacity: 64,
            batch_size: 8,
        },
    ];
    assert!(value.validate().is_err());
    value
        .inference_classes
        .sort_by_key(|class| class.legal_action_capacity);
    value.maximum_eligible_pairs = 3;
    assert!(value.validate().is_err());
    assert!(PromotionSamplingPolicyV1::new(0.0, 0.05).is_err());
    assert!(PromotionSamplingPolicyV1::new(1.0, 1.01).is_err());
}
