use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use paisho_ai::{Agent, AgentError, AgentTelemetry, MatchConfig, MatchResult, MatchTermination};
use paisho_core::{Action, GameRecord, Player, Position};

use super::*;
use crate::{
    build_promotion_schedule, run_promotion_schedule, sha256_text, EvaluationInferenceClassV1,
    EvaluationOpponentV1,
};

static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

fn temporary_root(name: &str) -> PathBuf {
    let ordinal = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "paisho-evaluation-{name}-{}-{ordinal}",
        std::process::id()
    ))
}

fn identity(maximum_attempted_pairs: u64, maximum_eligible_pairs: u64) -> EvaluationRunIdentityV1 {
    let source_sha256 = "11".repeat(32);
    let opponent = EvaluationOpponentV1::Random;
    let opponent_descriptor = opponent.descriptor(&source_sha256);
    EvaluationRunIdentityV1 {
        source_sha256,
        source_revision: "test-revision".to_owned(),
        source_dirty: false,
        service_sha256: "22".repeat(32),
        candidate_checkpoint_sha256: "33".repeat(32),
        opponent,
        opponent_sha256: sha256_text(&opponent_descriptor),
        opponent_descriptor,
        preset: "pure".to_owned(),
        optimization_level: 1,
        inference_classes: vec![EvaluationInferenceClassV1 {
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
        lower_elo: None,
        alpha: 0.49,
        beta: 0.49,
    }
}

fn terminal_result(
    scheduled: crate::PromotionScheduledGame,
    record_text: &str,
) -> crate::PlayedPromotionGame {
    let record: GameRecord = record_text.parse().unwrap();
    assert_eq!(record.setup(), scheduled.task.setup);
    let final_position = record.replay().unwrap();
    let (host_decisions, guest_decisions) = record_decision_counts(&record);
    crate::PlayedPromotionGame {
        result: Ok(MatchResult {
            task_id: scheduled.task.id,
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
        scheduled,
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
fn terminal_batch_reopens_and_publishes_replayable_analysis() {
    const HOST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002004-high-host.psr"
    );
    const GUEST_WIN: &str = include_str!(
        "../../../../benchmarks/results/mcts-128-vs-32-2156132-block-2000/records/pair-00000000000000002004-high-guest.psr"
    );
    let root = temporary_root("terminal");
    // Two games may both run on one worker of a two-worker pool.
    let mut identity = identity(1, 1);
    identity.workers = 2;
    let archive = EvaluationCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
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
        .publish_next_batch(&EvaluationProgress::default(), &execution)
        .unwrap();
    assert_eq!(progress.attempted_pairs, 1);
    assert_eq!(progress.eligible_pairs, 1);
    assert_eq!(archive.load_progress_integrity().unwrap(), progress);
    assert_eq!(progress.pentanomial.bins(), [0, 0, 0, 0, 1]);
    let analysis = archive.publish_conclusion(&progress).unwrap();
    assert_eq!(analysis.candidate_wins, 2);
    assert_eq!(analysis.candidate_losses, 0);
    assert_eq!(
        analysis.contextual_elo_point,
        ContextualEloPointV1::PositiveInfinity
    );
    assert!(analysis.mle.is_none());
    assert!(analysis.mle_error.is_some());

    let reopened = EvaluationCampaignArchive::open_or_create(&root, identity).unwrap();
    assert_eq!(reopened.load_progress().unwrap(), progress);
    assert_eq!(reopened.analysis().unwrap(), analysis);
    assert_eq!(reopened.publish_conclusion(&progress).unwrap(), analysis);
    fs::remove_dir_all(root).unwrap();
}

#[derive(Default)]
struct FirstLegalAgent {
    decisions: usize,
}

impl Agent for FirstLegalAgent {
    fn select_action(
        &mut self,
        _position: &Position,
        _legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
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
fn interrupted_pair_is_preserved_but_never_rated() {
    let root = temporary_root("interrupted");
    let mut identity = identity(1, 1);
    identity.decision_soft_limit = 1;
    let archive = EvaluationCampaignArchive::open_or_create(&root, identity.clone()).unwrap();
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
            |_scheduled, _player| FirstLegalAgent::default(),
            |_scheduled, _player| FirstLegalAgent::default(),
        )
    });
    let progress = archive
        .publish_next_batch(&EvaluationProgress::default(), &execution)
        .unwrap();
    assert_eq!(progress.eligible_pairs, 0);
    assert_eq!(progress.excluded_pairs, 1);
    assert_eq!(
        progress.conclusion(&identity).unwrap(),
        Some(EvaluationConclusion::InconclusiveMaximumAttemptedPairs)
    );
    let analysis = archive.publish_conclusion(&progress).unwrap();
    assert_eq!(analysis.empirical_score, None);
    assert_eq!(
        analysis.contextual_elo_point,
        ContextualEloPointV1::NoRatedGames
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn changed_batch_bytes_are_rejected_by_the_manifest() {
    let root = temporary_root("corruption");
    let mut identity = identity(1, 1);
    identity.decision_soft_limit = 1;
    let archive = EvaluationCampaignArchive::open_or_create(&root, identity).unwrap();
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
            |_scheduled, _player| FirstLegalAgent::default(),
            |_scheduled, _player| FirstLegalAgent::default(),
        )
    });
    archive
        .publish_next_batch(&EvaluationProgress::default(), &execution)
        .unwrap();
    let batch_path = root
        .join(BATCHES_DIRECTORY)
        .join(batch_name(0))
        .join("batch.json");
    let mut bytes = fs::read(&batch_path).unwrap();
    bytes.push(b' ');
    fs::write(batch_path, bytes).unwrap();
    assert!(EvaluationCampaignArchive::open_existing(&root).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn identity_rejects_a_forged_opponent_digest() {
    let mut identity = identity(1, 1);
    identity.opponent_sha256 = "44".repeat(32);
    assert!(identity.validate().is_err());
}
