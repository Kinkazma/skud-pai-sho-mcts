//! Durable orchestration for replay-driven MPSGraph training.

mod actor;
mod atomic_file;
mod commit;
pub mod compact_compare;
pub mod compact_learning;
pub mod micro_learning;
pub mod compact_selfplay;
mod curriculum;
mod curriculum_archive;
mod dashboard;
mod evaluation;
mod evaluation_archive;
mod evaluation_value_diagnostics;
mod generation;
mod generation_bootstrap;
pub mod highlighted_games;
mod learner;
mod learner_metrics;
pub mod live_actors;
pub mod live_learner;
mod neutral_start;
mod objective;
mod origin;
mod policy_diagnostics;
mod policy_update_diagnostics;
mod promotion_archive;
mod promotion_match;
mod replay_stats;
mod value_diagnostics;

pub use actor::{
    play_replay_match, play_replay_parallel, MctsReplayAgent, ParallelReplayResults,
    PlayedActionReplayAgent, RecordedNetworkAgent, ReplayActorAgent, ReplayActorChoice,
    ReplayMatchConfiguration, ReplayMatchError, ReplayMatchResult, ReplayMatchTask,
    UnrecordedReplayAgent,
};
pub use commit::{discover_latest_commit, LearnerCommit, LearnerCommitError, LearnerIdentityV1};
pub use curriculum::{
    assess_curriculum_evaluation, deferred_evaluation_assessment, self_play_assessment,
    CurriculumActionV1, CurriculumAssessmentV1, CurriculumDecisionError, CurriculumEloWindowV1,
    CurriculumReasonV1, CurriculumTierV1,
};
pub use curriculum_archive::{
    CurriculumArchiveError, CurriculumCampaignArchive, CurriculumCampaignIdentityV1,
    CurriculumDecisionV1, CurriculumEvaluationEvidenceV1, CurriculumEvidenceV1, CurriculumStateV1,
};
pub use dashboard::{
    build_dashboard_snapshot, render_dashboard_html, write_dashboard_html, DashboardCheckpointV1,
    DashboardEloObservationV1, DashboardEloSeriesV1, DashboardError, DashboardGameCountsV1,
    DashboardGenerationV1, DashboardSnapshotV1, DashboardVerificationV1,
};
pub use evaluation::{
    contextual_elo_from_score, evaluation_mcts_configuration, sha256_text, EvaluationConclusion,
    EvaluationIdentityError, EvaluationInferenceClassV1, EvaluationOpponentAgent,
    EvaluationOpponentV1, EvaluationProgress, EvaluationRunIdentityV1,
};
pub use evaluation_archive::{
    ContextualEloPointV1, EvaluationAnalysisV1, EvaluationArchiveError, EvaluationCampaignArchive,
    EvaluationEstimateV1, EvaluationLowerWindowV1, EvaluationMleV1, EvaluationUncertaintyV1,
};
pub use evaluation_value_diagnostics::{
    materialize_candidate_value_examples_v1, CandidateValueExampleV1,
    EvaluationValueMaterializationError,
};
pub use generation::{
    ActorGenerationPlanV1, ActorGenerationStageV1, CampaignIdentityV1, CheckpointReferenceV1,
    ExternalFileReferenceV1, GenerationArchiveError, GenerationCampaignArchive,
    GenerationCampaignChainV1, GenerationInferenceClassV1, GenerationOutcomeV1, GenerationPlanV1,
    GenerationRolesV1, LearnerGenerationPlanV1, LearnerGenerationStageV1,
    NeutralStartGenerationPlanV1, PromotionGenerationPlanV1, PromotionGenerationStageV1,
};
pub use generation_bootstrap::{
    open_or_initialize_generation_campaign, GenerationBootstrapConfiguration,
};
pub use highlighted_games::{
    export_highlighted_game, select_highlighted_game, write_highlighted_game,
    HighlightedGameExport, HighlightedGameMetadata, HighlightedGameSelection,
    HighlightedGamesError, HighlightedGamesMetadata,
};
pub use learner::{run_learner, LearnerConfiguration, LearnerError, LearnerReport};
pub use learner_metrics::{
    read_terminal_ppo_metric_segment_v1, LearnerMetricsError, ScalarMetricSummaryV1,
    TerminalPpoBatchMetricsV1, TerminalPpoMetricSegmentV1, TerminalPpoMetricsSummaryV1,
};
pub use live_actors::{
    collect_live_games, LiveActorsConfiguration, LiveActorsError, LiveActorsOpponent,
    LiveActorsReport,
};
pub use neutral_start::{
    NeutralStartConfigurationV1, NeutralStartError, NeutralStartProvenanceV1, NeutralStartV1,
    NEUTRAL_START_POLICY_V1,
};
pub use objective::{
    LearnerObjectiveConfiguration, LearnerObjectiveV1, TerminalPpoLearnerObjectiveV1,
    SUPERVISED_OBJECTIVE_V1,
};
pub use origin::{LearnerOriginV1, LearnerOriginV1Error, ParentCheckpointV1};
pub use policy_diagnostics::{
    observe_sampling_policy_v1, sampling_policy_probabilities_v1, summarize_policy_observations_v1,
    DiagnosticRangeV1, PolicyDiagnosticsError, PolicyDistributionObservationV1,
    PolicyDistributionSummaryV1, SamplingProfileV1,
};
pub use policy_update_diagnostics::{
    observe_played_policy_update_v1, summarize_policy_updates_v1, PolicyUpdateClassSummaryV1,
    PolicyUpdateDiagnosticsError, PolicyUpdateObservationV1, PolicyUpdateSummaryV1,
};
pub use promotion_archive::{
    PromotionArchiveError, PromotionCampaignArchive, PromotionCampaignConclusion,
    PromotionCampaignProgress, PromotionInferenceClassV1, PromotionNeutralStartV1,
    PromotionRunIdentityV1, PromotionSamplingPolicyV1,
};
pub use promotion_match::{
    build_promotion_schedule, build_promotion_schedule_with_neutral_starts, run_promotion_schedule,
    PairExclusionReason, PlayedPromotionGame, PromotionBatchExecution, PromotionBatchSummary,
    PromotionMatchError, PromotionPairEvaluation, PromotionScheduleError, PromotionScheduledGame,
};
pub use replay_stats::{
    analyze_behavior_replay_v1, BehaviorReplayStatisticsV1, ReplayClassStatisticsV1,
    ReplayRangeStatisticsV1, ReplayStatisticsError,
};
pub use value_diagnostics::{
    summarize_value_predictions_v1, ValueClassDiagnosticsV1, ValueDiagnosticsError,
    ValuePredictionObservationV1, ValuePredictionSummaryV1,
};

pub mod gen32;
