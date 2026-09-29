use core::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use paisho_ai::{AgentTelemetry, MatchResult, MatchTermination};
use paisho_core::{GameOutcome, GameRecord, Player, TurnPhase};
use paisho_rating::{
    evaluate_promotion_sprt, fit_davidson_for_agents, DavidsonOptions, ParameterEstimate,
    PentanomialCounts, RatedGame, RatedOutcome,
};
use serde::{Deserialize, Serialize};

use crate::{
    build_promotion_schedule_with_neutral_starts, contextual_elo_from_score, EvaluationConclusion,
    EvaluationIdentityError, EvaluationProgress, EvaluationRunIdentityV1, PromotionBatchExecution,
    PromotionMatchError, PromotionScheduleError,
};

mod storage;

use storage::{
    sha256_hex, sync_tree, temporary_directory, verify_manifest, write_json_new, write_manifest,
    write_new_synced,
};

const RUN_FORMAT: &str = "PAISHO-EVALUATION-RUN-1";
const BATCH_FORMAT: &str = "PAISHO-EVALUATION-BATCH-1";
const RESULT_FORMAT: &str = "PAISHO-EVALUATION-RESULT-1";
const RUN_FILE: &str = "run.json";
const RUN_DIGEST_FILE: &str = "run.sha256";
const BATCHES_DIRECTORY: &str = "batches";
const RESULT_DIRECTORY: &str = "result";

pub struct EvaluationCampaignArchive {
    root: PathBuf,
    identity: EvaluationRunIdentityV1,
}

impl EvaluationCampaignArchive {
    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self, EvaluationArchiveError> {
        let root = root.as_ref().to_owned();
        let identity = read_run_identity(&root)?;
        identity.validate()?;
        let archive = Self { root, identity };
        archive.load_progress()?;
        Ok(archive)
    }

    pub fn open_existing_integrity(root: impl AsRef<Path>) -> Result<Self, EvaluationArchiveError> {
        let root = root.as_ref().to_owned();
        let identity = read_run_identity(&root)?;
        identity.validate()?;
        let archive = Self { root, identity };
        archive.load_progress_integrity()?;
        Ok(archive)
    }

    pub fn open_or_create(
        root: impl AsRef<Path>,
        identity: EvaluationRunIdentityV1,
    ) -> Result<Self, EvaluationArchiveError> {
        identity.validate()?;
        let root = root.as_ref().to_owned();
        if root.exists() {
            if read_run_identity(&root)? != identity {
                return Err(invalid(
                    "existing evaluation archive has a different immutable identity",
                ));
            }
        } else {
            initialize_archive(&root, &identity)?;
        }
        let archive = Self { root, identity };
        archive.load_progress()?;
        Ok(archive)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn identity(&self) -> &EvaluationRunIdentityV1 {
        &self.identity
    }

    pub fn load_progress(&self) -> Result<EvaluationProgress, EvaluationArchiveError> {
        self.load_state(true).map(|state| state.progress)
    }

    fn load_progress_integrity(&self) -> Result<EvaluationProgress, EvaluationArchiveError> {
        self.load_state(false).map(|state| state.progress)
    }

    pub fn analysis(&self) -> Result<EvaluationAnalysisV1, EvaluationArchiveError> {
        let state = self.load_state(true)?;
        build_analysis(&self.identity, &state.progress, &state.rated_games)
    }

    pub fn published_analysis_integrity(
        &self,
    ) -> Result<EvaluationAnalysisV1, EvaluationArchiveError> {
        self.load_progress_integrity()?;
        read_result(&self.root)?
            .map(|result| result.analysis)
            .ok_or_else(|| invalid("evaluation has no published result"))
    }

    pub fn publish_next_batch(
        &self,
        progress: &EvaluationProgress,
        execution: &PromotionBatchExecution,
    ) -> Result<EvaluationProgress, EvaluationArchiveError> {
        let current = self.load_progress_integrity()?;
        if &current != progress {
            return Err(invalid(
                "evaluation progress changed before batch publication",
            ));
        }
        if current.conclusion(&self.identity)?.is_some() {
            return Err(invalid("evaluation campaign already reached a conclusion"));
        }
        let summary = execution.summary()?;
        let attempted_pairs = u64::try_from(summary.attempted_pairs)
            .map_err(|_| invalid("batch pair count does not fit u64"))?;
        let eligible_pairs = u64::try_from(summary.eligible_pairs)
            .map_err(|_| invalid("eligible batch pair count does not fit u64"))?;
        if attempted_pairs == 0
            || current
                .attempted_pairs
                .checked_add(attempted_pairs)
                .map_or(true, |total| total > self.identity.maximum_attempted_pairs)
            || current
                .eligible_pairs
                .checked_add(eligible_pairs)
                .map_or(true, |total| total > self.identity.maximum_eligible_pairs)
        {
            return Err(invalid(
                "evaluation batch exceeds the configured campaign budget",
            ));
        }
        let first_pair_id = current.next_pair_id(&self.identity)?;
        validate_execution_schedule(execution, &self.identity, first_pair_id)?;
        let batch = archived_batch(current.batches, first_pair_id, execution)?;
        publish_batch_directory(&self.root, &batch)?;
        let published_path = self
            .root
            .join(BATCHES_DIRECTORY)
            .join(batch_name(batch.batch_index));
        let verified = read_batch(
            &published_path,
            &self.identity,
            batch.batch_index,
            first_pair_id,
        )?;
        if verified.batch.batch_index != batch.batch_index
            || verified.batch.first_pair_id != batch.first_pair_id
        {
            return Err(invalid(
                "published evaluation batch identity changed during verification",
            ));
        }
        let mut next = current;
        absorb_batch(&mut next, &batch)?;
        Ok(next)
    }

    pub fn publish_conclusion(
        &self,
        progress: &EvaluationProgress,
    ) -> Result<EvaluationAnalysisV1, EvaluationArchiveError> {
        let state = self.load_state(true)?;
        if &state.progress != progress {
            return Err(invalid(
                "evaluation progress changed before result publication",
            ));
        }
        let analysis = build_analysis(&self.identity, progress, &state.rated_games)?;
        let final_directory = self.root.join(RESULT_DIRECTORY);
        if final_directory.exists() {
            read_result(&self.root)?
                .ok_or_else(|| invalid("evaluation result directory is unreadable"))?;
            if result_bytes_match(&self.root, &analysis)? {
                return Ok(analysis);
            }
            return Err(invalid(
                "evaluation result is already published with other contents",
            ));
        }
        let temporary = temporary_directory(&self.root, ".result.partial")?;
        write_json_new(
            &temporary.path.join("result.json"),
            &ResultFileV1 {
                format: RESULT_FORMAT.to_owned(),
                analysis: analysis.clone(),
            },
        )?;
        write_manifest(&temporary.path)?;
        verify_manifest(&temporary.path)?;
        sync_tree(&temporary.path)?;
        temporary.publish(&final_directory)?;
        read_result(&self.root)?
            .ok_or_else(|| invalid("published evaluation result disappeared"))?;
        if !result_bytes_match(&self.root, &analysis)? {
            return Err(invalid("published evaluation result failed verification"));
        }
        Ok(analysis)
    }
}

struct LoadedState {
    progress: EvaluationProgress,
    rated_games: Vec<RatedGame>,
}

impl EvaluationCampaignArchive {
    fn load_state(&self, replay_records: bool) -> Result<LoadedState, EvaluationArchiveError> {
        if read_run_identity(&self.root)? != self.identity {
            return Err(invalid("evaluation identity changed after opening"));
        }
        let mut paths = Vec::new();
        for entry in fs::read_dir(self.root.join(BATCHES_DIRECTORY))? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Some(index) = parse_batch_name(&name) else {
                return Err(invalid(format!("unexpected evaluation batch entry {name}")));
            };
            if !entry.file_type()?.is_dir() {
                return Err(invalid(format!(
                    "evaluation batch {name} is not a directory"
                )));
            }
            paths.push((index, entry.path()));
        }
        paths.sort_by_key(|(index, _)| *index);
        let mut progress = EvaluationProgress::default();
        let mut rated_games = Vec::new();
        for (expected, (actual, path)) in paths.into_iter().enumerate() {
            let expected = u64::try_from(expected)
                .map_err(|_| invalid("evaluation batch index does not fit u64"))?;
            if actual != expected {
                return Err(invalid("evaluation batch indices are not contiguous"));
            }
            let first_pair_id = progress.next_pair_id(&self.identity)?;
            let loaded = if replay_records {
                read_batch(&path, &self.identity, actual, first_pair_id)?
            } else {
                LoadedBatch {
                    batch: read_batch_integrity(&path, &self.identity, actual, first_pair_id)?,
                    rated_games: Vec::new(),
                }
            };
            absorb_batch(&mut progress, &loaded.batch)?;
            rated_games.extend(loaded.rated_games);
        }
        let expected_conclusion = progress.conclusion(&self.identity)?;
        if let Some(stored) = read_result(&self.root)? {
            let expected = expected_conclusion
                .ok_or_else(|| invalid("evaluation result exists before a stopping condition"))?;
            if stored.analysis.conclusion != expected
                || stored.analysis.attempted_pairs != progress.attempted_pairs
                || stored.analysis.eligible_pairs != progress.eligible_pairs
                || stored.analysis.excluded_pairs != progress.excluded_pairs
                || stored.analysis.pentanomial != progress.pentanomial.bins()
            {
                return Err(invalid("evaluation result differs from its batches"));
            }
            if replay_records {
                let expected_analysis = build_analysis(&self.identity, &progress, &rated_games)?;
                if !result_bytes_match(&self.root, &expected_analysis)? {
                    return Err(invalid(
                        "evaluation result differs from replayed rating evidence",
                    ));
                }
            }
        }
        Ok(LoadedState {
            progress,
            rated_games,
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RunFileV1 {
    format: String,
    identity: EvaluationRunIdentityV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct BatchFileV1 {
    format: String,
    batch_index: u64,
    first_pair_id: u64,
    attempted_pairs: u64,
    eligible_pairs: u64,
    excluded_pairs: u64,
    pentanomial: [u64; 5],
    elapsed_seconds: f64,
    observed_match_workers: usize,
    worker_capacity: usize,
    candidate_inference_positions: u64,
    games: Vec<ArchivedGameV1>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ArchivedGameV1 {
    game_id: u64,
    pair_id: u64,
    leg: u8,
    candidate_is_host: bool,
    termination: Option<ArchivedTermination>,
    record: Option<String>,
    error: Option<String>,
    host_telemetry: Option<ArchivedTelemetryV1>,
    guest_telemetry: Option<ArchivedTelemetryV1>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum ArchivedTermination {
    HostWin,
    GuestWin,
    Draw,
    DecisionLimit,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct ArchivedTelemetryV1 {
    decisions: usize,
    simulations: usize,
    evaluated_actions: usize,
    expanded_nodes: usize,
    generated_nodes: usize,
    generated_actions: usize,
    maximum_search_depth: usize,
    maximum_search_trees: usize,
    maximum_search_workers: usize,
    maximum_search_worker_capacity: usize,
    maximum_action_ranking_workers: usize,
    maximum_action_ranking_worker_capacity: usize,
    rollout_steps: usize,
}

impl From<AgentTelemetry> for ArchivedTelemetryV1 {
    fn from(value: AgentTelemetry) -> Self {
        Self {
            decisions: value.decisions,
            simulations: value.simulations,
            evaluated_actions: value.evaluated_actions,
            expanded_nodes: value.expanded_nodes,
            generated_nodes: value.generated_nodes,
            generated_actions: value.generated_actions,
            maximum_search_depth: value.maximum_search_depth,
            maximum_search_trees: value.maximum_search_trees,
            maximum_search_workers: value.maximum_search_workers,
            maximum_search_worker_capacity: value.maximum_search_worker_capacity,
            maximum_action_ranking_workers: value.maximum_action_ranking_workers,
            maximum_action_ranking_worker_capacity: value.maximum_action_ranking_worker_capacity,
            rollout_steps: value.rollout_steps,
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct ResultFileV1 {
    format: String,
    analysis: EvaluationAnalysisV1,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationUncertaintyV1 {
    pub standard_error: f64,
    pub interval_95_lower: f64,
    pub interval_95_upper: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationEstimateV1 {
    pub estimate: f64,
    pub model: EvaluationUncertaintyV1,
    pub paired_cluster: Option<EvaluationUncertaintyV1>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationMleV1 {
    pub candidate_minus_opponent: EvaluationEstimateV1,
    pub host_advantage_elo: Option<EvaluationEstimateV1>,
    pub draw_weight: f64,
    pub tie_model: String,
    pub log_likelihood: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ContextualEloPointV1 {
    NoRatedGames,
    NegativeInfinity,
    Finite { elo: f64 },
    PositiveInfinity,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationAnalysisV1 {
    pub conclusion: EvaluationConclusion,
    pub attempted_pairs: u64,
    pub eligible_pairs: u64,
    pub excluded_pairs: u64,
    pub pentanomial: [u64; 5],
    pub candidate_wins: u64,
    pub draws: u64,
    pub candidate_losses: u64,
    pub empirical_score: Option<f64>,
    pub contextual_elo_point: ContextualEloPointV1,
    pub mle: Option<EvaluationMleV1>,
    pub mle_error: Option<String>,
    pub elo0: f64,
    pub elo1: f64,
    pub log_likelihood_ratio: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_window: Option<EvaluationLowerWindowV1>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationLowerWindowV1 {
    pub lower_elo: f64,
    pub center_elo: f64,
    pub log_likelihood_ratio: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

fn initialize_archive(
    root: &Path,
    identity: &EvaluationRunIdentityV1,
) -> Result<(), EvaluationArchiveError> {
    let parent = root.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = temporary_directory(parent, ".evaluation-run.initializing")?;
    fs::create_dir(temporary.path.join(BATCHES_DIRECTORY))?;
    write_json_new(
        &temporary.path.join(RUN_FILE),
        &RunFileV1 {
            format: RUN_FORMAT.to_owned(),
            identity: identity.clone(),
        },
    )?;
    let digest = sha256_hex(&temporary.path.join(RUN_FILE))?;
    write_new_synced(
        &temporary.path.join(RUN_DIGEST_FILE),
        format!("{digest}  {RUN_FILE}\n").as_bytes(),
    )?;
    sync_tree(&temporary.path)?;
    temporary.publish(root)
}

fn read_run_identity(root: &Path) -> Result<EvaluationRunIdentityV1, EvaluationArchiveError> {
    if !root.is_dir() {
        return Err(invalid("evaluation archive root is not a directory"));
    }
    let expected = format!("{}  {RUN_FILE}\n", sha256_hex(&root.join(RUN_FILE))?);
    if fs::read_to_string(root.join(RUN_DIGEST_FILE))? != expected {
        return Err(invalid("evaluation run identity checksum mismatch"));
    }
    let run: RunFileV1 = serde_json::from_slice(&fs::read(root.join(RUN_FILE))?)?;
    if run.format != RUN_FORMAT {
        return Err(invalid("unsupported evaluation run format"));
    }
    Ok(run.identity)
}

fn archived_batch(
    batch_index: u64,
    first_pair_id: u64,
    execution: &PromotionBatchExecution,
) -> Result<BatchFileV1, EvaluationArchiveError> {
    let summary = execution.summary()?;
    let games = execution
        .games
        .iter()
        .map(|played| match &played.result {
            Ok(result) => ArchivedGameV1 {
                game_id: played.scheduled.task.id,
                pair_id: played.scheduled.pair_id,
                leg: played.scheduled.leg,
                candidate_is_host: played.scheduled.candidate_is_host,
                termination: Some(archived_termination(result)),
                record: Some(result.record.to_string()),
                error: None,
                host_telemetry: Some(result.host_telemetry.into()),
                guest_telemetry: Some(result.guest_telemetry.into()),
            },
            Err(source) => ArchivedGameV1 {
                game_id: played.scheduled.task.id,
                pair_id: played.scheduled.pair_id,
                leg: played.scheduled.leg,
                candidate_is_host: played.scheduled.candidate_is_host,
                termination: None,
                record: None,
                error: Some(sanitize(source)),
                host_telemetry: None,
                guest_telemetry: None,
            },
        })
        .collect();
    Ok(BatchFileV1 {
        format: BATCH_FORMAT.to_owned(),
        batch_index,
        first_pair_id,
        attempted_pairs: summary.attempted_pairs as u64,
        eligible_pairs: summary.eligible_pairs as u64,
        excluded_pairs: summary.excluded_pairs as u64,
        pentanomial: summary.pentanomial.bins(),
        elapsed_seconds: execution.elapsed_seconds,
        observed_match_workers: execution.observed_match_workers,
        worker_capacity: execution.worker_capacity,
        candidate_inference_positions: candidate_inference_positions(execution)?,
        games,
    })
}

fn publish_batch_directory(root: &Path, batch: &BatchFileV1) -> Result<(), EvaluationArchiveError> {
    let batches = root.join(BATCHES_DIRECTORY);
    let final_directory = batches.join(batch_name(batch.batch_index));
    if final_directory.exists() {
        return Err(invalid("evaluation batch directory already exists"));
    }
    let temporary = temporary_directory(
        &batches,
        &format!(".{}.partial", batch_name(batch.batch_index)),
    )?;
    write_json_new(&temporary.path.join("batch.json"), batch)?;
    write_manifest(&temporary.path)?;
    verify_manifest(&temporary.path)?;
    sync_tree(&temporary.path)?;
    temporary.publish(&final_directory)
}

struct LoadedBatch {
    batch: BatchFileV1,
    rated_games: Vec<RatedGame>,
}

fn read_batch(
    path: &Path,
    identity: &EvaluationRunIdentityV1,
    expected_index: u64,
    expected_first_pair_id: u64,
) -> Result<LoadedBatch, EvaluationArchiveError> {
    let batch = read_batch_integrity(path, identity, expected_index, expected_first_pair_id)?;
    let pair_count = usize::try_from(batch.attempted_pairs)
        .map_err(|_| invalid("archived evaluation pair count does not fit usize"))?;
    let schedule = build_promotion_schedule_with_neutral_starts(
        identity.first_pair_id,
        expected_first_pair_id,
        pair_count,
        identity.neutral_start_configuration()?,
    )?;
    if batch.games.len() != schedule.len() {
        return Err(invalid(
            "evaluation games length differs from its regenerated schedule",
        ));
    }
    let mut outcomes = Vec::with_capacity(batch.games.len());
    let mut candidate_inference_positions = 0_u64;
    for (game, scheduled) in batch.games.iter().zip(&schedule) {
        if game.game_id != scheduled.task.id
            || game.pair_id != scheduled.pair_id
            || game.leg != scheduled.leg
            || game.candidate_is_host != scheduled.candidate_is_host
        {
            return Err(invalid(
                "archived evaluation game differs from its schedule",
            ));
        }
        let outcome = validate_archived_game(identity, game, scheduled)?;
        if let (Some(host), Some(guest)) = (game.host_telemetry, game.guest_telemetry) {
            let candidate = if game.candidate_is_host { host } else { guest };
            candidate_inference_positions = candidate_inference_positions
                .checked_add(candidate.decisions as u64)
                .ok_or_else(|| invalid("candidate inference counter overflow"))?;
        }
        outcomes.push(outcome);
    }
    let summary = summarize_archived_outcomes(identity, &outcomes, &schedule)?;
    if batch.eligible_pairs != summary.eligible_pairs
        || batch.excluded_pairs != summary.excluded_pairs
        || batch.pentanomial != summary.pentanomial.bins()
        || batch.candidate_inference_positions != candidate_inference_positions
    {
        return Err(invalid(
            "evaluation batch summary differs from replayed games",
        ));
    }
    Ok(LoadedBatch {
        batch,
        rated_games: summary.rated_games,
    })
}

fn read_batch_integrity(
    path: &Path,
    identity: &EvaluationRunIdentityV1,
    expected_index: u64,
    expected_first_pair_id: u64,
) -> Result<BatchFileV1, EvaluationArchiveError> {
    verify_manifest(path)?;
    let batch: BatchFileV1 = serde_json::from_slice(&fs::read(path.join("batch.json"))?)?;
    let eligible_from_bins = batch
        .pentanomial
        .iter()
        .try_fold(0_u64, |sum, count| sum.checked_add(*count));
    if batch.format != BATCH_FORMAT
        || batch.batch_index != expected_index
        || batch.first_pair_id != expected_first_pair_id
        || batch.attempted_pairs == 0
        || batch.games.len() as u64 != batch.attempted_pairs.saturating_mul(2)
        || batch.eligible_pairs.checked_add(batch.excluded_pairs) != Some(batch.attempted_pairs)
        || eligible_from_bins != Some(batch.eligible_pairs)
        || !batch.elapsed_seconds.is_finite()
        || batch.elapsed_seconds < 0.0
        || batch.worker_capacity != identity.workers
        || batch.observed_match_workers > batch.worker_capacity
    {
        return Err(invalid("evaluation batch metadata is inconsistent"));
    }
    Ok(batch)
}

#[derive(Clone, Copy)]
enum ArchivedOutcome {
    Rated(GameOutcome),
    DecisionLimit,
    MatchError,
}

fn validate_archived_game(
    identity: &EvaluationRunIdentityV1,
    game: &ArchivedGameV1,
    scheduled: &crate::PromotionScheduledGame,
) -> Result<ArchivedOutcome, EvaluationArchiveError> {
    match (
        game.termination,
        game.record.as_deref(),
        game.error.as_deref(),
        game.host_telemetry,
        game.guest_telemetry,
    ) {
        (None, None, Some(error), None, None) if !error.is_empty() => {
            Ok(ArchivedOutcome::MatchError)
        }
        (Some(termination), Some(record_text), None, Some(host), Some(guest)) => {
            let record: GameRecord = record_text
                .parse()
                .map_err(|source| invalid(format!("cannot parse evaluation record: {source}")))?;
            let starting_decisions = scheduled.starting_record.actions().len();
            if record.setup() != scheduled.task.setup
                || !record
                    .actions()
                    .starts_with(scheduled.starting_record.actions())
                || record.actions().len()
                    > starting_decisions
                        .saturating_add(identity.decision_soft_limit)
                        .saturating_add(1)
            {
                return Err(invalid(
                    "evaluation record prefix, setup or length is inconsistent",
                ));
            }
            let final_position = record
                .replay()
                .map_err(|source| invalid(format!("cannot replay evaluation record: {source}")))?;
            let (host_decisions, guest_decisions) = decision_counts(&record, starting_decisions)?;
            if host.decisions != host_decisions || guest.decisions != guest_decisions {
                return Err(invalid(
                    "evaluation telemetry differs from replayed decisions",
                ));
            }
            match termination {
                ArchivedTermination::HostWin
                    if final_position.outcome() == GameOutcome::Win(Player::Host) =>
                {
                    Ok(ArchivedOutcome::Rated(GameOutcome::Win(Player::Host)))
                }
                ArchivedTermination::GuestWin
                    if final_position.outcome() == GameOutcome::Win(Player::Guest) =>
                {
                    Ok(ArchivedOutcome::Rated(GameOutcome::Win(Player::Guest)))
                }
                ArchivedTermination::Draw if final_position.outcome() == GameOutcome::Draw => {
                    Ok(ArchivedOutcome::Rated(GameOutcome::Draw))
                }
                ArchivedTermination::DecisionLimit
                    if final_position.outcome() == GameOutcome::Ongoing
                        && final_position.phase() == TurnPhase::Main
                        && valid_decision_limit_length(
                            &record,
                            starting_decisions,
                            identity.decision_soft_limit,
                        ) =>
                {
                    Ok(ArchivedOutcome::DecisionLimit)
                }
                _ => Err(invalid(
                    "evaluation termination differs from its replayed record",
                )),
            }
        }
        _ => Err(invalid(
            "evaluation game fields form an invalid combination",
        )),
    }
}

struct ArchivedSummary {
    eligible_pairs: u64,
    excluded_pairs: u64,
    pentanomial: PentanomialCounts,
    rated_games: Vec<RatedGame>,
}

fn summarize_archived_outcomes(
    identity: &EvaluationRunIdentityV1,
    outcomes: &[ArchivedOutcome],
    schedule: &[crate::PromotionScheduledGame],
) -> Result<ArchivedSummary, EvaluationArchiveError> {
    let mut pentanomial = PentanomialCounts::default();
    let mut rated_games = Vec::new();
    let mut eligible_pairs = 0_u64;
    let mut excluded_pairs = 0_u64;
    let candidate = identity.candidate_agent_id();
    let opponent = identity.opponent_agent_id();
    for (pair_outcomes, pair_schedule) in outcomes.chunks_exact(2).zip(schedule.chunks_exact(2)) {
        match (pair_outcomes[0], pair_outcomes[1]) {
            (ArchivedOutcome::Rated(first), ArchivedOutcome::Rated(second)) => {
                let half_points =
                    candidate_half_points(first, true) + candidate_half_points(second, false);
                pentanomial.observe_pair_half_points(half_points)?;
                eligible_pairs += 1;
                for (outcome, scheduled) in [first, second].into_iter().zip(pair_schedule) {
                    let (host, guest) = if scheduled.candidate_is_host {
                        (candidate.clone(), opponent.clone())
                    } else {
                        (opponent.clone(), candidate.clone())
                    };
                    rated_games.push(RatedGame::new(
                        scheduled.task.id,
                        scheduled.pair_id,
                        host,
                        guest,
                        rated_outcome(outcome),
                    )?);
                }
            }
            _ => excluded_pairs += 1,
        }
    }
    Ok(ArchivedSummary {
        eligible_pairs,
        excluded_pairs,
        pentanomial,
        rated_games,
    })
}

fn build_analysis(
    identity: &EvaluationRunIdentityV1,
    progress: &EvaluationProgress,
    rated_games: &[RatedGame],
) -> Result<EvaluationAnalysisV1, EvaluationArchiveError> {
    let conclusion = progress
        .conclusion(identity)?
        .ok_or_else(|| invalid("evaluation has not reached a stopping condition"))?;
    if rated_games.len() as u64 != progress.eligible_pairs.saturating_mul(2) {
        return Err(invalid("rated game count differs from eligible pair count"));
    }
    let candidate = identity.candidate_agent_id();
    let opponent = identity.opponent_agent_id();
    let mut wins = 0_u64;
    let mut draws = 0_u64;
    let mut losses = 0_u64;
    for game in rated_games {
        let candidate_is_host = game.host() == &candidate;
        match (game.outcome(), candidate_is_host) {
            (RatedOutcome::Draw, _) => draws += 1,
            (RatedOutcome::HostWin, true) | (RatedOutcome::GuestWin, false) => wins += 1,
            _ => losses += 1,
        }
    }
    let sprt = evaluate_promotion_sprt(progress.pentanomial, identity.sprt_configuration()?)?;
    let lower_window = identity
        .lower_sprt_configuration()?
        .map(|configuration| {
            evaluate_promotion_sprt(progress.pentanomial, configuration).map(|report| {
                EvaluationLowerWindowV1 {
                    lower_elo: configuration.elo0(),
                    center_elo: configuration.elo1(),
                    log_likelihood_ratio: report.log_likelihood_ratio,
                    lower_bound: report.lower_bound,
                    upper_bound: report.upper_bound,
                }
            })
        })
        .transpose()?;
    let contextual_elo_point = match sprt.empirical_score {
        None => ContextualEloPointV1::NoRatedGames,
        Some(score) if score == 0.0 => ContextualEloPointV1::NegativeInfinity,
        Some(score) if score == 1.0 => ContextualEloPointV1::PositiveInfinity,
        Some(score) => ContextualEloPointV1::Finite {
            elo: contextual_elo_from_score(score)
                .expect("a strict probability has a finite logistic Elo"),
        },
    };
    let (mle, mle_error) = match fit_davidson_for_agents(
        &[candidate.clone(), opponent.clone()],
        rated_games,
        DavidsonOptions::default(),
    ) {
        Ok(fit) => {
            let gap = fit
                .elo_difference(&candidate, &opponent)
                .ok_or_else(|| invalid("Davidson fit omitted the requested Elo difference"))?;
            (
                Some(EvaluationMleV1 {
                    candidate_minus_opponent: gap.into(),
                    host_advantage_elo: fit.host_advantage_elo.map(Into::into),
                    draw_weight: fit.draw_weight(),
                    tie_model: format!("{:?}", fit.tie_model),
                    log_likelihood: fit.log_likelihood,
                }),
                None,
            )
        }
        Err(source) => (None, Some(source.to_string())),
    };
    Ok(EvaluationAnalysisV1 {
        conclusion,
        attempted_pairs: progress.attempted_pairs,
        eligible_pairs: progress.eligible_pairs,
        excluded_pairs: progress.excluded_pairs,
        pentanomial: progress.pentanomial.bins(),
        candidate_wins: wins,
        draws,
        candidate_losses: losses,
        empirical_score: sprt.empirical_score,
        contextual_elo_point,
        mle,
        mle_error,
        elo0: identity.elo0,
        elo1: identity.elo1,
        log_likelihood_ratio: sprt.log_likelihood_ratio,
        lower_bound: sprt.lower_bound,
        upper_bound: sprt.upper_bound,
        lower_window,
    })
}

impl From<ParameterEstimate> for EvaluationEstimateV1 {
    fn from(value: ParameterEstimate) -> Self {
        Self {
            estimate: value.estimate,
            model: EvaluationUncertaintyV1 {
                standard_error: value.model.standard_error,
                interval_95_lower: value.model.interval_95.lower,
                interval_95_upper: value.model.interval_95.upper,
            },
            paired_cluster: value
                .paired_cluster
                .map(|uncertainty| EvaluationUncertaintyV1 {
                    standard_error: uncertainty.standard_error,
                    interval_95_lower: uncertainty.interval_95.lower,
                    interval_95_upper: uncertainty.interval_95.upper,
                }),
        }
    }
}

fn absorb_batch(
    progress: &mut EvaluationProgress,
    batch: &BatchFileV1,
) -> Result<(), EvaluationArchiveError> {
    progress.batches = progress
        .batches
        .checked_add(1)
        .ok_or_else(|| invalid("evaluation batch counter overflow"))?;
    progress.attempted_pairs = progress
        .attempted_pairs
        .checked_add(batch.attempted_pairs)
        .ok_or_else(|| invalid("attempted pair counter overflow"))?;
    progress.eligible_pairs = progress
        .eligible_pairs
        .checked_add(batch.eligible_pairs)
        .ok_or_else(|| invalid("eligible pair counter overflow"))?;
    progress.excluded_pairs = progress
        .excluded_pairs
        .checked_add(batch.excluded_pairs)
        .ok_or_else(|| invalid("excluded pair counter overflow"))?;
    let current = progress.pentanomial.bins();
    let mut combined = [0_u64; 5];
    for index in 0..5 {
        combined[index] = current[index]
            .checked_add(batch.pentanomial[index])
            .ok_or_else(|| invalid("pentanomial counter overflow"))?;
    }
    progress.pentanomial = PentanomialCounts::new(combined);
    progress.candidate_inference_positions = progress
        .candidate_inference_positions
        .checked_add(batch.candidate_inference_positions)
        .ok_or_else(|| invalid("candidate inference counter overflow"))?;
    progress.maximum_workers_observed = progress
        .maximum_workers_observed
        .max(batch.observed_match_workers);
    Ok(())
}

fn validate_execution_schedule(
    execution: &PromotionBatchExecution,
    identity: &EvaluationRunIdentityV1,
    first_pair_id: u64,
) -> Result<(), EvaluationArchiveError> {
    if execution.games.is_empty() || execution.games.len() % 2 != 0 {
        return Err(invalid(
            "evaluation execution is not made of complete pairs",
        ));
    }
    let expected = build_promotion_schedule_with_neutral_starts(
        identity.first_pair_id,
        first_pair_id,
        execution.games.len() / 2,
        identity.neutral_start_configuration()?,
    )?;
    if execution
        .games
        .iter()
        .map(|played| &played.scheduled)
        .ne(expected.iter())
        || execution.worker_capacity != identity.workers
    {
        return Err(invalid(
            "evaluation execution differs from its immutable schedule",
        ));
    }
    Ok(())
}

fn candidate_inference_positions(
    execution: &PromotionBatchExecution,
) -> Result<u64, EvaluationArchiveError> {
    execution.games.iter().try_fold(0_u64, |total, played| {
        let Some(result) = played.result.as_ref().ok() else {
            return Ok(total);
        };
        let decisions = if played.scheduled.candidate_is_host {
            result.host_telemetry.decisions
        } else {
            result.guest_telemetry.decisions
        };
        total
            .checked_add(decisions as u64)
            .ok_or_else(|| invalid("candidate inference counter overflow"))
    })
}

fn read_result(root: &Path) -> Result<Option<ResultFileV1>, EvaluationArchiveError> {
    let directory = root.join(RESULT_DIRECTORY);
    if !directory.exists() {
        return Ok(None);
    }
    verify_manifest(&directory)?;
    let result: ResultFileV1 = serde_json::from_slice(&fs::read(directory.join("result.json"))?)?;
    if result.format != RESULT_FORMAT {
        return Err(invalid("unsupported evaluation result format"));
    }
    Ok(Some(result))
}

fn result_bytes_match(
    root: &Path,
    analysis: &EvaluationAnalysisV1,
) -> Result<bool, EvaluationArchiveError> {
    let mut expected = serde_json::to_vec_pretty(&ResultFileV1 {
        format: RESULT_FORMAT.to_owned(),
        analysis: analysis.clone(),
    })?;
    expected.push(b'\n');
    Ok(fs::read(root.join(RESULT_DIRECTORY).join("result.json"))? == expected)
}

fn decision_counts(
    record: &GameRecord,
    starting_decisions: usize,
) -> Result<(usize, usize), EvaluationArchiveError> {
    let mut position = record.initial_position();
    let mut counts = [0_usize; 2];
    for (index, action) in record.actions().iter().enumerate() {
        let player = position.to_move();
        position
            .apply(*action)
            .map_err(|source| invalid(format!("record decision cannot apply: {source}")))?;
        if index >= starting_decisions {
            counts[player.index()] += 1;
        }
    }
    Ok((counts[Player::Host.index()], counts[Player::Guest.index()]))
}

fn valid_decision_limit_length(
    record: &GameRecord,
    starting_decisions: usize,
    decision_soft_limit: usize,
) -> bool {
    let new_decisions = record.actions().len().saturating_sub(starting_decisions);
    if new_decisions == decision_soft_limit {
        return true;
    }
    if new_decisions != decision_soft_limit.saturating_add(1) {
        return false;
    }
    let admitted = starting_decisions.saturating_add(decision_soft_limit);
    let mut position = record.initial_position();
    for action in record.actions().iter().take(admitted) {
        if position.apply(*action).is_err() {
            return false;
        }
    }
    position.outcome() == GameOutcome::Ongoing && position.phase() == TurnPhase::HarmonyBonus
}

fn archived_termination(result: &MatchResult) -> ArchivedTermination {
    match result.termination {
        MatchTermination::Rules(GameOutcome::Win(Player::Host)) => ArchivedTermination::HostWin,
        MatchTermination::Rules(GameOutcome::Win(Player::Guest)) => ArchivedTermination::GuestWin,
        MatchTermination::Rules(GameOutcome::Draw) => ArchivedTermination::Draw,
        MatchTermination::DecisionLimit => ArchivedTermination::DecisionLimit,
        MatchTermination::Rules(GameOutcome::Ongoing) => {
            unreachable!("match validation rejects ongoing rule outcomes")
        }
    }
}

const fn candidate_half_points(outcome: GameOutcome, candidate_is_host: bool) -> u8 {
    match outcome {
        GameOutcome::Draw => 1,
        GameOutcome::Win(Player::Host) if candidate_is_host => 2,
        GameOutcome::Win(Player::Guest) if !candidate_is_host => 2,
        GameOutcome::Win(_) => 0,
        GameOutcome::Ongoing => unreachable!(),
    }
}

const fn rated_outcome(outcome: GameOutcome) -> RatedOutcome {
    match outcome {
        GameOutcome::Win(Player::Host) => RatedOutcome::HostWin,
        GameOutcome::Win(Player::Guest) => RatedOutcome::GuestWin,
        GameOutcome::Draw => RatedOutcome::Draw,
        GameOutcome::Ongoing => unreachable!(),
    }
}

fn sanitize(value: &impl ToString) -> String {
    value.to_string().replace(['\t', '\n', '\r'], " ")
}

fn batch_name(index: u64) -> String {
    format!("batch-{index:020}")
}

fn parse_batch_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("batch-")?;
    (digits.len() == 20 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

fn invalid(message: impl Into<String>) -> EvaluationArchiveError {
    EvaluationArchiveError::InvalidData(message.into())
}

#[derive(Debug)]
pub enum EvaluationArchiveError {
    Io(io::Error),
    Json(serde_json::Error),
    Identity(EvaluationIdentityError),
    Match(PromotionMatchError),
    Schedule(PromotionScheduleError),
    Sprt(paisho_rating::PromotionSprtError),
    RatedGame(paisho_rating::RatedGameError),
    InvalidData(String),
}

impl fmt::Display for EvaluationArchiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => source.fmt(formatter),
            Self::Json(source) => source.fmt(formatter),
            Self::Identity(source) => source.fmt(formatter),
            Self::Match(source) => source.fmt(formatter),
            Self::Schedule(source) => source.fmt(formatter),
            Self::Sprt(source) => source.fmt(formatter),
            Self::RatedGame(source) => source.fmt(formatter),
            Self::InvalidData(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EvaluationArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::Identity(source) => Some(source),
            Self::Match(source) => Some(source),
            Self::Schedule(source) => Some(source),
            Self::Sprt(source) => Some(source),
            Self::RatedGame(source) => Some(source),
            Self::InvalidData(_) => None,
        }
    }
}

impl From<io::Error> for EvaluationArchiveError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for EvaluationArchiveError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

impl From<EvaluationIdentityError> for EvaluationArchiveError {
    fn from(source: EvaluationIdentityError) -> Self {
        Self::Identity(source)
    }
}

impl From<PromotionMatchError> for EvaluationArchiveError {
    fn from(source: PromotionMatchError) -> Self {
        Self::Match(source)
    }
}

impl From<PromotionScheduleError> for EvaluationArchiveError {
    fn from(source: PromotionScheduleError) -> Self {
        Self::Schedule(source)
    }
}

impl From<paisho_rating::PromotionSprtError> for EvaluationArchiveError {
    fn from(source: paisho_rating::PromotionSprtError) -> Self {
        Self::Sprt(source)
    }
}

impl From<paisho_rating::RatedGameError> for EvaluationArchiveError {
    fn from(source: paisho_rating::RatedGameError) -> Self {
        Self::RatedGame(source)
    }
}

#[cfg(test)]
#[path = "evaluation_archive/tests.rs"]
mod tests;
