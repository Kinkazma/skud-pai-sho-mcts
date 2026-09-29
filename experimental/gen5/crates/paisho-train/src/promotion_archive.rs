use core::fmt;
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use paisho_ai::{AgentTelemetry, MatchResult, MatchTermination};
use paisho_core::{GameOutcome, GameRecord, Player, TurnPhase};
use paisho_rating::{evaluate_promotion_sprt, PentanomialCounts, PromotionSprtError};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    build_promotion_schedule_with_neutral_starts, PairExclusionReason, PromotionBatchExecution,
    PromotionBatchSummary, PromotionMatchError,
};

mod identity;

pub use identity::{
    PromotionCampaignConclusion, PromotionCampaignProgress, PromotionInferenceClassV1,
    PromotionNeutralStartV1, PromotionRunIdentityV1, PromotionSamplingPolicyV1,
};

const RUN_FORMAT: &str = "PAISHO-PROMOTION-RUN-1";
const BATCH_FORMAT: &str = "PAISHO-PROMOTION-BATCH-1";
const DECISION_FORMAT: &str = "PAISHO-PROMOTION-DECISION-1";
const RUN_FILE: &str = "run.json";
const RUN_DIGEST_FILE: &str = "run.sha256";
const BATCHES_DIRECTORY: &str = "batches";
const DECISION_DIRECTORY: &str = "decision";
const MANIFEST_FILE: &str = "MANIFEST.sha256";

pub struct PromotionCampaignArchive {
    root: PathBuf,
    identity: PromotionRunIdentityV1,
}

impl PromotionCampaignArchive {
    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self, PromotionArchiveError> {
        let root = root.as_ref().to_owned();
        let identity = read_run_identity(&root)?;
        identity.validate()?;
        let archive = Self { root, identity };
        archive.load_progress()?;
        Ok(archive)
    }

    pub(crate) fn open_existing_integrity(
        root: impl AsRef<Path>,
    ) -> Result<(Self, PromotionCampaignProgress), PromotionArchiveError> {
        let root = root.as_ref().to_owned();
        let identity = read_run_identity(&root)?;
        identity.validate()?;
        let archive = Self { root, identity };
        let progress = archive.load_progress_integrity()?;
        Ok((archive, progress))
    }

    pub fn open_or_create(
        root: impl AsRef<Path>,
        identity: PromotionRunIdentityV1,
    ) -> Result<Self, PromotionArchiveError> {
        identity.validate()?;
        let root = root.as_ref().to_owned();
        if root.exists() {
            let stored = read_run_identity(&root)?;
            if stored != identity {
                return Err(invalid(
                    "existing promotion archive has a different immutable run identity",
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

    pub const fn identity(&self) -> &PromotionRunIdentityV1 {
        &self.identity
    }

    pub fn load_progress(&self) -> Result<PromotionCampaignProgress, PromotionArchiveError> {
        self.load_progress_with_record_replay(true)
    }

    pub(crate) fn load_progress_integrity(
        &self,
    ) -> Result<PromotionCampaignProgress, PromotionArchiveError> {
        self.load_progress_with_record_replay(false)
    }

    pub(crate) fn require_published_conclusion(
        &self,
        progress: &PromotionCampaignProgress,
        conclusion: PromotionCampaignConclusion,
    ) -> Result<(), PromotionArchiveError> {
        if progress.conclusion(&self.identity)? != Some(conclusion) {
            return Err(invalid(
                "promotion conclusion disagrees with the campaign progress",
            ));
        }
        let stored = read_decision(&self.root)?
            .ok_or_else(|| invalid("promotion stopping condition has no published decision"))?;
        if stored != decision_record(&self.identity, progress, conclusion)? {
            return Err(invalid(
                "stored promotion decision does not match its batches",
            ));
        }
        Ok(())
    }

    fn load_progress_with_record_replay(
        &self,
        replay_records: bool,
    ) -> Result<PromotionCampaignProgress, PromotionArchiveError> {
        let stored = read_run_identity(&self.root)?;
        if stored != self.identity {
            return Err(invalid("promotion run identity changed after opening"));
        }
        let batches_directory = self.root.join(BATCHES_DIRECTORY);
        let mut batch_paths = Vec::new();
        for entry in fs::read_dir(batches_directory)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Some(index) = parse_batch_name(&name) else {
                return Err(invalid(format!("unexpected promotion batch entry {name}")));
            };
            if !entry.file_type()?.is_dir() {
                return Err(invalid(format!(
                    "promotion batch {name} is not a directory"
                )));
            }
            batch_paths.push((index, entry.path()));
        }
        batch_paths.sort_by_key(|(index, _)| *index);
        let mut progress = PromotionCampaignProgress::default();
        for (expected_index, (actual_index, path)) in batch_paths.into_iter().enumerate() {
            let expected_index = u64::try_from(expected_index)
                .map_err(|_| invalid("promotion batch index does not fit u64"))?;
            if actual_index != expected_index {
                return Err(invalid("promotion batch indices are not contiguous"));
            }
            let expected_first_pair = progress.next_pair_id(&self.identity)?;
            let batch = if replay_records {
                read_batch(&path, &self.identity, actual_index, expected_first_pair)?
            } else {
                read_batch_integrity(&path, &self.identity, actual_index, expected_first_pair)?
            };
            absorb_batch(&mut progress, &batch)?;
        }
        let expected_conclusion = progress.conclusion(&self.identity)?;
        let stored_conclusion = read_decision(&self.root)?;
        match (expected_conclusion, stored_conclusion) {
            (Some(expected), Some(stored)) if expected == stored.conclusion => {
                let expected_record = decision_record(&self.identity, &progress, expected)?;
                if stored != expected_record {
                    return Err(invalid(
                        "stored promotion decision does not match its batches",
                    ));
                }
            }
            (None, None) | (Some(_), None) => {}
            (None, Some(_)) => {
                return Err(invalid(
                    "promotion decision exists before a stopping condition",
                ))
            }
            (Some(_), Some(_)) => {
                return Err(invalid(
                    "stored promotion conclusion differs from recomputation",
                ))
            }
        }
        Ok(progress)
    }

    pub fn publish_next_batch(
        &self,
        progress: &PromotionCampaignProgress,
        execution: &PromotionBatchExecution,
    ) -> Result<PromotionCampaignProgress, PromotionArchiveError> {
        // Opening or resuming a campaign has already replayed every published
        // record semantically. Between batches, verify immutable manifests and
        // counters without regenerating every historical neutral prefix.
        let current = self.load_progress_integrity()?;
        if &current != progress {
            return Err(invalid(
                "promotion progress changed before batch publication",
            ));
        }
        if current.conclusion(&self.identity)?.is_some() {
            return Err(invalid(
                "promotion campaign already reached a stopping condition",
            ));
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
                "promotion batch exceeds the configured campaign budget",
            ));
        }
        let first_pair_id = current.next_pair_id(&self.identity)?;
        validate_execution_schedule(
            execution,
            &self.identity,
            first_pair_id,
            summary.attempted_pairs,
        )?;
        let batch = archived_batch(current.batches, first_pair_id, execution, summary)?;
        publish_batch_directory(&self.root, &batch, execution)?;
        let mut next = current;
        absorb_batch(&mut next, &batch)?;
        Ok(next)
    }

    pub fn publish_conclusion(
        &self,
        progress: &PromotionCampaignProgress,
    ) -> Result<PromotionCampaignConclusion, PromotionArchiveError> {
        let current = self.load_progress()?;
        if &current != progress {
            return Err(invalid(
                "promotion progress changed before decision publication",
            ));
        }
        let conclusion = current
            .conclusion(&self.identity)?
            .ok_or_else(|| invalid("promotion campaign has not reached a stopping condition"))?;
        let record = decision_record(&self.identity, &current, conclusion)?;
        let final_directory = self.root.join(DECISION_DIRECTORY);
        if final_directory.exists() {
            let stored = read_decision(&self.root)?
                .ok_or_else(|| invalid("promotion decision directory is unreadable"))?;
            if stored == record {
                return Ok(conclusion);
            }
            return Err(invalid(
                "promotion decision is already published with other contents",
            ));
        }
        let temporary = temporary_directory(&self.root, ".decision.partial")?;
        write_json_new(&temporary.path.join("decision.json"), &record)?;
        write_manifest(&temporary.path)?;
        verify_manifest(&temporary.path)?;
        sync_tree(&temporary.path)?;
        temporary.publish(&final_directory)?;
        let stored = read_decision(&self.root)?
            .ok_or_else(|| invalid("published promotion decision disappeared"))?;
        if stored != record {
            return Err(invalid("published promotion decision failed verification"));
        }
        Ok(conclusion)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RunFileV1 {
    format: String,
    identity: PromotionRunIdentityV1,
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
    champion_inference_positions: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct GamesFileV1 {
    format: String,
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
struct DecisionFileV1 {
    format: String,
    conclusion: PromotionCampaignConclusion,
    attempted_pairs: u64,
    eligible_pairs: u64,
    excluded_pairs: u64,
    pentanomial: [u64; 5],
    empirical_score: Option<f64>,
    log_likelihood_ratio: f64,
    lower_bound: f64,
    upper_bound: f64,
}

fn initialize_archive(
    root: &Path,
    identity: &PromotionRunIdentityV1,
) -> Result<(), PromotionArchiveError> {
    let parent = root.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let temporary = temporary_directory(parent, ".promotion-run.initializing")?;
    fs::create_dir(temporary.path.join(BATCHES_DIRECTORY))?;
    let run = RunFileV1 {
        format: RUN_FORMAT.to_owned(),
        identity: identity.clone(),
    };
    write_json_new(&temporary.path.join(RUN_FILE), &run)?;
    let digest = sha256_hex(&temporary.path.join(RUN_FILE))?;
    write_new_synced(
        &temporary.path.join(RUN_DIGEST_FILE),
        format!("{digest}  {RUN_FILE}\n").as_bytes(),
    )?;
    sync_tree(&temporary.path)?;
    temporary.publish(root)?;
    Ok(())
}

fn read_run_identity(root: &Path) -> Result<PromotionRunIdentityV1, PromotionArchiveError> {
    if !root.is_dir() || !root.join(BATCHES_DIRECTORY).is_dir() {
        return Err(invalid("promotion archive structure is incomplete"));
    }
    let expected_digest = fs::read_to_string(root.join(RUN_DIGEST_FILE))?;
    let expected_digest = expected_digest
        .strip_suffix(&format!("  {RUN_FILE}\n"))
        .ok_or_else(|| invalid("malformed promotion run checksum"))?;
    if !is_lower_hex_digest(expected_digest) || sha256_hex(&root.join(RUN_FILE))? != expected_digest
    {
        return Err(invalid("promotion run identity checksum mismatch"));
    }
    let run: RunFileV1 = serde_json::from_slice(&fs::read(root.join(RUN_FILE))?)?;
    if run.format != RUN_FORMAT {
        return Err(invalid("unsupported promotion run format"));
    }
    run.identity.validate()?;
    Ok(run.identity)
}

fn archived_batch(
    batch_index: u64,
    first_pair_id: u64,
    execution: &PromotionBatchExecution,
    summary: PromotionBatchSummary,
) -> Result<BatchFileV1, PromotionArchiveError> {
    let (candidate_inference_positions, champion_inference_positions) =
        inference_positions(&execution.games)?;
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
        candidate_inference_positions,
        champion_inference_positions,
    })
}

fn publish_batch_directory(
    root: &Path,
    batch: &BatchFileV1,
    execution: &PromotionBatchExecution,
) -> Result<(), PromotionArchiveError> {
    let batches = root.join(BATCHES_DIRECTORY);
    let final_directory = batches.join(batch_name(batch.batch_index));
    if final_directory.exists() {
        return Err(invalid("promotion batch directory already exists"));
    }
    let temporary = temporary_directory(
        &batches,
        &format!(".{}.partial", batch_name(batch.batch_index)),
    )?;
    fs::create_dir(temporary.path.join("records"))?;
    let games = archive_games(&temporary.path, execution)?;
    write_json_new(&temporary.path.join("batch.json"), batch)?;
    write_json_new(
        &temporary.path.join("games.json"),
        &GamesFileV1 {
            format: BATCH_FORMAT.to_owned(),
            games,
        },
    )?;
    write_manifest(&temporary.path)?;
    verify_manifest(&temporary.path)?;
    sync_tree(&temporary.path)?;
    temporary.publish(&final_directory)?;
    Ok(())
}

fn archive_games(
    directory: &Path,
    execution: &PromotionBatchExecution,
) -> Result<Vec<ArchivedGameV1>, PromotionArchiveError> {
    execution
        .games
        .iter()
        .map(|played| {
            let scheduled = &played.scheduled;
            match &played.result {
                Ok(result) => {
                    let record_name = format!("records/game-{:020}.psr", scheduled.task.id);
                    write_new_synced(
                        &directory.join(&record_name),
                        result.record.to_string().as_bytes(),
                    )?;
                    Ok(ArchivedGameV1 {
                        game_id: scheduled.task.id,
                        pair_id: scheduled.pair_id,
                        leg: scheduled.leg,
                        candidate_is_host: scheduled.candidate_is_host,
                        termination: Some(archived_termination(result)),
                        record: Some(record_name),
                        error: None,
                        host_telemetry: Some(result.host_telemetry.into()),
                        guest_telemetry: Some(result.guest_telemetry.into()),
                    })
                }
                Err(source) => Ok(ArchivedGameV1 {
                    game_id: scheduled.task.id,
                    pair_id: scheduled.pair_id,
                    leg: scheduled.leg,
                    candidate_is_host: scheduled.candidate_is_host,
                    termination: None,
                    record: None,
                    error: Some(sanitize(source)),
                    host_telemetry: None,
                    guest_telemetry: None,
                }),
            }
        })
        .collect()
}

fn read_batch(
    path: &Path,
    identity: &PromotionRunIdentityV1,
    expected_index: u64,
    expected_first_pair: u64,
) -> Result<BatchFileV1, PromotionArchiveError> {
    let batch = read_batch_integrity(path, identity, expected_index, expected_first_pair)?;
    let games: GamesFileV1 = serde_json::from_slice(&fs::read(path.join("games.json"))?)?;
    if games.format != BATCH_FORMAT {
        return Err(invalid("promotion batch metadata is inconsistent"));
    }
    let pair_count = usize::try_from(batch.attempted_pairs)
        .map_err(|_| invalid("archived promotion pair count does not fit usize"))?;
    let expected = build_identity_schedule(identity, expected_first_pair, pair_count)?;
    if games.games.len() != expected.len() {
        return Err(invalid(
            "promotion games length differs from its batch schedule",
        ));
    }
    let mut outcomes = Vec::with_capacity(games.games.len());
    let mut referenced_records = BTreeSet::new();
    let mut candidate_inference_positions = 0_u64;
    let mut champion_inference_positions = 0_u64;
    for (game, scheduled) in games.games.iter().zip(&expected) {
        if game.game_id != scheduled.task.id
            || game.pair_id != scheduled.pair_id
            || game.leg != scheduled.leg
            || game.candidate_is_host != scheduled.candidate_is_host
        {
            return Err(invalid("archived promotion game differs from its schedule"));
        }
        outcomes.push(validate_archived_game(
            path,
            identity,
            game,
            scheduled,
            &mut referenced_records,
        )?);
        if let (Some(host), Some(guest)) = (game.host_telemetry, game.guest_telemetry) {
            let (candidate, champion) = if game.candidate_is_host {
                (host.decisions, guest.decisions)
            } else {
                (guest.decisions, host.decisions)
            };
            candidate_inference_positions = candidate_inference_positions
                .checked_add(candidate as u64)
                .ok_or_else(|| invalid("candidate inference counter overflow"))?;
            champion_inference_positions = champion_inference_positions
                .checked_add(champion as u64)
                .ok_or_else(|| invalid("champion inference counter overflow"))?;
        }
    }
    let actual_records = regular_files(&path.join("records"))?;
    if actual_records != referenced_records {
        return Err(invalid("promotion record files differ from games.json"));
    }
    let summary = summarize_archived_outcomes(&outcomes, &expected)?;
    if batch.eligible_pairs != summary.eligible_pairs as u64
        || batch.excluded_pairs != summary.excluded_pairs as u64
        || batch.pentanomial != summary.pentanomial.bins()
        || batch.candidate_inference_positions != candidate_inference_positions
        || batch.champion_inference_positions != champion_inference_positions
    {
        return Err(invalid(
            "promotion batch summary differs from replayed games",
        ));
    }
    Ok(batch)
}

fn read_batch_integrity(
    path: &Path,
    identity: &PromotionRunIdentityV1,
    expected_index: u64,
    expected_first_pair: u64,
) -> Result<BatchFileV1, PromotionArchiveError> {
    verify_manifest(path)?;
    let batch: BatchFileV1 = serde_json::from_slice(&fs::read(path.join("batch.json"))?)?;
    let eligible_from_bins = batch
        .pentanomial
        .iter()
        .try_fold(0_u64, |sum, count| sum.checked_add(*count));
    if batch.format != BATCH_FORMAT
        || batch.batch_index != expected_index
        || batch.first_pair_id != expected_first_pair
        || batch.attempted_pairs == 0
        || batch.eligible_pairs.checked_add(batch.excluded_pairs) != Some(batch.attempted_pairs)
        || eligible_from_bins != Some(batch.eligible_pairs)
        || !batch.elapsed_seconds.is_finite()
        || batch.elapsed_seconds < 0.0
        || batch.worker_capacity != identity.workers
        || batch.observed_match_workers > batch.worker_capacity
    {
        return Err(invalid("promotion batch metadata is inconsistent"));
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
    batch_directory: &Path,
    identity: &PromotionRunIdentityV1,
    game: &ArchivedGameV1,
    scheduled: &crate::PromotionScheduledGame,
    referenced_records: &mut BTreeSet<PathBuf>,
) -> Result<ArchivedOutcome, PromotionArchiveError> {
    match (
        game.termination,
        game.record.as_deref(),
        game.error.as_deref(),
        game.host_telemetry,
        game.guest_telemetry,
    ) {
        (None, None, Some(error), None, None) if !error.is_empty() => {
            return Ok(ArchivedOutcome::MatchError)
        }
        (Some(termination), Some(record_name), None, Some(host), Some(guest)) => {
            let expected_name = format!("records/game-{:020}.psr", game.game_id);
            if record_name != expected_name || !safe_relative_path(record_name) {
                return Err(invalid("promotion record path is not canonical"));
            }
            let relative = PathBuf::from(record_name);
            if !referenced_records.insert(PathBuf::from(
                relative
                    .strip_prefix("records")
                    .expect("canonical record path starts with records"),
            )) {
                return Err(invalid("promotion record is referenced more than once"));
            }
            let text = fs::read_to_string(batch_directory.join(relative))?;
            let record: GameRecord = text
                .parse()
                .map_err(|source| invalid(format!("cannot parse promotion record: {source}")))?;
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
                    "promotion record prefix, setup or length is inconsistent",
                ));
            }
            let final_position = record
                .replay()
                .map_err(|source| invalid(format!("cannot replay promotion record: {source}")))?;
            let (host_decisions, guest_decisions) = decision_counts(&record, starting_decisions)?;
            if host.decisions != host_decisions || guest.decisions != guest_decisions {
                return Err(invalid(
                    "promotion telemetry differs from replayed decisions",
                ));
            }
            return match termination {
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
                        && (record.actions().len()
                            == starting_decisions.saturating_add(identity.decision_soft_limit)
                            || valid_bonus_grace(
                                &record,
                                starting_decisions,
                                identity.decision_soft_limit,
                            )) =>
                {
                    Ok(ArchivedOutcome::DecisionLimit)
                }
                _ => Err(invalid(
                    "promotion termination differs from replayed position",
                )),
            };
        }
        _ => {}
    }
    Err(invalid("promotion game mixes success and error fields"))
}

fn summarize_archived_outcomes(
    outcomes: &[ArchivedOutcome],
    schedule: &[crate::PromotionScheduledGame],
) -> Result<PromotionBatchSummary, PromotionArchiveError> {
    let mut counts = PentanomialCounts::default();
    let mut eligible = 0_usize;
    for (outcomes, scheduled) in outcomes.chunks_exact(2).zip(schedule.chunks_exact(2)) {
        let exclusion = if outcomes
            .iter()
            .any(|outcome| matches!(outcome, ArchivedOutcome::MatchError))
        {
            Some(PairExclusionReason::MatchError)
        } else if outcomes
            .iter()
            .any(|outcome| matches!(outcome, ArchivedOutcome::DecisionLimit))
        {
            Some(PairExclusionReason::DecisionLimit)
        } else {
            None
        };
        if exclusion.is_some() {
            continue;
        }
        let [ArchivedOutcome::Rated(first), ArchivedOutcome::Rated(second)] = outcomes else {
            return Err(invalid("eligible promotion pair has a non-terminal game"));
        };
        let half_points = candidate_half_points(*first, scheduled[0].candidate_is_host)
            + candidate_half_points(*second, scheduled[1].candidate_is_host);
        counts.observe_pair_half_points(half_points)?;
        eligible += 1;
    }
    Ok(PromotionBatchSummary {
        attempted_pairs: outcomes.len() / 2,
        eligible_pairs: eligible,
        excluded_pairs: outcomes.len() / 2 - eligible,
        pentanomial: counts,
    })
}

fn validate_execution_schedule(
    execution: &PromotionBatchExecution,
    identity: &PromotionRunIdentityV1,
    first_pair_id: u64,
    pair_count: usize,
) -> Result<(), PromotionArchiveError> {
    let expected = build_identity_schedule(identity, first_pair_id, pair_count)?;
    if execution.games.len() != expected.len()
        || execution
            .games
            .iter()
            .zip(expected)
            .any(|(played, scheduled)| played.scheduled != scheduled)
        || execution.worker_capacity != identity.workers
        || execution.observed_match_workers > execution.worker_capacity
        || !execution.elapsed_seconds.is_finite()
        || execution.elapsed_seconds < 0.0
    {
        return Err(invalid(
            "promotion execution differs from its configured schedule",
        ));
    }
    Ok(())
}

fn inference_positions(
    games: &[crate::PlayedPromotionGame],
) -> Result<(u64, u64), PromotionArchiveError> {
    let mut candidate = 0_u64;
    let mut champion = 0_u64;
    for game in games {
        if let Ok(result) = &game.result {
            let (candidate_decisions, champion_decisions) = if game.scheduled.candidate_is_host {
                (
                    result.host_telemetry.decisions,
                    result.guest_telemetry.decisions,
                )
            } else {
                (
                    result.guest_telemetry.decisions,
                    result.host_telemetry.decisions,
                )
            };
            candidate = candidate
                .checked_add(candidate_decisions as u64)
                .ok_or_else(|| invalid("candidate inference counter overflow"))?;
            champion = champion
                .checked_add(champion_decisions as u64)
                .ok_or_else(|| invalid("champion inference counter overflow"))?;
        }
    }
    Ok((candidate, champion))
}

fn build_identity_schedule(
    identity: &PromotionRunIdentityV1,
    first_pair_id: u64,
    pair_count: usize,
) -> Result<Vec<crate::PromotionScheduledGame>, PromotionArchiveError> {
    Ok(build_promotion_schedule_with_neutral_starts(
        identity.first_pair_id,
        first_pair_id,
        pair_count,
        identity.neutral_start_configuration()?,
    )?)
}

fn absorb_batch(
    progress: &mut PromotionCampaignProgress,
    batch: &BatchFileV1,
) -> Result<(), PromotionArchiveError> {
    progress.batches = progress
        .batches
        .checked_add(1)
        .ok_or_else(|| invalid("promotion batch counter overflow"))?;
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
    progress.champion_inference_positions = progress
        .champion_inference_positions
        .checked_add(batch.champion_inference_positions)
        .ok_or_else(|| invalid("champion inference counter overflow"))?;
    progress.maximum_workers_observed = progress
        .maximum_workers_observed
        .max(batch.observed_match_workers);
    Ok(())
}

fn decision_record(
    identity: &PromotionRunIdentityV1,
    progress: &PromotionCampaignProgress,
    conclusion: PromotionCampaignConclusion,
) -> Result<DecisionFileV1, PromotionArchiveError> {
    let report = evaluate_promotion_sprt(progress.pentanomial, identity.sprt_configuration()?)?;
    Ok(DecisionFileV1 {
        format: DECISION_FORMAT.to_owned(),
        conclusion,
        attempted_pairs: progress.attempted_pairs,
        eligible_pairs: progress.eligible_pairs,
        excluded_pairs: progress.excluded_pairs,
        pentanomial: progress.pentanomial.bins(),
        empirical_score: report.empirical_score,
        log_likelihood_ratio: report.log_likelihood_ratio,
        lower_bound: report.lower_bound,
        upper_bound: report.upper_bound,
    })
}

fn read_decision(root: &Path) -> Result<Option<DecisionFileV1>, PromotionArchiveError> {
    let directory = root.join(DECISION_DIRECTORY);
    if !directory.exists() {
        return Ok(None);
    }
    verify_manifest(&directory)?;
    let decision: DecisionFileV1 =
        serde_json::from_slice(&fs::read(directory.join("decision.json"))?)?;
    if decision.format != DECISION_FORMAT {
        return Err(invalid("unsupported promotion decision format"));
    }
    Ok(Some(decision))
}

fn archived_termination(result: &MatchResult) -> ArchivedTermination {
    match result.termination {
        MatchTermination::Rules(GameOutcome::Win(Player::Host)) => ArchivedTermination::HostWin,
        MatchTermination::Rules(GameOutcome::Win(Player::Guest)) => ArchivedTermination::GuestWin,
        MatchTermination::Rules(GameOutcome::Draw) => ArchivedTermination::Draw,
        MatchTermination::DecisionLimit => ArchivedTermination::DecisionLimit,
        MatchTermination::Rules(GameOutcome::Ongoing) => {
            unreachable!("promotion match validation rejects ongoing rule outcomes")
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

fn decision_counts(
    record: &GameRecord,
    starting_decisions: usize,
) -> Result<(usize, usize), PromotionArchiveError> {
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

fn valid_bonus_grace(
    record: &GameRecord,
    starting_decisions: usize,
    decision_soft_limit: usize,
) -> bool {
    let admitted_decisions = starting_decisions.saturating_add(decision_soft_limit);
    if record.actions().len() != admitted_decisions.saturating_add(1) {
        return false;
    }
    let mut position = record.initial_position();
    for action in record.actions().iter().take(admitted_decisions) {
        if position.apply(*action).is_err() {
            return false;
        }
    }
    position.outcome() == GameOutcome::Ongoing && position.phase() == TurnPhase::HarmonyBonus
}

fn write_json_new<T: Serialize>(path: &Path, value: &T) -> Result<(), PromotionArchiveError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new_synced(path, &bytes)
}

fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), PromotionArchiveError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn write_manifest(directory: &Path) -> Result<(), PromotionArchiveError> {
    let files = archive_files(directory)?;
    let mut text = String::new();
    use core::fmt::Write as _;
    for relative in files {
        writeln!(
            text,
            "{}  {}",
            sha256_hex(&directory.join(&relative))?,
            relative.display()
        )
        .expect("writing a manifest to String cannot fail");
    }
    write_new_synced(&directory.join(MANIFEST_FILE), text.as_bytes())
}

fn verify_manifest(directory: &Path) -> Result<(), PromotionArchiveError> {
    let manifest_path = directory.join(MANIFEST_FILE);
    if !fs::symlink_metadata(&manifest_path)?.file_type().is_file() {
        return Err(invalid("promotion manifest must be a regular file"));
    }
    let manifest = fs::read_to_string(manifest_path)?;
    let mut expected = BTreeSet::new();
    for line in manifest.lines() {
        let (digest, relative_text) = line
            .split_once("  ")
            .ok_or_else(|| invalid("malformed promotion manifest row"))?;
        if !is_lower_hex_digest(digest) || !safe_relative_path(relative_text) {
            return Err(invalid("unsafe or malformed promotion manifest entry"));
        }
        let relative = PathBuf::from(relative_text);
        if !expected.insert(relative.clone()) || sha256_hex(&directory.join(relative))? != digest {
            return Err(invalid("promotion manifest checksum or path mismatch"));
        }
    }
    let actual: BTreeSet<_> = archive_files(directory)?.into_iter().collect();
    if actual != expected {
        return Err(invalid(
            "promotion manifest file set differs from directory",
        ));
    }
    Ok(())
}

fn archive_files(directory: &Path) -> Result<Vec<PathBuf>, PromotionArchiveError> {
    let mut files = Vec::new();
    collect_files(directory, directory, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), PromotionArchiveError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(invalid("promotion archives cannot contain symbolic links"));
        }
        if file_type.is_dir() {
            collect_files(root, &path, files)?;
        } else if file_type.is_file() && entry.file_name() != MANIFEST_FILE {
            files.push(
                path.strip_prefix(root)
                    .map_err(|_| invalid("promotion archive path escaped its root"))?
                    .to_owned(),
            );
        } else if !file_type.is_file() {
            return Err(invalid("promotion archive contains a non-regular entry"));
        }
    }
    Ok(())
}

fn regular_files(directory: &Path) -> Result<BTreeSet<PathBuf>, PromotionArchiveError> {
    if !directory.is_dir() {
        return Err(invalid("promotion records directory is missing"));
    }
    let mut files = BTreeSet::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            return Err(invalid("promotion records must be regular files"));
        }
        files.insert(PathBuf::from(entry.file_name()));
    }
    Ok(files)
}

fn sha256_hex(path: &Path) -> Result<String, PromotionArchiveError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
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

fn safe_relative_path(text: &str) -> bool {
    let path = Path::new(text);
    !text.is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn is_lower_hex_digest(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn sanitize(value: &impl ToString) -> String {
    value.to_string().replace(['\t', '\n', '\r'], " ")
}

fn sync_tree(directory: &Path) -> Result<(), PromotionArchiveError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            sync_tree(&entry.path())?;
        } else {
            File::open(entry.path())?.sync_all()?;
        }
    }
    File::open(directory)?.sync_all()?;
    Ok(())
}

struct TemporaryDirectory {
    path: PathBuf,
    published: bool,
}

impl TemporaryDirectory {
    fn publish(mut self, destination: &Path) -> Result<(), PromotionArchiveError> {
        fs::rename(&self.path, destination)?;
        self.published = true;
        File::open(destination.parent().unwrap_or_else(|| Path::new(".")))?.sync_all()?;
        Ok(())
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn temporary_directory(
    parent: &Path,
    stem: &str,
) -> Result<TemporaryDirectory, PromotionArchiveError> {
    for attempt in 0..1_000_u32 {
        let path = parent.join(format!("{stem}-{}-{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => {
                return Ok(TemporaryDirectory {
                    path,
                    published: false,
                })
            }
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(source.into()),
        }
    }
    Err(invalid("cannot reserve a temporary promotion directory"))
}

fn invalid(message: impl Into<String>) -> PromotionArchiveError {
    PromotionArchiveError::InvalidData(message.into())
}

#[derive(Debug)]
pub enum PromotionArchiveError {
    Io(io::Error),
    Json(serde_json::Error),
    Sprt(PromotionSprtError),
    Match(PromotionMatchError),
    Schedule(crate::PromotionScheduleError),
    InvalidData(String),
}

impl fmt::Display for PromotionArchiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => source.fmt(formatter),
            Self::Json(source) => source.fmt(formatter),
            Self::Sprt(source) => source.fmt(formatter),
            Self::Match(source) => source.fmt(formatter),
            Self::Schedule(source) => source.fmt(formatter),
            Self::InvalidData(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for PromotionArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::Sprt(source) => Some(source),
            Self::Match(source) => Some(source),
            Self::Schedule(source) => Some(source),
            Self::InvalidData(_) => None,
        }
    }
}

impl From<io::Error> for PromotionArchiveError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for PromotionArchiveError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

impl From<PromotionSprtError> for PromotionArchiveError {
    fn from(source: PromotionSprtError) -> Self {
        Self::Sprt(source)
    }
}

impl From<PromotionMatchError> for PromotionArchiveError {
    fn from(source: PromotionMatchError) -> Self {
        Self::Match(source)
    }
}

impl From<crate::PromotionScheduleError> for PromotionArchiveError {
    fn from(source: crate::PromotionScheduleError) -> Self {
        Self::Schedule(source)
    }
}

#[cfg(test)]
#[path = "promotion_archive/tests.rs"]
mod tests;
