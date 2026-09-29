use core::fmt;
use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

use paisho_ai::{
    play_match_from_record, Agent, MatchConfig, MatchError, MatchResult, MatchTask,
    MatchTermination,
};
use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId, StandardSetup, BASIC_FLOWERS};
use paisho_rating::{PentanomialCounts, PromotionSprtError};
use rayon::prelude::*;

use crate::NeutralStartConfigurationV1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromotionScheduledGame {
    pub pair_id: u64,
    pub leg: u8,
    pub candidate_is_host: bool,
    pub task: MatchTask,
    pub starting_record: GameRecord,
}

pub fn build_promotion_schedule(
    campaign_first_pair_id: u64,
    batch_first_pair_id: u64,
    pair_count: usize,
) -> Result<Vec<PromotionScheduledGame>, PromotionScheduleError> {
    build_promotion_schedule_with_neutral_starts(
        campaign_first_pair_id,
        batch_first_pair_id,
        pair_count,
        None,
    )
}

pub fn build_promotion_schedule_with_neutral_starts(
    campaign_first_pair_id: u64,
    batch_first_pair_id: u64,
    pair_count: usize,
    neutral_start: Option<NeutralStartConfigurationV1>,
) -> Result<Vec<PromotionScheduledGame>, PromotionScheduleError> {
    if pair_count == 0 {
        return Err(PromotionScheduleError::ZeroPairs);
    }
    let first_offset = batch_first_pair_id
        .checked_sub(campaign_first_pair_id)
        .ok_or(PromotionScheduleError::BatchBeforeCampaign)?;
    let game_count = pair_count
        .checked_mul(2)
        .ok_or(PromotionScheduleError::IdentifierOverflow)?;
    let pairs = (0..pair_count)
        .into_par_iter()
        .map(|pair_offset| {
            let pair_offset_u64 = u64::try_from(pair_offset)
                .map_err(|_| PromotionScheduleError::IdentifierOverflow)?;
            let pair_id = batch_first_pair_id
                .checked_add(pair_offset_u64)
                .ok_or(PromotionScheduleError::IdentifierOverflow)?;
            let opening_offset = first_offset
                .checked_add(pair_offset_u64)
                .ok_or(PromotionScheduleError::IdentifierOverflow)?;
            let opening_index = usize::try_from(opening_offset % BASIC_FLOWERS.len() as u64)
                .expect("a flower-cycle remainder always fits usize");
            let setup = StandardSetup::balanced(BASIC_FLOWERS[opening_index]);
            let starting_record = match neutral_start {
                Some(configuration) => configuration
                    .generate(setup, pair_id)
                    .map_err(|source| PromotionScheduleError::NeutralStart {
                        pair_id,
                        message: source.to_string(),
                    })?
                    .prefix()
                    .clone(),
                None => GameRecord::with_rules(setup, RuleProfileId::SkudPaiSho2022),
            };
            let mut games = Vec::with_capacity(2);
            for leg in 0..2_u8 {
                let game_id = pair_id
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(u64::from(leg)))
                    .ok_or(PromotionScheduleError::IdentifierOverflow)?;
                games.push(PromotionScheduledGame {
                    pair_id,
                    leg,
                    candidate_is_host: leg == 0,
                    task: MatchTask { id: game_id, setup },
                    starting_record: starting_record.clone(),
                });
            }
            <[_; 2]>::try_from(games).map_err(|_| PromotionScheduleError::CapacityUnavailable)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut schedule = Vec::new();
    schedule
        .try_reserve_exact(game_count)
        .map_err(|_| PromotionScheduleError::CapacityUnavailable)?;
    for pair in pairs {
        schedule.extend(pair);
    }
    Ok(schedule)
}

#[derive(Debug)]
pub struct PlayedPromotionGame {
    pub scheduled: PromotionScheduledGame,
    pub result: Result<MatchResult, MatchError>,
}

#[derive(Debug)]
pub struct PromotionBatchExecution {
    pub games: Vec<PlayedPromotionGame>,
    pub elapsed_seconds: f64,
    pub observed_match_workers: usize,
    pub worker_capacity: usize,
}

impl PromotionBatchExecution {
    pub fn summary(&self) -> Result<PromotionBatchSummary, PromotionMatchError> {
        summarize_games(&self.games)
    }

    pub fn pair_evaluations(&self) -> Result<Vec<PromotionPairEvaluation>, PromotionMatchError> {
        evaluate_pairs(&self.games)
    }
}

pub fn run_promotion_schedule<Candidate, Champion, MakeCandidate, MakeChampion>(
    schedule: &[PromotionScheduledGame],
    match_configuration: MatchConfig,
    make_candidate: MakeCandidate,
    make_champion: MakeChampion,
) -> PromotionBatchExecution
where
    Candidate: Agent + Send,
    Champion: Agent + Send,
    MakeCandidate: Fn(&PromotionScheduledGame, Player) -> Candidate + Sync,
    MakeChampion: Fn(&PromotionScheduledGame, Player) -> Champion + Sync,
{
    let observed_workers = Mutex::new(HashSet::new());
    let started = Instant::now();
    let games = schedule
        .par_iter()
        .map(|scheduled| {
            if let Some(worker) = rayon::current_thread_index() {
                observed_workers
                    .lock()
                    .expect("worker observation lock is not poisoned")
                    .insert(worker);
            }
            let candidate_player = if scheduled.candidate_is_host {
                Player::Host
            } else {
                Player::Guest
            };
            let champion_player = candidate_player.opponent();
            let mut candidate = make_candidate(scheduled, candidate_player);
            let mut champion = make_champion(scheduled, champion_player);
            let result = if scheduled.candidate_is_host {
                play_match_from_record(
                    scheduled.task.id,
                    scheduled.starting_record.clone(),
                    match_configuration,
                    &mut candidate,
                    &mut champion,
                )
            } else {
                play_match_from_record(
                    scheduled.task.id,
                    scheduled.starting_record.clone(),
                    match_configuration,
                    &mut champion,
                    &mut candidate,
                )
            };
            PlayedPromotionGame {
                scheduled: scheduled.clone(),
                result,
            }
        })
        .collect();
    PromotionBatchExecution {
        games,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        observed_match_workers: observed_workers
            .into_inner()
            .expect("worker observation lock is not poisoned")
            .len(),
        worker_capacity: rayon::current_num_threads(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairExclusionReason {
    MatchError,
    DecisionLimit,
}

impl PairExclusionReason {
    pub const fn code(self) -> &'static str {
        match self {
            Self::MatchError => "match-error",
            Self::DecisionLimit => "decision-limit",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromotionPairEvaluation {
    pub pair_id: u64,
    pub candidate_half_points: Option<u8>,
    pub exclusion: Option<PairExclusionReason>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PromotionBatchSummary {
    pub attempted_pairs: usize,
    pub eligible_pairs: usize,
    pub excluded_pairs: usize,
    pub pentanomial: PentanomialCounts,
}

fn summarize_games(
    games: &[PlayedPromotionGame],
) -> Result<PromotionBatchSummary, PromotionMatchError> {
    let evaluations = evaluate_pairs(games)?;
    let mut pentanomial = PentanomialCounts::default();
    let mut eligible_pairs = 0_usize;
    for evaluation in &evaluations {
        if let Some(half_points) = evaluation.candidate_half_points {
            pentanomial.observe_pair_half_points(half_points)?;
            eligible_pairs += 1;
        }
    }
    Ok(PromotionBatchSummary {
        attempted_pairs: evaluations.len(),
        eligible_pairs,
        excluded_pairs: evaluations.len() - eligible_pairs,
        pentanomial,
    })
}

fn evaluate_pairs(
    games: &[PlayedPromotionGame],
) -> Result<Vec<PromotionPairEvaluation>, PromotionMatchError> {
    if games.is_empty() || games.len() % 2 != 0 {
        return Err(PromotionMatchError::MalformedSchedule(
            "a promotion batch must contain one or more complete two-game pairs",
        ));
    }
    games
        .chunks_exact(2)
        .map(evaluate_pair)
        .collect::<Result<Vec<_>, _>>()
}

fn evaluate_pair(
    pair: &[PlayedPromotionGame],
) -> Result<PromotionPairEvaluation, PromotionMatchError> {
    let [first, second] = pair else {
        unreachable!("chunks_exact(2) always yields two games")
    };
    if first.scheduled.pair_id != second.scheduled.pair_id
        || first.scheduled.leg != 0
        || second.scheduled.leg != 1
        || !first.scheduled.candidate_is_host
        || second.scheduled.candidate_is_host
        || first.scheduled.task.setup != second.scheduled.task.setup
        || first.scheduled.starting_record != second.scheduled.starting_record
    {
        return Err(PromotionMatchError::MalformedSchedule(
            "promotion pair does not reverse seats on one setup",
        ));
    }
    let first_outcome = eligible_outcome(first)?;
    let second_outcome = eligible_outcome(second)?;
    let exclusion = match (first_outcome, second_outcome) {
        (GameEligibility::MatchError, _) | (_, GameEligibility::MatchError) => {
            Some(PairExclusionReason::MatchError)
        }
        (GameEligibility::DecisionLimit, _) | (_, GameEligibility::DecisionLimit) => {
            Some(PairExclusionReason::DecisionLimit)
        }
        (GameEligibility::Rated(_), GameEligibility::Rated(_)) => None,
    };
    let candidate_half_points = match (first_outcome, second_outcome) {
        (GameEligibility::Rated(first), GameEligibility::Rated(second)) => Some(
            candidate_game_half_points(first, true) + candidate_game_half_points(second, false),
        ),
        _ => None,
    };
    Ok(PromotionPairEvaluation {
        pair_id: first.scheduled.pair_id,
        candidate_half_points,
        exclusion,
    })
}

#[derive(Clone, Copy)]
enum GameEligibility {
    Rated(GameOutcome),
    DecisionLimit,
    MatchError,
}

fn eligible_outcome(game: &PlayedPromotionGame) -> Result<GameEligibility, PromotionMatchError> {
    let result = match &game.result {
        Ok(result) => result,
        Err(_) => return Ok(GameEligibility::MatchError),
    };
    if result.task_id != game.scheduled.task.id
        || result.record.setup() != game.scheduled.task.setup
        || !result
            .record
            .actions()
            .starts_with(game.scheduled.starting_record.actions())
        || result.record.replay().ok().as_ref() != Some(&result.final_position)
    {
        return Err(PromotionMatchError::MalformedResult {
            game_id: game.scheduled.task.id,
        });
    }
    match result.termination {
        MatchTermination::Rules(outcome)
            if outcome != GameOutcome::Ongoing && result.final_position.outcome() == outcome =>
        {
            Ok(GameEligibility::Rated(outcome))
        }
        MatchTermination::DecisionLimit
            if result.final_position.outcome() == GameOutcome::Ongoing =>
        {
            Ok(GameEligibility::DecisionLimit)
        }
        _ => Err(PromotionMatchError::MalformedResult {
            game_id: game.scheduled.task.id,
        }),
    }
}

const fn candidate_game_half_points(outcome: GameOutcome, candidate_is_host: bool) -> u8 {
    match outcome {
        GameOutcome::Draw => 1,
        GameOutcome::Win(Player::Host) if candidate_is_host => 2,
        GameOutcome::Win(Player::Guest) if !candidate_is_host => 2,
        GameOutcome::Win(_) => 0,
        GameOutcome::Ongoing => unreachable!(),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromotionScheduleError {
    ZeroPairs,
    BatchBeforeCampaign,
    IdentifierOverflow,
    CapacityUnavailable,
    NeutralStart { pair_id: u64, message: String },
}

impl fmt::Display for PromotionScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPairs => formatter.write_str("a promotion batch needs at least one pair"),
            Self::BatchBeforeCampaign => {
                formatter.write_str("a promotion batch cannot precede its campaign")
            }
            Self::IdentifierOverflow => {
                formatter.write_str("promotion pair or game identifiers overflow")
            }
            Self::CapacityUnavailable => {
                formatter.write_str("promotion schedule allocation is unavailable")
            }
            Self::NeutralStart { pair_id, message } => {
                write!(
                    formatter,
                    "promotion pair {pair_id} has no neutral start: {message}"
                )
            }
        }
    }
}

impl std::error::Error for PromotionScheduleError {}

#[derive(Debug)]
pub enum PromotionMatchError {
    MalformedSchedule(&'static str),
    MalformedResult { game_id: u64 },
    Sprt(PromotionSprtError),
}

impl fmt::Display for PromotionMatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedSchedule(reason) => {
                write!(formatter, "malformed promotion schedule: {reason}")
            }
            Self::MalformedResult { game_id } => {
                write!(
                    formatter,
                    "promotion game {game_id} has an inconsistent result"
                )
            }
            Self::Sprt(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for PromotionMatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sprt(source) => Some(source),
            _ => None,
        }
    }
}

impl From<PromotionSprtError> for PromotionMatchError {
    fn from(source: PromotionSprtError) -> Self {
        Self::Sprt(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_reverses_seats_and_cycles_openings_across_batches() {
        let first = build_promotion_schedule(100, 100, 4).unwrap();
        let second = build_promotion_schedule(100, 104, 3).unwrap();
        let combined = first.into_iter().chain(second).collect::<Vec<_>>();
        assert_eq!(combined.len(), 14);
        for (offset, pair) in combined.chunks_exact(2).enumerate() {
            for game in pair {
                assert_eq!(game.starting_record.rules(), RuleProfileId::SkudPaiSho2022);
            }
            assert_eq!(pair[0].pair_id, 100 + offset as u64);
            assert_eq!(pair[0].leg, 0);
            assert_eq!(pair[1].leg, 1);
            assert!(pair[0].candidate_is_host);
            assert!(!pair[1].candidate_is_host);
            assert_eq!(pair[0].task.setup, pair[1].task.setup);
            assert_eq!(
                pair[0].task.setup.starting_flower,
                BASIC_FLOWERS[offset % BASIC_FLOWERS.len()]
            );
        }
    }

    #[test]
    fn neutral_schedule_replays_one_identical_prefix_for_both_legs() {
        let configuration = NeutralStartConfigurationV1::new(71, 32, 4_096, 16).unwrap();
        let first =
            build_promotion_schedule_with_neutral_starts(100, 100, 1, Some(configuration)).unwrap();
        let repeated =
            build_promotion_schedule_with_neutral_starts(100, 100, 1, Some(configuration)).unwrap();

        assert_eq!(first, repeated);
        assert_eq!(first[0].starting_record, first[1].starting_record);
        assert!(!first[0].starting_record.actions().is_empty());
        let position = first[0].starting_record.replay().unwrap();
        assert_eq!(position.outcome(), GameOutcome::Ongoing);
        assert_eq!(position.phase(), paisho_core::TurnPhase::Main);
    }

    #[test]
    fn pair_score_uses_the_candidates_perspective_in_both_legs() {
        assert_eq!(
            candidate_game_half_points(GameOutcome::Win(Player::Host), true)
                + candidate_game_half_points(GameOutcome::Win(Player::Host), false),
            2
        );
        assert_eq!(
            candidate_game_half_points(GameOutcome::Win(Player::Host), true)
                + candidate_game_half_points(GameOutcome::Win(Player::Guest), false),
            4
        );
        assert_eq!(
            candidate_game_half_points(GameOutcome::Draw, true)
                + candidate_game_half_points(GameOutcome::Draw, false),
            2
        );
    }

    #[test]
    fn schedule_rejects_empty_backward_and_overflowing_ranges() {
        assert_eq!(
            build_promotion_schedule(0, 0, 0),
            Err(PromotionScheduleError::ZeroPairs)
        );
        assert_eq!(
            build_promotion_schedule(10, 9, 1),
            Err(PromotionScheduleError::BatchBeforeCampaign)
        );
        assert_eq!(
            build_promotion_schedule(0, u64::MAX, 1),
            Err(PromotionScheduleError::IdentifierOverflow)
        );
    }
}
