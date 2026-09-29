//! In-memory equivalent of the paisho-actors collection protocol.
//! The caller owns the inference service, frozen producer identity and reusable Rayon pool.
use std::error::Error;

use paisho_ai::{
    AgentError, AgentTelemetry, MctsAgent, NetworkPolicy, PolicyValueEvaluator, PureNetworkAgent,
    RandomAgent, SiteBotV1, StableRng,
};
use paisho_core::{Action, Player, Position, StandardSetup, BASIC_FLOWERS};
use paisho_replay::{ReplayDigestV1, ReplayGameV1, ReplayValidationError};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

use crate::{
    evaluation_mcts_configuration, play_replay_parallel, MctsReplayAgent,
    NeutralStartConfigurationV1, RecordedNetworkAgent, ReplayActorAgent, ReplayActorChoice,
    ReplayMatchConfiguration, ReplayMatchError, ReplayMatchResult, ReplayMatchTask,
    UnrecordedReplayAgent,
};

pub type LiveActorsError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LiveActorsOpponent {
    SelfPlay,
    Random,
    Site,
    Mcts { simulations: usize },
}

impl LiveActorsOpponent {
    fn paired(self) -> bool {
        !matches!(self, Self::SelfPlay)
    }
}

#[derive(Clone, Debug)]
pub struct LiveActorsConfiguration {
    /// Concurrent match tasks per round, not Rayon worker count.
    pub actors: usize,
    pub target_games: usize,
    pub maximum_attempts: usize,
    pub first_game_id: u64,
    pub decision_soft_limit: usize,
    pub actor_seed: u64,
    pub policy: NetworkPolicy,
    pub neutral_start: Option<NeutralStartConfigurationV1>,
    pub opponent: LiveActorsOpponent,
}

impl Default for LiveActorsConfiguration {
    fn default() -> Self {
        Self {
            actors: 80,
            target_games: 512,
            maximum_attempts: 2048,
            first_game_id: 0,
            decision_soft_limit: 256,
            actor_seed: 0x5041_4953_484f_2026,
            policy: NetworkPolicy::Sample {
                temperature: 1.0,
                uniform_mix: 0.05,
            },
            neutral_start: Some(
                NeutralStartConfigurationV1::new(0x4e45_5554_5241_4c31, 64, 16_384, 16)
                    .expect("constant neutral-start defaults are valid"),
            ),
            opponent: LiveActorsOpponent::Random,
        }
    }
}

impl LiveActorsConfiguration {
    pub fn validate(&self) -> Result<(), LiveActorsError> {
        if self.actors == 0 || self.target_games == 0 || self.decision_soft_limit == 0 {
            return Err("actors, target games and decision limit must be positive".into());
        }
        if self.maximum_attempts < self.target_games {
            return Err("maximum attempts cannot be smaller than target games".into());
        }
        if self.opponent.paired() && (self.actors % 2 != 0 || self.target_games % 2 != 0) {
            return Err("distinct opponents require even actors and target games".into());
        }
        if let LiveActorsOpponent::Mcts { simulations } = self.opponent {
            if !(1..=512).contains(&simulations) {
                return Err("MCTS opponent simulations must be in 1..=512".into());
            }
        }
        self.first_game_id
            .checked_add(u64::try_from(self.maximum_attempts - 1)?)
            .ok_or("game id overflow")?;
        self.policy.validate()?;
        Ok(())
    }
}

/// Partial results survive exhausted budgets and operational failures, as in the CLI.
/// Check `target_reached` and `abort_reason` before treating a collection as complete.
#[derive(Default, Debug)]
pub struct LiveActorsReport {
    pub start_preparation_seconds: f64,
    pub match_seconds: f64,
    pub retained: Vec<ReplayMatchResult>,
    pub attempts: usize,
    pub completed: usize,
    pub interrupted: usize,
    pub no_training_decision: usize,
    pub failed: usize,
    pub excluded_pairs: usize,
    pub maximum_workers_observed: usize,
    pub worker_capacity: usize,
    pub target_reached: bool,
    pub abort_reason: Option<String>,
}

impl LiveActorsReport {
    pub fn into_games(self) -> Vec<ReplayGameV1> {
        self.retained
            .into_iter()
            .map(|result| result.game)
            .collect()
    }

    fn abort(&mut self, reason: impl Into<String>) {
        if self.abort_reason.is_none() {
            self.abort_reason = Some(reason.into());
        }
    }
}

/// No broker launch, checkpoint access or publication. Reuse a caller-created pool
/// (e.g. 20 workers) across cycles. E accepts CapacityInferenceBrokerClient or CPU mocks.
/// Keep the evaluator's model frozen and bound to `producer` throughout this call.
/// Equal retained-game quotas, same frozen network, disjoint attempt IDs.
pub fn collect_live_mixed_games<E: PolicyValueEvaluator>(
    config: &LiveActorsConfiguration,
    pool: &rayon::ThreadPool,
    evaluator: E,
    producer: ReplayDigestV1,
) -> Result<LiveActorsReport, LiveActorsError> {
    if config.opponent == LiveActorsOpponent::SelfPlay {
        return collect_live_games(config, pool, evaluator, producer);
    }
    if config.target_games % 4 != 0 || config.maximum_attempts % 4 != 0 {
        return Err("50/50 paired collection needs games and attempts divisible by four".into());
    }
    collect_live_weighted_games(config, config.target_games / 2, pool, evaluator, producer)
}

/// Exact retained quotas with a proportional attempt budget; all games use one frozen network.
/// External games remain reversed-seat pairs. Zero external games means pure self-play.
pub fn collect_live_weighted_games<E: PolicyValueEvaluator>(
    config: &LiveActorsConfiguration,
    external_games: usize,
    pool: &rayon::ThreadPool,
    evaluator: E,
    producer: ReplayDigestV1,
) -> Result<LiveActorsReport, LiveActorsError> {
    config.validate()?;
    if external_games > config.target_games || external_games % 2 != 0 {
        return Err("external retained quota must be even and no larger than total games".into());
    }
    if config.opponent == LiveActorsOpponent::SelfPlay || external_games == 0 {
        let mut own = config.clone();
        own.opponent = LiveActorsOpponent::SelfPlay;
        return collect_live_games(&own, pool, evaluator, producer);
    }
    if external_games == config.target_games {
        return collect_live_games(config, pool, evaluator, producer);
    }
    let mut external = config.clone();
    external.target_games = external_games;
    external.maximum_attempts = usize::try_from(
        config.maximum_attempts as u128 * external_games as u128 / config.target_games as u128,
    )? & !1;
    let mut own = config.clone();
    own.target_games -= external_games;
    own.maximum_attempts -= external.maximum_attempts;
    own.opponent = LiveActorsOpponent::SelfPlay;
    own.first_game_id = config
        .first_game_id
        .checked_add(external.maximum_attempts as u64)
        .ok_or("mixed game id overflow")?;
    external.validate()?;
    own.validate()?;
    let mut a = LiveActorsReport {
        worker_capacity: pool.current_num_threads(),
        ..Default::default()
    };
    let mut b = LiveActorsReport {
        worker_capacity: pool.current_num_threads(),
        ..Default::default()
    };
    let opponent_digest = ReplayDigestV1::from_bytes(
        Sha256::digest(
            format!(
        "paisho-live-opponent-v1;opponent={:?};source={};seed-policy=game-seat-splitmix64-v1",
        config.opponent, env!("PAISHO_BUILD_SOURCE_SHA256"))
            .as_bytes(),
        )
        .into(),
    );
    loop {
        // Both quotas enter one task batch. Only the unfinished quota is replenished.
        let external_capacity =
            if b.retained.len() < own.target_games && b.attempts < own.maximum_attempts {
                (usize::try_from(
                    config.actors as u128 * external_games as u128 / config.target_games as u128,
                )?)
                .max(2)
                    & !1
            } else {
                config.actors
            };
        let external_count = round_size(&external, &a).min(external_capacity);
        let own_count = round_size(&own, &b).min(config.actors - external_count);
        if external_count + own_count == 0 || a.abort_reason.is_some() || b.abort_reason.is_some() {
            break;
        }
        let timer = std::time::Instant::now();
        let tasks = pool.install(|| -> Result<_, LiveActorsError> {
            let (left, right) = rayon::join(
                || build_tasks(&external, a.attempts, external_count),
                || build_tasks(&own, b.attempts, own_count),
            );
            let mut tasks = left?;
            tasks.extend(right?);
            Ok(tasks)
        });
        let tasks = match tasks {
            Ok(tasks) => tasks,
            Err(error) => {
                a.abort(error.to_string());
                break;
            }
        };
        a.start_preparation_seconds += timer.elapsed().as_secs_f64();
        let select = |task: &ReplayMatchTask, player| {
            let configuration = if task.game_id < own.first_game_id {
                &external
            } else {
                &own
            };
            make_agent(
                configuration,
                evaluator.clone(),
                producer,
                opponent_digest,
                task,
                player,
            )
        };
        let timer = std::time::Instant::now();
        let batch = pool.install(|| {
            play_replay_parallel(
                &tasks,
                ReplayMatchConfiguration {
                    decision_soft_limit: config.decision_soft_limit,
                },
                |task| select(task, Player::Host),
                |task| select(task, Player::Guest),
            )
        });
        a.match_seconds += timer.elapsed().as_secs_f64();
        a.maximum_workers_observed = a.maximum_workers_observed.max(batch.workers);
        let mut results = batch.matches;
        let own_results = results.split_off(external_count);
        a.attempts += external_count;
        b.attempts += own_count;
        absorb_matches(&mut a, results, external.opponent);
        absorb_matches(&mut b, own_results, own.opponent);
    }
    a.target_reached = a.retained.len() == external.target_games;
    b.target_reached = b.retained.len() == own.target_games;
    a.retained.append(&mut b.retained);
    a.start_preparation_seconds += b.start_preparation_seconds;
    a.match_seconds += b.match_seconds;
    a.attempts += b.attempts;
    a.completed += b.completed;
    a.interrupted += b.interrupted;
    a.no_training_decision += b.no_training_decision;
    a.failed += b.failed;
    a.excluded_pairs += b.excluded_pairs;
    a.maximum_workers_observed = a.maximum_workers_observed.max(b.maximum_workers_observed);
    a.target_reached &= b.target_reached;
    if a.abort_reason.is_none() {
        a.abort_reason = b.abort_reason;
    }
    Ok(a)
}

pub fn collect_live_games<E: PolicyValueEvaluator>(
    config: &LiveActorsConfiguration,
    pool: &rayon::ThreadPool,
    evaluator: E,
    producer: ReplayDigestV1,
) -> Result<LiveActorsReport, LiveActorsError> {
    config.validate()?;
    let opponent_digest = ReplayDigestV1::from_bytes(
        Sha256::digest(
            format!(
        "paisho-live-opponent-v1;opponent={:?};source={};seed-policy=game-seat-splitmix64-v1",
        config.opponent, env!("PAISHO_BUILD_SOURCE_SHA256"),
    )
            .as_bytes(),
        )
        .into(),
    );
    let mut report = LiveActorsReport {
        worker_capacity: pool.current_num_threads(),
        ..Default::default()
    };
    while report.retained.len() < config.target_games
        && report.attempts < config.maximum_attempts
        && report.abort_reason.is_none()
    {
        let count = round_size(config, &report);
        if count == 0 {
            break;
        }
        let timer = std::time::Instant::now();
        let tasks = match pool.install(|| build_tasks(config, report.attempts, count)) {
            Ok(tasks) => tasks,
            Err(error) => {
                report.abort(error.to_string());
                break;
            }
        };
        report.start_preparation_seconds += timer.elapsed().as_secs_f64();
        let timer = std::time::Instant::now();
        let batch = pool.install(|| {
            play_replay_parallel(
                &tasks,
                ReplayMatchConfiguration {
                    decision_soft_limit: config.decision_soft_limit,
                },
                |task| {
                    make_agent(
                        config,
                        evaluator.clone(),
                        producer,
                        opponent_digest,
                        task,
                        Player::Host,
                    )
                },
                |task| {
                    make_agent(
                        config,
                        evaluator.clone(),
                        producer,
                        opponent_digest,
                        task,
                        Player::Guest,
                    )
                },
            )
        });
        report.match_seconds += timer.elapsed().as_secs_f64();
        report.maximum_workers_observed = report.maximum_workers_observed.max(batch.workers);
        report.attempts += count;
        absorb_matches(&mut report, batch.matches, config.opponent);
    }
    report.target_reached = report.retained.len() == config.target_games;
    Ok(report)
}

// Dynamic dispatch only at the actor boundary; evaluator clones can borrow caller state.
struct LiveAgent<'a>(Box<dyn ReplayActorAgent + Send + 'a>);
impl ReplayActorAgent for LiveAgent<'_> {
    fn digest(&self) -> ReplayDigestV1 {
        self.0.digest()
    }
    fn select_for_replay(
        &mut self,
        position: &Position,
        actions: &[Action],
    ) -> Result<ReplayActorChoice, AgentError> {
        self.0.select_for_replay(position, actions)
    }
    fn telemetry(&self) -> AgentTelemetry {
        self.0.telemetry()
    }
    fn reset_telemetry(&mut self) {
        self.0.reset_telemetry();
    }
}

fn make_agent<'a, E: PolicyValueEvaluator + 'a>(
    config: &LiveActorsConfiguration,
    evaluator: E,
    producer: ReplayDigestV1,
    opponent_digest: ReplayDigestV1,
    task: &ReplayMatchTask,
    player: Player,
) -> LiveAgent<'a> {
    let candidate_is_host = (task.game_id - config.first_game_id) % 2 == 0;
    if !config.opponent.paired() || candidate_is_host == (player == Player::Host) {
        let network = PureNetworkAgent::new(
            evaluator,
            agent_seed(config.actor_seed, task.game_id, player, 0),
            config.policy,
        )
        .expect("policy validated before collection");
        return LiveAgent(Box::new(RecordedNetworkAgent::new(producer, network)));
    }
    let seed = agent_seed(config.actor_seed, task.game_id, player, 1);
    match config.opponent {
        LiveActorsOpponent::Random => LiveAgent(Box::new(UnrecordedReplayAgent::new(
            opponent_digest,
            RandomAgent::new(seed),
        ))),
        LiveActorsOpponent::Site => LiveAgent(Box::new(UnrecordedReplayAgent::new(
            opponent_digest,
            SiteBotV1::new(seed),
        ))),
        LiveActorsOpponent::Mcts { simulations } => LiveAgent(Box::new(MctsReplayAgent::new(
            opponent_digest,
            MctsAgent::new(seed, evaluation_mcts_configuration(simulations))
                .expect("MCTS budget validated"),
        ))),
        LiveActorsOpponent::SelfPlay => unreachable!("self-play always creates a network actor"),
    }
}

// Exact CLI seed splitting: game ID, seat and candidate/opponent family all contribute.
fn agent_seed(base: u64, game_id: u64, player: Player, family: u64) -> u64 {
    let side = match player {
        Player::Host => 0x484f_5354,
        Player::Guest => 0x0047_5545_5354,
    };
    StableRng::new(
        base ^ game_id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ side ^ family.rotate_left(29),
    )
    .next_u64()
}

fn round_size(config: &LiveActorsConfiguration, report: &LiveActorsReport) -> usize {
    let mut size = config
        .actors
        .min(config.maximum_attempts.saturating_sub(report.attempts))
        .min(config.target_games.saturating_sub(report.retained.len()));
    if config.opponent.paired() {
        size -= size % 2;
    }
    size
}

fn build_tasks(
    config: &LiveActorsConfiguration,
    first_attempt: usize,
    count: usize,
) -> Result<Vec<ReplayMatchTask>, LiveActorsError> {
    if config.opponent.paired() && (first_attempt % 2 != 0 || count % 2 != 0) {
        return Err("paired actor rounds must begin and end on pair boundaries".into());
    }
    let divisor = if config.opponent.paired() { 2 } else { 1 };
    let first_slot = first_attempt / divisor;
    let starts = config
        .neutral_start
        .map(|configuration| {
            (0..count / divisor)
                .into_par_iter()
                .map(|offset| -> Result<_, LiveActorsError> {
                    let slot = first_slot
                        .checked_add(offset)
                        .ok_or("neutral-start slot overflow")?;
                    let first_slot_attempt = slot
                        .checked_mul(divisor)
                        .ok_or("neutral-start attempt overflow")?;
                    let ordinal = config
                        .first_game_id
                        .checked_add(u64::try_from(first_slot_attempt)?)
                        .ok_or("neutral-start game id overflow")?;
                    Ok(configuration.generate(
                        StandardSetup::balanced(BASIC_FLOWERS[slot % BASIC_FLOWERS.len()]),
                        ordinal,
                    )?)
                })
                .collect::<Result<Vec<_>, LiveActorsError>>()
        })
        .transpose()?;
    (0..count)
        .map(|offset| {
            let attempt = first_attempt
                .checked_add(offset)
                .ok_or("actor attempt overflow")?;
            let game_id = config
                .first_game_id
                .checked_add(u64::try_from(attempt)?)
                .ok_or("game id overflow")?;
            let slot = attempt / divisor;
            Ok(match &starts {
                Some(starts) => ReplayMatchTask::from_neutral(game_id, &starts[slot - first_slot]),
                None => ReplayMatchTask::standard(
                    game_id,
                    StandardSetup::balanced(BASIC_FLOWERS[slot % BASIC_FLOWERS.len()]),
                ),
            })
        })
        .collect()
}

fn absorb_matches(
    report: &mut LiveActorsReport,
    matches: Vec<Result<ReplayMatchResult, ReplayMatchError>>,
    opponent: LiveActorsOpponent,
) {
    let mut matches = matches.into_iter();
    while let Some(first) = matches.next() {
        let first = classify_match(report, first);
        if !opponent.paired() {
            report.retained.extend(first);
            continue;
        }
        let second = classify_match(
            report,
            matches
                .next()
                .expect("paired batches contain complete pairs"),
        );
        match (first, second) {
            (Some(first), Some(second)) => report.retained.extend([first, second]),
            _ => report.excluded_pairs += 1,
        }
    }
}

fn classify_match(
    report: &mut LiveActorsReport,
    result: Result<ReplayMatchResult, ReplayMatchError>,
) -> Option<ReplayMatchResult> {
    match result {
        Ok(result) => {
            report.completed += 1;
            Some(result)
        }
        Err(ReplayMatchError::DecisionLimit { .. }) => {
            report.interrupted += 1;
            None
        }
        Err(ReplayMatchError::Replay(ReplayValidationError::NoTrainingDecision)) => {
            report.no_training_decision += 1;
            None
        }
        Err(error) => {
            report.failed += 1;
            report.abort(format!("actor match failed: {error}"));
            None
        }
    }
}

#[cfg(test)]
#[path = "live_actors/tests.rs"]
mod tests;
