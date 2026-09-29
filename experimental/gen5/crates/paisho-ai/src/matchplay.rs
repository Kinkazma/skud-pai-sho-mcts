use core::fmt;
use std::collections::HashSet;
use std::sync::Mutex;

use paisho_core::{
    legal_actions, Action, ApplyError, GameOutcome, GameRecord, Player, Position, ReplayError,
    StandardSetup, TurnPhase,
};
use rayon::prelude::*;

use crate::{Agent, AgentError, AgentTelemetry};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchConfig {
    /// Soft decision ceiling. A Harmony Bonus opened by the last admitted main
    /// action receives one grace decision so the archived position ends a turn.
    pub decision_soft_limit: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            decision_soft_limit: 2_048,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchTask {
    pub id: u64,
    pub setup: StandardSetup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchTermination {
    Rules(GameOutcome),
    DecisionLimit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchResult {
    pub task_id: u64,
    pub final_position: Position,
    pub record: GameRecord,
    pub termination: MatchTermination,
    pub host_telemetry: AgentTelemetry,
    pub guest_telemetry: AgentTelemetry,
}

impl MatchResult {
    /// Returns a result eligible for rated evaluation. A decision-limit stop
    /// is deliberately not converted into a draw.
    pub const fn scored_outcome(&self) -> Option<GameOutcome> {
        match self.termination {
            MatchTermination::Rules(outcome) => Some(outcome),
            MatchTermination::DecisionLimit => None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParallelMatchResults {
    /// Distinct Rayon workers that actually started at least one match.
    pub workers: usize,
    pub worker_capacity: usize,
    pub matches: Vec<Result<MatchResult, MatchError>>,
}

pub fn play_match<Host: Agent, Guest: Agent>(
    task: MatchTask,
    config: MatchConfig,
    host: &mut Host,
    guest: &mut Guest,
) -> Result<MatchResult, MatchError> {
    play_match_from_record(task.id, GameRecord::new(task.setup), config, host, guest)
}

/// Continues a match from a replayable, ongoing main-phase record. The decision
/// limit applies only to newly played decisions; the returned record retains
/// the complete prefix.
pub fn play_match_from_record<Host: Agent, Guest: Agent>(
    task_id: u64,
    mut record: GameRecord,
    config: MatchConfig,
    host: &mut Host,
    guest: &mut Guest,
) -> Result<MatchResult, MatchError> {
    host.reset_telemetry();
    guest.reset_telemetry();
    let mut position = record
        .replay()
        .map_err(|source| MatchError::StartingRecordRejected { task_id, source })?;
    if position.outcome() != GameOutcome::Ongoing {
        return Err(MatchError::StartingPositionTerminal {
            task_id,
            decisions: record.actions().len(),
        });
    }
    if position.phase() != TurnPhase::Main {
        return Err(MatchError::StartingPositionInBonus {
            task_id,
            decisions: record.actions().len(),
        });
    }
    let mut decisions = 0;

    loop {
        if position.outcome() != GameOutcome::Ongoing {
            return Ok(completed_result(
                task_id,
                position,
                record,
                host.telemetry(),
                guest.telemetry(),
            ));
        }
        if decisions >= config.decision_soft_limit && position.phase() == TurnPhase::Main {
            break;
        }

        let actions = legal_actions(&position);
        if actions.is_empty() {
            return Err(MatchError::NoLegalAction {
                player: position.to_move(),
            });
        }
        let player = position.to_move();
        let selected = match player {
            Player::Host => host.select_action(&position, &actions),
            Player::Guest => guest.select_action(&position, &actions),
        }
        .map_err(|source| MatchError::AgentFailure { player, source })?;
        let action = actions
            .get(selected)
            .copied()
            .ok_or(MatchError::AgentChoiceOutOfRange {
                player,
                selected,
                legal_action_count: actions.len(),
            })?;
        position
            .apply(action)
            .map_err(|source| MatchError::GeneratedActionRejected {
                player,
                action,
                source,
            })?;
        record.push(action);
        decisions += 1;
    }

    Ok(MatchResult {
        task_id,
        final_position: position,
        record,
        termination: MatchTermination::DecisionLimit,
        host_telemetry: host.telemetry(),
        guest_telemetry: guest.telemetry(),
    })
}

pub fn play_parallel<Host, Guest, MakeHost, MakeGuest>(
    tasks: &[MatchTask],
    config: MatchConfig,
    make_host: MakeHost,
    make_guest: MakeGuest,
) -> ParallelMatchResults
where
    Host: Agent + Send,
    Guest: Agent + Send,
    MakeHost: Fn(&MatchTask) -> Host + Sync,
    MakeGuest: Fn(&MatchTask) -> Guest + Sync,
{
    let observed_workers = Mutex::new(HashSet::new());
    let matches = tasks
        .par_iter()
        .map(|task| {
            if let Some(index) = rayon::current_thread_index() {
                observed_workers
                    .lock()
                    .expect("worker observation lock is not poisoned")
                    .insert(index);
            }
            let mut host = make_host(task);
            let mut guest = make_guest(task);
            play_match(*task, config, &mut host, &mut guest)
        })
        .collect();
    let workers = observed_workers
        .into_inner()
        .expect("worker observation lock is not poisoned")
        .len();
    ParallelMatchResults {
        workers,
        worker_capacity: rayon::current_num_threads(),
        matches,
    }
}

fn completed_result(
    task_id: u64,
    position: Position,
    record: GameRecord,
    host_telemetry: AgentTelemetry,
    guest_telemetry: AgentTelemetry,
) -> MatchResult {
    MatchResult {
        task_id,
        termination: MatchTermination::Rules(position.outcome()),
        final_position: position,
        record,
        host_telemetry,
        guest_telemetry,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MatchError {
    StartingRecordRejected {
        task_id: u64,
        source: ReplayError,
    },
    StartingPositionTerminal {
        task_id: u64,
        decisions: usize,
    },
    StartingPositionInBonus {
        task_id: u64,
        decisions: usize,
    },
    NoLegalAction {
        player: Player,
    },
    AgentChoiceOutOfRange {
        player: Player,
        selected: usize,
        legal_action_count: usize,
    },
    AgentFailure {
        player: Player,
        source: AgentError,
    },
    GeneratedActionRejected {
        player: Player,
        action: Action,
        source: ApplyError,
    },
}

impl fmt::Display for MatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StartingRecordRejected { task_id, source } => {
                write!(
                    formatter,
                    "match {task_id} has an invalid starting record: {source}"
                )
            }
            Self::StartingPositionTerminal { task_id, decisions } => write!(
                formatter,
                "match {task_id} starts from a terminal record of {decisions} decisions"
            ),
            Self::StartingPositionInBonus { task_id, decisions } => write!(
                formatter,
                "match {task_id} starts inside a Harmony Bonus after {decisions} decisions"
            ),
            Self::NoLegalAction { player } => {
                write!(
                    formatter,
                    "{player:?} has no legal action in an ongoing game"
                )
            }
            Self::AgentChoiceOutOfRange {
                player,
                selected,
                legal_action_count,
            } => write!(
                formatter,
                "{player:?} selected action {selected}, but only {legal_action_count} are legal"
            ),
            Self::AgentFailure { player, source } => {
                write!(
                    formatter,
                    "{player:?} agent failed to choose an action: {source}"
                )
            }
            Self::GeneratedActionRejected {
                player,
                action,
                source,
            } => write!(
                formatter,
                "the engine rejected generated action {action} for {player:?}: {source}"
            ),
        }
    }
}

impl std::error::Error for MatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::StartingRecordRejected { source, .. } => Some(source),
            Self::AgentFailure { source, .. } => Some(source),
            Self::GeneratedActionRejected { source, .. } => Some(source),
            _ => None,
        }
    }
}
