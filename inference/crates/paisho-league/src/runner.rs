use std::collections::HashSet;
use std::sync::Mutex;
use std::time::Instant;

use paisho_ai::{play_match, MatchConfig, MatchError, MatchResult, MatchTask};
use rayon::prelude::*;

use crate::{AgentDefinition, ScheduledGame};

#[derive(Debug)]
pub struct PlayedGame {
    pub scheduled: ScheduledGame,
    pub result: Result<MatchResult, MatchError>,
}

#[derive(Debug)]
pub struct LeagueExecution {
    pub games: Vec<PlayedGame>,
    pub elapsed_seconds: f64,
    /// Distinct outer match workers observed starting at least one game.
    pub observed_match_workers: usize,
    pub worker_capacity: usize,
    /// Logical CPUs made available to this process by the operating system.
    pub available_parallelism: usize,
}

pub fn run_schedule(
    schedule: &[ScheduledGame],
    agents: &[AgentDefinition],
    match_config: MatchConfig,
) -> LeagueExecution {
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
            let mut host = agents[scheduled.host_agent].make(scheduled.host_seed);
            let mut guest = agents[scheduled.guest_agent].make(scheduled.guest_seed);
            let result = play_match(
                MatchTask {
                    id: scheduled.game_id,
                    setup: scheduled.setup,
                },
                match_config,
                &mut host,
                &mut guest,
            );
            PlayedGame {
                scheduled: *scheduled,
                result,
            }
        })
        .collect();
    LeagueExecution {
        games,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        observed_match_workers: observed_workers
            .into_inner()
            .expect("worker observation lock is not poisoned")
            .len(),
        worker_capacity: rayon::current_num_threads(),
        available_parallelism: std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1),
    }
}
