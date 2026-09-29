use std::time::Instant;

use paisho_ai::{
    play_parallel, HeuristicWeights, MatchConfig, MatchResult, MatchTask, MctsAgent, MctsConfig,
    RandomAgent, EXHAUSTIVE_ACTION_RANKING,
};
use paisho_core::{BasicFlower, GameOutcome, Player, StandardSetup};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let games_per_side = argument(&mut arguments, "games per side", 4);
    let simulations = argument(&mut arguments, "simulations", 256);
    let decision_soft_limit = argument(&mut arguments, "decision soft limit", 256);
    if arguments.next().is_some() {
        eprintln!(
            "usage: mcts_vs_random [games-per-side] [simulations-per-decision] [decision-soft-limit]"
        );
        std::process::exit(2);
    }

    let tasks: Vec<_> = (0..games_per_side as u64)
        .map(|id| MatchTask {
            id,
            setup: StandardSetup::balanced(BasicFlower::Red3),
        })
        .collect();
    let match_config = MatchConfig {
        decision_soft_limit,
    };
    let mcts_config = MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 96,
        action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    };

    let started = Instant::now();
    let as_host = play_parallel(
        &tasks,
        match_config,
        |task| MctsAgent::new(mcts_seed(task.id), mcts_config).unwrap(),
        |task| RandomAgent::new(random_seed(task.id)),
    );
    let as_guest = play_parallel(
        &tasks,
        match_config,
        |task| RandomAgent::new(random_seed(task.id)),
        |task| MctsAgent::new(mcts_seed(task.id), mcts_config).unwrap(),
    );
    let elapsed = started.elapsed();

    let mut score = Score::default();
    collect(&mut score, as_host.matches, Player::Host);
    collect(&mut score, as_guest.matches, Player::Guest);
    let games = score.wins + score.draws + score.losses + score.unfinished;
    let games_per_second = games as f64 / elapsed.as_secs_f64();
    let decisions_per_second = score.decisions as f64 / elapsed.as_secs_f64();
    let simulations_per_second = score.simulations as f64 / elapsed.as_secs_f64();

    println!("MCTS heuristic versus random (paired seats)");
    println!(
        "observed match workers: {}/{}",
        as_host.workers.max(as_guest.workers),
        as_host.worker_capacity.max(as_guest.worker_capacity)
    );
    println!("simulations per MCTS decision: {simulations}");
    println!(
        "internal action-ranking batch: {}",
        action_batch_label(mcts_config.action_rank_batch_size)
    );
    println!(
        "MCTS action-ranking workers: {}/{}",
        score.maximum_action_ranking_workers, score.maximum_action_ranking_worker_capacity
    );
    println!("rollout depth: 0");
    println!("decision soft limit: {decision_soft_limit}");
    println!(
        "MCTS result: {} win / {} draw / {} loss / {} unfinished / {} error",
        score.wins, score.draws, score.losses, score.unfinished, score.errors
    );
    println!("decisions: {}", score.decisions);
    println!("MCTS simulations: {}", score.simulations);
    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    println!("throughput: {games_per_second:.3} games/s, {decisions_per_second:.1} decisions/s");
    println!("search throughput: {simulations_per_second:.0} simulations/s");
}

const fn mcts_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x4d43_5453_5f41_4931
}

const fn random_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x5241_4e44_4f4d_5f31
}

fn argument(arguments: &mut impl Iterator<Item = String>, name: &str, default: usize) -> usize {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            eprintln!("{name} must be a positive integer, got `{text}`");
            std::process::exit(2);
        })
}

#[derive(Default)]
struct Score {
    wins: usize,
    draws: usize,
    losses: usize,
    unfinished: usize,
    errors: usize,
    decisions: usize,
    simulations: usize,
    maximum_action_ranking_workers: usize,
    maximum_action_ranking_worker_capacity: usize,
}

fn collect(
    score: &mut Score,
    results: Vec<Result<MatchResult, paisho_ai::MatchError>>,
    candidate: Player,
) {
    for result in results {
        let Ok(result) = result else {
            score.errors += 1;
            continue;
        };
        score.decisions += result.record.actions().len();
        let telemetry = match candidate {
            Player::Host => result.host_telemetry,
            Player::Guest => result.guest_telemetry,
        };
        score.simulations += telemetry.simulations;
        score.maximum_action_ranking_workers = score
            .maximum_action_ranking_workers
            .max(telemetry.maximum_action_ranking_workers);
        score.maximum_action_ranking_worker_capacity = score
            .maximum_action_ranking_worker_capacity
            .max(telemetry.maximum_action_ranking_worker_capacity);
        match result.scored_outcome() {
            Some(GameOutcome::Win(winner)) if winner == candidate => score.wins += 1,
            Some(GameOutcome::Win(_)) => score.losses += 1,
            Some(GameOutcome::Draw) => score.draws += 1,
            Some(GameOutcome::Ongoing) | None => score.unfinished += 1,
        }
    }
}

fn action_batch_label(batch: usize) -> String {
    if batch == EXHAUSTIVE_ACTION_RANKING {
        "all".to_owned()
    } else {
        batch.to_string()
    }
}
