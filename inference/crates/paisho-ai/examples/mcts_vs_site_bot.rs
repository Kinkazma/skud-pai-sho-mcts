use std::time::Instant;

use paisho_ai::{
    play_parallel, HeuristicWeights, MatchConfig, MatchResult, MatchTask, MctsAgent, MctsConfig,
    SiteBotV1, EXHAUSTIVE_ACTION_RANKING, SITE_BOT_V1_SOURCE_COMMIT,
};
use paisho_core::{BasicFlower, GameOutcome, Player, StandardSetup};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let games_per_side = argument(&mut arguments, "games per side", 2);
    let simulations = argument(&mut arguments, "simulations", 128);
    let decision_soft_limit = argument(&mut arguments, "decision soft limit", 512);
    let rollout_depth = nonnegative_argument(&mut arguments, "rollout depth", 0);
    let exploration = float_argument(&mut arguments, "exploration", core::f32::consts::SQRT_2);
    if arguments.next().is_some() {
        eprintln!(
            "usage: mcts_vs_site_bot [games-per-side] [simulations-per-decision] [decision-soft-limit] [rollout-depth] [exploration]"
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
        rollout_depth,
        exploration,
        heuristic_weights: HeuristicWeights::default(),
    };

    let started = Instant::now();
    let as_host = play_parallel(
        &tasks,
        match_config,
        |task| MctsAgent::new(mcts_seed(task.id), mcts_config).unwrap(),
        |task| SiteBotV1::new(site_bot_seed(task.id)),
    );
    let as_guest = play_parallel(
        &tasks,
        match_config,
        |task| SiteBotV1::new(site_bot_seed(task.id)),
        |task| MctsAgent::new(mcts_seed(task.id), mcts_config).unwrap(),
    );
    let elapsed = started.elapsed();
    let workers = as_host.workers.max(as_guest.workers);
    let worker_capacity = as_host.worker_capacity.max(as_guest.worker_capacity);

    let mut score = Score::default();
    collect(&mut score, as_host.matches, Player::Host);
    collect(&mut score, as_guest.matches, Player::Guest);
    let games = score.wins + score.draws + score.losses + score.unfinished;
    let games_per_second = games as f64 / elapsed.as_secs_f64();
    let decisions_per_second = score.decisions as f64 / elapsed.as_secs_f64();
    let simulations_per_second = score.simulations as f64 / elapsed.as_secs_f64();

    println!("MCTS heuristic versus SiteBotV1 (paired seats)");
    println!("site source commit: {SITE_BOT_V1_SOURCE_COMMIT}");
    println!("observed match workers: {workers}/{worker_capacity}");
    println!("simulations per MCTS decision: {simulations}");
    println!(
        "internal action-ranking batch: {}",
        action_batch_label(mcts_config.action_rank_batch_size)
    );
    println!(
        "MCTS action-ranking workers: {}/{}",
        score.maximum_action_ranking_workers, score.maximum_action_ranking_worker_capacity
    );
    println!("rollout depth: {rollout_depth}");
    println!("exploration: {exploration}");
    println!("decision soft limit: {decision_soft_limit}");
    println!(
        "MCTS result: {} win / {} draw / {} loss / {} unfinished / {} error",
        score.wins, score.draws, score.losses, score.unfinished, score.errors
    );
    println!("MCTS as Host: {}", score.as_host);
    println!("MCTS as Guest: {}", score.as_guest);
    println!("decisions: {}", score.decisions);
    println!("MCTS simulations: {}", score.simulations);
    println!(
        "SiteBotV1 candidate actions evaluated: {}",
        score.site_evaluated_actions
    );
    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    println!("throughput: {games_per_second:.3} games/s, {decisions_per_second:.1} decisions/s");
    println!("search throughput: {simulations_per_second:.0} simulations/s");
}

const fn mcts_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x4d43_5453_5f41_4931
}

const fn site_bot_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x5349_5445_5f56_315f
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

fn nonnegative_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: usize,
) -> usize {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<usize>().unwrap_or_else(|_| {
        eprintln!("{name} must be a non-negative integer, got `{text}`");
        std::process::exit(2);
    })
}

fn float_argument(arguments: &mut impl Iterator<Item = String>, name: &str, default: f32) -> f32 {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or_else(|| {
            eprintln!("{name} must be a finite non-negative number, got `{text}`");
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
    site_evaluated_actions: usize,
    maximum_action_ranking_workers: usize,
    maximum_action_ranking_worker_capacity: usize,
    as_host: RoleScore,
    as_guest: RoleScore,
}

#[derive(Default)]
struct RoleScore {
    wins: usize,
    draws: usize,
    losses: usize,
    unfinished: usize,
}

impl std::fmt::Display for RoleScore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} win / {} draw / {} loss / {} unfinished",
            self.wins, self.draws, self.losses, self.unfinished
        )
    }
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
        let (candidate_telemetry, reference_telemetry) = match candidate {
            Player::Host => (result.host_telemetry, result.guest_telemetry),
            Player::Guest => (result.guest_telemetry, result.host_telemetry),
        };
        score.simulations += candidate_telemetry.simulations;
        score.maximum_action_ranking_workers = score
            .maximum_action_ranking_workers
            .max(candidate_telemetry.maximum_action_ranking_workers);
        score.maximum_action_ranking_worker_capacity = score
            .maximum_action_ranking_worker_capacity
            .max(candidate_telemetry.maximum_action_ranking_worker_capacity);
        score.site_evaluated_actions += reference_telemetry.evaluated_actions;
        let outcome = match result.scored_outcome() {
            Some(GameOutcome::Win(winner)) if winner == candidate => CandidateOutcome::Win,
            Some(GameOutcome::Win(_)) => CandidateOutcome::Loss,
            Some(GameOutcome::Draw) => CandidateOutcome::Draw,
            Some(GameOutcome::Ongoing) | None => CandidateOutcome::Unfinished,
        };
        match outcome {
            CandidateOutcome::Win => score.wins += 1,
            CandidateOutcome::Draw => score.draws += 1,
            CandidateOutcome::Loss => score.losses += 1,
            CandidateOutcome::Unfinished => score.unfinished += 1,
        }
        let role_score = match candidate {
            Player::Host => &mut score.as_host,
            Player::Guest => &mut score.as_guest,
        };
        match outcome {
            CandidateOutcome::Win => role_score.wins += 1,
            CandidateOutcome::Draw => role_score.draws += 1,
            CandidateOutcome::Loss => role_score.losses += 1,
            CandidateOutcome::Unfinished => role_score.unfinished += 1,
        }
    }
}

#[derive(Clone, Copy)]
enum CandidateOutcome {
    Win,
    Draw,
    Loss,
    Unfinished,
}

fn action_batch_label(batch: usize) -> String {
    if batch == EXHAUSTIVE_ACTION_RANKING {
        "all".to_owned()
    } else {
        batch.to_string()
    }
}
