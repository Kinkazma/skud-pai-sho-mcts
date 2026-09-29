use std::time::Instant;

use paisho_ai::{
    play_parallel, AgentTelemetry, GameScore, HeuristicWeights, MatchConfig, MatchResult,
    MatchTask, MctsAgent, MctsConfig, PairedComparison, EXHAUSTIVE_ACTION_RANKING,
};
use paisho_core::{GameOutcome, Player, StandardSetup, BASIC_FLOWERS};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let pair_count = positive_argument(&mut arguments, "pair count", 12);
    let simulations = positive_argument(&mut arguments, "simulations", 128);
    let candidate_batch = batch_argument(&mut arguments, "candidate batch", 64);
    let baseline_batch =
        batch_argument(&mut arguments, "baseline batch", EXHAUSTIVE_ACTION_RANKING);
    let decision_soft_limit = positive_argument(&mut arguments, "decision soft limit", 2_048);
    let first_pair_id = u64_argument(&mut arguments, "first pair id", 5_000);
    let candidate_root = positive_float_argument(&mut arguments, "candidate root factor", 1.0);
    let candidate_internal =
        positive_float_argument(&mut arguments, "candidate internal factor", 1.0);
    let baseline_root = positive_float_argument(&mut arguments, "baseline root factor", 1.0);
    let baseline_internal =
        positive_float_argument(&mut arguments, "baseline internal factor", 1.0);
    if arguments.next().is_some() {
        eprintln!(
            "usage: mcts_action_batch_match [pairs] [simulations] [candidate-batch] [baseline-batch|all] [decision-soft-limit] [first-pair-id] [candidate-root] [candidate-internal] [baseline-root] [baseline-internal]"
        );
        std::process::exit(2);
    }

    let tasks: Vec<_> = (0..pair_count as u64)
        .map(|offset| {
            let id = first_pair_id
                .checked_add(offset)
                .expect("pair id range must fit in u64");
            MatchTask {
                id,
                setup: StandardSetup::balanced(
                    BASIC_FLOWERS[(id % BASIC_FLOWERS.len() as u64) as usize],
                ),
            }
        })
        .collect();
    let match_config = MatchConfig {
        decision_soft_limit,
    };
    let candidate_config = config(
        simulations,
        candidate_batch,
        candidate_root,
        candidate_internal,
    );
    let baseline_config = config(
        simulations,
        baseline_batch,
        baseline_root,
        baseline_internal,
    );

    let started = Instant::now();
    let as_host = play_parallel(
        &tasks,
        match_config,
        |task| MctsAgent::new(host_seed(task.id), candidate_config).unwrap(),
        |task| MctsAgent::new(guest_seed(task.id), baseline_config).unwrap(),
    );
    let as_guest = play_parallel(
        &tasks,
        match_config,
        |task| MctsAgent::new(host_seed(task.id), baseline_config).unwrap(),
        |task| MctsAgent::new(guest_seed(task.id), candidate_config).unwrap(),
    );
    let elapsed = started.elapsed();
    let paired = paired_comparison(&as_host.matches, &as_guest.matches);
    let mut score = Score::default();
    collect(&mut score, &as_host.matches, Player::Host);
    collect(&mut score, &as_guest.matches, Player::Guest);

    println!("MCTS search-policy comparison (paired seats)");
    println!(
        "observed match workers: {}/{}",
        as_host.workers.max(as_guest.workers),
        as_host.worker_capacity.max(as_guest.worker_capacity)
    );
    println!("simulations per decision: {simulations}");
    println!(
        "candidate/baseline batch: {}/{}",
        batch_label(candidate_batch),
        batch_label(baseline_batch)
    );
    println!("candidate root/internal widening: {candidate_root}/{candidate_internal}");
    println!("baseline root/internal widening: {baseline_root}/{baseline_internal}");
    println!("decision soft limit: {decision_soft_limit}");
    println!(
        "candidate result: {} win / {} draw / {} loss / {} unfinished / {} error",
        score.wins, score.draws, score.losses, score.unfinished, score.errors
    );
    println!(
        "paired favorable/tied/unfavorable: {}/{}/{}",
        paired.favorable(),
        paired.tied(),
        paired.unfavorable()
    );
    println!(
        "exact two-sided paired sign-test p-value: {:.6}",
        paired.exact_two_sided_sign_test_p_value()
    );
    print_totals("candidate", score.candidate);
    print_totals("baseline", score.baseline);
    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    if score.errors > 0 {
        std::process::exit(1);
    }
}

fn config(
    simulations: usize,
    action_rank_batch_size: usize,
    root_widening_factor: f32,
    progressive_widening_factor: f32,
) -> MctsConfig {
    MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 96,
        action_rank_batch_size,
        root_widening_factor,
        progressive_widening_factor,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    }
}

fn paired_comparison(
    as_host: &[Result<MatchResult, paisho_ai::MatchError>],
    as_guest: &[Result<MatchResult, paisho_ai::MatchError>],
) -> PairedComparison {
    assert_eq!(as_host.len(), as_guest.len());
    let mut comparison = PairedComparison::default();
    for (host, guest) in as_host.iter().zip(as_guest) {
        comparison.observe(
            game_score(host, Player::Host),
            game_score(guest, Player::Guest),
        );
    }
    comparison
}

fn game_score(
    result: &Result<MatchResult, paisho_ai::MatchError>,
    candidate: Player,
) -> Option<GameScore> {
    GameScore::from_outcome(result.as_ref().ok()?.scored_outcome()?, candidate)
}

fn collect(
    score: &mut Score,
    results: &[Result<MatchResult, paisho_ai::MatchError>],
    candidate: Player,
) {
    for result in results {
        let Ok(result) = result else {
            score.errors += 1;
            continue;
        };
        let (candidate_telemetry, baseline_telemetry) = match candidate {
            Player::Host => (result.host_telemetry, result.guest_telemetry),
            Player::Guest => (result.guest_telemetry, result.host_telemetry),
        };
        score.candidate.add(candidate_telemetry);
        score.baseline.add(baseline_telemetry);
        match result.scored_outcome() {
            Some(GameOutcome::Win(winner)) if winner == candidate => score.wins += 1,
            Some(GameOutcome::Win(_)) => score.losses += 1,
            Some(GameOutcome::Draw) => score.draws += 1,
            Some(GameOutcome::Ongoing) | None => score.unfinished += 1,
        }
    }
}

fn print_totals(label: &str, totals: SearchTotals) {
    let per_simulation = if totals.simulations == 0 {
        0.0
    } else {
        totals.evaluated_actions as f64 / totals.simulations as f64
    };
    println!("{label} decisions: {}", totals.decisions);
    println!("{label} simulations: {}", totals.simulations);
    println!(
        "{label} heuristic action evaluations: {} ({per_simulation:.2}/simulation)",
        totals.evaluated_actions
    );
    println!(
        "{label} maximum action-ranking workers: {}/{}",
        totals.maximum_action_ranking_workers, totals.maximum_action_ranking_worker_capacity
    );
}

#[derive(Default)]
struct Score {
    wins: usize,
    draws: usize,
    losses: usize,
    unfinished: usize,
    errors: usize,
    candidate: SearchTotals,
    baseline: SearchTotals,
}

#[derive(Clone, Copy, Default)]
struct SearchTotals {
    decisions: usize,
    simulations: usize,
    evaluated_actions: usize,
    maximum_action_ranking_workers: usize,
    maximum_action_ranking_worker_capacity: usize,
}

impl SearchTotals {
    fn add(&mut self, telemetry: AgentTelemetry) {
        self.decisions += telemetry.decisions;
        self.simulations += telemetry.simulations;
        self.evaluated_actions += telemetry.evaluated_actions;
        self.maximum_action_ranking_workers = self
            .maximum_action_ranking_workers
            .max(telemetry.maximum_action_ranking_workers);
        self.maximum_action_ranking_worker_capacity = self
            .maximum_action_ranking_worker_capacity
            .max(telemetry.maximum_action_ranking_worker_capacity);
    }
}

const fn host_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x484f_5354_5f42_4154
}

const fn guest_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x4755_4553_545f_4241
}

fn positive_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: usize,
) -> usize {
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

fn batch_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: usize,
) -> usize {
    let Some(text) = arguments.next() else {
        return default;
    };
    if text == "all" {
        EXHAUSTIVE_ACTION_RANKING
    } else {
        text.parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .unwrap_or_else(|| {
                eprintln!("{name} must be `all` or a positive integer, got `{text}`");
                std::process::exit(2);
            })
    }
}

fn u64_argument(arguments: &mut impl Iterator<Item = String>, name: &str, default: u64) -> u64 {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<u64>().unwrap_or_else(|_| {
        eprintln!("{name} must be a non-negative integer, got `{text}`");
        std::process::exit(2);
    })
}

fn positive_float_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: f32,
) -> f32 {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<f32>()
        .ok()
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or_else(|| {
            eprintln!("{name} must be a positive finite number, got `{text}`");
            std::process::exit(2);
        })
}

fn batch_label(batch: usize) -> String {
    if batch == EXHAUSTIVE_ACTION_RANKING {
        "all".to_owned()
    } else {
        batch.to_string()
    }
}
