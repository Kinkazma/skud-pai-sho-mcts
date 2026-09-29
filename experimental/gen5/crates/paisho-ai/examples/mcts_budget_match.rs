use std::fmt::Write as _;
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use paisho_ai::{
    play_parallel, AgentTelemetry, GameScore, HeuristicWeights, MatchConfig, MatchResult,
    MatchTask, MatchTermination, MctsAgent, MctsConfig, PairedComparison,
    EXHAUSTIVE_ACTION_RANKING,
};
use paisho_core::{GameOutcome, Player, StandardSetup, BASIC_FLOWERS};
use sha2::{Digest, Sha256};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let games_per_side = argument(&mut arguments, "games per side", 4);
    let low_simulations = argument(&mut arguments, "low simulations", 32);
    let high_simulations = argument(&mut arguments, "high simulations", 128);
    let decision_soft_limit = argument(&mut arguments, "decision soft limit", 512);
    let rollout_depth = nonnegative_argument(&mut arguments, "rollout depth", 0);
    let exploration = float_argument(&mut arguments, "exploration", core::f32::consts::SQRT_2);
    let first_pair_id = u64_argument(&mut arguments, "first pair id", 0);
    let evidence_directory = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() || high_simulations <= low_simulations {
        eprintln!(
            "usage: mcts_budget_match [games-per-side] [low-simulations] [high-simulations] [decision-soft-limit] [rollout-depth] [exploration] [first-pair-id] [evidence-directory]"
        );
        eprintln!("high-simulations must be greater than low-simulations");
        std::process::exit(2);
    }
    let source_revision = evidence_directory.as_ref().map(|_| {
        verified_source_revision().unwrap_or_else(|error| {
            eprintln!("cannot start an evidence run: {error}");
            std::process::exit(2);
        })
    });

    let tasks: Vec<_> = (0..games_per_side as u64)
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
    let low_config = mcts_config(low_simulations, rollout_depth, exploration);
    let high_config = mcts_config(high_simulations, rollout_depth, exploration);

    let started = Instant::now();
    let as_host = play_parallel(
        &tasks,
        match_config,
        |task| MctsAgent::new(host_seed(task.id), high_config).unwrap(),
        |task| MctsAgent::new(guest_seed(task.id), low_config).unwrap(),
    );
    let as_guest = play_parallel(
        &tasks,
        match_config,
        |task| MctsAgent::new(host_seed(task.id), low_config).unwrap(),
        |task| MctsAgent::new(guest_seed(task.id), high_config).unwrap(),
    );
    let elapsed = started.elapsed();
    let workers = as_host.workers.max(as_guest.workers);
    let worker_capacity = as_host.worker_capacity.max(as_guest.worker_capacity);
    let paired = paired_comparison(&as_host.matches, &as_guest.matches);

    let mut score = Score::default();
    collect(&mut score, &as_host.matches, Player::Host);
    collect(&mut score, &as_guest.matches, Player::Guest);
    let games = score.wins + score.draws + score.losses + score.unfinished;

    println!("MCTS high budget versus low budget (paired seats)");
    println!("observed match workers: {workers}/{worker_capacity}");
    println!("low/high simulations per decision: {low_simulations}/{high_simulations}");
    println!(
        "internal action-ranking batch: {}",
        action_batch_label(high_config.action_rank_batch_size)
    );
    println!("rollout depth: {rollout_depth}");
    println!("exploration: {exploration}");
    println!("decision soft limit: {decision_soft_limit}");
    println!(
        "pair ids: {first_pair_id}..={}",
        tasks.last().expect("at least one task").id
    );
    println!(
        "high-budget result: {} win / {} draw / {} loss / {} unfinished / {} error",
        score.wins, score.draws, score.losses, score.unfinished, score.errors
    );
    println!("high-budget as Host: {}", score.as_host);
    println!("high-budget as Guest: {}", score.as_guest);
    println!(
        "paired pentanomial [0, 0.5, 1, 1.5, 2]: [{}, {}, {}, {}, {}] / {} excluded",
        paired.zero, paired.half, paired.one, paired.one_and_half, paired.two, paired.excluded
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
    println!(
        "pessimistic missing-as-loss sign-test p-value: {:.6} ({} tied / {} unfavorable excluded pairs)",
        paired.pessimistic_exact_two_sided_sign_test_p_value(),
        paired.excluded_pessimistic_ties,
        paired.excluded_pessimistic_losses
    );
    println!("decisions: {}", score.decisions);
    print_search_totals("high-budget", score.high);
    print_search_totals("low-budget", score.low);
    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
    println!(
        "throughput: {:.3} games/s, {:.0} simulations/s",
        games as f64 / elapsed.as_secs_f64(),
        (score.high.simulations + score.low.simulations) as f64 / elapsed.as_secs_f64()
    );
    if score.errors > 0 {
        eprintln!("the batch is invalid because at least one match failed");
        std::process::exit(1);
    }
    if let Some(directory) = evidence_directory {
        let run = RunMetadata {
            low_simulations,
            high_simulations,
            action_rank_batch_size: high_config.action_rank_batch_size,
            decision_soft_limit,
            rollout_depth,
            exploration,
            first_pair_id,
            pair_count: games_per_side,
            observed_match_workers: workers,
            match_worker_capacity: worker_capacity,
            elapsed_seconds: elapsed.as_secs_f64(),
            source_revision: source_revision
                .as_deref()
                .expect("an evidence directory has a verified revision"),
        };
        write_evidence(
            &directory,
            run,
            &as_host.matches,
            &as_guest.matches,
            paired,
            &score,
        )
        .unwrap_or_else(|error| {
            eprintln!("cannot write evidence to {}: {error}", directory.display());
            std::process::exit(1);
        });
        println!("evidence: {}", directory.display());
    }
}

fn verified_source_revision() -> io::Result<String> {
    let expected = std::env::var("PAISHO_SOURCE_REVISION").map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "PAISHO_SOURCE_REVISION is required when writing evidence",
        )
    })?;
    let actual = git_output(&["rev-parse", "HEAD"])?;
    if expected != actual {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("requested revision {expected}, but HEAD is {actual}"),
        ));
    }
    let status = git_output(&["status", "--porcelain"])?;
    if !status.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the Git worktree must be clean before an evidence run",
        ));
    }
    Ok(actual)
}

fn git_output(arguments: &[&str]) -> io::Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn paired_comparison(
    as_host: &[Result<MatchResult, paisho_ai::MatchError>],
    as_guest: &[Result<MatchResult, paisho_ai::MatchError>],
) -> PairedComparison {
    assert_eq!(as_host.len(), as_guest.len());
    let mut comparison = PairedComparison::default();
    for (host_game, guest_game) in as_host.iter().zip(as_guest) {
        comparison.observe(
            game_score(host_game, Player::Host),
            game_score(guest_game, Player::Guest),
        );
    }
    comparison
}

fn game_score(
    result: &Result<MatchResult, paisho_ai::MatchError>,
    candidate: Player,
) -> Option<GameScore> {
    let outcome = result.as_ref().ok()?.scored_outcome()?;
    GameScore::from_outcome(outcome, candidate)
}

#[derive(Clone, Copy)]
struct RunMetadata<'a> {
    low_simulations: usize,
    high_simulations: usize,
    action_rank_batch_size: usize,
    decision_soft_limit: usize,
    rollout_depth: usize,
    exploration: f32,
    first_pair_id: u64,
    pair_count: usize,
    observed_match_workers: usize,
    match_worker_capacity: usize,
    elapsed_seconds: f64,
    source_revision: &'a str,
}

fn write_evidence(
    directory: &Path,
    run: RunMetadata<'_>,
    as_host: &[Result<MatchResult, paisho_ai::MatchError>],
    as_guest: &[Result<MatchResult, paisho_ai::MatchError>],
    paired: PairedComparison,
    score: &Score,
) -> io::Result<()> {
    if directory.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "the evidence directory already exists",
        ));
    }
    let file_name = directory.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the evidence directory needs a final path component",
        )
    })?;
    let parent = directory.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let partial = parent.join(format!(".{}.partial", file_name.to_string_lossy()));
    if partial.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("stale partial evidence exists at {}", partial.display()),
        ));
    }
    fs::create_dir(&partial)?;
    fs::create_dir(partial.join("records"))?;

    let weights = HeuristicWeights::default();
    let mut metadata = String::new();
    writeln!(metadata, "PAISHO-MCTS-BUDGET-EVIDENCE\t2").unwrap();
    writeln!(metadata, "source_revision\t{}", run.source_revision).unwrap();
    writeln!(metadata, "rule_profile\tskud-pai-sho-2022-03-14").unwrap();
    writeln!(metadata, "pair_id_first\t{}", run.first_pair_id).unwrap();
    writeln!(metadata, "pair_count\t{}", run.pair_count).unwrap();
    writeln!(metadata, "opening_schedule\tsix-basic-flowers-modulo-id").unwrap();
    writeln!(metadata, "low_simulations\t{}", run.low_simulations).unwrap();
    writeln!(metadata, "high_simulations\t{}", run.high_simulations).unwrap();
    writeln!(metadata, "independent_trees\t1").unwrap();
    writeln!(metadata, "maximum_tree_depth\t96").unwrap();
    writeln!(
        metadata,
        "action_rank_batch_size\t{}",
        action_batch_label(run.action_rank_batch_size)
    )
    .unwrap();
    writeln!(metadata, "rollout_depth\t{}", run.rollout_depth).unwrap();
    writeln!(metadata, "exploration\t{}", run.exploration).unwrap();
    writeln!(metadata, "decision_soft_limit\t{}", run.decision_soft_limit).unwrap();
    writeln!(
        metadata,
        "decision_limit_policy\tcomplete-in-flight-harmony-bonus"
    )
    .unwrap();
    writeln!(
        metadata,
        "observed_match_workers\t{}",
        run.observed_match_workers
    )
    .unwrap();
    writeln!(
        metadata,
        "match_worker_capacity\t{}",
        run.match_worker_capacity
    )
    .unwrap();
    writeln!(
        metadata,
        "high_max_action_ranking_workers\t{}",
        score.high.maximum_action_ranking_workers
    )
    .unwrap();
    writeln!(
        metadata,
        "high_action_ranking_worker_capacity\t{}",
        score.high.maximum_action_ranking_worker_capacity
    )
    .unwrap();
    writeln!(
        metadata,
        "low_max_action_ranking_workers\t{}",
        score.low.maximum_action_ranking_workers
    )
    .unwrap();
    writeln!(
        metadata,
        "low_action_ranking_worker_capacity\t{}",
        score.low.maximum_action_ranking_worker_capacity
    )
    .unwrap();
    writeln!(metadata, "elapsed_seconds\t{:.6}", run.elapsed_seconds).unwrap();
    writeln!(
        metadata,
        "heuristic_weights\t{},{},{},{},{}",
        weights.harmony,
        weights.midline_harmony,
        weights.blooming_flower,
        weights.total_flower,
        weights.basic_reserve_progress
    )
    .unwrap();
    writeln!(
        metadata,
        "game_wdlue\t{}\t{}\t{}\t{}\t{}",
        score.wins, score.draws, score.losses, score.unfinished, score.errors
    )
    .unwrap();
    writeln!(
        metadata,
        "pentanomial\t{}\t{}\t{}\t{}\t{}\t{}",
        paired.zero, paired.half, paired.one, paired.one_and_half, paired.two, paired.excluded
    )
    .unwrap();
    writeln!(
        metadata,
        "paired_sign_test_two_sided_p\t{:.12}",
        paired.exact_two_sided_sign_test_p_value()
    )
    .unwrap();
    writeln!(
        metadata,
        "excluded_pessimistic_ties_losses\t{}\t{}",
        paired.excluded_pessimistic_ties, paired.excluded_pessimistic_losses
    )
    .unwrap();
    writeln!(
        metadata,
        "paired_sign_test_pessimistic_two_sided_p\t{:.12}",
        paired.pessimistic_exact_two_sided_sign_test_p_value()
    )
    .unwrap();
    fs::write(partial.join("run.tsv"), metadata)?;

    let mut games = String::from(
        "pair_id\tstart\tleg\thigh_role\thigh_result\ttermination\tdecisions\thost_seed\tguest_seed\thigh_simulations\tlow_simulations\thigh_evaluated_actions\tlow_evaluated_actions\thigh_max_depth\tlow_max_depth\trecord\terror\n",
    );
    for (host_game, guest_game) in as_host.iter().zip(as_guest) {
        append_game_evidence(&partial, &mut games, host_game, Player::Host)?;
        append_game_evidence(&partial, &mut games, guest_game, Player::Guest)?;
    }
    fs::write(partial.join("games.tsv"), games)?;
    write_sha256_manifest(&partial)?;
    fs::rename(partial, directory)
}

fn write_sha256_manifest(partial: &Path) -> io::Result<()> {
    let mut files = vec![PathBuf::from("games.tsv"), PathBuf::from("run.tsv")];
    for entry in fs::read_dir(partial.join("records"))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(PathBuf::from("records").join(entry.file_name()));
        }
    }
    files.sort();

    let mut manifest = String::new();
    for relative in files {
        let digest = sha256_hex(&partial.join(&relative))?;
        writeln!(manifest, "{digest}  {}", relative.display()).unwrap();
    }
    fs::write(partial.join("MANIFEST.sha256"), manifest)?;
    verify_sha256_manifest(partial)
}

fn sha256_hex(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(digest, "{byte:02x}").unwrap();
    }
    Ok(digest)
}

fn verify_sha256_manifest(directory: &Path) -> io::Result<()> {
    let manifest = fs::read_to_string(directory.join("MANIFEST.sha256"))?;
    for line in manifest.lines() {
        let (expected, relative) = line.split_once("  ").ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "malformed SHA-256 manifest")
        })?;
        let actual = sha256_hex(&directory.join(relative))?;
        if actual != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("SHA-256 mismatch for {relative}"),
            ));
        }
    }
    Ok(())
}

fn append_game_evidence(
    partial: &Path,
    games: &mut String,
    result: &Result<MatchResult, paisho_ai::MatchError>,
    high_role: Player,
) -> io::Result<()> {
    let Ok(result) = result else {
        let error = result.as_ref().unwrap_err().to_string();
        writeln!(
            games,
            "NA\tNA\tNA\t{high_role:?}\tE\tERROR\t0\tNA\tNA\t0\t0\t0\t0\t0\t0\tNA\t{}",
            error.replace(['\t', '\n'], " ")
        )
        .unwrap();
        return Ok(());
    };
    let replayed = result.record.replay().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "record for pair {} does not replay: {error}",
                result.task_id
            ),
        )
    })?;
    if replayed != result.final_position {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "record for pair {} replays to another position",
                result.task_id
            ),
        ));
    }

    let leg = match high_role {
        Player::Host => "high-host",
        Player::Guest => "high-guest",
    };
    let (high, low) = match high_role {
        Player::Host => (result.host_telemetry, result.guest_telemetry),
        Player::Guest => (result.guest_telemetry, result.host_telemetry),
    };
    let record_name = format!("pair-{:020}-{leg}.psr", result.task_id);
    fs::write(
        partial.join("records").join(&record_name),
        result.record.to_string(),
    )?;
    writeln!(
        games,
        "{}\t{}\t{}\t{high_role:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\trecords/{}\t-",
        result.task_id,
        result.record.setup().starting_flower.code(),
        leg,
        candidate_result_code(result, high_role),
        termination_code(result.termination),
        result.record.actions().len(),
        host_seed(result.task_id),
        guest_seed(result.task_id),
        high.simulations,
        low.simulations,
        high.evaluated_actions,
        low.evaluated_actions,
        high.maximum_search_depth,
        low.maximum_search_depth,
        record_name
    )
    .unwrap();
    Ok(())
}

fn candidate_result_code(result: &MatchResult, candidate: Player) -> &'static str {
    match result.scored_outcome() {
        Some(GameOutcome::Win(winner)) if winner == candidate => "W",
        Some(GameOutcome::Win(_)) => "L",
        Some(GameOutcome::Draw) => "D",
        Some(GameOutcome::Ongoing) | None => "U",
    }
}

fn termination_code(termination: MatchTermination) -> &'static str {
    match termination {
        MatchTermination::DecisionLimit => "DECISION_LIMIT",
        MatchTermination::Rules(GameOutcome::Win(Player::Host)) => "HOST_WIN",
        MatchTermination::Rules(GameOutcome::Win(Player::Guest)) => "GUEST_WIN",
        MatchTermination::Rules(GameOutcome::Draw) => "DRAW",
        MatchTermination::Rules(GameOutcome::Ongoing) => "INVALID_ONGOING",
    }
}

fn print_search_totals(label: &str, totals: SearchTotals) {
    println!("{label} decisions: {}", totals.decisions);
    println!("{label} simulations: {}", totals.simulations);
    println!(
        "{label} heuristic action evaluations: {}",
        totals.evaluated_actions
    );
    println!("{label} expanded nodes: {}", totals.expanded_nodes);
    println!("{label} generated nodes: {}", totals.generated_nodes);
    println!("{label} maximum search depth: {}", totals.maximum_depth);
    println!(
        "{label} maximum independent trees: {}",
        totals.maximum_trees
    );
    println!("{label} maximum search workers: {}", totals.maximum_workers);
    println!(
        "{label} maximum search worker capacity: {}",
        totals.maximum_worker_capacity
    );
    println!(
        "{label} maximum action-ranking workers: {}",
        totals.maximum_action_ranking_workers
    );
    println!(
        "{label} maximum action-ranking worker capacity: {}",
        totals.maximum_action_ranking_worker_capacity
    );
    println!(
        "{label} mean generated branching factor: {:.2}",
        totals.mean_branching_factor()
    );
    println!("{label} rollout steps: {}", totals.rollout_steps);
}

fn mcts_config(simulations: usize, rollout_depth: usize, exploration: f32) -> MctsConfig {
    MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 96,
        action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth,
        exploration,
        heuristic_weights: HeuristicWeights::default(),
    }
}

const fn host_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x484f_5354_5f4d_4354
}

const fn guest_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x4755_4553_545f_4d43
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

fn u64_argument(arguments: &mut impl Iterator<Item = String>, name: &str, default: u64) -> u64 {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<u64>().unwrap_or_else(|_| {
        eprintln!("{name} must be a non-negative integer, got `{text}`");
        std::process::exit(2);
    })
}

fn action_batch_label(batch: usize) -> String {
    if batch == EXHAUSTIVE_ACTION_RANKING {
        "all".to_owned()
    } else {
        batch.to_string()
    }
}

#[derive(Default)]
struct Score {
    wins: usize,
    draws: usize,
    losses: usize,
    unfinished: usize,
    errors: usize,
    decisions: usize,
    high: SearchTotals,
    low: SearchTotals,
    as_host: RoleScore,
    as_guest: RoleScore,
}

#[derive(Clone, Copy, Default)]
struct SearchTotals {
    decisions: usize,
    simulations: usize,
    evaluated_actions: usize,
    expanded_nodes: usize,
    generated_nodes: usize,
    generated_actions: usize,
    maximum_depth: usize,
    maximum_trees: usize,
    maximum_workers: usize,
    maximum_worker_capacity: usize,
    maximum_action_ranking_workers: usize,
    maximum_action_ranking_worker_capacity: usize,
    rollout_steps: usize,
}

impl SearchTotals {
    fn add(&mut self, telemetry: AgentTelemetry) {
        self.decisions += telemetry.decisions;
        self.simulations += telemetry.simulations;
        self.evaluated_actions += telemetry.evaluated_actions;
        self.expanded_nodes += telemetry.expanded_nodes;
        self.generated_nodes += telemetry.generated_nodes;
        self.generated_actions += telemetry.generated_actions;
        self.maximum_depth = self.maximum_depth.max(telemetry.maximum_search_depth);
        self.maximum_trees = self.maximum_trees.max(telemetry.maximum_search_trees);
        self.maximum_workers = self.maximum_workers.max(telemetry.maximum_search_workers);
        self.maximum_worker_capacity = self
            .maximum_worker_capacity
            .max(telemetry.maximum_search_worker_capacity);
        self.maximum_action_ranking_workers = self
            .maximum_action_ranking_workers
            .max(telemetry.maximum_action_ranking_workers);
        self.maximum_action_ranking_worker_capacity = self
            .maximum_action_ranking_worker_capacity
            .max(telemetry.maximum_action_ranking_worker_capacity);
        self.rollout_steps += telemetry.rollout_steps;
    }

    fn mean_branching_factor(self) -> f64 {
        if self.generated_nodes == 0 {
            0.0
        } else {
            self.generated_actions as f64 / self.generated_nodes as f64
        }
    }
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
    results: &[Result<MatchResult, paisho_ai::MatchError>],
    candidate: Player,
) {
    for result in results {
        let Ok(result) = result else {
            score.errors += 1;
            continue;
        };
        score.decisions += result.record.actions().len();
        let (high, low) = match candidate {
            Player::Host => (result.host_telemetry, result.guest_telemetry),
            Player::Guest => (result.guest_telemetry, result.host_telemetry),
        };
        score.high.add(high);
        score.low.add(low);
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    #[test]
    fn sealed_manifest_uses_relative_paths_and_verifies_after_rename() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "paisho-manifest-test-{}-{nonce}",
            std::process::id()
        ));
        let partial = root.join("partial");
        let sealed = root.join("sealed");
        fs::create_dir_all(partial.join("records")).unwrap();
        fs::write(partial.join("games.tsv"), "games\n").unwrap();
        fs::write(partial.join("run.tsv"), "run\n").unwrap();
        fs::write(partial.join("records/example.psr"), "record\n").unwrap();

        write_sha256_manifest(&partial).unwrap();
        fs::rename(&partial, &sealed).unwrap();

        let manifest = fs::read_to_string(sealed.join("MANIFEST.sha256")).unwrap();
        let paths: Vec<_> = manifest
            .lines()
            .map(|line| line.split_once("  ").unwrap().1)
            .collect();
        assert_eq!(paths, ["games.tsv", "records/example.psr", "run.tsv"]);
        assert!(!manifest.contains(&root.display().to_string()));

        assert_eq!(
            manifest,
            concat!(
                "f2ace8cdde1e6fa8e118ef1390357ddd985cf1f3dcfbb6bf9f794149fe736f93  games.tsv\n",
                "59772b9c70d6cc244274937445f7c5b56ec6fe0a11292c4ed68848655515a1e6  records/example.psr\n",
                "b5004f26a852b0d60ec1237432c1a33c2307ff2458c374d9d99749d045c7feb9  run.tsv\n",
            )
        );
        verify_sha256_manifest(&sealed).unwrap();
        fs::remove_dir_all(root).unwrap();
    }
}
