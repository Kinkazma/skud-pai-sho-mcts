//! Bounded paired matches against the historical CPU heuristic or a frozen
//! compact parent. Interrupted games remain unrated, never synthetic draws.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::time::{Duration, Instant};

use paisho_ai::{
    CompactValueModel, CpuMctsEvaluator, GameScore, MctsAgent, MctsConfig, MctsEvaluator,
    MctsSession, PairedComparison,
};
use paisho_core::{
    legal_actions, GameOutcome, GameRecord, Player, Position, StandardSetup, BASIC_FLOWERS,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::compact_learning::{invalid, save_json_new, sha256, ModelArtifact};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const LEGACY_REFERENCE: &str =
    "CpuMctsEvaluator with historical HeuristicWeights::default; no learned coefficients";

#[derive(Clone, Debug, PartialEq)]
struct Options {
    candidate: PathBuf,
    reference_model: Option<PathBuf>,
    candidate_reuse: bool,
    reference_reuse: bool,
    output: PathBuf,
    pairs: usize,
    simulations: usize,
    reference_simulations: usize,
    workers: usize,
    seconds: u64,
    decision_limit: usize,
    first_pair: u64,
    move_ms: Option<u64>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let args = if args.first().is_some_and(|arg| arg == "compare") {
            &args[1..]
        } else {
            args
        };
        let mut flags = BTreeMap::new();
        for pair in args.chunks(2) {
            if pair.len() != 2
                || !pair[0].starts_with("--")
                || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err(invalid("expected distinct --name value comparison options"));
            }
        }
        let candidate = PathBuf::from(
            flags
                .remove("--candidate")
                .ok_or_else(|| invalid("missing --candidate"))?,
        );
        let output = PathBuf::from(
            flags
                .remove("--output")
                .ok_or_else(|| invalid("missing --output"))?,
        );
        let reference_model = flags.remove("--reference-model").map(PathBuf::from);
        let candidate_reuse = flags
            .remove("--candidate-reuse")
            .unwrap_or("false")
            .parse::<bool>()?;
        let reference_reuse = flags
            .remove("--reference-reuse")
            .unwrap_or("false")
            .parse::<bool>()?;
        let pairs = bounded(flags.remove("--pairs"), "pairs", 12, 10_000)? as usize;
        let simulations = bounded(flags.remove("--simulations"), "simulations", 32, 8192)? as usize;
        let reference_simulations = bounded(
            flags.remove("--reference-simulations"),
            "reference-simulations",
            simulations as u64,
            8192,
        )? as usize;
        let workers = bounded(flags.remove("--workers"), "workers", 6, 64)? as usize;
        let seconds = bounded(flags.remove("--seconds"), "seconds", 120, 3600)?;
        let decision_limit = bounded(
            flags.remove("--decision-limit"),
            "decision-limit",
            512,
            8192,
        )? as usize;
        let first_pair = flags
            .remove("--first-pair")
            .unwrap_or("100000")
            .parse::<u64>()?;
        first_pair
            .checked_add(pairs as u64 - 1)
            .ok_or_else(|| invalid("pair identifiers overflow"))?;
        let move_ms = flags
            .remove("--move-ms")
            .map(|value| bounded(Some(value), "move-ms", 1, 60_000))
            .transpose()?;
        if !flags.is_empty() {
            return Err(invalid("unknown comparison option"));
        }
        if candidate.as_os_str().is_empty()
            || output.as_os_str().is_empty()
            || reference_model
                .as_ref()
                .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(invalid(
                "candidate, reference and output paths must be nonempty",
            ));
        }
        Ok(Self {
            candidate,
            reference_model,
            candidate_reuse,
            reference_reuse,
            output,
            pairs,
            simulations,
            reference_simulations,
            workers,
            seconds,
            decision_limit,
            first_pair,
            move_ms,
        })
    }

    fn mcts_config(&self) -> MctsConfig {
        MctsConfig {
            simulations: self.simulations,
            independent_trees: 1,
            ..MctsConfig::default()
        }
    }

    fn seat_configs(&self, candidate_host: bool) -> [MctsConfig; 2] {
        let candidate = self.mcts_config();
        let reference = MctsConfig {
            simulations: self.reference_simulations,
            ..candidate
        };
        if candidate_host {
            [candidate, reference]
        } else {
            [reference, candidate]
        }
    }
}

fn bounded(value: Option<&str>, name: &str, default: u64, maximum: u64) -> Result<u64> {
    let parsed = match value {
        Some(value) => value.parse::<u64>()?,
        None => default,
    };
    if parsed == 0 || parsed > maximum {
        return Err(invalid(format!("--{name} must be in 1..={maximum}")));
    }
    Ok(parsed)
}

/// Read and validate once, then publish exactly those bytes. Neither side reads
/// a mutable model file after play begins.
struct FrozenModel {
    path: PathBuf,
    bytes: Vec<u8>,
    artifact: ModelArtifact,
    model: CompactValueModel,
}

impl FrozenModel {
    fn load(path: &Path) -> Result<Self> {
        let path = path.canonicalize()?;
        let bytes = fs::read(&path)?;
        let artifact: ModelArtifact = serde_json::from_slice(&bytes)?;
        let model = artifact.model()?;
        Ok(Self {
            path,
            bytes,
            artifact,
            model,
        })
    }

    fn publish(&self, output: &Path, snapshot: &str) -> Result<Value> {
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join(snapshot))?;
        file.write_all(&self.bytes)?;
        file.sync_all()?;
        Ok(json!({"path":self.path,"sha256":sha256(&self.bytes),
            "snapshot":snapshot,"training_steps":self.artifact.training_steps}))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct EvaluatorIdentities {
    candidate_artifact_sha256: String,
    reference_kind: String,
    reference_artifact_sha256: Option<String>,
    source_sha256: String,
    #[serde(default)]
    candidate_reuse: bool,
    #[serde(default)]
    reference_reuse: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum TerminalOutcome {
    HostWin,
    GuestWin,
    Draw,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Termination {
    Rules { outcome: TerminalOutcome },
    DecisionLimit,
    WallLimit,
    NoLegalActions,
    Error { message: String },
    NotPlayed { reason: String },
}

impl Termination {
    fn from_rules(outcome: GameOutcome) -> Option<Self> {
        let outcome = match outcome {
            GameOutcome::Win(Player::Host) => TerminalOutcome::HostWin,
            GameOutcome::Win(Player::Guest) => TerminalOutcome::GuestWin,
            GameOutcome::Draw => TerminalOutcome::Draw,
            GameOutcome::Ongoing => return None,
        };
        Some(Self::Rules { outcome })
    }

    fn candidate_score(&self, candidate_host: bool) -> Option<GameScore> {
        match self {
            Self::Rules {
                outcome: TerminalOutcome::Draw,
            } => Some(GameScore::Draw),
            Self::Rules { outcome } => {
                let host_won = *outcome == TerminalOutcome::HostWin;
                Some(if host_won == candidate_host {
                    GameScore::Win
                } else {
                    GameScore::Loss
                })
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct SideTelemetry {
    attempted_decisions: usize,
    completed_decisions: usize,
    completed_searches: usize,
    simulations: usize,
    #[serde(default)]
    retained_searches: usize,
    #[serde(default)]
    inherited_root_visits: usize,
    #[serde(default)]
    reused_candidate_positions: usize,
    #[serde(default)]
    reused_leaf_values: usize,
    #[serde(default)]
    maximum_retained_tree_bytes: usize,
    #[serde(default)]
    retained_memory_resets: usize,
    wall_seconds: f64,
    maximum_decision_seconds: f64,
    maximum_move_budget_overshoot_seconds: f64,
}

impl SideTelemetry {
    fn add(&mut self, other: &Self) {
        self.attempted_decisions += other.attempted_decisions;
        self.completed_decisions += other.completed_decisions;
        self.completed_searches += other.completed_searches;
        self.simulations += other.simulations;
        self.retained_searches += other.retained_searches;
        self.inherited_root_visits += other.inherited_root_visits;
        self.reused_candidate_positions += other.reused_candidate_positions;
        self.reused_leaf_values += other.reused_leaf_values;
        self.maximum_retained_tree_bytes = self
            .maximum_retained_tree_bytes
            .max(other.maximum_retained_tree_bytes);
        self.retained_memory_resets += other.retained_memory_resets;
        self.wall_seconds += other.wall_seconds;
        self.maximum_decision_seconds = self
            .maximum_decision_seconds
            .max(other.maximum_decision_seconds);
        self.maximum_move_budget_overshoot_seconds = self
            .maximum_move_budget_overshoot_seconds
            .max(other.maximum_move_budget_overshoot_seconds);
    }

    fn record_elapsed(&mut self, elapsed: Duration, move_ms: Option<u64>) {
        let seconds = elapsed.as_secs_f64();
        self.wall_seconds += seconds;
        self.maximum_decision_seconds = self.maximum_decision_seconds.max(seconds);
        if let Some(move_ms) = move_ms {
            self.maximum_move_budget_overshoot_seconds = self
                .maximum_move_budget_overshoot_seconds
                .max((seconds - move_ms as f64 / 1000.0).max(0.0));
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GameMetadata {
    schema: String,
    game_index: usize,
    pair_id: u64,
    leg: usize,
    candidate_host: bool,
    starting_flower: String,
    host_seed: String,
    guest_seed: String,
    /// Candidate/reference roles stay fixed when Host/Guest seats reverse.
    #[serde(default)]
    evaluators: Option<EvaluatorIdentities>,
    termination: Termination,
    candidate: SideTelemetry,
    reference: SideTelemetry,
    simulation_counts_complete: bool,
    decisions: usize,
    completed_turns: u32,
    wall_seconds: f64,
    maximum_global_deadline_overshoot_seconds: f64,
    record: Option<String>,
    record_sha256: Option<String>,
}

struct GameResult {
    metadata: GameMetadata,
    record: Option<GameRecord>,
}

// A blocked ongoing position is an unknown result, not a failed search or draw.
fn actions_for_search(
    position: &Position,
) -> std::result::Result<Vec<paisho_core::Action>, Termination> {
    let actions = legal_actions(position);
    if actions.is_empty() {
        Err(Termination::NoLegalActions)
    } else {
        Ok(actions)
    }
}

fn seat_seed(pair: u64, seat: Player) -> u64 {
    let salt = if seat == Player::Host {
        0x484f_5354_3230_3236
    } else {
        0x4755_4553_5432_3032
    };
    let mut value = (pair ^ salt).wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn initial_metadata(index: usize, options: &Options) -> GameMetadata {
    let pair_id = options.first_pair + (index / 2) as u64;
    GameMetadata {
        schema: "paisho-compact-comparison-game-v1".into(),
        game_index: index,
        pair_id,
        leg: index % 2,
        candidate_host: index % 2 == 0,
        starting_flower: BASIC_FLOWERS[(pair_id % 6) as usize].code().into(),
        host_seed: format!("0x{:016x}", seat_seed(pair_id, Player::Host)),
        guest_seed: format!("0x{:016x}", seat_seed(pair_id, Player::Guest)),
        evaluators: None,
        termination: Termination::NotPlayed {
            reason: "global soft deadline reached before start".into(),
        },
        candidate: SideTelemetry::default(),
        reference: SideTelemetry::default(),
        simulation_counts_complete: true,
        decisions: 0,
        completed_turns: 0,
        wall_seconds: 0.0,
        maximum_global_deadline_overshoot_seconds: 0.0,
        record: None,
        record_sha256: None,
    }
}

fn play_game(
    index: usize,
    options: &Options,
    model: &CompactValueModel,
    reference: &dyn MctsEvaluator,
    deadline: Instant,
) -> GameResult {
    let started = Instant::now();
    let mut metadata = initial_metadata(index, options);
    if started >= deadline {
        return GameResult {
            metadata,
            record: None,
        };
    }
    let setup = StandardSetup::balanced(BASIC_FLOWERS[(metadata.pair_id % 6) as usize]);
    let mut position = Position::from_standard_setup(setup);
    let mut record = GameRecord::new(setup);
    let [host_config, guest_config] = options.seat_configs(metadata.candidate_host);
    let mut host = MctsAgent::new(seat_seed(metadata.pair_id, Player::Host), host_config)
        .expect("validated MCTS config");
    let mut guest = MctsAgent::new(seat_seed(metadata.pair_id, Player::Guest), guest_config)
        .expect("validated MCTS config");
    let candidate = if metadata.candidate_host {
        Player::Host
    } else {
        Player::Guest
    };
    let host_evaluator: &dyn MctsEvaluator = if metadata.candidate_host {
        model
    } else {
        reference
    };
    let guest_evaluator: &dyn MctsEvaluator = if metadata.candidate_host {
        reference
    } else {
        model
    };
    let [host_reuse, guest_reuse] = if metadata.candidate_host {
        [options.candidate_reuse, options.reference_reuse]
    } else {
        [options.reference_reuse, options.candidate_reuse]
    };
    let mut sessions = [
        host_reuse.then(|| {
            MctsSession::new(
                seat_seed(metadata.pair_id, Player::Host),
                host_config,
                host_evaluator,
            )
            .expect("validated single-tree MCTS config")
        }),
        guest_reuse.then(|| {
            MctsSession::new(
                seat_seed(metadata.pair_id, Player::Guest),
                guest_config,
                guest_evaluator,
            )
            .expect("validated single-tree MCTS config")
        }),
    ];
    loop {
        if let Some(terminal) = Termination::from_rules(position.outcome()) {
            metadata.termination = terminal;
            break;
        }
        if record.actions().len() >= options.decision_limit {
            metadata.termination = Termination::DecisionLimit;
            break;
        }
        if Instant::now() >= deadline {
            metadata.termination = Termination::WallLimit;
            break;
        }
        // The move allowance includes move generation and application, not just search.
        let move_started = Instant::now();
        let move_deadline = options
            .move_ms
            .map(|ms| move_started + Duration::from_millis(ms))
            .map_or(deadline, |limit| limit.min(deadline));
        let seat = position.to_move();
        let side = if seat == candidate {
            &mut metadata.candidate
        } else {
            &mut metadata.reference
        };
        side.attempted_decisions += 1;
        let actions = match actions_for_search(&position) {
            Ok(actions) => actions,
            Err(stop) => {
                side.record_elapsed(move_started.elapsed(), options.move_ms);
                metadata.termination = stop;
                break;
            }
        };
        let evaluator: &dyn MctsEvaluator = if seat == candidate { model } else { reference };
        let agent = if seat == Player::Host {
            &mut host
        } else {
            &mut guest
        };
        let (result, reuse) = if let Some(session) = &mut sessions[seat.index()] {
            let result = session.search_until(&position, &actions, Some(move_deadline));
            (result, Some(session.reuse_statistics()))
        } else {
            (
                agent.search_with_evaluator_until(
                    &position,
                    &actions,
                    evaluator,
                    Some(move_deadline),
                ),
                None,
            )
        };
        let configured_simulations = if seat == Player::Host {
            host_config.simulations
        } else {
            guest_config.simulations
        };
        let result = result.and_then(|report| {
            // In retained mode the action statistics include inherited visits;
            // report.simulations must count only the newly executed simulations.
            if report.simulations == 0 || report.simulations > configured_simulations {
                Err("invalid count of newly executed MCTS simulations".into())
            } else {
                Ok(report)
            }
        });
        let failure = match result {
            Ok(report) => {
                side.completed_searches += 1;
                side.simulations += report.simulations;
                if let Some(reuse) = reuse {
                    side.retained_searches += 1;
                    side.inherited_root_visits += reuse.inherited_root_visits;
                    side.reused_candidate_positions += reuse.reused_candidate_positions;
                    side.reused_leaf_values += reuse.reused_leaf_values;
                    side.maximum_retained_tree_bytes = side
                        .maximum_retained_tree_bytes
                        .max(reuse.tree_bytes_before_limit);
                    side.retained_memory_resets += usize::from(reuse.memory_limit_reset);
                }
                match actions.get(report.selected_index).copied() {
                    Some(action) => match position.apply(action) {
                        Ok(_) => {
                            record.push(action);
                            for session in sessions.iter_mut().flatten() {
                                session.advance(action);
                            }
                            side.completed_decisions += 1;
                            None
                        }
                        Err(error) => Some(format!("applying selected action: {error}")),
                    },
                    None => Some("MCTS returned an invalid action index".into()),
                }
            }
            Err(error) => {
                // The fallible search API cannot report work performed before an error.
                metadata.simulation_counts_complete = false;
                Some(format!("MCTS search: {error}"))
            }
        };
        let move_ended = Instant::now();
        side.record_elapsed(move_ended.duration_since(move_started), options.move_ms);
        metadata.maximum_global_deadline_overshoot_seconds = metadata
            .maximum_global_deadline_overshoot_seconds
            .max(move_ended.saturating_duration_since(deadline).as_secs_f64());
        if let Some(message) = failure {
            metadata.termination = Termination::Error { message };
            break;
        }
    }
    metadata.decisions = record.actions().len();
    metadata.completed_turns = position.completed_turns();
    metadata.wall_seconds = started.elapsed().as_secs_f64();
    GameResult {
        metadata,
        record: Some(record),
    }
}

fn save_game(
    output: &Path,
    mut result: GameResult,
    identities: &EvaluatorIdentities,
) -> Result<GameMetadata> {
    result.metadata.evaluators = Some(identities.clone());
    let stem = format!("game-{:08}", result.metadata.game_index);
    if let Some(record) = result.record {
        let text = record.to_string();
        let relative = format!("records/{stem}.psr");
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join(&relative))?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        result.metadata.record_sha256 = Some(sha256(text.as_bytes()));
        result.metadata.record = Some(relative);
    }
    save_json_new(
        &output.join("games").join(format!("{stem}.json")),
        &result.metadata,
    )?;
    Ok(result.metadata)
}

fn score_half_points(score: GameScore) -> usize {
    match score {
        GameScore::Win => 2,
        GameScore::Draw => 1,
        GameScore::Loss => 0,
    }
}

fn summarize(games: &[GameMetadata]) -> Value {
    let mut comparison = PairedComparison::default();
    let mut paired_wdl = [0_usize; 3];
    let mut known_half_points = 0;
    let mut candidate = SideTelemetry::default();
    let mut reference = SideTelemetry::default();
    let mut stop_counts = BTreeMap::<&str, usize>::new();
    let game_errors: Vec<_> = games
        .iter()
        .filter_map(|game| {
            if let Termination::Error { message } = &game.termination {
                Some(json!({"game_index":game.game_index,"message":message}))
            } else {
                None
            }
        })
        .collect();
    for pair in games.chunks_exact(2) {
        let scores = [
            pair[0].termination.candidate_score(pair[0].candidate_host),
            pair[1].termination.candidate_score(pair[1].candidate_host),
        ];
        comparison.observe(scores[0], scores[1]);
        for score in scores.into_iter().flatten() {
            known_half_points += score_half_points(score);
        }
        if let [Some(first), Some(second)] = scores {
            for score in [first, second] {
                paired_wdl[match score {
                    GameScore::Win => 0,
                    GameScore::Draw => 1,
                    GameScore::Loss => 2,
                }] += 1;
            }
        }
    }
    for game in games {
        candidate.add(&game.candidate);
        reference.add(&game.reference);
        let key = match game.termination {
            Termination::Rules { .. } => "rules",
            Termination::DecisionLimit => "decision_limit",
            Termination::WallLimit => "wall_limit",
            Termination::NoLegalActions => "no_legal_actions",
            Termination::Error { .. } => "error",
            Termination::NotPlayed { .. } => "not_played",
        };
        *stop_counts.entry(key).or_default() += 1;
    }
    let paired_half_points =
        comparison.half + 2 * comparison.one + 3 * comparison.one_and_half + 4 * comparison.two;
    let score = (comparison.rated_pairs() > 0)
        .then(|| paired_half_points as f64 / (4 * comparison.rated_pairs()) as f64);
    json!({
        "schema":"paisho-compact-comparison-summary-v1",
        "rules":paisho_core::RuleProfileId::CURRENT.as_str(),
        "scheduled_games":games.len(), "scheduled_pairs":games.len()/2,
        "game_terminations":stop_counts,
        "game_errors":game_errors,
        "status":if game_errors.is_empty() { "completed" } else { "failed" },
        "paired_comparison":{
            "zero":comparison.zero,"half":comparison.half,"one":comparison.one,
            "one_and_half":comparison.one_and_half,"two":comparison.two,
            "excluded":comparison.excluded,
            "excluded_pessimistic_ties":comparison.excluded_pessimistic_ties,
            "excluded_pessimistic_losses":comparison.excluded_pessimistic_losses,
            "rated_pairs":comparison.rated_pairs(),"score":score,
            "wins":paired_wdl[0],"draws":paired_wdl[1],"losses":paired_wdl[2],
            "exact_two_sided_sign_test_p_value":comparison.exact_two_sided_sign_test_p_value(),
            "pessimistic_exact_two_sided_sign_test_p_value":comparison.pessimistic_exact_two_sided_sign_test_p_value(),
        },
        "pessimistic_score_missing_games_as_candidate_losses": if games.is_empty() { None } else { Some(known_half_points as f64 / (2 * games.len()) as f64) },
        "candidate":candidate,"reference":reference,
        "simulation_counts_complete":games.iter().all(|game|game.simulation_counts_complete),
        "maximum_global_deadline_overshoot_seconds":games.iter().map(|game|game.maximum_global_deadline_overshoot_seconds).fold(0.0,f64::max),
        "interpretation":"Score includes only pairs with two rule-terminal outcomes; score is wins plus half draws, not win rate. Excluded games are not draws. No promotion is performed."
    })
}

/// Run a finite comparison, saving each completed game as soon as it is received.
/// The global deadline is soft: in-flight move generation and one simulation can
/// finish after it, and serialization/joining are included in total run wall time.
mod suite;
pub fn run_suite(args: &[String]) -> Result<()> {
    suite::run(args)
}

pub fn run(args: &[String]) -> Result<()> {
    run_limited(args, None, None)
}

fn run_limited(
    args: &[String],
    slots: Option<&suite::GameSlots>,
    global_deadline: Option<Instant>,
) -> Result<()> {
    let options = Options::parse(args)?;
    options.mcts_config().validate()?;
    let candidate = FrozenModel::load(&options.candidate)?;
    let reference_model = options
        .reference_model
        .as_deref()
        .map(FrozenModel::load)
        .transpose()?;
    let reference: &dyn MctsEvaluator = match &reference_model {
        Some(frozen) => &frozen.model,
        None => &CpuMctsEvaluator,
    };
    let identities = EvaluatorIdentities {
        candidate_artifact_sha256: sha256(&candidate.bytes),
        reference_kind: if reference_model.is_some() {
            "compact-model"
        } else if options.reference_reuse {
            "legacy-cpu-heuristic-retained"
        } else {
            "legacy-cpu-heuristic"
        }
        .into(),
        reference_artifact_sha256: reference_model.as_ref().map(|frozen| sha256(&frozen.bytes)),
        source_sha256: env!("PAISHO_BUILD_SOURCE_SHA256").into(),
        candidate_reuse: options.candidate_reuse,
        reference_reuse: options.reference_reuse,
    };
    let executable = std::env::current_exe()?.canonicalize()?;
    let executable_hash = sha256(&fs::read(&executable)?);
    // create_dir, rather than create_dir_all, rejects an existing experiment.
    if let Some(parent) = options
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&options.output)?;
    fs::create_dir(options.output.join("records"))?;
    fs::create_dir(options.output.join("games"))?;
    let candidate_artifact = candidate.publish(&options.output, "candidate.json")?;
    let reference_artifact = reference_model
        .as_ref()
        .map(|frozen| frozen.publish(&options.output, "reference.json"))
        .transpose()?;
    let config = options.mcts_config();
    let plan = json!({
        "schema":"paisho-compact-comparison-plan-v1",
        "candidate":candidate_artifact,
        "reference":if reference_model.is_some() { "CompactValueModel from frozen reference.json; learned coefficients" } else { LEGACY_REFERENCE },
        "reference_kind":identities.reference_kind,"reference_model":reference_artifact,
        "candidate_reuse":options.candidate_reuse,"reference_reuse":options.reference_reuse,
        "reuse_protocol":"opt-in immutable per-seat sessions; advance both after every actual action, including bonuses; simulation totals count new visits only; retained visits separate; 256MiB retained-tree limit per session",
        "evaluators":identities,
        "executable":{"path":executable,"sha256":executable_hash},
        "build":{"source_sha256":env!("PAISHO_BUILD_SOURCE_SHA256"),
            "git_revision":env!("PAISHO_BUILD_GIT_REVISION"),"git_dirty":env!("PAISHO_BUILD_GIT_DIRTY")},
        "pairs":options.pairs,"first_pair":options.first_pair,"workers":options.workers,
        "seconds":options.seconds,"decision_limit":options.decision_limit,"move_ms":options.move_ms,
        "reference_simulations":options.reference_simulations,
        "mcts":{"simulations":config.simulations,"independent_trees":config.independent_trees,
            "maximum_tree_depth":config.maximum_tree_depth,"action_rank_batch_size":config.action_rank_batch_size,
            "root_widening_factor":config.root_widening_factor,"progressive_widening_factor":config.progressive_widening_factor,
            "rollout_depth":config.rollout_depth,"exploration":config.exploration,
            "heuristic_weights":{"harmony":config.heuristic_weights.harmony,"midline_harmony":config.heuristic_weights.midline_harmony,
                "blooming_flower":config.heuristic_weights.blooming_flower,"total_flower":config.heuristic_weights.total_flower,
                "basic_reserve_progress":config.heuristic_weights.basic_reserve_progress}},
        "pair_protocol":"balanced BASIC_FLOWERS[pair_id%6], candidate Host then Guest; stable SplitMix64 pair-and-seat seeds remain identical across reversal",
        "suite_capacity":slots.map(|s|s.capacity),
        "deadline_scope":if global_deadline.is_some() {"shared-suite"} else {"single-comparison"},
        "execution":"persistent native worker threads, one-thread Rayon pool per worker, bounded completed-game channel; weights immutable",
        "deadline":"soft deadline between moves and simulations; move timer starts before legal action generation; initial ordering and one simulation can overshoot",
        "rules":paisho_core::RuleProfileId::CURRENT.as_str()
    });
    save_json_new(&options.output.join("plan.json"), &plan)?;
    let started = Instant::now();
    let deadline = global_deadline.unwrap_or(started + Duration::from_secs(options.seconds));
    let next_game = AtomicUsize::new(0);
    let cancel = AtomicBool::new(false);
    let total_games = options.pairs * 2;
    let workers = options.workers.min(total_games);
    let (sender, receiver) = sync_channel(workers * 2);
    let mut games = vec![None; total_games];
    let mut worker_errors = Vec::new();
    let mut archive_errors = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            let sender = sender.clone();
            let options = &options;
            let model = &candidate.model;
            let next_game = &next_game;
            let cancel = &cancel;
            handles.push(scope.spawn(move || -> std::result::Result<(), String> {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .map_err(|error| error.to_string())?;
                loop {
                    if cancel.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        break;
                    }
                    let _permit = match slots {
                        Some(slots) => match slots.acquire(deadline) {
                            Some(permit) => Some(permit),
                            None => break,
                        },
                        None => None,
                    };
                    if cancel.load(Ordering::Relaxed) {
                        break;
                    }
                    let index = next_game.fetch_add(1, Ordering::Relaxed);
                    if index >= total_games {
                        break;
                    }
                    let result =
                        pool.install(|| play_game(index, options, model, reference, deadline));
                    if sender.send(result).is_err() {
                        break;
                    }
                }
                Ok(())
            }));
        }
        drop(sender);
        for result in receiver {
            let index = result.metadata.game_index;
            match save_game(&options.output, result, &identities) {
                Ok(metadata) => {
                    games[index] = Some(metadata);
                }
                Err(error) => {
                    cancel.store(true, Ordering::Relaxed);
                    archive_errors.push(format!("saving game {index}: {error}"));
                }
            }
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => worker_errors.push(format!("worker startup: {error}")),
                Err(_) => worker_errors.push("worker panicked".into()),
            }
        }
    });
    for (index, slot) in games.iter_mut().enumerate() {
        if slot.is_none() {
            let mut metadata = initial_metadata(index, &options);
            if !worker_errors.is_empty() || !archive_errors.is_empty() {
                metadata.termination = Termination::NotPlayed {
                    reason:
                        "no saved result after worker or output failure; see summary error fields"
                            .into(),
                };
            }
            match save_game(
                &options.output,
                GameResult {
                    metadata,
                    record: None,
                },
                &identities,
            ) {
                Ok(metadata) => *slot = Some(metadata),
                Err(error) => archive_errors.push(format!("saving missing game {index}: {error}")),
            }
        }
    }
    let available: Vec<_> = games
        .iter()
        .enumerate()
        .map(|(index, game)| {
            game.clone().unwrap_or_else(|| {
                let mut metadata = initial_metadata(index, &options);
                metadata.termination = Termination::NotPlayed {
                    reason: "result persistence failed".into(),
                };
                metadata
            })
        })
        .collect();
    let mut summary = summarize(&available);
    let has_game_errors = available
        .iter()
        .any(|game| matches!(game.termination, Termination::Error { .. }));
    let failed = has_game_errors || !worker_errors.is_empty() || !archive_errors.is_empty();
    summary["run_wall_seconds"] = json!(started.elapsed().as_secs_f64());
    summary["worker_errors"] = json!(worker_errors);
    summary["archive_errors"] = json!(archive_errors);
    summary["status"] = json!(if failed { "failed" } else { "completed" });
    summary["complete_archive"] =
        json!(games.iter().all(Option::is_some) && archive_errors.is_empty());
    save_json_new(&options.output.join("summary.json"), &summary)?;
    println!(
        "{}",
        json!({"output":options.output,"status":summary["status"],"paired_comparison":summary["paired_comparison"],"game_terminations":summary["game_terminations"],"run_wall_seconds":summary["run_wall_seconds"],"complete_archive":summary["complete_archive"]})
    );
    if failed {
        return Err(invalid(
            "comparison encountered game, worker or archive errors; see summary.json",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
