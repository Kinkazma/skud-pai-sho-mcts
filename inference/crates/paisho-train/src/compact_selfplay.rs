//! Bounded CPU actor/learner pilot for the compact value model. Every actor uses
//! one immutable weight snapshot for an entire game. Rules outcomes and explicit
//! training repetition losses are archived separately; all PSRs remain truthful.

use std::collections::BTreeMap;
use std::error::Error;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use paisho_ai::{
    CompactValueFeatures, CompactValueModel, CpuMctsEvaluator, MctsAgent, MctsConfig,
    MctsEvaluator, MctsSession, StableRng, COMPACT_FEATURE_COUNT, COMPACT_VALUE_SCHEMA_V1,
};
use paisho_core::{
    legal_actions, GameOutcome, GameRecord, Player, Position, StandardSetup, BASIC_FLOWERS,
};
use serde::{Deserialize, Serialize};

use crate::compact_learning::{
    invalid, load_model, save_json_new, save_model_new, sha256, ModelArtifact,
};

mod diagnostic;
pub fn analyze_repetitions(args: &[String]) -> Result<()> {
    diagnostic::run(args)
}

pub(crate) mod reuse;
use reuse::{training_example, RepetitionLoss, Repetitions, ReplayMemory};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
type SharedModel = Arc<RwLock<(u64, Arc<CompactValueModel>)>>;

/// Frozen collection profile, COMPACT_MCTS_V5. No automatic retuning.
/// These limits never apply to strength comparisons.
fn fixed_game_seconds(simulations: usize) -> Option<f64> {
    match simulations {
        32 => Some(8.0),
        64 => Some(17.5),
        128 => Some(25.0),
        256 => Some(34.0),
        512 => Some(58.0),
        _ => None,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Options {
    pub games: usize,
    pub seconds: f64,
    #[serde(default)]
    pub game_seconds: Option<f64>,
    #[serde(default = "learning_enabled_by_default")]
    pub learn: bool,
    #[serde(default)]
    pub replay_capacity: usize,
    #[serde(default)]
    pub replay_input: Option<PathBuf>,
    #[serde(default)]
    pub replay_ratio: usize,
    #[serde(default)]
    pub repetition_cycles: usize,
    #[serde(default)]
    pub reuse_search: bool,
    pub workers: usize,
    pub simulations: usize,
    /// Empty preserves the single-budget profile; otherwise workers cycle through these budgets.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub budgets: Vec<usize>,
    pub decision_limit: usize,
    pub samples: usize,
    pub learning_rate: f64,
    pub lambda: f64,
    pub seed: u64,
    pub model: PathBuf,
    pub output: PathBuf,
}

impl Options {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut flags = BTreeMap::new();
        for pair in args.chunks(2) {
            if pair.len() != 2
                || !pair[0].starts_with("--")
                || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
            {
                return Err(invalid("expected distinct --name value options"));
            }
        }
        let budget_flag = flags.remove("--budgets");
        if budget_flag.is_some() && flags.contains_key("--simulations") {
            return Err(invalid(
                "--budgets and --simulations are mutually exclusive",
            ));
        }
        let budgets = budget_flag
            .map(|value| {
                value
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<Vec<usize>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        let simulations = flags.remove("--simulations").unwrap_or("8").parse()?;
        let game_seconds = flags
            .remove("--game-seconds")
            .map(str::parse)
            .transpose()?
            .or_else(|| {
                budgets
                    .is_empty()
                    .then(|| fixed_game_seconds(simulations))
                    .flatten()
            });
        let result = Self {
            games: flags.remove("--games").unwrap_or("128").parse()?,
            seconds: flags.remove("--seconds").unwrap_or("60").parse()?,
            game_seconds,
            learn: flags.remove("--learn").unwrap_or("true").parse()?,
            replay_input: flags.remove("--replay-input").map(PathBuf::from),
            replay_capacity: flags
                .remove("--replay-capacity")
                .unwrap_or("65536")
                .parse()?,
            replay_ratio: flags.remove("--replay-ratio").unwrap_or("4").parse()?,
            repetition_cycles: flags.remove("--repetition-cycles").unwrap_or("4").parse()?,
            reuse_search: flags.remove("--reuse-search").unwrap_or("true").parse()?,
            workers: flags.remove("--workers").unwrap_or("6").parse()?,
            simulations,
            budgets,
            decision_limit: flags.remove("--decision-limit").unwrap_or("512").parse()?,
            samples: flags.remove("--samples").unwrap_or("32").parse()?,
            learning_rate: flags.remove("--learning-rate").unwrap_or("0.01").parse()?,
            lambda: flags.remove("--lambda").unwrap_or("0.5").parse()?,
            seed: flags.remove("--seed").unwrap_or("1").parse()?,
            model: PathBuf::from(
                flags
                    .remove("--model")
                    .ok_or_else(|| invalid("missing --model"))?,
            ),
            output: PathBuf::from(
                flags
                    .remove("--output")
                    .ok_or_else(|| invalid("missing --output"))?,
            ),
        };
        if !flags.is_empty() {
            return Err(invalid("unknown compact self-play option"));
        }
        result.validate()?;
        Ok(result)
    }

    fn worker_options(&self, worker: usize) -> Self {
        let mut options = self.clone();
        if !self.budgets.is_empty() {
            options.simulations = self.budgets[worker % self.budgets.len()];
            options.game_seconds = self
                .game_seconds
                .or_else(|| fixed_game_seconds(options.simulations));
        }
        options
    }

    fn validate(&self) -> Result<()> {
        if !self.budgets.is_empty()
            && (self.budgets.iter().any(|&budget| budget == 0)
                || self
                    .budgets
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != self.budgets.len()
                || self.workers.min(self.games) < self.budgets.len())
        {
            return Err(invalid("--budgets requires distinct positive budgets and at least one worker/game per budget"));
        }
        if self.repetition_cycles == 1 || self.repetition_cycles > 32 {
            return Err(invalid("repetition cycles must be 0 (disabled) or 2..32"));
        }
        if self.games == 0
            || self.workers == 0
            || self.simulations == 0
            || self.decision_limit == 0
            || self.samples == 0
            || !self.seconds.is_finite()
            || self.seconds <= 0.0
            || self
                .game_seconds
                .is_some_and(|seconds| !seconds.is_finite() || seconds <= 0.0)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.lambda.is_finite()
            || !(0.0..=1.0).contains(&self.lambda)
        {
            return Err(invalid(
                "invalid compact self-play limits or learning options",
            ));
        }
        Duration::try_from_secs_f64(self.seconds).map_err(|error| invalid(error.to_string()))?;
        if let Some(seconds) = self.game_seconds {
            Duration::try_from_secs_f64(seconds).map_err(|error| invalid(error.to_string()))?;
        }
        Ok(())
    }
}

fn learning_enabled_by_default() -> bool {
    true
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Seat {
    Host,
    Guest,
}

impl Seat {
    fn from_player(player: Player) -> Self {
        match player {
            Player::Host => Self::Host,
            Player::Guest => Self::Guest,
        }
    }
    pub(crate) fn player(self) -> Player {
        match self {
            Self::Host => Player::Host,
            Self::Guest => Player::Guest,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RootSample {
    decision: usize,
    perspective: Seat,
    features: Vec<f64>,
    q: f64,
    selected_visits: usize,
    search_simulations: usize,
}

impl RootSample {
    fn features(&self) -> Result<CompactValueFeatures> {
        let values: [f64; COMPACT_FEATURE_COUNT] = self
            .features
            .clone()
            .try_into()
            .map_err(|_| invalid("root sample has an invalid feature count"))?;
        Ok(CompactValueFeatures::from_values(values, None)?)
    }
}

fn terminal_value(outcome: GameOutcome, perspective: Player) -> Option<f64> {
    match outcome {
        GameOutcome::Ongoing => None,
        GameOutcome::Draw => Some(0.0),
        GameOutcome::Win(winner) => Some(if winner == perspective { 1.0 } else { -1.0 }),
    }
}

fn mixed_target(sample: &RootSample, outcome: GameOutcome, lambda: f64) -> Option<f64> {
    terminal_value(outcome, sample.perspective.player())
        .map(|z| (1.0 - lambda) * sample.q + lambda * z)
}

fn reservoir_sample(
    samples: &mut Vec<RootSample>,
    sample: RootSample,
    seen: usize,
    limit: usize,
    rng: &mut StableRng,
) {
    if samples.len() < limit {
        samples.push(sample);
    } else {
        let index = rng.index(seen);
        if index < limit {
            samples[index] = sample;
        }
    }
}

fn weights_hash(model: &CompactValueModel) -> String {
    let mut bytes = COMPACT_VALUE_SCHEMA_V1.as_bytes().to_vec();
    bytes.push(0);
    for weight in model.weights() {
        bytes.extend(weight.to_le_bytes());
    }
    sha256(&bytes)
}

fn outcome_name(outcome: GameOutcome) -> &'static str {
    match outcome {
        GameOutcome::Ongoing => "ongoing",
        GameOutcome::Draw => "draw",
        GameOutcome::Win(Player::Host) => "host",
        GameOutcome::Win(Player::Guest) => "guest",
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct GameTiming {
    legal_actions_seconds: f64,
    search_seconds: f64,
    features_seconds: f64,
    apply_seconds: f64,
}

impl GameTiming {
    fn add(&mut self, other: Self) {
        self.legal_actions_seconds += other.legal_actions_seconds;
        self.search_seconds += other.search_seconds;
        self.features_seconds += other.features_seconds;
        self.apply_seconds += other.apply_seconds;
    }
}

#[derive(Debug)]
struct PlayedGame {
    id: usize,
    worker: usize,
    version: u64,
    simulations: usize,
    requested_game_seconds: Option<f64>,
    schedule_id: usize,
    weights_sha256: String,
    seeds: [u64; 3],
    self_play: bool,
    learner_seat: Option<Seat>,
    record: GameRecord,
    outcome: GameOutcome,
    repetition_loss: Option<RepetitionLoss>,
    completed_turns: u32,
    termination: &'static str,
    error: Option<String>,
    samples: Vec<RootSample>,
    eligible_roots: usize,
    actual_simulations: Vec<usize>,
    reused_root_visits: usize,
    reused_candidate_positions: usize,
    reused_leaf_values: usize,
    elapsed_seconds: f64,
    timing: GameTiming,
    effective_deadline_seconds_from_game_start: f64,
    game_deadline_overshoot_seconds: Option<f64>,
    campaign_deadline_overshoot_seconds: f64,
}

/// Global ID keeps seeds/archive names unique; schedule ID rotates opponents and
/// starting flowers independently per actor in a heterogeneous run.
#[derive(Clone, Copy)]
struct GameIdentity {
    id: usize,
    worker: usize,
    schedule_id: usize,
}

fn play_game(
    options: &Options,
    identity: GameIdentity,
    version: u64,
    model: &CompactValueModel,
    campaign_deadline: Instant,
    cancelled: &AtomicBool,
) -> PlayedGame {
    let GameIdentity {
        id,
        worker,
        schedule_id,
    } = identity;
    let started = Instant::now();
    let game_deadline = options.game_seconds.map(|seconds| {
        // A representable duration can exceed Instant's range. Such a game
        // allowance cannot bind before the already validated campaign deadline.
        started
            .checked_add(Duration::from_secs_f64(seconds))
            .unwrap_or(campaign_deadline)
    });
    let deadline = game_deadline.map_or(campaign_deadline, |limit| limit.min(campaign_deadline));
    let deadline_reason = if game_deadline.is_some_and(|limit| limit < campaign_deadline) {
        "game-deadline"
    } else {
        "campaign-deadline"
    };
    let mut seed_rng = StableRng::new(options.seed ^ (id as u64).wrapping_mul(0x9e3779b97f4a7c15));
    let seeds = [
        seed_rng.next_u64(),
        seed_rng.next_u64(),
        seed_rng.next_u64(),
    ];
    let self_play = schedule_id % 2 == 0;
    let learner_seat = if self_play {
        None
    } else if (schedule_id / 2) % 2 == 0 {
        Some(Seat::Host)
    } else {
        Some(Seat::Guest)
    };
    let setup = StandardSetup::balanced(BASIC_FLOWERS[(schedule_id / 2) % BASIC_FLOWERS.len()]);
    let mut position = Position::from_standard_setup(setup);
    let config = MctsConfig {
        simulations: options.simulations,
        ..MctsConfig::default()
    };
    let mut actors = [
        MctsAgent::new(seeds[0], config).expect("validated config"),
        MctsAgent::new(seeds[1], config).expect("validated config"),
    ];
    let legacy = CpuMctsEvaluator;
    let evaluators: [&dyn MctsEvaluator; 2] = [Player::Host, Player::Guest].map(|seat| {
        if self_play || learner_seat == Some(Seat::from_player(seat)) {
            model as &dyn MctsEvaluator
        } else {
            &legacy as &dyn MctsEvaluator
        }
    });
    let mut sessions = [
        MctsSession::new(seeds[0], config, evaluators[0]).expect("validated session"),
        MctsSession::new(seeds[1], config, evaluators[1]).expect("validated session"),
    ];
    let mut sampling_rng = StableRng::new(seeds[2]);
    let mut result = PlayedGame {
        id,
        worker,
        version,
        simulations: options.simulations,
        requested_game_seconds: options.game_seconds,
        schedule_id,
        weights_sha256: weights_hash(model),
        seeds,
        self_play,
        learner_seat,
        record: GameRecord::new(setup),
        outcome: GameOutcome::Ongoing,
        repetition_loss: None,
        completed_turns: 0,
        termination: "terminal",
        error: None,
        samples: Vec::new(),
        eligible_roots: 0,
        actual_simulations: Vec::new(),
        reused_root_visits: 0,
        reused_candidate_positions: 0,
        reused_leaf_values: 0,
        elapsed_seconds: 0.0,
        timing: GameTiming::default(),
        effective_deadline_seconds_from_game_start: deadline
            .saturating_duration_since(started)
            .as_secs_f64(),
        game_deadline_overshoot_seconds: None,
        campaign_deadline_overshoot_seconds: 0.0,
    };
    let mut repetitions = Repetitions::new(&position, options.repetition_cycles);
    let mut last_learner_sample = None;
    while position.outcome() == GameOutcome::Ongoing {
        if cancelled.load(Ordering::Relaxed) {
            result.termination = "cancelled";
            break;
        }
        if Instant::now() >= deadline {
            result.termination = deadline_reason;
            break;
        }
        if result.record.actions().len() >= options.decision_limit {
            result.termination = "decision-limit";
            break;
        }
        let perspective = position.to_move();
        let learner_turn = self_play || learner_seat == Some(Seat::from_player(perspective));
        let legal_started = Instant::now();
        let actions = legal_actions(&position);
        result.timing.legal_actions_seconds += legal_started.elapsed().as_secs_f64();
        if actions.is_empty() {
            // No rules-terminal result: retain the PSR but no Q/result target.
            result.termination = "no-legal-actions";
            break;
        }
        let evaluator: &dyn MctsEvaluator = if learner_turn {
            model
        } else {
            &CpuMctsEvaluator
        };
        let search_started = Instant::now();
        let searched = if options.reuse_search {
            sessions[perspective.index()].search_until(&position, &actions, Some(deadline))
        } else {
            actors[perspective.index()].search_with_evaluator_until(
                &position,
                &actions,
                evaluator,
                Some(deadline),
            )
        };
        if options.reuse_search {
            let reuse = sessions[perspective.index()].reuse_statistics();
            result.reused_root_visits += reuse.inherited_root_visits;
            result.reused_candidate_positions += reuse.reused_candidate_positions;
            result.reused_leaf_values += reuse.reused_leaf_values;
        }
        result.timing.search_seconds += search_started.elapsed().as_secs_f64();
        let report = match searched {
            Ok(report) => report,
            Err(error) => {
                result.error = Some(error);
                result.termination = "error";
                break;
            }
        };
        let selected = report.actions[report.selected_index];
        let q = selected.mean_value();
        if selected.visits == 0 || !q.is_finite() || q.abs() > 1.000001 {
            result.error = Some("selected MCTS action has no valid bounded root Q".into());
            result.termination = "error";
            break;
        }
        if learner_turn {
            let features_started = Instant::now();
            // Root Q is already in this exact player's perspective. A harmony
            // bonus may retain the player; never infer the sign from parity.
            let sample = RootSample {
                decision: result.record.actions().len() + 1,
                perspective: Seat::from_player(perspective),
                features: CompactValueFeatures::extract(&position, perspective)
                    .values()
                    .to_vec(),
                q: q.clamp(-1.0, 1.0),
                selected_visits: selected.visits,
                search_simulations: report.simulations,
            };
            last_learner_sample = Some(sample.clone());
            result.eligible_roots += 1;
            reservoir_sample(
                &mut result.samples,
                sample,
                result.eligible_roots,
                options.samples,
                &mut sampling_rng,
            );
            result.timing.features_seconds += features_started.elapsed().as_secs_f64();
        }
        let apply_started = Instant::now();
        let applied = position.apply(selected.action);
        result.timing.apply_seconds += apply_started.elapsed().as_secs_f64();
        if let Err(error) = applied {
            result.error = Some(error.to_string());
            result.termination = "error";
            break;
        }
        if options.reuse_search {
            for session in &mut sessions {
                session.advance(selected.action);
            }
        }
        result.record.push(selected.action);
        result.actual_simulations.push(report.simulations);
        if let Some(loss) =
            repetitions.observe(&position, perspective, result.record.actions().len())
        {
            if let Some(sample) = last_learner_sample
                .take()
                .filter(|s| s.decision == loss.last_decision)
            {
                if !result.samples.iter().any(|s| s.decision == sample.decision) {
                    if result.samples.len() >= options.samples {
                        result.samples.remove(0);
                    }
                    result.samples.push(sample);
                }
            }
            result.repetition_loss = Some(loss);
            result.termination = "repetition-training-loss";
            break;
        }
    }
    result.outcome = position.outcome();
    result.completed_turns = position.completed_turns();
    result.samples.sort_by_key(|sample| sample.decision);
    let ended = Instant::now();
    result.elapsed_seconds = ended.duration_since(started).as_secs_f64();
    result.game_deadline_overshoot_seconds = options
        .game_seconds
        .map(|seconds| (result.elapsed_seconds - seconds).max(0.0));
    result.campaign_deadline_overshoot_seconds = ended
        .saturating_duration_since(campaign_deadline)
        .as_secs_f64();
    result
}

enum WorkerMessage {
    Game(PlayedGame),
    Failure(String),
}

#[derive(Default, Serialize)]
struct Progress {
    consumed_games: usize,
    repetition_training_losses: usize,
    terminal_games: usize,
    unresolved_games: usize,
    self_play_games: usize,
    legacy_games: usize,
    learner_wins_vs_legacy: usize,
    learner_draws_vs_legacy: usize,
    learner_losses_vs_legacy: usize,
    updates: u64,
    published_version: u64,
    fresh_learner_roots: usize,
    sampled_roots: usize,
    actor_timing: GameTiming,
    coordinator_learning_seconds: f64,
    coordinator_persistence_seconds: f64,
    errors: Vec<String>,
    per_budget: BTreeMap<usize, BudgetProgress>,
}

#[derive(Default, Serialize)]
struct BudgetProgress {
    consumed_games: usize,
    repetition_training_losses: usize,
    terminal_games: usize,
    unresolved_games: usize,
    self_play_games: usize,
    legacy_games: usize,
    learner_wins_vs_legacy: usize,
    learner_draws_vs_legacy: usize,
    learner_losses_vs_legacy: usize,
    fresh_learner_roots: usize,
    sampled_roots: usize,
    updates: u64,
    decisions: usize,
    completed_turns: u64,
    occupied_worker_seconds: f64,
}

impl BudgetProgress {
    fn record(&mut self, game: &PlayedGame, updates: u64) {
        self.consumed_games += 1;
        self.repetition_training_losses += usize::from(game.repetition_loss.is_some());
        self.terminal_games += usize::from(game.outcome != GameOutcome::Ongoing);
        self.unresolved_games += usize::from(game.outcome == GameOutcome::Ongoing);
        self.self_play_games += usize::from(game.self_play);
        self.legacy_games += usize::from(!game.self_play);
        if let Some(seat) = game.learner_seat {
            match game.outcome {
                GameOutcome::Win(winner) if winner == seat.player() => {
                    self.learner_wins_vs_legacy += 1
                }
                GameOutcome::Win(_) => self.learner_losses_vs_legacy += 1,
                GameOutcome::Draw => self.learner_draws_vs_legacy += 1,
                GameOutcome::Ongoing => {}
            }
        }
        self.fresh_learner_roots += game.eligible_roots;
        self.sampled_roots += game.samples.len();
        self.updates += updates;
        self.decisions += game.record.actions().len();
        self.completed_turns += u64::from(game.completed_turns);
        self.occupied_worker_seconds += game.elapsed_seconds;
    }
}

fn persist_game(output: &Path, game: &PlayedGame, options: &Options) -> Result<()> {
    let stem = format!("game-{:08}", game.id);
    let record = game.record.to_string();
    write_new(
        &output.join("games").join(format!("{stem}.psr")),
        record.as_bytes(),
    )?;
    save_json_new(
        &output.join("games").join(format!("{stem}.json")),
        &serde_json::json!({
            "schema":"paisho-compact-selfplay-game-v1", "rules":game.record.rules().as_str(), "game_id":game.id,"worker":game.worker,
            "snapshot_version":game.version,"weights_sha256":game.weights_sha256,
            "build_source_sha256":env!("PAISHO_BUILD_SOURCE_SHA256"),
            "feature_schema":COMPACT_VALUE_SCHEMA_V1,"root_seeds":{"host":game.seeds[0],"guest":game.seeds[1],"reservoir":game.seeds[2]},
            "search_reuse":{"enabled":options.reuse_search,"inherited_root_visits":game.reused_root_visits,"prepared_positions":game.reused_candidate_positions,"leaf_values":game.reused_leaf_values},
            "simulations_requested_per_decision":game.simulations,"actual_simulations":game.actual_simulations,
            "self_play":game.self_play,"learner_seat":game.learner_seat,
            "opponent":if game.self_play {"same-game-frozen-compact-snapshot"} else {"CpuMctsEvaluator-legacy-default-weights"},
            "record_sha256":sha256(record.as_bytes()),"record_path":format!("{stem}.psr"),
            "decisions":game.record.actions().len(),"completed_turns":game.completed_turns,"outcome":outcome_name(game.outcome),
            "training_adjudication":game.repetition_loss,
            "eligible_for_repetition_training":game.repetition_loss.is_some() && game.error.is_none(),
            "termination":game.termination,"error":game.error,"elapsed_seconds":game.elapsed_seconds,
            "timing":game.timing,"learning_enabled":options.learn,
            "requested_game_seconds":game.requested_game_seconds,"schedule_id":game.schedule_id,
            "effective_deadline_seconds_from_game_start":game.effective_deadline_seconds_from_game_start,
            "game_deadline_overshoot_seconds":game.game_deadline_overshoot_seconds,
            "campaign_deadline_overshoot_seconds":game.campaign_deadline_overshoot_seconds,
            "eligible_for_terminal_training":game.outcome != GameOutcome::Ongoing && game.error.is_none(),
            "eligible_learner_roots":game.eligible_roots,"sample_method":if game.repetition_loss.is_some() {"uniform-reservoir-with-required-cycle-closing-root-v2"} else {"uniform-reservoir-without-replacement-v1"},
            "sampled_roots":game.samples
        }),
    )
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn publish_progress(output: &Path, progress: &Progress) -> Result<()> {
    let temp = output.join("progress.tmp");
    let bytes = serde_json::to_vec_pretty(progress)?;
    let mut file = fs::File::create(&temp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(temp, output.join("progress.json"))?;
    Ok(())
}

fn timed_persistence<T>(progress: &mut Progress, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let started = Instant::now();
    let result = work();
    progress.coordinator_persistence_seconds += started.elapsed().as_secs_f64();
    result
}

fn publish_progress_timed(output: &Path, progress: &mut Progress) -> Result<()> {
    let started = Instant::now();
    let result = publish_progress(output, progress);
    progress.coordinator_persistence_seconds += started.elapsed().as_secs_f64();
    result
}

mod learning;
use learning::learn_game;

/// Finite pilot only. The time limit is soft: root ordering, one search
/// simulation and artifact publication may finish after its deadline.
pub fn run(args: &[String]) -> Result<()> {
    run_options(Options::parse(args)?)
}

fn run_options(options: Options) -> Result<()> {
    options.validate()?;
    let input_bytes = fs::read(&options.model)?;
    let parent = load_model(&options.model)?;
    if fs::read(&options.model)? != input_bytes {
        return Err(invalid("initial model changed while loading"));
    }
    let mut replay = ReplayMemory::new(options.replay_capacity, options.seed ^ 0x5245504c4159);
    if let Some(path) = &options.replay_input {
        replay.load(path)?;
    }
    let mut model = parent.model()?;
    let seconds = Duration::try_from_secs_f64(options.seconds)?;
    let deadline = Instant::now()
        .checked_add(seconds)
        .ok_or_else(|| invalid("deadline overflow"))?;
    let started = Instant::now();
    if let Some(directory) = options
        .output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
    {
        fs::create_dir_all(directory)?;
    }
    fs::create_dir(&options.output)?;
    fs::create_dir(options.output.join("games"))?;
    fs::create_dir(options.output.join("models"))?;
    write_new(&options.output.join("initial-model.json"), &input_bytes)?;
    save_model_new(
        &options.output.join("models/version-00000000.json"),
        &parent,
    )?;
    let worker_count = options.workers.min(options.games);
    let assignments: Vec<_> = (0..worker_count).map(|worker| {
        let actor = options.worker_options(worker);
        serde_json::json!({"worker":worker,"simulations":actor.simulations,"game_seconds":actor.game_seconds})
    }).collect();
    save_json_new(
        &options.output.join("plan.json"),
        &serde_json::json!({
            "schema":"paisho-compact-selfplay-plan-v1","rules":paisho_core::RuleProfileId::CURRENT.as_str(),"options":options,
            "initial_model_sha256":sha256(&input_bytes),"initial_weights_sha256":weights_hash(&model),
            "build_source_sha256":env!("PAISHO_BUILD_SOURCE_SHA256"),
            "build_git_revision":env!("PAISHO_BUILD_GIT_REVISION"),"build_git_dirty":env!("PAISHO_BUILD_GIT_DIRTY"),
            "workers":worker_count,"rayon_threads_per_worker":1,"result_channel_capacity":worker_count,
            "learning_enabled":options.learn,"worker_assignments":assignments,
            "shared_learner_model":true,
            "collection_mode":if options.learn { "online-learning" } else { "frozen-initial-model" },
            "schedule":if options.budgets.is_empty() {
                "even game IDs self-play; odd IDs vs frozen CPU heuristic, learner seat alternates"
            } else {
                "each worker independently cycles self-play, legacy as Host, self-play, legacy as Guest; shared model snapshot adopted at game start"
            },
            "starts":"standard balanced setup; six starting flowers cycle by schedule-id/2 (global game ID in single-budget mode, local actor ordinal in mixed mode)",
            "search":{"simulations":if options.budgets.is_empty() {Some(options.simulations)} else {None},"budgets":options.budgets,"independent_trees":1,"maximum_tree_depth":96,
                "retained_subtree":options.reuse_search,"simulation_accounting":"actual_simulations counts new visits only; selected_visits may include inherited visits",
                "action_ranking":"exhaustive","root_widening":1,"internal_widening":1,"rollout_depth":0,
                "exploration":core::f32::consts::SQRT_2},
            "target":"(1-lambda)*played-most-visits-root-Q + lambda*terminal-result in root.to_move perspective",
            "terminal_filter":"rules terminal games use Q/result mixture; explicit repetition losses use -1 only for closing player roots inside repeated suffix; no fabricated opponent win; other ongoing/error games have no targets",
            "replay":{"capacity":options.replay_capacity,"extra_updates_per_fresh":options.replay_ratio,"input":options.replay_input,"initial_examples":replay.len(),"seed":options.seed ^ 0x5245504c4159_u64,"sampling":"uniform recent roots, immutable target and actor identity"},
            "repetition":{"cycles":options.repetition_cycles,"min_decisions":24,"max_period":32,"rule_outcome_unchanged":true},
            "deadline":"minimum of soft campaign and optional per-game deadlines, checked between moves and simulations; in-flight root ordering may overrun",
            "timing_scope":"actor components summed across games/workers may overlap; coordinator learning excludes persistence; startup and final summary write excluded from persistence telemetry; not a measured overhead comparison",
            "reproducibility":"seeded searches and saved model versions; asynchronous game completion order is recorded, not deterministic"
        }),
    )?;
    let shared: SharedModel = Arc::new(RwLock::new((0, Arc::new(model.clone()))));
    // Reserve one initial game per heterogeneous worker so a fast actor cannot
    // consume a small finite quota before a slower worker has even started.
    let next = Arc::new(AtomicUsize::new(if options.budgets.is_empty() {
        0
    } else {
        worker_count
    }));
    let cancelled = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::sync_channel(worker_count);
    let mut workers = Vec::new();
    for worker in 0..worker_count {
        let (sender, options, shared, next, cancelled) = (
            sender.clone(),
            options.worker_options(worker),
            Arc::clone(&shared),
            Arc::clone(&next),
            Arc::clone(&cancelled),
        );
        workers.push(thread::spawn(move || {
            let pool = match rayon::ThreadPoolBuilder::new().num_threads(1).build() {
                Ok(pool) => pool,
                Err(error) => {
                    let _ = sender.send(WorkerMessage::Failure(error.to_string()));
                    return;
                }
            };
            let mut local_ordinal = 0;
            while Instant::now() < deadline && !cancelled.load(Ordering::Relaxed) {
                let id = if !options.budgets.is_empty() && local_ordinal == 0 {
                    worker
                } else {
                    next.fetch_add(1, Ordering::Relaxed)
                };
                if id >= options.games {
                    break;
                }
                let (version, snapshot) = match shared.read() {
                    Ok(snapshot) => (snapshot.0, Arc::clone(&snapshot.1)),
                    Err(_) => {
                        let _ = sender
                            .send(WorkerMessage::Failure("shared model lock poisoned".into()));
                        break;
                    }
                };
                let game = pool.install(|| {
                    play_game(
                        &options,
                        GameIdentity {
                            id,
                            worker,
                            schedule_id: if options.budgets.is_empty() {
                                id
                            } else {
                                local_ordinal
                            },
                        },
                        version,
                        &snapshot,
                        deadline,
                        &cancelled,
                    )
                });
                local_ordinal += 1;
                if sender.send(WorkerMessage::Game(game)).is_err() {
                    break;
                }
            }
        }));
    }
    drop(sender);
    let mut progress = Progress::default();
    for message in receiver {
        let result: Result<()> = match message {
            WorkerMessage::Failure(error) => Err(invalid(error)),
            WorkerMessage::Game(game) => (|| {
                timed_persistence(&mut progress, || {
                    persist_game(&options.output, &game, &options)
                })?;
                progress.consumed_games += 1;
                progress.repetition_training_losses += usize::from(game.repetition_loss.is_some());
                progress.self_play_games += usize::from(game.self_play);
                progress.legacy_games += usize::from(!game.self_play);
                progress.fresh_learner_roots += game.eligible_roots;
                progress.sampled_roots += game.samples.len();
                progress.actor_timing.add(game.timing);
                if game.outcome == GameOutcome::Ongoing {
                    progress.unresolved_games += 1;
                } else {
                    progress.terminal_games += 1;
                    if let Some(seat) = game.learner_seat {
                        match game.outcome {
                            GameOutcome::Win(winner) if winner == seat.player() => {
                                progress.learner_wins_vs_legacy += 1;
                            }
                            GameOutcome::Win(_) => progress.learner_losses_vs_legacy += 1,
                            GameOutcome::Draw => progress.learner_draws_vs_legacy += 1,
                            GameOutcome::Ongoing => unreachable!("terminal branch"),
                        }
                    }
                }
                let updates_before = progress.updates;
                if progress.errors.is_empty() {
                    learn_game(
                        &game,
                        &options,
                        &parent,
                        &mut model,
                        &shared,
                        &mut progress,
                        &mut replay,
                    )?;
                }
                progress
                    .per_budget
                    .entry(game.simulations)
                    .or_default()
                    .record(&game, progress.updates - updates_before);
                publish_progress_timed(&options.output, &mut progress)?;
                if let Some(error) = game.error {
                    return Err(invalid(error));
                }
                Ok(())
            })(),
        };
        if let Err(error) = result {
            progress.errors.push(error.to_string());
            cancelled.store(true, Ordering::Relaxed);
        }
    }
    for worker in workers {
        if worker.join().is_err() {
            progress.errors.push("actor thread panicked".into());
        }
    }
    replay.persist(&options.output)?;
    let final_artifact = parent.with_model(&model,parent.training_steps + progress.updates,
        serde_json::json!({"rules":paisho_core::RuleProfileId::CURRENT.as_str(),"method":if options.learn { "compact-replay-and-repetition-sgd-v2" } else { "frozen-collection-no-updates-v1" },"version":progress.published_version,
            "learning_enabled":options.learn,
            "initial_model_sha256":sha256(&input_bytes),"run_plan":"plan.json",
            "build_source_sha256":env!("PAISHO_BUILD_SOURCE_SHA256"),"progress":progress}));
    timed_persistence(&mut progress, || {
        save_model_new(&options.output.join("final-model.json"), &final_artifact)
    })?;
    publish_progress_timed(&options.output, &mut progress)?;
    let summary = serde_json::json!({"schema":"paisho-compact-selfplay-summary-v1","rules":paisho_core::RuleProfileId::CURRENT.as_str(),"progress":progress,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"requested_seconds":options.seconds,
        "requested_games":options.games,"deadline_reached":Instant::now() >= deadline,
        "learning_enabled":options.learn,"collection_mode":if options.learn { "online-learning" } else { "frozen-initial-model" },
        "requested_game_seconds":options.game_seconds,
        "final_weights_sha256":weights_hash(&model),
        "final_model_sha256":sha256(&fs::read(options.output.join("final-model.json"))?),
        "status":if progress.errors.is_empty() {"completed-bounded-pilot"} else {"failed"},
        "strength_claim":"none: training-game outcomes are not independent promotion evidence"});
    save_json_new(&options.output.join("summary.json"), &summary)?;
    println!("{summary}");
    if !progress.errors.is_empty() {
        return Err(invalid(progress.errors.join("; ")));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
