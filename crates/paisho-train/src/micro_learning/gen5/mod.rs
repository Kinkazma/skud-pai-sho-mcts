//! Gen5: Gen4 weights/architecture, explicit rules, exploration and independent collection.
use super::*;
use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
mod archive;
mod ladder;
mod case_actor;
mod cases;
mod checkpoint;
mod durable;
mod proof_cache;
mod reanalysis;
mod revisions;
pub use cases::CaseOptions;
mod collector;
mod cpu;
mod history;
mod memory;
mod runtime;
pub use runtime::run;
pub const RULES: RuleProfileId = RuleProfileId::SkudPaiShoGen5V1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub case_curriculum: Option<CaseOptions>,
    pub legacy_replay: bool,
    pub control_stop: bool,
    #[serde(skip)]
    #[doc(hidden)]
    pub stop_signal: Option<Arc<std::sync::atomic::AtomicBool>>,
    #[serde(skip)]
    #[doc(hidden)]
    pub candidate_seat: Option<Player>,
    /// Zero preserves legacy per-step durability; otherwise RAM weights and periodic commits.
    pub checkpoint_seconds: f64,
    pub inline_learning: bool,
    pub archive_workers: usize,
    pub model: PathBuf,
    pub evaluation_anchor: Option<PathBuf>,
    pub resume_progress: Option<PathBuf>,
    pub reference: PathBuf,
    pub human_dataset: Option<PathBuf>,
    pub output: PathBuf,
    pub seconds: f64,
    /// Optional hard UTC epoch deadline; preparation consumes this budget too.
    pub end_unix_seconds: Option<f64>,
    pub games: usize,
    pub threads: usize,
    pub actors: usize,
    pub seed: u64,
    pub learn: bool,
    pub mode: String,
    pub dirichlet_fraction: f64,
    pub dirichlet_total: f64,
    pub forced_playout_strength: f64,
    pub proof_search: bool,
    pub correction_replay_fraction: f64,
    pub correction_capacity: usize,
    pub considered_actions: usize,
    pub budgets: Vec<(usize, f64)>,
    /// Use all sufficiently deep completed searches for policy learning.
    pub policy_min_budget: usize,
    pub game_seconds: f64,
    pub historical_seconds: Vec<(usize, f64)>,
    /// These references have only the campaign deadline; CPU admission is per job.
    pub historical_unlimited_budgets: Vec<usize>,
    pub historical: bool,
    pub secondary_threads: usize,
    pub reuse_idle_secondary: bool,
    pub macos_qos: bool,
    /// Conservative secondary-pool reservation budget / total CPU capacity.
    pub historical_capacity_fraction: f64,
    /// Frozen calibration only: independent one-CPU historical games, no learner.
    pub calibration_reference_budget: Option<usize>,
    pub checkpoint_fraction: f64,
    pub history_interval: f64,
    pub history_seconds: f64,
    /// Paired internal evaluation, independent of collection limits.
    pub history_initial_pairs: usize,
    pub history_decision_limit: usize,
    pub decision_limit: usize,
    pub replay_capacity: usize,
    pub replay_ratio: usize,
    pub replay_max_bytes: usize,
    pub human_fraction: f64,
    pub historical_replay_fraction: f64,
    pub rate: f64,
    pub replay_index: Option<PathBuf>,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            case_curriculum: None,
            legacy_replay: false,
            control_stop: true,
            stop_signal: None,
            candidate_seat: None,
            checkpoint_seconds: 0.0,
            inline_learning: false,
            archive_workers: 1,
            model: PathBuf::new(),
            evaluation_anchor: None,
            resume_progress: None,
            reference: PathBuf::new(),
            human_dataset: None,
            output: PathBuf::new(),
            seconds: 60.0,
            end_unix_seconds: None,
            games: 10_000_000,
            threads: 10,
            actors: 9,
            seed: 95000,
            learn: true,
            mode: "puct".into(),
            dirichlet_fraction: 0.25,
            dirichlet_total: 10.0,
            forced_playout_strength: 2.0,
            proof_search: true,
            correction_replay_fraction: 0.05,
            correction_capacity: 8192,
            considered_actions: 16,
            budgets: vec![(64, 0.8), (256, 0.2)],
            policy_min_budget: 256,
            game_seconds: 30.0,
            historical_seconds: vec![(32, 30.0), (64, 45.0), (128, 60.0)],
            historical_unlimited_budgets: vec![32, 64, 128],
            historical: true,
            secondary_threads: 1,
            reuse_idle_secondary: false,
            macos_qos: false,
            historical_capacity_fraction: 0.075,
            calibration_reference_budget: None,
            checkpoint_fraction: 0.05,
            history_interval: 900.0,
            history_seconds: 600.0,
            history_initial_pairs: 50,
            history_decision_limit: 800,
            decision_limit: 600,
            replay_capacity: 327680,
            replay_ratio: 4,
            replay_max_bytes: 24 * 1024 * 1024 * 1024,
            human_fraction: 0.02,
            historical_replay_fraction: 0.05,
            rate: 0.02,
            replay_index: None,
        }
    }
}
impl Options {
    pub fn search(
        &self,
        seed: u64,
        training: bool,
    ) -> std::result::Result<MicroSearchOptions, String> {
        let mode = match self.mode.as_str() {
            "puct" => MicroSearchMode::Puct,
            "gumbel" => MicroSearchMode::Gumbel,
            _ => return Err("mode must be puct or gumbel".into()),
        };
        let o = MicroSearchOptions {
            mode,
            seed,
            dirichlet_fraction: if training && mode == MicroSearchMode::Puct {
                self.dirichlet_fraction
            } else {
                0.0
            },
            dirichlet_total: self.dirichlet_total,
            gumbel_scale: if training { 1.0 } else { 0.0 },
            considered_actions: self.considered_actions,
            proof_search: self.proof_search,
            forced_playout_strength: if training && mode == MicroSearchMode::Puct {
                self.forced_playout_strength
            } else {
                0.0
            },
        };
        o.validate()?;
        Ok(o)
    }
    fn validate(&self) -> Result<()> {
        self.search(0, true).map_err(invalid)?;
        if self.legacy_replay && (self.case_curriculum.is_none() || self.historical || self.history_interval != 0.0 || self.reuse_idle_secondary) {
            return Err(invalid("legacy replay requires dedicated main actors without a secondary lane"));
        }
        if !self.checkpoint_seconds.is_finite()
            || self.checkpoint_seconds < 0.0
            || self.checkpoint_seconds > 300.0
            || self.checkpoint_seconds > 0.0 && self.case_curriculum.is_none()
        {
            return Err(invalid("invalid RAM checkpoint interval"));
        }
        if let Some(c) = &self.case_curriculum {
            if !self.learn
                || self.human_dataset.is_none()
                || c.archive.as_os_str().is_empty()
                || c.losses == 0
                || c.reversals == 0
                || c.unresolved_attempts == 0
                || c.reanalysis_positions == 0
                || c.reanalysis_positions > 16
                || ![256, 512].contains(&c.reanalysis_budget)
                || !(0.0..=0.25).contains(&c.durable_fraction)
                || c.durable_fraction == 0.0
                || self.budgets.iter().any(|(b, _)| ![256, 512].contains(b))
                || self.calibration_reference_budget.is_some()
                || self.checkpoint_fraction != 0.0
            {
                return Err(invalid("invalid human case curriculum configuration"));
            }
        }
        if !(1..=4).contains(&self.archive_workers)
            || self.games == 0
            || self.policy_min_budget == 0
            || self.policy_min_budget > 2048
            || !(0.0..=1.0).contains(&self.correction_replay_fraction)
            || !self.forced_playout_strength.is_finite()
            || !(0.0..=16.0).contains(&self.forced_playout_strength)
            || self.threads == 0
            || self.threads > std::thread::available_parallelism()?.get()
            || self.actors == 0
            || self.actors
                > if self.case_curriculum.is_some() && self.checkpoint_seconds > 0.0 {
                    self.threads * 4
                } else if self.reuse_idle_secondary {
                    self.threads
                } else {
                    self.main_threads()
                }
            || !self.historical_capacity_fraction.is_finite()
            || !(0.0..=0.09).contains(&self.historical_capacity_fraction)
            || self.secondary_threads == 0
            || self.main_threads() == 0
            || self.historical && self.threads < 2
            || self.calibration_reference_budget.is_some_and(|b| {
                ![32, 64, 128].contains(&b)
                    || self.learn
                    || self.historical
                    || self.history_interval > 0.0
            })
            || self.decision_limit == 0
            || self.history_initial_pairs == 0
            || self.history_initial_pairs.checked_mul(2).is_none()
            || self.history_decision_limit == 0
            || !self.seconds.is_finite()
            || self.seconds <= 0.0
            || self.seconds > 86400.0
            || self
                .end_unix_seconds
                .is_some_and(|s| !s.is_finite() || s <= 0.0)
            || !self.game_seconds.is_finite()
            || self.game_seconds <= 0.0
            || !self.history_interval.is_finite()
            || self.history_interval < 0.0
            || !self.history_seconds.is_finite()
            || self.history_seconds <= 0.0
            || !(0.0..=0.1).contains(&self.checkpoint_fraction)
            || !(0.0..=0.1).contains(&self.human_fraction)
            || !(0.0..=0.1).contains(&self.historical_replay_fraction)
            || self.replay_capacity == 0
            || self.replay_max_bytes == 0
            || self.replay_ratio > 16
            || !self.rate.is_finite()
            || self.rate <= 0.0
            || self.budgets.is_empty()
            || self
                .budgets
                .iter()
                .any(|(b, w)| *b == 0 || *b > 2048 || !w.is_finite() || *w <= 0.0)
            || (self.budgets.iter().map(|(_, w)| w).sum::<f64>() - 1.0).abs() > 1e-9
            || self.historical_seconds.len() != 3
            || self
                .historical_unlimited_budgets
                .iter()
                .any(|b| ![32, 64, 128].contains(b))
            || [32, 64, 128].iter().any(|b| {
                self.historical_seconds
                    .iter()
                    .filter(|(v, t)| v == b && t.is_finite() && *t > 0.0)
                    .count()
                    != 1
            })
            || self.human_fraction > 0.0 && self.learn && self.human_dataset.is_none()
        {
            return Err(invalid("invalid bounded Gen5 configuration"));
        }
        Ok(())
    }
    fn main_threads(&self) -> usize {
        self.threads.saturating_sub(self.secondary_capacity())
    }
    fn secondary_capacity(&self) -> usize {
        if self.historical || self.history_interval > 0.0 || self.reuse_idle_secondary {
            self.secondary_threads
        } else {
            0
        }
    }
}
#[derive(Clone)]
struct Snapshot {
    artifact: Option<Arc<MicroArtifact>>,
    version: u64,
    identity: String,
    model: Arc<MicroModel>,
    path: PathBuf,
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
enum Lane {
    Selfplay,
    Checkpoint,
    Historical,
    Reanalysis,
}
fn shuffle<T>(values: &mut [T], rng: &mut StableRng) {
    for i in (1..values.len()).rev() {
        let j = rng.index(i + 1);
        values.swap(i, j);
    }
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(value)?)?;
    fs::rename(tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests;
