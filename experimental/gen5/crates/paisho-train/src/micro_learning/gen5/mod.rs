//! Gen5: Gen4 weights/architecture, explicit rules, exploration and independent collection.
use super::*;
use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId};
use std::{
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
mod archive;
mod artifact_identity;
#[doc(hidden)]
pub use artifact_identity::benchmark as benchmark_artifact_identity;
mod case_actor;
mod cases;
mod checkpoint;
mod checkpoint_retention;
mod ensemble_actor;
mod ladder;
mod references;
mod depth_duel_probe;
#[doc(hidden)]
pub use depth_duel_probe::run as run_depth_duel_probe;
mod depth_confirmation;
#[doc(hidden)]
pub use depth_confirmation::run as run_depth_confirmation;
pub use references::probe as probe_opponents;
pub use references::{Frozen, OpponentSpec};
mod durable;
#[doc(hidden)]
pub use durable::verify_prefetch as verify_coverage_prefetch;
#[doc(hidden)]
pub use durable::verify_policy_distance as verify_recall_policy_distance;
mod proof_cache;
mod prefix_states;
#[doc(hidden)]
pub use prefix_states::benchmark as benchmark_prefix_states;
mod reanalysis;
mod search_cache;
#[doc(hidden)]
pub use search_cache::ReanalysisCache;
mod revisions;
mod relational_migration;
#[doc(hidden)]
pub use relational_migration::prepare as prepare_relational_recovery;
pub use cases::CaseOptions;
mod collector;
mod evaluation;
mod protection;
#[doc(hidden)]
pub use protection::probe_mechanism_repair;
pub use durable::probe_recall_repair;
mod publication;
mod publication_cadence;
pub use publication_cadence::WorkBudget as PublicationWorkBudget;
pub use publication::probe_dynamic_proof_control;
#[doc(hidden)]
pub use publication::probe_policy_relay;
pub use publication::{probe_gain_transfer,probe_gain_transfer_relinearized};
#[doc(hidden)]
pub use publication::benchmark_lazy_publication;
#[doc(hidden)]
pub use protection::verify_frontier as verify_consolidation_frontier;
#[doc(hidden)]
pub use protection::verify_kl_frontier;
#[doc(hidden)]
pub use protection::verify_fresh_gain_frontier;
#[doc(hidden)]
pub use protection::verify_validation_value_frontier;
#[doc(hidden)]
pub use protection::replay_captured_consolidation;
pub use protection::replay_capture_publication;
#[doc(hidden)]
pub use protection::measure_block_retention;
#[doc(hidden)]
pub use protection::measure_reader_context_drift;
#[doc(hidden)]
pub use protection::measure_final_block_retention;
mod resume_example;
mod block_retention_export;
#[doc(hidden)]
pub use block_retention_export::run as export_block_retention;
mod teaching;
/// Isolated representation probes use the exact production teaching transform.
#[doc(hidden)]
pub fn diagnostic_teaching_target(report: &MicroSearchReport, prior: &[f64]) -> Result<(Vec<f64>,Vec<f64>)> {
    teaching::target_with_prior(report,prior)
}
mod verify_v34;
mod loop_cycle_probe;
#[doc(hidden)]
pub use loop_cycle_probe::measure_search_transfer as measure_tape_search_transfer;
#[doc(hidden)]
pub use loop_cycle_probe::run as probe_learning_cycles;
#[doc(hidden)]
pub use loop_cycle_probe::run_final_evaluation as probe_final_evaluation;
#[doc(hidden)]
pub use loop_cycle_probe::{run_learner_tape as probe_learner_tape,preflight_learner_tape,run_step_fractions as probe_tape_step_fractions};
#[doc(hidden)]
pub use loop_cycle_probe::measure_base_policy as measure_tape_base_policy;
#[doc(hidden)]
pub use publication::{
    verify_transactional_candidate,
    verify_consolidation_cycle,
    benchmark_publication, verify_candidate as verify_learning_loop_candidate, verify_guard as verify_learning_loop_guard,
};
#[doc(hidden)]
pub use verify_v34::run as verify_v34_integration;
mod cpu;
mod history;
mod memory;
mod recall;
mod runtime;
pub use runtime::run;
mod fixed_bench;
pub use fixed_bench::benchmark_frozen_games;
pub const RULES: RuleProfileId = RuleProfileId::SkudPaiShoGen5V1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuredRecallAnchors {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    /// New V5 launches opt in with an explicitly migrated, neutral ~200k memory.
    /// Old saved options stay readable and retain their original model family.
    pub neural_memory: bool,
    #[serde(skip)]
    #[doc(hidden)]
    pub reanalysis_cache: Option<Arc<ReanalysisCache>>,
    /// V34: coherent raw targets, coverage, protected consolidation and branch publication.
    pub learning_loop_v2: bool,
    /// Transactional learning and actionable, decision-based publication checks.
    /// Explicit opt-in keeps old saved protocols reproducible.
    pub learning_loop_v3: bool,
    /// Relay learned regulatory choices into admitted actors and retain them.
    /// Explicit opt-in; older configurations retain their publication behavior.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub publication_transfer: bool,
    /// None preserves the historical 30-second clock. Some uses actual consumed
    /// examples only; all publication acceptance criteria remain unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub publication_work_budget: Option<PublicationWorkBudget>,
    /// Diagnostic persistence at due V3 consolidation boundaries; no extra inference.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub diagnostic_consolidation_capture: bool,
    /// Additional fixed retention sources, separate from the corrective focus.
    pub publication_validation: Option<PathBuf>,
    /// Integrated learning-loop protocol; old runs default to false.
    pub learning_loop_repair: bool,
    pub value_policy_strength: f64,
    pub publication_guard: Option<PathBuf>,
    #[serde(skip)]
    pub measurement: bool,
    #[serde(skip)]
    pub observed_origin: Option<(f64, String)>,
    /// Empty preserves the existing single-reference protocol.
    pub opponents: Vec<OpponentSpec>,
    pub checkpoint_keep: usize,
    pub proof_recall: bool,
    #[serde(skip)]
    #[doc(hidden)]
    pub match_reference: Option<(Arc<references::Frozen>, usize)>,
    pub case_curriculum: Option<CaseOptions>,
    /// V22: durable fraction of ALL examples, including fresh and human samples.
    pub structural_repair: bool,
    pub recall_fraction: f64,
    /// Hashed seed checkpoint restored inside the existing quota and 32 MiB cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_recall_anchors: Option<StructuredRecallAnchors>,
    /// Optional immutable recall sources; new lessons always go to case_curriculum.archive.
    pub recall_archive_sources: Vec<PathBuf>,
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
    /// Partition the existing CPU budget; actors keep their own trees and cases.
    pub search_pool_shards: usize,
    /// Reserve part of the SAME CPU budget for ordered learning/protection.
    /// Zero retains the shared search-worker scheduler.
    pub learner_threads: usize,
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
    /// PUCT leaf minimum, not a depth cap. Zero preserves old saved protocols.
    /// Positive values allow budget*minimum expansions for the same root rollouts.
    pub minimum_search_depth: usize,
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
            neural_memory: false,
            reanalysis_cache: None,
            learning_loop_v2: false,
            learning_loop_v3: false,
            publication_transfer: false,
            publication_work_budget: None,
            diagnostic_consolidation_capture: false,
            publication_validation: None,
            learning_loop_repair: false,
            value_policy_strength: 0.,
            publication_guard: None,
            measurement: false,
            observed_origin: None,
            opponents: vec![],
            checkpoint_keep: 64,
            proof_recall: false,
            match_reference: None,
            case_curriculum: None,
            structural_repair: false,
            recall_fraction: 0.5,
            structured_recall_anchors: None,
            recall_archive_sources: vec![],
            legacy_replay: false,
            control_stop: true,
            stop_signal: None,
            candidate_seat: None,
            checkpoint_seconds: 0.0,
            inline_learning: false,
            search_pool_shards: 1,
            learner_threads: 0,
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
            minimum_search_depth: 0,
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
        if self.minimum_search_depth > 96 || (self.minimum_search_depth > 0 && self.mode != "puct") {
            return Err("minimum search depth requires PUCT and must be 0..=96".into());
        }
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
        if self.structured_recall_anchors.is_some() && (!self.learning_loop_v3 || self.case_curriculum.is_none() || !self.structural_repair) {
            return Err(invalid("structured anchor recovery requires V3 structural curriculum"));
        }
        if self.publication_transfer && !self.learning_loop_v3 {
            return Err(invalid("publication transfer requires transactional V3"));
        }
        if let Some(budget)=&self.publication_work_budget {
            budget.validate()?;
            if !self.learning_loop_v3 || !self.structural_repair || !self.learn {
                return Err(invalid("publication work budget requires V3 learning and structural repair"));
            }
        }
        if self.publication_transfer && self.diagnostic_consolidation_capture {
            return Err(invalid("publication transfer cannot use the older consolidation capture path"));
        }
        if self.diagnostic_consolidation_capture && !self.learning_loop_v3 {
            return Err(invalid("consolidation capture requires transactional V3"));
        }
        if self.learning_loop_v2
            && (!self.learning_loop_repair
                || self.publication_validation.is_none()
                || !self.proof_recall)
        {
            return Err(invalid("V34 requires the repaired loop, direct proof recall and a separate validation panel"));
        }
        if !self.value_policy_strength.is_finite()
            || !(0.0..=16.0).contains(&self.value_policy_strength)
            || self.learning_loop_repair
                && (!self.structural_repair
                    || self.opponents.len() != 5
                    || self.publication_guard.is_none())
            || !self.learning_loop_repair && self.value_policy_strength != 0.
        {
            return Err(invalid("invalid integrated learning-loop protocol"));
        }

        self.search(0, true).map_err(invalid)?;
        if self.checkpoint_keep < 2 || self.checkpoint_keep > 512 {
            return Err(invalid("checkpoint retention must be between 2 and 512"));
        }
        if !self.opponents.is_empty()
            && (self.opponents.len() != 5 || !self.legacy_replay || !self.structural_repair)
        {
            return Err(invalid(
                "five-generation curriculum requires five references and structural case replay",
            ));
        }
        if self.learner_threads > 0 && (self.learner_threads >= self.main_threads()
            || !self.learning_loop_v2 || !self.inline_learning || !self.legacy_replay
            || self.secondary_capacity() > 0) {
            return Err(invalid("reserved learner workers require repaired inline Gen5 without secondary work"));
        }
        if self.search_pool_shards == 0
            || self.search_pool_shards > self.main_threads().saturating_sub(self.learner_threads)
            || self.main_threads().saturating_sub(self.learner_threads) % self.search_pool_shards != 0
            || self.search_pool_shards > 1 && (!self.legacy_replay || !self.inline_learning)
        {
            return Err(invalid("search pool shards must divide the CPU budget; partitioning requires inline legacy-replay learning"));
        }
        if self.recall_archive_sources.len() > 8
            || (!self.recall_archive_sources.is_empty() && !self.structural_repair)
        {
            return Err(invalid("invalid read-only recall sources"));
        }
        if self.structural_repair
            && (self.case_curriculum.is_none()
                || !self.recall_fraction.is_finite()
                || !(0.0..=1.0).contains(&self.recall_fraction)
                || self.recall_fraction * (self.replay_ratio + 1) as f64
                    > self.replay_ratio as f64 * (1.0 - self.human_fraction))
        {
            return Err(invalid(
                "recall fraction exceeds available replay after human quota",
            ));
        }
        if self.legacy_replay
            && (self.case_curriculum.is_none()
                || self.historical
                || self.history_interval != 0.0
                || self.reuse_idle_secondary)
        {
            return Err(invalid(
                "legacy replay requires dedicated main actors without a secondary lane",
            ));
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

#[cfg(test)]
mod publication_transfer_options_tests {
    use super::*;
    #[test]
    fn minimum_depth_is_explicit_serialized_and_puct_only() {
        let mut o:Options=serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(o.minimum_search_depth,0);
        o.minimum_search_depth=5;
        assert!(o.search(1,true).is_ok());
        let restored:Options=serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
        assert_eq!(restored.minimum_search_depth,5);
        o.mode="gumbel".into();assert!(o.search(1,true).is_err());
        o.mode="puct".into();o.minimum_search_depth=97;assert!(o.search(1,false).is_err());
    }
    #[test]
    fn publication_transfer_is_explicit_and_requires_v3() {
        let old: Options = serde_json::from_value(serde_json::json!({})).unwrap();
        assert!(!old.publication_transfer);
        assert!(serde_json::to_value(&old).unwrap().get("publication_transfer").is_none());
        let enabled = Options { publication_transfer: true, ..old };
        assert_eq!(enabled.validate().unwrap_err().to_string(),
            "publication transfer requires transactional V3");
        let encoded = serde_json::to_value(&enabled).unwrap();
        assert!(serde_json::from_value::<Options>(encoded).unwrap().publication_transfer);
    }
}

mod portable_recovery;
#[doc(hidden)]
pub use portable_recovery::{finalize as finalize_portable_recovery, verify as verify_portable_recovery};
