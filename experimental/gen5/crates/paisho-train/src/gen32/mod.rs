//! Independently versioned Gen3 continuation; never consumes Gen5 value weights.
use crate::compact_learning::{invalid, load_model, sha256, ModelArtifact};
use crate::micro_learning::{MicroArtifact, SavedMicroExample};
use paisho_ai::*;
use paisho_core::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const RULES: RuleProfileId = RuleProfileId::SkudPaiSho2022V2;
mod evaluation;
mod game;
mod runtime;
mod tactics;
pub use evaluation::{compare, compare_learning, compare_panel, compare_seed, compare_tactics};
pub use runtime::run;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub schema: String,
    pub generation: String,
    pub rules: String,
    pub compact: ModelArtifact,
    pub policy: MicroArtifact,
    pub parent_sha256: String,
    pub updates: u64,
    pub memory_manifest: Option<PathBuf>,
    pub memory_manifest_sha256: Option<String>,
}
impl Artifact {
    pub fn load(path: &Path) -> Result<Self> {
        let a: Self = serde_json::from_slice(&fs::read(path)?)?;
        a.model()?;
        Ok(a)
    }
    pub fn model(&self) -> Result<Gen32Model> {
        if self.schema != "paisho-gen3-policy-memory-v1" || self.rules != RULES.as_str() {
            return Err(invalid("Gen3 continuation schema/rules mismatch"));
        }
        let policy = self.policy.model()?;
        match (
            &self.memory_manifest,
            &self.memory_manifest_sha256,
            policy.sequence_memory(),
        ) {
            (Some(path), Some(hash), Some(bank)) => {
                let bytes = fs::read(path)?;
                let meta: serde_json::Value = serde_json::from_slice(&bytes)?;
                if sha256(&bytes) != *hash
                    || meta["rules"] != RULES.as_str()
                    || meta["sha256"] != bank.spec.sha256
                {
                    return Err(invalid("Gen3 memory rules/hash mismatch"));
                }
            }
            (None, None, None) => {}
            _ => return Err(invalid("Gen3 memory manifest required")),
        }
        Ok(Gen32Model {
            value: self.compact.model()?,
            policy,
        })
    }
    pub fn updated(&self, model: &Gen32Model, updates: u64) -> Self {
        let mut a = self.clone();
        a.updates = updates;
        a.compact=self.compact.with_model(&model.value,updates,serde_json::json!({"kind":"gen3-continuation","parent":self.parent_sha256,"rules":self.rules}));
        a.policy = MicroArtifact::new(
            &model.policy,
            updates,
            serde_json::json!({"kind":"gen3-policy","value_head":"compact64","rules":self.rules}),
        );
        a
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        self.model()?;
        atomic(path, self)
    }
}
pub fn bootstrap(parent: &Path, out: &Path, memory: Option<&Path>) -> Result<()> {
    let compact = load_model(parent)?;
    let mut model = Gen32Model::from_gen31(compact.model()?, 32001);
    let mut manifest = None;
    let mut manifest_hash = None;
    if let Some(directory) = memory {
        let path = directory.join("summary.json");
        let bytes = fs::read(&path)?;
        let meta: serde_json::Value = serde_json::from_slice(&bytes)?;
        if meta["rules"] != RULES.as_str() {
            return Err(invalid("memory must be rebuilt under Gen3 V2 rules"));
        }
        let spec = SequenceMemorySpec {
            path: directory.join("memory.bin").to_string_lossy().into_owned(),
            sha256: meta["sha256"]
                .as_str()
                .ok_or_else(|| invalid("memory hash missing"))?
                .into(),
        };
        model = model.with_memory(crate::micro_learning::load_sequence_memory(&spec)?);
        manifest = Some(path);
        manifest_hash = Some(sha256(&bytes));
    }
    Artifact {
        schema: "paisho-gen3-policy-memory-v1".into(),
        generation: "3.2".into(),
        rules: RULES.to_string(),
        policy: MicroArtifact::new(
            &model.policy,
            compact.training_steps,
            serde_json::json!({"neutral":true}),
        ),
        updates: compact.training_steps,
        compact,
        parent_sha256: sha256(&fs::read(parent)?),
        memory_manifest: manifest,
        memory_manifest_sha256: manifest_hash,
    }
    .save(out)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub model: PathBuf,
    pub output: PathBuf,
    pub seconds: f64,
    pub threads: usize,
    pub actors: usize,
    pub parallel_candidates: bool,
    pub fixed_actor_budgets: bool,
    pub archive_workers: usize,
    pub games: usize,
    pub budgets: Vec<usize>,
    pub caps: Vec<f64>,
    pub decisions: usize,
    pub samples: usize,
    pub rate: f64,
    pub lambda: f64,
    pub replay_capacity: usize,
    pub replay_ratio: usize,
    pub replay_max_bytes: usize,
    pub checkpoint_seconds: f64,
    pub seed: u64,
    pub replay_index: Option<PathBuf>,
    pub learn: bool,
    pub historical_reference: Option<PathBuf>,
    pub historical_reference_sha256: Option<String>,
    pub historical_every: usize,
    pub solver: bool,
    pub tactical_positions: usize,
    pub correction_replay: bool,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            output: PathBuf::new(),
            seconds: 3600.,
            threads: 10,
            actors: 10,
            parallel_candidates: false,
            fixed_actor_budgets: true,
            archive_workers: 2,
            games: 10_000_000,
            budgets: vec![512, 256, 128, 64, 32, 512],
            caps: vec![58., 34., 25., 17.5, 8., 58.],
            decisions: 2048,
            samples: 128,
            rate: 0.01,
            lambda: 0.5,
            replay_capacity: 65536,
            replay_ratio: 4,
            replay_max_bytes: 32 * 1024 * 1024 * 1024,
            checkpoint_seconds: 30.,
            seed: 32001,
            replay_index: None,
            learn: true,
            historical_reference: None,
            historical_reference_sha256: None,
            historical_every: 2,
            solver: false,
            tactical_positions: 0,
            correction_replay: false,
        }
    }
}
impl Options {
    pub(super) fn historical_game(&self, actor: usize, ordinal: usize) -> bool {
        (ordinal + actor) % self.historical_every == self.historical_every - 1
    }
    fn validate(&self) -> Result<()> {
        if self.tactical_positions > 100_000
            || self.historical_every < 2
            || self.historical_reference.is_some() != self.historical_reference_sha256.is_some()
            || self.threads == 0
            || self.threads > std::thread::available_parallelism()?.get()
            || self.actors == 0
            || self.actors
                > if self.parallel_candidates {
                    self.threads * 2
                } else {
                    self.threads
                }
            || !(1..=4).contains(&self.archive_workers)
            || !self.seconds.is_finite()
            || !(0.0..=86400.).contains(&self.seconds)
            || self.seconds == 0.
            || self.games == 0
            || self.budgets.is_empty()
            || self.budgets.len() != self.caps.len()
            || self
                .budgets
                .iter()
                .any(|b| ![8, 32, 64, 128, 256, 512].contains(b))
            || self.caps.iter().any(|t| !t.is_finite() || *t <= 0.)
            || self.decisions == 0
            || self.samples == 0
            || self.samples > 2048
            || self.replay_capacity == 0
            || self.replay_max_bytes == 0
            || self.replay_ratio > 16
            || !self.rate.is_finite()
            || self.rate <= 0.
            || !self.lambda.is_finite()
            || !(0.0..=1.0).contains(&self.lambda)
            || !self.checkpoint_seconds.is_finite()
            || self.checkpoint_seconds <= 0.
        {
            return Err(invalid("invalid Gen3 continuation options"));
        }
        Ok(())
    }
}
pub(super) fn atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp)?;
    f.write_all(&serde_json::to_vec(value)?)?;
    f.sync_all()?;
    fs::rename(tmp, path)?;
    Ok(())
}
/// Preserve the historical post-selfplay human value fit, including epoch zero.
pub fn human_fit(model: &Path, dataset: &Path, out: &Path) -> Result<()> {
    use crate::compact_learning::{fit_dataset, load_dataset, FitOptions};
    let mut artifact = Artifact::load(model)?;
    let data = load_dataset(dataset)?;
    let (compact, report) = fit_dataset(
        &data,
        &artifact.compact,
        FitOptions {
            epochs: 10000,
            patience: 200,
            learning_rate: 0.1,
            l2: 0.,
            seed: 1,
        },
    )?;
    artifact.updates += report.selected_updates;
    artifact.compact = compact;
    fs::create_dir(out)?;
    artifact.save(&out.join("model.json"))?;
    atomic(&out.join("report.json"), &report)
}

#[cfg(test)]
mod schedule_tests {
    use super::*;
    #[test]
    fn ten_percent_per_actor_and_balanced_seats_across_twenty_games() {
        let o = Options {
            historical_every: 10,
            ..Default::default()
        };
        for actor in 0..10 {
            let ordinals: Vec<_> = (0..20).filter(|n| o.historical_game(actor, *n)).collect();
            assert_eq!(ordinals.len(), 2);
            assert_eq!((ordinals[0] + actor) / 10 % 2, 0);
            assert_eq!((ordinals[1] + actor) / 10 % 2, 1);
        }
        assert_eq!((0..10).filter(|a| o.historical_game(*a, 0)).count(), 1);
    }
}
