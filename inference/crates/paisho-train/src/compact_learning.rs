//! Offline preparation and supervised fitting for the small CPU MCTS value
//! model. Dataset splits are by canonical game identity, never by position.

use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use paisho_ai::{
    CompactValueModel, COMPACT_FEATURE_COUNT, COMPACT_FEATURE_NAMES, COMPACT_VALUE_SCHEMA_V1,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

mod cost;
mod dataset;
mod fitting;

pub use cost::run_cost;
pub use dataset::{
    load_dataset, prepare_dataset, CompactDataset, DatasetExample, DatasetGame, OriginalRecord,
    PrepareOptions,
};
pub use fitting::{fit_dataset, EpochMetrics, FitOptions, FitReport, LossMetrics};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
pub const COMPACT_MODEL_SCHEMA_V1: &str = "paisho-compact-value-model-v1";

/// Historical files predate explicit rules metadata. Never reinterpret them as
/// current-rule training targets merely because the engine default changed.
pub(crate) fn legacy_training_rules() -> String {
    paisho_core::RuleProfileId::SkudPaiSho2022.to_string()
}

pub(crate) fn require_current_training_rules(rules: &str) -> Result<()> {
    if rules != paisho_core::RuleProfileId::CURRENT.as_str() {
        return Err(invalid(format!(
            "training targets use {rules}; current training requires {}. Reinterpret the source PSRs up to their first terminal state and prepare fresh targets; old weights remain usable as a warm start",
            paisho_core::RuleProfileId::CURRENT
        )));
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelArtifact {
    pub schema: String,
    pub feature_schema: String,
    pub feature_names: Vec<String>,
    pub weights: Vec<f64>,
    pub training_steps: u64,
    pub provenance: serde_json::Value,
}

impl ModelArtifact {
    pub fn legacy() -> Self {
        Self {
            schema: COMPACT_MODEL_SCHEMA_V1.into(),
            feature_schema: COMPACT_VALUE_SCHEMA_V1.into(),
            feature_names: COMPACT_FEATURE_NAMES
                .iter()
                .map(|name| (*name).into())
                .collect(),
            weights: CompactValueModel::default().weights().to_vec(),
            training_steps: 0,
            provenance: serde_json::json!({"initialization":"exact-legacy-heuristic-v1"}),
        }
    }

    pub fn model(&self) -> Result<CompactValueModel> {
        if self.schema != COMPACT_MODEL_SCHEMA_V1
            || self.feature_schema != COMPACT_VALUE_SCHEMA_V1
            || self.feature_names != COMPACT_FEATURE_NAMES
        {
            return Err(invalid("compact model schema or feature order mismatch"));
        }
        let weights: [f64; COMPACT_FEATURE_COUNT] = self
            .weights
            .clone()
            .try_into()
            .map_err(|_| invalid("compact model requires exactly 64 coefficients"))?;
        Ok(CompactValueModel::from_weights(weights)?)
    }

    /// `training_steps` is the total count including the parent model's steps.
    pub fn with_model(
        &self,
        model: &CompactValueModel,
        training_steps: u64,
        provenance: serde_json::Value,
    ) -> Self {
        Self {
            weights: model.weights().to_vec(),
            training_steps,
            provenance,
            ..self.clone()
        }
    }
}

pub fn load_model(path: &Path) -> Result<ModelArtifact> {
    let artifact: ModelArtifact = serde_json::from_slice(&fs::read(path)?)?;
    artifact.model()?;
    Ok(artifact)
}

pub fn save_model_new(path: &Path, artifact: &ModelArtifact) -> Result<()> {
    artifact.model()?;
    save_json_new(path, artifact)
}

pub(crate) fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidData, message.into()))
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn save_json_new(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

/// CLI dispatch used by the compact-training executable. These commands only
/// read the explicitly supplied corpus/model and write new local artifacts.
pub fn run_offline(args: &[String]) -> Result<()> {
    let command = args
        .first()
        .ok_or_else(|| invalid("expected prepare or train"))?;
    let mut flags = std::collections::BTreeMap::new();
    for pair in args[1..].chunks(2) {
        if pair.len() != 2
            || !pair[0].starts_with("--")
            || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err(invalid("expected distinct --name value options"));
        }
    }
    let output = PathBuf::from(
        flags
            .remove("--output")
            .ok_or_else(|| invalid("missing --output"))?,
    );
    let seed = flags.remove("--seed").unwrap_or("1").parse()?;
    match command.as_str() {
        "prepare" => {
            let source = PathBuf::from(
                flags
                    .remove("--source")
                    .ok_or_else(|| invalid("missing --source"))?,
            );
            let max_games = flags.remove("--max-games").unwrap_or("1000").parse()?;
            let positions_per_game = flags
                .remove("--positions-per-game")
                .unwrap_or("32")
                .parse()?;
            if !flags.is_empty() {
                return Err(invalid("unknown prepare option"));
            }
            if output.exists() {
                return Err(invalid("dataset output already exists"));
            }
            let dataset = prepare_dataset(
                &source,
                PrepareOptions {
                    max_games,
                    positions_per_game,
                    seed,
                },
            )?;
            save_json_new(&output, &dataset)?;
            println!(
                "{}",
                serde_json::json!({"dataset":output,"games":dataset.games.len(),"train_games":dataset.games.iter().filter(|game| !game.held_out).count(),"held_out_games":dataset.games.iter().filter(|game| game.held_out).count(),"examples":dataset.games.iter().map(|game|game.examples.len()).sum::<usize>(),"extraction_seconds":dataset.extraction_seconds})
            );
        }
        "train" => {
            let dataset_path = PathBuf::from(
                flags
                    .remove("--dataset")
                    .ok_or_else(|| invalid("missing --dataset"))?,
            );
            let parent_path = flags.remove("--model").map(PathBuf::from);
            let epochs = flags.remove("--epochs").unwrap_or("20").parse()?;
            let patience = flags.remove("--patience").unwrap_or("0").parse()?;
            let learning_rate = flags.remove("--learning-rate").unwrap_or("0.1").parse()?;
            let l2 = flags.remove("--l2").unwrap_or("0").parse()?;
            if !flags.is_empty() {
                return Err(invalid("unknown train option"));
            }
            let report_path = output.with_extension("report.json");
            if output.exists() || report_path.exists() {
                return Err(invalid("model or report output already exists"));
            }
            let parent = match &parent_path {
                Some(path) => load_model(path)?,
                None => ModelArtifact::legacy(),
            };
            let dataset = load_dataset(&dataset_path)?;
            let (mut artifact, report) = fit_dataset(
                &dataset,
                &parent,
                FitOptions {
                    epochs,
                    patience,
                    learning_rate,
                    l2,
                    seed,
                },
            )?;
            artifact.provenance["dataset_file"] = serde_json::json!({"path":dataset_path.canonicalize()?,"sha256":sha256(&fs::read(&dataset_path)?)});
            if let Some(path) = parent_path {
                artifact.provenance["parent_model_file"] = serde_json::json!({"path":path.canonicalize()?,"sha256":sha256(&fs::read(path)?)});
            }
            save_model_new(&output, &artifact)?;
            save_json_new(&report_path, &report)?;
            println!(
                "{}",
                serde_json::json!({"model":output,"report":report_path,"selected_epoch":report.selected_epoch,"before":report.before,"after":report.after,"update_seconds":report.update_seconds})
            );
        }
        _ => return Err(invalid("expected prepare or train")),
    }
    Ok(())
}

#[cfg(test)]
mod tests;
