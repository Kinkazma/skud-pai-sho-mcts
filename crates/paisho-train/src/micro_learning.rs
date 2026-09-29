//! Gen4 artifact and bounded CPU learning. Gen1–3 artifacts stay unchanged.
use crate::compact_learning::{invalid, save_json_new, sha256};
use paisho_ai::*;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
mod compare;
mod sequence;
pub use sequence::*;
pub mod gen5;
mod history;
mod human;
mod selfplay;
mod tactics;
pub use compare::*;
pub use human::*;
pub use selfplay::*;
pub use tactics::TacticalEvidence;
pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicroArtifact {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_memory: Option<SequenceMemorySpec>,
    pub schema: String,
    pub feature_schema: String,
    pub parameters: Vec<f64>,
    pub updates: u64,
    pub provenance: serde_json::Value,
}
impl MicroArtifact {
    pub fn new(model: &MicroModel, updates: u64, provenance: serde_json::Value) -> Self {
        Self {
            sequence_memory: model.sequence_memory().map(|b| b.spec.clone()),
            schema: model.schema().into(),
            feature_schema: MICRO_FEATURE_SCHEMA.into(),
            parameters: model.parameters().to_vec(),
            updates,
            provenance,
        }
    }
    pub fn model(&self) -> Result<MicroModel> {
        let model = MicroModel::from_parameters(self.parameters.clone()).map_err(invalid)?;
        if self.schema != model.schema() || self.feature_schema != MICRO_FEATURE_SCHEMA {
            return Err(invalid("Gen4 model/feature schema mismatch"));
        }
        match &self.sequence_memory {
            Some(spec) if self.schema == MICRO_MEMORY_MODEL_SCHEMA => Ok(model.with_sequence_memory(load_sequence_memory(spec)?)),
            None if self.schema != MICRO_MEMORY_MODEL_SCHEMA => Ok(model),
            _ => Err(invalid("micro sequence memory dependency mismatch")),
        }
    }
    pub fn load(path: &Path) -> Result<Self> {
        let a: Self = serde_json::from_slice(&fs::read(path)?)?;
        a.model()?;
        Ok(a)
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        self.model()?;
        save_json_new(path, self)
    }
    pub fn identity(&self) -> String {
        sha256(&serde_json::to_vec(self).expect("validated artifact serializes"))
    }
}

/// Durable compact targets; legal action features and visits remain aligned.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SavedMicroExample {
    #[serde(default = "crate::compact_learning::legacy_training_rules")]
    pub rules: String,
    pub source_run: String,
    pub game_id: String,
    /// One-based index of the recorded decision (PSR action number).
    pub decision: usize,
    pub collector: String,
    pub budget: usize,
    pub inherited_visits: usize,
    pub new_visits: Vec<usize>,
    /// Optional audit of corrected PUCT targets; raw visits include retained search.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub policy_raw_visits: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub policy_pruned_visits: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tactical: Option<TacticalEvidence>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub correction_priority: bool,
    pub actions: Vec<String>,
    pub state: Vec<f64>,
    pub action_features: Vec<Vec<f64>>,
    pub policy: Vec<f64>,
    pub value: f64,
    pub policy_weight: f64,
    pub reason: String,
}
impl SavedMicroExample {
    pub fn example(&self) -> Result<MicroExample> {
        crate::compact_learning::require_current_training_rules(&self.rules)?;
        self.example_for_rules(paisho_core::RuleProfileId::CURRENT)
    }
    pub fn example_for_rules(&self, rules: paisho_core::RuleProfileId) -> Result<MicroExample> {
        if self.rules != rules.as_str() {
            return Err(invalid("micro example rules mismatch"));
        }
        if let Some(t) = &self.tactical {
            t.validate(self)?;
        }
        if self.correction_priority && !self.tactical.as_ref().is_some_and(|t| t.informative()) {
            return Err(invalid(
                "correction priority requires informative tactical evidence",
            ));
        }
        if self.source_run.is_empty()
            || self.game_id.is_empty()
            || self.collector.len() != 64
            || self.actions.len() != self.action_features.len()
            || (!self.new_visits.is_empty() && self.new_visits.len() != self.actions.len())
        {
            return Err(invalid(
                "invalid micro example provenance or action alignment",
            ));
        }
        if !self.policy_raw_visits.is_empty() || !self.policy_pruned_visits.is_empty() {
            if self.policy_raw_visits.len() != self.policy.len()
                || self.policy_pruned_visits.len() != self.policy.len()
                || self
                    .policy_raw_visits
                    .iter()
                    .zip(&self.policy_pruned_visits)
                    .any(|(a, b)| b > a)
            {
                return Err(invalid("invalid corrected policy visit provenance"));
            }
            let counts: Vec<_> = self
                .policy_raw_visits
                .iter()
                .zip(&self.policy_pruned_visits)
                .map(|(a, b)| a - b)
                .collect();
            let total: usize = counts.iter().sum();
            if total == 0
                || self
                    .policy
                    .iter()
                    .zip(counts)
                    .any(|(p, n)| (*p - n as f64 / total as f64).abs() > 1e-10)
            {
                return Err(invalid("corrected policy does not match archived visits"));
            }
        }
        let state = self
            .state
            .clone()
            .try_into()
            .map_err(|_| invalid("micro state requires 128 inputs"))?;
        let actions = self
            .action_features
            .iter()
            .map(|x| {
                x.clone()
                    .try_into()
                    .map_err(|_| invalid("micro action requires 32 inputs"))
            })
            .collect::<Result<Vec<_>>>()?;
        let ex = MicroExample {
            sequence_source: sequence_source(&format!("{}/{}", self.source_run, self.game_id)),
            state,
            actions,
            policy: self.policy.clone(),
            value: self.value,
            policy_weight: self.policy_weight,
        };
        ex.validate().map_err(invalid)?;
        Ok(ex)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn artifact_schema_roundtrip_and_legacy_rejection() {
        let a = MicroArtifact::new(&MicroModel::seeded(4), 7, serde_json::json!({"test":true}));
        let loaded: MicroArtifact =
            serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        assert_eq!(a.identity(), loaded.identity());
        assert_eq!(a.model().unwrap(), loaded.model().unwrap());
        let mut wrong = loaded;
        wrong.feature_schema = "paisho-compact-value-features-v1".into();
        assert!(wrong.model().is_err());
    }
    #[test]
    fn target_alignment_and_normalization_are_checked_on_resume() {
        let s = SavedMicroExample {
            tactical: None,
            correction_priority: false,
            policy_raw_visits: vec![],
            policy_pruned_visits: vec![],
            rules: paisho_core::RuleProfileId::CURRENT.to_string(),
            source_run: "run".into(),
            game_id: "1".into(),
            decision: 7,
            collector: "a".repeat(64),
            budget: 256,
            inherited_visits: 0,
            new_visits: vec![256],
            actions: vec!["a".into()],
            state: vec![0.0; 128],
            action_features: vec![vec![0.0; 32]],
            policy: vec![1.0],
            value: 1.0,
            policy_weight: 1.0,
            reason: "rules-terminal-q-mix".into(),
        };
        assert!(s.example().is_ok());
        let legacy_json = serde_json::to_value(&s).unwrap();
        assert!(legacy_json.get("policy_raw_visits").is_none());
        let mut corrected = s.clone();
        corrected.actions = vec!["a".into(), "b".into()];
        corrected.action_features = vec![vec![0.0; 32]; 2];
        corrected.new_visits = vec![80, 20];
        corrected.policy_raw_visits = vec![80, 20];
        corrected.policy_pruned_visits = vec![0, 10];
        corrected.policy = vec![8.0 / 9.0, 1.0 / 9.0];
        let restored: SavedMicroExample =
            serde_json::from_value(serde_json::to_value(&corrected).unwrap()).unwrap();
        assert!(restored.example().is_ok());
        corrected.policy = vec![0.8, 0.2];
        assert!(corrected
            .example()
            .unwrap_err()
            .to_string()
            .contains("archived visits"));
        corrected.policy_pruned_visits[1] = 21;
        assert!(corrected.example().is_err());
        let mut historical = serde_json::to_value(&s).unwrap();
        historical.as_object_mut().unwrap().remove("rules");
        let historical: SavedMicroExample = serde_json::from_value(historical).unwrap();
        assert_eq!(
            historical.rules,
            paisho_core::RuleProfileId::SkudPaiSho2022.as_str()
        );
        assert!(historical
            .example()
            .unwrap_err()
            .to_string()
            .contains("current training requires"));
        let mut bad = s.clone();
        bad.actions.clear();
        assert!(bad.example().is_err());
        let mut bad = s;
        bad.policy[0] = 0.5;
        assert!(bad.example().is_err());
    }
}

/// Gzip level 1 archives every target losslessly without a second JSON buffer.
/// Plain JSON remains readable for all previously saved replay snapshots.
fn save_examples_new(path: &Path, examples: &[SavedMicroExample]) -> Result<()> {
    save_examples_batch(path,examples,false)
}
fn save_examples_batch(path: &Path, examples: &[SavedMicroExample], pending_commit:bool) -> Result<()> {
    use std::io::Write;
    let file = std::io::BufWriter::new(
        fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)?,
    );
    let mut file = if path.extension().is_some_and(|e| e == "gz") {
        let gzip = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        // JSON emits tiny tokens. Buffer BEFORE compression as well as before
        // the file, avoiding a deflate call for every number/punctuation token.
        let mut encoded = std::io::BufWriter::with_capacity(256 * 1024, gzip);
        serde_json::to_writer(&mut encoded, examples)?;
        encoded.write_all(b"\n")?;
        encoded.into_inner()?.finish()?
    } else {
        let mut file = file;
        serde_json::to_writer(&mut file, examples)?;
        file.write_all(b"\n")?;
        file
    };
    file.flush()?;
    if pending_commit { paisho_platform::sync_before_batch_commit(file.get_ref())?; }
    else { file.get_ref().sync_all()?; }
    Ok(())
}
pub(crate) fn load_examples(path: &Path) -> Result<Vec<SavedMicroExample>> {
    use std::io::Read;
    let bytes = fs::read(path)?;
    if bytes.starts_with(&[0x1f, 0x8b]) {
        let mut decoded = Vec::new();
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut decoded)?;
        Ok(serde_json::from_slice(&decoded)?)
    } else {
        Ok(serde_json::from_slice(&bytes)?)
    }
}
#[cfg(test)]
mod archive_tests {
    use super::*;
    #[test]
    fn gzip_empty_archive_roundtrips_and_truncation_fails() {
        let p =
            std::env::temp_dir().join(format!("paisho-micro-archive-{}.gz", std::process::id()));
        let _ = fs::remove_file(&p);
        save_examples_new(&p, &[]).unwrap();
        assert!(load_examples(&p).unwrap().is_empty());
        assert!(save_examples_new(&p, &[]).is_err());
        let mut bytes = fs::read(&p).unwrap();
        bytes.truncate(bytes.len() - 4);
        fs::write(&p, bytes).unwrap();
        assert!(load_examples(&p).is_err());
        fs::remove_file(p).unwrap();
    }
}
