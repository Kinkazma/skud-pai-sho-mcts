//! Periodic consistent recovery point, independent of per-step RAM publication.
use super::*;
pub(super) struct Checkpoint {
    seconds: f64,
    sequence: usize,
    last: Option<Instant>,
    pub version: u64,
    previous_index: Option<PathBuf>,
}
impl Checkpoint {
    pub fn new(seconds: f64) -> Self {
        Self {
            seconds,
            sequence: 0,
            last: None,
            version: 0,
            previous_index: None,
        }
    }
    pub fn due(&self) -> bool {
        self.last
            .map_or(true, |t| t.elapsed().as_secs_f64() >= self.seconds)
    }
    pub fn commit(
        &mut self,
        out: &Path,
        snapshot: &Snapshot,
        memory: &memory::Memory,
        progress: &mut serde_json::Value,
    ) -> Result<()> {
        // Every source referenced by Memory was written before Ready. Flush the
        // batch before atomically exposing the model/index/progress checkpoint.
        fs::File::open(out.join("games"))?.sync_all()?;
        let model_path = out
            .join("models")
            .join(format!("model-{:07}.json", snapshot.version));
        if !model_path.exists() {
            if let Some(artifact) = &snapshot.artifact {
                artifact.save(&model_path)?;
            } else {
                fs::copy(&snapshot.path, &model_path)?;
                fs::File::open(&model_path)?.sync_all()?;
            }
        }
        let index = out.join(format!(
            "replay-checkpoint-{:07}-{:07}.index.json",
            snapshot.version, self.sequence
        ));
        self.sequence += 1;
        memory.save(&index)?;
        if let Some(bank) = snapshot.model.sequence_memory() {
            durable::write_pending(&out.join("sequence-usage.json"), &serde_json::json!({"bank":bank.spec,"query_blocks":8192,"queries":bank.telemetry()[0],"rows":bank.recent_usage()}))?;
            progress["sequence_memory"] = serde_json::json!({"games":bank.games,"human_games":bank.human_games,"segments":bank.entries.len(),"sha256":bank.spec.sha256,"queries":bank.telemetry()});
        }
        progress["checkpoint_model"] = model_path.to_string_lossy().into_owned().into();
        progress["checkpoint_replay_index"] = index.to_string_lossy().into_owned().into();
        progress["durable_version"] = snapshot.version.into();
        progress["checkpoint_seconds"] = self.seconds.into();
        durable::write(&out.join("durable-progress.json"), progress)?;
        if let Some(old) = self.previous_index.replace(index.clone()) {
            if old != index {
                fs::remove_file(old)?;
            }
        }
        self.version = snapshot.version;
        self.last = Some(Instant::now());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ram_weights_do_not_replace_recovery_point_until_commit() {
        let out = std::env::temp_dir().join(format!("gen5-checkpoint-{}", std::process::id()));
        fs::create_dir_all(out.join("games")).unwrap();
        fs::create_dir_all(out.join("models")).unwrap();
        let model = MicroModel::seeded(9001);
        let artifact = MicroArtifact::new(&model, 12, serde_json::json!({"ram":true}));
        let first = Snapshot {
            artifact: Some(Arc::new(artifact.clone())),
            version: 1,
            identity: artifact.identity(),
            model: Arc::new(model.clone()),
            path: out.join("not-saved.json"),
        };
        let memory = memory::Memory::new(&Options::default());
        let mut checkpoint = Checkpoint::new(30.0);
        let mut progress = serde_json::json!({"version":1,"updates":12});
        checkpoint
            .commit(&out, &first, &memory, &mut progress)
            .unwrap();
        assert_eq!(
            MicroArtifact::load(&out.join("models/model-0000001.json"))
                .unwrap()
                .identity(),
            first.identity
        );
        assert!(!first.path.exists());
        let first_index = progress["checkpoint_replay_index"]
            .as_str()
            .unwrap()
            .to_string();
        // Draining an interrupted game can change FIFO without changing weights.
        checkpoint
            .commit(&out, &first, &memory, &mut progress)
            .unwrap();
        assert_ne!(
            progress["checkpoint_replay_index"].as_str().unwrap(),
            first_index
        );
        assert!(!Path::new(&first_index).exists());
        let previous = fs::read(out.join("durable-progress.json")).unwrap();
        let second_artifact = MicroArtifact::new(&model, 13, serde_json::json!({"ram":true}));
        let second = Snapshot {
            artifact: Some(Arc::new(second_artifact.clone())),
            version: 2,
            identity: second_artifact.identity(),
            ..first
        };
        assert_eq!(
            fs::read(out.join("durable-progress.json")).unwrap(),
            previous
        );
        assert!(!out.join("models/model-0000002.json").exists());
        progress["version"] = 2.into();
        progress["updates"] = 13.into();
        checkpoint
            .commit(&out, &second, &memory, &mut progress)
            .unwrap();
        let restored: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("durable-progress.json")).unwrap()).unwrap();
        assert_eq!(restored["version"], 2);
        assert!(Path::new(restored["checkpoint_replay_index"].as_str().unwrap()).exists());
        fs::remove_dir_all(out).unwrap();
    }
}
