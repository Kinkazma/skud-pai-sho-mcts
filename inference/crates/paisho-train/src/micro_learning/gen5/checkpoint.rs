//! Periodic consistent recovery point, independent of per-step RAM publication.
use super::*;
pub(super) struct Checkpoint {
    seconds: f64,
    sequence: usize,
    last: Option<Instant>,
    pub version: u64,
    previous_index: Option<PathBuf>,
    keep: usize,
    models: checkpoint_retention::Retention,
    pruned: usize,
}
impl Checkpoint {
    pub fn new(seconds: f64) -> Self {
        Self {
            seconds,
            sequence: 0,
            last: None,
            version: 0,
            previous_index: None,
            keep: usize::MAX,
            models: Default::default(),
            pruned: 0,
        }
    }
    pub fn with_retention(seconds: f64, keep: usize) -> Self {
        Self {
            keep,
            ..Self::new(seconds)
        }
    }
    pub fn observe(&mut self, game: &collector::Played, specs: &[OpponentSpec]) {
        if self.seconds > 0.0 {
            self.models.observe(game, specs);
        }
    }
    pub fn retained(&self) -> usize {
        self.models.entries.len()
    }
    pub fn pruned(&self) -> usize {
        self.pruned
    }
    pub fn due(&self) -> bool {
        self.last
            .map_or(true, |t| paisho_platform::training_time::elapsed(t).as_secs_f64() >= self.seconds)
    }
    fn save_candidates(&mut self, out: &Path, recovery: &Path) -> Result<Vec<usize>> {
        let mut pending = self.models.take_candidates();
        pending.retain_mut(|p| {
            if let Some(e) = self
                .models
                .entries
                .iter_mut()
                .find(|e| e.version == p.entry.version && e.identity == p.entry.identity)
            {
                // The current recovery model may have been saved just above.
                // Preserve its existing path, including older unsuffixed paths.
                e.results.extend(p.entry.results.clone());
                return false;
            }
            // Several actor artifacts can share a learner update version. Their
            // identities, not a growing list of provenance kinds, distinguish
            // the exact weights that earned these passive observations. This
            // also works for snapshots whose artifact is only present on disk.
            p.entry.path = out.join("models").join(format!(
                "model-{:07}-{}.json", p.entry.version, p.entry.identity
            ));
            // An existing file not owned by this writer is never adopted/pruned.
            if p.entry.path.exists() {
                return false;
            }
            self.models.entries.push(p.entry.clone());
            true
        });
        let selected: std::collections::BTreeSet<_> = self
            .models
            .select(self.keep, recovery)
            .iter()
            .map(|&i| self.models.entries[i].path.clone())
            .collect();
        for p in &pending {
            if selected.contains(&p.entry.path) {
                if let Some(artifact) = &p.snapshot.artifact {
                    artifact.save(&p.entry.path)?;
                } else {
                    fs::copy(&p.snapshot.path, &p.entry.path)?;
                    fs::File::open(&p.entry.path)?.sync_all()?;
                }
            }
        }
        // Rejected RAM candidates are never serialized, and are not disk deletions.
        self.models.entries.retain(|e| {
            selected.contains(&e.path) || !pending.iter().any(|p| p.entry.path == e.path)
        });
        Ok(self
            .models
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| selected.contains(&e.path).then_some(i))
            .collect())
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
            self.models.add(model_path.clone(), snapshot)?;
        }
        let chosen = self.save_candidates(out, &model_path)?;
        let index = out.join(format!(
            "replay-checkpoint-{:07}-{:07}.index.json",
            snapshot.version, self.sequence
        ));
        self.sequence += 1;
        memory.save(&index)?;
        if let Some(bank) = snapshot.model.sequence_memory() {
            durable::write_pending(
                &out.join("sequence-usage.json"),
                &serde_json::json!({"bank":bank.spec,"query_blocks":8192,"queries":bank.telemetry()[0],"rows":bank.recent_usage()}),
            )?;
            progress["sequence_memory"] = serde_json::json!({"games":bank.games,"human_games":bank.human_games,"segments":bank.entries.len(),"sha256":bank.spec.sha256,"queries":bank.telemetry()});
        }
        progress["checkpoint_model"] = model_path.to_string_lossy().into_owned().into();
        progress["checkpoint_replay_index"] = index.to_string_lossy().into_owned().into();
        progress["durable_version"] = snapshot.version.into();
        progress["checkpoint_seconds"] = self.seconds.into();
        progress["checkpoint_models_retained"] = self.models.entries.len().min(self.keep).into();
        progress["checkpoint_models_pruned"] =
            (self.pruned + self.models.entries.len().saturating_sub(self.keep)).into();
        durable::write(&out.join("durable-progress.json"), progress)?;
        // Only files created by this checkpoint writer are eligible. Parent,
        // references and recovery inputs from earlier campaigns are untouched.
        durable::write(
            &out.join("checkpoint-retention.json"),
            &self.models.metadata(&chosen, self.keep),
        )?;
        let mut retained = vec![];
        for (i, entry) in self.models.entries.drain(..).enumerate() {
            if chosen.contains(&i) {
                retained.push(entry);
            } else {
                fs::remove_file(entry.path)?;
                self.pruned += 1;
            }
        }
        self.models.entries = retained;
        progress["checkpoint_models_retained"] = self.models.entries.len().into();
        progress["checkpoint_models_pruned"] = self.pruned.into();
        if let Some(old) = self.previous_index.replace(index.clone()) {
            if old != index {
                fs::remove_file(old)?;
            }
        }
        self.version = snapshot.version;
        self.last = Some(paisho_platform::training_time::now());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn actors_and_learner_with_same_version_keep_checkpoints_by_identity() {
        let out = std::env::temp_dir().join(format!(
            "gen5-actor-learner-collision-{}",
            std::process::id()
        ));
        fs::create_dir_all(out.join("games")).unwrap();
        fs::create_dir_all(out.join("models")).unwrap();
        let make = |seed, kind: &str| {
            let model = Arc::new(MicroModel::seeded(seed));
            let artifact = Arc::new(MicroArtifact::new(
                &model,
                7,
                serde_json::json!({"kind":kind}),
            ));
            Arc::new(Snapshot {
                version: 7,
                identity: artifact.identity(),
                model,
                artifact: Some(artifact),
                path: "unused".into(),
            })
        };
        let mut actors: Vec<_> = [
            "gen5-guarded-interpolation",
            "gen5-transactional-policy-repair-v1",
            "gen5-policy-relay-v1",
            "unrecognized-provenance-kind",
        ]
        .iter()
        .enumerate()
        .map(|(i, kind)| make(i as u64 + 1, kind))
        .collect();
        // Disk-only snapshots must receive the same identity-based treatment.
        let source = out.join("disk-only-actor.json");
        actors[3].artifact.as_ref().unwrap().save(&source).unwrap();
        actors[3] = Arc::new(Snapshot {
            artifact: None,
            path: source,
            version: actors[3].version,
            identity: actors[3].identity.clone(),
            model: actors[3].model.clone(),
        });
        let learner = make(5, "learner");
        let mut checkpoint = Checkpoint::with_retention(30., 8);
        for actor in &actors {
            checkpoint
                .models
                .observe_result(actor, "reference".into(), "case", 0, true);
        }
        checkpoint
            .models
            .observe_result(&learner, "reference".into(), "learner-case", 1, true);
        let memory = memory::Memory::new(&Options::default());
        let mut progress = serde_json::json!({});
        checkpoint
            .commit(&out, &learner, &memory, &mut progress)
            .unwrap();
        assert_eq!(checkpoint.models.entries.len(), actors.len() + 1);
        for e in &checkpoint.models.entries {
            assert_eq!(MicroArtifact::load(&e.path).unwrap().identity(), e.identity);
            assert!(!e.results.is_empty());
        }
        for actor in &actors {
            let entry = checkpoint.models.entries.iter()
                .find(|e| e.identity == actor.identity).unwrap();
            assert_eq!(entry.path, out.join("models")
                .join(format!("model-0000007-{}.json", actor.identity)));
        }
        assert_eq!(Path::new(progress["checkpoint_model"].as_str().unwrap()),
            out.join("models/model-0000007.json"));
        assert_eq!(
            MicroArtifact::load(Path::new(progress["checkpoint_model"].as_str().unwrap()))
                .unwrap()
                .identity(),
            learner.identity
        );
        // Another observation updates the same identity without duplicating or
        // renaming either the suffixed actors or the legacy recovery path.
        checkpoint.models.observe_result(&actors[0], "reference".into(), "later", 1, true);
        checkpoint.commit(&out, &learner, &memory, &mut progress).unwrap();
        assert_eq!(checkpoint.models.entries.len(), actors.len() + 1);
        let entry = checkpoint.models.entries.iter()
            .find(|e| e.identity == actors[0].identity).unwrap();
        assert_eq!(serde_json::to_value(&entry.results).unwrap()["reference"]["wins"],
            serde_json::json!([1, 1]));
        fs::remove_dir_all(out).unwrap();
    }

    use super::*;
    #[test]
    fn preexisting_recovery_is_not_adopted_or_deleted() {
        let out =
            std::env::temp_dir().join(format!("gen5-unowned-recovery-{}", std::process::id()));
        fs::create_dir_all(out.join("games")).unwrap();
        fs::create_dir_all(out.join("models")).unwrap();
        let model = Arc::new(MicroModel::seeded(17));
        let artifact = Arc::new(MicroArtifact::new(
            &model,
            1,
            serde_json::json!({"test":true}),
        ));
        let path = out.join("models/model-0000001.json");
        artifact.save(&path).unwrap();
        let before = fs::read(&path).unwrap();
        let snapshot = Snapshot {
            version: 1,
            identity: artifact.identity(),
            model,
            artifact: Some(artifact),
            path: path.clone(),
        };
        let mut writer = Checkpoint::with_retention(30., 64);
        let mut progress = serde_json::json!({"version":1});
        writer
            .commit(
                &out,
                &snapshot,
                &memory::Memory::new(&Options::default()),
                &mut progress,
            )
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), before);
        assert_eq!(writer.retained(), 0);
        fs::remove_dir_all(out).unwrap();
    }
    #[test]
    fn measured_collector_is_saved_exactly_without_replacing_recovery_weights() {
        let out =
            std::env::temp_dir().join(format!("gen5-measured-retention-{}", std::process::id()));
        fs::create_dir_all(out.join("games")).unwrap();
        fs::create_dir_all(out.join("models")).unwrap();
        let make = |version, seed| {
            let model = Arc::new(MicroModel::seeded(seed));
            let artifact = Arc::new(MicroArtifact::new(
                &model,
                version,
                serde_json::json!({"test":true}),
            ));
            Arc::new(Snapshot {
                version,
                identity: artifact.identity(),
                model,
                artifact: Some(artifact),
                path: out.join("not-a-file"),
            })
        };
        let measured = make(7, 17);
        let current = make(10, 18);
        let identity_before = current.identity.clone();
        let memory = memory::Memory::new(&Options::default());
        let mut checkpoint = Checkpoint::with_retention(30., 8);
        checkpoint
            .models
            .observe_result(&measured, "ref:MCTS8".into(), "case", 0, true);
        let mut progress = serde_json::json!({"version":10,"updates":10});
        checkpoint
            .commit(&out, &current, &memory, &mut progress)
            .unwrap();
        let retained = &checkpoint.models.entries;
        assert_eq!(retained.len(), 2);
        let old = retained.iter().find(|e| e.version == 7).unwrap();
        assert_eq!(
            MicroArtifact::load(&old.path).unwrap().identity(),
            measured.identity
        );
        let current_file = Path::new(progress["checkpoint_model"].as_str().unwrap());
        assert_eq!(
            MicroArtifact::load(current_file).unwrap().identity(),
            identity_before
        );
        assert!(retained
            .iter()
            .find(|e| e.version == 10)
            .unwrap()
            .results
            .is_empty());
        assert_eq!(
            serde_json::to_value(&old.results).unwrap()["ref:MCTS8"]["wins"],
            serde_json::json!([1, 0])
        );
        // Same collector observed before its regular save: merge the measurement.
        let next = make(11, 19);
        checkpoint
            .models
            .observe_result(&next, "ref:MCTS8".into(), "other", 1, true);
        progress["version"] = 11.into();
        checkpoint
            .commit(&out, &next, &memory, &mut progress)
            .unwrap();
        let retained_next: Vec<_> = checkpoint
            .models
            .entries
            .iter()
            .filter(|e| e.version == 11)
            .collect();
        assert_eq!(retained_next.len(), 1);
        assert!(!retained_next[0].results.is_empty());
        assert_eq!(current.identity, identity_before);
        fs::remove_dir_all(out).unwrap();
    }
    #[test]
    fn retention_only_prunes_owned_models_after_committing_the_latest() {
        let out = std::env::temp_dir().join(format!("gen5-retention-{}", std::process::id()));
        fs::create_dir_all(out.join("games")).unwrap();
        fs::create_dir_all(out.join("models")).unwrap();
        fs::write(out.join("models/unowned.json"), b"external").unwrap();
        let model = MicroModel::seeded(17);
        let memory = memory::Memory::new(&Options::default());
        let mut writer = Checkpoint::with_retention(30., 2);
        for version in 1..=5 {
            let a = MicroArtifact::new(&model, version, serde_json::json!({"test":true}));
            let snapshot = Snapshot {
                artifact: Some(Arc::new(a.clone())),
                version,
                identity: a.identity(),
                model: Arc::new(model.clone()),
                path: out.join("unused"),
            };
            let mut progress = serde_json::json!({"version":version,"updates":version});
            writer
                .commit(&out, &snapshot, &memory, &mut progress)
                .unwrap();
            let durable: serde_json::Value =
                serde_json::from_slice(&fs::read(out.join("durable-progress.json")).unwrap())
                    .unwrap();
            assert_eq!(durable["version"], version);
            assert!(Path::new(durable["checkpoint_model"].as_str().unwrap()).exists());
            assert_eq!(writer.retained(), (version as usize).min(2));
        }
        assert_eq!(writer.pruned(), 3);
        assert!(out.join("models/unowned.json").exists());
        assert!(!out.join("models/model-0000003.json").exists());
        assert!(out.join("models/model-0000004.json").exists());
        fs::remove_dir_all(out).unwrap();
    }
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
