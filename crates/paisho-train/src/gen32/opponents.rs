//! Frozen historical generations, budget-specific experts loaded once in RAM.
use super::*;
use std::collections::{BTreeSet, HashMap};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoricalExpert {
    pub generation: String,
    pub budget: usize,
    pub path: PathBuf,
    pub sha256: String,
    pub solver: bool,
}
pub(super) struct LoadedExpert {
    pub spec: HistoricalExpert,
    pub model: Arc<dyn MctsEvaluator + Send>,
}
pub(super) struct HistoricalPool {
    pub entries: Vec<LoadedExpert>,
    names: Vec<String>,
    pub unique_models: usize,
}
impl HistoricalPool {
    pub fn load(o: &Options) -> Result<Self> {
        let mut seen = BTreeSet::new();
        let mut names = vec![];
        let mut models: HashMap<String, Arc<dyn MctsEvaluator + Send>> = HashMap::new();
        let mut entries = vec![];
        if !o.historical_pool.is_empty() && o.historical_reference.is_some() {
            return Err(invalid(
                "historical pool and single reference are exclusive",
            ));
        }
        for spec in &o.historical_pool {
            if !["Gen3.1", "Gen3.2", "Gen3.3", "Gen3.4"].contains(&spec.generation.as_str())
                || !seen.insert((spec.generation.clone(), spec.budget))
            {
                return Err(invalid("invalid/duplicate historical expert"));
            }
            let bytes = fs::read(&spec.path)?;
            if sha256(&bytes) != spec.sha256 {
                return Err(invalid("historical expert hash mismatch"));
            }
            let model = if let Some(m) = models.get(&spec.sha256) {
                m.clone()
            } else {
                let meta: serde_json::Value = serde_json::from_slice(&bytes)?;
                let m: Arc<dyn MctsEvaluator + Send> = if meta["schema"]
                    == "paisho-gen3-policy-memory-v1"
                    || meta["schema"] == "paisho-gen34-value128-memory-v1"
                    || meta["schema"] == GEN3_VALUE_RESIDUAL_SCHEMA
                {
                    Arc::new(Artifact::load(&spec.path)?.model()?)
                } else {
                    let a: ModelArtifact = serde_json::from_slice(&bytes)?;
                    Arc::new(a.model()?)
                };
                models.insert(spec.sha256.clone(), m.clone());
                m
            };
            if !names.contains(&spec.generation) {
                names.push(spec.generation.clone());
            }
            entries.push(LoadedExpert {
                spec: spec.clone(),
                model,
            });
        }
        for name in &names {
            for budget in &o.budgets {
                if !seen.contains(&(name.clone(), *budget)) {
                    return Err(invalid("historical generation missing a configured budget"));
                }
            }
        }
        Ok(Self {
            entries,
            names,
            unique_models: models.len(),
        })
    }
    pub fn select_budget(
        &self,
        actor: usize,
        ordinal: usize,
        budget: usize,
        o: &Options,
    ) -> Option<&LoadedExpert> {
        if self.names.is_empty() || !o.historical_game(actor, ordinal) {
            return None;
        }
        let name = &self.names[((ordinal + actor) / o.historical_every + actor) % self.names.len()];
        self.entries
            .iter()
            .find(|e| &e.spec.generation == name && e.spec.budget == budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn four_generations_share_weights_and_balance_eighty_twenty() {
        let dir = std::env::temp_dir().join(format!("gen35-pool-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("reference.json");
        let raw = serde_json::to_vec(&ModelArtifact::legacy()).unwrap();
        fs::write(&path, &raw).unwrap();
        let mut o = Options {
            historical_every: 5,
            ..Default::default()
        };
        for generation in ["Gen3.1", "Gen3.2", "Gen3.3", "Gen3.4"] {
            for budget in [32, 64, 128, 256, 512] {
                o.historical_pool.push(HistoricalExpert {
                    generation: generation.into(),
                    budget,
                    path: path.clone(),
                    sha256: sha256(&raw),
                    solver: true,
                });
            }
        }
        let pool = HistoricalPool::load(&o).unwrap();
        assert_eq!(pool.unique_models, 1);
        assert!(pool
            .entries
            .iter()
            .all(|e| Arc::ptr_eq(&pool.entries[0].model, &e.model)));
        let mut counts = std::collections::BTreeMap::new();
        let mut selfplay = 0;
        for actor in 0..10 {
            for ordinal in 0..1000 {
                if let Some(e) = pool.select_budget(actor, ordinal, 512, &o) {
                    *counts.entry(e.spec.generation.clone()).or_insert(0) += 1;
                } else {
                    selfplay += 1;
                }
            }
        }
        assert_eq!(selfplay, 8000);
        assert_eq!(counts.len(), 4);
        assert!(counts.values().all(|n| *n == 500));
        o.historical_pool.pop();
        assert!(HistoricalPool::load(&o).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
