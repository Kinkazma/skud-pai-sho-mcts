//! Immutable correction files survive ordinary FIFO eviction. A bounded RAM
//! cache serves rehearsal; deterministic cold rotation revisits the entire bank.
use super::*;
#[derive(Default)]
pub(super) struct Lessons {
    pub catalog: Vec<Row>,
    cache: VecDeque<Entry>,
    bytes: usize,
    pub cursor: usize,
    pub draws: usize,
    pub reread: usize,
}
impl Lessons {
    fn load(&mut self, row: Row) -> Result<()> {
        let raw = fs::read(&row.path)?;
        if sha256(&raw) != row.sha256 {
            return Err(invalid("correction hash mismatch"));
        }
        let saved = crate::micro_learning::load_examples(&row.path)?;
        let s = saved
            .get(row.index)
            .ok_or_else(|| invalid("correction index"))?;
        if !s.correction_priority {
            return Err(invalid("non-correction in durable bank"));
        }
        let example = Arc::new(s.example()?);
        self.cache(row, example);
        Ok(())
    }
    fn cache(&mut self, row: Row, example: Arc<MicroExample>) {
        let bytes = size(&example);
        // One unusually wide position may exceed the RAM cache; keep it on disk.
        if bytes > 64 * 1024 * 1024 {
            return;
        }
        self.bytes += bytes;
        self.cache.push_back(Entry {
            row,
            example,
            bytes,
        });
        while self.cache.len() > 512 || self.bytes > 64 * 1024 * 1024 {
            self.bytes -= self.cache.pop_front().unwrap().bytes;
        }
    }
    pub fn restore(v: &serde_json::Value) -> Result<Self> {
        let mut bank = Self::default();
        if let Some(rows) = v.get("corrections") {
            bank.catalog = serde_json::from_value(rows.clone())?;
        }
        bank.cursor = v["correction_cursor"].as_u64().unwrap_or(0) as usize;
        bank.draws = v["correction_draws"].as_u64().unwrap_or(0) as usize;
        bank.reread = v["correction_reread"].as_u64().unwrap_or(0) as usize;
        for _ in 0..bank.catalog.len().min(64) {
            bank.rotate()?;
        }
        Ok(bank)
    }
    fn rotate(&mut self) -> Result<()> {
        if !self.catalog.is_empty() {
            let row = self.catalog[self.cursor % self.catalog.len()].clone();
            self.cursor = (self.cursor + 1) % self.catalog.len();
            self.load(row)?;
        }
        Ok(())
    }
    #[cfg(test)]
    pub fn add(&mut self, rows: Vec<Row>) -> Result<()> {
        for row in rows {
            self.load(row.clone())?;
            self.catalog.push(row);
        }
        Ok(())
    }
    pub fn add_saved(&mut self, rows: Vec<Row>, saved: &[SavedMicroExample]) -> Result<()> {
        let corrections: Vec<_> = saved
            .iter()
            .filter(|s| s.correction_priority)
            .take(4)
            .collect();
        if rows.len() != corrections.len() {
            return Err(invalid("correction archive count mismatch"));
        }
        for (row, s) in rows.into_iter().zip(corrections) {
            // The archive worker has just hash-checked and synced these exact
            // bytes; reuse the already-validated in-memory target here.
            self.cache(row.clone(), Arc::new(s.example()?));
            self.catalog.push(row);
        }
        Ok(())
    }
    /// Replace, rather than add to, 10% of ordinary replay draws.
    pub fn sample(&mut self, rng: &mut StableRng) -> Result<Option<Arc<MicroExample>>> {
        self.draws += 1;
        if self.draws % 1000 == 0 {
            self.rotate()?;
        }
        if self.draws % 10 != 0 || self.cache.is_empty() {
            return Ok(None);
        }
        self.reread += 1;
        Ok(Some(
            self.cache[rng.index(self.cache.len())].example.clone(),
        ))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn durable_lessons_survive_fifo_and_are_rehearsed_after_restore() {
        let dir = std::env::temp_dir().join(format!("gen32-lessons-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lessons.json.gz");
        let mut s = SavedMicroExample { structured: Vec::new(), evidence: None,
            rules: RULES.to_string(),
            source_run: "test".into(),
            game_id: "1".into(),
            decision: 1,
            collector: "a".repeat(64),
            budget: 64,
            inherited_visits: 0,
            new_visits: vec![1, 99],
            policy_raw_visits: vec![],
            policy_pruned_visits: vec![],
            actions: vec!["a".into(), "b".into()],
            state: vec![0.; 128],
            action_features: vec![vec![0.; 32]; 2],
            policy: vec![1., 0.],
            value: 1.,
            policy_weight: 1.,
            reason: "search-proven-value".into(),
            correction_priority: true,
            tactical: Some(crate::micro_learning::TacticalEvidence {
                schema: "paisho-mcts-proof-v1".into(),
                root_value: Some(1),
                action_values: vec![Some(1), None],
                network_value: 0.,
                network_best_action: Some(1),
            }),
        };
        s.rules = RULES.to_string();
        s.correction_priority = true;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        serde_json::to_writer(&mut encoder, &vec![s]).unwrap();
        let raw = encoder.finish().unwrap();
        fs::write(&path, &raw).unwrap();
        let mut bank = Lessons::default();
        bank.add(vec![Row {
            path: path.clone(),
            sha256: sha256(&raw),
            index: 0,
        }])
        .unwrap();
        let v = serde_json::json!({"corrections":bank.catalog,"correction_cursor":bank.cursor});
        let mut restored = Lessons::restore(&v).unwrap();
        let mut rng = StableRng::new(12);
        let mut used = 0;
        for _ in 0..2000 {
            if restored.sample(&mut rng).unwrap().is_some() {
                used += 1;
            }
        }
        assert_eq!(used, 200);
        assert_eq!(restored.reread, 200);
        assert_eq!(restored.catalog.len(), 1);
        fs::write(&path, b"corrupt").unwrap();
        assert!(Lessons::restore(&v).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
