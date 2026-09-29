//! Bounded direct recall of verified wins, independent of ordinary bundle density.
use super::*;
use std::collections::{HashSet, VecDeque};
#[derive(Default)]
pub(super) struct Winning {
    pub policy_only: bool,
    pub policy_support: bool,
    pub focus: Vec<Arc<MicroExample>>,
    pub(super) files: Vec<PathBuf>,
    pending: VecDeque<PathBuf>,
    seen: HashSet<PathBuf>,
    cursor: usize,
    examples: VecDeque<(String, Arc<MicroExample>)>,
    bytes: usize,
    next_refresh: usize,
    draws: usize,
    directory: Option<PathBuf>,
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_recall_reconstructs_verified_wins_and_then_reuses_ram() {
        let dir = std::env::temp_dir().join(format!("gen5-winning-{}", std::process::id()));
        fs::create_dir_all(dir.join("proofs")).unwrap();
        let r: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
        ))
        .parse()
        .unwrap();
        let p = r.replay().unwrap();
        let model = MicroModel::seeded(31).with_spatial_policy();
        let mut session = MicroMctsSession::new(Arc::new(model.clone()));
        session
            .search_with_options(
                &p,
                512,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let cert = session.certificate(10000).unwrap();
        let key = sha256(r.to_string().as_bytes());
        let path = dir.join("proofs").join(format!("{key}.json"));
        fs::write(&path,serde_json::to_vec(&serde_json::json!({"rules":RULES.to_string(),"prefix":r.to_string(),"certificate":cert})).unwrap()).unwrap();
        let mut pool = Winning::default();
        pool.add_root(&dir).unwrap();
        let mut rng = StableRng::new(12);
        let rows = pool.draw(1, &mut rng, &model).unwrap();
        assert_eq!(rows.len(), 1);
        let legal = paisho_core::legal_actions(&p);
        for (a, q) in legal.iter().zip(&rows[0].policy) {
            if *q > 0. {
                let mut n = p.clone();
                n.apply(*a).unwrap();
                assert_eq!(n.outcome(), GameOutcome::Win(p.to_move()));
            }
        }
        fs::remove_file(path).unwrap();
        assert_eq!(pool.draw(10, &mut rng, &model).unwrap().len(), 10);
        fs::remove_dir_all(dir).unwrap();
    }
}
impl Winning {
    pub(super) fn examples_empty(&self) -> bool {
        self.examples.is_empty()
    }
    pub fn register_persisted_proof(&mut self, path: PathBuf) -> bool {
        if self.seen.insert(path.clone()) {
            self.files.push(path.clone());
            if self.pending.len() == 1024 {
                self.pending.pop_front();
            }
            self.pending.push_back(path);
            true
        } else {
            false
        }
    }
    pub fn add_root(&mut self, root: &Path) -> Result<()> {
        let dir = root.join("proofs");
        if self.directory.is_none() {
            self.directory = Some(dir.clone());
        }
        if !dir.exists() {
            return Ok(());
        }
        let mut files = fs::read_dir(dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.sort();
        for p in files {
            if p.extension().is_some_and(|x| x == "json") && self.seen.insert(p.clone()) {
                self.files.push(p);
            }
        }
        Ok(())
    }
    pub fn admit(&mut self, key: String, mut ex: Arc<MicroExample>) {
        if self.policy_only && ex.value_weight != 0. {
            Arc::make_mut(&mut ex).value_weight = 0.;
        }
        if ex.value != 1. || ex.policy_weight <= 0. || ex.policy.is_empty() {
            return;
        }
        if let Some(dir) = &self.directory {
            let path = dir.join(format!("{key}.json"));
            if !self.seen.contains(&path) && path.exists() {
                self.seen.insert(path.clone());
                self.files.push(path);
            }
        }
        let bytes = ex.state.len() * 8 + ex.actions.len() * 256 + ex.policy.len() * 8;
        if bytes > 32 * 1024 * 1024 || self.examples.iter().any(|(k, _)| k == &key) {
            return;
        }
        while !self.examples.is_empty()
            && (self.examples.len() >= 256 || self.bytes + bytes > 32 * 1024 * 1024)
        {
            let (_, old) = self.examples.pop_front().unwrap();
            self.bytes -= old.state.len() * 8 + old.actions.len() * 256 + old.policy.len() * 8;
        }
        self.bytes += bytes;
        self.examples.push_back((key, ex));
    }
    fn load_one(&mut self, model: &MicroModel) -> Result<()> {
        if self.files.is_empty() {
            return Ok(());
        }
        let path = self.files[self.cursor % self.files.len()].clone();
        self.cursor += 1;
        self.load_path(&path, model)
    }
    fn load_path(&mut self, path: &Path, model: &MicroModel) -> Result<()> {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        let text = v["prefix"]
            .as_str()
            .ok_or_else(|| invalid("winning prefix missing"))?;
        let key = sha256(text.as_bytes());
        if path.file_stem().and_then(|s| s.to_str()) != Some(&key) || v["rules"] != RULES.as_str() {
            return Err(invalid("winning proof identity mismatch"));
        }
        let record: GameRecord = text.parse()?;
        let p = record.replay()?;
        if record.rules() != RULES {
            return Err(invalid("winning proof rules mismatch"));
        }
        let cert: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
        cert.verify(&p).map_err(invalid)?;
        let outcome = if p.to_move() == Player::Host { 1 } else { -1 };
        if cert.outcome != outcome {
            return Ok(());
        }
        let actions = paisho_core::legal_actions(&p);
        let mut policy = vec![0.; actions.len()];
        for (a, child) in &cert.children {
            if child.outcome == outcome {
                let action: paisho_core::Action = a.parse()?;
                let i = actions
                    .iter()
                    .position(|a| *a == action)
                    .ok_or_else(|| invalid("winning proof action illegal"))?;
                policy[i] = 1.;
            }
        }
        let n: f64 = policy.iter().sum();
        if n == 0. {
            return Err(invalid("winning proof lacks winning child"));
        }
        for p in &mut policy {
            *p /= n;
        }
        if self.policy_support {policy=action_values::verified_winning_policy(&p,&actions,&cert)?;}
        let mut action_values=certificate_action_values(&p,&actions,&cert)?;
        if self.policy_support {action_values::complete_verified_winning_values(&mut action_values,&policy)?;}
        let ex = Arc::new(MicroExample { policy_support:self.policy_support, action_values,
            value_weight: if self.policy_only { 0. } else { 1. },
            sequence_source: 0,
            state: model.state_features(&p),
            actions: actions
                .iter()
                .map(|a| micro_action_features(&p, *a))
                .collect(),
            policy,
            value: 1.,
            policy_weight: 1.,
        });
        ex.validate().map_err(invalid)?;
        self.admit(key, ex);
        Ok(())
    }
    pub fn draw(
        &mut self,
        n: usize,
        rng: &mut StableRng,
        model: &MicroModel,
    ) -> Result<Vec<Arc<MicroExample>>> {
        if n == 0 {
            return Ok(vec![]);
        }
        for _ in 0..2 {
            if let Some(path) = self.pending.pop_front() {
                self.load_path(&path, model)?;
            } else {
                break;
            }
        }
        if self.examples.is_empty() {
            for _ in 0..32.min(self.files.len()) {
                self.load_one(model)?;
            }
            self.next_refresh = self.draws + 64;
        } else if self.draws >= self.next_refresh {
            for _ in 0..2.min(self.files.len()) {
                self.load_one(model)?;
            }
            self.next_refresh = self.draws + 64;
        }
        if self.examples.is_empty() {
            return Ok(vec![]);
        }
        let out = (0..n)
            .map(|j| {
                if !self.focus.is_empty() && (self.draws + j) % 4 == 0 {
                    return self.focus[((self.draws + j) / 4) % self.focus.len()].clone();
                }
                let i = rng.index(self.examples.len());
                self.examples[i].1.clone()
            })
            .collect();
        self.draws += n;
        Ok(out)
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"cursor":self.cursor,"draws":self.draws,"cache_positions":self.examples.len(),"cache_bytes":self.bytes,"catalogue":self.files.len(),"pending_new_proofs":self.pending.len()})
    }
    pub fn restore(&mut self, v: &serde_json::Value) {
        self.cursor = v["cursor"].as_u64().unwrap_or(0) as usize;
        self.draws = v["draws"].as_u64().unwrap_or(0) as usize;
    }
}
