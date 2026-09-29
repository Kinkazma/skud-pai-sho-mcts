//! Compact, independently persistent lessons. Never owned by the recent FIFO.
pub(super) use super::proof_cache::Proofs;
use super::*;
mod balanced;
mod coverage;
pub use coverage::verify_prefetch;
mod rehearsal;
mod distance_probe;
pub use distance_probe::verify as verify_policy_distance;
mod winning;
pub(super) use coverage::audit as audit_coverage;
use rehearsal::Pool;
use std::{
    io::{Read, Write},
    sync::RwLock,
};

/// Distance to the policy objective used by this lesson. A verified support
/// allows any distribution on its winning actions, so only escaped mass is an
/// error. Keep the original arithmetic for legacy and singleton targets.
fn policy_distance(prediction: &[f64], ex: &MicroExample) -> f64 {
    if ex.policy_support && ex.policy.iter().filter(|&&t| t > 0.).take(2).count() > 1 {
        2. * prediction
            .iter()
            .zip(&ex.policy)
            .filter(|(_, t)| **t <= 0.)
            .map(|(p, _)| *p)
            .sum::<f64>()
    } else {
        prediction
            .iter()
            .zip(&ex.policy)
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
    }
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Lesson {
    #[serde(default, skip_serializing_if="Option::is_none")]
    tactical: Option<TacticalEvidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) evidence: Option<TargetEvidence>,
    decision: usize,
    pub(super) value: f64,
    pub(super) policy_weight: f64,
    policy: Vec<(String, f64)>,
    pub(super) reason: String,
    budget: usize,
    collector: String,
}
impl Lesson {
    pub(super) fn from_saved(s: &SavedMicroExample) -> Self {
        Self {
            tactical: s.tactical.clone(),
            evidence: s.evidence.clone(),
            decision: s.decision,
            value: s.value,
            policy_weight: s.policy_weight,
            policy: s
                .actions
                .iter()
                .cloned()
                .zip(s.policy.iter().copied())
                .filter(|(_, p)| *p > 0.0)
                .collect(),
            reason: s.reason.clone(),
            budget: s.budget,
            collector: s.collector.clone(),
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Bundle {
    schema: String,
    rules: String,
    case: cases::Attempt,
    psr: String,
    psr_sha256: String,
    source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_run: Option<String>,
    game_id: usize,
    lessons: Vec<Lesson>,
    proofs: Vec<(usize, MicroProofCertificate)>,
}
/// fsync before publication; an interrupted temporary file is never an archive entry.
pub(super) fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    write_pending(path, value)?;
    fs::File::open(path.parent().unwrap())?.sync_all()?;
    Ok(())
}
/// The caller must perform a full barrier on this filesystem before acknowledging.
pub(super) fn write_pending(path: &Path, value: &impl Serialize) -> Result<()> {
    let tmp = path.with_extension("tmp");
    let mut f = fs::File::create(&tmp)?;
    f.write_all(&serde_json::to_vec(value)?)?;
    paisho_platform::sync_before_batch_commit(&f)?;
    fs::rename(tmp, path)?;
    paisho_platform::sync_before_batch_commit(&fs::File::open(path.parent().unwrap())?)?;
    Ok(())
}
fn read(path: &Path) -> Result<Bundle> {
    let bytes = fs::read(path)?;
    let expected = path
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_suffix(".json.gz"))
        .ok_or_else(|| invalid("invalid durable bundle name"))?;
    if sha256(&bytes) != expected {
        return Err(invalid("durable lesson hash mismatch"));
    }
    let mut decoder = flate2::read::GzDecoder::new(bytes.as_slice());
    let mut bytes = vec![];
    decoder.read_to_end(&mut bytes)?;
    let b: Bundle = serde_json::from_slice(&bytes)?;
    if b.schema != "paisho-gen5-durable-lessons-v1"
        || b.rules != RULES.as_str()
        || sha256(b.psr.as_bytes()) != b.psr_sha256
    {
        return Err(invalid("durable lesson identity mismatch"));
    }
    Ok(b)
}
#[cfg(test)]
pub(super) fn persist(
    dir: &Path,
    game: &collector::Played,
    saved: &[SavedMicroExample],
    proofs: &Proofs,
) -> Result<Option<PathBuf>> {
    persist_mode(dir, game, saved, proofs, false)
}
pub(super) fn persist_mode(
    dir: &Path,
    game: &collector::Played,
    saved: &[SavedMicroExample],
    proofs: &Proofs,
    buffered: bool,
) -> Result<Option<PathBuf>> {
    let historical = cases::Attempt {
        opponent_generation: None,
        actor: usize::MAX,
        case: format!("historical-{}", game.reference_budget.unwrap_or(0)),
        human_source: "historical-reference-not-human".into(),
        zone: 0,
        prefix_decisions: 0,
        before: cases::State::default(),
        after: cases::State::default(),
        kind: "historical".into(),
    };
    let case = match &game.case {
        Some(c) => c,
        None if game.lane == Lane::Historical => &historical,
        None => return Ok(None),
    };
    if saved.is_empty() {
        return Ok(None);
    }
    let psr = game.record.to_string();
    let bundle = Bundle {
        schema: "paisho-gen5-durable-lessons-v1".into(),
        rules: RULES.to_string(),
        case: case.clone(),
        psr_sha256: sha256(psr.as_bytes()),
        psr,
        source: game.snapshot.identity.clone(),
        source_run: saved.first().map(|s| s.source_run.clone()),
        game_id: game.id,
        lessons: saved.iter().map(Lesson::from_saved).collect(),
        proofs: game.certificates.clone(),
    };
    // Certificates are validated against exact states before becoming reusable.
    register_proofs(&bundle, proofs)?;
    if !bundle.proofs.is_empty() {
        let proof_dir = dir.join("proofs");
        fs::create_dir_all(&proof_dir)?;
        for (decision, certificate) in &bundle.proofs {
            let prefix = cases::prefix(&game.record, decision - 1).to_string();
            let value = serde_json::json!({"rules":RULES.as_str(),"prefix":prefix,"certificate":certificate,"human_source":case.human_source});
            let hash = sha256(prefix.as_bytes());
            let path = proof_dir.join(format!("{hash}.json"));
            if !path.exists() {
                write_pending(&path, &value)?;
                proofs.write().unwrap().disk_count += 1;
            }
        }
    }
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&serde_json::to_vec(&bundle)?)?;
    let bytes = encoder.finish()?;
    let path = dir.join(format!("{}.json.gz", sha256(&bytes)));
    if !path.exists() {
        let tmp = path.with_extension("tmp");
        let mut f = fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        paisho_platform::sync_before_batch_commit(&f)?;
        fs::rename(tmp, &path)?;
        paisho_platform::sync_before_batch_commit(&fs::File::open(dir)?)?;
    }
    super::revisions::store(dir, game, saved, &path)?;
    if !buffered {
        fs::File::open(dir)?.sync_all()?;
    }
    Ok(Some(path))
}
fn register_proofs(b: &Bundle, shared: &Proofs) -> Result<()> {
    if b.proofs.is_empty() {
        return Ok(());
    }
    let record: GameRecord = b.psr.parse()?;
    let mut position = record.initial_position();
    let mut next = 0;
    let mut proofs: Vec<_> = b.proofs.iter().collect();
    proofs.sort_by_key(|(decision, _)| *decision);
    for (decision, certificate) in proofs {
        if *decision == 0 || *decision > record.actions().len() + 1 {
            return Err(invalid("certificate position missing"));
        }
        let prefix = cases::prefix(&record, decision - 1);
        let key = sha256(prefix.to_string().as_bytes());
        let old = proof_cache::lookup(shared, &key)?;
        if let Some(old) = old {
            if old.outcome != certificate.outcome {
                return Err(invalid("conflicting exact certificates"));
            }
            if old == *certificate {
                continue;
            }
        }
        while next < decision - 1 {
            position.apply(record.actions()[next])?;
            next += 1;
        }
        certificate.verify(&position).map_err(invalid)?;
        shared.write().unwrap().insert(key, certificate.clone());
    }
    Ok(())
}

pub(super) struct Archive {
    deferred_winning: std::collections::VecDeque<Arc<MicroExample>>,
    deferred_ordinary: std::collections::VecDeque<Arc<MicroExample>>,
    deferred_checkpoint: serde_json::Value,
    deferred_retired: Vec<PathBuf>,
    coverage: Option<coverage::Coverage>,
    balanced: balanced::Balanced,
    corrective_values: Vec<Arc<MicroExample>>,
    corrective_value_slots: usize,
    entries: Vec<PathBuf>,
    seen: std::collections::HashSet<PathBuf>,
    cursor: usize,
    pub cache_enabled: bool,
    pub loop_repair: bool,
    pub trusted_action_values: bool,
    pool: Pool,
    winning: winning::Winning,
    pub proof_recall: bool,
    proof_credit: f64,
    pub last_winning_draws: usize,
    pub loads: usize,
    refresh_at: usize,
    pub proofs: Proofs,
    pub draws: usize,
    pub revisit: Option<reanalysis::Request>,
}
impl Archive {
    pub fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let mut entries: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.to_string_lossy().ends_with(".json.gz"))
            .collect();
        entries.sort();
        let proofs = Arc::new(RwLock::new(proof_cache::Cache::open(dir)?));
        let mut winning = winning::Winning::default();
        winning.add_root(dir)?;
        Ok(Self {
            deferred_winning: Default::default(),
            deferred_ordinary: Default::default(),
            deferred_checkpoint: serde_json::Value::Null,
            deferred_retired: vec![],
            coverage: None,
            balanced: balanced::Balanced::default(),
            corrective_values: vec![],
            corrective_value_slots: 0,
            seen: entries.iter().cloned().collect(),
            entries,
            cursor: 0,
            cache_enabled: false,
            loop_repair: false,
            trusted_action_values: false,
            pool: Pool::default(),
            winning,
            proof_recall: false,
            proof_credit: 0.,
            last_winning_draws: 0,
            loads: 0,
            refresh_at: 0,
            proofs,
            draws: 0,
            revisit: None,
        })
    }
    pub fn add_read_only(&mut self, root: &Path) -> Result<()> {
        self.winning.add_root(root)?;
        for entry in fs::read_dir(root)? {
            let path = entry?.path();
            if path.to_string_lossy().ends_with(".json.gz") {
                self.add(path);
            }
        }
        self.entries.sort();
        self.proofs.write().unwrap().add_read_only(root)?;
        Ok(())
    }
    pub fn add(&mut self, path: PathBuf) {
        if self.seen.insert(path.clone()) {
            if let Some(c) = &self.coverage {
                c.add_bundle(path.clone());
            }
            self.entries.push(path);
        }
    }
    pub fn focus_policy(&mut self, examples: Vec<Arc<MicroExample>>) {
        let (winning, values)=examples.into_iter().partition(|e|e.value==1. && e.policy_weight>0.);
        self.winning.focus = winning;
        self.corrective_values = values;
    }
    pub fn policy_consolidation(&mut self, active: bool) {
        self.winning.policy_only = active;
        self.loop_repair = active;
    }
    pub fn register_persisted_proof(&mut self, path: PathBuf) {
        if self.winning.register_persisted_proof(path.clone()) {
            if let Some(c) = &self.coverage {
                c.add_proof(path);
            }
        }
    }
    pub fn enable_coverage(
        &mut self,
        out: &Path,
        progress: &serde_json::Value,
        model: &MicroModel,
        writable: &Path,
    ) -> Result<()> {
        self.balanced.restore(&progress["balanced_value"]);
        self.corrective_value_slots=progress["corrective_value_slots"].as_u64().unwrap_or(0) as usize;
        let saved = &progress["deferred_checkpoint"];
        if let Some(path) = saved["path"].as_str() {
            let bytes = fs::read(path)?;
            if sha256(&bytes) != saved["sha256"].as_str().unwrap_or("") {
                return Err(invalid("deferred recall changed"));
            }
            let rows: [Vec<super::resume_example::ResumeExample>; 2] =
                serde_json::from_slice(&bytes)?;
            let [winning, ordinary] = rows;
            self.deferred_winning = winning
                .into_iter()
                .map(|e| e.example_with_trusted_q(self.trusted_action_values))
                .collect::<Result<_>>()?;
            self.deferred_ordinary = ordinary
                .into_iter()
                .map(|e| e.example_with_trusted_q(self.trusted_action_values))
                .collect::<Result<_>>()?;
            self.deferred_checkpoint = saved.clone();
        }
        self.winning.policy_support=self.trusted_action_values;
        self.coverage = Some(coverage::Coverage::open_with_trusted_q(
            self.winning.files.clone(),
            self.entries.clone(),
            out,
            &progress["coverage"],
            model,
            &self.proofs,
            writable,
            self.trusted_action_values,
        )?);
        Ok(())
    }
    pub fn defer(&mut self, ex: Arc<MicroExample>, winning: bool) {
        if self.coverage.is_some() {
            if winning {
                self.deferred_winning.push_back(ex);
            } else {
                self.deferred_ordinary.push_back(ex);
            }
        }
    }
    pub fn checkpoint(&mut self, out: &Path) -> Result<()> {
        if self.coverage.is_none() {
            return Ok(());
        }
        let rows = [&self.deferred_winning, &self.deferred_ordinary].map(|rows| {
            rows.iter()
                .map(|e| super::resume_example::ResumeExample::from_with_trusted_q(e.as_ref(),self.trusted_action_values))
                .collect::<Vec<_>>()
        });
        let hash = sha256(&serde_json::to_vec(&rows)?);
        let path = out.join(format!("deferred-recall-{hash}.json"));
        if !path.exists() {
            write(&path, &rows)?;
        }
        if let Some(old) = self.deferred_checkpoint["path"].as_str() {
            let old = PathBuf::from(old);
            if old != path && old.starts_with(out) {
                self.deferred_retired.push(old);
            }
        }
        self.deferred_checkpoint = serde_json::json!({"path":path,"sha256":hash});
        Ok(())
    }
    pub fn committed(&mut self) -> Result<()> {
        for p in self.deferred_retired.drain(..) {
            if p.exists() {
                fs::remove_file(p)?;
            }
        }
        Ok(())
    }
    pub fn enable_parallel(&mut self,pools:&[Arc<rayon::ThreadPool>]) {self.pool.enable_parallel(pools);}
    pub fn seed_values(&mut self, examples: Vec<Arc<MicroExample>>) {
        for ex in examples {
            let key = sha256(&serde_json::to_vec(&ex.state).unwrap());
            self.balanced.admit(key, ex);
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn admit(&mut self, key: String, group: String, example: Arc<MicroExample>, proof: bool) {
        if proof && self.coverage.is_some() {
            self.balanced.admit(key.clone(), example.clone());
        }
        if proof && self.proof_recall {
            self.winning.admit(key.clone(), example.clone());
        }
        // Fresh proofs enter after their first update; new empirical Q targets
        // only refresh an existing lesson, avoiding a cache full of recent singletons.
        if proof || self.pool.contains(&key) {
            let mut example = example;
            if proof && self.coverage.is_some() && example.policy_weight > 0. {
                Arc::make_mut(&mut example).value_weight = 0.;
            }
            self.pool.insert(key, group, example, 0.0, proof);
        }
    }
    pub fn restore(&mut self, value: &serde_json::Value) -> Result<()> {
        if value.is_null() {
            return Ok(());
        }
        self.winning.restore(&value["winning"]);
        self.proof_credit = value["proof_credit"].as_f64().unwrap_or(0.);
        if !self.proof_credit.is_finite() || !(0.0..1.0).contains(&self.proof_credit) {
            return Err(invalid("invalid proof recall credit"));
        }
        self.cursor = value.get("cursor").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        self.draws = value.get("draws").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        self.loads = value.get("loads").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        if let Some(path) = value.get("next_bundle").and_then(|v| v.as_str()) {
            if let Some(i) = self
                .entries
                .iter()
                .position(|p| p.to_string_lossy() == path)
            {
                self.cursor = i;
            }
        }
        // Dense cache is reconstructible; the next cold catalogue position is durable.
        Ok(())
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"corrective_value_slots":self.corrective_value_slots,"corrective_value_positions":self.corrective_values.len(),"scoring":self.pool.scoring_progress(),"deferred_checkpoint":self.deferred_checkpoint,"deferred_winning":self.deferred_winning.len(),"deferred_ordinary":self.deferred_ordinary.len(),"coverage":self.coverage.as_ref().map(|c|c.progress()),"balanced_value":self.balanced.progress(),"winning":self.winning.progress(),"proof_credit":self.proof_credit,"cursor":self.cursor,"draws":self.draws,"loads":self.loads,
            "next_bundle":if self.entries.is_empty(){None}else{Some(&self.entries[self.cursor % self.entries.len()])},"cache_positions":self.pool.len(),"cache_bytes":self.pool.bytes(),"cache_groups":self.pool.groups()})
    }
    pub fn rehearse_cached(
        &mut self,
        n: usize,
        rng: &mut StableRng,
        model: &MicroModel,
    ) -> Result<Vec<Arc<MicroExample>>> {
        self.last_winning_draws = 0;
        self.winning.policy_support=self.trusted_action_values;
        if n == 0 {
            return Ok(vec![]);
        }
        if self.coverage.is_some() {
            self.pool.begin_scoring();
            let result=self.rehearse_coverage(n,rng,model);
            self.pool.end_scoring();
            return result;
        }
        self.cache_enabled = true;
        let before = self.draws;
        // Initial warmup is bounded. Afterwards two compressed files per 64 draws,
        // independent of receipt size; old catalogue traversal never starts over.
        if self.pool.len() == 0 {
            for _ in 0..32.min(self.entries.len()) {
                self.rehearse(1, rng, model)?;
            }
            self.refresh_at = before + 64;
        } else if before >= self.refresh_at {
            self.rehearse(1, rng, model)?;
            self.refresh_at = before + 64;
        }
        self.draws = before;
        let mut out = vec![];
        if self.proof_recall {
            self.proof_credit += n as f64 * 0.5;
            let requested = self.proof_credit.floor() as usize;
            self.proof_credit -= requested as f64;
            out = self.winning.draw(requested, rng, model)?;
            self.last_winning_draws = out.len();
        }
        out.extend(self.pool.draw(n - out.len(), rng, model)?);
        self.draws += out.len();
        Ok(out)
    }
    fn rehearse_coverage(
        &mut self,
        n: usize,
        rng: &mut StableRng,
        model: &MicroModel,
    ) -> Result<Vec<Arc<MicroExample>>> {
        self.proof_credit += n as f64 * 0.5;
        let winning = self.proof_credit.floor() as usize;
        self.proof_credit -= winning as f64;
        let mut out = Vec::with_capacity(n);
        for _ in 0..winning {
            if let Some(ex) = self.deferred_winning.pop_front() {
                out.push(ex);
                continue;
            }
            let c = self.coverage.as_mut().unwrap();
            let cover = c.winning_slots % 2 == 0;
            c.winning_slots += 1;
            if cover || self.winning.examples_empty() {
                loop {
                    let row = self.coverage.as_mut().unwrap().proof()?;
                    self.balanced.admit(row.key.clone(), row.example.clone());
                    if row.example.value == 1. {
                        let mut ex = row.example.as_ref().clone();
                        ex.value_weight = 0.;
                        let ex = Arc::new(ex);
                        self.winning.admit(row.key.clone(), ex.clone());
                        self.pool.insert(row.key, row.group, ex.clone(), 0., true);
                        out.push(ex);
                        break;
                    }
                }
            } else {
                out.extend(self.winning.draw(1, rng, model)?);
            }
        }
        self.last_winning_draws = out.len();
        for _ in out.len()..n {
            if let Some(ex) = self.deferred_ordinary.pop_front() {
                out.push(ex);
                continue;
            }
            let c = self.coverage.as_mut().unwrap();
            let slot = c.ordinary_slots % 4;
            c.ordinary_slots += 1;
            if slot % 2 == 0 || self.pool.len() == 0 {
                let row = self.coverage.as_mut().unwrap().ordinary()?;
                if row.proof {
                    self.balanced.admit(row.key.clone(), row.example.clone());
                }
                let mut ex = row.example;
                // Verified value learning is sampled by class in slot 1 below.
                // Empirical values retain their original learning weight here.
                if row.proof && ex.policy_weight > 0. {
                    Arc::make_mut(&mut ex).value_weight = 0.;
                }
                self.pool
                    .insert(row.key, row.group, ex.clone(), 0., row.proof);
                out.push(ex);
            } else if slot == 1 {
                let focus_slot=self.corrective_value_slots;
                self.corrective_value_slots+=1;
                if !self.corrective_values.is_empty() && focus_slot % 2 == 0 {
                    out.push(self.corrective_values[(focus_slot/2)%self.corrective_values.len()].clone());
                } else if let Some(ex) = self.balanced.draw() {
                    out.push(ex);
                } else {
                    out.extend(self.pool.draw(1, rng, model)?);
                }
            } else {
                out.extend(self.pool.draw(1, rng, model)?);
            }
        }
        self.draws += out.len();
        Ok(out)
    }
    /// Half the candidate pool revisits old shards in order; half is uniform.
    /// Within it, half the draws target current disagreement, half retain diversity.
    pub fn rehearse(
        &mut self,
        n: usize,
        rng: &mut StableRng,
        model: &MicroModel,
    ) -> Result<Vec<Arc<MicroExample>>> {
        if n == 0 || self.entries.is_empty() {
            return Ok(vec![]);
        }
        let mut candidates = vec![];
        let mut revisit_error = -1.0;
        for shard_index in [
            self.cursor % self.entries.len(),
            rng.index(self.entries.len()),
        ] {
            let b = read(&self.entries[shard_index])?;
            self.loads += 1;
            let record: GameRecord = b.psr.parse()?;
            let mut selected: Vec<_> = (0..b.lessons.len()).collect();
            shuffle(&mut selected, rng);
            selected.truncate(32);
            selected.sort_by_key(|i| b.lessons[*i].decision);
            let mut position = record.initial_position();
            let mut next = 0;
            for i in selected {
                let original = &b.lessons[i];
                let key = sha256(
                    cases::prefix(&record, original.decision.saturating_sub(1))
                        .to_string()
                        .as_bytes(),
                );
                let revision =
                    super::revisions::lookup(self.entries[shard_index].parent().unwrap(), &key)?;
                let l = revisions::resolve(original, revision, self.loop_repair);
                if l.decision == 0 || l.decision > record.actions().len() + 1 {
                    return Err(invalid("durable decision missing"));
                }
                while next < l.decision - 1 {
                    position.apply(record.actions()[next])?;
                    next += 1;
                }
                if position.outcome() != GameOutcome::Ongoing {
                    return Err(invalid("durable lesson at terminal state"));
                }
                let legal = if l.policy_weight > 0.0 {
                    paisho_core::legal_actions(&position)
                } else {
                    vec![]
                };
                let indices: std::collections::HashMap<_, _> = legal
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, a)| (a, i))
                    .collect();
                let mut policy = vec![0.0; legal.len()];
                for (action, p) in &l.policy {
                    let action: paisho_core::Action = action.parse()?;
                    let j = *indices
                        .get(&action)
                        .ok_or_else(|| invalid("durable policy action illegal"))?;
                    policy[j] = *p;
                }
                let mut ex = MicroExample { policy_support: self.trusted_action_values && l.evidence.as_ref().is_some_and(|e|e.policy_support), action_values: if self.trusted_action_values {
                    action_values::action_value_targets_v3(legal.len(),l.evidence.as_ref(),l.tactical.as_ref(),&[])
                } else {action_value_targets(legal.len(),l.evidence.as_ref(),l.tactical.as_ref())},
                    value_weight: if self.loop_repair && l.reason == "fresh-reanalysis-q" {
                        0.25
                    } else {
                        l.evidence.as_ref().map_or(1., |e| e.value_weight)
                    },
                    sequence_source: b.source_run.as_ref().map_or(u64::MAX, |source| {
                        paisho_ai::sequence_source(&format!("{source}/{}", b.game_id))
                    }),
                    state: model.state_features(&position),
                    actions: legal
                        .iter()
                        .map(|a| micro_action_features(&position, *a))
                        .collect(),
                    policy,
                    value: l.value,
                    policy_weight: l.policy_weight,
                };
                let key = sha256(
                    cases::prefix(&record, l.decision - 1)
                        .to_string()
                        .as_bytes(),
                );
                let certificate = proof_cache::lookup(&self.proofs, &key)?;
                let certified = certificate.is_some();
                if let Some(certificate) = certificate {
                    if !ex.actions.is_empty(){ex.action_values=certificate_action_values(&position,&legal,&certificate)?;}
                    let winner = match certificate.outcome {
                        1 => Some(Player::Host),
                        -1 => Some(Player::Guest),
                        _ => None,
                    };
                    ex.value =
                        winner.map_or(0.0, |w| if w == position.to_move() { 1.0 } else { -1.0 });
                    ex.value_weight = 1.;
                    ex.policy_support = self.trusted_action_values && ex.value==1.;
                    if ex.value == -1.0 {
                        ex.policy_weight = 0.0;
                        ex.actions.clear();
                        ex.policy.clear();
                        ex.action_values.clear();
                    } else if self.trusted_action_values && ex.value==1. && !ex.policy.is_empty() {
                        ex.policy=action_values::verified_winning_policy(&position,&legal,&certificate)?;
                        action_values::complete_verified_winning_values(&mut ex.action_values,&ex.policy)?;
                    } else if !ex.policy.is_empty() {
                        let mut valid = vec![false; legal.len()];
                        for (action, child) in &certificate.children {
                            if child.outcome == certificate.outcome {
                                let action: paisho_core::Action = action.parse()?;
                                if let Some(i) = indices.get(&action) {
                                    valid[*i] = true;
                                }
                            }
                        }
                        let n = valid.iter().filter(|v| **v).count();
                        if n > 0 {
                            ex.policy = valid
                                .iter()
                                .map(|v| if *v { 1.0 / n as f64 } else { 0.0 })
                                .collect();
                        }
                    }
                }
                if model.has_spatial() && l.reason == "repetition-training-loss" && !certified {
                    continue;
                }
                ex.validate().map_err(invalid)?;
                let embedding = model.embed(&ex.state);
                let mut error = (embedding.value - ex.value).abs();
                if !ex.policy.is_empty() {
                    let logits = MicroModel::logits(&embedding, &ex.actions);
                    let base = micro_softmax(&logits).map_err(invalid)?;
                    let prediction = model
                        .memory_priors(&ex.state, &ex.actions, &base, ex.sequence_source)
                        .map_err(invalid)?;
                    error += policy_distance(&prediction, &ex);
                }
                if shard_index == self.cursor % self.entries.len()
                    && error > revisit_error
                    && !b.case.human_source.starts_with("historical-")
                {
                    revisit_error = error;
                    self.revisit = Some(reanalysis::Request {
                        prefix: cases::prefix(&record, l.decision - 1),
                        observed_origin: l
                            .evidence
                            .as_ref()
                            .and_then(|e| e.observed_value.zip(e.observed_psr.clone()))
                            .map(|(z, h)| {
                                (
                                    if position.to_move() == Player::Host {
                                        z
                                    } else {
                                        -z
                                    },
                                    h,
                                )
                            })
                            .or_else(|| {
                                (original.reason == "rules-terminal-z").then(|| {
                                    (
                                        if position.to_move() == Player::Host {
                                            original.value
                                        } else {
                                            -original.value
                                        },
                                        b.psr_sha256.clone(),
                                    )
                                })
                            }),
                        origin: b.case.clone(),
                    });
                }
                let ex = Arc::new(ex);
                if self.cache_enabled {
                    let group = if b.case.human_source.starts_with("historical-") {
                        b.psr_sha256.clone()
                    } else {
                        b.case.human_source.clone()
                    };
                    self.pool.insert(key, group, ex.clone(), error, certified);
                }
                candidates.push((error, ex));
            }
        }
        self.cursor += 1;
        candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
        if candidates.is_empty() {
            return Ok(vec![]);
        }
        let out = (0..n)
            .map(|i| {
                let range = if i % 2 == 0 {
                    candidates.len().div_ceil(2)
                } else {
                    candidates.len()
                };
                candidates[rng.index(range)].1.clone()
            })
            .collect();
        self.draws += n;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scoring_example(policy: Vec<f64>, policy_support: bool) -> MicroExample {
        MicroExample {
            state: vec![0.; 417], actions: vec![[0.; 32]; policy.len()], policy,
            value: 1., policy_weight: 1., value_weight: 1., action_values: vec![],
            sequence_source: 0, policy_support,
        }
    }
    #[test]
    fn support_distance_ignores_redistribution_between_verified_wins() {
        let ex = scoring_example(vec![0.5, 0.5, 0.], true);
        let balanced = [0.375, 0.375, 0.25];
        let concentrated = [0.625, 0.125, 0.25];
        assert_eq!(policy_distance(&balanced, &ex).to_bits(), 0.5f64.to_bits());
        assert_eq!(policy_distance(&concentrated, &ex).to_bits(), 0.5f64.to_bits());
        assert_eq!(policy_distance(&[1., 0., 0.], &ex).to_bits(), 0.0f64.to_bits());
    }
    #[test]
    fn support_distance_prioritizes_missing_winning_mass_over_uniformity() {
        let ex = scoring_example(vec![0.5, 0.5, 0.], true);
        let concentrated_winner = [0.875, 0.0625, 0.0625];
        let leaking_balanced = [0.375, 0.375, 0.25];
        let old = |p: &[f64]| p.iter().zip(&ex.policy).map(|(p,t)| (p-t).abs()).sum::<f64>();
        assert!(old(&concentrated_winner) > old(&leaking_balanced));
        assert!(policy_distance(&concentrated_winner, &ex) < policy_distance(&leaking_balanced, &ex));
    }
    #[test]
    fn support_distance_preserves_legacy_and_singleton_bits() {
        let predictions = [[0.1, 0.2, 0.7], [1., 0., -0.], [0.25, 0.5, 0.25]];
        for ex in [scoring_example(vec![0.5, 0.5, 0.], false),
            scoring_example(vec![0., 1., 0.], true),
            scoring_example(vec![0., 0., 0.], true)] {
            for prediction in predictions {
                let old = prediction.iter().zip(&ex.policy).map(|(p,t)| (p-t).abs()).sum::<f64>();
                assert_eq!(policy_distance(&prediction, &ex).to_bits(), old.to_bits());
            }
        }
    }
    #[test]
    fn independent_archive_survives_fifo_and_model_replacement_with_exact_proof() {
        let dir = std::env::temp_dir().join(format!("paisho-durable-case-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let old: GameRecord = include_str!(
            "../../../../../crates/paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr"
        )
        .parse()
        .unwrap();
        let (record, _) = old.replay_prefix_with_rules(RULES).unwrap();
        let prefix = cases::prefix(&record, record.actions().len() - 1);
        let position = prefix.replay().unwrap();
        let action = *record.actions().last().unwrap();
        let terminal = record.replay().unwrap().outcome();
        let value = match terminal {
            GameOutcome::Win(Player::Host) => 1,
            GameOutcome::Win(Player::Guest) => -1,
            _ => panic!("winning fixture"),
        };
        let certificate = MicroProofCertificate {
            outcome: value,
            children: vec![(
                action.to_string(),
                MicroProofCertificate {
                    outcome: value,
                    children: vec![],
                },
            )],
        };
        certificate.verify(&position).unwrap();
        let mut archive = Archive::open(&dir).unwrap();
        archive
            .proofs
            .write()
            .unwrap()
            .insert(sha256(prefix.to_string().as_bytes()), certificate);
        let model = MicroModel::seeded(1);
        let artifact = MicroArtifact::new(&model, 1, serde_json::json!({"test":true}));
        let snapshot = Arc::new(Snapshot {
            artifact: None,
            version: 1,
            identity: artifact.identity(),
            model: Arc::new(model),
            path: PathBuf::new(),
        });
        let options = Options {
            budgets: vec![(256, 1.0)],
            ..Default::default()
        };
        let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
        let mut game = collector::play_from(
            0,
            snapshot.clone(),
            snapshot,
            None,
            &options,
            paisho_platform::training_time::now() + Duration::from_secs(10),
            &pool,
            "deleted-campaign",
            Some(&prefix),
            false,
            Some(&archive.proofs),
        );
        assert_eq!(game.outcome, terminal);
        game.case = Some(cases::Attempt {
            opponent_generation: None,
            actor: 0,
            case: "case".into(),
            human_source: "human".into(),
            zone: 2,
            prefix_decisions: prefix.actions().len(),
            before: cases::State::default(),
            after: cases::State::default(),
            kind: "continuation".into(),
        });
        let fresh = collector::targets(&mut game);
        let expected = if position.to_move() == Player::Host {
            value as f64
        } else {
            -(value as f64)
        };
        assert_eq!(fresh[0].value, expected);
        let path = persist(&dir, &game, &fresh, &archive.proofs)
            .unwrap()
            .unwrap();
        archive.add(path.clone());
        archive.add(path.clone());
        assert_eq!(archive.len(), 1);
        let key = sha256(prefix.to_string().as_bytes());
        let immutable = fs::read(&path).unwrap();
        let mut uncertain = fresh.clone();
        uncertain[0].value = -expected;
        game.snapshot = Arc::new(Snapshot {
            version: 0,
            ..(*game.snapshot).clone()
        });
        super::super::revisions::store(&dir, &game, &uncertain, &path).unwrap();
        assert_eq!(
            super::super::revisions::lookup(&dir, &key)
                .unwrap()
                .unwrap()
                .value,
            expected
        );
        game.snapshot = Arc::new(Snapshot {
            version: 1001,
            ..(*game.snapshot).clone()
        });
        super::super::revisions::store(&dir, &game, &uncertain, &path).unwrap();
        assert_eq!(
            super::super::revisions::lookup(&dir, &key)
                .unwrap()
                .unwrap()
                .value,
            -expected
        );
        assert_eq!(fs::read(&path).unwrap(), immutable); // revisions never erase evidence
        drop(archive); // no FIFO, campaign file, model or in-memory lesson survives
        let mut reopened = Archive::open(&dir).unwrap();
        let current = MicroModel::seeded(1001);
        let examples = reopened
            .rehearse(8, &mut StableRng::new(2), &current)
            .unwrap();
        assert_eq!(examples.len(), 8);
        assert!(examples.iter().all(|e| e.value == expected));
        assert_eq!(reopened.proofs.read().unwrap().len(), 1);
        assert_eq!(read(&path).unwrap().psr, record.to_string());
        assert!(examples
            .iter()
            .all(|e| e.sequence_source == sequence_source("deleted-campaign/0")));
        let loads = reopened.loads;
        for _ in 0..80 {
            assert_eq!(
                reopened
                    .rehearse_cached(1, &mut StableRng::new(5), &current)
                    .unwrap()
                    .len(),
                1
            );
        }
        assert!(
            reopened.loads - loads <= 4,
            "singletons must not reload files each time"
        );
        assert!(reopened.pool.bytes() <= 64 * 1024 * 1024);
        let state = reopened.progress();
        let mut resumed = Archive::open(&dir).unwrap();
        resumed.restore(&state).unwrap();
        assert_eq!(resumed.draws, reopened.draws);
        assert_eq!(resumed.cursor, reopened.cursor % reopened.entries.len());
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(reopened
            .rehearse(1, &mut StableRng::new(3), &current)
            .is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
