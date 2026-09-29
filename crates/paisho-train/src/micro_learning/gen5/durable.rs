//! Compact, independently persistent lessons. Never owned by the recent FIFO.
pub(super) use super::proof_cache::Proofs;
use super::*;
use std::{
    io::{Read, Write},
    sync::RwLock,
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Lesson {
    decision: usize,
    value: f64,
    policy_weight: f64,
    policy: Vec<(String, f64)>,
    reason: String,
    budget: usize,
    collector: String,
}
impl Lesson {
    pub(super) fn from_saved(s: &SavedMicroExample) -> Self {
        Self {
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
    entries: Vec<PathBuf>,
    seen: std::collections::HashSet<PathBuf>,
    cursor: usize,
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
        Ok(Self {
            seen: entries.iter().cloned().collect(),
            entries,
            cursor: 0,
            proofs,
            draws: 0,
            revisit: None,
        })
    }
    pub fn add(&mut self, path: PathBuf) {
        if self.seen.insert(path.clone()) {
            self.entries.push(path);
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
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
                let l = revision.as_ref().unwrap_or(original);
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
                let mut ex = MicroExample {
            sequence_source: paisho_ai::sequence_source(&format!("{}/{}", b.source, b.game_id)),
                    state: micro_state_features(&position),
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
                if let Some(certificate) = proof_cache::lookup(&self.proofs, &key)? {
                    let winner = match certificate.outcome {
                        1 => Some(Player::Host),
                        -1 => Some(Player::Guest),
                        _ => None,
                    };
                    ex.value =
                        winner.map_or(0.0, |w| if w == position.to_move() { 1.0 } else { -1.0 });
                    if ex.value == -1.0 {
                        ex.policy_weight = 0.0;
                        ex.actions.clear();
                        ex.policy.clear();
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
                ex.validate().map_err(invalid)?;
                let embedding = model.embed(&ex.state);
                let mut error = (embedding.value - ex.value).abs();
                if !ex.policy.is_empty() {
                    let logits = MicroModel::logits(&embedding, &ex.actions);
                    let prediction = micro_softmax(&logits).map_err(invalid)?;
                    error += prediction
                        .iter()
                        .zip(&ex.policy)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>();
                }
                if shard_index == self.cursor % self.entries.len()
                    && error > revisit_error
                    && !b.case.human_source.starts_with("historical-")
                {
                    revisit_error = error;
                    self.revisit = Some(reanalysis::Request {
                        prefix: cases::prefix(&record, l.decision - 1),
                        origin: b.case.clone(),
                    });
                }
                candidates.push((error, Arc::new(ex)));
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
            Instant::now() + Duration::from_secs(10),
            &pool,
            "deleted-campaign",
            Some(&prefix),
            false,
            Some(&archive.proofs),
        );
        assert_eq!(game.outcome, terminal);
        game.case = Some(cases::Attempt {
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
        let mut bytes = fs::read(&path).unwrap();
        bytes[0] ^= 1;
        fs::write(&path, bytes).unwrap();
        assert!(reopened
            .rehearse(1, &mut StableRng::new(3), &current)
            .is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
