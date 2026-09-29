//! Durable coverage epochs with independent, bounded ordered preload lanes.
use super::*;
use std::collections::{BTreeMap, HashSet};
use std::sync::{mpsc, Mutex};

#[derive(Clone, Default, Serialize, Deserialize)]
struct Cursor {
    manifest: PathBuf,
    sha256: String,
    epoch: usize,
    block: usize,
    item: usize,
}
pub(super) struct Row {
    pub key: String,
    pub group: String,
    pub example: Arc<MicroExample>,
    pub proof: bool,
    after: Cursor,
    position: Option<Arc<paisho_core::Position>>,
    proof_generation: u64,
}
struct Lane {
    catalogue: Arc<Mutex<Vec<PathBuf>>>,
    receiver: mpsc::Receiver<std::result::Result<Row, String>>,
    cursor: Cursor,
    draws: usize,
    wait_seconds: f64,
    preload_capacity: usize,
}
pub(super) struct Coverage {
    policy_support: bool,
    proofs: Lane,
    ordinary: Lane,
    pub winning_slots: usize,
    pub ordinary_slots: usize,
    proof_cache: Proofs,
}
fn epoch(
    root: &Path,
    name: &str,
    files: &[PathBuf],
    number: usize,
) -> Result<(Cursor, Vec<PathBuf>)> {
    let mut files = files.to_vec();
    if name == "proofs" {
        // Same position may occur in several read-only archives. New writable root wins.
        let mut seen = HashSet::new();
        files.retain(|p| seen.insert(p.file_name().unwrap().to_owned()));
    }
    files.sort();
    if files.is_empty() {
        return Err(invalid("empty coverage catalogue"));
    }
    let bytes = serde_json::to_vec(&files)?;
    let hash = sha256(&bytes);
    let path = root.join(format!("coverage-{name}-{hash}.json"));
    if !path.exists() {
        write(&path, &files)?;
    }
    Ok((
        Cursor {
            manifest: path,
            sha256: hash,
            epoch: number,
            block: 0,
            item: 0,
        },
        files,
    ))
}
fn restore(cursor: &Cursor) -> Result<Vec<PathBuf>> {
    let bytes = fs::read(&cursor.manifest)?;
    if sha256(&bytes) != cursor.sha256 {
        return Err(invalid("coverage manifest changed"));
    }
    let paths: Vec<PathBuf> = serde_json::from_slice(&bytes)?;
    if paths.is_empty() || cursor.block > paths.len() {
        return Err(invalid("invalid coverage cursor"));
    }
    Ok(paths)
}
impl Lane {
    fn open(
        name: &'static str,
        paths: Vec<PathBuf>,
        out: PathBuf,
        saved: &serde_json::Value,
        model: MicroModel,
        proofs: Proofs,
        writable: PathBuf,
        preload_capacity: usize,
        trusted_action_values: bool,
    ) -> Result<Self> {
        let (cursor, files) = if saved["cursor"].is_object() {
            let c: Cursor = serde_json::from_value(saved["cursor"].clone())?;
            let f = restore(&c)?;
            (c, f)
        } else {
            epoch(&out, name, &paths, 0)?
        };
        let catalogue = Arc::new(Mutex::new(paths));
        let worker_catalogue = catalogue.clone();
        let (tx, receiver) = mpsc::sync_channel(preload_capacity);
        let mut next = cursor.clone();
        std::thread::Builder::new()
            .name(format!("gen5-{name}-coverage"))
            .spawn(move || {
                let result = (|| -> Result<()> {
                    let mut files = files;
                    loop {
                        if next.block >= files.len() {
                            (next, files) = epoch(
                                &out,
                                name,
                                &worker_catalogue.lock().unwrap(),
                                next.epoch + 1,
                            )?;
                        }
                        if name == "proofs" {
                            let mut row = decode_proof(&files[next.block], &model, trusted_action_values)?;
                            next.block += 1;
                            next.item = 0;
                            row.after = next.clone();
                            if tx.send(Ok(row)).is_err() {
                                return Ok(());
                            }
                        } else {
                            // Four compressed records; the bounded dense queue
                            // prepares one maximum-size learner batch ahead.
                            // All lessons participate; source groups interleave within each block.
                            let end = (next.block + 4).min(files.len());
                            let bundles = files[next.block..end]
                                .iter()
                                .map(|p| read(p))
                                .collect::<Result<Vec<_>>>()?;
                            let records = bundles
                                .iter()
                                .map(|b| b.psr.parse::<GameRecord>())
                                .collect::<std::result::Result<Vec<_>, _>>()?;
                            let mut replay = records.iter().map(prefix_states::PrefixStates::new).collect::<Vec<_>>();
                            let mut groups: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
                            for (bi, b) in bundles.iter().enumerate() {
                                for li in 0..b.lessons.len() {
                                    groups
                                        .entry(b.case.human_source.clone())
                                        .or_default()
                                        .push((bi, li));
                                }
                            }
                            let mut order = vec![];
                            for i in 0..groups.values().map(Vec::len).max().unwrap_or(0) {
                                for g in groups.values() {
                                    if let Some(pair) = g.get(i) {
                                        order.push(*pair);
                                    }
                                }
                            }
                            if next.item > order.len() {
                                return Err(invalid("coverage lesson cursor out of bounds"));
                            }
                            while next.item < order.len() {
                                let (bi, li) = order[next.item];
                                next.item += 1;
                                if let Some(mut row) = decode_lesson(
                                    &bundles[bi],
                                    &records[bi],
                                    &mut replay[bi],
                                    li,
                                    &files[next.block + bi],
                                    &writable,
                                    &model,
                                    &proofs,
                                    trusted_action_values,
                                )? {
                                    row.after = next.clone();
                                    if tx.send(Ok(row)).is_err() {
                                        return Ok(());
                                    }
                                }
                            }
                            next.block = end;
                            next.item = 0;
                        }
                    }
                })();
                if let Err(e) = result {
                    let _ = tx.send(Err(e.to_string()));
                }
            })?;
        Ok(Self {
            catalogue,
            receiver,
            cursor,
            draws: saved["draws"].as_u64().unwrap_or(0) as usize,
            wait_seconds: 0.,
            preload_capacity,
        })
    }
    fn draw(&mut self) -> Result<Row> {
        let start = paisho_platform::training_time::now();
        let row = self
            .receiver
            .recv()
            .map_err(|_| invalid("coverage worker stopped"))?
            .map_err(invalid)?;
        self.wait_seconds += paisho_platform::training_time::elapsed(start).as_secs_f64();
        self.cursor = row.after.clone();
        self.draws += 1;
        Ok(row)
    }
    fn progress(&self) -> serde_json::Value {
        serde_json::json!({"cursor":self.cursor,"draws":self.draws,"wait_seconds":self.wait_seconds,"preload_capacity":self.preload_capacity})
    }
}
impl Coverage {
    pub fn open(
        proof_paths: Vec<PathBuf>,
        bundles: Vec<PathBuf>,
        out: &Path,
        state: &serde_json::Value,
        model: &MicroModel,
        proofs: &Proofs,
        writable: &Path,
    ) -> Result<Self> {
        Self::open_with_trusted_q(proof_paths,bundles,out,state,model,proofs,writable,false)
    }
    pub fn open_with_trusted_q(
        proof_paths: Vec<PathBuf>, bundles: Vec<PathBuf>, out: &Path,
        state: &serde_json::Value, model: &MicroModel, proofs: &Proofs,
        writable: &Path, trusted_action_values: bool,
    ) -> Result<Self> {
        Ok(Self {
            policy_support: trusted_action_values,
            proof_cache: proofs.clone(),
            proofs: Lane::open(
                "proofs",
                proof_paths,
                out.into(),
                &state["proofs"],
                model.clone(),
                proofs.clone(),
                writable.into(),
                64,
                trusted_action_values,
            )?,
            ordinary: Lane::open(
                "ordinary",
                bundles,
                out.into(),
                &state["ordinary"],
                model.clone(),
                proofs.clone(),
                writable.into(),
                64,
                trusted_action_values,
            )?,
            winning_slots: state["winning_slots"].as_u64().unwrap_or(0) as usize,
            ordinary_slots: state["ordinary_slots"].as_u64().unwrap_or(0) as usize,
        })
    }
    pub fn proof(&mut self) -> Result<Row> {
        self.proofs.draw()
    }
    pub fn ordinary(&mut self) -> Result<Row> {
        let mut row = self.ordinary.draw()?;
        refresh_proof(&mut row, &self.proof_cache, self.policy_support)?;
        Ok(row)
    }
    pub fn add_proof(&self, path: PathBuf) {
        self.proofs.catalogue.lock().unwrap().push(path);
    }
    pub fn add_bundle(&self, path: PathBuf) {
        self.ordinary.catalogue.lock().unwrap().push(path);
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"proofs":self.proofs.progress(),"ordinary":self.ordinary.progress(),"winning_slots":self.winning_slots,"ordinary_slots":self.ordinary_slots})
    }
}

fn decode_proof(path: &Path, model: &MicroModel, policy_support: bool) -> Result<Row> {
    let v: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    let text = v["prefix"]
        .as_str()
        .ok_or_else(|| invalid("proof prefix missing"))?;
    let key = sha256(text.as_bytes());
    if path.file_stem().and_then(|s| s.to_str()) != Some(&key) || v["rules"] != RULES.as_str() {
        return Err(invalid("coverage proof identity mismatch"));
    }
    let record: GameRecord = text.parse()?;
    let p = record.replay()?;
    if record.rules() != RULES || p.outcome() != GameOutcome::Ongoing {
        return Err(invalid("invalid coverage proof position"));
    }
    let c: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
    c.verify(&p).map_err(invalid)?;
    let value = (c.outcome * if p.to_move() == Player::Host { 1 } else { -1 }) as f64;
    let legal = paisho_core::legal_actions(&p);
    let policy = proved_policy(&p, &legal, &c, value)?;
    let mut action_values=certificate_action_values(&p,&legal,&c)?;
    if policy_support && value==1. {action_values::complete_verified_winning_values(&mut action_values,&policy)?;}
    let example = MicroExample { policy_support: policy_support && value==1., action_values,
        state: model.state_features(&p),
        actions: legal
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect(),
        policy,
        value,
        policy_weight: if value >= 0. { 1. } else { 0. },
        value_weight: 1.,
        sequence_source: 0,
    };
    example.validate().map_err(invalid)?;
    Ok(Row {
        key,
        group: v["human_source"]
            .as_str()
            .unwrap_or("unknown-proof-source")
            .into(),
        example: Arc::new(example),
        proof: true,
        position: None,
        proof_generation: 0,
        after: Cursor::default(),
    })
}
fn proved_policy(
    position: &paisho_core::Position,
    legal: &[paisho_core::Action],
    c: &MicroProofCertificate,
    value: f64,
) -> Result<Vec<f64>> {
    let mut policy = vec![0.; legal.len()];
    for (i, a) in legal.iter().enumerate() {
        let certified = c
            .children
            .iter()
            .any(|(s, k)| s == &a.to_string() && k.outcome == c.outcome);
        let immediate = if value == 1. && !certified {
            let mut n = position.clone();
            n.apply(*a)?;
            n.outcome() == GameOutcome::Win(position.to_move())
        } else {
            false
        };
        if certified || immediate {
            policy[i] = 1.;
        }
    }
    let total: f64 = policy.iter().sum();
    if total == 0. {
        return Err(invalid("coverage proof lacks legal witness"));
    }
    for p in &mut policy {
        *p /= total;
    }
    Ok(policy)
}
fn decode_lesson(
    b: &Bundle,
    record: &GameRecord,
    replay: &mut prefix_states::PrefixStates<'_>,
    i: usize,
    path: &Path,
    writable: &Path,
    model: &MicroModel,
    proofs: &Proofs,
    trusted_action_values: bool,
) -> Result<Option<Row>> {
    let original = &b.lessons[i];
    if original.decision == 0 || original.decision > record.actions().len() + 1 {
        return Err(invalid("coverage decision out of bounds"));
    }
    let prefix = cases::prefix(record, original.decision - 1);
    let key = sha256(prefix.to_string().as_bytes());
    let position = replay.at(original.decision - 1)?;
    if record.rules() != RULES || position.outcome() != GameOutcome::Ongoing {
        return Err(invalid("invalid coverage lesson position"));
    }
    let mut revision = revisions::lookup(writable, &key)?;
    if revision.is_none() && path.parent() != Some(writable) {
        revision = revisions::lookup(path.parent().unwrap(), &key)?;
    }
    let l = revisions::resolve(original, revision, true);
    let proof_generation = proofs.read().unwrap().generation;
    let cert = proof_cache::lookup(proofs, &key)?;
    if cert.is_none() && l.reason == "repetition-training-loss" {
        return Ok(None);
    }
    let legacy = l
        .evidence
        .as_ref()
        .is_some_and(TargetEvidence::legacy_coupled_policy);
    let legal = if l.policy_weight > 0. && !legacy || cert.is_some() {
        paisho_core::legal_actions(&position)
    } else {
        vec![]
    };
    let mut policy = vec![0.; legal.len()];
    if l.policy_weight > 0. && !legacy {
        for (a, q) in &l.policy {
            let a: paisho_core::Action = a.parse()?;
            let j = legal
                .iter()
                .position(|x| *x == a)
                .ok_or_else(|| invalid("coverage policy action illegal"))?;
            policy[j] = *q;
        }
    }
    let mut ex = MicroExample { policy_support: trusted_action_values && l.evidence.as_ref().is_some_and(|e|e.policy_support), action_values: if trusted_action_values {
        action_values::action_value_targets_v3(legal.len(),l.evidence.as_ref(),l.tactical.as_ref(),&[])
    } else {action_value_targets(legal.len(),l.evidence.as_ref(),l.tactical.as_ref())},
        state: model.state_features(&position),
        actions: legal
            .iter()
            .map(|a| micro_action_features(&position, *a))
            .collect(),
        policy,
        value: l.value,
        policy_weight: l.policy_weight,
        value_weight: if l.reason == "fresh-reanalysis-q" {
            0.25
        } else {
            l.evidence.as_ref().map_or(1., |e| e.value_weight)
        },
        sequence_source: b.source_run.as_ref().map_or(u64::MAX, |s| {
            paisho_ai::sequence_source(&format!("{s}/{}", b.game_id))
        }),
    };
    if l.evidence
        .as_ref()
        .is_some_and(TargetEvidence::legacy_coupled_policy)
    {
        ex.policy_weight = 0.;
    }
    if let Some(c) = &cert {
        ex.action_values=certificate_action_values(&position,&legal,c)?;
        ex.value = (c.outcome
            * if position.to_move() == Player::Host {
                1
            } else {
                -1
            }) as f64;
        ex.value_weight = 1.;
        ex.policy_support = trusted_action_values && ex.value==1.;
        ex.policy = proved_policy(&position, &legal, c, ex.value)?;
        if ex.policy_support {action_values::complete_verified_winning_values(&mut ex.action_values,&ex.policy)?;}
        ex.policy_weight = if ex.value >= 0. { 1. } else { 0. };
    }
    if ex.policy_weight == 0. {
        ex.actions.clear();
        ex.policy.clear();
        ex.action_values.clear();
    }
    ex.validate().map_err(invalid)?;
    Ok(Some(Row {
        key,
        group: b.case.human_source.clone(),
        example: Arc::new(ex),
        proof: cert.is_some(),
        position: Some(Arc::new(position)),
        proof_generation,
        after: Cursor::default(),
    }))
}

pub(crate) fn audit(source: &Path, model: &MicroModel, out: &Path) -> Result<serde_json::Value> {
    fs::create_dir_all(out)?;
    let proof_files = fs::read_dir(source.join("proofs"))?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let bundles = fs::read_dir(source)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| p.to_string_lossy().ends_with(".json.gz"))
        .collect::<Vec<_>>();
    let proofs = Arc::new(RwLock::new(proof_cache::Cache::open(source)?));
    let mut c = Coverage::open(
        proof_files.clone(),
        bundles.clone(),
        out,
        &serde_json::Value::Null,
        model,
        &proofs,
        source,
    )?;
    let unique = proof_files
        .iter()
        .map(|p| p.file_name().unwrap().to_owned())
        .collect::<HashSet<_>>()
        .len();
    let mut first = HashSet::new();
    let mut wins = 0;
    for _ in 0..unique {
        let r = c.proof()?;
        assert!(first.insert(r.key));
        wins += usize::from(r.example.value == 1.);
    }
    assert_eq!(first.len(), unique);
    for _ in 0..73 {
        c.proof()?;
    }
    for _ in 0..79 {
        c.ordinary()?;
    }
    let state = c.progress();
    let mut resumed = Coverage::open(proof_files, bundles, out, &state, model, &proofs, source)?;
    for _ in 0..256 {
        let a = c.proof()?;
        let b = resumed.proof()?;
        assert_eq!(a.key, b.key);
        assert_eq!(
            serde_json::to_value(super::super::resume_example::ResumeExample::from(
                a.example.as_ref()
            ))?,
            serde_json::to_value(super::super::resume_example::ResumeExample::from(
                b.example.as_ref()
            ))?
        );
        let a = c.ordinary()?;
        let b = resumed.ordinary()?;
        assert_eq!(a.key, b.key);
        assert_eq!(
            serde_json::to_value(super::super::resume_example::ResumeExample::from(
                a.example.as_ref()
            ))?,
            serde_json::to_value(super::super::resume_example::ResumeExample::from(
                b.example.as_ref()
            ))?
        );
    }
    Ok(
        serde_json::json!({"unique_proofs_before_repeat":unique,"winning_proofs_before_repeat":wins,"resumed_proof_draws_exact":256,"resumed_ordinary_draws_exact":256,"coverage":c.progress()}),
    )
}

// Recheck only after a semantic proof admission, never after an ordinary disk cache fill.
fn refresh_proof(row: &mut Row, proofs: &Proofs, policy_support: bool) -> Result<()> {
    if row.proof_generation == proofs.read().unwrap().generation {
        return Ok(());
    }
    let Some(c) = proof_cache::lookup(proofs, &row.key)? else {
        return Ok(());
    };
    let Some(position) = &row.position else {
        return Ok(());
    };
    let legal = paisho_core::legal_actions(position);
    let ex = Arc::make_mut(&mut row.example);
    ex.value = (c.outcome
        * if position.to_move() == Player::Host {
            1
        } else {
            -1
        }) as f64;
    ex.value_weight = 1.;
    ex.policy_support = policy_support && ex.value==1.;
    ex.policy_weight = if ex.value >= 0. { 1. } else { 0. };
    if ex.policy_weight > 0. {
        ex.action_values=certificate_action_values(position,&legal,&c)?;
        ex.policy = proved_policy(position, &legal, &c, ex.value)?;
        if ex.policy_support {action_values::complete_verified_winning_values(&mut ex.action_values,&ex.policy)?;}
        ex.actions = legal
            .iter()
            .map(|a| micro_action_features(position, *a))
            .collect();
    } else {
        ex.policy.clear();
        ex.actions.clear();
        ex.action_values.clear();
    }
    ex.validate().map_err(invalid)?;
    row.proof = true;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preloaded_estimate_yields_to_new_proof_without_cache_load_invalidations() {
        let r: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
        ))
        .parse()
        .unwrap();
        let position = r.replay().unwrap();
        let model = MicroModel::seeded(31).with_spatial_policy();
        let mut session = MicroMctsSession::new(Arc::new(model.clone()));
        session
            .search_with_options(
                &position,
                512,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let certificate = session.certificate(10000).unwrap();
        let proofs = Arc::new(RwLock::new(proof_cache::Cache::default()));
        let key = sha256(r.to_string().as_bytes());
        let mut row = Row {
            key: key.clone(),
            group: "test".into(),
            proof: false,
            after: Cursor::default(),
            position: Some(Arc::new(position.clone())),
            proof_generation: 0,
            example: Arc::new(MicroExample { policy_support:false, action_values: vec![], 
                state: model.state_features(&position),
                actions: vec![],
                policy: vec![],
                value: -0.75,
                policy_weight: 0.,
                value_weight: 0.25,
                sequence_source: 0,
            }),
        };
        refresh_proof(&mut row, &proofs, false).unwrap();
        assert_eq!(row.example.value, -0.75);
        proofs
            .write()
            .unwrap()
            .insert(key.clone(), certificate.clone());
        assert_eq!(proofs.read().unwrap().generation, 1);
        refresh_proof(&mut row, &proofs, false).unwrap();
        assert!(row.proof);
        assert_eq!(row.example.value, 1.);
        assert_eq!(row.example.value_weight, 1.);
        let legal = paisho_core::legal_actions(&position);
        for (a, q) in legal.iter().zip(&row.example.policy) {
            if *q > 0. {
                let mut n = position.clone();
                n.apply(*a).unwrap();
                assert_eq!(n.outcome(), GameOutcome::Win(position.to_move()));
            }
        }
        proofs.write().unwrap().insert(key, certificate);
        assert_eq!(proofs.read().unwrap().generation, 1);
    }
}

/// Frozen-file comparison: queue depth changes preparation, never consumption order.
pub fn verify_prefetch(config:&Path,out:&Path)->Result<serde_json::Value> {
    fs::create_dir_all(out)?;
    let c:serde_json::Value=serde_json::from_slice(&fs::read(config)?)?;
    let saved:serde_json::Value=serde_json::from_slice(&fs::read(c["resume_progress"].as_str().ok_or_else(||invalid("resume"))?)?)?;
    let model=MicroArtifact::load(Path::new(c["model"].as_str().ok_or_else(||invalid("model"))?))?.model()?;
    let mut proofs=proof_cache::Cache::default();
    for root in c["recall_archive_sources"].as_array().ok_or_else(||invalid("archives"))? {proofs.add_read_only(Path::new(root.as_str().ok_or_else(||invalid("root"))?))?;}
    let proofs=Arc::new(RwLock::new(proofs));let mut reports=vec![];
    for name in ["proofs","ordinary"] {
        let state=&saved["durable_recall"]["coverage"][name];
        let cursor:Cursor=serde_json::from_value(state["cursor"].clone())?;let paths=restore(&cursor)?;
        let mut a=Lane::open(name,paths.clone(),out.join("eight"),state,model.clone(),proofs.clone(),out.into(),8,false)?;
        let mut b=Lane::open(name,paths,out.join("sixty-four"),state,model.clone(),proofs.clone(),out.into(),64,false)?;
        let fingerprint=|r:&Row| {
            let e=&r.example;let bits=|v:&[f64]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
            serde_json::json!({"key":r.key,"group":r.group,"proof":r.proof,"after":r.after,"state":bits(&e.state),"actions":e.actions.iter().map(|v|bits(v)).collect::<Vec<_>>(),"policy":bits(&e.policy),"action_values":e.action_values.iter().map(|v|v.map(f64::to_bits)).collect::<Vec<_>>(),"value":e.value.to_bits(),"value_weight":e.value_weight.to_bits(),"policy_weight":e.policy_weight.to_bits(),"source":e.sequence_source})
        };
        for _ in 0..1024 {assert_eq!(fingerprint(&a.draw()?),fingerprint(&b.draw()?));}
        assert_eq!(serde_json::to_value(&a.cursor)?,serde_json::to_value(&b.cursor)?);
        reports.push(serde_json::json!({"lane":name,"rows":1024,"bits_and_consumed_cursor_exact":true}));
    }
    let report=serde_json::json!({"comparisons":reports,"capacities":[8,64]});write(&out.join("report.json"),&report)?;Ok(report)
}
