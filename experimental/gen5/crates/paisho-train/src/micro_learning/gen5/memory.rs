//! RAM examples have one numeric representation. Durable indices reference
//! immutable target archives, avoiding a second full replay JSON copy in RAM.
use super::*;
use rayon::prelude::*;
use std::{
    collections::{HashMap, VecDeque},
    sync::Weak,
};
#[derive(Clone, Serialize, Deserialize)]
struct Source {
    path: PathBuf,
    sha256: String,
}
struct Entry {
    example: Arc<MicroExample>,
    source: Arc<Source>,
    index: usize,
    lane: Lane,
    bytes: usize,
    trainable: bool,
    /// Separate lifetime from the example Arc (minibatches may outlive eviction).
    _correction_lifetime: Option<Arc<()>>,
}
struct Correction {
    example: Weak<MicroExample>,
    lifetime: Weak<()>,
    lane: Lane,
}
#[derive(Serialize, Deserialize)]
struct IndexRow {
    source: Source,
    indices: Vec<usize>,
    lane: Lane,
}
#[derive(Serialize, Deserialize)]
struct Index {
    schema: String,
    rules: String,
    rows: Vec<IndexRow>,
}
pub(super) struct Memory {
    coherent_policy: bool,
    trusted_action_values: bool,
    loop_repair: bool,
    source_counts: HashMap<PathBuf, usize>,
    obsolete: Vec<(PathBuf, Lane)>,
    entries: VecDeque<Entry>,
    historical: VecDeque<Weak<MicroExample>>,
    corrections: VecDeque<Correction>,
    correction_capacity: usize,
    capacity: usize,
    max_bytes: usize,
    pub bytes: usize,
    pub evicted: usize,
}
fn size(e: &MicroExample) -> usize {
    std::mem::size_of::<Entry>()
        + std::mem::size_of::<MicroExample>()
        + e.state.capacity() * 8
        + e.actions.capacity() * MICRO_ACTION_INPUTS * 8
        + e.policy.capacity() * 8
        + e.action_values.capacity() * std::mem::size_of::<Option<f64>>()
        + e.structured.capacity()*std::mem::size_of::<Option<MicroStructuredTarget>>()
        + e.structured.iter().flatten().map(|t|match &t.threat {MicroThreatEvidence::Present{reply}=>reply.capacity(),_=>0}).sum::<usize>()
}
impl Memory {
    pub fn new(o: &Options) -> Self {
        Self {
            coherent_policy: o.learning_loop_v2,
            trusted_action_values: o.learning_loop_v3,
            loop_repair: o.learning_loop_repair,
            source_counts: HashMap::new(),
            obsolete: vec![],
            entries: VecDeque::new(),
            historical: VecDeque::new(),
            corrections: VecDeque::new(),
            correction_capacity: o.correction_capacity.min(o.replay_capacity),
            capacity: o.replay_capacity,
            max_bytes: o.replay_max_bytes,
            bytes: 0,
            evicted: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn add(
        &mut self,
        values: Vec<Arc<MicroExample>>,
        path: &Path,
        hash: &str,
        lane: Lane,
        corrections: &[bool],
    ) {
        assert!(corrections.is_empty() || corrections.len() == values.len());
        let source = Arc::new(Source {
            path: path.to_path_buf(),
            sha256: hash.into(),
        });
        for (index, example) in values.into_iter().enumerate() {
            self.push(
                example,
                source.clone(),
                index,
                lane,
                corrections.get(index).copied().unwrap_or(false),
                true,
            );
        }
    }
    fn push(
        &mut self,
        example: Arc<MicroExample>,
        source: Arc<Source>,
        index: usize,
        lane: Lane,
        correction: bool,
        trainable: bool,
    ) {
        let bytes = size(&example)
            + if correction && self.correction_capacity > 0 {
                2 * std::mem::size_of::<usize>()
            } else {
                0
            };
        self.bytes += bytes;
        if lane == Lane::Historical && trainable {
            self.historical.push_back(Arc::downgrade(&example));
        }
        let correction_lifetime =
            (correction && trainable && self.correction_capacity > 0).then(|| Arc::new(()));
        if let Some(lifetime) = &correction_lifetime {
            self.corrections.push_back(Correction {
                example: Arc::downgrade(&example),
                lifetime: Arc::downgrade(lifetime),
                lane,
            });
        }
        *self.source_counts.entry(source.path.clone()).or_default() += 1;
        self.entries.push_back(Entry {
            example,
            source,
            index,
            lane,
            bytes,
            _correction_lifetime: correction_lifetime,
            trainable,
        });
        while self.entries.len() > self.capacity || self.bytes > self.max_bytes {
            if let Some(old) = self.entries.pop_front() {
                let count = self.source_counts.get_mut(&old.source.path).unwrap();
                *count -= 1;
                if *count == 0 {
                    self.source_counts.remove(&old.source.path);
                    self.obsolete.push((old.source.path.clone(), old.lane));
                }
                self.bytes -= old.bytes;
                self.evicted += 1;
            } else {
                break;
            }
        }
        while self.historical.len() > self.capacity {
            self.historical.pop_front();
        }
        while self.corrections.len() > self.correction_capacity
            || self
                .corrections
                .front()
                .is_some_and(|c| c.lifetime.strong_count() == 0)
        {
            self.corrections.pop_front();
        }
    }
    pub fn take_obsolete(&mut self) -> Vec<(PathBuf, Lane)> {
        std::mem::take(&mut self.obsolete)
            .into_iter()
            .filter(|(path, _)| !self.source_counts.contains_key(path))
            .collect()
    }
    pub fn correction_len(&self) -> usize {
        self.corrections.len()
    }
    pub fn correction_reference_bytes(&self) -> usize {
        self.corrections.capacity() * std::mem::size_of::<Correction>()
    }
    pub fn draw_correction(&self, rng: &mut StableRng) -> Option<(Arc<MicroExample>, Lane)> {
        if self.corrections.is_empty() {
            return None;
        }
        let c = &self.corrections[rng.index(self.corrections.len())];
        let _live = c.lifetime.upgrade()?;
        Some((c.example.upgrade()?, c.lane))
    }
    pub fn draw(
        &mut self,
        rng: &mut StableRng,
        prefer_history: bool,
    ) -> Option<(Arc<MicroExample>, Lane)> {
        if prefer_history {
            for _ in 0..16 {
                if self.historical.is_empty() {
                    break;
                }
                let i = rng.index(self.historical.len());
                if let Some(ex) = self.historical[i].upgrade() {
                    return Some((ex, Lane::Historical));
                }
                self.historical.swap_remove_back(i);
            }
        }
        if self.entries.is_empty() {
            None
        } else {
            for _ in 0..32 {
                let e = &self.entries[rng.index(self.entries.len())];
                if e.trainable {
                    return Some((e.example.clone(), e.lane));
                }
            }
            self.entries
                .iter()
                .find(|e| e.trainable)
                .map(|e| (e.example.clone(), e.lane))
        }
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut rows: Vec<IndexRow> = Vec::new();
        for entry in &self.entries {
            if rows
                .last()
                .map_or(true, |r| r.source.path != entry.source.path)
            {
                rows.push(IndexRow {
                    source: (*entry.source).clone(),
                    indices: vec![],
                    lane: entry.lane,
                });
            }
            rows.last_mut().unwrap().indices.push(entry.index);
        }
        save_json_new(
            path,
            &Index {
                schema: "paisho-gen5-replay-index-v1".into(),
                rules: RULES.to_string(),
                rows,
            },
        )
    }
    #[cfg(test)]
    pub fn load(&mut self, path: &Path) -> Result<()> {
        self.load_for_model(path, false)
    }
    pub fn load_for_model(&mut self, path: &Path, spatial: bool) -> Result<()> {
        let index: Index = serde_json::from_slice(&fs::read(path)?)?;
        if index.schema != "paisho-gen5-replay-index-v1" || index.rules != RULES.as_str() {
            return Err(invalid("replay index identity mismatch"));
        }
        // At most two decoded source files in flight, using the caller's pool.
        // Ordered commit preserves FIFO, correction lifetimes and seeded draws.
        let width = rayon::current_num_threads().min(2);
        for rows in index.rows.chunks(width) {
            let loaded: Vec<_> = rows
                .par_iter()
                .map(|row| load_source(row, spatial).map_err(|e| e.to_string()))
                .collect();
            for (row, loaded) in rows.iter().zip(loaded) {
                let (saved, mut positions) = loaded.map_err(invalid)?;
                let source = Arc::new(row.source.clone());
                for &index in &row.indices {
                    let s = saved
                        .get(index)
                        .ok_or_else(|| invalid("replay index out of bounds"))?;
                    let mut example = s.example_for_rules_with_trusted_q(RULES,self.trusted_action_values)?;
                    if self.loop_repair && s.evidence.is_none() && s.reason == "fresh-reanalysis-q"
                    {
                        example.value_weight = 0.25;
                    }
                    if self.coherent_policy
                        && s.evidence
                            .as_ref()
                            .is_some_and(TargetEvidence::legacy_coupled_policy)
                    {
                        example.policy_weight = 0.;
                    }
                    if let Some(map) = &mut positions {
                        if let Some(state) = map.remove(&index) {
                            example.state = state;
                        }
                    }
                    self.push(
                        Arc::new(example),
                        source.clone(),
                        index,
                        row.lane,
                        s.correction_priority,
                        !spatial || s.reason != "repetition-training-loss",
                    );
                }
            }
        }
        Ok(())
    }
}
type LoadedSource = (Vec<SavedMicroExample>, Option<HashMap<usize, Vec<f64>>>);
fn load_source(row: &IndexRow, spatial: bool) -> Result<LoadedSource> {
    let bytes = fs::read(&row.source.path)?;
    if sha256(&bytes) != row.source.sha256 {
        return Err(invalid("replay target changed"));
    }
    let saved = decode_examples(&bytes)?;
    drop(bytes);
    let positions = if spatial
        && row
            .indices
            .iter()
            .any(|i| saved.get(*i).is_some_and(|s| s.state.len() == 128))
    {
        Some(spatial_positions(&row.source, &saved, &row.indices)?)
    } else {
        None
    };
    Ok((saved, positions))
}
/// Old dense targets remain immutable. Rehydrate only the missing board map,
/// validating both the archived PSR digest and the original 128 input values.
fn spatial_positions(
    source: &Source,
    saved: &[SavedMicroExample],
    wanted: &[usize],
) -> Result<HashMap<usize, Vec<f64>>> {
    let name = source
        .path
        .file_name()
        .and_then(|s| s.to_str())
        .and_then(|s| s.strip_suffix(".targets.json.gz"))
        .ok_or_else(|| invalid("cannot locate replay PSR"))?;
    let dir = source.path.parent().unwrap();
    let bytes = fs::read(dir.join(format!("{name}.psr")))?;
    let receipt: serde_json::Value =
        serde_json::from_slice(&fs::read(dir.join(format!("{name}.json")))?)?;
    if receipt["psr_sha256"] != sha256(&bytes) || receipt["targets_sha256"] != source.sha256 {
        return Err(invalid("replay PSR/target binding mismatch"));
    }
    let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
    if record.rules() != RULES {
        return Err(invalid("replay spatial rules mismatch"));
    }
    let mut selected = wanted.to_vec();
    selected.sort_by_key(|i| saved[*i].decision);
    let mut position = record.initial_position();
    let mut next = 0;
    let mut result = HashMap::new();
    for i in selected {
        let ex = &saved[i];
        if ex.decision == 0 || ex.decision > record.actions().len() + 1 {
            return Err(invalid("spatial replay decision missing"));
        }
        while next < ex.decision - 1 {
            position.apply(record.actions()[next])?;
            next += 1;
        }
        let features = micro_spatial_state_features(&position);
        if ex.state.len() == 128
            && ex
                .state
                .iter()
                .zip(&features)
                .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err(invalid("spatial replay legacy features changed"));
        }
        result.insert(i, features);
    }
    Ok(result)
}
/// Revalidate the human training split under Gen5. Terminal values are derived
/// anew; historical search-Q targets and all held-out games are excluded.
pub(super) fn human(
    path: &Path,
    output: &Path,
    pool: &rayon::ThreadPool,
    spatial: bool,
) -> Result<Vec<Arc<MicroExample>>> {
    let data = crate::compact_learning::load_dataset(path)?;
    let dataset_hash = sha256(&fs::read(path)?);
    let started = paisho_platform::training_time::now();
    let rows=pool.install(||data.games.par_iter().filter(|g|!g.held_out).map(|game| {
        (||->Result<(serde_json::Value,Vec<Arc<MicroExample>>)>{
            let original=&game.originals[0];let bytes=fs::read(&original.path)?;
            if sha256(&bytes)!=original.sha256{return Err(invalid("human original changed"));}
            let old:GameRecord=std::str::from_utf8(&bytes)?.parse()?;
            if sha256(old.to_string().as_bytes())!=game.game_sha256 || old.actions().len()!=game.decisions || old.rules().as_str()!=data.rules {return Err(invalid("human canonical identity mismatch"));}
            let (record,terminal)=old.replay_prefix_with_rules(RULES)?;
            let outcome=match terminal.outcome(){GameOutcome::Win(p)=>p.code().to_string(),GameOutcome::Draw=>"draw".into(),GameOutcome::Ongoing=>{
                if record.actions().len()!=old.actions().len(){return Err(invalid("incomplete human prefix"));}
                let ext=game.external_outcome.as_ref().ok_or_else(||invalid("human game lacks a terminal result"))?;
                if ext.record_sha256!=game.game_sha256 || ext.outcome!=game.outcome {return Err(invalid("unbound external human result"));}ext.outcome.clone()
            }};
            let mut position=record.initial_position();let mut wanted=game.examples.iter().peekable();let mut examples=vec![];
            for (decision,&action) in record.actions().iter().enumerate(){
                if wanted.peek().is_some_and(|e|e.decision_index==decision){
                    let old=wanted.next().unwrap();if old.perspective!=position.to_move().code().to_string(){return Err(invalid("human perspective mismatch"));}
                    let legal=paisho_core::legal_actions(&position);let index=legal.iter().position(|a|*a==action).ok_or_else(||invalid("illegal human action"))?;
                    let mut policy=vec![0.0;legal.len()];policy[index]=1.0;
                    let value=if outcome=="draw"{0.0}else if outcome==position.to_move().code().to_string(){1.0}else{-1.0};
                    let ex=MicroExample{ structured: Vec::new(), policy_support: false, action_values: vec![], value_weight: 1.0, sequence_source:paisho_ai::sequence_source(&format!("human/{}",game.game_sha256)),state:if spatial {micro_spatial_state_features(&position)}else{micro_state_features(&position).to_vec()},actions:legal.iter().map(|a|micro_action_features(&position,*a)).collect(),policy,value,policy_weight:1.0};ex.validate().map_err(invalid)?;examples.push(Arc::new(ex));
                }position.apply(action)?;
            }
            let receipt=serde_json::json!({"source_sha256":game.game_sha256,"split_identity":game.split_identity_sha256,"held_out":false,"decisions_before":old.actions().len(),"decisions_after":record.actions().len(),"outcome_before":game.outcome,"outcome_after":outcome,"examples":examples.len(),"gen5_psr_sha256":sha256(record.to_string().as_bytes())});
            Ok((receipt,examples))
        })().map_err(|e|e.to_string())
    }).collect::<Vec<_>>());
    let rows = rows
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(invalid)?;
    let receipts: Vec<_> = rows.iter().map(|(r, _)| r).collect();
    save_json_new(
        &output.join("human-revalidation.json"),
        &serde_json::json!({"rules":RULES.as_str(),"dataset_sha256":dataset_hash,"held_out_used":0,"games":receipts,"seconds":paisho_platform::training_time::elapsed(started).as_secs_f64()}),
    )?;
    Ok(rows.into_iter().flat_map(|(_, e)| e).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correction_refs_expire_on_fifo_eviction_even_with_live_minibatch_arcs() {
        let o = Options {
            replay_capacity: 2,
            correction_capacity: 1,
            ..Default::default()
        };
        let mut m = Memory::new(&o);
        let ex = Arc::new(
            super::super::super::tactics::fixture()
                .example_for_rules(RULES)
                .unwrap(),
        );
        let held = ex.clone();
        m.add(vec![ex], Path::new("a"), "hash", Lane::Selfplay, &[true]);
        assert_eq!(m.correction_len(), 1);
        let mut rng = StableRng::new(1);
        assert!(m.draw_correction(&mut rng).is_some());
        m.add(
            vec![held.clone(); 2],
            Path::new("b"),
            "hash",
            Lane::Selfplay,
            &[false, false],
        );
        assert_eq!(m.correction_len(), 0);
        assert!(m.draw_correction(&mut rng).is_none());
        assert_eq!(held.value, 1.0);
    }
    #[test]
    fn correction_flags_survive_durable_replay_and_pool_capacity() {
        let dir =
            std::env::temp_dir().join(format!("paisho-correction-replay-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let target = dir.join("targets.json.gz");
        let mut s = super::super::super::tactics::fixture();
        s.correction_priority = true;
        let saved = vec![s; 3];
        save_examples_new(&target, &saved).unwrap();
        let hash = sha256(&fs::read(&target).unwrap());
        let o = Options {
            replay_capacity: 3,
            correction_capacity: 2,
            ..Default::default()
        };
        let mut m = Memory::new(&o);
        m.add(
            saved
                .iter()
                .map(|s| Arc::new(s.example_for_rules(RULES).unwrap()))
                .collect(),
            &target,
            &hash,
            Lane::Selfplay,
            &[true, true, true],
        );
        assert_eq!(m.correction_len(), 2);
        let index = dir.join("index.json");
        m.save(&index).unwrap();
        let mut restored = Memory::new(&o);
        restored.load(&index).unwrap();
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.correction_len(), 2);
        let (ex, lane) = restored.draw_correction(&mut StableRng::new(1)).unwrap();
        assert_eq!(ex.value, 1.0);
        assert_eq!(lane, Lane::Selfplay);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn bounded_parallel_restore_preserves_spatial_fifo_and_seeded_recalls() {
        let dir =
            std::env::temp_dir().join(format!("paisho-parallel-replay-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let old: GameRecord =
            include_str!("../../../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let (record, _) = old.replay_prefix_with_rules(RULES).unwrap();
        let mut p = record.initial_position();
        let mut examples = vec![];
        for (i, &action) in record.actions().iter().take(8).enumerate() {
            let mut s = super::super::super::tactics::fixture();
            let legal = paisho_core::legal_actions(&p);
            s.tactical = None;
            s.correction_priority = false;
            s.decision = i + 1;
            s.source_run = dir.to_string_lossy().into_owned();
            s.game_id = (i / 2).to_string();
            s.state = paisho_ai::micro_state_features(&p).to_vec();
            s.actions = legal.iter().map(ToString::to_string).collect();
            s.action_features = legal
                .iter()
                .map(|a| micro_action_features(&p, *a).to_vec())
                .collect();
            s.new_visits = vec![0; legal.len()];
            s.new_visits[0] = 1;
            s.policy = vec![0.; legal.len()];
            s.policy[0] = 1.;
            s.policy_raw_visits.clear();
            s.policy_pruned_visits.clear();
            examples.push(s);
            p.apply(action).unwrap();
        }
        let mut rows = vec![];
        for (i, saved) in examples.chunks(2).enumerate() {
            let stem = format!("game-{i}");
            let path = dir.join(format!("{stem}.targets.json.gz"));
            save_examples_new(&path, saved).unwrap();
            let hash = sha256(&fs::read(&path).unwrap());
            let psr = record.to_string();
            fs::write(dir.join(format!("{stem}.psr")), &psr).unwrap();
            save_json_new(
                &dir.join(format!("{stem}.json")),
                &serde_json::json!({"psr_sha256":sha256(psr.as_bytes()),"targets_sha256":hash}),
            )
            .unwrap();
            rows.push(IndexRow {
                source: Source { path, sha256: hash },
                indices: vec![1, 0],
                lane: if i % 2 == 0 {
                    Lane::Historical
                } else {
                    Lane::Selfplay
                },
            });
        }
        let index = dir.join("index.json");
        save_json_new(
            &index,
            &Index {
                schema: "paisho-gen5-replay-index-v1".into(),
                rules: RULES.to_string(),
                rows,
            },
        )
        .unwrap();
        for spatial in [false, true] {
            let mut signatures = vec![];
            for threads in [1, 2] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut memory = Memory::new(&Options {
                    replay_capacity: 5,
                    correction_capacity: 3,
                    ..Default::default()
                });
                pool.install(|| {
                    memory
                        .load_for_model(&index, spatial)
                        .map_err(|e| e.to_string())
                })
                .unwrap();
                let path = dir.join(format!("restored-{spatial}-{threads}.json"));
                memory.save(&path).unwrap();
                let encode = |e: &MicroExample| {
                    serde_json::json!({"source":e.sequence_source,
                    "state":e.state.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "actions":e.actions.iter().flatten().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "policy":e.policy.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                    "value":e.value.to_bits(),"weight":e.policy_weight.to_bits()})
                };
                let entries: Vec<_> = memory
                    .entries
                    .iter()
                    .map(|e| {
                        serde_json::json!([
                            e.index,
                            e.trainable,
                            e._correction_lifetime.is_some(),
                            encode(&e.example)
                        ])
                    })
                    .collect();
                let mut rng = StableRng::new(517);
                let mut draws = vec![];
                for _ in 0..64 {
                    let (e, lane) = memory.draw(&mut rng, true).unwrap();
                    draws.push(serde_json::json!([format!("{lane:?}"), encode(&e)]));
                    if let Some((e, lane)) = memory.draw_correction(&mut rng) {
                        draws.push(serde_json::json!([format!("{lane:?}"), encode(&e)]));
                    }
                }
                signatures.push(serde_json::json!([
                    sha256(&fs::read(path).unwrap()),
                    memory.bytes,
                    memory.evicted,
                    entries,
                    draws
                ]));
            }
            assert_eq!(signatures[0], signatures[1]);
        }
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn source_metadata_requires_headroom_when_restoring_a_byte_full_replay() {
        let ex = Arc::new(MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight: 1.0,
            sequence_source: 1,
            state: [0.; MICRO_INPUTS].to_vec(),
            actions: vec![],
            policy: vec![],
            value: 0.,
            policy_weight: 0.,
        });
        let count = 4;
        let legacy_bytes = count * (size(&ex) - std::mem::size_of::<u64>());
        let mut old = Memory::new(&Options {
            replay_capacity: count,
            replay_max_bytes: legacy_bytes,
            ..Default::default()
        });
        old.add(
            vec![ex.clone(); count],
            Path::new("legacy"),
            "hash",
            Lane::Selfplay,
            &[],
        );
        assert!(old.len() < count);
        let mut upgraded = Memory::new(&Options {
            replay_capacity: count,
            replay_max_bytes: legacy_bytes + count * std::mem::size_of::<u64>(),
            ..Default::default()
        });
        upgraded.add(
            vec![ex; count],
            Path::new("legacy"),
            "hash",
            Lane::Selfplay,
            &[],
        );
        assert_eq!(upgraded.len(), count);
        assert_eq!(upgraded.evicted, 0);
    }
    #[test]
    fn replay_has_exact_capacity_and_expired_historical_refs_are_not_drawn() {
        let o = Options {
            replay_capacity: 3,
            ..Default::default()
        };
        let mut m = Memory::new(&o);
        let ex = Arc::new(MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight: 1.0,
            sequence_source: 0,
            state: [0.0; MICRO_INPUTS].to_vec(),
            actions: vec![],
            policy: vec![],
            value: 0.25,
            policy_weight: 0.0,
        });
        m.add(vec![ex], Path::new("a"), "hash", Lane::Historical, &[]);
        for i in 0..4 {
            m.add(
                vec![Arc::new(MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
                    value_weight: 1.0,
                    sequence_source: 0,
                    state: [0.0; MICRO_INPUTS].to_vec(),
                    actions: vec![],
                    policy: vec![],
                    value: i as f64 / 4.0,
                    policy_weight: 0.0,
                })],
                Path::new("b"),
                "hash",
                Lane::Selfplay,
                &[],
            );
        }
        assert_eq!(m.len(), 3);
        assert_eq!(m.evicted, 2);
        let mut rng = StableRng::new(1);
        for _ in 0..20 {
            assert_eq!(m.draw(&mut rng, true).unwrap().1, Lane::Selfplay);
        }
    }
}
