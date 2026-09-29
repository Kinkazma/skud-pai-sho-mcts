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
        + e.actions.capacity() * MICRO_ACTION_INPUTS * 8
        + e.policy.capacity() * 8
}
impl Memory {
    pub fn new(o: &Options) -> Self {
        Self {
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
    ) {
        let bytes = size(&example)
            + if correction && self.correction_capacity > 0 {
                2 * std::mem::size_of::<usize>()
            } else {
                0
            };
        self.bytes += bytes;
        if lane == Lane::Historical {
            self.historical.push_back(Arc::downgrade(&example));
        }
        let correction_lifetime =
            (correction && self.correction_capacity > 0).then(|| Arc::new(()));
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
            let e = &self.entries[rng.index(self.entries.len())];
            Some((e.example.clone(), e.lane))
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
    pub fn load(&mut self, path: &Path) -> Result<()> {
        let index: Index = serde_json::from_slice(&fs::read(path)?)?;
        if index.schema != "paisho-gen5-replay-index-v1" || index.rules != RULES.as_str() {
            return Err(invalid("replay index identity mismatch"));
        }
        for row in index.rows {
            if sha256(&fs::read(&row.source.path)?) != row.source.sha256 {
                return Err(invalid("replay target changed"));
            }
            let saved = load_examples(&row.source.path)?;
            let source = Arc::new(row.source);
            for index in row.indices {
                let s = saved
                    .get(index)
                    .ok_or_else(|| invalid("replay index out of bounds"))?;
                self.push(
                    Arc::new(s.example_for_rules(RULES)?),
                    source.clone(),
                    index,
                    row.lane,
                    s.correction_priority,
                );
            }
        }
        Ok(())
    }
}
/// Revalidate the human training split under Gen5. Terminal values are derived
/// anew; historical search-Q targets and all held-out games are excluded.
pub(super) fn human(
    path: &Path,
    output: &Path,
    pool: &rayon::ThreadPool,
) -> Result<Vec<Arc<MicroExample>>> {
    let data = crate::compact_learning::load_dataset(path)?;
    let dataset_hash = sha256(&fs::read(path)?);
    let started = Instant::now();
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
                    let ex=MicroExample{sequence_source:paisho_ai::sequence_source(&format!("human/{}",game.game_sha256)),state:micro_state_features(&position),actions:legal.iter().map(|a|micro_action_features(&position,*a)).collect(),policy,value,policy_weight:1.0};ex.validate().map_err(invalid)?;examples.push(Arc::new(ex));
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
        &serde_json::json!({"rules":RULES.as_str(),"dataset_sha256":dataset_hash,"held_out_used":0,"games":receipts,"seconds":started.elapsed().as_secs_f64()}),
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
    fn source_metadata_requires_headroom_when_restoring_a_byte_full_replay() {
        let ex=Arc::new(MicroExample {sequence_source:1,state:[0.;MICRO_INPUTS],actions:vec![],policy:vec![],value:0.,policy_weight:0.});
        let count=4;
        let legacy_bytes=count*(size(&ex)-std::mem::size_of::<u64>());
        let mut old=Memory::new(&Options{replay_capacity:count,replay_max_bytes:legacy_bytes,..Default::default()});
        old.add(vec![ex.clone();count],Path::new("legacy"),"hash",Lane::Selfplay,&[]);
        assert!(old.len()<count);
        let mut upgraded=Memory::new(&Options{replay_capacity:count,replay_max_bytes:legacy_bytes+count*std::mem::size_of::<u64>(),..Default::default()});
        upgraded.add(vec![ex;count],Path::new("legacy"),"hash",Lane::Selfplay,&[]);
        assert_eq!(upgraded.len(),count);assert_eq!(upgraded.evicted,0);
    }
    #[test]
    fn replay_has_exact_capacity_and_expired_historical_refs_are_not_drawn() {
        let o = Options {
            replay_capacity: 3,
            ..Default::default()
        };
        let mut m = Memory::new(&o);
        let ex = Arc::new(MicroExample {
            sequence_source: 0,
            state: [0.0; MICRO_INPUTS],
            actions: vec![],
            policy: vec![],
            value: 0.25,
            policy_weight: 0.0,
        });
        m.add(vec![ex], Path::new("a"), "hash", Lane::Historical, &[]);
        for i in 0..4 {
            m.add(
                vec![Arc::new(MicroExample {
            sequence_source: 0,
                    state: [0.0; MICRO_INPUTS],
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
