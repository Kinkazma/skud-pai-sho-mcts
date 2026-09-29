//! Bounded independent lesson cache; eviction drops reconstructible RAM only.
use super::*;
use std::collections::BTreeMap;
struct Entry {
    key: String,
    group: String,
    example: Arc<MicroExample>,
    error: f64,
    proof: bool,
    bytes: usize,
    score_epoch: Option<u64>,
    pending: Option<Arc<MicroModel>>,
}

#[cfg(test)]
mod exact_tests {
    use super::*;
    #[test]
    fn support_scores_match_eager_lazy_and_parallel_draws() {
        let pools = cpu::build_search_pools(2, 1, None).unwrap();
        let mut eager = Pool::default();
        let mut lazy = Pool::default();
        let mut parallel = Pool::default();
        parallel.enable_parallel(&pools);
        let model = MicroModel::seeded(11);
        for i in 0..12 {
            let mut state = vec![0.; 417]; state[i] = (i + 1) as f64 / 16.;
            let ex = Arc::new(MicroExample { structured: Vec::new(),
                state, actions: vec![[0.1; 32], [0.2; 32], [0.3; 32]],
                policy: if i % 3 == 1 { vec![1., 0., 0.] } else { vec![0.5, 0.5, 0.] },
                value: 0.5, policy_weight: 1., value_weight: 1., action_values: vec![],
                sequence_source: 0, policy_support: i % 3 != 2,
            });
            for pool in [&mut eager, &mut lazy, &mut parallel] {
                pool.insert(i.to_string(), (i % 2).to_string(), ex.clone(), 0., true);
            }
        }
        let mut a = StableRng::new(81);
        let mut b = StableRng::new(81);
        let mut c = StableRng::new(81);
        for _ in 0..4 {
            lazy.begin_scoring(); parallel.begin_scoring();
            for _ in 0..32 {
                let x = eager.draw(2, &mut a, &model).unwrap();
                let y = lazy.draw(2, &mut b, &model).unwrap();
                let z = parallel.draw(2, &mut c, &model).unwrap();
                assert!(x.iter().zip(&y).zip(&z).all(|((x,y),z)| Arc::ptr_eq(x,y) && Arc::ptr_eq(x,z)));
            }
            let ids = (0..12).collect::<Vec<_>>();
            lazy.materialize(&ids).unwrap(); parallel.materialize(&ids).unwrap();
            for ((x,y),z) in eager.entries.iter().zip(&lazy.entries).zip(&parallel.entries) {
                assert_eq!(x.error.to_bits(), y.error.to_bits());
                assert_eq!(x.error.to_bits(), z.error.to_bits());
                let emb = model.embed(&x.example.state);
                let p = micro_softmax(&MicroModel::logits(&emb, &x.example.actions)).unwrap();
                let p = model.memory_priors(&x.example.state, &x.example.actions, &p, 0).unwrap();
                let expected = (emb.value - x.example.value).abs() + policy_distance(&p, &x.example);
                assert_eq!(x.error.to_bits(), expected.to_bits());
            }
            lazy.end_scoring(); parallel.end_scoring();
        }
        let next = a.next_f64().to_bits();
        assert_eq!(next, b.next_f64().to_bits());
        assert_eq!(next, c.next_f64().to_bits());
    }
    #[test]
    fn lazy_scores_keep_the_original_model_across_epochs_and_bound_resident_snapshots() {
        let mut eager=Pool::default();let mut lazy=Pool::default();
        let base=MicroModel::seeded(11).with_neural_memory(19);
        for i in 0..512 {
            let mut state=vec![0.;417];state[i%417]=i as f64/1024.;
            let ex=Arc::new(MicroExample { structured: Vec::new(), policy_support: false,state,actions:vec![[0.1;32],[0.2;32]],policy:vec![0.8,0.2],value:(i%3) as f64-1.,policy_weight:1.,value_weight:1.,action_values:vec![],sequence_source:0});
            for p in [&mut eager,&mut lazy] {p.insert(i.to_string(),(i%13).to_string(),ex.clone(),0.,i%3==0);}
        }
        let mut a=StableRng::new(175);let mut b=StableRng::new(175);
        for epoch in 0..96 {
            let mut w=base.parameters().to_vec();w[4160]+=epoch as f64/1000.;
            let model=MicroModel::from_parameters(w).unwrap();
            lazy.begin_scoring();
            let x=eager.draw(1,&mut a,&model).unwrap();let y=lazy.draw(1,&mut b,&model).unwrap();
            assert!(Arc::ptr_eq(&x[0],&y[0]));
            lazy.end_scoring();
            assert!(lazy.snapshots.len()<=16);
            assert_eq!(a.next_f64().to_bits(),b.next_f64().to_bits());
        }
        lazy.materialize(&(0..lazy.entries.len()).collect::<Vec<_>>()).unwrap();
        assert!(eager.entries.iter().zip(&lazy.entries).all(|(a,b)|a.error.to_bits()==b.error.to_bits()));
        assert_eq!(lazy.peak_snapshots,16);
        assert!(lazy.evaluated<=eager.evaluated);
    }
    #[test]
    fn parallel_rescoring_and_scoped_reuse_keep_every_draw_and_error_exact() {
        let pools=cpu::build_search_pools(4,2,None).unwrap();
        let mut old=Pool::default();let mut new=Pool::default();
        new.enable_parallel(&pools);
        let mut model=MicroModel::seeded(7).with_neural_memory(19);
        let mut rows=vec![];
        for i in 0..19 {
            let mut state=vec![0.;417];state[i]=0.1+i as f64/20.;
            let ex=Arc::new(MicroExample { structured: Vec::new(), policy_support: false,state, actions:vec![[0.1;32],[0.2;32]],policy:vec![0.8,0.2],value:0.5,policy_weight:1.,value_weight:1.,action_values:vec![Some(1.),Some(-1.)],sequence_source:0});
            rows.push(ex.clone());
            for p in [&mut old,&mut new] {p.insert(i.to_string(),(i%3).to_string(),ex.clone(),0.,i%2==0);}
        }
        model.train_step(&rows[0],0.01,0.).unwrap();
        let mut a=StableRng::new(17);let mut b=StableRng::new(17);
        for epoch in 0..2 {
            new.begin_scoring();
            for j in 0..80 {
                if j==40 {
                    let mut ex=rows[2].as_ref().clone();ex.value=-0.25;
                    let ex=Arc::new(ex);
                    for p in [&mut old,&mut new] {p.insert("2".into(),"2".into(),ex.clone(),0.,true);}
                }
                let x=old.draw(3,&mut a,&model).unwrap();let y=new.draw(3,&mut b,&model).unwrap();
                assert!(x.iter().zip(y).all(|(x,y)|Arc::ptr_eq(x,&y)));
                assert_eq!(old.refresh,new.refresh);

            }
            new.materialize(&(0..new.entries.len()).collect::<Vec<_>>()).unwrap();
            assert!(old.entries.iter().zip(&new.entries).all(|(a,b)|a.error.to_bits()==b.error.to_bits()));
            new.end_scoring();
            if epoch==0 {model.train_step(&rows[3],0.02,0.).unwrap();}
        }
        assert!(new.reused>1000);
        assert!(new.evaluated<old.evaluated/10);
        assert_eq!(a.next_f64().to_bits(),b.next_f64().to_bits());
    }
}
#[derive(Default)]
pub(super) struct Pool {
    entries: Vec<Entry>,
    cursor: usize,
    bytes: usize,
    refresh: usize,
    parallel: Option<cpu::Ordered>,
    epoch: u64,
    active_epoch: Option<u64>,
    evaluated: usize,
    reused: usize,
    snapshots: std::collections::VecDeque<Arc<MicroModel>>,
    discarded: usize,
    peak_snapshots: usize,
}
impl Pool {
    pub fn enable_parallel(&mut self,pools:&[Arc<rayon::ThreadPool>]) {self.parallel=Some(cpu::Ordered::new(pools));}
    pub fn begin_scoring(&mut self) {self.epoch=self.epoch.checked_add(1).expect("recall scoring epoch overflow");self.active_epoch=Some(self.epoch);}
    pub fn end_scoring(&mut self) {self.active_epoch=None;}
    pub fn scoring_progress(&self)->serde_json::Value {serde_json::json!({"evaluated":self.evaluated,"reused":self.reused,"unused_scores_discarded":self.discarded,"pending_models":self.snapshots.len(),"peak_pending_models":self.peak_snapshots})}

    pub fn contains(&self, key: &str) -> bool {
        self.entries.iter().any(|e| e.key == key)
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn groups(&self) -> usize {
        self.entries
            .iter()
            .map(|e| &e.group)
            .collect::<std::collections::HashSet<_>>()
            .len()
    }
    pub fn insert(
        &mut self,
        key: String,
        group: String,
        mut example: Arc<MicroExample>,
        error: f64,
        proof: bool,
    ) {
        let bytes = std::mem::size_of::<Entry>()
            + std::mem::size_of::<MicroExample>()
            + example.state.capacity() * 8
            + example.actions.capacity() * 32 * 8
            + example.policy.capacity() * 8
            + example.action_values.capacity() * std::mem::size_of::<Option<f64>>()
            + example.structured.capacity()*std::mem::size_of::<Option<MicroStructuredTarget>>()
            + example.structured.iter().flatten().map(|t|match &t.threat {MicroThreatEvidence::Present{reply}=>reply.capacity(),_=>0}).sum::<usize>()
            + key.capacity()
            + group.capacity();
        if bytes > 64 * 1024 * 1024 {
            return;
        }
        if let Some(i) = self.entries.iter().position(|e| e.key == key) {
            // A weaker empirical revision must never replace an exact proof.
            if self.entries[i].proof && !proof {
                return;
            }
            if !proof && self.entries[i].example.value_weight == 1. && example.value_weight < 1. {
                let value = self.entries[i].example.value;
                let ex = Arc::make_mut(&mut example);
                ex.value = value;
                ex.value_weight = 1.;
            }
            self.bytes -= self.entries.swap_remove(i).bytes;
        }
        while !self.entries.is_empty()
            && (self.entries.len() >= 512 || self.bytes + bytes > 64 * 1024 * 1024)
        {
            let i = self.cursor % self.entries.len();
            self.bytes -= self.entries.swap_remove(i).bytes;
            self.cursor += 1;
        }
        self.bytes += bytes;
        self.entries.push(Entry {
            key,
            group,
            example,
            error,
            score_epoch: None,
            pending: None,
            proof,
            bytes,
        });
    }
    fn materialize(&mut self,ids:&[usize])->Result<()> {
        let pending=ids.iter().copied().filter(|&i|self.entries[i].pending.is_some()).collect::<Vec<_>>();
        let jobs=pending.iter().map(|&i| {
            let e=&self.entries[i];(e.pending.as_ref().unwrap().clone(),e.example.clone())
        }).collect::<Vec<_>>();
        let score=|(model,x):&(Arc<MicroModel>,Arc<MicroExample>)| -> std::result::Result<f64,String> {
            let emb=model.embed(&x.state);
            let mut error=(emb.value-x.value).abs();
            if !x.actions.is_empty() {
                let p=micro_softmax(&MicroModel::logits(&emb,&x.actions))?;
                let p=model.memory_priors(&x.state,&x.actions,&p,x.sequence_source)?;
                error+=policy_distance(&p,x);
            }
            Ok(error)
        };
        let values:Vec<_>=match &self.parallel {Some(p)=>p.map_owned(jobs,|(_,e)|e.actions.len(),score),None=>jobs.iter().map(score).collect()};
        for (i,value) in pending.into_iter().zip(values) {
            self.entries[i].error=value.map_err(invalid)?;
            self.entries[i].pending=None;
            self.evaluated+=1;
        }
        Ok(())
    }
    pub fn draw(
        &mut self,
        n: usize,
        rng: &mut StableRng,
        model: &MicroModel,
    ) -> Result<Vec<Arc<MicroExample>>> {
        if self.entries.is_empty() {
            return Ok(vec![]);
        }
        // Keep the original refresh cursor and draw order. Independent scores
        // use the frozen model; repeats within one recall request reuse that score.
        let mut selected=vec![];
        for _ in 0..8.min(self.entries.len()) {
            let i=self.refresh%self.entries.len();self.refresh+=1;
            if self.active_epoch.is_some() && self.entries[i].score_epoch==self.active_epoch {self.reused+=1;}
            else {selected.push(i);}
        }
        if self.active_epoch.is_some() {
            // An overwritten, unobserved score has no effect on a draw. Keep its
            // original immutable model until the score is needed, never rescore
            // it against newer weights. At most 16 snapshots remain resident.
            self.snapshots.retain(|m| self.entries.iter().any(|e| e.pending.as_ref().is_some_and(|p|Arc::ptr_eq(p,m))));
            let snapshot=if let Some(m)=self.snapshots.iter().find(|m|m.shares_storage_with(model)) {m.clone()} else {
                if self.snapshots.len()==16 {
                    let old=self.snapshots.pop_front().unwrap();
                    let ids=self.entries.iter().enumerate().filter(|(_,e)|e.pending.as_ref().is_some_and(|m|Arc::ptr_eq(m,&old))).map(|(i,_)|i).collect::<Vec<_>>();
                    self.materialize(&ids)?;
                }
                let m=Arc::new(model.clone());self.snapshots.push_back(m.clone());
                self.peak_snapshots=self.peak_snapshots.max(self.snapshots.len());m
            };
            for i in selected {
                self.discarded+=usize::from(self.entries[i].pending.is_some());
                self.entries[i].pending=Some(snapshot.clone());
                self.entries[i].score_epoch=self.active_epoch;
            }
        } else {
            for i in &selected {self.entries[*i].pending=Some(Arc::new(model.clone()));}
            self.materialize(&selected)?;
        }
        let mut groups: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        let mut proofs: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for (i, e) in self.entries.iter().enumerate() {
            groups.entry(&e.group).or_default().push(i);
            if e.proof {
                proofs.entry(&e.group).or_default().push(i);
            }
        }
        let groups: Vec<_> = groups.into_values().collect();
        let proofs: Vec<_> = proofs.into_values().collect();
        (0..n)
            .map(|_| -> Result<Arc<MicroExample>> {
                // Half diversity; a quarter verified lessons; a quarter disagreement.
                // All strata first sample source groups uniformly, then positions.
                // Random strata preserve the mixture for singleton receipts too.
                let stratum = rng.index(4);
                let bank = if stratum == 0 && !proofs.is_empty() {
                    &proofs
                } else {
                    &groups
                };
                let group = &bank[rng.index(bank.len())];
                let mut i = group[rng.index(group.len())];
                if stratum == 2 {
                    let compared=std::array::from_fn::<_,3,_>(|_|group[rng.index(group.len())]);
                    let mut ids=vec![i];
                    for &j in &compared {if !ids.contains(&j) {ids.push(j);}}
                    self.materialize(&ids)?;
                    for j in compared {
                        if self.entries[j].error > self.entries[i].error {i=j;}
                    }
                }
                Ok(self.entries[i].example.clone())
            })
            .collect()
    }
}
