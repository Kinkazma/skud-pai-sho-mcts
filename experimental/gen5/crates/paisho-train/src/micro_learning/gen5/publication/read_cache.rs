//! Scores use immutable snapshots; lazy successor cells share exact value weights.
//! No pruning result is stored as a value. Complete coefficient reads remain exact.
use super::*;
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex, OnceLock,
    },
};
mod lazy;
pub(super) use lazy::{CoupledLogits, ValueRow};
pub(super) type ValueTable = Vec<Option<Arc<ValueRow>>>;
enum Prepared {
    Hit(Score),
    Miss(Option<Arc<ValueTable>>),
}
pub(super) struct Reads {
    pub(super) reuse_root_embedding: bool,
    pub(super) compact_inputs: bool,
    pub(super) joint_panels: bool,
    pub(super) parallel_prefilter: bool,
    pub(super) prefilter_audit: Option<Arc<prefilter::Audit>>,
    original: Option<Box<super::transaction_bench::OriginalReads>>,
    pub(super) audit: Option<Arc<super::transaction_bench::Audit>>,
    scores: Mutex<VecDeque<(MicroModel, Score)>>,
    values: Mutex<VecDeque<(Vec<u64>, Arc<ValueTable>)>>,
    bounded_inputs: OnceLock<Vec<bool>>,
    lazy: bool,
    forward_reads: Arc<AtomicUsize>,
}
impl Default for Reads {
    fn default() -> Self {
        // Targeted parallel materialization preserves exact coefficients while
        // avoiding serial deferred forwards during margin repair. Five frozen
        // complete transactions pass bit parity and preparation + control cost.
        Self::new(true)
    }
}
impl Reads {
    /// False is the eager diagnostic comparator; all values still use the same
    /// row cache representation and scorer arithmetic.
    pub fn new(lazy: bool) -> Self {
        Self {
            reuse_root_embedding: false,
            // Frozen full-control ABBA comparisons retain every score, model
            // bit and registry decision. Configure OFF only in the comparator.
            compact_inputs: true,
            joint_panels: true,
            // Four frozen OFF/ON/ON/OFF transactions preserve every result bit
            // and save 0.8–1.5 seconds per publication on this machine.
            parallel_prefilter: true,
            prefilter_audit: None,
            original: None,
            audit: None,
            scores: Default::default(),
            values: Default::default(),
            bounded_inputs: Default::default(),
            lazy,
            forward_reads: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub(super) fn original() -> Self {
        let mut out = Self::new(false);
        out.original = Some(Box::new(super::transaction_bench::OriginalReads::default()));
        out
    }
    /// Prefill only coefficients the caller will necessarily consume. Selection
    /// stays lazy too: the eager backend does not even inspect the iterator.
    /// No value or reduction arithmetic moves out of its existing implementation.
    pub(super) fn prefill<'a>(
        &self,
        logits: impl Iterator<Item = &'a CoupledLogits>,
        parallel: Option<&cpu::Ordered>,
    ) {
        if !self.lazy || self.original.is_some() {
            return;
        }
        let rows: Vec<_> = logits.filter(|row| !row.is_materialized()).cloned().collect();
        if let Some(pool) = parallel {
            let _: Vec<()> = pool.map_owned(
                rows,
                CoupledLogits::action_count,
                CoupledLogits::materialize,
            );
        } else {
            for row in rows {
                row.materialize();
            }
        }
    }
    pub fn evaluate(
        &self,
        rows: &[Arc<Witness>],
        model: &MicroModel,
        beta: f64,
        parallel: Option<&cpu::Ordered>,
    ) -> Result<Score> {
        let score = self.evaluate_backend(rows, model, beta, parallel)?;
        if let Some(audit) = &self.audit { audit.score(model, &score); }
        Ok(score)
    }
    fn evaluate_backend(
        &self, rows: &[Arc<Witness>], model: &MicroModel, beta: f64,
        parallel: Option<&cpu::Ordered>,
    ) -> Result<Score> {
        match self.prepare(rows, model, beta, parallel)? {
            Prepared::Hit(score) => Ok(score),
            Prepared::Miss(values) => {
                let score = measure_ordered(rows, model, beta, true, parallel, values, self.lazy,
                    self.reuse_root_embedding, self.compact_inputs)?;
                self.store(model, score.clone());
                Ok(score)
            }
        }
    }
    fn prepare(&self, rows: &[Arc<Witness>], model: &MicroModel, beta: f64,
        parallel: Option<&cpu::Ordered>) -> Result<Prepared> {
        if let Some(original) = &self.original { return original.evaluate(rows, model, beta, parallel).map(Prepared::Hit); }
        if let Some((_, score)) = self
            .scores
            .lock()
            .unwrap()
            .iter()
            .find(|(m, _)| m.shares_storage_with(model))
        {
            return Ok(Prepared::Hit(score.clone()));
        }
        let values = if model.has_deep_value() {
            let key: Vec<u64> = std::iter::once(model.parameters().len() as u64)
                .chain(
                    model
                        .parameters()
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| branches::value_parameter(*i))
                        .map(|(_, w)| w.to_bits()),
                )
                .collect();
            let old = self
                .values
                .lock()
                .unwrap()
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.clone());
            Some(if let Some(old) = old {
                old
            } else {
                // Input bounds are model-independent; scan each immutable
                // successor vector only once, rather than once per value model.
                if self.bounded_inputs.get().is_none() {
                    let mut bounds = Vec::with_capacity(rows.len());
                    for r in rows {
                        if r.example.value != 1. {
                            bounds.push(false);
                            continue;
                        }
                        if r.successors.get().is_none() {
                            let _ = r.successors.set(successor_inputs(r, model)?);
                        }
                        bounds.push(lazy::bounded_inputs(r.successors.get().unwrap()));
                    }
                    let _ = self.bounded_inputs.set(bounds);
                }
                let bounds = self.bounded_inputs.get().unwrap();
                if bounds.len() != rows.len() {
                    return Err(invalid("successor cache witness count changed"));
                }
                let weights_safe = lazy::bounded_weights(model);
                let table: Arc<ValueTable> = Arc::new(
                    rows.iter()
                        .enumerate()
                        .map(|(i, r)| {
                            (r.example.value == 1.).then(|| {
                                Arc::new(ValueRow::new(
                                    r.clone(),
                                    model,
                                    weights_safe && bounds[i],
                                    self.forward_reads.clone(),
                                ))
                            })
                        })
                        .collect(),
                );
                let mut cache = self.values.lock().unwrap();
                if cache.len() == 4 {
                    cache.pop_front();
                }
                cache.push_back((key, table.clone()));
                table
            })
        } else {
            None
        };
        Ok(Prepared::Miss(values))
    }
    fn store(&self, model: &MicroModel, score: Score) {
        let mut cache = self.scores.lock().unwrap();
        if cache.len() == 8 {
            cache.pop_front();
        }
        cache.push_back((model.clone(), score));
    }
    pub fn value_forwards(&self) -> usize {
        self.forward_reads.load(Ordering::Relaxed)
    }
    pub fn loaded_value_cells(&self) -> usize {
        self.values
            .lock()
            .unwrap()
            .iter()
            .flat_map(|(_, rows)| rows.iter().flatten())
            .map(|r| r.loaded())
            .sum()
    }
}

/// Feed independent panels into one existing worker queue, so a short panel's
/// last expensive row does not leave the remaining workers idle. The individual
/// caches, row order, reductions, errors and final score audit stay per panel.
pub(super) fn evaluate_panels(panels: &[(&Reads, &[Arc<Witness>], f64)], model: &MicroModel,
    parallel: &cpu::Ordered) -> Result<Vec<Score>> {
    struct Job {
        row: Arc<Witness>, values: Option<Arc<ValueRow>>, beta: f64,
        lazy: bool, reuse: bool, compact: bool,
    }
    let mut prepared = Vec::with_capacity(panels.len());
    let mut jobs = vec![];
    for (reads, rows, beta) in panels {
        let item = reads.prepare(rows, model, *beta, Some(parallel));
        if let Ok(Prepared::Miss(values)) = &item {
            for (i, row) in rows.iter().enumerate() {
                jobs.push(Job { row: row.clone(), values: values.as_ref().and_then(|v|v[i].clone()),
                    beta: *beta, lazy: reads.lazy, reuse: reads.reuse_root_embedding,
                    compact: reads.compact_inputs });
            }
        }
        // Keep an earlier panel's ordinary error ahead of later worker panics.
        let failed = item.is_err();
        prepared.push(item);
        if failed { break; }
    }
    let m = model.clone();
    let parts = parallel.map_owned(jobs, |j|j.row.example.actions.len(), move |j| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            measure_one(&j.row,&m,j.beta,true,j.values.clone(),j.lazy,j.reuse,j.compact)
                .map_err(|e|e.to_string())
        }))
    });
    let mut parts = parts.into_iter();
    let mut scores = Vec::with_capacity(panels.len());
    for ((reads, panel_rows, _), prepared) in panels.iter().zip(prepared) {
        let score = match prepared? {
            Prepared::Hit(score) => score,
            Prepared::Miss(_) => {
                let mut rows = Vec::with_capacity(panel_rows.len());
                for _ in *panel_rows {
                    let part = match parts.next().expect("missing panel result") {
                        Ok(part) => part,
                        Err(panic) => std::panic::resume_unwind(panic),
                    };
                    rows.push(part);
                }
                let score = fold_scores(panel_rows, rows)?;
                reads.store(model, score.clone());
                score
            }
        };
        if let Some(audit)=&reads.audit { audit.score(model,&score); }
        scores.push(score);
    }
    Ok(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows(model: &MicroModel, n:usize) -> Vec<Arc<Witness>> {
        let record: GameRecord=include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gen5-multiple-proved-wins.psr")).parse().unwrap();
        let position=record.replay().unwrap();
        let legal=paisho_core::legal_actions(&position);
        (0..n).map(|i| {
            let valid:Vec<_>=legal.iter().enumerate().map(|(j,_)|j==i%legal.len()).collect();
            Arc::new(Witness {position:position.clone(),valid:valid.clone(),successors:Default::default(),
                example:Arc::new(MicroExample { structured: Vec::new(),state:model.state_features(&position),
                    actions:legal.iter().map(|a|micro_action_features(&position,*a)).collect(),
                    policy:valid.iter().map(|b|usize::from(*b) as f64).collect(),value:[1.,0.,-1.][i%3],
                    action_values:vec![],value_weight:0.,policy_weight:1.,sequence_source:0,policy_support:false})})
        }).collect()
    }
    fn bits(a:&Score,b:&Score) {
        assert_eq!(a.raw,b.raw);assert_eq!(a.coupled,b.coupled);
        assert_eq!(a.mass.to_bits(),b.mass.to_bits());assert_eq!(a.value_mse.to_bits(),b.value_mse.to_bits());
        for (a,b) in a.errors.iter().zip(b.errors.iter()) {assert_eq!(a.to_bits(),b.to_bits());}
        for (a,b) in a.priors.iter().zip(b.priors.iter()) {
            assert_eq!(a.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        for (a,b) in a.coupled_logits.iter().zip(b.coupled_logits.iter()) {
            assert_eq!(a.as_slice().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                b.as_slice().iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
    }
    #[test]
    fn relational_lazy_scorer_matches_eager_values_and_invalidates_shared_encoder() {
        let base=MicroModel::seeded(31).with_relational(72);
        let mut weights=base.parameters().to_vec();*weights.last_mut().unwrap()=0.2;
        let n=weights.len();weights[n-33]=0.3;
        let model=MicroModel::from_parameters(weights.clone()).unwrap();let rows=rows(&model,4);
        assert!(lazy::bounded_weights(&model));
        let eager=Reads::original();let cached=Reads::new(true);
        for j in [0,1,7] {
            weights[paisho_ai::MICRO_RELATIONAL_START+j]+=0.01;
            let m=MicroModel::from_parameters(weights.clone()).unwrap();
            let expected=eager.evaluate(&rows,&m,2.,None).unwrap();let actual=cached.evaluate(&rows,&m,2.,None).unwrap();
            bits(&expected,&actual);
        }
    }
    #[test]
    fn combined_panels_keep_bits_with_partial_cache_hits_and_changed_models() {
        let model=MicroModel::seeded(31).with_neural_memory(72);
        let mut w=model.parameters().to_vec();*w.last_mut().unwrap()=0.25;
        let model=MicroModel::from_parameters(w).unwrap();
        let a=rows(&model,5);let b=rows(&model,17);
        for workers in [1,3] {
            let pool=cpu::Ordered::new(&[cpu::build_pool(workers,None).unwrap().0]);
            let old=[Reads::new(true),Reads::new(true)];
            let new=[Reads::new(true),Reads::new(true)];
            for i in 0..12 {
                let mut w=model.parameters().to_vec();w[0]+=i as f64*0.0001;
                let m=MicroModel::from_parameters(w).unwrap();
                if i%2==0 {new[0].evaluate(&a,&m,16.,Some(&pool)).unwrap();}
                let expected=[old[0].evaluate(&a,&m,16.,Some(&pool)).unwrap(),old[1].evaluate(&b,&m,16.,Some(&pool)).unwrap()];
                for _ in 0..2 {
                    let actual=evaluate_panels(&[(&new[0],&a,16.),(&new[1],&b,16.)],&m,&pool).unwrap();
                    for (a,b) in expected.iter().zip(&actual) {bits(a,b);}
                }
            }
            assert_eq!(new[0].scores.lock().unwrap().len(),8);
            assert_eq!(new[1].scores.lock().unwrap().len(),8);
        }
    }
    #[test]
    fn combined_panels_preserve_failure_precedence_and_do_not_cache_failures() {
        let model=MicroModel::seeded(31).with_deep_value(72);
        let pool=cpu::Ordered::new(&[cpu::build_pool(2,None).unwrap().0]);
        let good=rows(&model,1).remove(0);
        let mut erroneous=(*good).clone();
        Arc::make_mut(&mut erroneous.example).state.fill(f64::NAN);
        let mut panicking=(*good).clone();panicking.valid.clear();
        let error_rows=vec![Arc::new(erroneous)];
        let panic_rows=vec![Arc::new(panicking)];
        let first=Reads::new(true);let second=Reads::new(true);
        let original=first.evaluate(&error_rows,&model,16.,Some(&pool)).err().unwrap().to_string();
        let combined=evaluate_panels(&[(&first,&error_rows,16.),(&second,&panic_rows,16.)],&model,&pool)
            .err().unwrap().to_string();
        assert_eq!(original,combined);
        assert!(first.scores.lock().unwrap().is_empty());assert!(second.scores.lock().unwrap().is_empty());
        // Within one panel, Ordered propagates a worker panic before ordinary
        // row errors. Preserve that behavior as well as the cross-panel order.
        let both=vec![error_rows[0].clone(),panic_rows[0].clone()];
        for joint in [false,true] {
            let reads=Reads::new(true);
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if joint {evaluate_panels(&[(&reads,&both,16.)],&model,&pool).map(|_|())}
                else {reads.evaluate(&both,&model,16.,Some(&pool)).map(|_|())}
            })).is_err());
            assert!(reads.scores.lock().unwrap().is_empty());
        }
    }
}
