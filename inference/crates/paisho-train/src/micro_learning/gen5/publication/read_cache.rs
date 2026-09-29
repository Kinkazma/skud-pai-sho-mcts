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
pub(super) struct Reads {
    pub(super) reuse_root_embedding: bool,
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
        if let Some(original) = &self.original { return original.evaluate(rows, model, beta, parallel); }
        if let Some((_, score)) = self
            .scores
            .lock()
            .unwrap()
            .iter()
            .find(|(m, _)| m.shares_storage_with(model))
        {
            return Ok(score.clone());
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
        let score = measure_ordered(rows, model, beta, true, parallel, values, self.lazy, self.reuse_root_embedding)?;
        let mut cache = self.scores.lock().unwrap();
        if cache.len() == 8 {
            cache.pop_front();
        }
        cache.push_back((model.clone(), score.clone()));
        Ok(score)
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
