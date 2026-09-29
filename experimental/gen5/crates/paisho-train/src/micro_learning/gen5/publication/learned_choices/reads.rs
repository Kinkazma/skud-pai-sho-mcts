//! Bounded exact raw reads for the immutable active population.
use super::*;
use std::{any::Any, sync::Mutex};

// The measured transaction found 37 repeated reads in a window of eight models.
// Match that existing fixed-panel cache bound; retrieval banks remain shared.
const MODELS: usize = 8;
type Outcome = std::result::Result<std::result::Result<bool, String>, Box<dyn Any + Send>>;

struct Entry {
    model: MicroModel,
    rows: Vec<Arc<Witness>>,
    wins: Arc<Vec<bool>>,
}
impl Entry {
    fn matches(&self, model: &MicroModel, rows: &[Arc<Witness>]) -> bool {
        self.model.shares_storage_with(model) && self.rows.len() == rows.len()
            && self.rows.iter().zip(rows).all(|(a, b)| Arc::ptr_eq(a, b))
    }
}
#[derive(Default)]
struct Cache {
    entries: VecDeque<Entry>,
    hits: usize,
    misses: usize,
    computed_rows: usize,
}

#[derive(Clone, Default)]
pub(super) struct Reads {
    enabled: bool,
    parallel: Option<cpu::Ordered>,
    // Sharing across transactional clones is safe because the complete active
    // population is in each key, not a generation number that forks can reuse.
    cache: Arc<Mutex<Cache>>,
}

impl Reads {
    pub(super) fn new(enabled: bool, parallel: Option<&cpu::Ordered>) -> Self {
        Self { enabled, parallel: if enabled { parallel.cloned() } else { None },
            cache: Default::default() }
    }
    pub(super) fn enabled(&self) -> bool { self.enabled }
    pub(super) fn progress(&self) -> serde_json::Value {
        let c = self.cache.lock().unwrap();
        serde_json::json!({"enabled":self.enabled,"parallel":self.parallel.is_some(),
            "capacity":MODELS,"entries":c.entries.len(),"hits":c.hits,
            "misses":c.misses,"computed_rows":c.computed_rows})
    }
    pub(super) fn evaluate(&self, model: &MicroModel, rows: Vec<Arc<Witness>>) -> Result<Arc<Vec<bool>>> {
        {
            let mut c = self.cache.lock().unwrap();
            if let Some(index) = c.entries.iter().position(|e| e.matches(model, &rows)) {
                let entry = c.entries.remove(index).unwrap();
                let wins = entry.wins.clone();
                c.entries.push_back(entry);
                c.hits += 1;
                return Ok(wins);
            }
            c.misses += 1;
            c.computed_rows += rows.len();
        }
        // No lock is held during inference or when a panic is rethrown.
        let wins = if let Some(pool) = &self.parallel {
            let m = model.clone();
            let outcomes = pool.map_owned(rows.clone(), |r| r.example.actions.len(), move |r| {
                // Ordered normally rethrows any worker panic before returning
                // its Vec. Capture inside the job so an earlier ordinary error
                // still wins over a later panic, exactly as in the serial path.
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    raw_wins(&m, r).map_err(|e| e.to_string())
                }))
            });
            collect_ordered(outcomes)?
        } else {
            rows.iter().map(|r| raw_wins(model, r)).collect::<Result<Vec<_>>>()?
        };
        let wins = Arc::new(wins);
        let mut c = self.cache.lock().unwrap();
        // Only fully successful evaluations are cached. Keep strong references
        // to model and rows, preventing address reuse from becoming a false hit.
        if !c.entries.iter().any(|e| e.matches(model, &rows)) {
            if c.entries.len() == MODELS { c.entries.pop_front(); }
            c.entries.push_back(Entry { model: model.clone(), rows, wins: wins.clone() });
        }
        Ok(wins)
    }
}

fn collect_ordered(outcomes: Vec<Outcome>) -> Result<Vec<bool>> {
    let mut wins = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        match outcome {
            Ok(won) => wins.push(won.map_err(invalid)?),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    Ok(wins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_errors_and_panics_keep_the_first_serial_failure() {
        let later_panic: Outcome = Err(Box::new("later panic"));
        let error = collect_ordered(vec![Ok(Ok(true)), Ok(Err("first error".into())), later_panic])
            .unwrap_err();
        assert_eq!(error.to_string(), "first error");
        let early_panic: Outcome = Err(Box::new("first panic"));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            collect_ordered(vec![early_panic, Ok(Err("later error".into()))])
        })).unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"first panic"));
    }

    #[test]
    fn immutable_model_and_exact_population_are_both_part_of_the_key() {
        let (model, proof) = super::super::tests::fixture();
        let a = verified(proof.clone(), &model, 0).unwrap().row;
        let b = verified(proof, &model, 1).unwrap().row;
        let reads = Reads::new(true, None);
        let first = reads.evaluate(&model, vec![a.clone(), b.clone()]).unwrap();
        let hit = reads.evaluate(&model.clone(), vec![a.clone(), b.clone()]).unwrap();
        assert!(Arc::ptr_eq(&first, &hit));
        assert_eq!(reads.progress()["hits"], 1);
        // Equal input values in fresh Witness allocations do not masquerade as
        // the same ordered population; this also covers separately restored rows.
        reads.evaluate(&model, vec![b.clone(), a.clone()]).unwrap();
        assert_eq!(reads.progress()["misses"], 2);
        let independent = MicroModel::from_parameters(model.parameters().to_vec()).unwrap();
        reads.evaluate(&independent, vec![a.clone(), b]).unwrap();
        assert_eq!(reads.progress()["misses"], 3);
        // Retaining no raw pointer avoids accidental hits after old data drops.
        for seed in 100..110 {
            reads.evaluate(&MicroModel::seeded(seed).with_spatial_policy(), vec![a.clone()]).unwrap();
        }
        assert_eq!(reads.progress()["entries"], MODELS);
        assert_eq!(reads.progress()["capacity"], MODELS);
    }

    #[test]
    fn retrieval_bank_identity_invalidates_same_parameter_storage() {
        let (model, proof) = super::super::tests::fixture();
        let row = verified(proof, &model, 0).unwrap().row;
        let entry = SequenceEntry { key: [0.; 64], patterns: [[0; 32]; 4], source: 1,
            game: 0, decision: 0, end_decision: 1, outcome: 1, phase: 0 };
        let bank1 = Arc::new(SequenceBank::build(vec![entry.clone()], 1, 0, 1));
        let bank2 = Arc::new(SequenceBank::build(vec![entry], 1, 0, 1));
        let a = model.clone().with_sequence_memory_owned(bank1);
        let b = model.with_sequence_memory_owned(bank2);
        assert_eq!(a.parameters().as_ptr(), b.parameters().as_ptr());
        assert!(!a.shares_storage_with(&b));
        let reads = Reads::new(true, None);
        reads.evaluate(&a, vec![row.clone()]).unwrap();
        reads.evaluate(&a, vec![row.clone()]).unwrap();
        reads.evaluate(&b, vec![row]).unwrap();
        assert_eq!(reads.progress()["hits"], 1);
        assert_eq!(reads.progress()["misses"], 2);
    }

    #[test]
    fn failed_parallel_reads_never_populate_the_success_cache() {
        let (model, proof) = super::super::tests::fixture();
        let row = verified(proof, &model, 0).unwrap().row;
        let bad = Arc::new(Witness { position: row.position.clone(), valid: vec![],
            example: row.example.clone(), successors: Default::default() });
        let expected = raw_wins(&model, &bad).unwrap_err().to_string();
        let pool = cpu::Ordered::new(&[cpu::build_pool(2, None).unwrap().0]);
        let reads = Reads::new(true, Some(&pool));
        for _ in 0..2 {
            assert_eq!(reads.evaluate(&model, vec![row.clone(), bad.clone()])
                .unwrap_err().to_string(), expected);
        }
        assert_eq!(reads.progress()["entries"], 0);
        assert_eq!(reads.progress()["hits"], 0);
        assert_eq!(reads.progress()["misses"], 2);
    }
}
