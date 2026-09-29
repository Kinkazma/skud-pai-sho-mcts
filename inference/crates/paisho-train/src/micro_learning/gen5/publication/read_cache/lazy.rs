//! Exact lazy reads: an absent value is None, never a fabricated Q or NaN.
use super::*;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// For the supported separated tanh value networks, each input affine sum has
/// at most 417 products. |x|,|w|<=1e100 keeps even their absolute sum <1e203.
/// Subsequent hidden tanhs are bounded by one; all later sums are <1e103.
/// The final sum of the two value heads is finite, so its tanh is in [-1,1].
/// This deliberately conservative domain excludes finite overflow candidates.
pub fn bounded_weights(model: &MicroModel) -> bool {
    model.has_spatial()
        && model.has_deep_value()
        && [
            MICRO_DEEP_VALUE_MODEL_SCHEMA,
            MICRO_NEURAL_MEMORY_MODEL_SCHEMA,
        ]
        .contains(&model.schema())
        && model
            .parameters()
            .iter()
            .enumerate()
            .filter(|(i, _)| branches::value_parameter(*i))
            .all(|(_, w)| w.is_finite() && w.abs() <= 1e100)
}
pub fn bounded_inputs(inputs: &[(f64, Vec<f64>)]) -> bool {
    inputs.iter().all(|(sign, state)| {
        if state.is_empty() {
            sign.is_finite() && sign.abs() <= 1.
        } else {
            (*sign == 1. || *sign == -1.)
                && [MICRO_INPUTS, MICRO_SPATIAL_INPUTS].contains(&state.len())
                && state.iter().all(|x| x.is_finite() && x.abs() <= 1e100)
        }
    })
}

pub struct ValueRow {
    witness: Arc<Witness>,
    model: MicroModel,
    values: Mutex<Vec<Option<f64>>>,
    bounded: bool,
    forward_reads: Arc<AtomicUsize>,
}
impl ValueRow {
    pub fn new(
        witness: Arc<Witness>,
        model: &MicroModel,
        bounded: bool,
        forward_reads: Arc<AtomicUsize>,
    ) -> Self {
        let inputs = witness
            .successors
            .get()
            .expect("initialized successor inputs");
        let values = inputs
            .iter()
            .map(|(q, state)| state.is_empty().then_some(*q))
            .collect();
        Self {
            witness,
            model: model.clone(),
            values: Mutex::new(values),
            bounded,
            forward_reads,
        }
    }
    fn get(&self, i: usize) -> f64 {
        let mut values = self.values.lock().unwrap();
        if let Some(q) = values[i] {
            return q;
        }
        let (sign, state) = &self.witness.successors.get().unwrap()[i];
        // Preserve the original scorer's sign operation and actual value bits.
        let value = self.model.value(state);
        self.forward_reads.fetch_add(1, Ordering::Relaxed);
        let q = if *sign == 1. { value } else { -value };
        values[i] = Some(q);
        q
    }
    fn terminal_winner(&self, i: usize) -> bool {
        let (q, state) = &self.witness.successors.get().unwrap()[i];
        state.is_empty() && *q == 1.
    }
    fn terminal(&self, i: usize) -> Option<f64> {
        let (q, state) = &self.witness.successors.get().unwrap()[i];
        state.is_empty().then_some(*q)
    }
    pub fn loaded(&self) -> usize {
        self.values
            .lock()
            .unwrap()
            .iter()
            .filter(|q| q.is_some())
            .count()
    }
}

enum Logits {
    Complete(Vec<f64>),
    Pending {
        logp: Vec<f64>,
        beta: f64,
        values: Arc<ValueRow>,
        complete: OnceLock<Vec<f64>>,
        allow_skip: bool,
    },
}
/// Every consumer that requests coefficients still receives the complete,
/// bit-exact vector. Scoring and a certified zero hinge need not request it.
#[derive(Clone)]
pub struct CoupledLogits(Arc<Logits>);
impl From<Vec<f64>> for CoupledLogits {
    fn from(values: Vec<f64>) -> Self {
        Self(Arc::new(Logits::Complete(values)))
    }
}
fn add_value(logp: f64, beta: f64, q: f64) -> f64 {
    let mut out = logp;
    out += beta * q;
    out
}
fn better(a: usize, av: f64, b: usize, bv: f64) -> bool {
    av.total_cmp(&bv).then_with(|| b.cmp(&a)).is_gt()
}
fn best_full(values: &[f64]) -> usize {
    (0..values.len())
        .max_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
impl CoupledLogits {
    /// Diagnostic-only exact offsets, sharing each immutable cached native Q.
    /// Complete-only vectors cannot reconstruct Q without rounding and refuse.
    pub fn diagnostic_frozen_value_offsets(&self) -> Result<Vec<f64>> {
        match self.0.as_ref() {
            Logits::Pending { logp, beta, values, .. } => {
                let out:Vec<_>=(0..logp.len()).map(|i| beta*values.get(i)).collect();
                if out.iter().any(|v|!v.is_finite()) {return Err(invalid("non-finite diagnostic frozen offsets"));}
                Ok(out)
            }
            Logits::Complete(_) => Err(invalid("diagnostic frozen offsets require native ValueRow")),
        }
    }

    /// Scheduling must not dereference this object: `.len()` via Deref would
    /// run all missing forwards on the submitting thread before dispatch.
    pub fn action_count(&self) -> usize {
        match self.0.as_ref() {
            Logits::Complete(values) => values.len(),
            Logits::Pending { logp, .. } => logp.len(),
        }
    }
    pub fn is_materialized(&self) -> bool {
        match self.0.as_ref() {
            Logits::Complete(_) => true,
            Logits::Pending { complete, .. } => complete.get().is_some(),
        }
    }
    /// Keep the existing arithmetic and per-row cache path unchanged.
    pub fn materialize(&self) {
        let _ = self.deref();
    }
    pub fn pending(logp: Vec<f64>, beta: f64, values: Arc<ValueRow>, lazy: bool) -> Self {
        let allow_skip = lazy
            && values.bounded
            && beta.is_finite()
            && beta > 0.
            && beta <= 16.
            && logp.iter().all(|x| x.is_finite());
        Self(Arc::new(Logits::Pending {
            logp,
            beta,
            values,
            complete: OnceLock::new(),
            allow_skip,
        }))
    }
    fn terminal_lower_bound(&self, valid: &[bool]) -> Option<(usize, f64)> {
        let Logits::Pending {
            logp,
            beta,
            values,
            allow_skip: true,
            ..
        } = self.0.as_ref()
        else {
            return None;
        };
        if valid.len() != logp.len() {
            return None;
        }
        let mut best = None;
        for i in 0..logp.len() {
            if valid[i] && values.terminal_winner(i) {
                let score = add_value(logp[i], *beta, 1.);
                if best.map_or(true, |(j, x)| better(i, score, j, x)) {
                    best = Some((i, score));
                }
            }
        }
        best
    }
    pub fn winner(&self, valid: &[bool]) -> usize {
        let Some((mut selected, mut score)) = self.terminal_lower_bound(valid) else {
            return best_full(self);
        };
        let Logits::Pending {
            logp, beta, values, ..
        } = self.0.as_ref()
        else {
            unreachable!()
        };
        for i in 0..logp.len() {
            let upper = add_value(logp[i], *beta, values.terminal(i).unwrap_or(1.));
            // Include total_cmp's signed-zero ordering and original index ties.
            if !better(i, upper, selected, score) {
                continue;
            }
            let q = values.get(i);
            if !q.is_finite() || q.abs() > 1. {
                return best_full(self);
            }
            let actual = add_value(logp[i], *beta, q);
            if better(i, actual, selected, score) {
                selected = i;
                score = actual;
            }
        }
        selected
    }
    /// A strictly negative upper bound implies the original finite hinge is
    /// exactly +0. Boundary/equality cases materialize the old expression.
    pub fn zero_hinge(&self, valid: &[bool], margin: f64) -> bool {
        let Some((_, good)) = self.terminal_lower_bound(valid) else {
            return false;
        };
        if !margin.is_finite() || margin < 0. {
            return false;
        }
        let Logits::Pending {
            logp, beta, values, ..
        } = self.0.as_ref()
        else {
            unreachable!()
        };
        let mut bad = None;
        for i in 0..logp.len() {
            if !valid[i] {
                let upper = add_value(logp[i], *beta, values.terminal(i).unwrap_or(1.));
                bad = Some(bad.map_or(upper, |b: f64| b.max(upper)));
            }
        }
        bad.map_or(true, |b| b - good + margin < 0.)
    }
}
impl Deref for CoupledLogits {
    type Target = Vec<f64>;
    fn deref(&self) -> &Self::Target {
        match self.0.as_ref() {
            Logits::Complete(values) => values,
            Logits::Pending {
                logp,
                beta,
                values,
                complete,
                ..
            } => complete.get_or_init(|| {
                logp.iter()
                    .enumerate()
                    .map(|(i, p)| add_value(*p, *beta, values.get(i)))
                    .collect()
            }),
        }
    }
}

#[cfg(test)]
mod tests;
