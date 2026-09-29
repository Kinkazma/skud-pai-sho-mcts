//! Diagnostic copy of pre-lazy Reads and scoring, preserving eager Vec tables.
//! Only the final Vec logits are wrapped in the new Score storage; no cell Mutex.
//! A policy-only proposal cannot change successor values. Keep these exact reads
//! across proposal checks; keys contain EVERY value coefficient, as raw f64 bits.
//! Scores use immutable snapshot identity. Both caches have small fixed bounds.
use super::*;
use std::{collections::VecDeque, sync::Mutex};
#[derive(Default)]
pub(in crate::micro_learning::gen5::publication) struct Reads {
    scores: Mutex<VecDeque<(MicroModel, Score)>>,
    values: Mutex<VecDeque<(Vec<u64>, Arc<Vec<Vec<f64>>>)>>,
}
impl Reads {
    pub fn evaluate(
        &self,
        rows: &[Arc<Witness>],
        model: &MicroModel,
        beta: f64,
        parallel: Option<&cpu::Ordered>,
    ) -> Result<Score> {
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
                let model = model.clone();
                let f = move |r: &Arc<Witness>| -> std::result::Result<Vec<f64>, String> {
                    if r.example.value != 1. {
                        return Ok(vec![]);
                    }
                    if r.successors.get().is_none() {
                        let _ = r
                            .successors
                            .set(successor_inputs(r, &model).map_err(|e| e.to_string())?);
                    }
                    Ok(r.successors
                        .get()
                        .unwrap()
                        .iter()
                        .map(|(sign, state)| {
                            if state.is_empty() {
                                *sign
                            } else {
                                let value = model.value(state);
                                if *sign == 1. {
                                    value
                                } else {
                                    -value
                                }
                            }
                        })
                        .collect())
                };
                let parts: Vec<_> = match parallel {
                    Some(p) => p.map_owned(rows.to_vec(), |r| r.example.actions.len(), f),
                    None => rows.iter().map(f).collect(),
                };
                let table = Arc::new(
                    parts
                        .into_iter()
                        .collect::<std::result::Result<Vec<_>, _>>()
                        .map_err(invalid)?,
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
        let score = measure_ordered(rows, model, beta, true, parallel, values)?;
        let mut cache = self.scores.lock().unwrap();
        if cache.len() == 8 {
            cache.pop_front();
        }
        cache.push_back((model.clone(), score.clone()));
        Ok(score)
    }
}

#[derive(Clone)]
struct RowScore {
    raw: bool,
    coupled: bool,
    mass: f64,
    error: f64,
    class: usize,
    priors: Vec<f64>,
    coupled_logits: Vec<f64>,
}
fn measure_one(
    r: &Witness,
    model: &MicroModel,
    beta: f64,
    cached: bool,
    values: Option<&[f64]>,
) -> Result<RowScore> {
    let e = model.embed(&r.example.state);
    let class = (r.example.value as i8 + 1) as usize;
    let error = (e.value - r.example.value).powi(2);
    if cached && r.example.value != 1. {
        return Ok(RowScore {
            raw: false,
            coupled: false,
            mass: 0.,
            error,
            class,
            priors: vec![],
            coupled_logits: vec![],
        });
    }
    let base = micro_softmax(&MicroModel::logits(&e, &r.example.actions)).map_err(invalid)?;
    let p = model
        .memory_priors(&r.example.state, &r.example.actions, &base, 0)
        .map_err(invalid)?;
    let best = |p: &[f64]| {
        (0..p.len())
            .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
            .unwrap()
    };
    let raw = r.example.value == 1. && r.valid[best(&p)];
    let mass = if r.example.value == 1. {
        p.iter()
            .zip(&r.valid)
            .filter(|(_, v)| **v)
            .map(|(p, _)| *p)
            .sum::<f64>()
    } else {
        0.
    };
    let mut logits: Vec<f64> = p.iter().map(|p| p.max(1e-300).ln()).collect();
    let rebuilt;
    let inputs = if cached {
        if r.successors.get().is_none() {
            let _ = r.successors.set(successor_inputs(r, model)?);
        }
        r.successors.get().unwrap()
    } else {
        rebuilt = successor_inputs(r, model)?;
        &rebuilt
    };
    for (i, (l, (sign, state))) in logits.iter_mut().zip(inputs).enumerate() {
        let q = values.map_or_else(
            || {
                if state.is_empty() {
                    *sign
                } else {
                    let v = model.value(state);
                    if *sign == 1. {
                        v
                    } else {
                        -v
                    }
                }
            },
            |v| v[i],
        );
        *l += beta * q;
    }
    Ok(RowScore {
        raw,
        coupled: r.example.value == 1. && r.valid[best(&logits)],
        mass,
        error,
        class,
        priors: p,
        coupled_logits: logits,
    })
}
fn measure_ordered(
    rows: &[Arc<Witness>],
    model: &MicroModel,
    beta: f64,
    cached: bool,
    parallel: Option<&cpu::Ordered>,
    values: Option<Arc<Vec<Vec<f64>>>>,
) -> Result<Score> {
    let indexed: Vec<_> = rows.iter().cloned().enumerate().collect();
    let model = model.clone();
    let f = move |(i, r): &(usize, Arc<Witness>)| {
        measure_one(
            r,
            &model,
            beta,
            cached,
            values.as_ref().map(|v| v[*i].as_slice()),
        )
        .map_err(|e| e.to_string())
    };
    let parts: Vec<_> = match parallel {
        Some(p) => p.map_owned(indexed, |(_, r)| r.example.actions.len(), f),
        None => indexed.iter().map(f).collect(),
    };
    let mut raw = vec![];
    let mut coupled = vec![];
    let mut mass = 0.;
    let mut errors = [0.; 3];
    let mut counts = [0usize; 3];
    let mut priors = vec![];
    let mut row_errors = vec![];
    let mut coupled_logits = vec![];
    for (r, part) in rows.iter().zip(parts) {
        let part = part.map_err(invalid)?;
        raw.push(part.raw);
        coupled.push(part.coupled);
        if r.example.value == 1. {
            mass += part.mass;
        }
        errors[part.class] += part.error;
        counts[part.class] += 1;
        priors.push(part.priors);
        row_errors.push(part.error);
        coupled_logits.push(part.coupled_logits.into());
    }
    let value_mse = errors
        .iter()
        .zip(counts)
        .filter(|(_, n)| *n > 0)
        .map(|(e, n)| e / n as f64)
        .sum::<f64>()
        / counts.iter().filter(|n| **n > 0).count() as f64;
    Ok(Score {
        raw,
        coupled,
        mass: mass / rows.iter().filter(|r| r.example.value == 1.).count().max(1) as f64,
        value_mse,
        priors: Arc::new(priors),
        coupled_logits: Arc::new(coupled_logits),
        errors: Arc::new(row_errors),
    })
}
