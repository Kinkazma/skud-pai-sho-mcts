//! Necessary raw-policy conditions for an interpolation trial. Passing this
//! filter is never admission: the caller must still run the complete V3 guard.
//! No successor state/value is read or initialized here, and no feedback changes.
use super::*;

#[derive(Clone, Debug, Serialize)]
pub(super) struct PanelResult {
    pub reject: bool,
    pub raw_choice_lost: Option<usize>,
    /// None means a raw-choice failure ended the panel before its full KL was
    /// available. A partial sum must not be compared with the mean-KL limit.
    pub policy_kl: Option<f64>,
    pub policy_rows_evaluated: usize,
    /// Includes speculative rows whose result was not consumed before early exit.
    pub policy_rows_computed: usize,
    pub kl_partial_sum: f64,
    pub kl_rows_reduced: usize,
}

/// An exact necessary-condition check for one frozen panel. Use only to skip
/// unsuccessful interpolation trials; full candidates still produce all-panel
/// corrective feedback, and every survivor still needs `checked_v3`.
pub(super) fn check(
    rows: &[Arc<Witness>],
    old: &Score,
    model: &MicroModel,
    max_kl: f64,
) -> Result<PanelResult> {
    if !max_kl.is_finite() || max_kl < 0. {
        return Err(invalid("invalid publication prefilter KL limit"));
    }
    if old.raw.len() != rows.len() || old.priors.len() != rows.len() {
        return Err(invalid("publication prefilter panel/score mismatch"));
    }
    let mut sum = 0.;
    let mut count = 0usize;
    let mut evaluated = 0usize;
    for (i, row) in rows.iter().enumerate() {
        // Cached measure_one gives non-winning rows false/empty policy scores.
        // Their value errors remain the responsibility of the complete guard.
        if row.example.value != 1. {
            if old.raw[i] || !old.priors[i].is_empty() {
                return Err(invalid("publication prefilter expects cached panel scores"));
            }
            continue;
        }
        let embedding = model.embed(&row.example.state);
        let base = micro_softmax(&MicroModel::logits(&embedding, &row.example.actions))
            .map_err(invalid)?;
        let priors = model
            .memory_priors(&row.example.state, &row.example.actions, &base, 0)
            .map_err(invalid)?;
        evaluated += 1;
        if priors.len() != row.valid.len() || priors.is_empty() {
            return Err(invalid("publication prefilter action/support mismatch"));
        }
        // Same action ordering and first-index tie break as measure_one.
        let best = (0..priors.len())
            .max_by(|&a, &b| priors[a].total_cmp(&priors[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        if old.raw[i] && !row.valid[best] {
            return Ok(PanelResult {
                reject: true,
                raw_choice_lost: Some(i),
                policy_kl: None,
                policy_rows_evaluated: evaluated,
                policy_rows_computed: evaluated,
                kl_partial_sum: sum,
                kl_rows_reduced: count,
            });
        }
        let previous = &old.priors[i];
        if previous.is_empty() {
            continue;
        }
        if previous.len() != priors.len() {
            // repair::kl also rejects this shape through positive infinity.
            return Ok(PanelResult {
                reject: true,
                raw_choice_lost: None,
                policy_kl: Some(f64::INFINITY),
                policy_rows_evaluated: evaluated,
                policy_rows_computed: evaluated,
                kl_partial_sum: sum,
                kl_rows_reduced: count,
            });
        }
        // Keep both reductions and their order exactly equal to repair::kl.
        // In particular, do not clamp each row or threshold a partial sum.
        sum += previous
            .iter()
            .zip(&priors)
            .filter(|(p, _)| **p > 0.)
            .map(|(p, q)| p * (p.max(1e-300).ln() - q.max(1e-300).ln()))
            .sum::<f64>();
        count += 1;
    }
    let shift = sum.max(0.) / count.max(1) as f64;
    Ok(PanelResult {
        reject: !shift.is_finite() || shift > max_kl,
        raw_choice_lost: None,
        policy_kl: Some(shift),
        policy_rows_evaluated: evaluated,
        policy_rows_computed: evaluated,
        kl_partial_sum: sum,
        kl_rows_reduced: count,
    })
}

// Keep very early failures cheap, then bound extra work to one resident window.
const SERIAL_WINNING_PREFIX: usize = 8;
const PARALLEL_WINDOW: usize = 16;
fn row_priors(row: &Witness, model: &MicroModel) -> std::result::Result<Option<Vec<f64>>, String> {
    if row.example.value != 1. {
        return Ok(None);
    }
    let embedding = model.embed(&row.example.state);
    let base = micro_softmax(&MicroModel::logits(&embedding, &row.example.actions))?;
    let p = model.memory_priors(&row.example.state, &row.example.actions, &base, 0)?;
    if p.len() != row.valid.len() || p.is_empty() {
        return Err("publication prefilter action/support mismatch".into());
    }
    Ok(Some(p))
}
/// Same logical prefix, failure, actions and f64 reduction; only independent reads move.
pub(super) fn check_ordered(
    rows: &[Arc<Witness>],
    old: &Score,
    model: &MicroModel,
    max_kl: f64,
    parallel: Option<&cpu::Ordered>,
) -> Result<PanelResult> {
    let Some(pool) = parallel else {
        return check(rows, old, model, max_kl);
    };
    if !max_kl.is_finite() || max_kl < 0. {
        return Err(invalid("invalid publication prefilter KL limit"));
    }
    if old.raw.len() != rows.len() || old.priors.len() != rows.len() {
        return Err(invalid("publication prefilter panel/score mismatch"));
    }
    let (mut sum, mut count, mut evaluated, mut computed, mut start) =
        (0., 0usize, 0usize, 0usize, 0usize);
    while start < rows.len() {
        let parallel_read = evaluated >= SERIAL_WINNING_PREFIX;
        let end = (start + if parallel_read { PARALLEL_WINDOW } else { 1 }).min(rows.len());
        let part = rows[start..end].to_vec();
        computed += part.iter().filter(|r| r.example.value == 1.).count();
        let m = model.clone();
        let reads = if parallel_read {
            pool.map_owned(
                part,
                |r| r.example.actions.len(),
                move |r| row_priors(r, &m),
            )
        } else {
            part.iter().map(|r| row_priors(r, &m)).collect()
        };
        // Consume exactly in baseline order. A speculative later error cannot
        // override the first baseline raw failure or error.
        for (offset, result) in reads.into_iter().enumerate() {
            let i = start + offset;
            let row = &rows[i];
            if row.example.value != 1. {
                if old.raw[i] || !old.priors[i].is_empty() {
                    return Err(invalid("publication prefilter expects cached panel scores"));
                }
                continue;
            }
            let p = result
                .map_err(invalid)?
                .ok_or_else(|| invalid("missing winning prefilter read"))?;
            evaluated += 1;
            let best = (0..p.len())
                .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            if old.raw[i] && !row.valid[best] {
                return Ok(PanelResult {
                    reject: true,
                    raw_choice_lost: Some(i),
                    policy_kl: None,
                    policy_rows_evaluated: evaluated,
                    policy_rows_computed: computed,
                    kl_partial_sum: sum,
                    kl_rows_reduced: count,
                });
            }
            let previous = &old.priors[i];
            if previous.is_empty() {
                continue;
            }
            if previous.len() != p.len() {
                return Ok(PanelResult {
                    reject: true,
                    raw_choice_lost: None,
                    policy_kl: Some(f64::INFINITY),
                    policy_rows_evaluated: evaluated,
                    policy_rows_computed: computed,
                    kl_partial_sum: sum,
                    kl_rows_reduced: count,
                });
            }
            sum += previous
                .iter()
                .zip(&p)
                .filter(|(p, _)| **p > 0.)
                .map(|(p, q)| p * (p.max(1e-300).ln() - q.max(1e-300).ln()))
                .sum::<f64>();
            count += 1;
        }
        start = end;
    }
    let shift = sum.max(0.) / count.max(1) as f64;
    Ok(PanelResult {
        reject: !shift.is_finite() || shift > max_kl,
        raw_choice_lost: None,
        policy_kl: Some(shift),
        policy_rows_evaluated: evaluated,
        policy_rows_computed: computed,
        kl_partial_sum: sum,
        kl_rows_reduced: count,
    })
}
/// Diagnostic collection keeps only immutable models and small results. Hashing
/// happens after the transaction timer, never in an interpolation kernel.
#[derive(Default)]
pub(super) struct Audit {
    rows: std::sync::Mutex<Vec<(MicroModel, PanelResult, f64)>>,
}
impl Audit {
    pub(super) fn record(&self, model: &MicroModel, result: &PanelResult, seconds: f64) {
        self.rows
            .lock()
            .unwrap()
            .push((model.clone(), result.clone(), seconds));
    }
    pub(super) fn report(&self) -> serde_json::Value {
        use sha2::{Digest, Sha256};
        serde_json::json!(self.rows.lock().unwrap().iter().map(|(m,r,seconds)|{
            let mut h=Sha256::new();for w in m.parameters(){h.update(w.to_bits().to_le_bytes());}
            serde_json::json!({"parameter_bits_sha256":format!("{:x}",h.finalize()),"result":r,"kl_partial_sum_bits":r.kl_partial_sum.to_bits(),"policy_kl_bits":r.policy_kl.map(f64::to_bits),"seconds":seconds})
        }).collect::<Vec<_>>())
    }
}

impl Guard {
    /// Private diagnostic opt-in. The production/default path stays serial.
    #[doc(hidden)]
    pub fn diagnostic_parallel_prefilter(&mut self, enabled: bool, audit: bool) {
        self.reads.parallel_prefilter = enabled;
        self.reads.prefilter_audit = audit.then(|| Arc::new(Audit::default()));
        if let Some(v) = &mut self.validation {
            v.diagnostic_parallel_prefilter(enabled, audit);
        }
    }
    #[doc(hidden)]
    pub fn diagnostic_prefilter_profile(&self) -> serde_json::Value {
        serde_json::json!({"primary":self.reads.prefilter_audit.as_ref().map(|a|a.report()),
            "validation":self.validation.as_ref().map(|v|v.diagnostic_prefilter_profile())})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic policy-support rows exercise numerical equivalence only; they
    // are not certificates or a tactical-strength benchmark.
    fn rows(model: &MicroModel) -> Vec<Arc<Witness>> {
        let position = Position::from_standard_setup_with_rules(
            paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3),
            RULES,
        );
        [-1., 1., 0., 1., 1.]
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let example = Arc::new(MicroExample {
                    policy_support: false,
                    state: (0..model.state_features(&position).len())
                        .map(|j| (((j * 17 + i * 11) % 59) as f64 - 29.) / 59.)
                        .collect(),
                    actions: (0..3)
                        .map(|a| {
                            std::array::from_fn(|j| {
                                (((j * 7 + a * 13 + i * 3) % 31) as f64 - 15.) / 31.
                            })
                        })
                        .collect(),
                    policy: vec![1. / 3.; 3],
                    value,
                    policy_weight: 1.,
                    value_weight: 1.,
                    action_values: vec![],
                    sequence_source: 0,
                });
                Arc::new(Witness {
                    position: position.clone(),
                    valid: vec![true; 3],
                    example,
                    successors: Default::default(),
                })
            })
            .collect()
    }

    fn reference(rows: &[Arc<Witness>], model: &MicroModel) -> Score {
        // A private copy prevents measure_one from populating the filter's
        // witnesses. Zero synthetic successor values keep this a small test.
        let copy: Vec<_> = rows
            .iter()
            .map(|row| {
                let row = row.as_ref().clone();
                row.successors
                    .set(vec![(0., vec![]); row.example.actions.len()])
                    .unwrap();
                Arc::new(row)
            })
            .collect();
        measure_cached(&copy, model, 16., true).unwrap()
    }

    fn reference_kl(old: &Score, new: &Score) -> f64 {
        let mut sum = 0.;
        let mut count = 0;
        for (a, b) in old.priors.iter().zip(new.priors.iter()) {
            if a.is_empty() {
                continue;
            }
            if a.len() != b.len() {
                return f64::INFINITY;
            }
            sum += a
                .iter()
                .zip(b)
                .filter(|(p, _)| **p > 0.)
                .map(|(p, q)| p * (p.max(1e-300).ln() - q.max(1e-300).ln()))
                .sum::<f64>();
            count += 1;
        }
        sum.max(0.) / count.max(1) as f64
    }

    #[test]
    fn prefilter_kl_matches_full_score_without_successor_reads() {
        let base = MicroModel::seeded(937).with_neural_memory(617);
        let rows = rows(&base);
        let old = reference(&rows, &base);
        let mut candidate = base.clone();
        for _ in 0..3 {
            let mut example = rows[1].example.as_ref().clone();
            example.policy = vec![0., 1., 0.];
            candidate.train_step(&example, 0.02, 0.).unwrap();
        }
        let scored = reference(&rows, &candidate);
        let expected = reference_kl(&old, &scored);
        assert!(expected > 0.);
        for limit in [0., expected, 0.001, expected * 2.] {
            let filtered = check(&rows, &old, &candidate, limit).unwrap();
            assert_eq!(filtered.policy_kl.unwrap().to_bits(), expected.to_bits());
            assert_eq!(filtered.reject, expected > limit);
            assert_eq!(filtered.raw_choice_lost, None);
            assert_eq!(filtered.policy_rows_evaluated, 3);
        }
        assert!(rows.iter().all(|r| r.successors.get().is_none()));
        assert_eq!(check(&rows, &old, &base, 0.).unwrap().reject, false);
    }

    #[test]
    fn prefilter_stops_on_first_lost_raw_choice_only() {
        let model = MicroModel::seeded(77).with_neural_memory(3);
        let mut rows = rows(&model);
        let old = reference(&rows, &model);
        // Each selected action is outside this synthetic support. Later rows
        // need no calculation once this necessary retention condition fails.
        Arc::make_mut(&mut rows[1]).valid.fill(false);
        let filtered = check(&rows, &old, &model, 0.001).unwrap();
        assert!(filtered.reject);
        assert_eq!(filtered.raw_choice_lost, Some(1));
        assert_eq!(filtered.policy_kl, None);
        assert_eq!(filtered.policy_rows_evaluated, 1);
        assert!(rows.iter().all(|r| r.successors.get().is_none()));
        let mut previously_wrong = old;
        previously_wrong.raw[1] = false;
        assert!(
            !check(&rows, &previously_wrong, &model, 0.001)
                .unwrap()
                .reject
        );
    }

    #[test]
    fn prefilter_requires_aligned_scores_and_a_valid_limit() {
        let model = MicroModel::seeded(88);
        let rows = rows(&model);
        let old = reference(&rows, &model);
        for limit in [f64::NAN, f64::INFINITY, -0.001] {
            assert!(check(&rows, &old, &model, limit).is_err());
        }
        assert!(check(&rows[..3], &old, &model, 0.001).is_err());
    }
    fn logical(r: &PanelResult) -> serde_json::Value {
        let mut value = serde_json::to_value(r).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("policy_rows_computed");
        value
    }
    #[test]
    fn ordered_prefilter_keeps_kl_bits_and_counts_speculative_work() {
        let model = MicroModel::seeded(33).with_neural_memory(7);
        let seed = rows(&model);
        let mut rows: Vec<_> = (0..50).map(|i| seed[i % seed.len()].clone()).collect();
        let old = reference(&rows, &model);
        let pools = cpu::build_search_pools(4, 2, None).unwrap();
        let pool = cpu::Ordered::new(&pools);
        let a = check(&rows, &old, &model, 0.001).unwrap();
        let b = check_ordered(&rows, &old, &model, 0.001, Some(&pool)).unwrap();
        assert_eq!(logical(&a), logical(&b));
        assert_eq!(a.kl_partial_sum.to_bits(), b.kl_partial_sum.to_bits());
        assert_eq!(a.policy_kl.map(f64::to_bits), b.policy_kl.map(f64::to_bits));
        assert_eq!(a.policy_rows_evaluated, b.policy_rows_computed);
        // This failure lies after the serial eight-win prefix, inside a window.
        Arc::make_mut(&mut rows[16]).valid.fill(false);
        let a = check(&rows, &old, &model, 0.001).unwrap();
        let b = check_ordered(&rows, &old, &model, 0.001, Some(&pool)).unwrap();
        assert_eq!(logical(&a), logical(&b));
        assert_eq!(a.kl_partial_sum.to_bits(), b.kl_partial_sum.to_bits());
        assert_eq!(a.policy_kl.map(f64::to_bits), b.policy_kl.map(f64::to_bits));
        assert_eq!(b.raw_choice_lost, Some(16));
        assert!(b.policy_rows_computed > b.policy_rows_evaluated);
    }
    #[test]
    fn ordered_prefilter_preserves_early_failure_and_ignores_later_speculative_error() {
        let model = MicroModel::seeded(21);
        let seed = rows(&model);
        let mut rows: Vec<_> = (0..50).map(|i| seed[i % seed.len()].clone()).collect();
        let old = reference(&rows, &model);
        let pools = cpu::build_search_pools(4, 2, None).unwrap();
        let pool = cpu::Ordered::new(&pools);
        Arc::make_mut(&mut rows[16]).valid.fill(false);
        Arc::make_mut(&mut rows[18]).valid.clear();
        let a = check(&rows, &old, &model, 0.001).unwrap();
        let b = check_ordered(&rows, &old, &model, 0.001, Some(&pool)).unwrap();
        assert_eq!(logical(&a), logical(&b));
        assert_eq!(a.kl_partial_sum.to_bits(), b.kl_partial_sum.to_bits());
        assert_eq!(a.policy_kl.map(f64::to_bits), b.policy_kl.map(f64::to_bits));
        Arc::make_mut(&mut rows[1]).valid.fill(false);
        let a = check(&rows, &old, &model, 0.001).unwrap();
        let b = check_ordered(&rows, &old, &model, 0.001, Some(&pool)).unwrap();
        assert_eq!(logical(&a), logical(&b));
        assert_eq!(a.kl_partial_sum.to_bits(), b.kl_partial_sum.to_bits());
        assert_eq!(a.policy_kl.map(f64::to_bits), b.policy_kl.map(f64::to_bits));
        assert_eq!(b.policy_rows_computed, 1);
    }
}
