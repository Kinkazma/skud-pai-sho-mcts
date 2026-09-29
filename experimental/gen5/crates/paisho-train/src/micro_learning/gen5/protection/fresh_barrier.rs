//! Diagnostic finite fresh criterion. No learner/replay/runtime option selects it.
use super::*;

const TOLERANCE: f64 = 1e-12;
pub(super) struct Barrier {
    pub ceiling: f64,
    pub active: bool,
    pub maximum_constraints: usize,
    rows: Vec<Arc<MicroExample>>,
    gradient_calls: usize,
    gradient_rows: usize,
    gradient_seconds: f64,
    gradient_failures: Vec<String>,
    scalar_calls: usize,
    scalar_rows: usize,
    scalar_seconds: f64,
    states: Vec<serde_json::Value>,
    trials: Vec<serde_json::Value>,
}
impl Barrier {
    pub fn new(
        rows: &[Arc<MicroExample>],
        anchor: f64,
        learner: f64,
        initial_seconds: f64,
    ) -> Result<Self> {
        if rows.len() > 64 || !anchor.is_finite() || !learner.is_finite() {
            return Err(invalid("invalid bounded fresh-active criterion"));
        }
        // Identical to the existing final finite test; never update this ceiling
        // when changing the candidate, trying a scale, or accepting a substep.
        let ceiling = if learner < anchor {
            anchor - 0.05 * (anchor - learner)
        } else {
            learner
        };
        Ok(Self {
            ceiling,
            active: false,
            maximum_constraints: 0,
            rows: rows.to_vec(),
            gradient_calls: 0,
            gradient_rows: 0,
            gradient_seconds: 0.,
            gradient_failures: vec![],
            scalar_calls: 2,
            scalar_rows: 2 * rows.len(),
            scalar_seconds: initial_seconds,
            states: vec![],
            trials: vec![],
        })
    }
    pub fn observe(&mut self, iteration: usize, value: f64, old_criteria_pass: bool) {
        // Delay all additional backward work until a real finite violation.
        // Once active, retain its current slack in subsequent linearizations.
        self.active |= !self.accepts(value);
        self.states.push(serde_json::json!({"iteration":iteration,"loss":value,
            "old_criteria_pass":old_criteria_pass,"fresh_pass":self.accepts(value),"normal_active":self.active}));
    }
    fn accepts(&self, value: f64) -> bool {
        value.is_finite() && value <= self.ceiling + TOLERANCE
    }
    pub fn merit(&self, value: f64) -> f64 {
        if !value.is_finite() {
            return f64::INFINITY;
        }
        ((value - (self.ceiling + TOLERANCE)).max(0.) / (self.ceiling.abs() + TOLERANCE)).powi(2)
    }
    pub fn loss(
        &mut self,
        model: &MicroModel,
        parallel: Option<&cpu::Ordered>,
        loop_v3: bool,
    ) -> Result<f64> {
        let start = paisho_platform::training_time::now();
        let result = fresh_loss_mode(model, &self.rows, parallel, loop_v3);
        self.scalar_calls += 1;
        self.scalar_rows += self.rows.len();
        self.scalar_seconds += paisho_platform::training_time::elapsed(start).as_secs_f64();
        let value = result?;
        if !value.is_finite() {
            return Err(invalid("non-finite fresh-active loss"));
        }
        Ok(value)
    }
    pub fn gradient(
        &mut self,
        model: &MicroModel,
        parallel: Option<&cpu::Ordered>,
    ) -> Result<Vec<f64>> {
        let start = paisho_platform::training_time::now();
        self.gradient_calls += 1;
        self.gradient_rows += self.rows.len();
        let result = joint_gradient(model, &self.rows, parallel);
        self.gradient_seconds += paisho_platform::training_time::elapsed(start).as_secs_f64();
        if let Err(error) = &result {
            self.gradient_failures.push(error.to_string());
        }
        result
    }
    pub fn trial(
        &mut self,
        iteration: usize,
        halve: i32,
        value: f64,
        before: f64,
        after: f64,
        admitted: bool,
    ) {
        self.trials.push(serde_json::json!({"iteration":iteration,"halve":halve,"scale":0.5_f64.powi(halve),
            "fresh":value,"fresh_pass":self.accepts(value),"merit_before":before,"merit_after":after,"admitted":admitted}));
    }
    pub fn rejected_error(&mut self, iteration: usize, halve: i32, error: &str) {
        self.trials.push(serde_json::json!({"iteration":iteration,"halve":halve,"admitted":false,"fresh_error":error}));
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"diagnostic_only":true,"ceiling":self.ceiling,"acceptance_tolerance":TOLERANCE,
            "normal_activated":self.active,"gradient_calls":self.gradient_calls,"gradient_rows":self.gradient_rows,
            "gradient_seconds":self.gradient_seconds,"gradient_failures":self.gradient_failures,"scalar_calls":self.scalar_calls,"scalar_rows":self.scalar_rows,
            "scalar_seconds":self.scalar_seconds,"maximum_constraints":self.maximum_constraints,
            "gradient_mode":"joint exact derivative of the current scalar, including reader value context; no L2",
            "maximum_corrections":6,"maximum_trials_per_correction":6,"states":self.states,"trials":self.trials})
    }
}
fn joint_gradient(
    model: &MicroModel,
    rows: &[Arc<MicroExample>],
    parallel: Option<&cpu::Ordered>,
) -> Result<Vec<f64>> {
    let count = rows.len().max(1) as f64;
    let mut sum = vec![0.; model.parameters().len()];
    let model = model.clone();
    // V3 detaches the neural reader's V context ONLY for ordinary SGD. The
    // current forward scalar still varies with that context. Its finite barrier
    // therefore needs the joint derivative, retaining all existing weights,
    // policy-support semantics and provenance-masked auxiliary targets.
    let balance = model
        .has_relational()
        .then(|| MicroStructuredBatchBalance::new(rows.iter().map(AsRef::as_ref)));
    let f = move |row: &Arc<MicroExample>| {
        match balance.as_ref() {
            Some(balance) => {
                model.loss_gradient_structured_batch_reusing(row, balance, false, Vec::new())
            }
            None => model.loss_gradient(row),
        }
        .map(|(_, g)| g)
    };
    for chunk in rows.chunks(16) {
        let parts: Vec<_> = match parallel {
            Some(p) => p.map_owned(chunk.to_vec(), |e| e.actions.len(), f.clone()),
            None => chunk.iter().map(&f).collect(),
        };
        for part in parts {
            for (sum, g) in sum.iter_mut().zip(part.map_err(invalid)?) {
                *sum += g / count;
            }
        }
    }
    if sum.iter().any(|g| !g.is_finite()) {
        return Err(invalid("non-finite fresh-active gradient"));
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relational_fresh_barrier_differentiates_the_batch_balanced_scalar() {
        let record: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr"
        ))
        .parse()
        .unwrap();
        let p = record.initial_position();
        let base = MicroModel::seeded(37).with_relational(41);
        let mut w = base.parameters().to_vec();
        for v in &mut w[paisho_ai::MICRO_RELATIONAL_START..] {
            *v = 0.01;
        }
        let model = MicroModel::from_parameters(w.clone()).unwrap();
        let mut example = MicroExample {
            state: model.state_features(&p),
            actions: vec![[0.; 32]; 4],
            policy: vec![0.25; 4],
            action_values: vec![],
            policy_weight: 0.,
            value_weight: 0.,
            value: 0.,
            policy_support: false,
            sequence_source: 0,
            structured: vec![
                Some(MicroStructuredTarget {
                    counts: [0.; 10],
                    events: [None; 10],
                    threat: MicroThreatEvidence::Unknown
                });
                4
            ],
        };
        for t in example.structured.iter_mut().flatten() {
            t.events[0] = Some(false);
        }
        let mut positive = example.clone();
        positive.structured[0].as_mut().unwrap().events[0] = Some(true);
        let rows = vec![
            Arc::new(positive),
            Arc::new(example.clone()),
            Arc::new(example),
        ];
        let g = joint_gradient(&model, &rows, None).unwrap();
        let mut indices: Vec<_> = (paisho_ai::MICRO_RELATIONAL_START..g.len()).collect();
        indices.sort_by(|a, b| g[*b].abs().total_cmp(&g[*a].abs()));
        assert!(g[indices[0]].abs() > 1e-6);
        for &i in indices.iter().take(4) {
            let old = w[i];
            w[i] = old + 1e-5;
            let high = fresh_loss_mode(
                &MicroModel::from_parameters(w.clone()).unwrap(),
                &rows,
                None,
                true,
            )
            .unwrap();
            w[i] = old - 1e-5;
            let low = fresh_loss_mode(
                &MicroModel::from_parameters(w.clone()).unwrap(),
                &rows,
                None,
                true,
            )
            .unwrap();
            w[i] = old;
            assert!(((high - low) / 2e-5 - g[i]).abs() < 2e-7);
        }
    }
    fn row(value: f64, input: f64) -> Arc<MicroExample> {
        let mut state = vec![0.; 128];
        state[0] = input;
        Arc::new(MicroExample {
            structured: Vec::new(),
            state,
            actions: vec![],
            policy: vec![],
            action_values: vec![],
            value,
            policy_weight: 0.,
            value_weight: 1.,
            policy_support: false,
            sequence_source: 0,
        })
    }
    fn fixture() -> (MicroModel, MicroModel) {
        let mut w = vec![0.; MicroModel::seeded(1).parameters().len()];
        w[0] = 1.;
        w[4160] = 0.2;
        let anchor = MicroModel::from_parameters(w.clone()).unwrap();
        w[4160] += 1e-5;
        (anchor, MicroModel::from_parameters(w).unwrap())
    }
    fn bits(a: &MicroModel, b: &MicroModel) -> bool {
        a.parameters()
            .iter()
            .zip(b.parameters())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }
    #[test]
    fn fresh_ceiling_is_fixed_and_activation_does_not_relax_it() {
        let mut b = Barrier::new(&[], 4., 3., 0.).unwrap();
        assert_eq!(b.ceiling, 3.95);
        assert_eq!(b.merit(3.95 + 0.5e-12), 0.);
        b.observe(0, 3., false);
        assert!(!b.active);
        b.observe(1, 4., true);
        assert!(b.active);
        b.observe(2, 3., true);
        assert!(b.active);
        assert_eq!(b.ceiling, 3.95);
        assert!(!b.accepts(3.95 + 2e-12));
        assert_eq!(Barrier::new(&[], 3., 4., 0.).unwrap().ceiling, 4.);
        assert!(Barrier::new(&[], f64::NAN, 4., 0.).is_err());
    }
    #[test]
    fn fresh_active_repairs_a_criterion_the_historical_path_only_rejects() {
        let (anchor, candidate) = fixture();
        let refs = vec![row(0., 0.)];
        let fresh = vec![row(1., 1.)];
        let make = || {
            let mut p = Protection::new(&anchor, refs.clone()).unwrap();
            p.enable_loop_v3();
            p.observe(&fresh);
            p.updates = 17;
            p
        };
        let mut old = make();
        let mut old_model = candidate.clone();
        old.consolidate(&mut old_model).unwrap();
        assert_eq!(old.last["accepted"], false);
        assert_eq!(old.last["fresh_after_measured"], true);
        assert!(
            old.last["reference_after"][2].as_f64().unwrap()
                <= old.anchor_losses[2] + FINITE_LOSS_TOLERANCE
        );
        assert!(bits(&old_model, &anchor));
        let mut active = make();
        let mut result = candidate.clone();
        active
            .diagnostic_consolidate_fresh_active(&mut result, |_| {})
            .unwrap();
        assert_eq!(active.last["accepted"], true, "{}", active.last);
        assert!(active.last["iterations"].as_u64().unwrap() <= 6);
        assert!(active.last["line_search_trials"].as_u64().unwrap() <= 36);
        assert!(
            active.last["fresh_barrier"]["gradient_calls"]
                .as_u64()
                .unwrap()
                > 0
        );
        let ceiling = active.last["fresh_barrier"]["ceiling"].as_f64().unwrap();
        assert!(fresh_loss_mode(&result, &fresh, None, true).unwrap() <= ceiling + TOLERANCE);
        assert_eq!(active.updates, 17);
        assert_eq!(active.fresh.len(), 1);
        assert!(
            losses(&result, &refs, None).unwrap().0[2]
                <= old.anchor_losses[2] + FINITE_LOSS_TOLERANCE
        );
        assert!(!bits(&result, &anchor));
    }
    #[test]
    fn conflicting_fresh_criterion_still_rolls_back_exactly_without_consuming_data() {
        let (anchor, mut model) = fixture();
        let refs = vec![row(0., 0.)];
        let fresh = vec![row(1., 0.)];
        let mut p = Protection::new(&anchor, refs).unwrap();
        p.enable_loop_v3();
        p.observe(&fresh);
        p.updates = 23;
        let gradients = p.gradients.clone();
        p.diagnostic_consolidate_fresh_active(&mut model, |_| {})
            .unwrap();
        assert_eq!(p.last["accepted"], false);
        assert!(bits(&model, &anchor));
        assert!(bits(&p.anchor, &anchor));
        assert_eq!(p.updates, 23);
        assert_eq!(p.fresh.len(), 1);
        assert_eq!(p.gradients, gradients);
        assert!(p.last["iterations"].as_u64().unwrap() <= 6);
        assert!(p.last["line_search_trials"].as_u64().unwrap() <= 36);
    }
    #[test]
    fn six_normalized_constraints_are_supported_without_changing_outer_tests() {
        let refs = (0..6)
            .map(|i| {
                let mut r = vec![0.; 6];
                r[i] = 1e-8;
                r
            })
            .collect::<Vec<_>>();
        let correction = projection::affine(&[0.; 6], &refs, &[1e-9; 6]).unwrap();
        for v in correction {
            assert!((v - 0.1).abs() < 1e-15);
        }
    }
    #[test]
    fn joint_barrier_matches_current_v3_scalar_in_value_and_reader_coordinates() {
        use paisho_ai::{MICRO_NEURAL_MEMORY_START, MICRO_VALUE_TRUNK};
        let mut rng = StableRng::new(1807);
        let mut example = MicroExample {
            structured: Vec::new(),
            policy_support: false,
            state: (0..417).map(|_| rng.next_f64() - 0.5).collect(),
            actions: (0..3)
                .map(|_| std::array::from_fn(|_| rng.next_f64() - 0.5))
                .collect(),
            policy: vec![0.8, 0.15, 0.05],
            value: -0.4,
            policy_weight: 1.,
            value_weight: 1.,
            action_values: vec![Some(0.9), None, Some(-0.8)],
            sequence_source: 0,
        };
        let mut model = MicroModel::seeded(893).with_neural_memory(390);
        for _ in 0..8 {
            model.train_step(&example, 0.01, 0.).unwrap();
        }
        // Retain a pure policy/Q example so the reader-context derivative is
        // visible without direct value supervision; also cover set support.
        example.value_weight = 0.;
        example.policy_support = true;
        example.policy = vec![0.5, 0.5, 0.];
        let mut other = example.clone();
        other.value_weight = 0.7;
        other.policy_support = false;
        other.policy = vec![0.1, 0.3, 0.6];
        let rows = vec![Arc::new(example), Arc::new(other)];
        for row in &rows {
            assert_eq!(
                model.loss(row).unwrap().total(row.policy_weight).to_bits(),
                model
                    .loss_loop_v3(row)
                    .unwrap()
                    .total(row.policy_weight)
                    .to_bits()
            );
        }
        let g = joint_gradient(&model, &rows, None).unwrap();
        let (_, joint) = model.loss_gradient(&rows[0]).unwrap();
        let (_, detached) = model
            .loss_gradient_loop_v3_reusing(&rows[0], vec![])
            .unwrap();
        let value_index = (4128..=4160)
            .chain(MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START)
            .max_by(|a, b| {
                (joint[*a] - detached[*a])
                    .abs()
                    .total_cmp(&(joint[*b] - detached[*b]).abs())
            })
            .unwrap();
        assert!((joint[value_index] - detached[value_index]).abs() > 1e-12);
        let reader_index = (MICRO_NEURAL_MEMORY_START..g.len())
            .max_by(|a, b| g[*a].abs().total_cmp(&g[*b].abs()))
            .unwrap();
        for i in [value_index, 4160, 4161, reader_index] {
            let eps = 1e-5;
            let mut hi = model.parameters().to_vec();
            let mut lo = hi.clone();
            hi[i] += eps;
            lo[i] -= eps;
            let a = fresh_loss_mode(&MicroModel::from_parameters(hi).unwrap(), &rows, None, true)
                .unwrap();
            let b = fresh_loss_mode(&MicroModel::from_parameters(lo).unwrap(), &rows, None, true)
                .unwrap();
            let numeric = (a - b) / (2. * eps);
            assert!(
                (numeric - g[i]).abs() < 2e-7,
                "coordinate{i}: {numeric} != {}",
                g[i]
            );
        }
    }
}
