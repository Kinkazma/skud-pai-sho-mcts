//! Fixed, input-bound transaction benchmark. No collection, SGD, or promotion.
//! Original/lazy trace runs are deliberately separate from the untraced ABBA legs.
use super::*;
use sha2::{Digest, Sha256};
use std::sync::Mutex;
mod original;
mod retention;
pub(super) use original::Reads as OriginalReads;

/// Diagnostic-only inclusive wall times; absent in ordinary publications.
#[derive(Default)]
pub(super) struct Cost {
    rows: Mutex<std::collections::BTreeMap<&'static str, (usize, f64)>>,
    counters: Mutex<std::collections::BTreeMap<&'static str, usize>>,
    registry_models: Mutex<RegistryModels>,
}
#[derive(Default)]
struct RegistryModels {
    // Keep one immutable model alive per bank, preventing pointer reuse. The
    // frozen capsule uses one bank; coefficients are not copied by these clones.
    banks: Vec<MicroModel>,
    seen_bits: std::collections::BTreeSet<(usize, String)>,
    recent: std::collections::VecDeque<(MicroModel, usize, String)>,
}
pub(super) struct CostSpan {
    owner: Arc<Cost>,
    name: &'static str,
    start: Instant,
}
impl Cost {
    pub(super) fn scope(cost: &Option<Arc<Self>>, name: &'static str) -> Option<CostSpan> {
        cost.as_ref().map(|owner| CostSpan { owner: owner.clone(), name, start: Instant::now() })
    }
    fn report(&self) -> serde_json::Value {
        let rows = self.rows.lock().unwrap();
        serde_json::json!({"inclusive_wall_times_do_not_sum_nested_scopes":true,
            "registry_recent_storage_capacity":8,"counters":*self.counters.lock().unwrap(),
            "registry_bit_hashes_scoped_by_exact_bank_storage":true,
            "scopes":rows.iter().map(|(name,(calls,seconds))|
                serde_json::json!({"name":name,"calls":calls,"seconds":seconds})).collect::<Vec<_>>()})
    }
    pub(super) fn registry_read(&self, model: &MicroModel) {
        let start = Instant::now();
        let hash = float_hash(model.parameters());
        let mut models = self.registry_models.lock().unwrap();
        let bank = models.banks.iter().position(|m| match (m.sequence_memory(), model.sequence_memory()) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }).unwrap_or_else(|| { models.banks.push(model.clone()); models.banks.len() - 1 });
        let unique = models.seen_bits.insert((bank, hash.clone()));
        let found = models.recent.iter().position(|(m, _, _)| m.shares_storage_with(model));
        let recent_bits = models.recent.iter().any(|(_, b, h)| *b == bank && *h == hash);
        if let Some(index) = found { models.recent.remove(index); }
        models.recent.push_front((model.clone(), bank, hash));
        models.recent.truncate(8);
        drop(models);
        let mut counts = self.counters.lock().unwrap();
        *counts.entry("registry_reads").or_default() += 1;
        *counts.entry("registry_recent_same_storage").or_default() += usize::from(found.is_some());
        *counts.entry("registry_unique_parameter_bits_and_bank").or_default() += usize::from(unique);
        *counts.entry("registry_parameter_bit_repeats").or_default() += usize::from(!unique);
        *counts.entry("registry_recent_same_bits").or_default() += usize::from(recent_bits);
        *counts.entry("registry_recent_same_bits_distinct_storage").or_default()
            += usize::from(found.is_none() && recent_bits);
        drop(counts);
        let elapsed = start.elapsed().as_secs_f64();
        let mut rows = self.rows.lock().unwrap();
        let row = rows.entry("diagnostic.registry_bit_hash").or_default();
        row.0 += 1;
        row.1 += elapsed;
    }
}
impl Drop for CostSpan {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        let mut rows = self.owner.rows.lock().unwrap();
        let row = rows.entry(self.name).or_default();
        row.0 += 1;
        row.1 += elapsed;
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn bytes(&self) -> Result<Vec<u8>> {
        let b = fs::read(&self.path)?;
        if sha256(&b) != self.sha256 {
            return Err(invalid(format!(
                "benchmark input changed: {}",
                self.path.display()
            )));
        }
        Ok(b)
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Cycle {
    cycle: usize,
    actor: Input,
    before_consolidation: Input,
    candidate: Input,
    expected_actor: Input,
    previous_report: Option<Input>,
    report: Input,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    config: Input,
    source_plan: Input,
    primary: Input,
    validation: Input,
    cycles: Vec<Cycle>,
    order: Vec<String>,
    trace_before_timing: bool,
}

/// Enabled only for the two untimed verification transactions. Keeping Scores
/// preserves lazy cells without forcing full materialization during a transaction.
#[derive(Default)]
pub(super) struct Audit {
    scores: Mutex<Vec<(MicroModel, Score)>>,
    gradients: Mutex<Vec<serde_json::Value>>,
}
impl Audit {
    pub(super) fn score(&self, model: &MicroModel, score: &Score) {
        self.scores
            .lock()
            .unwrap()
            .push((model.clone(), score.clone()));
    }
    pub(super) fn gradient(&self, model: &MicroModel, value: Option<(&[f64], f64)>) {
        self.gradients.lock().unwrap().push(serde_json::json!({
            "model_parameters":float_hash(model.parameters()),
            "gradient":value.map(|(g,_)|float_hash(g)),
            "gap_bits":value.map(|(_,gap)|gap.to_bits()),
            "coefficients":value.map_or(0,|(g,_)|g.len())}));
    }
    fn report(&self, anchor: &Score) -> serde_json::Value {
        let scores = self
            .scores
            .lock()
            .unwrap()
            .iter()
            .map(|(m, s)| {
                serde_json::json!({
            "model_parameters":float_hash(m.parameters()),"score":score_hash(s),
            "kl_bits":repair::kl(anchor,s).to_bits()})
            })
            .collect::<Vec<_>>();
        serde_json::json!({"scores":scores,"gradients":*self.gradients.lock().unwrap()})
    }
}
fn float_hash(v: &[f64]) -> String {
    let mut h = Sha256::new();
    h.update((v.len() as u64).to_le_bytes());
    for x in v {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn score_hash(s: &Score) -> String {
    let mut h = Sha256::new();
    for v in [&s.raw, &s.coupled] {
        h.update((v.len() as u64).to_le_bytes());
        for b in v {
            h.update([u8::from(*b)]);
        }
    }
    h.update(s.mass.to_bits().to_le_bytes());
    h.update(s.value_mse.to_bits().to_le_bytes());
    h.update(float_hash(&s.errors).as_bytes());
    for v in [
        s.priors.iter().map(|v| v.as_slice()).collect::<Vec<_>>(),
        s.coupled_logits
            .iter()
            .map(|v| v.as_slice())
            .collect::<Vec<_>>(),
    ] {
        h.update((v.len() as u64).to_le_bytes());
        for row in v {
            h.update(float_hash(row).as_bytes());
        }
    }
    format!("{:x}", h.finalize())
}
fn snapshot(input: &Input, version: u64, base: Option<&MicroModel>) -> Result<Arc<Snapshot>> {
    let artifact: MicroArtifact = serde_json::from_slice(&input.bytes()?)?;
    if let Some(base) = base {
        return load_snapshot(&input.path, &artifact.identity(), version, base);
    }
    let model = artifact.model()?;
    Ok(Arc::new(Snapshot {
        identity: artifact.identity(),
        artifact: Some(Arc::new(artifact)),
        model: Arc::new(model),
        version,
        path: input.path.clone(),
    }))
}
fn fork(template: &Guard, out: &Path, lazy: bool, trace: bool) -> Result<Guard> {
    fs::create_dir_all(out.join("accepted"))?;
    let mut reads = if lazy {
        read_cache::Reads::new(true)
    } else {
        read_cache::Reads::original()
    };
    reads.reuse_root_embedding = template.reads.reuse_root_embedding;
    if trace {
        reads.audit = Some(Arc::new(Audit::default()));
    }
    Ok(Guard {
        learned_choices: template.learned_choices.clone(),
        publication_fresh: template.publication_fresh.clone(),
        publication_cost: template.publication_cost.as_ref().map(|_| Arc::new(Cost::default())),
        fast_interpolation: template.fast_interpolation,
        dynamic_interpolation_prefilter: template.dynamic_interpolation_prefilter,
        separate_step_scales: template.separate_step_scales,
        repair_interpolations: template.repair_interpolations,
        cached_margin_reads: template.cached_margin_reads,
        adaptive_margin: template.adaptive_margin,
        parallel: template.parallel.clone(),
        reads,
        validation: template
            .validation
            .as_deref()
            .map(|v| fork(v, &out.join("validation"), lazy, trace).map(Box::new))
            .transpose()?,
        rows: template.rows.clone(),
        state: template.state.clone(),
        score: template.score.clone(),
        beta: template.beta,
        accepted: template.accepted.clone(),
        out: out.into(),
        last: paisho_platform::training_time::now(),
        retired: vec![],
    })
}
fn state_without_output_path(g: &Guard) -> serde_json::Value {
    let mut s = g.progress();
    s.as_object_mut().unwrap().remove("accepted_path");
    s
}
fn focus_hash(rows: &[Arc<MicroExample>]) -> String {
    let mut h = Sha256::new();
    h.update((rows.len() as u64).to_le_bytes());
    for e in rows {
        h.update(float_hash(&e.state).as_bytes());
        for a in &e.actions {
            h.update(float_hash(a).as_bytes());
        }
        h.update(float_hash(&e.policy).as_bytes());
        for q in &e.action_values {
            h.update([u8::from(q.is_some())]);
            if let Some(q) = q {
                h.update(q.to_bits().to_le_bytes());
            }
        }
        for x in [e.value, e.policy_weight, e.value_weight] {
            h.update(x.to_bits().to_le_bytes());
        }
        h.update(e.sequence_source.to_le_bytes());
        h.update([u8::from(e.policy_support)]);
    }
    format!("{:x}", h.finalize())
}
struct Leg {
    report: serde_json::Value,
    exact: serde_json::Value,
    trace: serde_json::Value,
}
fn leg(
    template: &Guard,
    pre: &Snapshot,
    candidate: Arc<Snapshot>,
    expected: &Snapshot,
    out: &Path,
    lazy: bool,
    trace: bool,
) -> Result<Leg> {
    leg_mode(template, pre, candidate, expected, out, lazy, trace, false, None)
}
fn leg_mode(
    template: &Guard,
    pre: &Snapshot,
    candidate: Arc<Snapshot>,
    expected: &Snapshot,
    out: &Path,
    lazy: bool,
    trace: bool,
    retention: bool,
    optimization: Option<(retention::Optimization, bool)>,
) -> Result<Leg> {
    let fork_start = Instant::now();
    let mut guard = fork(template, out, lazy, trace)?;
    if let Some((kind, enabled)) = optimization {
        kind.configure(&mut guard, enabled)?;
    }
    let fork_seconds = fork_start.elapsed().as_secs_f64();
    // Same pre-publication calls as the recorded learning-cycle helper. They are
    // timed separately rather than pretending the candidate cache is free.
    let start = Instant::now();
    guard.diagnostic_measure(&guard.accepted.model)?;
    guard.observe_candidate(&pre.model)?;
    guard.diagnostic_measure(&candidate.model)?;
    let preparation_seconds = start.elapsed().as_secs_f64();
    let reads_before = guard.diagnostic_successor_reads();
    let start = Instant::now();
    let focus = guard.consider(candidate.clone(), true)?.unwrap_or_default();
    let transaction_seconds = start.elapsed().as_secs_f64();
    let reads_after = guard.diagnostic_successor_reads();
    // Timer is stopped BEFORE any complete logit digest or audit processing.
    let verification = Instant::now();
    let actor = guard.accepted();
    if float_hash(actor.model.parameters()) != float_hash(expected.model.parameters()) {
        return Err(invalid(
            "replayed transaction differs from recorded accepted actor",
        ));
    }
    let panels = std::iter::once(&guard)
        .chain(guard.validation.as_deref())
        .collect::<Vec<_>>();
    let anchors = std::iter::once(template)
        .chain(template.validation.as_deref())
        .collect::<Vec<_>>();
    let mut scores = vec![];
    let mut traces = vec![];
    for (p, a) in panels.iter().zip(&anchors) {
        let s = p.evaluate(&actor.model)?;
        scores.push(serde_json::json!({"score":score_hash(&s),"kl_bits":repair::kl(&a.score,&s).to_bits(),
            "raw":s.raw,"coupled":s.coupled,"mass_bits":s.mass.to_bits(),"mse_bits":s.value_mse.to_bits()}));
        if let Some(audit) = &p.reads.audit {
            traces.push(audit.report(&a.score));
        }
    }
    let state = if retention {
        retention::semantic_state(&guard.progress(), &actor.identity, &expected.identity)?
    } else { state_without_output_path(&guard) };
    let exact = serde_json::json!({"parameters":float_hash(actor.model.parameters()),
        "identity":if retention {&expected.identity}else{&actor.identity},
        "state":state,"scores":scores,"focus":focus_hash(&focus),"focus_rows":focus.len()});
    let report = serde_json::json!({"backend":if lazy {"lazy"}else{"original-eager-vec"},"trace_enabled":trace,
        "fork_seconds":fork_seconds,"preparation_seconds":preparation_seconds,"transaction_seconds":transaction_seconds,
        "preparation_plus_transaction_seconds":preparation_seconds+transaction_seconds,
        "verification_after_timer_seconds":verification.elapsed().as_secs_f64(),"reads_before_transaction":reads_before,
        "reads_after_transaction_before_materializing_logits":reads_after,"reads_after_verification":guard.diagnostic_successor_reads(),
        "original_value_counter_available":false,"decision":guard.state.last_decision,"repair":guard.state.repair,
        "accepted_fraction":guard.state.accepted_fraction,"actor_parameters_sha256":exact["parameters"],"exact":exact});
    let mut report = report;
    if let Some(cost) = &guard.publication_cost {
        report["publication_profile"] = cost.report();
    }
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(Leg {
        report,
        exact,
        trace: serde_json::json!(traces),
    })
}

pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("publication benchmark output must be new"));
    }
    let started = Instant::now();
    let plan_bytes = fs::read(plan_path)?;
    if serde_json::from_slice::<serde_json::Value>(&plan_bytes)?["schema"]
        == "paisho-gen5-publication-retention-bench-plan-v1"
    {
        return retention::run(plan_path, out, &plan_bytes);
    }
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "paisho-gen5-publication-lazy-bench-v1"
        || !plan.trace_before_timing
        || plan.cycles.iter().map(|c| c.cycle).collect::<Vec<_>>() != [1, 6, 12, 16, 24]
        || plan.order != ["original", "lazy", "lazy", "original"]
    {
        return Err(invalid(
            "benchmark requires fixed five cycles and original/lazy ABBA",
        ));
    }
    let inputs = std::iter::once(&plan.config)
        .chain(std::iter::once(&plan.source_plan))
        .chain(std::iter::once(&plan.primary))
        .chain(std::iter::once(&plan.validation))
        .chain(plan.cycles.iter().flat_map(|c| {
            [
                Some(&c.actor),
                Some(&c.before_consolidation),
                Some(&c.candidate),
                Some(&c.expected_actor),
                c.previous_report.as_ref(),
                Some(&c.report),
            ]
            .into_iter()
            .flatten()
        }))
        .collect::<Vec<_>>();
    for input in &inputs {
        input.bytes()?;
    }
    let options: Options = serde_json::from_slice(&plan.config.bytes()?)?;
    if options.threads == 0
        || options.threads > 32
        || options.value_policy_strength != 16.
        || options.publication_guard.as_ref() != Some(&plan.primary.path)
        || options.publication_validation.as_ref() != Some(&plan.validation.path)
    {
        return Err(invalid("benchmark config/panel mismatch"));
    }
    fs::create_dir(out)?;
    fs::write(out.join("frozen-plan.json"), &plan_bytes)?;
    let first = snapshot(&plan.cycles[0].actor, 0, None)?;
    let (pool, _) = cpu::build_pool(options.threads, None)?;
    let loading_seconds = started.elapsed().as_secs_f64();
    let mut records = vec![];
    for c in &plan.cycles {
        let cycle_start = Instant::now();
        let dir = out.join(format!("cycle-{:03}", c.cycle));
        fs::create_dir(&dir)?;
        let report: serde_json::Value = serde_json::from_slice(&c.report.bytes()?)?;
        let prior = match &c.previous_report {
            Some(p) => {
                serde_json::from_slice::<serde_json::Value>(&p.bytes()?)?["publication"].clone()
            }
            None => serde_json::Value::Null,
        };
        let actor_version = prior["accepted_version"].as_u64().unwrap_or(0);
        let actor = snapshot(&c.actor, actor_version, Some(&first.model))?;
        let pre = snapshot(&c.before_consolidation, c.cycle as u64, Some(&first.model))?;
        let candidate = snapshot(&c.candidate, c.cycle as u64, Some(&first.model))?;
        let expected = snapshot(&c.expected_actor, c.cycle as u64, Some(&first.model))?;
        if report["cycle"] != c.cycle
            || report["old_actor"] != actor.identity
            || report["learner"] != candidate.identity
            || report["actor"] != expected.identity
        {
            return Err(invalid(
                "cycle artifact identity does not match its recorded transaction",
            ));
        }
        let mut template = Guard::open(
            &plan.primary.path,
            &dir.join("setup"),
            options.value_policy_strength,
            actor.clone(),
            &prior,
        )?;
        template.enable_v2(&plan.validation.path)?;
        template.enable_v3()?;
        template.enable_parallel(&[pool.clone()]);
        // Initial scores and rule-derived successor arrays are resident for both
        // algorithms. Candidate value tables are rebuilt identically per leg.
        let setup_seconds = cycle_start.elapsed().as_secs_f64();
        let old = leg(
            &template,
            &pre,
            candidate.clone(),
            &expected,
            &dir.join("verify-original"),
            false,
            true,
        )?;
        let new = leg(
            &template,
            &pre,
            candidate.clone(),
            &expected,
            &dir.join("verify-lazy"),
            true,
            true,
        )?;
        if old.exact != new.exact || old.trace != new.trace {
            return Err(invalid("original/lazy transaction intermediate score, KL, gradient, decision or weights differ"));
        }
        fs::write(
            dir.join("trace-exact.json"),
            serde_json::to_vec_pretty(&serde_json::json!({"exact":true,"trace":old.trace}))?,
        )?;
        let mut legs = vec![];
        for (i, backend) in plan.order.iter().enumerate() {
            let leg = leg(
                &template,
                &pre,
                candidate.clone(),
                &expected,
                &dir.join(format!("abba-{i}-{backend}")),
                backend == "lazy",
                false,
            )?;
            if leg.exact != old.exact {
                return Err(invalid(
                    "timed transaction differs from verified transaction",
                ));
            }
            legs.push(leg.report);
        }
        let row = serde_json::json!({"cycle":c.cycle,"model_and_panel_setup_seconds":setup_seconds,
            "verification_legs":[old.report,new.report],"abba":legs,"all_intermediate_scores_kl_gradients_and_final_weights_bit_exact":true,
            "recorded_accepted_actor_reproduced":true});
        fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&row)?)?;
        records.push(row);
        println!(
            "{}",
            serde_json::json!({"publication_benchmark_cycle":c.cycle,"completed":true})
        );
    }
    for input in &inputs {
        input.bytes()?;
    }
    if fs::read(plan_path)? != plan_bytes {
        return Err(invalid("benchmark plan changed during execution"));
    }
    let result = serde_json::json!({"schema":"paisho-gen5-publication-lazy-bench-result-v1","plan_sha256":sha256(&plan_bytes),
        "loading_bank_and_input_verification_seconds":loading_seconds,"workers":options.threads,"cycles":records,
        "total_seconds":started.elapsed().as_secs_f64(),"diagnostic_only":true,"all_exact":true,
        "comparator":"pre-lazy Reads Vec tables and original ordered scorer; only Score logits wrapper adapted, never new per-cell Mutex cache",
        "timing_scope":"complete Guard::consider including branches, interpolation, margin gradients, validation and isolated accepted artifact write; pre-publication cache preparation reported separately; hash/materialization after timer",
        "cache_scope":"resident shared bank and immutable rule successors; each leg starts identical empty score/value LRUs then replays actor/pre-consolidation/candidate preparation; not historical cross-cycle LRU reconstruction",
        "not_a_full_campaign_throughput_or_strength_test":true});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
