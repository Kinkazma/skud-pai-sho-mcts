//! Three frozen recurrent-publication transactions. The first stage must
//! reproduce the recorded actors before any optimized transaction is attempted.
use super::*;
use serde_json::Value;

/// Keep one selector for complete-transaction comparisons, so future panel
/// optimizations can join this harness without copying its exactness checks.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Optimization {
    #[default]
    DynamicPrefilter,
    RegistryReads,
    RegistryAndRootRead,
    CompactInputs,
    JointPanels,
    CompactAndJoint,
}
impl Optimization {
    pub(super) fn configure(self, guard: &mut Guard, enabled: bool) -> Result<()> {
        guard.dynamic_interpolation_prefilter = matches!(self, Self::DynamicPrefilter) && enabled;
        guard.diagnostic_reuse_root_embedding(matches!(self, Self::RegistryAndRootRead) && enabled);
        guard.diagnostic_compact_inputs(matches!(self, Self::CompactInputs | Self::CompactAndJoint) && enabled);
        guard.reads.joint_panels = matches!(self, Self::JointPanels | Self::CompactAndJoint) && enabled;
        let registry = guard.learned_choices.as_mut()
            .ok_or_else(|| invalid("retention benchmark requires a resident registry"))?;
        // Configuration creates an independent EMPTY cache after the Guard
        // fork. A previous leg or template cannot warm this transaction's reads.
        registry.configure_read_optimization(
            matches!(self, Self::RegistryAndRootRead | Self::CompactInputs | Self::JointPanels | Self::CompactAndJoint)
                || matches!(self, Self::RegistryReads) && enabled,
            guard.parallel.as_ref());
        Ok(())
    }
    fn optimized_label(self) -> &'static str {
        match self {
            Self::DynamicPrefilter => "prefilter",
            Self::RegistryReads | Self::RegistryAndRootRead | Self::CompactInputs | Self::JointPanels | Self::CompactAndJoint => "optimized",
        }
    }
    fn comparison(self) -> &'static str {
        match self {
            Self::CompactAndJoint => "registry optimized in both arms; exact compact inputs and independent panel scheduling toggled together",
            Self::JointPanels => "registry optimized in both arms; only independent fixed panel rows share one ordered worker queue",
            Self::CompactInputs => "registry optimized in both arms; only exact compact neural input construction in fixed-panel inference toggled",
            Self::DynamicPrefilter => "same production scorer and solver; only necessary dynamic interpolation prefilter toggled",
            Self::RegistryReads => "same production scorer and solver; only registry read execution/cache toggled; cache starts independently empty per leg",
            Self::RegistryAndRootRead => "registry reads optimized in both arms with independent empty caches; only exact root embedding/prior read reuse toggled",
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Capsule {
    cycle: usize,
    actor: Input,
    before_consolidation: Input,
    candidate: Input,
    expected_actor: Input,
    previous_report: Input,
    report: Input,
    restored_guard: Input,
    fresh: Input,
    fresh_provenance: Input,
    contract_from_report: String,
    fresh_ceiling_recorded: f64,
    fresh_ceiling_bits_le: String,
    expected_parameter_bits_sha256: String,
    expected_transaction_bench_parameter_hash: String,
    precommit_active: usize,
    precommit_pending: usize,
    new_pending_this_cycle: usize,
    original_publication_seconds: f64,
    original_transmit_seconds: f64,
    expected_branch: String,
}
impl Capsule {
    fn inputs(&self) -> [&Input; 9] {
        [&self.actor, &self.before_consolidation, &self.candidate, &self.expected_actor,
            &self.previous_report, &self.report, &self.restored_guard, &self.fresh,
            &self.fresh_provenance]
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RetentionPlan {
    schema: String,
    requires_bounded_transaction_bench_extension: bool,
    config: Input,
    source_plan: Input,
    primary: Input,
    validation: Input,
    cycles: Vec<Capsule>,
    order: Vec<String>,
    mandatory_first_stage: String,
    cache_scope: String,
    timing_scope: String,
    native_runs_so_far: usize,
    #[serde(default)]
    profile_cycle: Option<usize>,
    #[serde(default)]
    optimization: Optimization,
}

fn replace_identity(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(s) if s == from => *s = to.into(),
        Value::Array(a) => a.iter_mut().for_each(|v| replace_identity(v, from, to)),
        Value::Object(o) => o.values_mut().for_each(|v| replace_identity(v, from, to)),
        _ => {}
    }
}
fn remove_transfer_times(value: &mut Value) {
    if let Some(o) = value.as_object_mut() {
        o.remove("total_seconds");
        o.remove("relay_seconds");
    }
}
/// Artifact identity incorporates measured relay time. Compare all nonvolatile
/// state, binding ONLY this transaction's actor identity to the recorded actor.
/// The persisted registry digest is verified before normalizing its content.
pub(super) fn semantic_state(state: &Value, actual: &str, expected: &str) -> Result<Value> {
    let envelope = &state["learned_choices"];
    if envelope["sha256"].as_str()
        != Some(sha256(&serde_json::to_vec(&envelope["payload"])?).as_str())
    {
        return Err(invalid("benchmark registry digest mismatch"));
    }
    if state["accepted_identity"] != actual
        || envelope["payload"]["last_validated_actor"] != actual
    {
        return Err(invalid("benchmark final actor/registry binding mismatch"));
    }
    let mut value = state.clone();
    value.as_object_mut().ok_or_else(|| invalid("benchmark state is not an object"))?
        .remove("accepted_path");
    remove_transfer_times(&mut value["transfer"]);
    remove_transfer_times(&mut value["repair"]["transfer"]);
    replace_identity(&mut value, actual, expected);
    // Retain the digest in the comparison, recomputed from normalized content.
    value["learned_choices"]["sha256"] = sha256(
        &serde_json::to_vec(&value["learned_choices"]["payload"])?).into();
    Ok(value)
}

fn bit_hash(values: &[f64]) -> String {
    let mut h = Sha256::new();
    for value in values { h.update(value.to_bits().to_le_bytes()); }
    format!("{:x}", h.finalize())
}
fn bits_hex(value: f64) -> String {
    value.to_bits().to_le_bytes().iter().map(|b| format!("{b:02x}")).collect()
}
struct Prepared {
    cycle: usize,
    template: Guard,
    pre: Arc<Snapshot>,
    candidate: Arc<Snapshot>,
    expected: Arc<Snapshot>,
    baseline: Leg,
    setup_seconds: f64,
    recorded: Value,
}

pub(super) fn run(plan_path: &Path, out: &Path, plan_bytes: &[u8]) -> Result<Value> {
    let started = Instant::now();
    let plan: RetentionPlan = serde_json::from_slice(plan_bytes)?;
    if plan.schema != "paisho-gen5-publication-retention-bench-plan-v1"
        || !plan.requires_bounded_transaction_bench_extension
        || plan.cycles.iter().map(|c| c.cycle).collect::<Vec<_>>() != [3, 6, 7]
        || plan.order != ["baseline", plan.optimization.optimized_label(), plan.optimization.optimized_label(), "baseline"]
        || plan.native_runs_so_far != 0
        || plan.profile_cycle.is_some_and(|c| c != 6)
        || plan.mandatory_first_stage.is_empty()
    { return Err(invalid("retention benchmark requires fixed 03/06/07 baseline then ABBA")); }
    let inputs = [&plan.config, &plan.source_plan, &plan.primary, &plan.validation].into_iter()
        .chain(plan.cycles.iter().flat_map(Capsule::inputs)).collect::<Vec<_>>();
    for input in &inputs { input.bytes()?; }
    let options: Options = serde_json::from_slice(&plan.config.bytes()?)?;
    // The diagnostic loop selects transfer through its own bound plan; its
    // untouched production input config predates that optional runtime flag.
    let source_plan: Value = serde_json::from_slice(&plan.source_plan.bytes()?)?;
    if source_plan["publication_transfer"] != true
        || source_plan["immediate_lesson_admission"] != true
        || !options.learning_loop_v3
        || options.threads == 0 || options.threads > 32
        || options.value_policy_strength != 16.
        || options.publication_guard.as_ref() != Some(&plan.primary.path)
        || options.publication_validation.as_ref() != Some(&plan.validation.path)
    { return Err(invalid("retention benchmark config/panel mismatch")); }
    fs::create_dir(out)?;
    fs::write(out.join("frozen-plan.json"), plan_bytes)?;
    let first = snapshot(&plan.cycles[0].actor, 0, None)?;
    let (pool, _) = cpu::build_pool(options.threads, None)?;
    let loading_seconds = started.elapsed().as_secs_f64();
    let mut prepared = vec![];
    // Finish ALL historical reproductions before enabling the new prefilter.
    for c in plan.cycles.iter().filter(|c| plan.profile_cycle.map_or(true, |n| c.cycle == n)) {
        let setup = Instant::now();
        let dir = out.join(format!("cycle-{:03}", c.cycle));
        fs::create_dir(&dir)?;
        let recorded: Value = serde_json::from_slice(&c.report.bytes()?)?;
        let previous: Value = serde_json::from_slice(&c.previous_report.bytes()?)?;
        let prior: Value = serde_json::from_slice(&c.restored_guard.bytes()?)?;
        let prior_book = &prior["learned_choices"]["payload"];
        let previous_book = &previous["publication"]["learned_choices"]["payload"];
        if recorded["cycle"] != c.cycle
            || prior_book["active"].as_array().map(Vec::len) != Some(c.precommit_active)
            || prior_book["pending"].as_array().map(Vec::len) != Some(c.precommit_pending)
            || prior_book["active"] != previous_book["active"]
            || prior_book["publications"] != previous_book["publications"]
            || prior_book["pending_evictions"] != previous_book["pending_evictions"]
            || prior_book["active_retirements"] != previous_book["active_retirements"]
            || previous_book["pending"].as_array().map(Vec::len)
                .and_then(|n| n.checked_add(c.new_pending_this_cycle)) != Some(c.precommit_pending)
            || c.expected_branch != recorded["publication"]["accepted_branch"]
        { return Err(invalid("retention benchmark pretransaction registry mismatch")); }
        let actor_version = prior["accepted_version"].as_u64()
            .ok_or_else(|| invalid("missing preceding actor version"))?;
        let actor = snapshot(&c.actor, actor_version, Some(&first.model))?;
        let pre = snapshot(&c.before_consolidation, c.cycle as u64, Some(&first.model))?;
        let candidate = snapshot(&c.candidate, c.cycle as u64, Some(&first.model))?;
        let expected = snapshot(&c.expected_actor, c.cycle as u64, Some(&first.model))?;
        if recorded["old_actor"] != actor.identity || recorded["learner"] != candidate.identity
            || recorded["actor"] != expected.identity
            || bit_hash(expected.model.parameters()) != c.expected_parameter_bits_sha256
            || float_hash(expected.model.parameters()) != c.expected_transaction_bench_parameter_hash
        { return Err(invalid("retention benchmark original artifact binding mismatch")); }
        let source: Value = serde_json::from_slice(&c.fresh_provenance.bytes()?)?;
        let provenance = source["rows"].as_array().ok_or_else(|| invalid("fresh provenance missing"))?;
        let saved: Vec<SavedMicroExample> = serde_json::from_slice(&c.fresh.bytes()?)?;
        if saved.len() != 64 || provenance.len() != 64 {
            return Err(invalid("retention benchmark requires exact last64 fresh rows"));
        }
        let mut verified_sources = std::collections::BTreeSet::new();
        for (s, binding) in saved.iter().zip(provenance) {
            let input: Input = serde_json::from_value(binding["source"].clone())?;
            if verified_sources.insert((input.path.clone(), input.sha256.clone())) { input.bytes()?; }
            if binding["game_id"] != s.game_id || binding["decision"] != s.decision
                || binding["collector"] != s.collector {
                return Err(invalid("retention benchmark Saved provenance mismatch"));
            }
        }
        let fresh = saved.iter().map(|s| s.example_for_rules_with_trusted_q(RULES, true)
            .map(Arc::new)).collect::<std::result::Result<Vec<_>, _>>()?;
        let last = &recorded["protection"]["last"];
        let old = last["fresh_anchor"].as_f64().ok_or_else(|| invalid("fresh anchor missing"))?;
        let learned = last["fresh_before"].as_f64().ok_or_else(|| invalid("fresh learned loss missing"))?;
        let ceiling = if learned < old { old - 0.05 * (old - learned) } else { learned };
        if last["accepted"] != true || !ceiling.is_finite() || c.contract_from_report.is_empty()
            || ceiling.to_bits() != c.fresh_ceiling_recorded.to_bits()
            || bits_hex(ceiling) != c.fresh_ceiling_bits_le {
            return Err(invalid("retention benchmark consolidation fresh contract mismatch"));
        }
        let mut template = Guard::open(&plan.primary.path, &dir.join("setup"),
            options.value_policy_strength, actor, &prior)?;
        template.enable_v2(&plan.validation.path)?;
        template.enable_v3()?;
        template.enable_parallel(&[pool.clone()]);
        template.enable_transfer()?;
        template.set_publication_fresh(Some((fresh, ceiling)))?;
        template.dynamic_interpolation_prefilter = false;
        if plan.profile_cycle.is_some() {
            template.publication_cost = Some(Arc::new(Cost::default()));
        }
        let setup_seconds = setup.elapsed().as_secs_f64();
        let baseline = leg_mode(&template, &pre, candidate.clone(), &expected,
            &dir.join("verify-recorded-baseline"), true, false, true,
            Some((plan.optimization, false)))?;
        let expected_state = semantic_state(&recorded["publication"], &expected.identity, &expected.identity)?;
        if baseline.exact["state"] != expected_state {
            fs::write(dir.join("recorded-state-expected.json"), serde_json::to_vec_pretty(&expected_state)?)?;
            return Err(invalid(format!("cycle {} replay state differs from recorded transaction", c.cycle)));
        }
        println!("{}", serde_json::json!({"recorded_replay_cycle":c.cycle,"all_actor_parameter_bits_exact":true,"state_exact_except_declared_timing_identity":true}));
        prepared.push(Prepared { cycle: c.cycle, template, pre, candidate, expected,
            baseline, setup_seconds, recorded: serde_json::json!({
                "publication_seconds":c.original_publication_seconds,"transmit_seconds":c.original_transmit_seconds}) });
    }
    if let Some(cycle) = plan.profile_cycle {
        let result = serde_json::json!({"schema":"paisho-gen5-publication-cost-profile-v1",
            "cycle":cycle,"plan_sha256":sha256(plan_bytes),"loading_seconds":loading_seconds,
            "verification":prepared[0].baseline.report,"all_exact":true,"diagnostic_only":true,
            "total_seconds":started.elapsed().as_secs_f64(),"profile_is_not_abba_timing":true});
        for input in inputs { input.bytes()?; }
        if fs::read(plan_path)? != plan_bytes { return Err(invalid("profile plan changed")); }
        fs::write(out.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
        return Ok(result);
    }
    fs::write(out.join("recorded-replay-passed.json"), serde_json::to_vec_pretty(
        &serde_json::json!({"cycles":[3,6,7],"all_recorded_actor_bits_and_semantic_states_exact":true}))?)?;
    let mut records = vec![];
    for p in &mut prepared {
        let dir = out.join(format!("cycle-{:03}", p.cycle));
        let mut legs = vec![];
        for (i, variant) in plan.order.iter().enumerate() {
            let mut value = leg_mode(&p.template, &p.pre, p.candidate.clone(), &p.expected,
                &dir.join(format!("abba-{i}-{variant}")), true, false, true,
                Some((plan.optimization, variant == plan.optimization.optimized_label())))?;
            if value.exact != p.baseline.exact {
                return Err(invalid(format!("cycle {} {variant} transaction changes final bits/state/scores/focus", p.cycle)));
            }
            value.report["variant"] = variant.clone().into();
            fs::write(dir.join(format!("abba-{i}-{variant}")).join("report.json"),
                serde_json::to_vec_pretty(&value.report)?)?;
            legs.push(value.report);
        }
        let row = serde_json::json!({"cycle":p.cycle,"model_and_panel_setup_seconds":p.setup_seconds,
            "recorded":p.recorded,"verification":p.baseline.report,"abba":legs,
            "recorded_actor_reproduced":true,"all_final_parameters_scores_state_focus_exact":true});
        fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&row)?)?;
        records.push(row);
        println!("{}", serde_json::json!({"retention_cost_benchmark_cycle":p.cycle,"complete":true}));
    }
    for input in inputs { input.bytes()?; }
    if fs::read(plan_path)? != plan_bytes { return Err(invalid("retention benchmark plan changed")); }
    let result = serde_json::json!({"schema":"paisho-gen5-publication-retention-bench-result-v1",
        "plan_sha256":sha256(plan_bytes),"loading_seconds":loading_seconds,"workers":options.threads,
        "optimization":plan.optimization,
        "cycles":records,"all_exact":true,"diagnostic_only":true,"total_seconds":started.elapsed().as_secs_f64(),
        "cache_scope":plan.cache_scope,"timing_scope":plan.timing_scope,
        "comparison":plan.optimization.comparison(),
        "normalization":"only accepted_path, relay times, and this new actor identity bound to recorded identity; registry digest verified then recomputed after normalization",
        "not_full_campaign_throughput_or_strength":true});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn state(actor: &str) -> Value {
        let payload = serde_json::json!({"last_validated_actor":actor,"pending":[],
            "active":[{"key":"proof","acquisition":{"actor":actor}}]});
        serde_json::json!({"accepted_identity":actor,"accepted_path":"temporary.json",
            "transfer":{"relay_seconds":1.,"total_seconds":2.,"registry_commit":{"actor":actor}},
            "repair":{"transfer":{"relay_seconds":1.,"total_seconds":2.,"fresh_pass":true}},
            "learned_choices":{"sha256":sha256(&serde_json::to_vec(&payload).unwrap()),"payload":payload}})
    }
    #[test]
    fn retention_benchmark_normalizes_only_volatile_transaction_fields() {
        let a = state("actual");
        let mut b = state("expected");
        b["accepted_path"] = "other.json".into();
        b["transfer"]["total_seconds"] = 9.0.into();
        let normalized = semantic_state(&a, "actual", "expected").unwrap();
        assert_eq!(normalized, semantic_state(&b, "expected", "expected").unwrap());
        b["repair"]["transfer"]["fresh_pass"] = false.into();
        assert_ne!(normalized, semantic_state(&b, "expected", "expected").unwrap());
        b["learned_choices"]["payload"]["active"][0]["key"] = "different-proof".into();
        assert!(semantic_state(&b, "expected", "expected").is_err());
        assert!(semantic_state(&a, "wrong-actor", "expected").is_err());
    }
}
