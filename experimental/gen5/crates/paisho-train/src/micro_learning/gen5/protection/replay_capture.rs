//! Replay the first four captured consolidations; never collect, learn, or promote.
use super::*;
use sha2::{Digest, Sha256};
mod publication_transfer;

pub use publication_transfer::run as replay_capture_publication;
mod retention_measure;
#[doc(hidden)]
pub use retention_measure::run as measure_block_retention;
#[doc(hidden)]
pub use retention_measure::measure_reader_context_drift;
#[doc(hidden)]
pub use retention_measure::measure_final_block_retention;

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
                "capture input changed: {}",
                self.path.display()
            )));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    selection: String,
    capture_root: PathBuf,
    config: Input,
    primary: Input,
    validation: Input,
    reports: Vec<Input>,
    order: Vec<String>,
    #[serde(default)]
    continuous_choices: bool,
    #[serde(default)]
    fresh_interior: bool,
    #[serde(default)]
    two_choice_constraints: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    file: String,
    sha256: String,
    bytes: usize,
}
fn child(directory: &Path, value: &serde_json::Value, expected: &str) -> Result<(Input, Vec<u8>)> {
    let d: Descriptor = serde_json::from_value(value.clone())?;
    if d.file != expected
        || Path::new(&d.file).components().count() != 1
        || Path::new(&d.file).is_absolute()
    {
        return Err(invalid("capture child name escaped its fixed role"));
    }
    let input = Input {
        path: directory.join(&d.file),
        sha256: d.sha256,
    };
    let bytes = input.bytes()?;
    if bytes.len() != d.bytes {
        return Err(invalid("capture child byte count changed"));
    }
    Ok((input, bytes))
}
fn bits_equal(a: &MicroModel, b: &MicroModel) -> bool {
    a.parameters().len() == b.parameters().len()
        && a.parameters()
            .iter()
            .zip(b.parameters())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}
fn bits_hash(model: &MicroModel) -> String {
    let mut h = Sha256::new();
    for x in model.parameters() {
        h.update(x.to_bits().to_le_bytes());
    }
    format!("{:x}", h.finalize())
}
fn value_coordinate(i: usize) -> bool {
    let start = (MICRO_INPUTS + 1) * MICRO_HIDDEN;
    (start..=start + MICRO_HIDDEN).contains(&i)
        || (MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START).contains(&i)
}
fn changed(a: &MicroModel, b: &MicroModel) -> serde_json::Value {
    let mut value = 0;
    let mut policy = 0;
    for (i, (x, y)) in a.parameters().iter().zip(b.parameters()).enumerate() {
        if x.to_bits() != y.to_bits() {
            if value_coordinate(i) {
                value += 1;
            } else {
                policy += 1;
            }
        }
    }
    serde_json::json!({"value_coefficients":value,"policy_and_reader_coefficients":policy,
        "before":bits_hash(a),"after":bits_hash(b)})
}
/// Capture fields must remain exact; a newer diagnostic may add extra fields.
fn same_observed(expected: &serde_json::Value, actual: &serde_json::Value) -> bool {
    match (expected, actual) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => a
            .iter()
            .all(|(k, v)| b.get(k).is_some_and(|x| same_observed(v, x))),
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_observed(a, b))
        }
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => a
            .as_f64()
            .zip(b.as_f64())
            .is_some_and(|(a, b)| a.to_bits() == b.to_bits()),
        _ => expected == actual,
    }
}
fn without_times(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(v) => serde_json::Value::Object(
            v.iter()
                .filter(|(k, _)| !k.ends_with("seconds") && !k.ends_with("_seconds"))
                .map(|(k, v)| (k.clone(), without_times(v)))
                .collect(),
        ),
        serde_json::Value::Array(v) => {
            serde_json::Value::Array(v.iter().map(without_times).collect())
        }
        _ => value.clone(),
    }
}
fn rows(bytes: &[u8]) -> Result<Vec<Arc<MicroExample>>> {
    let encoded: serde_json::Value = serde_json::from_slice(bytes)?;
    if !encoded
        .as_array()
        .is_some_and(|r| r.iter().all(|r| r["trusted_action_values"] == true))
    {
        return Err(invalid("capture row lost explicit trusted-Q provenance"));
    }
    let saved: Vec<resume_example::ResumeExample> = serde_json::from_slice(bytes)?;
    let restored = saved
        .into_iter()
        .map(|r| r.example_with_trusted_q(true))
        .collect::<Result<Vec<_>>>()?;
    if encoded_rows(&restored)? != bytes {
        return Err(invalid(
            "native optimizer inputs changed in trusted-Q/support roundtrip",
        ));
    }
    Ok(restored)
}
fn encoded_rows(rows: &[Arc<MicroExample>]) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(
        &rows
            .iter()
            .map(|r| resume_example::ResumeExample::from_with_trusted_q(r, true))
            .collect::<Vec<_>>(),
    )?)
}
fn model(artifact: &MicroArtifact, base: Option<&MicroModel>) -> Result<MicroModel> {
    let result = if let Some(base) = base {
        if serde_json::to_value(&artifact.sequence_memory)?
            != serde_json::to_value(base.sequence_memory().map(|b| &b.spec))?
        {
            return Err(invalid("capture sequence bank specifications differ"));
        }
        let mut m = MicroModel::from_parameters(artifact.parameters.clone()).map_err(invalid)?;
        if m.schema() != artifact.schema || m.feature_schema() != artifact.feature_schema {
            return Err(invalid("capture model schema mismatch"));
        }
        if let Some(bank) = base.sequence_memory() {
            m = m.with_sequence_memory_owned(bank.clone());
        }
        m
    } else {
        artifact.model()?
    };
    if result.parameters().len() != 292363
        || !result.has_deep_value()
        || !result.has_neural_memory()
    {
        return Err(invalid(
            "capture is not the frozen 292363-parameter V5 model",
        ));
    }
    if let (Some(a), Some(b)) = (
        base.and_then(|b| b.sequence_memory()),
        result.sequence_memory(),
    ) {
        if !Arc::ptr_eq(a, b) {
            return Err(invalid("capture model reloaded its immutable bank"));
        }
    }
    Ok(result)
}
struct Case {
    report: serde_json::Value,
    input: Input,
    dependencies: Vec<Input>,
    anchor_artifact: MicroArtifact,
    anchor: MicroModel,
    incoming: MicroModel,
    attempted: MicroModel,
    fresh: Vec<Arc<MicroExample>>,
    references: Vec<Arc<MicroExample>>,
    validation: Vec<Arc<MicroExample>>,
    validation_bytes: Vec<u8>,
}
fn load(input: &Input, base: Option<&MicroModel>) -> Result<Case> {
    let report: serde_json::Value = serde_json::from_slice(&input.bytes()?)?;
    if report["schema"] != "paisho-gen5-consolidation-capture-v1"
        || report["status"] != "complete"
        || report["context"]["rules"] != RULES.as_str()
        || report["example_format"] != "ResumeExample::from_with_trusted_q(example,true)"
        || report["counts"]
            != serde_json::json!({"fresh":64,"references":57,"validation_value":339})
        || report["limits"]["maximum_correction_iterations"] != 6
        || report["limits"]["maximum_line_search_halvings"] != 5
        || report["limits"]["finite_loss_tolerance"] != FINITE_LOSS_TOLERANCE
        || report["limits"]["fresh_loss_tolerance"] != 1e-12
        || report["limits"]["fresh_gain_retained_fraction"] != 0.05
        || report["limits"]["correction_target_tolerance_fraction"] != 0.5
        || report["context"]["check"].as_u64().is_none()
    {
        return Err(invalid(
            "incomplete or incompatible native consolidation capture",
        ));
    }
    let directory = input
        .path
        .parent()
        .ok_or_else(|| invalid("capture has no directory"))?;
    let mut dependencies = vec![];
    let mut artifacts = vec![];
    for role in ["anchor", "incoming", "attempted"] {
        let (source, bytes) = child(directory, &report["models"][role], &format!("{role}.json"))?;
        let a: MicroArtifact = serde_json::from_slice(&bytes)?;
        if a.provenance["context"] != report["context"]
            || a.provenance["role"] != role
            || a.provenance["diagnostic_only"] != true
        {
            return Err(invalid("capture model provenance/context mismatch"));
        }
        let expected = if role == "anchor" {
            report["context"]["accepted_actor_updates"].as_u64()
        } else {
            report["context"]["updates_consumed"].as_u64()
        };
        if expected != Some(a.updates) {
            return Err(invalid("capture model update counter mismatch"));
        }
        dependencies.push(source);
        artifacts.push(a);
    }
    let anchor_artifact = artifacts.remove(0);
    let anchor = model(&anchor_artifact, base)?;
    let incoming = model(&artifacts[0], Some(&anchor))?;
    let attempted = model(&artifacts[1], Some(&anchor))?;
    let (source, fresh_bytes) = child(directory, &report["examples"]["fresh"], "fresh.json")?;
    dependencies.push(source);
    let (source, reference_bytes) = child(
        directory,
        &report["examples"]["references"],
        "references.json",
    )?;
    dependencies.push(source);
    let (source, validation_bytes) = child(
        directory,
        &report["examples"]["validation_value"],
        "validation-value.json",
    )?;
    dependencies.push(source);
    let fresh = rows(&fresh_bytes)?;
    let references = rows(&reference_bytes)?;
    let validation = rows(&validation_bytes)?;
    if fresh.len() != 64 || references.len() != 57 || validation.len() != 339 {
        return Err(invalid("native captured row counts differ"));
    }
    Ok(Case {
        report,
        input: input.clone(),
        dependencies,
        anchor_artifact,
        anchor,
        incoming,
        attempted,
        fresh,
        references,
        validation,
        validation_bytes,
    })
}
fn counters(p: &Protection) -> serde_json::Value {
    serde_json::json!({"updates":p.updates,"checks":p.checks,"accepted":p.accepted,"reference_evaluations":p.reference_evaluations})
}
fn make(case: &Case, pools: &[Arc<rayon::ThreadPool>]) -> Result<Protection> {
    let mut p = Protection::new(&case.anchor, case.references.clone())?;
    p.enable_loop_v3();
    p.enable_parallel(pools);
    p.enable_validation_value(case.validation.clone())?;
    p.observe(&case.fresh);
    p.diagnostic_set_policy_gradient_constraint(
        case.report["limits"]["policy_gradient_constraint"]
            .as_bool()
            .ok_or_else(|| invalid("missing policy routing flag"))?,
    );
    let state = &case.report["anchor_state"];
    let panel = p.validation_value.as_ref().unwrap();
    if !same_observed(&state["losses"], &serde_json::json!(p.anchor_losses))
        || state["raw_choices"] != serde_json::json!(p.anchor_choices)
        || !same_observed(
            &state["validation_value"],
            &serde_json::json!({"loss":panel.anchor_loss,"class_counts":panel.counts}),
        )
        || encoded_rows(&panel.rows)? != case.validation_bytes
    {
        return Err(invalid(
            "reconstructed Protection anchor/validation differs from native capture",
        ));
    }
    let number = |key: &str| {
        state[key]
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| invalid(format!("missing capture counter {key}")))
    };
    p.updates = number("updates")?;
    p.checks = number("checks")?;
    p.accepted = number("accepted")?;
    p.reference_evaluations = number("reference_evaluations")?;
    Ok(p)
}
fn finite(
    case: &Case,
    model: &MicroModel,
    parallel: Option<&cpu::Ordered>,
) -> Result<serde_json::Value> {
    let (loss, choices) = losses(model, &case.references, parallel)?;
    let limits: [f64; 4] = serde_json::from_value(case.report["anchor_state"]["losses"].clone())?;
    let old: Vec<bool> =
        serde_json::from_value(case.report["anchor_state"]["raw_choices"].clone())?;
    let panel =
        validation_value::ValidationValue::new(case.validation.clone(), &case.anchor, parallel)?;
    let validation = panel.loss(model, parallel)?;
    let fresh_anchor = fresh_loss_mode(&case.anchor, &case.fresh, parallel, true)?;
    let fresh_incoming = fresh_loss_mode(&case.incoming, &case.fresh, parallel, true)?;
    let fresh = fresh_loss_mode(model, &case.fresh, parallel, true)?;
    let ceiling = if fresh_incoming < fresh_anchor {
        fresh_anchor - 0.05 * (fresh_anchor - fresh_incoming)
    } else {
        fresh_incoming
    };
    let classes = loss
        .iter()
        .enumerate()
        .all(|(i, v)| v.is_finite() && (i == 0 || *v <= limits[i] + FINITE_LOSS_TOLERANCE));
    let retained = old.iter().zip(&choices).all(|(a, b)| !*a || *b);
    let value_ok = validation <= panel.anchor_loss + FINITE_LOSS_TOLERANCE;
    let fresh_ok = fresh.is_finite() && fresh <= ceiling + 1e-12;
    Ok(
        serde_json::json!({"all_original_consolidation_criteria":classes&&retained&&value_ok&&fresh_ok,
        "old_raw_retained":retained,"old_raw_choices":old,"raw_choices":choices,"class_losses":loss,"class_limits":limits,
        "value_classes_pass":classes,"validation339":validation,"validation339_anchor":panel.anchor_loss,"validation339_pass":value_ok,
        "fresh":fresh,"fresh_anchor":fresh_anchor,"fresh_incoming":fresh_incoming,"fixed_fresh_ceiling":ceiling,"fresh_pass":fresh_ok}),
    )
}
fn guard(
    case: &Case,
    plan: &Plan,
    o: &Options,
    out: &Path,
    pools: &[Arc<rayon::ThreadPool>],
) -> Result<publication::Guard> {
    let snapshot = Arc::new(Snapshot {
        identity: case.anchor_artifact.identity(),
        artifact: Some(Arc::new(case.anchor_artifact.clone())),
        model: Arc::new(case.anchor.clone()),
        version: case.report["context"]["accepted_actor_version"]
            .as_u64()
            .ok_or_else(|| invalid("missing original actor version"))?,
        path: case.dependencies[0].path.clone(),
    });
    let mut g = publication::Guard::open(
        &plan.primary.path,
        out,
        o.value_policy_strength,
        snapshot,
        &serde_json::Value::Null,
    )?;
    g.enable_v2(&plan.validation.path)?;
    g.enable_v3()?;
    g.enable_parallel(pools);
    if encoded_rows(&g.reference_examples())? != encoded_rows(&case.references)? {
        return Err(invalid(
            "captured corrective rows differ from rule-verified Guard",
        ));
    }
    let native_validation = validation_value::ValidationValue::new(
        g.diagnostic_validation_examples()?,
        &case.anchor,
        Some(&cpu::Ordered::new(pools)),
    )?;
    if encoded_rows(&native_validation.rows)? != case.validation_bytes {
        return Err(invalid(
            "captured value rows differ from rule-verified secondary Guard",
        ));
    }
    Ok(g)
}
/// Descriptive SGD-context reads only, not a new acceptance criterion or recertification.
fn fresh_support(
    case: &Case,
    attempted: &MicroModel,
    applied: &MicroModel,
) -> Result<serde_json::Value> {
    let mut rows = Vec::new();
    let mut counts = [0usize; 4];
    let mut gained_incoming = 0usize;
    let mut retained_incoming_gain = 0usize;
    let mut lost_anchor = 0usize;
    for (index, row) in case
        .fresh
        .iter()
        .enumerate()
        .filter(|(_, row)| row.policy_support)
    {
        if row.actions.is_empty() || row.policy.len() != row.actions.len() {
            return Err(invalid(
                "explicit fresh support is not aligned with actions",
            ));
        }
        let support: Vec<_> = row
            .policy
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (*p > 0.).then_some(i))
            .collect();
        if support.is_empty() {
            return Err(invalid("explicit fresh support is empty"));
        }
        let mut states = Vec::new();
        let mut correct = [false; 4];
        for (i, (role, model)) in [
            ("anchor", &case.anchor),
            ("incoming", &case.incoming),
            ("attempted", attempted),
            ("applied", applied),
        ]
        .into_iter()
        .enumerate()
        {
            let p = prior(model, row)?;
            if p.len() != row.actions.len() || p.iter().any(|p| !p.is_finite()) {
                return Err(invalid("fresh support produced invalid raw priors"));
            }
            let chosen = best(&p);
            let mass: f64 = p
                .iter()
                .zip(&row.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(p, _)| *p)
                .sum();
            correct[i] = row.policy[chosen] > 0.;
            counts[i] += usize::from(correct[i]);
            let mut hash = Sha256::new();
            for value in &p {
                hash.update(value.to_bits().to_le_bytes());
            }
            states.push(serde_json::json!({"role":role,"argmax":chosen,"support_mass":mass,"argmax_in_support":correct[i],"priors_bits_sha256":format!("{:x}",hash.finalize())}));
        }
        gained_incoming += usize::from(!correct[0] && correct[1]);
        retained_incoming_gain += usize::from(!correct[0] && correct[1] && correct[3]);
        lost_anchor += usize::from(correct[0] && !correct[3]);
        rows.push(serde_json::json!({"fresh_index":index,"sequence_source":row.sequence_source,"policy_weight":row.policy_weight,
            "support":support,"states":states}));
    }
    Ok(
        serde_json::json!({"context":"native SGD sequence_source, not source0 collector inference","support":"explicit policy_support from exact optimizer capture; not recertified here; no successor coupling",
        "row_count":rows.len(),"correct_counts_anchor_incoming_attempted_applied":counts,
        "incoming_gains_vs_anchor":gained_incoming,"incoming_gains_retained_applied":retained_incoming_gain,
        "anchor_successes_lost_applied":lost_anchor,"rows":rows}),
    )
}

struct Run {
    attempted: MicroModel,
    applied: MicroModel,
    diagnostic: serde_json::Value,
    counters: serde_json::Value,
    seconds: f64,
    setup_seconds: f64,
}
fn trial(case: &Case, pools: &[Arc<rayon::ThreadPool>], active: bool, continuous: bool, interior: bool, two: bool) -> Result<Run> {
    let setup = Instant::now();
    let mut p = make(case, pools)?;
    let mut applied = case.incoming.clone();
    let mut attempted = None;
    let setup_seconds = setup.elapsed().as_secs_f64();
    let start = Instant::now();
    if two {
        if !active || !continuous {return Err(invalid("two-choice replay requires composed comparator"));}
        p.diagnostic_consolidate_two_choices(&mut applied, active, interior, |m| attempted = Some(m.clone()))?;
    } else if interior {
        if !active || !continuous {return Err(invalid("interior replay requires the composed comparator"));}
        p.diagnostic_consolidate_fresh_interior(&mut applied, true, |m| attempted = Some(m.clone()))?;
    } else if continuous {
        p.diagnostic_consolidate_continuous_choices(&mut applied, active, |m| attempted = Some(m.clone()))?;
    } else if active {
        p.diagnostic_consolidate_fresh_active(&mut applied, |m| attempted = Some(m.clone()))?;
    } else {
        p.consolidate_target(&mut applied, 0.5, |m| attempted = Some(m.clone()))?;
    }
    let seconds = start.elapsed().as_secs_f64();
    let attempted =
        attempted.ok_or_else(|| invalid("native consolidation omitted final attempted model"))?;
    Ok(Run {
        attempted,
        applied,
        diagnostic: p.last.clone(),
        counters: counters(&p),
        seconds,
        setup_seconds,
    })
}
fn verify_original(case: &Case, run: &Run) -> Result<()> {
    let expected = match case.report["applied_role"].as_str() {
        Some("anchor") => &case.anchor,
        Some("attempted") => &case.attempted,
        _ => return Err(invalid("invalid capture applied role")),
    };
    if !bits_equal(&case.attempted, &run.attempted)
        || !bits_equal(expected, &run.applied)
        || !same_observed(&case.report["result"], &run.diagnostic)
        || case.report["counters_after"] != run.counters
    {
        return Err(invalid(
            "original consolidation is not exactly reproduced; no variant may run",
        ));
    }
    Ok(())
}
/// PLAN NEW_OUTPUT. The four chronological original replays all precede variants.
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("capture replay output already exists"));
    }
    let begin = Instant::now();
    let plan_bytes = fs::read(plan_path)?;
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "paisho-gen5-consolidation-replay-plan-v1"
        || plan.selection != "first-four-captures-by-check-no-replacement"
        || plan.reports.len() != 4
        || plan.fresh_interior && !plan.continuous_choices
        || plan.two_choice_constraints && !plan.continuous_choices
        || plan.order != if plan.two_choice_constraints && plan.fresh_interior {
            ["fresh-active-continuous-interior", "fresh-active-continuous-interior-two", "fresh-active-continuous-interior-two", "fresh-active-continuous-interior"]
        } else if plan.two_choice_constraints {
            ["fresh-active-continuous", "fresh-active-continuous-two", "fresh-active-continuous-two", "fresh-active-continuous"]
        } else if plan.fresh_interior {
            ["fresh-active-continuous", "fresh-active-continuous-interior", "fresh-active-continuous-interior", "fresh-active-continuous"]
        } else if plan.continuous_choices {
            ["fresh-active", "fresh-active-continuous", "fresh-active-continuous", "fresh-active"]
        } else { ["original", "fresh-active", "fresh-active", "original"] }
    {
        return Err(invalid(
            "capture replay must use the four predeclared chronological cases and ABBA",
        ));
    }
    let mut directories = fs::read_dir(&plan.capture_root)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    directories.retain(|p| {
        p.is_dir()
            && p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("check-"))
    });
    directories.sort();
    if directories.len() < 4
        || plan
            .reports
            .iter()
            .zip(&directories[..4])
            .any(|(r, d)| r.path != d.join("report.json"))
    {
        return Err(invalid(
            "selected reports are not the first four native capture directories",
        ));
    }
    let o: Options = serde_json::from_slice(&plan.config.bytes()?)?;
    if !o.learning_loop_v3
        || !o.diagnostic_consolidation_capture
        || o.publication_guard.as_ref() != Some(&plan.primary.path)
        || o.publication_validation.as_ref() != Some(&plan.validation.path)
        || o.value_policy_strength != 16.
        || o.main_threads() != 10
        || o.search_pool_shards != 5
        || o.learner_threads != 0
        || o.macos_qos
    {
        return Err(invalid(
            "replay configuration differs from captured V3 protocol",
        ));
    }
    plan.primary.bytes()?;
    plan.validation.bytes()?;
    let mut cases = vec![];
    for report in &plan.reports {
        let case = load(report, cases.first().map(|c: &Case| &c.anchor))?;
        if cases.last().is_some_and(|previous: &Case| {
            previous.report["context"]["check"].as_u64() >= case.report["context"]["check"].as_u64()
        }) {
            return Err(invalid("capture checks are not strictly chronological"));
        }
        cases.push(case);
    }
    let pools = cpu::build_search_pools(o.main_threads(), o.search_pool_shards, None)?;
    fs::create_dir(out)?;
    let mut originals = vec![];
    // Baseline identity is a prerequisite, not a score-based case filter.
    for (i, case) in cases.iter().enumerate() {
        let run = trial(case, &pools, false, false, false, false)?;
        verify_original(case, &run)?;
        save_json_new(
            &out.join(format!("original-verified-{i:02}.json")),
            &serde_json::json!({"context":case.report["context"],
            "attempted_bits":bits_hash(&run.attempted),"applied_bits":bits_hash(&run.applied),"diagnostic":run.diagnostic,"counters":run.counters,
            "all_original_fields_exact":true,"seconds":run.seconds,"setup_seconds":run.setup_seconds}),
        )?;
        originals.push(run);
    }
    let originals_verified_seconds = begin.elapsed().as_secs_f64();
    let mut results = vec![];
    for (i, case) in cases.iter().enumerate() {
        let g = guard(case, &plan, &o, &out.join(format!("guard-{i:02}")), &pools)?;
        let state_before = g.progress();
        let ordered = cpu::Ordered::new(&pools);
        let mut legs = vec![];
        let mut modes: Vec<((bool, bool, bool, bool), Run)> = vec![];
        let configurations = if plan.two_choice_constraints {
            let interior=plan.fresh_interior;
            [(true,true,interior,false),(true,true,interior,true),(true,true,interior,true),(true,true,interior,false)]
        } else if plan.fresh_interior {
            [(true,true,false,false),(true,true,true,false),(true,true,true,false),(true,true,false,false)]
        } else if plan.continuous_choices {
            [(true,false,false,false),(true,true,false,false),(true,true,false,false),(true,false,false,false)]
        } else {[(false,false,false,false),(true,false,false,false),(true,false,false,false),(false,false,false,false)]};
        for (leg, (active, continuous, interior, two)) in configurations.into_iter().enumerate() {
            let run = trial(case, &pools, active, continuous, interior, two)?;
            if !active {
                verify_original(case, &run)?;
            }
            if let Some((_, old)) = modes.iter().find(|(a, _)| *a == (active, continuous, interior, two)) {
                if !bits_equal(&old.attempted, &run.attempted)
                    || !bits_equal(&old.applied, &run.applied)
                    || without_times(&old.diagnostic) != without_times(&run.diagnostic)
                    || old.counters != run.counters
                {
                    return Err(invalid(
                        "ABBA algorithm repetition changed parameters or finite diagnostics",
                    ));
                }
            }
            let evaluation_started = Instant::now();
            let attempt_finite = finite(case, &run.attempted, Some(&ordered))?;
            let applied_finite = finite(case, &run.applied, Some(&ordered))?;
            if run.diagnostic["accepted"] == true
                && applied_finite["all_original_consolidation_criteria"] != true
            {
                return Err(invalid(
                    "accepted repair failed independent finite consolidation criteria",
                ));
            }
            let attempted_guard = g.diagnostic_finite_v3(&run.attempted)?;
            let applied_guard = g.diagnostic_finite_v3(&run.applied)?;
            let support = fresh_support(case, &run.attempted, &run.applied)?;
            let evaluation_seconds = evaluation_started.elapsed().as_secs_f64();
            let applied = MicroArtifact::new(
                &run.applied,
                case.report["context"]["updates_consumed"].as_u64().unwrap(),
                serde_json::json!({"diagnostic_only":true,"kind":"capture-consolidation-replay","case":i,"leg":leg,"fresh_active":active,"continuous_choices":continuous,"fresh_interior":interior,"two_choice_constraints":two}),
            );
            save_json_new(
                &out.join(format!("case-{i:02}-leg-{leg}-applied.json")),
                &applied,
            )?;
            let attempted = MicroArtifact::new(&run.attempted,
                case.report["context"]["updates_consumed"].as_u64().unwrap(),
                serde_json::json!({"diagnostic_only":true,"kind":"capture-consolidation-replay-attempted",
                    "case":i,"leg":leg,"fresh_active":active,"continuous_choices":continuous,"fresh_interior":interior,"two_choice_constraints":two,
                    "applied_only_if_consolidation_accepted":true}));
            let attempted_path=out.join(format!("case-{i:02}-leg-{leg}-attempted.json"));
            save_json_new(&attempted_path,&attempted)?;
            let attempted_descriptor=serde_json::json!({"path":attempted_path,
                "sha256":sha256(&fs::read(&attempted_path)?),"parameter_bits_sha256":bits_hash(&run.attempted)});
            legs.push(serde_json::json!({"fresh_active":active,"continuous_choices":continuous,"fresh_interior":interior,"two_choice_constraints":two,"attempted_model":attempted_descriptor,"seconds":run.seconds,"setup_seconds":run.setup_seconds,"finite_verification_seconds":evaluation_seconds,
                "diagnostic":run.diagnostic,"counters":run.counters,"attempted_finite":attempt_finite,"applied_finite":applied_finite,
                "attempted_guard":attempted_guard,"applied_guard":applied_guard,"fresh_support_reads":support,
                "anchor_to_applied":changed(&case.anchor,&run.applied),"incoming_to_attempted":changed(&case.incoming,&run.attempted),
                "consolidation_and_guard_admissible":run.diagnostic["accepted"]==true&&applied_guard["accepted"]==true}));
            modes.push(((active,continuous,interior,two), run));
        }
        if g.progress() != state_before {
            return Err(invalid("finite verification mutated Guard state"));
        }
        results.push(serde_json::json!({"capture":case.input,"context":case.report["context"],"original_exact":true,
            "original_applied_role":case.report["applied_role"],"legs":legs,"guard_anchor_uses_captured_parameter_bits":true,
            "guard_identity_is_private_diagnostic_artifact_not_original_actor":true,
            "raw_support_rows":case.fresh.iter().filter(|r|r.policy_support).count(),
            "known_auxiliary_q":case.fresh.iter().flat_map(|r|&r.action_values).filter(|q|q.is_some()).count()}));
    }
    plan.config.bytes()?;
    plan.primary.bytes()?;
    plan.validation.bytes()?;
    for case in &cases {
        case.input.bytes()?;
        for input in &case.dependencies {
            input.bytes()?;
        }
    }
    if fs::read(plan_path)? != plan_bytes {
        return Err(invalid("replay plan changed during diagnostic"));
    }
    let report = serde_json::json!({"schema":"paisho-gen5-consolidation-capture-replay-v1","plan_sha256":sha256(&plan_bytes),"cases":results,
        "all_four_originals_verified_before_any_variant":true,"continuous_choice_comparison":plan.continuous_choices,"fresh_interior_comparison":plan.fresh_interior,"two_choice_comparison":plan.two_choice_constraints,"originals_verified_seconds_including_load":originals_verified_seconds,
        "total_seconds":begin.elapsed().as_secs_f64(),"one_shared_external_bank":true,"loaded_fifo":false,"new_sgd_or_games":false,
        "scope":"four fixed known native blocks; no promotion, long-term retention or strength claim; ABBA timings are isolated consolidation kernels without actor contention"});
    save_json_new(&out.join("report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observed_numbers_preserve_float_bits_and_allow_only_extra_keys() {
        assert!(same_observed(
            &serde_json::json!({"loss":0.2}),
            &serde_json::json!({"loss":0.2,"extra":true})
        ));
        assert!(!same_observed(
            &serde_json::json!({"loss":-0.0}),
            &serde_json::json!({"loss":0.0})
        ));
        assert!(!same_observed(
            &serde_json::json!({"loss":0.2}),
            &serde_json::json!({"extra":true})
        ));
    }
    #[test]
    fn resume_preserves_sparse_q_support_and_source_exactly() {
        let row = Arc::new(MicroExample { structured: Vec::new(),
            state: vec![0.; 417],
            actions: vec![[0.; 32]; 3],
            policy: vec![0.5, 0.5, 0.],
            policy_support: true,
            action_values: vec![Some(1.), None, Some(-0.3)],
            sequence_source: 73,
            policy_weight: 0.4,
            value_weight: 0.7,
            value: 1.,
        });
        let bytes = encoded_rows(&[row.clone()]).unwrap();
        let restored = rows(&bytes).unwrap();
        assert_eq!(restored[0].action_values, row.action_values);
        assert!(restored[0].policy_support);
        assert_eq!(restored[0].sequence_source, 73);
        let mut legacy: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        legacy[0]
            .as_object_mut()
            .unwrap()
            .remove("trusted_action_values");
        assert!(rows(&serde_json::to_vec(&legacy).unwrap()).is_err());
    }
}
