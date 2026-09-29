//! Descriptive weighted reads of already-consumed fresh examples; never train or admit.
use super::*;
mod final_family;
#[doc(hidden)]
pub use final_family::run as measure_final_block_retention;
use std::collections::{BTreeMap, BTreeSet};
mod context_drift;
pub use context_drift::run as measure_reader_context_drift;

/// Native archive descriptors may include a byte length; plan descriptors omit it.
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bytes: Option<usize>,
}
impl Input {
    fn validate(&self, data: &[u8]) -> Result<()> {
        if sha256(data) != self.sha256 || self.bytes.is_some_and(|n| n != data.len()) {
            return Err(invalid(format!("measurement input hash/size changed: {}", self.path.display())));
        }
        Ok(())
    }
    fn bytes(&self) -> Result<Vec<u8>> {
        let data = fs::read(&self.path)?;
        self.validate(&data)?;
        Ok(data)
    }
}
const ROLES: [&str; 8] = [
    "anchor",
    "incoming",
    "original-attempted",
    "original-applied",
    "fresh-active-attempted",
    "fresh-active-applied",
    "fresh-continuous-attempted",
    "fresh-continuous-applied",
];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReadPlan {
    schema: String,
    export: Input,
    replay: Input,
    blocks: Vec<BlockInput>,
    roles: Vec<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockInput {
    capture: Input,
    selection: Input,
    examples: Input,
    models: Vec<ModelInput>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInput {
    role: String,
    input: Input,
    parameter_bits_sha256: String,
}
#[derive(Clone, Deserialize)]
struct Entry {
    union_index: usize,
    weight: f64,
    weight_numerator: u64,
    weight_denominator: u64,
}
#[derive(Deserialize)]
struct View {
    name: String,
    estimand: String,
    entries: Vec<Entry>,
}
#[derive(Deserialize, Serialize)]
struct Metadata {
    lane: String,
    value_class: String,
    stratum: String,
    source_sha256: String,
}
#[derive(Deserialize)]
struct Selection {
    event: usize,
    capture: Input,
    rows: usize,
    union: Vec<Metadata>,
    views: Vec<View>,
    heldout: bool,
    admission_criterion: bool,
}
#[derive(Clone, Deserialize, Serialize)]
struct Reading {
    total: f64,
    weighted_policy: f64,
    value_and_q: f64,
    direct_value: f64,
    q_residual: f64,
    predicted_value: f64,
    raw_argmax: Option<usize>,
    support_mass: Option<f64>,
    support_correct: Option<bool>,
}
fn read_one(model: &MicroModel, example: &MicroExample) -> std::result::Result<Reading, String> {
    let loss = model.loss_loop_v3(example)?;
    let value = model.value(&example.state);
    let delta = value - example.value;
    // Identical arithmetic order to native MicroLoss.value before auxiliary addition.
    let direct_value = 0.5 * delta * delta * example.value_weight;
    let q_residual = loss.value - direct_value;
    let (chosen, mass, correct) = if example.actions.is_empty() {
        (None, None, None)
    } else {
        let p = prior(model, example).map_err(|e| e.to_string())?;
        let chosen = best(&p);
        let mass = example.policy_support.then(|| {
            p.iter()
                .zip(&example.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(p, _)| *p)
                .sum::<f64>()
        });
        let correct = example.policy_support.then(|| example.policy[chosen] > 0.);
        (Some(chosen), mass, correct)
    };
    if ![value, direct_value, q_residual]
        .iter()
        .all(|x| x.is_finite())
        || q_residual < -1e-12
    {
        return Err("nonfinite or inconsistent native loss decomposition".into());
    }
    Ok(Reading {
        total: loss.total(example.policy_weight),
        weighted_policy: example.policy_weight * loss.policy,
        value_and_q: loss.value,
        direct_value,
        q_residual,
        predicted_value: value,
        raw_argmax: chosen,
        support_mass: mass,
        support_correct: correct,
    })
}
#[derive(Clone, Serialize)]
struct Aggregate {
    rows: usize,
    weight_sum: f64,
    total: f64,
    weighted_policy: f64,
    value_and_q: f64,
    direct_value: f64,
    q_residual: f64,
    support_rows: usize,
    support_weight: f64,
    support_correct_weight: f64,
    support_mass_weighted: f64,
    support_correct_conditional: Option<f64>,
    support_mass_conditional: Option<f64>,
}
fn aggregate(values: &[Reading], entries: &[Entry]) -> Result<Aggregate> {
    let mut a = Aggregate {
        rows: entries.len(),
        weight_sum: 0.,
        total: 0.,
        weighted_policy: 0.,
        value_and_q: 0.,
        direct_value: 0.,
        q_residual: 0.,
        support_rows: 0,
        support_weight: 0.,
        support_correct_weight: 0.,
        support_mass_weighted: 0.,
        support_correct_conditional: None,
        support_mass_conditional: None,
    };
    for e in entries {
        let r = values
            .get(e.union_index)
            .ok_or_else(|| invalid("measurement view index out of bounds"))?;
        a.weight_sum += e.weight;
        a.total += e.weight * r.total;
        a.weighted_policy += e.weight * r.weighted_policy;
        a.value_and_q += e.weight * r.value_and_q;
        a.direct_value += e.weight * r.direct_value;
        a.q_residual += e.weight * r.q_residual;
        if let (Some(mass), Some(correct)) = (r.support_mass, r.support_correct) {
            a.support_rows += 1;
            a.support_weight += e.weight;
            a.support_mass_weighted += e.weight * mass;
            if correct {
                a.support_correct_weight += e.weight;
            }
        }
    }
    if a.support_weight > 0. {
        a.support_correct_conditional = Some(a.support_correct_weight / a.support_weight);
        a.support_mass_conditional = Some(a.support_mass_weighted / a.support_weight);
    }
    Ok(a)
}
fn transitions(
    anchor: &[Reading],
    incoming: &[Reading],
    current: &[Reading],
    entries: &[Entry],
) -> serde_json::Value {
    let (mut gains, mut losses, mut incoming_gains, mut retained) = (0., 0., 0., 0.);
    for e in entries {
        if let (Some(a), Some(i), Some(c)) = (
            anchor[e.union_index].support_correct,
            incoming[e.union_index].support_correct,
            current[e.union_index].support_correct,
        ) {
            if !a && c {
                gains += e.weight;
            }
            if a && !c {
                losses += e.weight;
            }
            if !a && i {
                incoming_gains += e.weight;
                if c {
                    retained += e.weight;
                }
            }
        }
    }
    serde_json::json!({"new_support_success_weight_vs_anchor":gains,"lost_anchor_support_success_weight":losses,
        "incoming_acquisition_weight":incoming_gains,"incoming_acquisition_retained_weight":retained})
}
fn validate_view(view: &View, n: usize) -> Result<()> {
    if view.entries.is_empty() || view.entries.len() > 128 {
        return Err(invalid("fixed view size invalid"));
    }
    let mut ids = BTreeSet::new();
    let mut weight = 0.;
    for e in &view.entries {
        if e.union_index >= n
            || !ids.insert(e.union_index)
            || e.weight_numerator == 0
            || e.weight_denominator == 0
            || !e.weight.is_finite()
            || e.weight.to_bits()
                != (e.weight_numerator as f64 / e.weight_denominator as f64).to_bits()
        {
            return Err(invalid(
                "sampling indices/weights differ from frozen exact fractions",
            ));
        }
        weight += e.weight;
    }
    if (weight - 1.).abs() > 1e-12 {
        return Err(invalid("sampling weights do not sum to one"));
    }
    Ok(())
}
fn verify_model_link(
    i: usize,
    role: &str,
    input: &Input,
    a: &MicroArtifact,
    capture: &serde_json::Value,
    replay: &serde_json::Value,
) -> Result<()> {
    let c = &replay["cases"][i];
    if c["context"] != capture["context"] {
        return Err(invalid("replay context differs from measured capture"));
    }
    let capture_role = match role {
        "anchor" => Some("anchor"),
        "incoming" => Some("incoming"),
        "original-attempted" => Some("attempted"),
        "original-applied" => capture["applied_role"].as_str(),
        _ => None,
    };
    if let Some(role) = capture_role {
        if input.sha256 != capture["models"][role]["sha256"]
            || a.provenance["context"] != capture["context"]
            || a.provenance["role"] != role
        {
            return Err(invalid("captured measurement role binding differs"));
        }
    } else {
        let continuous = role.starts_with("fresh-continuous");
        let leg = usize::from(continuous);
        let row = &c["legs"][leg];
        let attempted = role.ends_with("-attempted");
        if row["fresh_active"] != true
            || row["continuous_choices"] != continuous
            || a.provenance["case"].as_u64() != Some(i as u64)
            || a.provenance["leg"].as_u64() != Some(leg as u64)
            || a.provenance["fresh_active"] != true
            || a.provenance["continuous_choices"] != continuous
            || a.provenance["kind"]
                != if attempted {
                    "capture-consolidation-replay-attempted"
                } else {
                    "capture-consolidation-replay"
                }
        {
            return Err(invalid("diagnostic attempt/applied role mismatch"));
        }
        if attempted && input.sha256 != row["attempted_model"]["sha256"] {
            return Err(invalid("attempt artifact hash mismatch"));
        }
    }
    if a.provenance["diagnostic_only"] != true {
        return Err(invalid("measurement accepts diagnostic models only"));
    }
    Ok(())
}
/// Weighted diagnostic only; inputs are frozen before any model calculation.
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("measurement output must be new"));
    }
    let started = Instant::now();
    let plan_bytes = fs::read(plan_path)?;
    let plan: ReadPlan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "paisho-gen5-c1-weighted-read-plan-v1"
        || plan.blocks.len() != 4
        || plan.roles != ROLES
    {
        return Err(invalid(
            "four fixed blocks and all eight fixed roles required",
        ));
    }
    let export: serde_json::Value = serde_json::from_slice(&plan.export.bytes()?)?;
    let replay: serde_json::Value = serde_json::from_slice(&plan.replay.bytes()?)?;
    if export["schema"] != "paisho-gen5-c1-block-retention-native-export-v1"
        || export["blocks"].as_array().map(|a| a.len()) != Some(4)
        || replay["all_four_originals_verified_before_any_variant"] != true
        || replay["continuous_choice_comparison"] != true
        || replay["cases"].as_array().map(|a| a.len()) != Some(4)
    {
        return Err(invalid("export or composed replay prerequisite missing"));
    }
    let pools = cpu::build_search_pools(10, 5, None)?;
    let ordered = cpu::Ordered::new(&pools);
    let mut bank_model: Option<MicroModel> = None;
    let mut results = vec![];
    fs::create_dir(out)?;
    for (i, block) in plan.blocks.iter().enumerate() {
        let capture: serde_json::Value = serde_json::from_slice(&block.capture.bytes()?)?;
        let selection: Selection = serde_json::from_slice(&block.selection.bytes()?)?;
        selection.capture.bytes()?;
        let exported = &export["blocks"][i];
        if selection.event != i
            || selection.capture.sha256 != block.capture.sha256
            || selection.rows != selection.union.len()
            || selection.rows > 256
            || selection.heldout
            || selection.admission_criterion
            || selection.views.len() != 2
            || block.models.len() != 8
            || exported["capture"]["sha256"] != block.capture.sha256
            || exported["selection"]["sha256"] != block.selection.sha256
            || exported["examples"]["sha256"] != block.examples.sha256
            || exported["original_source_bytes_and_consumption_checked"] != true
            || exported["captured_tail_state_overlap"] != 0
            || capture["status"] != "complete"
        {
            return Err(invalid("weighted selection/export binding differs"));
        }
        let examples = rows(&block.examples.bytes()?)?;
        if examples.len() != selection.rows {
            return Err(invalid("native exported row count changed"));
        }
        for (v, view) in selection.views.iter().enumerate() {
            if view.name != if v == 0 { "A-position" } else { "B-source" } {
                return Err(invalid("measurement view order changed"));
            }
            validate_view(view, examples.len())?;
        }
        let mut all: Vec<Vec<Reading>> = vec![];
        let mut hashes = vec![];
        let mut timings = vec![];
        for (r, m) in block.models.iter().enumerate() {
            if m.role != ROLES[r] {
                return Err(invalid("model roles reordered or omitted"));
            }
            let a: MicroArtifact = serde_json::from_slice(&m.input.bytes()?)?;
            verify_model_link(i, &m.role, &m.input, &a, &capture, &replay)?;
            let model = model(&a, bank_model.as_ref())?;
            if bank_model.is_none() {
                bank_model = Some(model.clone());
            }
            let hash = bits_hash(&model);
            if hash != m.parameter_bits_sha256 {
                return Err(invalid("model parameter bits differ from frozen role"));
            }
            if r >= 4 {
                let leg = usize::from(r >= 6);
                let expected = if r % 2 == 0 {
                    &replay["cases"][i]["legs"][leg]["incoming_to_attempted"]["after"]
                } else {
                    &replay["cases"][i]["legs"][leg]["anchor_to_applied"]["after"]
                };
                if expected != &serde_json::Value::String(hash.clone()) {
                    return Err(invalid(
                        "measured diagnostic parameters differ from completed replay",
                    ));
                }
            }
            let start = Instant::now();
            let (previous, values) = if let Some(previous) = hashes.iter().position(|h| h == &hash)
            {
                (Some(previous), all[previous].clone())
            } else {
                let copied = model.clone();
                let values = ordered.map_owned(
                    examples.clone(),
                    |e| e.actions.len(),
                    move |e| read_one(&copied, e),
                );
                (
                    None,
                    values
                        .into_iter()
                        .map(|r| r.map_err(invalid))
                        .collect::<Result<Vec<_>>>()?,
                )
            };
            timings.push(serde_json::json!({"role":m.role,"read_seconds":start.elapsed().as_secs_f64(),"same_parameter_bits_reused_from_role_index":previous}));
            hashes.push(hash);
            all.push(values);
        }
        let mut views = vec![];
        for view in &selection.views {
            let mut measures = vec![];
            let anchor = aggregate(&all[0], &view.entries)?;
            let incoming = aggregate(&all[1], &view.entries)?;
            let mut groups: BTreeMap<&str, Vec<Entry>> = BTreeMap::new();
            for entry in &view.entries {
                groups
                    .entry(&selection.union[entry.union_index].stratum)
                    .or_default()
                    .push(entry.clone());
            }
            for (r, values) in all.iter().enumerate() {
                let a = aggregate(values, &view.entries)?;
                let by_stratum = groups
                    .iter()
                    .map(|(name, entries)| {
                        Ok((
                            name.to_string(),
                            serde_json::to_value(aggregate(values, entries)?)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                measures.push(serde_json::json!({"role":ROLES[r],"aggregate":a,"by_stratum_weighted_contributions":by_stratum,
                    "delta_total_vs_anchor":a.total-anchor.total,"delta_total_vs_incoming":a.total-incoming.total,
                    "incoming_total_gain_retained_fraction":if incoming.total<anchor.total {Some((anchor.total-a.total)/(anchor.total-incoming.total))}else{None},
                    "support_transitions":transitions(&all[0],&all[1],values,&view.entries)}));
            }
            views.push(
                serde_json::json!({"name":view.name,"estimand":view.estimand,"models":measures}),
            );
        }
        let row_readings = all
            .iter()
            .enumerate()
            .map(|(i, v)| serde_json::json!({"role":ROLES[i],"rows":v}))
            .collect::<Vec<_>>();
        let detail = serde_json::json!({"metadata":selection.union,"models":row_readings,"rows":examples.len(),
            "policy_support":examples.iter().map(|e|e.policy_support).collect::<Vec<_>>(),"known_q":examples.iter().map(|e|e.action_values.iter().flatten().count()).collect::<Vec<_>>()});
        let detail_path = out.join(format!("block-{i:03}-rows.json"));
        save_json_new(&detail_path, &detail)?;
        results.push(serde_json::json!({"event":i,"check":capture["context"]["check"],"examples":block.examples,"selection":block.selection,
            "views":views,"model_parameter_hashes_in_role_order":hashes,"timings":timings,"row_reads":{"path":detail_path,"sha256":sha256(&fs::read(&detail_path)?)}}));
    }
    // Authenticate again after reads, including all model inputs; no source is mutable input.
    plan.export.bytes()?;
    plan.replay.bytes()?;
    for b in &plan.blocks {
        b.capture.bytes()?;
        b.selection.bytes()?;
        b.examples.bytes()?;
        for m in &b.models {
            m.input.bytes()?;
        }
    }
    if fs::read(plan_path)? != plan_bytes {
        return Err(invalid("weighted read plan changed"));
    }
    let report = serde_json::json!({"schema":"paisho-gen5-c1-weighted-retention-v1","plan_sha256":sha256(&plan_bytes),"blocks":results,
        "roles":ROLES,"one_bank_arc":true,"new_sgd_or_games":false,"heldout":false,"admission_criterion":false,
        "auxiliary_q_definition":"native value loss minus direct weighted half squared value loss; subtraction inherits floating rounding, not inferred proof",
        "policy_definition":"native V3 categorical or explicit support objective, multiplied by each original policy_weight",
        "support_definition":"explicit trusted optimizer support only; source-aware SGD raw priors; no rule recertification or successor coupling",
        "strata_definition":"weighted contributions to full view (weight_sum provided); target-sign classes do not certify outcomes",
        "seconds_including_load_and_output":started.elapsed().as_secs_f64(),"timings_are_not_campaign_throughput_or_causal_abba":true});
    save_json_new(&out.join("report.json"), &report)?;
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn reading(total: f64, correct: Option<bool>) -> Reading {
        Reading {
            total,
            weighted_policy: total / 2.,
            value_and_q: total / 2.,
            direct_value: total / 4.,
            q_residual: total / 4.,
            predicted_value: 0.,
            raw_argmax: Some(0),
            support_mass: correct.map(|v| if v { 0.8 } else { 0.2 }),
            support_correct: correct,
        }
    }
    fn entry(i: usize, n: u64, d: u64) -> Entry {
        Entry {
            union_index: i,
            weight: n as f64 / d as f64,
            weight_numerator: n,
            weight_denominator: d,
        }
    }
    #[test]
    fn descriptor_bytes_are_optional_but_checked_when_present() {
        let hash=sha256(b"abc");
        let short:Input=serde_json::from_value(serde_json::json!({"path":"unused","sha256":hash})).unwrap();
        short.validate(b"abc").unwrap();
        let full:Input=serde_json::from_value(serde_json::json!({"path":"unused","sha256":hash,"bytes":3})).unwrap();
        full.validate(b"abc").unwrap();
        let wrong:Input=serde_json::from_value(serde_json::json!({"path":"unused","sha256":hash,"bytes":4})).unwrap();
        assert!(wrong.validate(b"abc").is_err());
        assert!(full.validate(b"abd").is_err());
    }
    #[test]
    fn fixed_weighted_mean_and_conditional_support_are_distinct() {
        let a = aggregate(
            &[reading(2., Some(true)), reading(6., None)],
            &[entry(0, 1, 4), entry(1, 3, 4)],
        )
        .unwrap();
        assert_eq!(a.total, 5.);
        assert_eq!(a.support_weight, 0.25);
        assert_eq!(a.support_correct_conditional, Some(1.));
        assert_eq!(a.support_mass_conditional, Some(0.8));
    }
    #[test]
    fn exact_sampling_weights_and_unique_indices_required() {
        let mut view = View {
            name: "A-position".into(),
            estimand: "test".into(),
            entries: vec![entry(0, 1, 4), entry(1, 3, 4)],
        };
        validate_view(&view, 2).unwrap();
        view.entries[1].union_index = 0;
        assert!(validate_view(&view, 2).is_err());
        view.entries[1].union_index = 1;
        view.entries[1].weight += 1e-16;
        assert!(validate_view(&view, 2).is_err());
    }
}
