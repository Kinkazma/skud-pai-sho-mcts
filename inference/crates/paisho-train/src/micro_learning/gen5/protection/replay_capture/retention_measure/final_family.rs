//! Supplement to an authenticated earlier read; same sampled rows, final fixed roles.
use super::*;
const FINAL_ROLES: [&str; 11] = [
    "anchor",
    "incoming",
    "original-attempted",
    "original-applied",
    "fresh-continuous-interior-attempted",
    "fresh-continuous-interior-applied",
    "fresh-continuous-interior-top2-attempted",
    "fresh-continuous-interior-top2-applied",
    "original-published",
    "fresh-continuous-interior-published",
    "fresh-continuous-interior-top2-published",
];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FinalPlan {
    schema: String,
    earlier_plan: Input,
    earlier_report: Input,
    final_replay: Input,
    publication: Input,
    publication_plan: Input,
    models: Vec<Vec<ModelInput>>,
    roles: Vec<String>,
}
fn final_leg(v: &serde_json::Value, top2: bool) -> bool {
    v["fresh_active"] == true
        && v["continuous_choices"] == true
        && v["fresh_interior"] == true
        && v["two_choice_constraints"] == top2
}
fn validate_reports(replay: &serde_json::Value, publication: &serde_json::Value) -> Result<()> {
    if replay["schema"] != "paisho-gen5-consolidation-capture-replay-v1"
        || replay["all_four_originals_verified_before_any_variant"] != true
        || replay["continuous_choice_comparison"] != true
        || replay["fresh_interior_comparison"] != true
        || replay["two_choice_comparison"] != true
        || replay["cases"].as_array().map(Vec::len) != Some(4)
        || publication["schema"] != "paisho-gen5-captured-publication-transfer-result-v1"
        || publication["family"] != "continuous-interior-vs-top2"
        || publication["all_four_native_original_actors_and_decisions_exact_before_variants"]
            != true
        || publication["blocks"].as_array().map(Vec::len) != Some(4)
        || publication["roles"]
            != serde_json::json!([
                "original",
                "fresh-continuous-interior",
                "fresh-continuous-interior-top2"
            ])
    {
        return Err(invalid(
            "completed final consolidation and exact native-publication prerequisites required",
        ));
    }
    for c in replay["cases"].as_array().unwrap() {
        let legs = c["legs"]
            .as_array()
            .ok_or_else(|| invalid("missing final replay legs"))?;
        if legs.len() != 4
            || !legs
                .iter()
                .zip([false, true, true, false])
                .all(|(v, t)| final_leg(v, t))
        {
            return Err(invalid("final consolidation ABBA flags/order changed"));
        }
    }
    Ok(())
}
fn verify_final_model(
    i: usize,
    r: usize,
    m: &ModelInput,
    a: &MicroArtifact,
    capture: &serde_json::Value,
    old_replay: &serde_json::Value,
    replay: &serde_json::Value,
    publication: &serde_json::Value,
    capture_input_sha: &str,
) -> Result<()> {
    if m.role != FINAL_ROLES[r] || a.provenance["diagnostic_only"] != true {
        return Err(invalid("final model role/order/provenance changed"));
    }
    if r < 4 {
        return verify_model_link(i, &m.role, &m.input, a, capture, old_replay);
    }
    if replay["cases"][i]["context"] != capture["context"] {
        return Err(invalid("final replay/capture context changed"));
    }
    if r < 8 {
        let j = (r - 4) / 2;
        let attempted = r % 2 == 0;
        let leg = &replay["cases"][i]["legs"][j];
        let expected = if attempted {
            &leg["incoming_to_attempted"]["after"]
        } else {
            &leg["anchor_to_applied"]["after"]
        };
        if !final_leg(&a.provenance, j == 1)
            || a.provenance["case"] != i
            || a.provenance["leg"] != j
            || a.provenance["kind"]
                != if attempted {
                    "capture-consolidation-replay-attempted"
                } else {
                    "capture-consolidation-replay"
                }
            || expected != &serde_json::Value::String(m.parameter_bits_sha256.clone())
            || (attempted && leg["attempted_model"]["sha256"] != m.input.sha256)
        {
            return Err(invalid(
                "final attempted/applied model is not the fixed native role",
            ));
        }
    } else {
        let j = r - 8;
        let block = &publication["blocks"][i];
        let expected_role = [
            "original",
            "fresh-continuous-interior",
            "fresh-continuous-interior-top2",
        ][j];
        let leg = &block["legs"][j];
        if block["capture"]["sha256"] != capture_input_sha
            || block["check"] != capture["context"]["check"]
            || block["legs"].as_array().map(Vec::len) != Some(3)
            || leg["role"] != expected_role
            || leg["adopted_model"]["sha256"] != m.input.sha256
            || leg["adopted_parameter_bits"] != m.parameter_bits_sha256
            || leg["adopted_true_guard_against_original_anchor"]["accepted"] != true
            || a.provenance["kind"] != "capture-publication-adopted"
            || a.provenance["role"] != expected_role
            || a.provenance["context"] != capture["context"]
        {
            return Err(invalid(
                "published actor is not the actual adopted diagnostic model",
            ));
        }
    }
    Ok(())
}
fn readings(detail: &serde_json::Value, role: usize, n: usize) -> Result<Vec<Reading>> {
    if detail["models"][role]["role"] != ROLES[role] {
        return Err(invalid("earlier row reads reordered"));
    }
    let rows: Vec<Reading> = serde_json::from_value(detail["models"][role]["rows"].clone())?;
    if rows.len() != n
        || rows.iter().any(|r| {
            ![
                r.total,
                r.weighted_policy,
                r.value_and_q,
                r.direct_value,
                r.q_residual,
                r.predicted_value,
            ]
            .iter()
            .all(|x| x.is_finite())
                || r.support_mass.is_some_and(|m| !m.is_finite())
        })
    {
        return Err(invalid("earlier row reads missing or nonfinite"));
    }
    Ok(rows)
}
/// No SGD, no guard decisions, no new sampling: evaluate only unseen parameter bits.
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("final read output must be new"));
    }
    let started = Instant::now();
    let bytes = fs::read(plan_path)?;
    let p: FinalPlan = serde_json::from_slice(&bytes)?;
    if p.schema != "paisho-gen5-c1-final-weighted-read-plan-v1"
        || p.roles != FINAL_ROLES
        || p.models.len() != 4
        || p.models.iter().any(|m| m.len() != FINAL_ROLES.len())
    {
        return Err(invalid(
            "final read requires four fixed blocks and all eleven roles",
        ));
    }
    let old: ReadPlan = serde_json::from_slice(&p.earlier_plan.bytes()?)?;
    let old_report: serde_json::Value = serde_json::from_slice(&p.earlier_report.bytes()?)?;
    let replay: serde_json::Value = serde_json::from_slice(&p.final_replay.bytes()?)?;
    let pubreport: serde_json::Value = serde_json::from_slice(&p.publication.bytes()?)?;
    validate_reports(&replay, &pubreport)?;
    let pubplan: serde_json::Value = serde_json::from_slice(&p.publication_plan.bytes()?)?;
    if pubreport["plan_sha256"] != p.publication_plan.sha256
        || pubplan["composed"]["sha256"] != p.final_replay.sha256
        || pubplan["family"] != "continuous-interior-vs-top2"
    {
        return Err(invalid(
            "publication must consume this exact final replay family/report",
        ));
    }
    if old.schema != "paisho-gen5-c1-weighted-read-plan-v1"
        || old.roles != ROLES
        || old.blocks.len() != 4
        || old_report["schema"] != "paisho-gen5-c1-weighted-retention-v1"
        || old_report["plan_sha256"] != p.earlier_plan.sha256
        || old_report["roles"] != serde_json::json!(ROLES)
        || old_report["blocks"].as_array().map(Vec::len) != Some(4)
        || old_report["one_bank_arc"] != true
        || old_report["new_sgd_or_games"] != false
    {
        return Err(invalid(
            "authenticated completed earlier eight-role measurement required",
        ));
    }
    let export: serde_json::Value = serde_json::from_slice(&old.export.bytes()?)?;
    let old_replay: serde_json::Value = serde_json::from_slice(&old.replay.bytes()?)?;
    let pools = cpu::build_search_pools(10, 5, None)?;
    let ordered = cpu::Ordered::new(&pools);
    let mut bank_model: Option<MicroModel> = None;
    let mut results = vec![];
    let mut dependencies: Vec<Input> = vec![];
    fs::create_dir(out)?;
    for (i, block) in old.blocks.iter().enumerate() {
        let capture: serde_json::Value = serde_json::from_slice(&block.capture.bytes()?)?;
        let selection: Selection = serde_json::from_slice(&block.selection.bytes()?)?;
        selection.capture.bytes()?;
        let examples = rows(&block.examples.bytes()?)?;
        let previous = &old_report["blocks"][i];
        if selection.event != i
            || selection.rows != examples.len()
            || selection.union.len() != examples.len()
            || selection.rows > 256
            || selection.heldout
            || selection.admission_criterion
            || selection.views.len() != 2
            || selection.capture.sha256 != block.capture.sha256
            || block.models.len() != 8
            || previous["examples"]["sha256"] != block.examples.sha256
            || previous["selection"]["sha256"] != block.selection.sha256
            || previous["check"] != capture["context"]["check"]
            || previous["event"] != i
            || export["blocks"][i]["examples"]["sha256"] != block.examples.sha256
            || export["blocks"][i]["captured_tail_state_overlap"] != 0
            || export["blocks"][i]["original_source_bytes_and_consumption_checked"] != true
        {
            return Err(invalid(
                "final supplement changed the frozen sample or earlier measurement binding",
            ));
        }
        dependencies.push(selection.capture.clone());
        for (j, v) in selection.views.iter().enumerate() {
            if v.name != if j == 0 { "A-position" } else { "B-source" } {
                return Err(invalid("frozen view order changed"));
            }
            validate_view(v, examples.len())?;
        }
        let detail_input: Input = serde_json::from_value(previous["row_reads"].clone())?;
        let detail: serde_json::Value = serde_json::from_slice(&detail_input.bytes()?)?;
        if detail["metadata"] != serde_json::to_value(&selection.union)?
            || detail["rows"] != examples.len()
            || detail["models"].as_array().map(Vec::len) != Some(8)
            || detail["policy_support"]
                != serde_json::json!(examples
                    .iter()
                    .map(|e| e.policy_support)
                    .collect::<Vec<_>>())
            || detail["known_q"]
                != serde_json::json!(examples
                    .iter()
                    .map(|e| e.action_values.iter().flatten().count())
                    .collect::<Vec<_>>())
        {
            return Err(invalid(
                "cached reads do not describe the exact native optimizer rows",
            ));
        }
        dependencies.push(detail_input);
        let mut cache: BTreeMap<String, (String, Vec<Reading>)> = BTreeMap::new();
        // Revalidate native artifacts and their common bank before trusting cached reads.
        for (j, m) in block.models.iter().enumerate() {
            if m.role != ROLES[j] {
                return Err(invalid("earlier model role reordered"));
            }
            let a: MicroArtifact = serde_json::from_slice(&m.input.bytes()?)?;
            verify_model_link(i, &m.role, &m.input, &a, &capture, &old_replay)?;
            let model = model(&a, bank_model.as_ref())?;
            if bank_model.is_none() {
                bank_model = Some(model.clone());
            }
            let hash = bits_hash(&model);
            if hash != m.parameter_bits_sha256
                || previous["model_parameter_hashes_in_role_order"][j] != hash
            {
                return Err(invalid("cached model coefficient binding changed"));
            }
            let values = readings(&detail, j, examples.len())?;
            if let Some((_, prev)) = cache.get(&hash) {
                if serde_json::to_vec(prev)? != serde_json::to_vec(&values)? {
                    return Err(invalid("earlier same-bit cache disagrees"));
                }
            } else {
                cache.insert(hash, (format!("earlier:{}", m.role), values));
            }
        }
        let mut all = vec![];
        let mut hashes = vec![];
        let mut timings = vec![];
        let mut new_reads = 0;
        for (r, m) in p.models[i].iter().enumerate() {
            let a: MicroArtifact = serde_json::from_slice(&m.input.bytes()?)?;
            verify_final_model(
                i,
                r,
                m,
                &a,
                &capture,
                &old_replay,
                &replay,
                &pubreport,
                &block.capture.sha256,
            )?;
            let model = model(&a, bank_model.as_ref())?;
            let hash = bits_hash(&model);
            if hash != m.parameter_bits_sha256 {
                return Err(invalid("final coefficient bits differ from frozen role"));
            }
            let t = Instant::now();
            let (reused, values) = if let Some((source, v)) = cache.get(&hash) {
                (Some(source.clone()), v.clone())
            } else {
                new_reads += 1;
                let copied = model.clone();
                let values = ordered
                    .map_owned(
                        examples.clone(),
                        |e| e.actions.len(),
                        move |e| read_one(&copied, e),
                    )
                    .into_iter()
                    .map(|r| r.map_err(invalid))
                    .collect::<Result<Vec<_>>>()?;
                cache.insert(
                    hash.clone(),
                    (format!("current:{}", m.role), values.clone()),
                );
                (None, values)
            };
            timings.push(serde_json::json!({"role":m.role,"read_seconds":t.elapsed().as_secs_f64(),"readings_reused_from":reused}));
            hashes.push(hash);
            all.push(values);
        }
        let mut views = vec![];
        for view in &selection.views {
            let anchor = aggregate(&all[0], &view.entries)?;
            let incoming = aggregate(&all[1], &view.entries)?;
            let mut groups: BTreeMap<&str, Vec<Entry>> = BTreeMap::new();
            for e in &view.entries {
                groups
                    .entry(&selection.union[e.union_index].stratum)
                    .or_default()
                    .push(e.clone());
            }
            let mut measured = vec![];
            for (r, values) in all.iter().enumerate() {
                let a = aggregate(values, &view.entries)?;
                let by_stratum = groups
                    .iter()
                    .map(|(name, e)| {
                        Ok((
                            name.to_string(),
                            serde_json::to_value(aggregate(values, e)?)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                measured.push(serde_json::json!({"role":FINAL_ROLES[r],"aggregate":a,"by_stratum_weighted_contributions":by_stratum,
                    "delta_total_vs_anchor":a.total-anchor.total,"delta_total_vs_incoming":a.total-incoming.total,
                    "incoming_total_gain_retained_fraction":if incoming.total<anchor.total{Some((anchor.total-a.total)/(anchor.total-incoming.total))}else{None},
                    "support_transitions":transitions(&all[0],&all[1],values,&view.entries)}));
            }
            views.push(
                serde_json::json!({"name":view.name,"estimand":view.estimand,"models":measured}),
            );
        }
        let detail_path = out.join(format!("block-{i:03}-rows.json"));
        save_json_new(
            &detail_path,
            &serde_json::json!({"metadata":selection.union,"rows":examples.len(),
            "models":all.iter().enumerate().map(|(r,v)|serde_json::json!({"role":FINAL_ROLES[r],"rows":v})).collect::<Vec<_>>(),
            "policy_support":examples.iter().map(|e|e.policy_support).collect::<Vec<_>>()}),
        )?;
        results.push(serde_json::json!({"event":i,"check":capture["context"]["check"],"examples":block.examples,"selection":block.selection,
            "views":views,"model_parameter_hashes_in_role_order":hashes,"timings":timings,"new_parameter_sets_evaluated":new_reads,
            "rows_evaluated":new_reads*examples.len(),"row_reads":{"path":detail_path,"sha256":sha256(&fs::read(&detail_path)?)}}));
    }
    p.earlier_plan.bytes()?;
    p.earlier_report.bytes()?;
    p.final_replay.bytes()?;
    p.publication.bytes()?;
    p.publication_plan.bytes()?;
    old.export.bytes()?;
    old.replay.bytes()?;
    for b in &old.blocks {
        b.capture.bytes()?;
        b.selection.bytes()?;
        b.examples.bytes()?;
        for m in &b.models {
            m.input.bytes()?;
        }
    }
    for m in p.models.iter().flatten() {
        m.input.bytes()?;
    }
    for d in &dependencies {
        d.bytes()?;
    }
    if fs::read(plan_path)? != bytes {
        return Err(invalid("final read plan changed"));
    }
    let report = serde_json::json!({"schema":"paisho-gen5-c1-final-weighted-retention-v1","plan_sha256":sha256(&bytes),"blocks":results,"roles":FINAL_ROLES,
        "earlier_report":p.earlier_report,"final_replay":p.final_replay,"publication":p.publication,"publication_plan":p.publication_plan,"one_bank_arc":true,
        "new_sgd_or_games":false,"heldout":false,"admission_criterion":false,
        "scope":"same preselected views of already consumed fresh rows outside last64; exact source-aware native loss and support reads reused only for same example/coefficient hashes with native common bank checked; not complete blocks, recall presentations, rule recertification or MCTS",
        "seconds_including_load_output":started.elapsed().as_secs_f64(),"timings_are_not_campaign_throughput_or_causal_abba":true});
    save_json_new(&out.join("report.json"), &report)?;
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_flags_are_all_explicit() {
        let mut v = serde_json::json!({"fresh_active":true,"continuous_choices":true,"fresh_interior":true,"two_choice_constraints":false});
        assert!(final_leg(&v, false));
        assert!(!final_leg(&v, true));
        v.as_object_mut().unwrap().remove("two_choice_constraints");
        assert!(!final_leg(&v, false));
    }
    #[test]
    fn cache_requires_finite_complete_reads() {
        let v = serde_json::json!({"models":[{"role":"anchor","rows":[]}]});
        assert!(readings(&v, 0, 1).is_err());
    }
}
