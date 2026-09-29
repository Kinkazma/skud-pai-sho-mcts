//! Fixed-example factorial read: policy weights versus the reader's scalar V input.
//! Neither changes trained weights nor claims to attribute the SGD trajectory.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContextPlan {
    schema: String,
    measurement_plan: Input,
    measured: Input,
}
fn decomposition(p00: f64, p10: f64, p01: f64, p11: f64) -> [f64; 3] {
    [p11 - p00, 0.5 * ((p10 - p00) + (p11 - p01)),
        0.5 * ((p01 - p00) + (p11 - p10))]
}

pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() { return Err(invalid("context read output must be new")); }
    let started = Instant::now();
    let bytes = fs::read(plan_path)?;
    let p: ContextPlan = serde_json::from_slice(&bytes)?;
    if p.schema != "paisho-gen5-reader-context-drift-plan-v1" {
        return Err(invalid("unknown context read plan"));
    }
    let source_bytes = p.measurement_plan.bytes()?;
    let plan: ReadPlan = serde_json::from_slice(&source_bytes)?;
    let measured: serde_json::Value = serde_json::from_slice(&p.measured.bytes()?)?;
    if plan.schema != "paisho-gen5-c1-weighted-read-plan-v1" || plan.blocks.len() != 4
        || plan.roles != ROLES || measured["schema"] != "paisho-gen5-c1-weighted-retention-v1"
        || measured["plan_sha256"] != sha256(&source_bytes)
        || measured["blocks"].as_array().map(|a| a.len()) != Some(4) {
        return Err(invalid("requires completed exact same four-block weighted read"));
    }
    let replay: serde_json::Value = serde_json::from_slice(&plan.replay.bytes()?)?;
    plan.export.bytes()?;
    fs::create_dir(out)?;
    let mut bank_model = None;
    let mut results = vec![];
    for (i, b) in plan.blocks.iter().enumerate() {
        let capture: serde_json::Value = serde_json::from_slice(&b.capture.bytes()?)?;
        let selection: Selection = serde_json::from_slice(&b.selection.bytes()?)?;
        selection.capture.bytes()?;
        let examples = rows(&b.examples.bytes()?)?;
        let previous = &measured["blocks"][i];
        if previous["examples"]["sha256"] != b.examples.sha256
            || previous["selection"]["sha256"] != b.selection.sha256
            || selection.event != i || selection.rows != examples.len()
            || selection.views.len() != 2 || b.models.len() != 8 {
            return Err(invalid("context examples/selection no longer match previous read"));
        }
        let descriptor: Input = serde_json::from_value(previous["row_reads"].clone())?;
        let old: serde_json::Value = serde_json::from_slice(&descriptor.bytes()?)?;
        let mut models = vec![];
        for (r, m) in b.models.iter().take(2).enumerate() {
            if m.role != ROLES[r] { return Err(invalid("context model role changed")); }
            let a: MicroArtifact = serde_json::from_slice(&m.input.bytes()?)?;
            verify_model_link(i, &m.role, &m.input, &a, &capture, &replay)?;
            let model = model(&a, bank_model.as_ref())?;
            if bank_model.is_none() { bank_model = Some(model.clone()); }
            if bits_hash(&model) != m.parameter_bits_sha256
                || previous["model_parameter_hashes_in_role_order"][r] != m.parameter_bits_sha256
                || old["models"][r]["role"] != m.role
                || old["models"][r]["rows"].as_array().map(|x| x.len()) != Some(examples.len()) {
                return Err(invalid("context model/previous reading binding changed"));
            }
            models.push(model);
        }
        let timer = Instant::now();
        let mut readings = vec![];
        for (j, e) in examples.iter().enumerate() {
            let v = [models[0].value(&e.state), models[1].value(&e.state)];
            let mut cells = [0.; 4];
            for r in 0..2 {
                let own = models[r].loss_with_value_context(e, v[r]).map_err(invalid)?;
                let weighted = own.policy * e.policy_weight;
                let cached = old["models"][r]["rows"][j]["weighted_policy"].as_f64()
                    .ok_or_else(|| invalid("missing cached weighted policy"))?;
                let cached_v = old["models"][r]["rows"][j]["predicted_value"].as_f64()
                    .ok_or_else(|| invalid("missing cached predicted value"))?;
                if weighted.to_bits() != cached.to_bits() || v[r].to_bits() != cached_v.to_bits() {
                    return Err(invalid("own-context read fails exact previous native reading"));
                }
                cells[if r == 0 {0} else {3}] = weighted;
                cells[if r == 0 {2} else {1}] = models[r].loss_with_value_context(e, v[1-r])
                    .map_err(invalid)?.policy * e.policy_weight;
            }
            readings.push(cells);
        }
        let mut views = vec![];
        for view in &selection.views {
            validate_view(view, examples.len())?;
            let mut cells = [0.; 4];
            for entry in &view.entries {
                for (sum, value) in cells.iter_mut().zip(readings[entry.union_index]) {
                    *sum += entry.weight * value;
                }
            }
            let [total, parameters, context] = decomposition(cells[0], cells[1], cells[2], cells[3]);
            views.push(serde_json::json!({"name":view.name,"policy_00_10_01_11":cells,
                "total_delta":total,"symmetric_parameter_contribution":parameters,
                "symmetric_value_context_contribution":context,
                "decomposition_rounding_residual":total-parameters-context}));
        }
        let path = out.join(format!("block-{i:03}-rows.json"));
        save_json_new(&path, &serde_json::json!({"policy_00_10_01_11":readings}))?;
        results.push(serde_json::json!({"event":i,"check":capture["context"]["check"],"rows":examples.len(),
            "all_own_context_reads_bit_exact":true,"views":views,"read_seconds":timer.elapsed().as_secs_f64(),
            "rows_file":{"path":path,"sha256":sha256(&fs::read(&path)?)}}));
    }
    p.measurement_plan.bytes()?; p.measured.bytes()?;
    for b in &plan.blocks {
        b.capture.bytes()?; b.selection.bytes()?; b.examples.bytes()?;
        for m in b.models.iter().take(2) { m.input.bytes()?; }
    }
    if fs::read(plan_path)? != bytes {return Err(invalid("context plan changed"));}
    let report = serde_json::json!({"schema":"paisho-gen5-reader-context-drift-v1", "plan_sha256":sha256(&bytes),
        "blocks":results,"one_shared_bank":true,"new_sgd_or_games":false,"heldout":false,
        "interpretation":"two-order symmetric finite decomposition of observed policy loss drift; not SGD gradient causality or playing strength",
        "elapsed_seconds":started.elapsed().as_secs_f64()});
    save_json_new(&out.join("report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn symmetric_decomposition_accounts_for_interaction_without_double_counting() {
        assert_eq!(decomposition(1., 3., 5., 11.), [10., 4., 6.]);
        assert_eq!(decomposition(3., 1., 11., 5.), [2., -4., 6.]);
    }
}
