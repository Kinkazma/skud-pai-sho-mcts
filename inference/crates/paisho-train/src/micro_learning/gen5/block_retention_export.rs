//! Bounded source export only. No model, forward, gradient, replay or campaign.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Deserialize, Serialize)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn bytes(&self) -> Result<Vec<u8>> {
        let bytes = fs::read(&self.path)?;
        if sha256(&bytes) != self.sha256 {
            return Err(invalid(format!(
                "changed block-retention input {}",
                self.path.display()
            )));
        }
        Ok(bytes)
    }
}
#[derive(Deserialize)]
struct Manifest {
    schema: String,
    plan: Input,
    blocks: Vec<Input>,
    captures: Vec<Input>,
}
#[derive(Deserialize)]
struct Row {
    receipt_ordinal: usize,
    receipt_id: usize,
    saved_index: usize,
    decision: usize,
    state_sha256: String,
    source_sha256: String,
    source_identity_basis: String,
    targets: Input,
    receipt: Input,
    psr: Input,
    rules: String,
    collector: String,
}
#[derive(Deserialize)]
struct Entry {
    union_index: usize,
    weight: f64,
    weight_numerator: u64,
    weight_denominator: u64,
}
#[derive(Deserialize)]
struct View {
    name: String,
    entries: Vec<Entry>,
}
#[derive(Deserialize)]
struct Block {
    schema: String,
    event: usize,
    plan: Input,
    capture: Input,
    boundary_receipt: usize,
    union_saved: Input,
    union: Vec<Row>,
    rows: usize,
    views: Vec<View>,
    already_learned: bool,
    heldout: bool,
    admission_criterion: bool,
}
fn state_sha(state: &[f64]) -> String {
    sha256(
        &state
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn linked_file(directory: &Path, value: &serde_json::Value) -> Result<Input> {
    let name = value["file"]
        .as_str()
        .ok_or_else(|| invalid("missing linked filename"))?;
    if Path::new(name).file_name().and_then(|s| s.to_str()) != Some(name) {
        return Err(invalid("unsafe linked filename"));
    }
    Ok(Input {
        path: directory.join(name),
        sha256: value["sha256"]
            .as_str()
            .ok_or_else(|| invalid("missing linked digest"))?
            .into(),
    })
}
fn save(path: &Path, value: &impl Serialize) -> Result<Input> {
    use std::io::Write;
    let bytes = serde_json::to_vec(value)?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(Input {
        path: path.to_path_buf(),
        sha256: sha256(&bytes),
    })
}
/// MANIFEST NEW_OUTPUT; the output contains native trusted ResumeExamples only.
/// This does not load any model or label an estimated target as a regulatory proof.
pub fn run(manifest_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("block-retention output must be new"));
    }
    let bytes = fs::read(manifest_path)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    if manifest.schema != "paisho-gen5-c1-block-retention-manifest-v1"
        || manifest.blocks.len() != 4
        || manifest.captures.len() != 4
    {
        return Err(invalid("block-retention requires the four fixed captures"));
    }
    let plan: serde_json::Value = serde_json::from_slice(&manifest.plan.bytes()?)?;
    if plan["schema"] != "paisho-gen5-c1-block-retention-plan-v1"
        || plan["maximum_rows_per_view"] != 128
        || plan["maximum_union_rows_per_block"] != 256
        || plan["seed"] != "gen5-c1-block-retention-2026-09-13-v1"
    {
        return Err(invalid("unsupported fixed block-retention plan"));
    }
    // Hold only <=256 converted rows per block, no model bank or dense FIFO.
    fs::create_dir_all(out)?;
    let mut reports = vec![];
    for (i, input) in manifest.blocks.iter().enumerate() {
        let block_bytes = input.bytes()?;
        let block: Block = serde_json::from_slice(&block_bytes)?;
        if block.schema != "paisho-gen5-c1-block-retention-selection-v1"
            || block.event != i
            || block.plan.sha256 != manifest.plan.sha256
            || block.capture.sha256 != manifest.captures[i].sha256
            || block.rows != block.union.len()
            || block.rows > 256
            || block.views.len() != 2
            || !block.already_learned
            || block.heldout
            || block.admission_criterion
        {
            return Err(invalid("changed block-retention scope or row count"));
        }
        let capture: serde_json::Value = serde_json::from_slice(&block.capture.bytes()?)?;
        if capture["status"] != "complete"
            || capture["context"]["receipt_id"].as_u64() != Some(block.boundary_receipt as u64)
        {
            return Err(invalid("block-retention capture mismatch"));
        }
        let directory = block
            .capture
            .path
            .parent()
            .ok_or_else(|| invalid("capture has no directory"))?;
        let tail_input = linked_file(directory, &capture["examples"]["fresh"])?;
        let tail: Vec<resume_example::ResumeExample> =
            serde_json::from_slice(&tail_input.bytes()?)?;
        let mut tail_states = BTreeSet::new();
        for row in tail {
            tail_states.insert(state_sha(&row.example_with_trusted_q(true)?.state));
        }
        let selected: Vec<SavedMicroExample> = decode_examples(&block.union_saved.bytes()?)?;
        if selected.len() != block.rows {
            return Err(invalid("selected Saved row count changed"));
        }
        let mut current_path = None;
        let mut saved = vec![];
        let mut current_receipt = serde_json::Value::Null;
        let mut resume = vec![];
        let mut keys = BTreeSet::new();
        let mut sources = BTreeSet::new();
        let mut previous = None;
        for (index, row) in block.union.iter().enumerate() {
            let key = (row.receipt_ordinal, row.saved_index);
            if !keys.insert(key) || previous.is_some_and(|p| p >= key) {
                return Err(invalid("selection chronology or uniqueness changed"));
            }
            previous = Some(key);
            if current_path.as_ref() != Some(&row.targets.path) {
                saved = decode_examples(&row.targets.bytes()?)?;
                current_receipt = serde_json::from_slice(&row.receipt.bytes()?)?;
                row.psr.bytes()?;
                if current_receipt["fully_learned"] != true
                    || current_receipt["fresh_used"] != current_receipt["eligible_examples"]
                    || current_receipt["eligible_examples"].as_u64() != Some(saved.len() as u64)
                    || current_receipt["targets_sha256"] != row.targets.sha256
                    || current_receipt["psr_sha256"] != row.psr.sha256
                {
                    return Err(invalid(
                        "source receipt is not an exact fully consumed fresh bundle",
                    ));
                }
                current_path = Some(row.targets.path.clone());
            }
            let source = saved
                .get(row.saved_index)
                .ok_or_else(|| invalid("selected Saved index is absent"))?;
            if current_receipt["id"].as_u64() != Some(row.receipt_id as u64)
                || source.game_id != row.receipt_id.to_string()
                || source.decision != row.decision
                || source.collector != row.collector
                || source.rules != row.rules
                || source.rules != RULES.as_str()
                || serde_json::to_vec(source)? != serde_json::to_vec(&selected[index])?
            {
                return Err(invalid("selected row differs from hashed native archive"));
            }
            let source_state = state_sha(&source.state);
            if source_state != row.state_sha256 || tail_states.contains(&source_state) {
                return Err(invalid(
                    "row identity changed or overlaps captured last64 state",
                ));
            }
            let evidence = serde_json::to_value(&source.evidence)?;
            let observed = evidence["observed_psr"].as_str();
            let source_hash = observed.unwrap_or(&row.psr.sha256);
            let basis = if observed.is_some() {
                "observed_psr"
            } else {
                "receipt_psr"
            };
            if source_hash != row.source_sha256 || basis != row.source_identity_basis {
                return Err(invalid("source grouping differs from original evidence"));
            }
            sources.insert(row.source_sha256.clone());
            let example = source.example_for_rules_with_trusted_q(RULES, true)?;
            let converted = resume_example::ResumeExample::from_with_trusted_q(&example, true);
            let restored = converted.clone().example_with_trusted_q(true)?;
            if serde_json::to_vec(&converted)?
                != serde_json::to_vec(&resume_example::ResumeExample::from_with_trusted_q(
                    &restored, true,
                ))?
            {
                return Err(invalid("native trusted ResumeExample roundtrip changed"));
            }
            resume.push(converted);
        }
        let mut weights = BTreeMap::new();
        for (v, view) in block.views.iter().enumerate() {
            if view.name != if v == 0 { "A-position" } else { "B-source" }
                || view.entries.len() > 128
            {
                return Err(invalid("measurement views or budget changed"));
            }
            let mut ids = BTreeSet::new();
            let mut total = 0.;
            for entry in &view.entries {
                if entry.union_index >= block.rows
                    || !ids.insert(entry.union_index)
                    || entry.weight_denominator == 0
                    || entry.weight_numerator == 0
                    || !entry.weight.is_finite()
                    || (entry.weight
                        - entry.weight_numerator as f64 / entry.weight_denominator as f64)
                        .abs()
                        > 1e-15
                {
                    return Err(invalid("invalid fixed sampling weight"));
                }
                total += entry.weight;
            }
            if !view.entries.is_empty() && (total - 1.).abs() > 1e-12 {
                return Err(invalid("measurement weights do not sum to one"));
            }
            weights.insert(
                view.name.clone(),
                serde_json::json!({"rows":view.entries.len(),"sum":total}),
            );
        }
        let path = out.join(format!("block-{i:03}.resume.json"));
        let exported = save(&path, &resume)?;
        reports.push(serde_json::json!({"event":i,"selection":input,"capture":block.capture,
            "examples":exported,"rows":resume.len(),"sources":sources.len(),"views":weights,
            "sampling_weights_and_original_indices_in_selection":true,"captured_tail_state_overlap":0,
            "native_conversion":"SavedMicroExample::example_for_rules_with_trusted_q(RULES,true) -> ResumeExample::from_with_trusted_q(true)",
            "original_source_bytes_and_consumption_checked":true,"resume_roundtrip_exact":true,
            "regulatory_proof_replay_performed":false,"model_reads":0,"model_calculations":0}));
    }
    let result = serde_json::json!({"schema":"paisho-gen5-c1-block-retention-native-export-v1",
        "manifest":{"path":manifest_path,"sha256":sha256(&bytes)},"blocks":reports,
        "already_learned":true,"heldout":false,"used_for_training_or_admission":false,
        "model_reads":0,"model_calculations":0});
    save(&out.join("report.json"), &result)?;
    Ok(result)
}
