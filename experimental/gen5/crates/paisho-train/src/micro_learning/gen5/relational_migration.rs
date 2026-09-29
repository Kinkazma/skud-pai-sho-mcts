//! Explicit offline V6 -> neutral V7 recovery conversion. Never starts a runtime.
use super::*;
use serde_json::{json, Value};

fn read(path: &Path, inputs: &mut Vec<(PathBuf, String)>) -> Result<Vec<u8>> {
    let bytes = fs::read(path)?;
    inputs.push((path.into(), sha256(&bytes)));
    Ok(bytes)
}
fn model_from(a: &MicroArtifact, base: &MicroModel) -> Result<MicroModel> {
    if a.schema != paisho_ai::MICRO_NEURAL_MEMORY_MODEL_SCHEMA
        || a.feature_schema != base.feature_schema()
        || serde_json::to_value(&a.sequence_memory)?
            != serde_json::to_value(base.sequence_memory().map(|b| &b.spec))?
    {
        return Err(invalid(
            "neutral recovery migration requires matching V6 schema and bank",
        ));
    }
    let mut m = MicroModel::from_parameters(a.parameters.clone()).map_err(invalid)?;
    if let Some(bank) = base.sequence_memory() {
        m = m.with_sequence_memory_owned(bank.clone());
    }
    Ok(m)
}
fn migrate(a: &MicroArtifact, base: &MicroModel, seed: u64, out: &Path) -> Result<MicroArtifact> {
    let old = model_from(a, base)?;
    let new = old.with_relational(seed);
    if old
        .parameters()
        .iter()
        .zip(new.parameters())
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(invalid("neutral migration changed an existing weight"));
    }
    let artifact = MicroArtifact::new(
        &new,
        a.updates,
        json!({"kind":"neutral-v6-to-v7-recovery",
        "source_identity":a.identity(),"source_provenance":a.provenance,"seed":seed,"no_optimizer_step":true}),
    );
    save_json_new(out, &artifact)?;
    Ok(artifact)
}
fn exact_reads(old: &MicroModel, new: &MicroModel, rows: &[Arc<MicroExample>]) -> Result<usize> {
    for e in rows {
        let a = old
            .policy_value_priors(&e.state, &e.actions, e.sequence_source)
            .map_err(invalid)?;
        let b = new
            .policy_value_priors(&e.state, &e.actions, e.sequence_source)
            .map_err(invalid)?;
        if a.0.to_bits() != b.0.to_bits()
            || a.1
                .iter()
                .zip(&b.1)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || a.1.len() != b.1.len()
        {
            return Err(invalid(
                "recovery migration changed policy/value on a protected row",
            ));
        }
    }
    Ok(rows.len())
}

/// Writes a private recovery proposal, retaining separate learner, actor, anchor,
/// FIFO and historical proof provenance. `out` must not exist. No weights promoted.
pub fn prepare(config: &Path, progress: &Path, seeds: &Path, out: &Path) -> Result<Value> {
    if out.exists() {
        return Err(invalid("migration output must be a new directory"));
    }
    fs::create_dir(out)?;
    let out = fs::canonicalize(out)?;
    let mut inputs = vec![];
    let o: Options = serde_json::from_slice(&read(config, &mut inputs)?)?;
    let mut p: Value = serde_json::from_slice(&read(progress, &mut inputs)?)?;
    if !o.learning_loop_v3
        || !o.publication_transfer
        || p["publication_guard"]["learning_loop_v3"] != true
    {
        return Err(invalid(
            "relational recovery requires existing V3 transfer protections",
        ));
    }
    let old: MicroArtifact = serde_json::from_slice(&read(
        Path::new(
            p["checkpoint_model"]
                .as_str()
                .ok_or_else(|| invalid("missing learner"))?,
        ),
        &mut inputs,
    )?)?;
    if p["updates"].as_u64() != Some(old.updates) {
        return Err(invalid("learner counters differ"));
    }
    let base = old.model()?;
    let seed = 20260926;
    let learner_path = out.join("learner-v7.json");
    let learner = migrate(&old, &base, seed, &learner_path)?;
    let actor_old: MicroArtifact = serde_json::from_slice(&read(
        Path::new(
            p["publication_guard"]["accepted_path"]
                .as_str()
                .ok_or_else(|| invalid("missing actor"))?,
        ),
        &mut inputs,
    )?)?;
    if actor_old.identity() != p["publication_guard"]["accepted_identity"] {
        return Err(invalid("actor identity mismatch"));
    }
    let actor_path = out.join("actor-v7.json");
    let actor = migrate(&actor_old, &base, seed, &actor_path)?;
    p["checkpoint_model"] = json!(learner_path);
    p["publication_guard"]["accepted_path"] = json!(actor_path);
    p["publication_guard"]["accepted_identity"] = json!(actor.identity());
    let registry = &mut p["publication_guard"]["learned_choices"];
    if registry["sha256"] != sha256(&serde_json::to_vec(&registry["payload"])?)
        || registry["payload"]["last_validated_actor"] != actor_old.identity()
    {
        return Err(invalid("registry source mismatch"));
    }
    registry["payload"]["last_validated_actor"] = json!(actor.identity());
    registry["sha256"] = json!(sha256(&serde_json::to_vec(&registry["payload"])?));
    let anchor_bytes = read(
        Path::new(
            p["protection"]["checkpoint"]["path"]
                .as_str()
                .ok_or_else(|| invalid("missing anchor"))?,
        ),
        &mut inputs,
    )?;
    if sha256(&anchor_bytes) != p["protection"]["checkpoint"]["sha256"] {
        return Err(invalid("anchor hash mismatch"));
    }
    let mut anchor: Value = serde_json::from_slice(&anchor_bytes)?;
    let anchor_old = MicroModel::from_parameters(serde_json::from_value(anchor["anchor"].clone())?)
        .map_err(invalid)?;
    if anchor_old.schema() != paisho_ai::MICRO_NEURAL_MEMORY_MODEL_SCHEMA {
        return Err(invalid("anchor schema mismatch"));
    }
    anchor["anchor"] = json!(anchor_old.with_relational(seed).parameters());
    let anchor_path = out.join("protection-v7.json");
    let bytes = serde_json::to_vec(&anchor)?;
    fs::write(&anchor_path, &bytes)?;
    p["protection"]["checkpoint"] = json!({"path":anchor_path,"sha256":sha256(&bytes)});
    // FIFO entries and target archives are deliberately unchanged. The native
    // reader validates their hashes; this tool never creates an empty replay.
    let replay_path = Path::new(
        p["checkpoint_replay_index"]
            .as_str()
            .ok_or_else(|| invalid("missing replay"))?,
    );
    let replay_bytes = read(replay_path, &mut inputs)?;
    let replay_copy = out.join("replay.index.json");
    fs::write(&replay_copy, &replay_bytes)?;
    p["checkpoint_replay_index"] = json!(replay_copy);
    let mut memory = memory::Memory::new(&o);
    memory.load_for_model(&replay_copy, true)?;
    if Some(memory.len() as u64) != p["replay_positions"].as_u64() {
        return Err(invalid("migration lost FIFO positions"));
    }
    let learner_model = base.with_relational(seed);
    let seed_examples: Vec<SavedMicroExample> =
        serde_json::from_slice(&read(&seeds.join("seed-examples.json"), &mut inputs)?)?;
    let seed_sources: Vec<Value> =
        serde_json::from_slice(&read(&seeds.join("seed-sources.json"), &mut inputs)?)?;
    p["durable_recall"]["structured"] = durable::prepare_seed_recall(
        &seed_examples,
        &seed_sources,
        &learner_model,
        &p["durable_recall"]["structured"],
        &out.join("structured-recall"),
    )?;
    let initial = Arc::new(Snapshot {
        model: Arc::new(learner_model.clone()),
        artifact: Some(Arc::new(learner.clone())),
        path: learner_path,
        identity: learner.identity(),
        version: p["version"].as_u64().ok_or_else(|| invalid("version"))?,
    });
    let mut guard = publication::Guard::open(
        o.publication_guard
            .as_ref()
            .ok_or_else(|| invalid("primary guard"))?,
        &out.join("validation"),
        o.value_policy_strength,
        initial,
        &p["publication_guard"],
    )?;
    guard.enable_v2(
        o.publication_validation
            .as_ref()
            .ok_or_else(|| invalid("secondary guard"))?,
    )?;
    guard.enable_v3()?;
    let pools = vec![Arc::new(
        rayon::ThreadPoolBuilder::new().num_threads(1).build()?,
    )];
    guard.enable_parallel(&pools);
    guard.enable_transfer()?;
    let rows = guard.reference_examples();
    let secondary = guard.diagnostic_validation_examples()?;
    let actor_model = model_from(&actor_old, &base)?;
    let exact = exact_reads(&base, &learner_model, &rows)?
        + exact_reads(&base, &learner_model, &secondary)?
        + exact_reads(&actor_model, &guard.accepted().model, &rows)?
        + exact_reads(&actor_model, &guard.accepted().model, &secondary)?;
    let mut protection = protection::Protection::new(&learner_model, rows)?;
    protection.enable_parallel(&pools);
    protection.enable_loop_v3();
    protection.enable_validation_value(secondary)?;
    protection.restore(&p["protection"])?;
    p["initial_weight_migration"] = json!({"kind":"neutral-v6-to-v7-recovery","learner_before":old.identity(),
        "actor_before":actor_old.identity(),"learner_after":learner.identity(),"actor_after":actor.identity(),"seed":seed,
        "updates_unchanged":true,"historical_acquisitions_unchanged":true,"no_optimizer_step":true,
        "structured_seed_sources":seeds,"seed_examples":seed_examples.len(),"recall_fraction_unchanged":true});
    save_json_new(&out.join("resume-progress.json"), &p)?;
    for (path, hash) in &inputs {
        if sha256(&fs::read(path)?) != *hash {
            return Err(invalid("source changed during migration"));
        }
    }
    let result = json!({"schema":"paisho-gen5-neutral-relational-recovery-v1","prepared_only":true,"activated":false,
        "exact_policy_value_reads":exact,"native_guard_and_registry_restored":true,"native_protection_restored":true,
        "fifo_positions":memory.len(),"updates":old.updates,"source_inputs":inputs,"learner":learner.identity(),"actor":actor.identity(),
        "structured_recall":p["durable_recall"]["structured"]});
    save_json_new(&out.join("verification.json"), &result)?;
    Ok(result)
}
