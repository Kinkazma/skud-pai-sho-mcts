//! Offline path/identity migration and native recovery checks. No training process.
use super::*;
use serde_json::{json, Value};

fn read(p: &Path) -> Result<Value> { Ok(serde_json::from_slice(&fs::read(p)?)?) }
fn file(v: &Value) -> Result<PathBuf> { Ok(PathBuf::from(v.as_str().ok_or_else(||invalid("missing portable path"))?)) }
fn write(p: &Path, v: &Value) -> Result<()> { fs::write(p,serde_json::to_vec(v)?)?;Ok(()) }
fn refresh(v: &mut Value) -> Result<()> {
    if let Some(o)=v.as_object_mut() {
        if let (Some(path),Some(_))=(o.get("path").and_then(Value::as_str),o.get("sha256")) {
            let h=sha256(&fs::read(path)?);o.insert("sha256".into(),json!(h));
        }
        for child in o.values_mut() {refresh(child)?;}
    } else if let Some(rows)=v.as_array_mut() { for child in rows {refresh(child)?;} }
    Ok(())
}

/// Reconcile only metadata identities after privacy-preserving file relocation.
/// Numeric model tokens are never rewritten by this command.
pub fn finalize(config: &Path) -> Result<Value> {
    let dir=config.parent().ok_or_else(||invalid("config parent"))?;
    let mut c=read(config)?;
    let mut p=read(&dir.join("progress.partial.json"))?;
    let initial=MicroArtifact::load(&file(&c["model"])?)?;
    if p["updates"] != initial.updates {return Err(invalid("learner update counter changed"));}
    let actor:MicroArtifact=serde_json::from_slice(&fs::read(file(&p["publication_guard"]["accepted_path"])?)?)?;
    let identity=actor.identity();
    p["publication_guard"]["accepted_identity"]=json!(identity);
    let registry=&mut p["publication_guard"]["learned_choices"];
    registry["payload"]["last_validated_actor"]=json!(identity);
    registry["sha256"]=json!(sha256(&serde_json::to_vec(&registry["payload"])?));
    for reference in c["opponents"].as_array_mut().ok_or_else(||invalid("opponents"))? {
        let path=file(&reference["model"])?;let mut model=read(&path)?;
        if let Some(manifest)=model.get("memory_manifest").and_then(Value::as_str) {
            let hash=sha256(&fs::read(manifest)?);
            model["memory_manifest_sha256"]=json!(hash);write(&path,&model)?;
        }
        reference["sha256"]=json!(sha256(&fs::read(&path)?));
        let generation=reference["generation"].as_str().ok_or_else(||invalid("generation"))?;
        p["opponent_ladders"][generation]["reference"]=reference["sha256"].clone();
    }
    p["legacy_ladder"]["reference"]=json!(sha256(&fs::read(file(&c["reference"])?)?));
    for (i,e) in p["frozen_evaluations"].as_array_mut().ok_or_else(||invalid("evaluations"))?.iter_mut().enumerate() {
        let model:MicroArtifact=serde_json::from_slice(&fs::read(file(&e["path"])?)?)?;
        e["model"]=json!(model.identity());e["reference"]=c["opponents"][i]["sha256"].clone();
    }
    for (field,key) in [("manifest","publication_guard"),("validation_manifest","publication_validation")] {
        p["publication_guard"][field]=json!(sha256(&fs::read(file(&c[key])?)?));
    }
    refresh(&mut p)?;
    let case_out=dir.join("case-check");fs::create_dir_all(&case_out)?;
    // This checks all legal human prefixes, their order and the exact split.
    cases::load(&file(&c["human_dataset"])?,c["seed"].as_u64().ok_or_else(||invalid("seed"))?,&case_out)?;
    let actual=read(&case_out.join("human-cases.json"))?;
    let mut expected=read(&dir.join("human-cases.original.json"))?;
    expected["dataset_sha256"]=actual["dataset_sha256"].clone();
    if actual!=expected {return Err(invalid("portable human cases/order changed"));}
    p["case_manifest_sha256"]=json!(sha256(&fs::read(case_out.join("human-cases.json"))?));
    write(&file(&c["resume_progress"])?,&p)?;write(config,&c)?;
    let report=json!({"schema":"paisho-gen5-portable-finalization-v1","updates":initial.updates,
        "learner":initial.identity(),"actor":identity,"version":p["version"],
        "actor_version":p["publication_guard"]["accepted_version"],"replay_positions":p["replay_positions"],
        "human_cases_order_exact":true,"weights_rewritten":false,"training_started":false});
    write(&dir.join("finalization.json"),&report)?;Ok(report)
}

pub fn verify(config: &Path, out: &Path) -> Result<Value> {
    if out.exists(){return Err(invalid("verification output must be new"));}fs::create_dir_all(out)?;
    let o:Options=serde_json::from_slice(&fs::read(config)?)?;
    // Validate every opponent's nested memory dependency before the large FIFO.
    let opponents=references::load_all(&o.opponents)?;
    let p=read(o.resume_progress.as_ref().ok_or_else(||invalid("resume state"))?)?;
    let a=MicroArtifact::load(&o.model)?;let model=Arc::new(a.model()?);
    let initial=Arc::new(Snapshot{artifact:Some(Arc::new(a.clone())),model:model.clone(),
        identity:a.identity(),version:p["version"].as_u64().ok_or_else(||invalid("version"))?,path:o.model.clone()});
    let mut guard=publication::Guard::open(o.publication_guard.as_ref().unwrap(),out,o.value_policy_strength,initial,&p["publication_guard"])?;
    guard.enable_v2(o.publication_validation.as_ref().unwrap())?;guard.enable_v3()?;
    let pools=vec![Arc::new(rayon::ThreadPoolBuilder::new().num_threads(4).build()?)];
    guard.enable_parallel(&pools);guard.enable_transfer()?;
    let mut protection=protection::Protection::new(&model,guard.reference_examples())?;
    protection.enable_parallel(&pools);protection.enable_loop_v3();protection.enable_validation_value(guard.diagnostic_validation_examples()?)?;
    protection.restore(&p["protection"])?;
    let mut memory=memory::Memory::new(&o);memory.load_for_model(o.replay_index.as_ref().unwrap(),model.has_spatial())?;
    if json!(memory.len())!=p["replay_positions"] {return Err(invalid("FIFO count changed"));}
    let catalogue=cases::load(o.human_dataset.as_ref().unwrap(),o.seed,out)?;
    if json!(sha256(&fs::read(out.join("human-cases.json"))?))!=p["case_manifest_sha256"] {return Err(invalid("case manifest changed"));}
    let _evaluations=evaluation::Evaluations::open(out,&catalogue,&o,&p["frozen_evaluations"],&model)?;
    let mut recall=durable::Archive::open(&o.case_curriculum.as_ref().unwrap().archive)?;
    recall.policy_consolidation(o.learning_loop_repair);
    for root in &o.recall_archive_sources {recall.add_read_only(root)?;}
    recall.restore(&p["durable_recall"])?;
    recall.cache_enabled=o.structural_repair;recall.proof_recall=o.proof_recall;
    recall.trusted_action_values=o.learning_loop_v3;
    recall.enable_coverage(out,&p["durable_recall"],&model,&o.case_curriculum.as_ref().unwrap().archive)?;
    recall.seed_values(guard.reference_examples());
    if let Some(anchors)=&o.structured_recall_anchors {recall.restore_structured_anchors(anchors)?;}
    recall.focus_policy(guard.correction_examples()?);
    let mut rng=StableRng::new(p["learner_rng"].as_u64().ok_or_else(||invalid("learner RNG"))?);
    use sha2::{Digest,Sha256};
    let mut recall_hash=Sha256::new();
    for _ in 0..32 {
        for ex in recall.rehearse_cached(32,&mut rng,&model)? {memory::example_digest(&mut recall_hash,&ex)?;}
    }
    let recall_fingerprint=format!("{:x}",recall_hash.finalize());
    // Native load validates all FIFO source hashes and parses all retained rows.
    let result=json!({"schema":"paisho-gen5-native-recovery-check-v1","updates":a.updates,
        "learner_identity":a.identity(),"actor_identity":guard.accepted().identity,
        "replay_positions":memory.len(),"fifo_bits":memory.portable_digest()?,"guard_and_acquisitions_restored":true,
        "frozen_opponents_loaded":opponents.len(),
        "protection_restored":true,"frozen_evaluations_restored":true,
        "recall_draws":1024,"recall_sha256":recall_fingerprint,"recall_after":recall.progress(),
        "training_started":false,"production_writes":0});
    write(&out.join("report.json"),&result)?;Ok(result)
}
