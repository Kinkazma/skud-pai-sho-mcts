//! The first V2-to-V3 transition is a weight rebase, never a counter reset.
//! Pure snapshot preparation: runtime owns the single diagnostic-provenance write.
use super::*;

pub(super) fn migration_required(
    requested_v3: bool,
    has_resume: bool,
    saved_guard: &serde_json::Value,
) -> Result<bool> {
    if !requested_v3 || !has_resume { return Ok(false); }
    let was_v3 = match saved_guard.get("learning_loop_v3") {
        None => false,
        Some(value) => value.as_bool().ok_or_else(|| invalid("invalid saved publication protocol marker"))?,
    };
    if was_v3 { return Ok(false); }
    if !saved_guard["accepted_identity"].as_str().is_some_and(|s| !s.is_empty()) {
        // Guard::open would otherwise initialize from the old learner and turn
        // an unvalidated shadow into a supposedly accepted migration baseline.
        return Err(invalid("V2-to-V3 resume requires the saved accepted actor metadata"));
    }
    Ok(true)
}

pub(super) fn working_snapshot(
    migrate: bool,
    original: &Arc<Snapshot>,
    accepted: &Arc<Snapshot>,
    consumed_updates: u64,
    resumed_version: u64,
    path: PathBuf,
) -> Result<Arc<Snapshot>> {
    if !migrate {
        // The learner can legitimately differ from the actor halfway through a
        // V3 transaction. Do not infer a migration from unequal weights.
        return Ok(original.clone());
    }
    if original.model.schema()!=accepted.model.schema()
        || original.model.feature_schema()!=accepted.model.feature_schema()
        || original.model.parameters().len()!=accepted.model.parameters().len()
        || serde_json::to_value(original.model.sequence_memory().map(|b| &b.spec))?
            != serde_json::to_value(accepted.model.sequence_memory().map(|b| &b.spec))?
    {
        return Err(invalid("protocol weight rebase requires matching architecture and memory"));
    }
    let artifact = Arc::new(MicroArtifact::new(&accepted.model, consumed_updates,
        serde_json::json!({"kind":"gen5-protocol-migration-v2-to-v3",
            "source_learner":original.identity,"source_learner_path":original.path,
            "accepted_actor":accepted.identity,"accepted_actor_path":accepted.path,
            "updates_consumed":consumed_updates,"version":resumed_version,
            "no_extra_optimizer_step":true,"replay_and_recall_not_reset":true,
            "reason":"start the first V3 transaction from accepted weights"})));
    Ok(Arc::new(Snapshot {
        identity: artifact.identity(), artifact: Some(artifact),
        model: accepted.model.clone(), version: resumed_version, path,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot(seed: u64, updates: u64, version: u64) -> Arc<Snapshot> {
        let model=Arc::new(MicroModel::seeded(seed).with_deep_value(23));
        let artifact=Arc::new(MicroArtifact::new(&model,updates,serde_json::json!({"fixture":seed})));
        Arc::new(Snapshot {identity:artifact.identity(),artifact:Some(artifact),
            model,version,path:PathBuf::from(format!("fixture-{seed}.json"))})
    }
    #[test]
    fn only_saved_legacy_protocol_triggers_rebase() {
        let legacy=serde_json::json!({"accepted_identity":"accepted"});
        assert!(migration_required(true,true,&legacy).unwrap());
        assert!(migration_required(true,true,&serde_json::json!({"learning_loop_v3":false,"accepted_identity":"accepted"})).unwrap());
        assert!(!migration_required(false,true,&legacy).unwrap());
        assert!(!migration_required(true,false,&serde_json::Value::Null).unwrap());
        assert!(!migration_required(true,true,&serde_json::json!({"learning_loop_v3":true})).unwrap());
        assert!(migration_required(true,true,&serde_json::Value::Null).is_err());
        assert!(migration_required(true,true,&serde_json::json!({"learning_loop_v3":"true"})).is_err());
    }
    #[test]
    fn migration_uses_actor_weights_but_consumed_learner_counters() {
        let shadow=snapshot(1,900,27);
        let actor=snapshot(2,12,3);
        let original_artifact=serde_json::to_vec(shadow.artifact.as_ref().unwrap().as_ref()).unwrap();
        let original_identity=shadow.identity.clone();
        let selected=working_snapshot(true,&shadow,&actor,900,27,PathBuf::from("migration.json")).unwrap();
        assert!(Arc::ptr_eq(&selected.model,&actor.model));
        assert_eq!(selected.version,27);
        let artifact=selected.artifact.as_ref().unwrap();
        assert_eq!(artifact.updates,900);
        assert_eq!(artifact.provenance["source_learner"],original_identity);
        assert_eq!(artifact.provenance["accepted_actor"],actor.identity);
        assert_eq!(artifact.provenance["updates_consumed"],900);
        assert_eq!(serde_json::to_vec(shadow.artifact.as_ref().unwrap().as_ref()).unwrap(),original_artifact);
        assert_eq!(shadow.identity,original_identity);
        assert_eq!(actor.artifact.as_ref().unwrap().updates,12);
    }
    #[test]
    fn real_v3_mid_transaction_keeps_the_exact_shadow_snapshot() {
        let shadow=snapshot(1,900,27);
        let actor=snapshot(2,12,3);
        assert_ne!(shadow.model.parameters(),actor.model.parameters());
        let saved=serde_json::json!({"learning_loop_v3":true,"accepted_identity":actor.identity});
        let migration=migration_required(true,true,&saved).unwrap();
        let selected=working_snapshot(migration,&shadow,&actor,900,27,PathBuf::from("unused.json")).unwrap();
        assert!(Arc::ptr_eq(&selected,&shadow));
        assert_eq!(selected.artifact.as_ref().unwrap().updates,900);
        assert_eq!(selected.version,27);
        assert_eq!(selected.path,shadow.path);
    }
}
