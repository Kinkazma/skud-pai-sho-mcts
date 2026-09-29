use paisho_ai::*;
use paisho_train::micro_learning::MicroArtifact;
#[test]
fn legacy_and_residual_artifacts_roundtrip_with_strict_schema_identity() {
    let old = MicroArtifact::new(
        &MicroModel::seeded(17),
        43,
        serde_json::json!({"test":true}),
    );
    let new = MicroArtifact::new(
        &old.model().unwrap().with_residual_policy(29),
        old.updates,
        serde_json::json!({"parent":old.identity()}),
    );
    assert_eq!(old.schema, MICRO_MODEL_SCHEMA);
    assert_eq!(new.schema, MICRO_RESIDUAL_MODEL_SCHEMA);
    assert_ne!(old.identity(), new.identity());
    for a in [old, new] {
        let loaded: MicroArtifact =
            serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        assert_eq!(a.identity(), loaded.identity());
        assert_eq!(a.model().unwrap(), loaded.model().unwrap());
        let mut bad = a.clone();
        bad.schema = if a.schema == MICRO_MODEL_SCHEMA {
            MICRO_RESIDUAL_MODEL_SCHEMA.into()
        } else {
            MICRO_MODEL_SCHEMA.into()
        };
        assert!(bad.model().is_err());
    }
}
