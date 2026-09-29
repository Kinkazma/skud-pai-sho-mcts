use paisho_ai::*;
use paisho_train::micro_learning::MicroArtifact;
#[test]
fn new_memory_identity_and_reload_preserve_all_parameters_and_next_update() {
    let old = MicroModel::seeded(7).with_deep_value(17);
    let mut m = old.with_neural_memory(9478);
    let ex = MicroExample { policy_support: false,
        action_values: vec![Some(1.), Some(-0.4)],
        value_weight: 1.,
        sequence_source: 0,
        state: vec![0.1; 417],
        actions: vec![[0.2; 32], [-0.3; 32]],
        policy: vec![0.9, 0.1],
        value: 0.7,
        policy_weight: 1.,
    };
    for _ in 0..4 {
        m.train_step(&ex, 0.01, 0.).unwrap();
    }
    let a = MicroArtifact::new(&m, 51, serde_json::json!({"test":true}));
    let b: MicroArtifact = serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
    assert_eq!(a.identity(), b.identity());
    let mut loaded = b.model().unwrap();
    assert_eq!(m.parameters(), loaded.parameters());
    m.train_step(&ex, 0.01, 0.).unwrap();
    loaded.train_step(&ex, 0.01, 0.).unwrap();
    assert_eq!(m, loaded);
    let mut wrong = a.clone();
    wrong.schema = MICRO_DEEP_VALUE_MODEL_SCHEMA.into();
    assert!(wrong.model().is_err());
    let before = a.identity();
    let mut changed = a;
    changed.parameters[MICRO_NEURAL_MEMORY_START] += 0.001;
    assert_ne!(before, changed.identity());
}
