use paisho_ai::*;
use paisho_train::micro_learning::*;
use sha2::{Digest,Sha256};
use std::sync::Arc;
#[test]
fn deep_checkpoint_preserves_bank_identity_and_continues_exactly_after_reload() {
    let dir=std::env::temp_dir().join(format!("deep-value-artifact-{}",std::process::id()));std::fs::create_dir_all(&dir).unwrap();
    let entry=SequenceEntry {key:[0.;64],patterns:[[0;32];4],source:1,game:0,decision:0,end_decision:1,outcome:0,phase:0};
    let bank=SequenceBank::build_spatial(vec![entry],vec![SequenceGeometry::from_state(&vec![0.;417]).unwrap()],1,0,1);
    let mut bytes=vec![];bank.write_to(&mut bytes).unwrap();let path=dir.join("memory.bin");std::fs::write(&path,&bytes).unwrap();
    let bank=load_sequence_memory(&SequenceMemorySpec {path:path.to_string_lossy().into(),sha256:format!("{:x}",Sha256::digest(&bytes))}).unwrap();
    let old=MicroModel::seeded(4).with_spatial_policy().with_sequence_memory(bank.clone());let mut model=old.with_deep_value(9473);
    assert!(Arc::ptr_eq(model.sequence_memory().unwrap(),&bank));
    let ex=MicroExample { policy_support: false, action_values: vec![], value_weight: 1.0, sequence_source:0,state:(0..417).map(|i|(i as f64*0.73).sin()*0.2).collect(),actions:vec![],policy:vec![],policy_weight:0.,value:0.8};
    model.train_step(&ex,0.01,0.).unwrap();
    let artifact=MicroArtifact::new(&model,19,serde_json::json!({"parent":"fixture","test":true}));
    let path=dir.join("model.json");artifact.save(&path).unwrap();assert!(artifact.save(&path).is_err());
    let restored=MicroArtifact::load(&path).unwrap();assert_eq!(restored.identity(),artifact.identity());assert_eq!(restored.updates,19);
    let mut next=restored.model().unwrap();assert!(Arc::ptr_eq(next.sequence_memory().unwrap(),&bank));
    assert_eq!(model.embed(&ex.state).value.to_bits(),next.embed(&ex.state).value.to_bits());
    model.train_step(&ex,0.01,0.).unwrap();next.train_step(&ex,0.01,0.).unwrap();assert_eq!(model.parameters(),next.parameters());
    assert_ne!(artifact.identity(),MicroArtifact::new(&model,20,serde_json::json!({})).identity());
    let mut bad=artifact.clone();bad.schema=MICRO_SPATIAL_MODEL_SCHEMA.into();assert!(bad.model().is_err());
    let mut bad=artifact.clone();bad.parameters.truncate(MICRO_SPATIAL_PARAMETERS);assert!(bad.model().is_err());
    let mut bad=artifact.clone();bad.sequence_memory.as_mut().unwrap().sha256="0".repeat(64);assert!(bad.model().is_err());
    assert_eq!(old.parameters().len(),MICRO_SPATIAL_PARAMETERS);
    std::fs::remove_dir_all(dir).unwrap();
}
