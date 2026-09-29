//! Verify a neutral migration on real recorded states; no search or training.
use paisho_ai::{MctsEvaluator, GEN3_VALUE_RESIDUAL_PARAMETERS};
use paisho_core::{GameRecord, Player, legal_actions};
use paisho_train::gen32::Artifact;
use serde_json::json;
use std::{fs,path::Path,sync::Arc};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=std::env::args().collect();
    if a.len()!=5 {return Err("usage: PARENT RESIDUAL HUMAN.json OUTPUT.json".into());}
    let before=Artifact::load(Path::new(&a[1]))?;
    let after=Artifact::load(Path::new(&a[2]))?;
    assert_eq!(before.generation,after.generation);
    assert_eq!(before.updates,after.updates);
    assert_eq!(before.compact.weights,after.compact.weights);
    assert_eq!(before.value128_extra,after.value128_extra);
    assert_eq!(before.policy.parameters,after.policy.parameters);
    assert_eq!(before.policy.sequence_memory,after.policy.sequence_memory);
    let old=before.model()?;let new=after.model()?;
    assert!(!new.value_residual.as_ref().unwrap().active());
    assert_eq!(new.value_residual.as_ref().unwrap().parameters().len(),GEN3_VALUE_RESIDUAL_PARAMETERS);
    if let (Some(x),Some(y))=(old.policy.sequence_memory(),new.policy.sequence_memory()) {assert!(Arc::ptr_eq(x,y));}
    let input:serde_json::Value=serde_json::from_slice(&fs::read(&a[3])?)?;
    let record:GameRecord=input["moves"].as_str().ok_or("missing PSR")?.parse()?;
    let mut p=record.initial_position();
    let mut values=0;let mut policies=0;
    for i in 0..=record.actions().len() {
        for player in [Player::Host,Player::Guest] {assert_eq!(old.value_at(&p,player),new.value_at(&p,player));values+=1;}
        if [5,9,13].contains(&i) {
            let actions=legal_actions(&p);
            assert_eq!(old.policy_bias(&p,&actions)?,new.policy_bias(&p,&actions)?);
            policies+=1;
        }
        if let Some(action)=record.actions().get(i) {p.apply(*action)?;}
    }
    fs::write(&a[4],serde_json::to_vec_pretty(&json!({"parent_updates":before.updates,"generation":after.generation,"schema":after.schema,"residual_parameters":GEN3_VALUE_RESIDUAL_PARAMETERS,"old_value_policy_weights_exact":true,"resident_bank_shared":old.policy.sequence_memory().is_some(),"identical_value_checks":values,"identical_policy_checks":policies,"searches":0,"training_steps":0,"strength_claim":false}))?)?;
    Ok(())
}
