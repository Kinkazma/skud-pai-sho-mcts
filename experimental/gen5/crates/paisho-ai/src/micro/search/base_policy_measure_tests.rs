use super::*;
#[test]
fn diagnostic_base_logits_match_actual_internal_policy_bits() {
    let record:paisho_core::GameRecord=include_str!("../../../tests/fixtures/micro-alias-0-a.psr").parse().unwrap();
    let mut position=record.initial_position();let mut cases=vec![];
    for (i,&a) in record.actions().iter().enumerate() {
        if i%7==0 {cases.push(position.clone());}position.apply(a).unwrap();
    }
    let mut neural=MicroModel::seeded(619).with_neural_memory(77);
    // Make the root neural path non-neutral, without using it in the test read.
    let p=&cases[0];let actions=legal_actions(p);let mut target=vec![0.;actions.len()];target[0]=1.;
    let ex=MicroExample{ structured: Vec::new(),policy_support:false,action_values:vec![],sequence_source:0,value_weight:0.,
        state:neural.state_features(p),actions:actions.iter().map(|&a|micro_action_features(p,a)).collect(),
        policy:target,value:0.,policy_weight:1.};
    neural.train_step(&ex,0.01,0.).unwrap();
    for model in [MicroModel::seeded(17),neural] {
        for p in &cases {
            let mut session=MicroMctsSession::new(Arc::new(model.clone()));
            let node=session.cache.get(p.clone(),&model);
            let internal=node.policy(&model).unwrap();
            let logits=model.diagnostic_interior_policy_logits(&node.state,&internal.features);
            let priors=micro_softmax(&logits).unwrap();
            assert_eq!(priors.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),internal.priors.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            let embedding=model.embed(&node.state);
            assert_eq!(logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                MicroModel::logits(&embedding,&internal.features).iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
    }
}
