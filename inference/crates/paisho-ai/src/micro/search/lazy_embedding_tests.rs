use super::*;
#[test]
fn lazy_policy_keeps_values_search_and_retention_exact() {
    let pool=rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    pool.install(|| {
        let text=include_str!("../../../tests/fixtures/micro-alias-0-a.psr");
        let record:paisho_core::GameRecord=text.parse().unwrap();
        let mut cases=vec![];let mut p=record.initial_position();
        for (i,&a) in record.actions().iter().enumerate() {
            if i%5==0 {cases.push(p.clone());}p.apply(a).unwrap();
        }
        let mut model=MicroModel::seeded(393).with_neural_memory(29);
        let p=&cases[0];let legal=legal_actions(p);
        let e=MicroExample { policy_support: false,state:model.state_features(p),actions:legal.iter().map(|&a|micro_action_features(p,a)).collect(),policy:vec![1./legal.len() as f64;legal.len()],value:0.7,policy_weight:1.,value_weight:1.,sequence_source:0,action_values:vec![]};
        model.train_step(&e,0.01,0.).unwrap();
        let model=Arc::new(model);let mut times=[0.;2];let mut searches=0;
        for (i,p) in cases.iter().enumerate() {
            let state=model.state_features(p);let full=model.embed(&state);
            let value=model.embed_value_for_search(&state);let complete=model.embed_policy_for_search(&state,&value);
            assert_eq!(full.value.to_bits(),value.value.to_bits());
            for a in legal_actions(p) {
                let f=micro_action_features(p,a);assert_eq!(MicroModel::logit(&full,&f).to_bits(),MicroModel::logit(&complete,&f).to_bits());
            }
            let mut reference=None;
            for eager in [true,false,false,true] {
                let mut s=MicroMctsSession::new(model.clone());s.cache.eager_embedding=eager;s.set_root_value_strength(16.).unwrap();
                s.set_limits(1024,if i%2==0 {512*1024}else{32*1024*1024});
                let opts=MicroSearchOptions {seed:998+i as u64,proof_search:true,forced_playout_strength:2.,dirichlet_fraction:0.25,..Default::default()};
                let t=Instant::now();let a=s.search_with_options(p,256,None,opts).unwrap();let b=s.search_with_options(p,64,None,opts).unwrap();times[usize::from(!eager)]+=t.elapsed().as_secs_f64();
                if let Some((ra,rb))=&reference {
                    successor_reuse_tests::equal(&a,ra);successor_reuse_tests::equal(&b,rb);
                    assert_eq!((a.retained_bytes,a.memory_reset,b.retained_bytes,b.memory_reset),(ra.retained_bytes,ra.memory_reset,rb.retained_bytes,rb.memory_reset));
                } else {reference=Some((a,b));}searches+=2;
            }
        }
        println!("lazy policy {} positions {searches} searches eager {}s lazy {}s",cases.len(),times[0],times[1]);
    });
}
