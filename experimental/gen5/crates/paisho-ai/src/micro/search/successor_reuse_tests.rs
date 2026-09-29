use super::*;
fn bits(x:&[f64])->Vec<u64> {x.iter().map(|v|v.to_bits()).collect()}
pub(super) fn equal(a:&MicroSearchReport,b:&MicroSearchReport) {
 assert_eq!(a.selected_index,b.selected_index);assert_eq!(a.actions,b.actions);
 for (x,y) in [(&a.policy_target,&b.policy_target),(&a.priors,&b.priors),(&a.search_priors,&b.search_priors),(&a.values,&b.values),(&a.state,&b.state)] {assert_eq!(bits(x),bits(y));}
 assert_eq!(a.raw_priors.as_ref().map(|p|bits(p)),b.raw_priors.as_ref().map(|p|bits(p)));
 assert_eq!(a.visits,b.visits);assert_eq!(a.new_visits,b.new_visits);assert_eq!(a.new_forced_visits,b.new_forced_visits);assert_eq!(a.pruned_visits,b.pruned_visits);
 assert_eq!(a.proven_value,b.proven_value);assert_eq!(a.proven_action_values,b.proven_action_values);
 assert_eq!(a.network_value.to_bits(),b.network_value.to_bits());assert_eq!(a.simulations,b.simulations);assert_eq!(a.tactical_evaluations,b.tactical_evaluations);
 assert_eq!(a.inherited_visits,b.inherited_visits);assert_eq!(a.inference_cache_hits,b.inference_cache_hits);assert_eq!(a.inference_evaluations,b.inference_evaluations);
 assert_eq!(a.action_features.len(),b.action_features.len());for (x,y) in a.action_features.iter().zip(b.action_features.iter()) {assert_eq!(bits(x),bits(y));}
}
#[test]
fn guard_successors_keep_coupled_search_and_cache_admission_exact() {
 let pool=rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
 pool.install(|| {
  let mut cases=vec![];
  for text in [include_str!("../../../tests/fixtures/micro-alias-0-a.psr"),include_str!("../../../tests/fixtures/micro-alias-1-a.psr"),include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr")] {
   let r:paisho_core::GameRecord=text.parse().unwrap();let (r,_)=r.replay_prefix_with_rules(paisho_core::RuleProfileId::SkudPaiShoGen5V1).unwrap();let mut p=r.initial_position();
   for (i,&a) in r.actions().iter().enumerate() {if i%8==0 && p.outcome()==GameOutcome::Ongoing {cases.push(p.clone());}p.apply(a).unwrap();}
   if p.outcome()==GameOutcome::Ongoing {cases.push(p);}
  }
  let mut model=MicroModel::seeded(91).with_neural_memory(193);
  let p=&cases[0];let actions=legal_actions(p);
  let e=MicroExample { structured: Vec::new(), policy_support: false,state:model.state_features(p),actions:actions.iter().map(|&a|micro_action_features(p,a)).collect(),policy:vec![1./actions.len() as f64;actions.len()],value:0.7,policy_weight:1.,value_weight:1.,action_values:vec![],sequence_source:0};model.train_step(&e,0.01,0.).unwrap();let model=Arc::new(model);
  let mut time=[0.;2];let mut checked=0;
  for (i,p) in cases.iter().enumerate() {
   let mut reference=None;
   for old in [true,false,false,true] {
    let mut s=MicroMctsSession::new(model.clone());s.set_root_value_strength(16.).unwrap();s.recompute_root_successors=old;
    let t=Instant::now();let opts=MicroSearchOptions {seed:151+i as u64,proof_search:true,forced_playout_strength:2.,dirichlet_fraction:0.25,..Default::default()};
    let a=s.search_with_options(p,256,None,opts).unwrap();let b=s.search_with_options(p,64,None,opts).unwrap();time[usize::from(!old)]+=t.elapsed().as_secs_f64();
    if let Some((ra,rb))=&reference {equal(&a,ra);equal(&b,rb);} else {reference=Some((a,b));}checked+=2;
   }
  }
  assert!(cases.len()>=10);println!("guard reuse: {} positions, {checked} searches, original {}s reused {}s",cases.len(),time[0],time[1]);
 });
}
