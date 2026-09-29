use super::*;

fn fixture() -> (MicroModel, MicroExample) {
    let mut rng = StableRng::new(1807);
    let ex = MicroExample { structured: Vec::new(),
        policy_support: false,
        state: (0..417).map(|_| rng.next_f64() - 0.5).collect(),
        actions: (0..3).map(|_| std::array::from_fn(|_| rng.next_f64() - 0.5)).collect(),
        policy: vec![0.8, 0.15, 0.05], value: -0.4,
        policy_weight: 1., value_weight: 1.,
        action_values: vec![Some(0.9), None, Some(-0.8)], sequence_source: 0,
    };
    let mut model = MicroModel::seeded(893).with_neural_memory(390);
    for _ in 0..8 {model.train_step(&ex, 0.01, 0.).unwrap();}
    (model, ex)
}
fn value_coordinate(i: usize) -> bool {
    (VALUE_W..=VALUE_B).contains(&i) || (MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START).contains(&i)
}
#[test]
fn detached_value_keeps_forward_and_routes_only_direct_value_supervision() {
    let (model, ex) = fixture();
    let context = model.value(&ex.state);
    let (joint, gj) = model.loss_gradient(&ex).unwrap();
    let (detached, gd) = model.loss_gradient_detached_value(&ex).unwrap();
    let (fixed, gf) = model.loss_gradient_with_value_context(&ex, context).unwrap();
    assert_eq!(joint.value.to_bits(), detached.value.to_bits());
    assert_eq!(joint.policy.to_bits(), detached.policy.to_bits());
    assert_eq!(fixed.total(1.).to_bits(), detached.total(1.).to_bits());
    assert!(gd.iter().zip(&gf).all(|(a,b)| a.to_bits()==b.to_bits()));
    let mut value = ex.clone();
    value.structured.clear();value.actions.clear(); value.policy.clear(); value.action_values.clear(); value.policy_weight=0.;
    let (_, gv) = model.loss_gradient(&value).unwrap();
    assert!(gd.iter().zip(&gv).enumerate().filter(|(i,_)|value_coordinate(*i)).all(|(_, (a,b))|a.to_bits()==b.to_bits()));
    assert!(gj.iter().zip(&gd).enumerate().any(|(i,(a,b))|value_coordinate(i) && a!=b));
    assert!(gj.iter().zip(&gd).enumerate().filter(|(i,_)|!value_coordinate(*i)).all(|(_, (a,b))|a.to_bits()==b.to_bits()));
    let mut policy = ex; policy.value_weight=0.;
    let (_, gp) = model.loss_gradient_detached_value(&policy).unwrap();
    assert!(gp.iter().enumerate().filter(|(i,_)|value_coordinate(*i)).all(|(_,g)|*g==0.));
    assert!(gp[MICRO_NEURAL_MEMORY_START..].iter().any(|g|*g!=0.));
    let (_, reused) = model.loss_gradient_detached_value_reusing(&policy, vec![f64::NAN;gp.len()+17]).unwrap();
    assert!(gp.iter().zip(reused).all(|(a,b)|a.to_bits()==b.to_bits()));
}
#[test]
fn fixed_context_gradient_is_a_finite_partial_derivative() {
    let (model, ex) = fixture();
    let context = model.value(&ex.state);
    let (_, gradient) = model.loss_gradient_with_value_context(&ex, context).unwrap();
    let n=MICRO_NEURAL_MEMORY_START;
    let points = [VALUE_W,VALUE_B,POLICY_W,POLICY_B,MICRO_VALUE_TRUNK,MICRO_DEEP_VALUE_START,
        n,n+449*130+3,n+481*130+20,n+482*130+77,n+62_790+8,
        n+63_000,n+130_000,n+198_900,n+199_290,n+199_291];
    let eps=1e-5;
    let mut worst:f64=0.;
    for i in points {
        let mut hi=model.parameters().to_vec();hi[i]+=eps;
        let mut lo=model.parameters().to_vec();lo[i]-=eps;
        let a=MicroModel::from_parameters(hi).unwrap().loss_with_value_context(&ex,context).unwrap().total(ex.policy_weight);
        let b=MicroModel::from_parameters(lo).unwrap().loss_with_value_context(&ex,context).unwrap().total(ex.policy_weight);
        worst=worst.max(((a-b)/(2.*eps)-gradient[i]).abs());
    }
    assert!(worst<1e-7,"partial derivative error {worst}");
    for invalid in [f64::NAN,f64::INFINITY,-1.0001,1.0001] {
        assert!(model.loss_gradient_with_value_context(&ex,invalid).is_err());
        assert!(model.loss_with_value_context(&ex,invalid).is_err());
    }
}
#[test]
fn experimental_all_action_normalization_does_not_amplify_remaining_targets() {
    let (model, mut ex) = fixture();ex.policy_weight=0.;
    let (old, go)=model.loss_gradient_detached_value(&ex).unwrap();
    let (new, gn)=model.loss_gradient_all_actions_auxiliary_reusing(&ex,Vec::new()).unwrap();
    let direct=0.5*(model.value(&ex.state)-ex.value).powi(2)*ex.value_weight;
    assert!(((new.value-direct)-(old.value-direct)*2./3.).abs()<1e-14);
    for (a,b) in go[MICRO_NEURAL_MEMORY_START..].iter().zip(&gn[MICRO_NEURAL_MEMORY_START..]) {
        assert!((a*2./3.-b).abs()<1e-13);
    }
    assert!(go.iter().zip(&gn).enumerate().filter(|(i,_)|value_coordinate(*i)).all(|(_, (a,b))|a.to_bits()==b.to_bits()));
    ex.action_values[1]=Some(0.5);
    let a=model.loss_gradient_detached_value(&ex).unwrap();
    let b=model.loss_gradient_all_actions_auxiliary_reusing(&ex,vec![f64::NAN;gn.len()]).unwrap();
    assert_eq!(a.0.value.to_bits(),b.0.value.to_bits());
    assert!(a.1.iter().zip(b.1).all(|(a,b)|a.to_bits()==b.to_bits()));
}

#[test]
fn v3_sparse_proven_q_retains_its_original_auxiliary_budget() {
    let (model, mut ex) = fixture();
    // One supplied proven action among 96 legal feature rows. This checks the
    // optimizer coefficient independently of the certificate parser's tests.
    ex.actions=(0..96).map(|i|ex.actions[i%3]).collect();
    ex.policy=vec![0.;96];ex.policy[7]=1.;
    ex.action_values=vec![None;96];ex.action_values[7]=Some(1.);
    let joint=model.loss_gradient(&ex).unwrap();
    let expected=model.loss_gradient_detached_value(&ex).unwrap();
    let actual=model.loss_gradient_loop_v3_reusing(&ex,vec![f64::NAN;expected.1.len()+17]).unwrap();
    let forward=model.loss_loop_v3(&ex).unwrap();
    assert_eq!(joint.0.value.to_bits(),actual.0.value.to_bits());
    assert_eq!(joint.0.policy.to_bits(),actual.0.policy.to_bits());
    assert_eq!(forward.value.to_bits(),actual.0.value.to_bits());
    assert_eq!(forward.policy.to_bits(),actual.0.policy.to_bits());
    assert!(expected.1.iter().zip(&actual.1).all(|(a,b)|a.to_bits()==b.to_bits()));
    assert!(joint.1.iter().zip(&actual.1).enumerate().filter(|(i,_)|!value_coordinate(*i)).all(|(_, (a,b))|a.to_bits()==b.to_bits()));
}

#[test]
fn singleton_support_matches_categorical_loss_and_every_gradient_bit() {
    let (model,mut ex)=fixture();
    ex.policy=vec![0.,1.,0.];
    let old=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).unwrap();
    let old_joint=model.loss_gradient(&ex).unwrap();
    ex.policy_support=true;
    let new=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).unwrap();
    let new_joint=model.loss_gradient(&ex).unwrap();
    for (a,b) in [(&old,&new),(&old_joint,&new_joint)] {
        assert_eq!(a.0.value.to_bits(),b.0.value.to_bits());
        assert_eq!(a.0.policy.to_bits(),b.0.policy.to_bits());
        assert!(a.1.iter().zip(&b.1).all(|(a,b)|a.to_bits()==b.to_bits()));
    }
    assert_eq!(model.loss_loop_v3(&ex).unwrap().policy.to_bits(),new.0.policy.to_bits());
}

#[test]
fn multi_action_support_gradient_matches_its_fixed_context_loss() {
    let (model,mut ex)=fixture();
    ex.policy=vec![0.2,0.8,0.];ex.policy_support=true;
    let context=model.value(&ex.state);
    let (_,gradient)=model.loss_gradient_with_value_context(&ex,context).unwrap();
    let eps=1e-5;
    let n=MICRO_NEURAL_MEMORY_START;
    for i in [VALUE_W,VALUE_B,POLICY_W,POLICY_B,MICRO_VALUE_TRUNK,MICRO_DEEP_VALUE_START,n,n+449*130+3,n+481*130+20,n+62_790+8,n+130_000,n+198_900,n+199_291] {
        let mut hi=model.parameters().to_vec();hi[i]+=eps;
        let mut lo=model.parameters().to_vec();lo[i]-=eps;
        let a=MicroModel::from_parameters(hi).unwrap().loss_with_value_context(&ex,context).unwrap().total(ex.policy_weight);
        let b=MicroModel::from_parameters(lo).unwrap().loss_with_value_context(&ex,context).unwrap().total(ex.policy_weight);
        assert!(((a-b)/(2.*eps)-gradient[i]).abs()<1e-7,"support partial derivative at {i}");
    }
    let first=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).unwrap();
    ex.policy=vec![0.5,0.5,0.];
    let same=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).unwrap();
    assert_eq!(first.0.policy.to_bits(),same.0.policy.to_bits());
    assert!(first.1.iter().zip(&same.1).all(|(a,b)|a.to_bits()==b.to_bits()));
}

#[test]
fn full_winning_support_has_no_gradient_toward_uniform_preferences() {
    let (model,mut ex)=fixture();ex.value_weight=0.;ex.action_values.clear();
    ex.policy=vec![1./3.;3];ex.policy_support=true;
    let (loss,g)=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).unwrap();
    assert_eq!(loss.policy,0.);assert!(g.iter().all(|g|*g==0.));
    ex.policy_support=false;
    assert!(model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).unwrap().1.iter().any(|g|g.abs()>1e-10));
    // Independent logits: support ascent improves every good/bad margin and
    // the winning mass, without prescribing a distribution within the support.
    let logits=vec![2.,-1.,0.5,-0.3];let p=micro_softmax(&logits).unwrap();
    let support=[true,false,true,false];
    let mass:f64=p.iter().zip(support).filter(|(_,s)|*s).map(|(p,_)|p).sum();
    let delta:Vec<_>=p.iter().zip(support).map(|(p,s)|p-if s {p/mass}else{0.}).collect();
    let after=micro_softmax(&logits.iter().zip(&delta).map(|(l,g)|l-0.1*g).collect::<Vec<_>>()).unwrap();
    let new_mass:f64=after.iter().zip(support).filter(|(_,s)|*s).map(|(p,_)|p).sum();
    assert!(new_mass>mass);
    for i in [0,2] {for j in [1,3] {assert!(delta[i]<delta[j]);}}
}
