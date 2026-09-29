use super::*;
fn example()->MicroExample {
    let mut rng=StableRng::new(3381);
    MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], value_weight: 1.0, sequence_source:0,state:(0..417).map(|_|rng.next_f64()*0.8-0.4).collect(),
        actions:vec![],policy:vec![],policy_weight:0.,value:0.7}
}
fn active_model()->MicroModel {
    let mut m=MicroModel::seeded(57).with_deep_value(9473);
    for (j,x) in std::sync::Arc::make_mut(&mut m.parameters)[OUT..].iter_mut().enumerate() {*x=(j as f64*0.711).sin()*0.15;}
    MicroModel::from_parameters(m.parameters.as_ref().clone()).unwrap()
}
#[test]
fn neutral_upgrade_preserves_old_parameters_predictions_and_idempotence() {
    let old=MicroModel::seeded(43).with_spatial_policy();let new=old.with_deep_value(9473);
    assert_eq!(new.parameters.len(),93071);assert!(new.has_spatial());assert!(new.has_deep_value());
    assert_eq!(new.schema(),MICRO_DEEP_VALUE_MODEL_SCHEMA);
    assert_eq!(new.feature_schema(),old.feature_schema());
    assert_eq!(&new.parameters[..MICRO_DEEP_VALUE_START],old.parameters());
    assert_eq!(new,new.with_deep_value(12));assert_eq!(new,new.with_spatial_policy());
    let mut ex=example();
    for i in 0..16 {
        ex.state[128..].rotate_left(i);
        assert_eq!(old.embed(&ex.state).value.to_bits(),new.embed(&ex.state).value.to_bits());
        let (a,ga)=old.loss_gradient(&ex).unwrap();let (b,gb)=new.loss_gradient(&ex).unwrap();
        assert_eq!(a.value.to_bits(),b.value.to_bits());
        assert!(ga.iter().zip(&gb).all(|(a,b)|a.to_bits()==b.to_bits()));
    }
}
#[test]
fn all_deep_derivatives_and_existing_value_parameters_match_finite_differences() {
    let mut model=active_model();let ex=example();let (_,g)=model.loss_gradient(&ex).unwrap();
    let mut max_error=0.0_f64;
    for i in (W1..MICRO_DEEP_VALUE_PARAMETERS).chain([VALUE_W,VALUE_B,MICRO_VALUE_TRUNK,MICRO_VALUE_BOARD]) {
        let old=model.parameters[i];let mut values=[0.;2];
        for (k,s) in [-1.,1.].into_iter().enumerate() {
            std::sync::Arc::make_mut(&mut model.parameters)[i]=old+s*1e-5;
            model.deep_value_weights=DeepValueWeights::from_parameters(&model.parameters).map(std::sync::Arc::new);
            values[k]=0.5*(model.embed(&ex.state).value-ex.value).powi(2);
        }
        std::sync::Arc::make_mut(&mut model.parameters)[i]=old;
        model.deep_value_weights=DeepValueWeights::from_parameters(&model.parameters).map(std::sync::Arc::new);
        let error=((values[1]-values[0])/2e-5-g[i]).abs();max_error=max_error.max(error);
        assert!(error<1e-7+1e-5*g[i].abs(),"gradient {i}: error {error}");
    }
    eprintln!("63873 deep derivatives + 4 existing value coordinates; max error {max_error}");
}
#[test]
fn all_layers_learn_after_neutral_start_and_policy_only_keeps_value_exact() {
    let mut model=MicroModel::seeded(57).with_deep_value(9473);let original=model.clone();let ex=example();
    let (_,g)=model.loss_gradient(&ex).unwrap();
    assert!(g[W1..OUT].iter().all(|x|*x==0.));assert!(g[OUT..].iter().any(|x|x.abs()>1e-10));
    for _ in 0..8 {model.train_step(&ex,0.02,0.).unwrap();}
    for (start,end) in [(W1,B1),(W2,B2),(W3,B3),(OUT,BOUT)] {
        assert!(model.parameters[start..end].iter().zip(&original.parameters[start..end]).any(|(a,b)|a!=b),"inactive layer {start}");
    }
    assert!(model.deep_value_active);
    let mut policy=ex.clone();policy.actions=vec![[0.;32],[0.5;32]];policy.policy=vec![1.,0.];policy.policy_weight=1.;
    let before=model.clone();let v=model.embed(&ex.state).value;
    model.train_policy_step(&policy,0.02).unwrap();
    assert_eq!(&model.parameters[W1..],&before.parameters[W1..]);
    assert_eq!(model.embed(&ex.state).value.to_bits(),v.to_bits());
    assert_eq!(original,MicroModel::seeded(57).with_deep_value(9473));
}
#[test]
fn deep_batches_are_ordered_and_invalid_updates_atomic() {
    let mut a=active_model();let mut b=a.clone();let ex=example();let mut other=ex.clone();other.value=-0.4;
    let pool=rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
    for batch in [vec![&ex,&other],vec![&other,&ex,&other]] {
        a.train_batch_inline(&batch,0.003,1e-5).unwrap();
        pool.install(||b.train_batch(&batch,0.003,1e-5)).unwrap();
        assert_eq!(a.parameters,b.parameters);
    }
    let before=a.clone();let mut invalid=ex.clone();invalid.value=f64::NAN;
    assert!(a.train_batch_inline(&[&ex,&invalid],0.003,1e-5).is_err());assert_eq!(a,before);
    let mut w=a.parameters.as_ref().clone();w.pop();assert!(MicroModel::from_parameters(w).is_err());
    let mut w=a.parameters.as_ref().clone();w[W1]=f64::INFINITY;assert!(MicroModel::from_parameters(w).is_err());
}

#[test]
fn grouped_weight_views_match_original_scalar_layers_including_signed_zero() {
    let model=active_model();let w=&model.parameters;
    let mut state=example().state;
    for kind in 0..40 {
        if kind==0 {state.fill(0.0);}
        else if kind==1 {state.fill(-0.0);}
        else {
            for (i,x) in state.iter_mut().enumerate() {
                *x=if (i+kind)%3==0 {0.0} else {((i*131+kind*17) as f64).sin()*0.8};
            }
        }
        let occupied=OccupiedFeatures::new(&state);
        let mut a=DeepValueActivations::default();let mut b=DeepValueActivations::default();
        let actual=model.deep_value_forward(&state,&occupied,&mut a);
        for (j,h) in b.first.iter_mut().enumerate() {
            let start=W1+j*417;
            let dense=w[B1+j]+state[..128].iter().zip(&w[start..start+128]).map(|(x,w)|x*w).sum::<f64>();
            *h=(dense+occupied.dot(&state,&w[start+128..start+417])).tanh();
        }
        for (j,h) in b.second.iter_mut().enumerate() {
            *h=(w[B2+j]+b.first.iter().zip(&w[W2+j*128..W2+(j+1)*128]).map(|(x,w)|x*w).sum::<f64>()).tanh();
        }
        for (j,h) in b.third.iter_mut().enumerate() {
            *h=(w[B3+j]+b.second.iter().zip(&w[W3+j*64..W3+(j+1)*64]).map(|(x,w)|x*w).sum::<f64>()).tanh();
        }
        let expected=w[BOUT]+b.third.iter().zip(&w[OUT..BOUT]).map(|(x,w)|x*w).sum::<f64>();
        for (x,y) in a.first.iter().chain(&a.second).chain(&a.third).zip(b.first.iter().chain(&b.second).chain(&b.third)) {
            assert_eq!(x.to_bits(),y.to_bits(),"kind {kind}");
        }
        assert_eq!(actual.to_bits(),expected.to_bits());
    }
    let clone=model.clone();
    assert!(std::sync::Arc::ptr_eq(model.deep_value_weights.as_ref().unwrap(),clone.deep_value_weights.as_ref().unwrap()));
}

#[test]
fn sparse_dots_preserve_cancellation_subnormals_and_signed_zero() {
    let values=[0.0,-0.0,1.0,-1.0,f64::from_bits(1),-f64::from_bits(1),f64::MAX,-f64::MAX];
    for shift in 0..values.len() {
        let mut x=[0.0;128];
        for (i,v) in x.iter_mut().enumerate() { *v=values[(i+shift)%values.len()]; }
        let weights:Vec<[f64;4]>=(0..128).map(|i|std::array::from_fn(|lane|values[(i*3+lane)%values.len()])).collect();
        let indices:Vec<u8>=x.iter().enumerate().filter(|(_,x)|**x!=0.0).map(|(i,_)|i as u8).collect();
        let got=sparse_dots(&x,&indices,&weights);
        for lane in 0..4 {
            let expected=x.iter().zip(&weights).map(|(x,w)|x*w[lane]).sum::<f64>();
            assert_eq!(got[lane].to_bits(),expected.to_bits());
        }
    }
    for zero in [0.0,-0.0] {
        let x=[zero;128];let weights=vec![[-1.0,1.0,-0.0,0.0];128];
        let got=sparse_dots(&x,&[],&weights);
        for lane in 0..4 { assert_eq!(got[lane].to_bits(),x.iter().zip(&weights).map(|(x,w)|x*w[lane]).sum::<f64>().to_bits()); }
    }
}
