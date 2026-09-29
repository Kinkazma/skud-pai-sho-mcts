use super::*;
#[test]
fn compact_root_inputs_preserve_every_forward_activation_bit() {
    let model = MicroModel::seeded(29).with_neural_memory(53);
    let mut rng = StableRng::new(8712);
    let w = &model.parameters()[MICRO_NEURAL_MEMORY_START..];
    for count in [0,1,31,32,33,65,130,189,524,1024,1025] {
        let state: Vec<_> = (0..417).map(|i| if i%7==0 {-0.} else {rng.next_f64()-0.5}).collect();
        let actions: Vec<_> = (0..count).map(|_|std::array::from_fn(|_|rng.next_f64()-0.5)).collect();
        let memory = std::array::from_fn(|i|if i%3==0 {-0.} else {rng.next_f64()-0.5});
        let logits: Vec<_> = (0..count).map(|_|rng.next_f64()-0.5).collect();
        for length in [400,417] {
            let a=Cache::new(w,&state[..length],&actions,&memory,-0.3,&logits);
            let b=Cache::inference(w,&state[..length],&actions,&memory,-0.3,&logits);
            for (left,right) in [(&a.q,&b.q),(&a.u,&b.u),(&a.h,&b.h),(&a.erfc,&b.erfc),
                (&a.normalized,&b.normalized),(&a.r,&b.r),(&a.output,&b.output)] {
                assert_eq!(left.len(),right.len());
                assert!(left.iter().zip(right.iter()).all(|(x,y)|x.to_bits()==y.to_bits()),
                    "forward count={count} state={length}");
            }
            assert_eq!(a.norms.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.norms.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            assert_eq!(a.inverse.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.inverse.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
    }
}
fn example() -> MicroExample {
    let mut rng = StableRng::new(913);
    MicroExample { structured: Vec::new(), policy_support: false,
        action_values: vec![Some(0.9), None, Some(-0.8)],
        sequence_source: 0,
        value_weight: 1.,
        state: (0..417).map(|_| rng.next_f64() - 0.5).collect(),
        actions: (0..3)
            .map(|_| std::array::from_fn(|_| rng.next_f64() - 0.5))
            .collect(),
        policy: vec![0.8, 0.15, 0.05],
        value: -0.3,
        policy_weight: 1.,
    }
}
#[test]
fn root_prior_read_reuses_embedding_without_changing_value_or_policy_bits() {
    let mut trained=MicroModel::seeded(29).with_neural_memory(53);
    trained.train_step(&example(),0.01,0.).unwrap();
    let mut rng=StableRng::new(713);
    for model in [MicroModel::seeded(29),MicroModel::seeded(29).with_deep_value(53),
        MicroModel::seeded(29).with_neural_memory(53),trained] {
        for count in [1,3,32,65,189] {
            let state:Vec<_>=(0..417).map(|i| if i%11==0 {-0.} else {rng.next_f64()-0.5}).collect();
            let actions:Vec<_>=(0..count).map(|_|std::array::from_fn(|_|rng.next_f64()-0.5)).collect();
            for excluded in [0,17,u64::MAX] {
                let embedding=model.embed(&state);
                let base=micro_softmax(&MicroModel::logits(&embedding,&actions)).unwrap();
                let expected=model.memory_priors(&state,&actions,&base,excluded).unwrap();
                let compact=model.memory_priors_compact(&state,&actions,&base,excluded).unwrap();
                assert_eq!(compact.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
                let (value,actual)=model.policy_value_priors(&state,&actions,excluded).unwrap();
                assert_eq!(value.to_bits(),embedding.value.to_bits());
                assert_eq!(actual.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
            }
        }
    }
}
#[test]
fn reused_policy_mass_read_and_forward_only_loss_match_original_gradient_path() {
    let mut trained=MicroModel::seeded(29).with_neural_memory(53);
    let mut ex=example();
    trained.train_step(&ex,0.01,0.).unwrap();
    ex.policy=vec![0.8,0.,0.2];
    for model in [MicroModel::seeded(29),MicroModel::seeded(29).with_deep_value(53),MicroModel::seeded(29).with_neural_memory(53),trained] {
        let mut target=ex.clone();
        let base=micro_softmax(&MicroModel::logits(&model.embed(&ex.state),&ex.actions)).unwrap();
        let p=model.memory_priors(&ex.state,&ex.actions,&base,ex.sequence_source).unwrap();
        let mass:f64=p.iter().zip(&ex.policy).filter(|(_,t)|**t>0.).map(|(p,_)|p).sum();
        target.policy=p.iter().zip(&ex.policy).map(|(p,t)|if *t>0. {p/mass}else{0.}).collect();
        target.policy_weight=1.;target.value_weight=0.;
        let original=model.loss_gradient(&target).unwrap().1;
        let cached=model.policy_mass_gradient(&ex).unwrap();
        assert!(original.iter().zip(cached).all(|(a,b)|a.to_bits()==b.to_bits()));
        for (policy,value) in [(1.,1.),(0.,1.),(1.,0.),(0.,0.)] {
            let mut e=ex.clone();e.policy_weight=policy;e.value_weight=value;
            let a=model.loss_gradient(&e).unwrap().0;let b=model.loss(&e).unwrap();
            assert_eq!(a.value.to_bits(),b.value.to_bits());
            assert_eq!(a.policy.to_bits(),b.policy.to_bits());
        }
    }
}
#[test]
fn neutral_migration_keeps_old_value_policy_and_gradients_exact() {
    let old = MicroModel::seeded(13).with_deep_value(77);
    let mut w = old.parameters().to_vec();
    *w.last_mut().unwrap() = 0.4;
    let old = MicroModel::from_parameters(w).unwrap();
    let new = old.with_neural_memory(17);
    let mut ex = example();
    ex.action_values.clear();
    assert_eq!(new.parameters().len(), 292_363);
    assert_eq!(
        &new.parameters()[..MICRO_NEURAL_MEMORY_START],
        old.parameters()
    );
    assert_eq!(new.with_neural_memory(8), new);
    assert_eq!(new.schema(), MICRO_NEURAL_MEMORY_MODEL_SCHEMA);
    let (a, b) = (old.embed(&ex.state), new.embed(&ex.state));
    assert_eq!(a.value.to_bits(), b.value.to_bits());
    let p = micro_softmax(&MicroModel::logits(&a, &ex.actions)).unwrap();
    assert_eq!(
        old.memory_priors(&ex.state, &ex.actions, &p, 0).unwrap(),
        new.memory_priors(&ex.state, &ex.actions, &p, 0).unwrap()
    );
    let (lo, go) = old.loss_gradient(&ex).unwrap();
    let (ln, gn) = new.loss_gradient(&ex).unwrap();
    assert_eq!(lo.total(1.).to_bits(), ln.total(1.).to_bits());
    assert_eq!(go, &gn[..go.len()]);
    assert!(gn[MICRO_NEURAL_MEMORY_START + OUT..]
        .iter()
        .any(|v| *v != 0.));
}
#[test]
fn joint_gradients_include_current_value_and_policy_inputs_and_all_layers_learn() {
    let mut m = MicroModel::seeded(3).with_neural_memory(17);
    let initial = m.parameters().to_vec();
    let ex = example();
    let before = m.loss_gradient(&ex).unwrap().0.total(1.);
    for _ in 0..20 {
        m.train_step(&ex, 0.01, 0.).unwrap();
    }
    assert!(m.loss_gradient(&ex).unwrap().0.total(1.) < before);
    for (a, b) in [
        (W0, B0),
        (B0, W1),
        (W1, B1),
        (B1, W2),
        (W2, B2),
        (B2, GAMMA),
        (GAMMA, BETA),
        (BETA, OUT),
        (OUT, BOUT),
        (BOUT, BOUT + 2),
    ] {
        assert!(
            m.parameters()[MICRO_NEURAL_MEMORY_START + a..MICRO_NEURAL_MEMORY_START + b]
                .iter()
                .zip(&initial[MICRO_NEURAL_MEMORY_START + a..MICRO_NEURAL_MEMORY_START + b])
                .any(|(a, b)| a != b)
        );
    }
    let (_, g) = m.loss_gradient(&ex).unwrap();
    let mut worst: f64 = 0.;
    let points = [
        VALUE_W,
        VALUE_B,
        POLICY_W,
        POLICY_B,
        MICRO_VALUE_TRUNK,
        MICRO_DEEP_VALUE_START,
        MICRO_NEURAL_MEMORY_START + W0,
        MICRO_NEURAL_MEMORY_START + 449 * WIDTH + 3,
        MICRO_NEURAL_MEMORY_START + 481 * WIDTH + 20,
        MICRO_NEURAL_MEMORY_START + 482 * WIDTH + 77,
        MICRO_NEURAL_MEMORY_START + B0 + 8,
        MICRO_NEURAL_MEMORY_START + W1 + 34,
        MICRO_NEURAL_MEMORY_START + B1 + 5,
        MICRO_NEURAL_MEMORY_START + W2 + 7,
        MICRO_NEURAL_MEMORY_START + B2 + 50,
        MICRO_NEURAL_MEMORY_START + GAMMA + 80,
        MICRO_NEURAL_MEMORY_START + BETA + 45,
        MICRO_NEURAL_MEMORY_START + OUT + 20,
        MICRO_NEURAL_MEMORY_START + OUT + 21,
        MICRO_NEURAL_MEMORY_START + BOUT + 1,
    ];
    for i in points {
        let eps = 1e-5;
        let mut w = m.parameters().to_vec();
        w[i] += eps;
        let hi = MicroModel::from_parameters(w)
            .unwrap()
            .loss_gradient(&ex)
            .unwrap()
            .0
            .total(1.);
        let mut w = m.parameters().to_vec();
        w[i] -= eps;
        let lo = MicroModel::from_parameters(w)
            .unwrap()
            .loss_gradient(&ex)
            .unwrap()
            .0
            .total(1.);
        worst = worst.max(((hi - lo) / (2. * eps) - g[i]).abs());
    }
    assert!(worst < 1e-7, "joint derivative error {worst}");
    let old = m.clone();
    assert!(m.train_step(&ex, f64::NAN, 0.).is_err());
    assert_eq!(m, old);
    let mut restored = MicroModel::from_parameters(m.parameters().to_vec()).unwrap();
    m.train_step(&ex, 0.01, 0.).unwrap();
    restored.train_step(&ex, 0.01, 0.).unwrap();
    assert_eq!(m, restored);
}
#[test]
fn memory_side_gradients_and_action_permutations_are_consistent() {
    let mut m = MicroModel::seeded(3).with_neural_memory(5);
    let ex = example();
    for _ in 0..3 {
        m.train_step(&ex, 0.01, 0.).unwrap();
    }
    let mem = [0.3; 32];
    let logits = [0.2, -0.5, 0.8];
    let d = [0.7, -0.3, 0.2, 0.8, -0.9, 0.1];
    let c = m.neural_forward(&ex.state, &ex.actions, &mem, 0.4, &logits);
    let mut g = vec![0.; m.parameters().len()];
    let side = m.neural_backward(&c, &d, &mut g);
    let evaluate = |mem: &[f64; 32], value: f64, logits: &[f64]| {
        m.neural_forward(&ex.state, &ex.actions, mem, value, logits)
            .output
            .iter()
            .zip(d)
            .map(|(a, b)| a * b)
            .sum::<f64>()
    };
    let eps = 1e-5;
    let mut a = mem;
    a[4] += eps;
    let mut b = mem;
    b[4] -= eps;
    assert!(
        ((evaluate(&a, 0.4, &logits) - evaluate(&b, 0.4, &logits)) / (2. * eps) - side.memory[4])
            .abs()
            < 1e-7
    );
    assert!(
        ((evaluate(&mem, 0.4 + eps, &logits) - evaluate(&mem, 0.4 - eps, &logits)) / (2. * eps)
            - side.value)
            .abs()
            < 1e-7
    );
    let mut e = ex.clone();
    e.actions.reverse();
    e.policy.reverse();
    e.action_values.reverse();
    let (l, g) = m.loss_gradient(&ex).unwrap();
    let (lp, gp) = m.loss_gradient(&e).unwrap();
    assert!((l.total(1.) - lp.total(1.)).abs() < 1e-12);
    assert!(g.iter().zip(gp).all(|(a, b)| (a - b).abs() < 1e-12));
    let p = m
        .memory_priors(
            &ex.state,
            &ex.actions,
            &micro_softmax(&MicroModel::logits(&m.embed(&ex.state), &ex.actions)).unwrap(),
            0,
        )
        .unwrap();
    let ce = -p
        .iter()
        .zip(&ex.policy)
        .map(|(p, t)| p.ln() * t)
        .sum::<f64>();
    assert!((ce - l.policy).abs() < 1e-12);
}

#[test]
fn cloned_snapshots_share_weights_and_learning_preserves_the_old_snapshot() {
    let mut learner=MicroModel::seeded(13).with_neural_memory(17);
    let snapshot=learner.clone();
    assert!(std::sync::Arc::ptr_eq(&learner.parameters,&snapshot.parameters));
    let before=snapshot.parameters().iter().map(|x|x.to_bits()).collect::<Vec<_>>();
    learner.train_step(&example(),0.01,0.).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&learner.parameters,&snapshot.parameters));
    assert_eq!(snapshot.parameters().iter().map(|x|x.to_bits()).collect::<Vec<_>>(),before);
    assert!(learner.parameters().iter().zip(before).any(|(a,b)|a.to_bits()!=b));
}
