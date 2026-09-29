use super::*;
/// Isolated acceptance/rejection exercise; the source model is never changed.
#[doc(hidden)]
pub fn run(model_path: &Path, manifest: &Path, out: &Path) -> Result<serde_json::Value> {
    fs::create_dir(out)?;
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(model_path)?)?;
    let model = artifact.model()?;
    let make = |model: MicroModel, version: u64| {
        let a = Arc::new(MicroArtifact::new(
            &model,
            artifact.updates,
            serde_json::json!({"diagnostic_version":version}),
        ));
        Arc::new(Snapshot {
            identity: a.identity(),
            artifact: Some(a),
            version,
            model: Arc::new(model),
            path: "unused".into(),
        })
    };
    let first = make(model.clone(), 0);
    let mut guard = Guard::open(manifest, out, 16., first.clone(), &serde_json::Value::Null)?;
    let neutral = make(model.clone(), 1);
    assert!(guard.consider(neutral.clone(), true)?.unwrap().is_empty());
    assert_eq!(guard.accepted().identity, neutral.identity);
    let baseline = guard.score.clone();
    let mut gradient = vec![0.; model.parameters().len()];
    let mut classes = [0usize; 3];
    for r in &guard.rows {
        classes[(r.example.value as i8 + 1) as usize] += 1;
    }
    for r in &guard.rows {
        let mut ex = r.example.as_ref().clone();
        ex.value_weight = 1.;
        ex.policy_weight = 0.;
        let (_, g) = model.loss_gradient(&ex).map_err(invalid)?;
        let n = classes[(ex.value as i8 + 1) as usize] as f64;
        for (sum, g) in gradient.iter_mut().zip(g) {
            *sum += g / n;
        }
    }
    let mut bad = MicroModel::from_parameters(
        model
            .parameters()
            .iter()
            .zip(&gradient)
            .map(|(w, g)| w + 0.01 * g)
            .collect(),
    )
    .map_err(invalid)?;
    if let Some(b) = model.sequence_memory() {
        bad = bad.with_sequence_memory(b.clone());
    }
    let rejected = guard.consider(make(bad, 2), true)?.unwrap();
    assert!(!guard.state.reasons.is_empty());
    assert_eq!(guard.accepted().identity, neutral.identity);
    assert!(rejected.len() <= 32);
    assert!(rejected.iter().all(|e| e.value_weight == 0.));
    let examples: Vec<_> = guard
        .rows
        .iter()
        .filter(|r| r.example.value == 1.)
        .map(|r| r.example.clone())
        .collect();
    let refs: Vec<_> = examples.iter().map(AsRef::as_ref).collect();
    let mut accepted_rate = None;
    for k in 0..16 {
        let rate = 0.001 / (1u64 << k) as f64;
        let mut improved = model.clone();
        improved
            .train_batch_inline(&refs, rate, 0.)
            .map_err(invalid)?;
        let result = guard.consider(make(improved, 3 + k), true)?.unwrap();
        if result.is_empty() && guard.state.reasons.is_empty() {
            accepted_rate = Some(rate);
            break;
        }
    }
    if accepted_rate.is_none() {
        return Err(invalid(
            "policy-only descent could not pass the fixed guard",
        ));
    }
    guard.committed()?;
    let restored = Guard::open(manifest, out, 16., first, &guard.progress())?;
    assert_eq!(
        restored.accepted().model.parameters(),
        guard.accepted().model.parameters()
    );
    Ok(
        serde_json::json!({"neutral_accepted":true,"value_regression_rejected":true,"focus_policy_only":true,"policy_step_accepted_rate":accepted_rate,"before":{"raw_wins":baseline.raw.iter().filter(|b|**b).count(),"coupled_wins":baseline.coupled.iter().filter(|b|**b).count(),"mass":baseline.mass,"value_mse":baseline.value_mse},"after":{"raw_wins":guard.score.raw.iter().filter(|b|**b).count(),"coupled_wins":guard.score.coupled.iter().filter(|b|**b).count(),"mass":guard.score.mass,"value_mse":guard.score.value_mse},"restored_exact":true,"state":guard.progress()}),
    )
}
#[doc(hidden)]
pub fn candidate(
    initial_path: &Path,
    candidate_path: &Path,
    manifest: &Path,
    out: &Path,
) -> Result<serde_json::Value> {
    fs::create_dir(out)?;
    let first: MicroArtifact = serde_json::from_slice(&fs::read(initial_path)?)?;
    let model = first.model()?;
    let initial = Arc::new(Snapshot {
        identity: first.identity(),
        artifact: Some(Arc::new(first)),
        version: 0,
        model: Arc::new(model.clone()),
        path: initial_path.into(),
    });
    let candidate: MicroArtifact = serde_json::from_slice(&fs::read(candidate_path)?)?;
    let snapshot = load_snapshot(candidate_path, &candidate.identity(), 1, &model)?;
    let mut guard = Guard::open(manifest, out, 16., initial, &serde_json::Value::Null)?;
    let before = serde_json::to_value(&guard.score)?;
    let full = serde_json::to_value(measure(&guard.rows, &snapshot.model, 16.)?)?;
    let focus = guard.consider(snapshot, true)?.unwrap();
    let result = serde_json::json!({"before":before,"full_learner":full,"accepted":guard.score,"state":guard.progress(),"focus":focus.len()});
    fs::write(
        out.join("verification.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}
