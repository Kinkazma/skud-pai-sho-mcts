//! Bounded real-archive integration controls. Diagnostic weights never enter a campaign.
use super::*;
#[doc(hidden)]
pub fn run(config: &Path, out: &Path) -> Result<serde_json::Value> {
    let started = paisho_platform::training_time::now();
    fs::create_dir(out)?;
    let o: Options = serde_json::from_slice(&fs::read(config)?)?;
    o.validate()?;
    let artifact = MicroArtifact::load(&o.model)?;
    let model = artifact.model()?;
    let frozen = sha256(&fs::read(&o.model)?);
    let resume: serde_json::Value =
        serde_json::from_slice(&fs::read(o.resume_progress.as_ref().unwrap())?)?;
    let first = Arc::new(Snapshot {
        identity: artifact.identity(),
        artifact: Some(Arc::new(artifact.clone())),
        model: Arc::new(model.clone()),
        version: resume["version"].as_u64().unwrap(),
        path: o.model.clone(),
    });
    let mut guard = publication::Guard::open(
        o.publication_guard.as_ref().unwrap(),
        out,
        16.,
        first,
        &resume["publication_guard"],
    )?;
    guard.enable_v2(o.publication_validation.as_ref().unwrap())?;
    let mut p = protection::Protection::new(&model, guard.reference_examples())?;
    let mut changed = model.clone();
    let rows = guard.reference_examples();
    let batch = rows
        .iter()
        .filter(|e| e.value == 1.)
        .cloned()
        .collect::<Vec<_>>();
    p.observe(&batch);
    let refs = batch.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    for _ in 0..4 {
        p.train(&mut changed, &refs, 0.002, 0.)?;
    }
    p.consolidate(&mut changed)?;
    p.checkpoint(out)?;
    let mut restored = protection::Protection::new(&changed, rows.clone())?;
    restored.restore(&p.progress())?;
    let mut next = changed.clone();
    p.train(&mut changed, &refs, 0.002, 1e-5)?;
    restored.train(&mut next, &refs, 0.002, 1e-5)?;
    assert_eq!(changed.parameters(), next.parameters());
    let publication_cache = guard.verify_cached_measurements(&[&model, &changed])?;
    let mut finite_regression = serde_json::Value::Null;
    if o.output.join("model.json").exists() {
        let mut candidate = MicroArtifact::load(&o.output.join("model.json"))?.model()?;
        let mut control = protection::Protection::new(&model, rows.clone())?;
        control.observe(&batch);
        control.consolidate(&mut candidate)?;
        finite_regression = control.progress();
    }
    let boundary = protection::verify_boundary(&model, &rows)?;
    let old = guard.accepted();
    let mut attempts = vec![];
    for rate in [0.001, 0.0001, 0.00001] {
        let mut candidate = old.model.as_ref().clone();
        candidate
            .train_batch_inline(&refs, rate, 0.)
            .map_err(invalid)?;
        let a = Arc::new(MicroArtifact::new(
            &candidate,
            artifact.updates,
            serde_json::json!({"diagnostic_only":true,"rate":rate}),
        ));
        let s = Arc::new(Snapshot {
            identity: a.identity(),
            artifact: Some(a),
            version: old.version + 1,
            model: Arc::new(candidate),
            path: PathBuf::new(),
        });
        guard.consider(s, true)?;
        attempts.push(guard.progress());
        if guard.accepted().identity != old.identity {
            break;
        }
    }
    let source = o
        .recall_archive_sources
        .first()
        .ok_or_else(|| invalid("verification archive required"))?;
    let coverage = durable::audit_coverage(source, &model, &out.join("coverage-audit"))?;
    let archive_root = out.join("recall");
    let mut d = durable::Archive::open(&archive_root)?;
    d.add_read_only(source)?;
    d.policy_consolidation(true);
    d.proof_recall = true;
    d.enable_coverage(out, &serde_json::Value::Null, &model, &archive_root)?;
    d.seed_values(rows.clone());
    let mut rng = StableRng::new(20260912);
    let samples = d.rehearse_cached(1024, &mut rng, &model)?;
    assert_eq!(samples.len(), 1024);
    assert_eq!(d.last_winning_draws, 512);
    assert!(samples[..512]
        .iter()
        .all(|e| e.value == 1. && e.value_weight == 0. && e.policy_weight > 0.));
    let pending = samples[0].clone();
    d.defer(pending.clone(), true);
    d.checkpoint(out)?;
    let state = d.progress();
    let mut d2 = durable::Archive::open(&out.join("recall-restored"))?;
    d2.add_read_only(source)?;
    d2.policy_consolidation(true);
    d2.proof_recall = true;
    d2.restore(&state)?;
    d2.enable_coverage(out, &state, &model, &archive_root)?;
    d2.seed_values(rows);
    let next = d2.rehearse_cached(2, &mut rng, &model)?;
    assert_eq!(
        serde_json::to_value(resume_example::ResumeExample::from(next[0].as_ref()))?,
        serde_json::to_value(resume_example::ResumeExample::from(pending.as_ref()))?
    );
    assert_eq!(frozen, sha256(&fs::read(&o.model)?));
    let result = serde_json::json!({"boundary_control":boundary,"finite_regression_control":finite_regression,"publication_cache":publication_cache,"parameters":model.parameters().len(),"input_model_unchanged":true,"next_protected_update_exact_after_reload":true,"protection":p.progress(),"publication_attempts":attempts,"publication_changed":guard.accepted().identity!=old.identity,"coverage":coverage,"mixed_recall":state,"deferred_recall_restored":true,"seconds":paisho_platform::training_time::elapsed(started).as_secs_f64(),"production_writes":0,"new_games":0});
    fs::write(
        out.join("verification.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(result)
}
