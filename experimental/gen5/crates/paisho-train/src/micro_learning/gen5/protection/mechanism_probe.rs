//! Frozen production anchor + actual fresh examples, bounded consolidation A/B.
//! No actors, MCTS, production writes or privately learned weight promotion.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn bytes(&self) -> Result<Vec<u8>> {
        let b = fs::read(&self.path)?;
        if sha256(&b) != self.sha256 {
            return Err(invalid("repair probe input hash changed"));
        }
        Ok(b)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    config: Input,
    progress: Input,
    actor: Input,
    protection: Input,
    #[serde(default)]
    learner: Option<Input>,
    steps: Vec<usize>,
    repetitions: usize,
    threads: usize,
    max_seconds: u64,
}
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    let bytes = fs::read(plan_path)?;
    let plan: Plan = serde_json::from_slice(&bytes)?;
    if out.exists()
        || plan.steps.is_empty()
        || plan.steps.len() > 6
        || plan
            .steps
            .iter()
            .any(|&n| n > 256 || (n == 0 && plan.learner.is_none()))
        || !(1..=2).contains(&plan.repetitions)
        || !(1..=10).contains(&plan.threads)
        || !(30..=1800).contains(&plan.max_seconds)
    {
        return Err(invalid("invalid bounded repair plan"));
    }
    let o: Options = serde_json::from_slice(&plan.config.bytes()?)?;
    let progress: serde_json::Value = serde_json::from_slice(&plan.progress.bytes()?)?;
    let saved: serde_json::Value = serde_json::from_slice(&plan.protection.bytes()?)?;
    let artifact: MicroArtifact = serde_json::from_slice(&plan.actor.bytes()?)?;
    let started = Instant::now();
    let model = artifact.model()?;
    let load_seconds = started.elapsed().as_secs_f64();
    if saved["anchor"] != serde_json::to_value(model.parameters())? {
        return Err(invalid("probe actor/anchor mismatch"));
    }
    let anchor = Arc::new(Snapshot {
        model: Arc::new(model.clone()),
        artifact: Some(Arc::new(artifact.clone())),
        path: plan.actor.path.clone(),
        version: progress["publication_guard"]["accepted_version"]
            .as_u64()
            .unwrap(),
        identity: artifact.identity(),
    });
    if anchor.identity != progress["publication_guard"]["accepted_identity"] {
        return Err(invalid("probe actor identity mismatch"));
    }
    fs::create_dir(out)?;
    fs::write(out.join("plan.json"), bytes)?;
    let pool = cpu::build_pool(plan.threads, None)?.0;
    let mut old_guard = progress["publication_guard"].clone();
    old_guard["accepted_path"] = serde_json::json!(plan.actor.path);
    let make_guard = |path: &Path| -> Result<publication::Guard> {
        let mut g = publication::Guard::open(
            o.publication_guard.as_ref().unwrap(),
            path,
            o.value_policy_strength,
            anchor.clone(),
            &old_guard,
        )?;
        g.enable_v2(o.publication_validation.as_ref().unwrap())?;
        g.enable_v3()?;
        g.enable_parallel(&[pool.clone()]);
        g.enable_transfer()?;
        Ok(g)
    };
    let template = make_guard(&out.join("template"))?;
    let references = template.reference_examples();
    let validation = template.diagnostic_validation_examples()?;
    let restored = serde_json::json!({"checkpoint":{"path":plan.protection.path,"sha256":plan.protection.sha256}});
    let make_protection = || -> Result<Protection> {
        let mut p = Protection::new(&model, references.clone())?;
        p.enable_loop_v3();
        p.enable_parallel(&[pool.clone()]);
        p.enable_validation_value(validation.clone())?;
        p.restore(&restored)?;
        Ok(p)
    };
    let fresh: Vec<resume_example::ResumeExample> = serde_json::from_value(saved["fresh"].clone())?;
    let fresh = fresh
        .into_iter()
        .map(|e| e.example_with_trusted_q(true))
        .collect::<Result<Vec<_>>>()?;
    if fresh.len() != 64 {
        return Err(invalid(
            "probe expects actual complete 64-row fresh barrier",
        ));
    }
    let actual = plan
        .learner
        .as_ref()
        .map(|input| -> Result<MicroModel> {
            let a: MicroArtifact = serde_json::from_slice(&input.bytes()?)?;
            if a.schema != artifact.schema
                || a.feature_schema != artifact.feature_schema
                || serde_json::to_value(&a.sequence_memory)?
                    != serde_json::to_value(&artifact.sequence_memory)?
            {
                return Err(invalid("actual learner architecture/bank changed"));
            }
            weights(&model, a.parameters)
        })
        .transpose()?;
    let mut results = vec![];
    let active = Instant::now();
    for steps in plan.steps {
        let mut p = make_protection()?;
        let mut incoming = actual.clone().unwrap_or_else(|| model.clone());
        let train_start = Instant::now();
        for i in 0..steps {
            let batch = (0..8)
                .map(|j| fresh[(8 * i + j) % fresh.len()].clone())
                .collect::<Vec<_>>();
            p.train_shared(&mut incoming, &batch, o.rate, 1e-6)?;
        }
        let train_seconds = train_start.elapsed().as_secs_f64();
        for repeat in 0..plan.repetitions {
            for corrected in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                if active.elapsed().as_secs() >= plan.max_seconds {
                    return Err(invalid("bounded repair probe time exhausted"));
                }
                let path = out.join(format!(
                    "steps-{steps}-repeat-{repeat}-corrected-{corrected}"
                ));
                fs::create_dir(&path)?;
                let mut guard = make_guard(&path)?;
                guard.observe_candidate(&incoming)?;
                let mut p = make_protection()?;
                let mut candidate = incoming.clone();
                let t = Instant::now();
                if corrected {
                    p.consolidate_for_publication(&mut candidate)?;
                } else {
                    p.diagnostic_consolidate_two_choices(&mut candidate, true, true, |_| {})?;
                }
                let consolidation_seconds = t.elapsed().as_secs_f64();
                durable::write(&path.join("consolidation.json"), &p.last)?;
                guard.set_publication_fresh(p.publication_fresh_contract()?)?;
                let a = Arc::new(MicroArtifact::new(
                    &candidate,
                    if actual.is_some() {
                        progress["updates"].as_u64().unwrap() + steps as u64
                    } else {
                        artifact.updates + steps as u64
                    },
                    serde_json::json!({"diagnostic":"repair-probe"}),
                ));
                let snap = Arc::new(Snapshot {
                    identity: a.identity(),
                    artifact: Some(a),
                    model: Arc::new(candidate),
                    version: if actual.is_some() {
                        progress["version"].as_u64().unwrap() + steps as u64
                    } else {
                        anchor.version + steps as u64
                    },
                    path: path.join("candidate.json"),
                });
                let t = Instant::now();
                guard.consider(snap, true)?;
                let publication_seconds = t.elapsed().as_secs_f64();
                let published_fresh = fresh_loss_mode(&guard.accepted().model, &fresh, p.parallel.as_ref(), true)?;
                let result = serde_json::json!({"steps":steps,"repeat":repeat,"corrected":corrected,
                    "train_seconds":train_seconds,"consolidation_seconds":consolidation_seconds,
                    "publication_seconds":publication_seconds,"published_fresh":published_fresh,"consolidation":p.last,
                    "actor_changed":guard.accepted().model.parameters()!=model.parameters(),"guard":guard.progress()});
                println!("steps={steps} corrected={corrected} consolidated={} actor_changed={} seconds={:.3}",
                    result["consolidation"]["accepted"],result["actor_changed"],consolidation_seconds+publication_seconds);
                durable::write(&path.join("result.json"), &result)?;
                results.push(result);
            }
        }
    }
    let report = serde_json::json!({"diagnostic_only":true,"load_seconds":load_seconds,
        "active_seconds":active.elapsed().as_secs_f64(),"fresh_examples":64,"results":results});
    durable::write(&out.join("report.json"), &report)?;
    Ok(report)
}
