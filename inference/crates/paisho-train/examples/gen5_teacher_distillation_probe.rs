//! Offline proposal test: distill complete verified search into frozen-clone policies.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, sync::Arc, time::Instant};
struct Row {
    ex: MicroExample,
    valid: Vec<bool>,
    source: String,
    cohort: String,
    train: bool,
    prefix: String,
}
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn probabilities(m: &MicroModel, x: &MicroExample) -> Vec<f64> {
    m.memory_priors(
        &x.state,
        &x.actions,
        &micro_softmax(&MicroModel::logits(&m.embed(&x.state), &x.actions)).unwrap(),
        0,
    )
    .unwrap()
}
fn score(m: &MicroModel, rows: &[Row], cohort: &str, train: bool, base: &MicroModel) -> Value {
    let mut n = 0;
    let mut top = 0;
    let mut mass = 0.;
    let mut ce = 0.;
    let mut lost = 0;
    let mut gained = 0;
    for r in rows
        .iter()
        .filter(|r| r.cohort == cohort && r.train == train)
    {
        let p = probabilities(m, &r.ex);
        let b = probabilities(base, &r.ex);
        let best = |v: &[f64]| {
            (0..v.len())
                .max_by(|a, b| v[*a].total_cmp(&v[*b]).then_with(|| b.cmp(a)))
                .unwrap()
        };
        let yes = r.valid[best(&p)];
        let before = r.valid[best(&b)];
        top += usize::from(yes);
        lost += usize::from(before && !yes);
        gained += usize::from(!before && yes);
        n += 1;
        mass += p
            .iter()
            .zip(&r.valid)
            .filter(|(_, t)| **t)
            .map(|(p, _)| p)
            .sum::<f64>();
        ce -= p
            .iter()
            .zip(&r.ex.policy)
            .filter(|(_, t)| **t > 0.)
            .map(|(p, t)| t * p.max(1e-300).ln())
            .sum::<f64>();
        assert_eq!(
            m.embed(&r.ex.state).value.to_bits(),
            base.embed(&r.ex.state).value.to_bits()
        );
    }
    json!({"n":n,"certified_top":top,"mass":mass/n as f64,"ce":ce/n as f64,"lost":lost,"gained":gained})
}
fn row(
    model: &MicroModel,
    record: &GameRecord,
    source: String,
    cohort: &str,
    policy: Vec<f64>,
) -> Row {
    let p = record.replay().unwrap();
    let actions = legal_actions(&p);
    let train = Sha256::digest(source.as_bytes())[0] % 3 != 0;
    let ex = MicroExample { policy_support: false, action_values: vec![], 
        state: model.state_features(&p),
        actions: actions
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect(),
        policy,
        value: 1.,
        policy_weight: 1.,
        value_weight: 1.0, sequence_source: 0,
    };
    ex.validate().unwrap();
    let valid = ex.policy.iter().map(|p| *p > 0.).collect();
    Row {
        ex,
        valid,
        source,
        cohort: cohort.into(),
        train,
        prefix: hash(record.to_string().as_bytes()),
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let artifact: MicroArtifact =
        serde_json::from_slice(&fs::read(manifest["model"].as_str().unwrap())?)?;
    let base = Arc::new(artifact.model()?);
    let mut rows = vec![];
    let mut searches = vec![];
    let mut used_sources = HashSet::new();
    for cohort in ["search", "opponent"] {
        let specs = manifest[if cohort == "search" {
            "teacher_roots"
        } else {
            "opponents"
        }]
        .as_array()
        .unwrap();
        for spec in specs {
            let source = spec["source"].as_str().unwrap().to_string();
            used_sources.insert(source.clone());
            let record: GameRecord = if cohort == "search" {
                spec["prefix"].as_str().unwrap().parse()?
            } else {
                let b = fs::read(spec["path"].as_str().unwrap())?;
                assert_eq!(hash(&b), spec["sha256"]);
                let full: GameRecord = std::str::from_utf8(&b)?.parse()?;
                full.replay()?;
                let mut prefix = GameRecord::with_rules(full.setup(), full.rules());
                for a in full
                    .actions()
                    .iter()
                    .take(spec["prefix_decisions"].as_u64().unwrap() as usize)
                {
                    prefix.push(*a);
                }
                prefix
            };
            let p = record.replay()?;
            let mut session = MicroMctsSession::new(base.clone());
            let options = MicroSearchOptions {
                proof_search: true,
                ..Default::default()
            };
            let mut r = session.search_with_options(&p, 512, None, options)?;
            let mut imported = false;
            if r.proven_value != Some(1) {
                assert_eq!(cohort, "search");
                let b = fs::read(spec["spec"]["path"].as_str().unwrap())?;
                assert_eq!(hash(&b), spec["spec"]["sha256"]);
                let data: Value = serde_json::from_slice(&b)?;
                let c: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
                assert_eq!(c.verify(&p)?, GameOutcome::Win(p.to_move()));
                session.install_certificate(&p, &c)?;
                r = session.search_with_options(&p, 512, None, options)?;
                imported = true;
            }
            assert_eq!(r.proven_value, Some(1));
            let cert = session.certificate(10000).ok_or("certificate missing")?;
            assert_eq!(cert.verify(&p)?, GameOutcome::Win(p.to_move()));
            let sign = if p.to_move() == Player::Host { 1 } else { -1 };
            let mut valid: HashSet<Action> = cert
                .children
                .iter()
                .filter(|(_, c)| c.outcome == sign)
                .map(|(a, _)| a.parse().unwrap())
                .collect();
            assert!(r
                .policy_target
                .iter()
                .zip(&r.actions)
                .all(|(t, a)| *t == 0. || valid.contains(a)));
            if cohort == "search" {
                let b = fs::read(spec["spec"]["path"].as_str().unwrap())?;
                assert_eq!(hash(&b), spec["spec"]["sha256"]);
                let data: Value = serde_json::from_slice(&b)?;
                let c: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
                assert_eq!(c.verify(&p)?, GameOutcome::Win(p.to_move()));
                for (a, c) in c.children {
                    if c.outcome == sign {
                        valid.insert(a.parse()?);
                    }
                }
            }
            let valid_mask: Vec<_> = r.actions.iter().map(|a| valid.contains(a)).collect();
            searches.push(json!({"cohort":cohort,"source":source,"prefix":record.to_string(),"certificate":cert,"target":r.policy_target,"valid":valid_mask,"actions":r.actions.iter().map(ToString::to_string).collect::<Vec<_>>(),"imported":imported,"simulations":r.simulations}));
            let mut teacher = row(&base, &record, source, cohort, r.policy_target);
            teacher.valid = valid_mask;
            rows.push(teacher);
        }
    }
    let panel: Value = serde_json::from_slice(&fs::read(manifest["old_panel"].as_str().unwrap())?)?;
    for spec in panel["positions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|r| r["kind"] == "certificate")
    {
        let b = fs::read(spec["path"].as_str().unwrap())?;
        assert_eq!(hash(&b), spec["sha256"]);
        let data: Value = serde_json::from_slice(&b)?;
        let source = data["human_source"].as_str().unwrap().to_string();
        if used_sources.contains(&source) {
            continue;
        }
        let record: GameRecord = data["prefix"].as_str().unwrap().parse()?;
        let p = record.replay()?;
        let c: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
        if c.verify(&p)? != GameOutcome::Win(p.to_move()) {
            continue;
        }
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let legal = legal_actions(&p);
        let mut policy = vec![0.; legal.len()];
        for (a, c) in c.children {
            if c.outcome == sign {
                let a: Action = a.parse()?;
                policy[legal.iter().position(|x| *x == a).unwrap()] = 1.;
            }
        }
        let total: f64 = policy.iter().sum();
        for p in &mut policy {
            *p /= total;
        }
        rows.push(row(&base, &record, source, "old", policy));
    }
    let fresh: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.train && r.cohort != "old")
        .map(|(i, _)| i)
        .collect();
    let old: Vec<_> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.train && r.cohort == "old")
        .map(|(i, _)| i)
        .collect();
    let train_sources: HashSet<_> = rows.iter().filter(|r| r.train).map(|r| &r.source).collect();
    assert!(rows
        .iter()
        .filter(|r| !r.train)
        .all(|r| !train_sources.contains(&r.source)));
    let report = |m: &MicroModel| -> Value {
        json!(["search","opponent","old"].iter().map(|name|json!({"cohort":name,"train":score(m,&rows,name,true,&base),"validation":score(m,&rows,name,false,&base)})).collect::<Vec<_>>())
    };
    let mut trials = vec![];
    let started = Instant::now();
    for seed in [37, 913, 4421] {
        for arm in [
            "uniform_no_recall",
            "uniform_recall50",
            "focused_recall50",
            "winning_set_recall50",
        ] {
            let mut m = base.as_ref().clone();
            let mut rng = StableRng::new(seed);
            let mut errors = vec![1.; rows.len()];
            let mut trace = vec![];
            for step in 0..=16384 {
                if [0, 1024, 4096, 8192, 16384].contains(&step) {
                    trace.push(json!({"step":step,"metrics":report(&m)}));
                }
                if step == 16384 {
                    break;
                }
                if step % 256 == 0 {
                    for &i in &fresh {
                        let p = probabilities(&m, &rows[i].ex);
                        errors[i] = 1.
                            - p.iter()
                                .zip(&rows[i].valid)
                                .filter(|(_, t)| **t)
                                .map(|(p, _)| p)
                                .sum::<f64>();
                    }
                }
                let mut candidates = [0; 4];
                for i in &mut candidates {
                    *i = fresh[rng.index(fresh.len())];
                }
                let oi = old[rng.index(old.len())];
                let i = if arm != "uniform_no_recall" && step % 2 == 1 {
                    oi
                } else if (arm == "focused_recall50" || arm == "winning_set_recall50")
                    && step % 4 == 0
                {
                    *candidates
                        .iter()
                        .max_by(|a, b| errors[**a].total_cmp(&errors[**b]))
                        .unwrap()
                } else {
                    candidates[0]
                };
                if arm == "winning_set_recall50" {
                    let p = probabilities(&m, &rows[i].ex);
                    let mut ex = rows[i].ex.clone();
                    let mass: f64 = p
                        .iter()
                        .zip(&rows[i].valid)
                        .filter(|(_, v)| **v)
                        .map(|(p, _)| p)
                        .sum();
                    assert!(mass > 0.);
                    ex.policy = p
                        .iter()
                        .zip(&rows[i].valid)
                        .map(|(p, v)| if *v { *p / mass } else { 0. })
                        .collect();
                    // Exact gradient of -log(sum of probabilities of verified wins).
                    // The conditional target is detached for the native CE update.
                    m.train_policy_step(&ex, 0.02 / 64.)?;
                } else {
                    m.train_policy_step(&rows[i].ex, 0.02 / 64.)?;
                }
            }
            trials.push(json!({"seed":seed,"arm":arm,"trace":trace}));
            fs::write(
                &a[2],
                serde_json::to_vec(
                    &json!({"trials":trials,"teacher_searches":searches,"rows":rows.iter().map(|r|json!({"cohort":r.cohort,"source":r.source,"train":r.train,"prefix":r.prefix})).collect::<Vec<_>>(),"neural_value_unchanged":true,"production_writes":0,"model_identity":artifact.identity()}),
                )?,
            )?;
            eprintln!(
                "{seed} {arm} elapsed {:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}
