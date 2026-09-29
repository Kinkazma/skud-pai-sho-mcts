//! Offline flow diagnostics and bounded proposals. No production write or activation.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use std::{fs, path::Path};
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
#[path = "../src/micro_learning/gen5/protection/projection.rs"]
mod projection;
type E = Box<dyn std::error::Error>;
struct Row {
    ex: MicroExample,
    group: usize,
    q: Vec<f64>,
}
#[derive(Clone)]
struct Score {
    raw: Vec<bool>,
    coupled: Vec<bool>,
    mass: [f64; 3],
    counts: [usize; 3],
    priors: Vec<Vec<f64>>,
}
fn value_index(i: usize) -> bool {
    (4128..4161).contains(&i) || (15822..93071).contains(&i)
}
fn model(base: &MicroModel, w: Vec<f64>) -> Result<MicroModel, E> {
    let mut m = MicroModel::from_parameters(w)?;
    if let Some(b) = base.sequence_memory() {
        m = m.with_sequence_memory_owned(b.clone());
    }
    Ok(m)
}
fn prior(m: &MicroModel, e: &MicroExample) -> Result<Vec<f64>, E> {
    let p = micro_softmax(&MicroModel::logits(&m.embed(&e.state), &e.actions))?;
    Ok(m.memory_priors(&e.state, &e.actions, &p, e.sequence_source)?)
}
fn best(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn score(m: &MicroModel, rows: &[Row]) -> Result<Score, E> {
    let mut s = Score {
        raw: vec![],
        coupled: vec![],
        mass: [0.; 3],
        counts: [0; 3],
        priors: vec![],
    };
    for r in rows {
        let p = prior(m, &r.ex)?;
        let win = r.ex.value == 1.;
        s.priors.push(p.clone());
        s.raw.push(win && r.ex.policy[best(&p)] > 0.);
        let logits = p
            .iter()
            .zip(&r.q)
            .map(|(p, q)| p.max(1e-300).ln() + 16. * q)
            .collect::<Vec<_>>();
        s.coupled.push(win && r.ex.policy[best(&logits)] > 0.);
        if win {
            s.mass[r.group] += p
                .iter()
                .zip(&r.ex.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(p, _)| p)
                .sum::<f64>();
            s.counts[r.group] += 1;
        }
    }
    for i in 0..3 {
        s.mass[i] /= s.counts[i].max(1) as f64;
    }
    Ok(s)
}
fn view(s: &Score, rows: &[Row]) -> Value {
    json!({"raw":(0..3).map(|k|s.raw.iter().zip(rows).filter(|(b,r)|**b&&r.group==k).count()).collect::<Vec<_>>(),"coupled":(0..3).map(|k|s.coupled.iter().zip(rows).filter(|(b,r)|**b&&r.group==k).count()).collect::<Vec<_>>(),"mass":s.mass,"wins":s.counts})
}
fn safe(old: &Score, new: &Score, rows: &[Row]) -> bool {
    old.raw
        .iter()
        .zip(&new.raw)
        .zip(old.coupled.iter().zip(&new.coupled))
        .zip(rows)
        .all(|(((a, b), (c, d)), r)| r.group == 2 || (!a || *b) && (!c || *d))
        && old
            .priors
            .iter()
            .zip(&new.priors)
            .zip(rows)
            .filter(|(_, r)| r.group < 2)
            .map(|((a, b), _)| {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| {
                        if *a > 0. {
                            a * (a.max(1e-300) / b.max(1e-300)).ln()
                        } else {
                            0.
                        }
                    })
                    .sum::<f64>()
            })
            .sum::<f64>()
            / rows.iter().filter(|r| r.group < 2).count() as f64
            <= 0.001
}
fn step(m: &MicroModel, g: &[f64], rate: f64, freeze_value: bool) -> Result<MicroModel, E> {
    let norm = dot(g, g).sqrt();
    let scale = if norm > 10. { 10. / norm } else { 1. };
    model(
        m,
        m.parameters()
            .iter()
            .zip(g)
            .enumerate()
            .map(|(i, (w, g))| {
                if freeze_value && value_index(i) {
                    *w
                } else {
                    w - rate * scale * g
                }
            })
            .collect(),
    )
}
fn mean_gradient(m: &MicroModel, ex: &[MicroExample], mass: bool) -> Result<Vec<f64>, E> {
    let mut g = vec![0.; m.parameters().len()];
    for e in ex {
        let p = if mass {
            m.policy_mass_gradient(e)?
        } else {
            m.loss_gradient(e)?.1
        };
        for (a, b) in g.iter_mut().zip(p) {
            *a += b / ex.len() as f64;
        }
    }
    Ok(g)
}
fn main() -> Result<(), E> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let args = std::env::args().collect::<Vec<_>>();
    let out = Path::new(&args[2]);
    fs::create_dir_all(out)?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let spec = &manifest["models"];
    let actor = MicroArtifact::load(Path::new(spec[2]["path"].as_str().unwrap()))?.model()?;
    let learner = MicroArtifact::load(Path::new(spec[3]["path"].as_str().unwrap()))?.model()?;
    let mut rows = vec![];
    for spec in manifest["positions"].as_array().unwrap() {
        let v: Value = serde_json::from_slice(&fs::read(spec["path"].as_str().unwrap())?)?;
        let rec: GameRecord = v["prefix"].as_str().unwrap().parse()?;
        let p = rec.replay()?;
        let cert: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
        cert.verify(&p)?;
        let legal = legal_actions(&p);
        let value = (cert.outcome * if p.to_move() == Player::Host { 1 } else { -1 }) as f64;
        let valid = legal
            .iter()
            .map(|a| {
                cert.children
                    .iter()
                    .any(|(s, c)| s == &a.to_string() && c.outcome == cert.outcome)
                    || (value == 1. && {
                        let mut n = p.clone();
                        n.apply(*a).unwrap();
                        n.outcome() == GameOutcome::Win(p.to_move())
                    })
            })
            .collect::<Vec<_>>();
        let n = valid.iter().filter(|b| **b).count();
        let policy = valid
            .iter()
            .map(|b| if *b { 1. / n as f64 } else { 0. })
            .collect();
        let q = legal
            .iter()
            .map(|a| {
                let mut next = p.clone();
                next.apply(*a).unwrap();
                match next.outcome() {
                    GameOutcome::Win(w) => {
                        if w == p.to_move() {
                            1.
                        } else {
                            -1.
                        }
                    }
                    GameOutcome::Draw => 0.,
                    _ => {
                        actor.value(&actor.state_features(&next))
                            * if next.to_move() == p.to_move() {
                                1.
                            } else {
                                -1.
                            }
                    }
                }
            })
            .collect();
        rows.push(Row {
            ex: MicroExample { structured: Vec::new(), policy_support: false,
                state: actor.state_features(&p),
                actions: legal
                    .iter()
                    .map(|a| micro_action_features(&p, *a))
                    .collect(),
                policy,
                value,
                policy_weight: if value == 1. { 1. } else { 0. },
                value_weight: 0.,
                action_values: vec![],
                sequence_source: 0,
            },
            group: match spec["group"].as_str().unwrap() {
                "guard" => 0,
                "validation" => 1,
                _ => 2,
            },
            q,
        });
    }
    let mut leak = vec![];
    for (label, m) in [("actor", &actor), ("learner", &learner)] {
        for (i, r) in rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.group == 0 && r.ex.value == 1.)
        {
            let g = m.loss_gradient(&r.ex)?.1;
            let vnorm = g
                .iter()
                .enumerate()
                .filter(|(i, _)| value_index(*i))
                .map(|(_, g)| g * g)
                .sum::<f64>()
                .sqrt();
            leak.push(json!({"model":label,"row":i,"value_gradient_norm":vnorm,"gradient_norm":dot(&g,&g).sqrt()}));
        }
    }
    let mut systems = vec![];
    for (label, m) in [("actor", &actor), ("learner", &learner)] {
        let mut grads = vec![vec![0.; 292363]; 3];
        let mut ns = [0; 3];
        let mut losses = [0.; 3];
        for r in rows.iter().filter(|r| r.group == 0) {
            let k = (r.ex.value as i8 + 1) as usize;
            let mut e = r.ex.clone();
            e.policy_weight = 0.;
            e.value_weight = 1.;
            e.structured.clear();e.actions.clear();
            e.policy.clear();
            let g = m.loss_gradient(&e)?.1;
            for (a, b) in grads[k].iter_mut().zip(g) {
                *a += b;
            }
            losses[k] += 0.5 * (m.value(&e.state) - e.value).powi(2);
            ns[k] += 1;
        }
        for k in 0..3 {
            for x in &mut grads[k] {
                *x /= ns[k].max(1) as f64;
            }
            losses[k] /= ns[k].max(1) as f64;
        }
        systems.push(json!({"model":label,"losses":losses,"gram_diagonal":grads.iter().map(|g|dot(g,g)).collect::<Vec<_>>()}));
    }
    let initial = score(&actor, &rows)?;
    let mut current = actor.clone();
    let mut history = vec![];
    for iteration in 0..16 {
        let s = score(&current, &rows)?;
        let mut candidates = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.group < 2 && r.ex.value == 1.)
            .map(|(i, r)| {
                let p = prior(&current, &r.ex).unwrap();
                let mass = p
                    .iter()
                    .zip(&r.ex.policy)
                    .filter(|(_, t)| **t > 0.)
                    .map(|(p, _)| p)
                    .sum::<f64>();
                (i, mass)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        let examples = candidates
            .iter()
            .take(32)
            .map(|(i, _)| rows[*i].ex.clone())
            .collect::<Vec<_>>();
        let g = mean_gradient(&current, &examples, true)?;
        let mut accepted = None;
        for k in 0..12 {
            let rate = 0.02 * 0.5f64.powi(k);
            let m = step(&current, &g, rate, true)?;
            let next = score(&m, &rows)?;
            if safe(&s, &next, &rows) {
                accepted = Some((m, next, rate));
                break;
            }
        }
        match accepted {
            Some((m, next, rate)) => {
                current = m;
                history.push(json!({"iteration":iteration,"rate":rate,"score":view(&next,&rows)}));
            }
            None => {
                history.push(json!({"iteration":iteration,"rate":0}));
                break;
            }
        }
    }
    let mut projected_history = vec![];
    for iteration in 0..12 {
        let old = score(&current, &rows)?;
        let mut ranked = vec![];
        let mut refs = vec![vec![0.; 292363]; 2];
        let mut counts = [0; 2];
        for (i, r) in rows
            .iter()
            .enumerate()
            .filter(|(_, r)| r.group < 2 && r.ex.value == 1.)
        {
            let p = prior(&current, &r.ex)?;
            let mass: f64 = p
                .iter()
                .zip(&r.ex.policy)
                .filter(|(_, t)| **t > 0.)
                .map(|(p, _)| p)
                .sum();
            ranked.push((i, mass));
            let part = current.policy_mass_gradient(&r.ex)?;
            counts[r.group] += 1;
            for (j, (g, x)) in refs[r.group].iter_mut().zip(part).enumerate() {
                if !value_index(j) {
                    *g += mass * x;
                }
            }
        }
        for k in 0..2 {
            for x in &mut refs[k] {
                *x /= counts[k] as f64;
            }
        }
        ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
        let examples = ranked
            .iter()
            .take(32)
            .map(|(i, _)| rows[*i].ex.clone())
            .collect::<Vec<_>>();
        let mut g = mean_gradient(&current, &examples, true)?;
        for (j, x) in g.iter_mut().enumerate() {
            if value_index(j) {
                *x = 0.;
            }
        }
        let before_norm = dot(&g, &g).sqrt();
        let g = projection::homogeneous(&g, &refs, &projection::gram(&refs))
            .ok_or("invalid projection")?;
        let mut accepted = None;
        for k in 0..16 {
            let rate = 0.02 * 0.5f64.powi(k);
            let m = step(&current, &g, rate, true)?;
            let next = score(&m, &rows)?;
            if safe(&old, &next, &rows) {
                accepted = Some((m, next, rate));
                break;
            }
        }
        match accepted {
            Some((m, next, rate)) => {
                current = m;
                projected_history.push(json!({"iteration":iteration,"rate":rate,"retained_norm":dot(&g,&g).sqrt()/before_norm,"score":view(&next,&rows)}));
            }
            None => {
                projected_history.push(json!({"iteration":iteration,"rate":0}));
                break;
            }
        }
    }
    let repaired = score(&current, &rows)?;
    assert!(safe(&initial, &repaired, &rows));
    assert!(actor
        .parameters()
        .iter()
        .zip(current.parameters())
        .enumerate()
        .filter(|(i, _)| value_index(*i))
        .all(|(_, (a, b))| a.to_bits() == b.to_bits()));
    let samples: Vec<SavedMicroExample> =
        serde_json::from_slice(&fs::read(out.join("sample-examples.json"))?)?;
    let examples = samples
        .iter()
        .filter(|x| x.policy_weight > 0.)
        .take(256)
        .map(|x| {
            x.example_for_rules(RuleProfileId::SkudPaiShoGen5V1)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut plain = current.clone();
    let mut protected = current.clone();
    let mut interference = vec![];
    for (i, chunk) in examples.chunks(32).enumerate() {
        let g = mean_gradient(&plain, chunk, false)?;
        plain = step(&plain, &g, 0.01, true)?;
        let old = score(&protected, &rows)?;
        let g = mean_gradient(&protected, chunk, false)?;
        let mut accepted = 0.;
        for k in 0..12 {
            let rate = 0.01 * 0.5f64.powi(k);
            let m = step(&protected, &g, rate, true)?;
            let s = score(&m, &rows)?;
            if safe(&old, &s, &rows) {
                protected = m;
                accepted = rate;
                break;
            }
        }
        interference.push(json!({"batch":i,"accepted_rate":accepted}));
    }
    let mut reg = vec![];
    for rhs in [vec![0.1, 1e-9], vec![0.1, -1e-9]] {
        let refs = vec![vec![1., 0.], vec![0., 1e-8]];
        let normalized = refs
            .iter()
            .map(|r| {
                let n = dot(r, r).sqrt();
                r.iter().map(|x| x / n).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let nrhs = rhs
            .iter()
            .zip(&refs)
            .map(|(b, r)| b / dot(r, r).sqrt())
            .collect::<Vec<_>>();
        reg.push(json!({"rhs":rhs,"original":projection::project(&[0.,0.],&refs,&rhs),"normalized":projection::project(&[0.,0.],&normalized,&nrhs)}));
    }
    MicroArtifact::new(
        &current,
        0,
        json!({"diagnostic_only":true,"kind":"proof_mass_trust_region"}),
    )
    .save(&out.join("proposed-model.json"))?;
    fs::write(
        out.join("native-results.json"),
        serde_json::to_vec_pretty(
            &json!({"policy_to_value":leak,"value_systems":systems,"initial":view(&initial,&rows),"repaired":view(&repaired,&rows),"history":history,"projected_history":projected_history,"interference":interference,"unprotected_after":view(&score(&plain,&rows)?,&rows),"protected_after":view(&score(&protected,&rows)?,&rows),"affine_controls":reg,"value_parameters_exact":true,"gate":"all previous raw/coupled winning choices retained; mean KL <= 0.001 per accepted step; mean mass no longer a hard monotonic constraint","scope":"Frozen value; policy and neural corrections trained. 57+339 known positions constrain proposals; 24 historical extra positions only scored. Not a full production-loop trial or unseen evaluation."}),
        )?,
    )?;
    Ok(())
}
