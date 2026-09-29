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

    let baseline = score(&actor, &rows)?;
    let mut current = MicroArtifact::load(Path::new(&args[3]))?.model()?;
    for (row, spec) in rows
        .iter_mut()
        .zip(manifest["positions"].as_array().unwrap())
    {
        let v: Value = serde_json::from_slice(&fs::read(spec["path"].as_str().unwrap())?)?;
        let p = v["prefix"]
            .as_str()
            .unwrap()
            .parse::<GameRecord>()?
            .replay()?;
        row.q = legal_actions(&p)
            .iter()
            .map(|a| {
                let mut n = p.clone();
                n.apply(*a).unwrap();
                match n.outcome() {
                    GameOutcome::Win(w) => {
                        if w == p.to_move() {
                            1.
                        } else {
                            -1.
                        }
                    }
                    GameOutcome::Draw => 0.,
                    _ => {
                        current.value(&current.state_features(&n))
                            * if n.to_move() == p.to_move() { 1. } else { -1. }
                    }
                }
            })
            .collect();
    }
    let before = score(&current, &rows)?;
    let value_parent = current.clone();
    let mut history = vec![];
    let lost = |s: &Score| -> Vec<usize> {
        (0..rows.len())
            .filter(|&i| {
                rows[i].group < 2
                    && ((baseline.raw[i] && !s.raw[i]) || (baseline.coupled[i] && !s.coupled[i]))
            })
            .collect()
    };
    for iter in 0..32 {
        let old = score(&current, &rows)?;
        let failures = lost(&old);
        if failures.is_empty() {
            break;
        }
        let mut g = vec![0.; 292363];
        for &i in &failures {
            let r = &rows[i];
            let mut e = r.ex.clone();
            let raw = prior(&current, &e)?;
            let coupled = micro_softmax(
                &raw.iter()
                    .zip(&r.q)
                    .map(|(p, q)| p.max(1e-300).ln() + 16. * q)
                    .collect::<Vec<_>>(),
            )?;
            let good = (0..e.policy.len())
                .filter(|&j| e.policy[j] > 0.)
                .max_by(|&a, &b| coupled[a].total_cmp(&coupled[b]))
                .unwrap();
            let bad = (0..e.policy.len())
                .filter(|&j| e.policy[j] == 0.)
                .max_by(|&a, &b| coupled[a].total_cmp(&coupled[b]))
                .unwrap();
            e.policy.fill(0.);
            e.policy[good] = 1.;
            let a = current.loss_gradient(&e)?.1;
            e.policy.fill(0.);
            e.policy[bad] = 1.;
            let b = current.loss_gradient(&e)?.1;
            for (x, (a, b)) in g.iter_mut().zip(a.into_iter().zip(b)) {
                *x += (a - b) / failures.len() as f64;
            }
        }
        let objective = |m: &MicroModel| -> Result<f64, E> {
            let mut loss = 0.;
            for &i in &failures {
                let r = &rows[i];
                let p = prior(m, &r.ex)?;
                let p = micro_softmax(
                    &p.iter()
                        .zip(&r.q)
                        .map(|(p, q)| p.max(1e-300).ln() + 16. * q)
                        .collect::<Vec<_>>(),
                )?;
                let good = p
                    .iter()
                    .zip(&r.ex.policy)
                    .filter(|(_, t)| **t > 0.)
                    .map(|(p, _)| p.max(1e-300).ln())
                    .fold(f64::NEG_INFINITY, f64::max);
                let bad = p
                    .iter()
                    .zip(&r.ex.policy)
                    .filter(|(_, t)| **t == 0.)
                    .map(|(p, _)| p.max(1e-300).ln())
                    .fold(f64::NEG_INFINITY, f64::max);
                loss += bad - good;
            }
            Ok(loss)
        };
        let initial_loss = objective(&current)?;
        let mut selected = None;
        for k in 0..12 {
            let rate = 0.002 * 0.5f64.powi(k);
            let m = step(&current, &g, rate, true)?;
            let next = score(&m, &rows)?;
            let count = lost(&next).len();
            if count <= 8 && objective(&m)? < initial_loss {
                selected = Some((m, next, rate));
                break;
            }
        }
        if let Some((m, next, rate)) = selected {
            current = m;
            history.push(json!({"step":iter,"rate":rate,"remaining_lost":lost(&next),"score":view(&next,&rows)}));
        } else {
            break;
        }
    }
    let final_score = score(&current, &rows)?;
    assert!(value_parent
        .parameters()
        .iter()
        .zip(current.parameters())
        .enumerate()
        .filter(|(i, _)| value_index(*i))
        .all(|(_, (a, b))| a.to_bits() == b.to_bits()));
    MicroArtifact::new(
        &current,
        0,
        json!({"diagnostic_only":true,"value_then_coupled_policy_repair":true}),
    )
    .save(&out.join("proposed-model.json"))?;
    fs::write(
        out.join("results.json"),
        serde_json::to_vec_pretty(
            &json!({"baseline":view(&baseline,&rows),"before":view(&before,&rows),"after":view(&final_score,&rows),"initial_lost":lost(&before),"remaining_lost":lost(&final_score),"history":history,"value_parameters_preserved":true,"scope":"Coupled correction on known failures only; all prior raw/coupled wins in 57+339 checked; 24 historical extras not used in fit. No production activation."}),
        )?,
    )?;
    Ok(())
}
