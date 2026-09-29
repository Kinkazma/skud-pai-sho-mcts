use super::*;
use data::{Data, Proof};
#[path = "projection.rs"]
mod projection;
#[derive(Clone, serde::Serialize)]
struct Score {
    raw: Vec<bool>,
    coupled: Vec<bool>,
    mass: f64,
    mse: [f64; 3],
    balanced_mse: f64,
    any_wins: usize,
}
fn score(m: &MicroModel, rows: &[&Proof], coupled: bool) -> Score {
    let mut s = Score {
        raw: vec![],
        coupled: vec![],
        mass: 0.,
        mse: [0.; 3],
        balanced_mse: 0.,
        any_wins: 0,
    };
    let mut ns = [0; 3];
    let mut wins = 0;
    for r in rows {
        let k = (r.ex.value as i8 + 1) as usize;
        ns[k] += 1;
        s.mse[k] += (m.embed(&r.ex.state).value - r.ex.value).powi(2);
        if r.ex.value == 1. {
            let p = prior(m, &r.ex);
            let i = best(&p);
            s.raw.push(r.certificate[i]);
            s.any_wins += usize::from(r.union[i]);
            s.mass += p
                .iter()
                .zip(&r.certificate)
                .filter(|(_, v)| **v)
                .map(|(p, _)| p)
                .sum::<f64>();
            wins += 1;
            if coupled {
                let logits: Vec<_> = p
                    .iter()
                    .zip(&r.successors)
                    .map(|(p, s)| p.max(1e-300).ln() + 16. * s.value(m))
                    .collect();
                s.coupled.push(r.certificate[best(&logits)]);
            }
        }
    }
    for k in 0..3 {
        s.mse[k] /= ns[k].max(1) as f64;
    }
    s.balanced_mse = s
        .mse
        .iter()
        .zip(ns)
        .filter(|(_, n)| *n > 0)
        .map(|(x, _)| x)
        .sum::<f64>()
        / ns.iter().filter(|n| **n > 0).count() as f64;
    s.mass /= wins.max(1) as f64;
    s
}
fn passes(s: &Score, old: &Score) -> bool {
    s.raw.iter().zip(&old.raw).all(|(a, b)| !*b || *a)
        && s.coupled.iter().zip(&old.coupled).all(|(a, b)| !*b || *a)
        && s.mass + 1e-12 >= old.mass
        && s.balanced_mse <= old.balanced_mse + 1e-12
}
fn merged(policy: &MicroModel, value: &MicroModel) -> MicroModel {
    clone_weights(
        policy,
        policy
            .parameters()
            .iter()
            .zip(value.parameters())
            .enumerate()
            .map(|(i, (p, v))| if value_parameter(i) { *v } else { *p })
            .collect(),
    )
}
pub fn run(d: &Data, review: &Value, out: &std::path::Path) -> Result<Value> {
    let guard: Vec<_> = d.proofs.iter().filter(|p| p.group == "guard").collect();
    let outside: Vec<_> = d.proofs.iter().filter(|p| p.group == "outside").collect();
    let accepted = &d.models[4];
    let final_model = &d.models[5];
    let old = score(accepted, &guard, true);
    let mut composites = vec![];
    for spec in review["models"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|s| s["version"].as_u64().unwrap() > 1029160)
    {
        let a: MicroArtifact = serde_json::from_slice(&fs::read(spec["path"].as_str().unwrap())?)?;
        let candidate = clone_weights(accepted, a.parameters.clone());
        for take_policy in [true, false] {
            let model = if take_policy {
                merged(&candidate, accepted)
            } else {
                merged(accepted, &candidate)
            };
            // Branch composition is verified as a function, not only by parameter masks.
            for r in &guard {
                assert_eq!(
                    model.embed(&r.ex.state).value.to_bits(),
                    if take_policy { accepted } else { &candidate }
                        .embed(&r.ex.state)
                        .value
                        .to_bits()
                );
                if r.ex.value == 1. {
                    assert_eq!(
                        prior(&model, &r.ex),
                        prior(if take_policy { &candidate } else { accepted }, &r.ex)
                    );
                }
            }
            let s = score(&model, &guard, true);
            let ok = passes(&s, &old);
            let ext = score(&model, &outside, false);
            if ok {
                let artifact = MicroArtifact::new(
                    &model,
                    a.updates,
                    json!({"diagnostic_only":true,"kind":"separate-branch-proposal","source_version":spec["version"],"candidate_policy":take_policy}),
                );
                fs::write(
                    out.join(format!(
                        "composite-{}-{}.json",
                        spec["version"],
                        if take_policy { "policy" } else { "value" }
                    )),
                    serde_json::to_vec(&artifact)?,
                )?;
            }
            composites.push(json!({"version":spec["version"],"candidate_policy":take_policy,"guard":s,"passes_original_guard":ok,"outside":ext}));
        }
    }
    let mut repairs = vec![];
    let mut counts = [0usize; 3];
    for r in &guard {
        counts[(r.ex.value as i8 + 1) as usize] += 1;
    }
    let smallest = *counts.iter().filter(|n| **n > 0).min().unwrap() as f64;
    for mode in 0..3 {
        let balanced_value = mode > 0;
        let mut model = final_model.clone();
        let training: Vec<_> = guard
            .iter()
            .map(|r| {
                let mut e = r.ex.clone();
                e.value_weight = if balanced_value {
                    smallest / counts[(e.value as i8 + 1) as usize] as f64
                } else {
                    0.
                };
                e
            })
            .collect();
        let mut history = vec![];
        let mut first_pass = None;
        let started = Instant::now();
        for n in 0..=32 {
            if [0, 1, 4, 8, 16, 32].contains(&n) {
                let s = score(&model, &guard, true);
                let ok = passes(&s, &old);
                history.push(json!({"updates":n,"guard":s,"outside":score(&model,&outside,false),"passes_original_guard":ok}));
                if ok && first_pass.is_none() {
                    first_pass = Some(n);
                    let artifact = MicroArtifact::new(
                        &model,
                        4818997,
                        json!({"diagnostic_only":true,"kind":"balanced-guard-repair","balanced_value":balanced_value,"additional_steps":n}),
                    );
                    fs::write(
                        out.join(format!("repair-mode{}-{}.json", mode, n)),
                        serde_json::to_vec(&artifact)?,
                    )?;
                }
            }
            if n < 32 {
                let mut actual = training.clone();
                if mode == 2 {
                    let now = score(&model, &guard, true);
                    let critical: Vec<_> = now
                        .raw
                        .iter()
                        .zip(&old.raw)
                        .zip(now.coupled.iter().zip(&old.coupled))
                        .map(|((new, old), (nc, oc))| *old && !*new || *oc && !*nc)
                        .collect();
                    let count = critical.iter().filter(|v| **v).count();
                    let mut i = 0;
                    for ex in &mut actual {
                        if ex.value == 1. {
                            if count > 0 && count < critical.len() {
                                ex.policy_weight = if critical[i] {
                                    critical.len() as f64 / (2. * count as f64)
                                } else {
                                    critical.len() as f64 / (2. * (critical.len() - count) as f64)
                                };
                            }
                            i += 1;
                        }
                    }
                }
                let refs: Vec<_> = actual.iter().collect();
                model.train_batch_inline(&refs, 0.02 * refs.len() as f64 / 64., 0.)?;
            }
        }
        repairs.push(json!({"mode":mode,"balanced_value":balanced_value,"targeted_lost_choices":mode==2,"history":history,"first_pass":first_pass,"seconds":started.elapsed().as_secs_f64()}));
    }
    let constraints = projection::run(d, &outside)?;
    Ok(
        json!({"accepted":{"guard":old,"outside":score(accepted,&outside,false)},"composites":composites,"repairs":repairs,"projected_updates":constraints,"scope":"All corrected/composed weights are diagnostic clones; accepting the historical guard does not establish strength or unseen-source retention."}),
    )
}
