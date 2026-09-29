//! Bounded read-only diagnosis of teacher feedback, direct recall and gradient directions.
use paisho_train::micro_learning::certificate_action_values;
use paisho_ai::*;
use paisho_train::micro_learning as action_values;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const RULES: RuleProfileId = RuleProfileId::SkudPaiShoGen5V1;
fn invalid(s: impl Into<String>) -> Box<dyn std::error::Error> {
    s.into().into()
}
fn sha256(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
#[path = "../src/micro_learning/gen5/teaching.rs"]
mod actual_teaching;
#[path = "../src/micro_learning/gen5/durable/winning.rs"]
mod actual_winning;
#[path = "gen5_v33_flow_probe/gradients.rs"]
mod gradients;
fn best(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn prior(model: &MicroModel, ex: &MicroExample) -> Result<Vec<f64>> {
    let e = model.embed(&ex.state);
    let p = micro_softmax(&MicroModel::logits(&e, &ex.actions))?;
    Ok(model.memory_priors(&ex.state, &ex.actions, &p, ex.sequence_source)?)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let spec: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut models = vec![];
    let mut identities = vec![];
    for m in spec["models"].as_array().ok_or("models")? {
        let a: MicroArtifact =
            serde_json::from_slice(&fs::read(m["path"].as_str().ok_or("path")?)?)?;
        assert_eq!(a.identity(), m["identity"]);
        identities.push(a.identity());
        models.push(a.model()?);
    }
    let mut proof_sets: BTreeMap<String, Vec<MicroExample>> = BTreeMap::new();
    for p in spec["positions"].as_array().ok_or("positions")? {
        if p["kind"] != "certificate" {
            continue;
        }
        let v: Value = serde_json::from_slice(&fs::read(p["path"].as_str().ok_or("path")?)?)?;
        let record: GameRecord = v["prefix"].as_str().ok_or("prefix")?.parse()?;
        let position = record.replay()?;
        let cert: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
        cert.verify(&position)?;
        let sign = if position.to_move() == Player::Host {
            1
        } else {
            -1
        };
        let z = (cert.outcome * sign) as f64;
        let legal = legal_actions(&position);
        let mut target = vec![0.; legal.len()];
        if z == 1. {
            for (a, c) in cert.children {
                if c.outcome == sign {
                    let a: Action = a.parse()?;
                    target[legal.iter().position(|x| *x == a).ok_or("proof action")?] = 1.;
                }
            }
            let n: f64 = target.iter().sum();
            assert!(n > 0.);
            for t in &mut target {
                *t /= n;
            }
        }
        let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            state: models[0].state_features(&position),
            actions: if z == 1. {
                legal
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect()
            } else {
                vec![]
            },
            policy: if z == 1. { target } else { vec![] },
            value: z,
            value_weight: 1.,
            policy_weight: if z == 1. { 1. } else { 0. },
            sequence_source: 0,
        };
        if p["group"] == "outside" && z == 1. {
            let mut union = ex.clone();
            for (i, action) in legal.iter().enumerate() {
                let mut next = position.clone();
                next.apply(*action)?;
                if next.outcome() == GameOutcome::Win(position.to_move()) {
                    union.policy[i] = 1.;
                }
            }
            let total: f64 = union.policy.iter().sum();
            for t in &mut union.policy {
                *t /= total;
            }
            proof_sets
                .entry("outside_any_1".into())
                .or_default()
                .push(union);
        }
        proof_sets
            .entry(format!("{}_{}", p["group"].as_str().unwrap(), z as i8))
            .or_default()
            .push(ex);
    }
    let mut fresh = vec![];
    let mut teacher_rows = vec![];
    let mut kinds = BTreeMap::new();
    for path in spec["bundles"].as_array().ok_or("bundles")? {
        let bytes = fs::read(path.as_str().ok_or("bundle")?)?;
        let b: Value = serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
        let record: GameRecord = b["psr"].as_str().ok_or("psr")?.parse()?;
        let mut position = record.initial_position();
        let mut next = 0;
        let mi = identities
            .iter()
            .position(|i| Some(i.as_str()) == b["source"].as_str());
        for l in b["lessons"].as_array().ok_or("lessons")? {
            let d = l["decision"].as_u64().ok_or("decision")? as usize;
            while next < d - 1 {
                position.apply(record.actions()[next])?;
                next += 1;
            }
            let pw = l["policy_weight"].as_f64().ok_or("policy weight")?;
            let legal = if pw > 0. {
                legal_actions(&position)
            } else {
                vec![]
            };
            let mut target = vec![0.; legal.len()];
            for pair in l["policy"].as_array().ok_or("policy")? {
                let a: Action = pair[0].as_str().ok_or("action")?.parse()?;
                target[legal.iter().position(|x| *x == a).ok_or("target action")?] =
                    pair[1].as_f64().ok_or("target")?;
            }
            let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
                state: models[0].state_features(&position),
                actions: legal
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect(),
                policy: target,
                value: l["value"].as_f64().ok_or("value")?,
                value_weight: l["evidence"]["value_weight"].as_f64().ok_or("vw")?,
                policy_weight: pw,
                sequence_source: sequence_source(&format!(
                    "{}/{}",
                    b["source_run"].as_str().ok_or("source_run")?,
                    b["game_id"].as_u64().ok_or("id")?
                )),
            };
            ex.validate()?;
            let kind = l["evidence"]["policy_source"].as_str().ok_or("source")?;
            *kinds.entry(kind.to_owned()).or_insert(0usize) += 1;
            if let Some(mi) = mi.filter(|_| kind == "full-search-estimate" && pw > 0.) {
                let mut raw_ex = ex.clone();
                raw_ex.sequence_source = 0;
                let raw = prior(&models[mi], &raw_ex)?;
                let saved: Vec<f64> =
                    serde_json::from_value(l["evidence"]["target_prior"].clone())?;
                let q: Vec<f64> =
                    serde_json::from_value(l["evidence"]["completed_action_values"].clone())?;
                let excluded: Vec<bool> =
                    serde_json::from_value(l["evidence"]["excluded_actions"].clone())?;
                let mut successor = vec![];
                for a in &legal {
                    let mut n = position.clone();
                    n.apply(*a)?;
                    let v = match n.outcome() {
                        GameOutcome::Win(w) => {
                            if w == position.to_move() {
                                1.
                            } else {
                                -1.
                            }
                        }
                        GameOutcome::Draw => 0.,
                        _ => {
                            models[mi].embed(&models[mi].state_features(&n)).value
                                * if n.to_move() == position.to_move() {
                                    1.
                                } else {
                                    -1.
                                }
                        }
                    };
                    successor.push(v);
                }
                let coupled = micro_softmax(
                    &raw.iter()
                        .zip(&successor)
                        .map(|(p, q)| p.max(1e-300).ln() + 16. * q)
                        .collect::<Vec<_>>(),
                )?;
                let deviation = coupled
                    .iter()
                    .zip(&saved)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0., f64::max);
                let alt_logits: Vec<_> = raw
                    .iter()
                    .zip(&q)
                    .enumerate()
                    .map(|(i, (p, q))| {
                        if excluded[i] {
                            -1e100
                        } else {
                            p.max(1e-300).ln() + q
                        }
                    })
                    .collect();
                let alt = micro_softmax(&alt_logits)?;
                let deployed = micro_softmax(
                    &ex.policy
                        .iter()
                        .zip(&successor)
                        .enumerate()
                        .map(|(i, (p, v))| {
                            if excluded[i] {
                                -1e100
                            } else {
                                p.max(1e-300).ln() + 16. * v
                            }
                        })
                        .collect::<Vec<_>>(),
                )?;
                let corrected_deployed = micro_softmax(
                    &alt.iter()
                        .zip(&successor)
                        .enumerate()
                        .map(|(i, (p, v))| {
                            if excluded[i] {
                                -1e100
                            } else {
                                p.max(1e-300).ln() + 16. * v
                            }
                        })
                        .collect::<Vec<_>>(),
                )?;
                let expected_q = |p: &[f64]| p.iter().zip(&q).map(|(p, q)| p * q).sum::<f64>();
                let qmax = (0..q.len())
                    .filter(|i| !excluded[*i])
                    .map(|i| q[i])
                    .fold(-1., f64::max);
                let ti = best(&ex.policy);
                let ai = best(&alt);
                teacher_rows.push(json!({"game":b["game_id"],"decision":d,"model":mi,"coupled_prior_max_error":deviation,
     "raw_argmax":best(&raw),"coupled_argmax":best(&saved),"teacher_argmax":ti,"raw_plus_q_argmax":ai,
     "teacher_q_regret":qmax-q[ti],"raw_plus_q_regret":qmax-q[ai],"teacher_value":successor[ti],"raw_plus_q_value":successor[ai],
     "teacher_q":q[ti],"raw_plus_q":q[ai],"max_successor_value":successor.iter().copied().fold(-1.,f64::max),"q_range":qmax-q.iter().copied().fold(1.,f64::min),
     "expected_search_q":{"coupled_before":expected_q(&saved),"teacher":expected_q(&ex.policy),"deployed_after_exact_raw_fit":expected_q(&deployed),"deployed_after_corrected_raw_fit":expected_q(&corrected_deployed)},
     "coordinate_correction_max_error":corrected_deployed.iter().zip(&ex.policy).map(|(a,b)|(a-b).abs()).fold(0.,f64::max)}));
            }
            fresh.push(ex);
        }
    }
    eprintln!(
        "reconstructed {} fresh targets; {} exact-collector teachers",
        fresh.len(),
        teacher_rows.len()
    );
    let mut gradient_results = vec![];
    for (mi, model) in models.iter().enumerate() {
        if spec["models"][mi]["gradients"] != true {
            continue;
        }
        gradient_results
            .push(json!({"model":mi,"results":gradients::run(model,&fresh,&proof_sets)?}));
    }
    // Replay the production direct sampler on a frozen old catalogue, without live proof arrivals.
    let mut winning = actual_winning::Winning::default();
    winning.policy_only = true;
    winning.add_root(Path::new(spec["old_proofs"].as_str().ok_or("old proofs")?))?;
    winning.restore(&spec["resume_winning"]);
    let focus: Vec<_> = proof_sets["guard_1"]
        .iter()
        .cloned()
        .map(|mut x| {
            x.value_weight = 0.;
            Arc::new(x)
        })
        .collect();
    winning.focus = focus.clone();
    let mut rng = StableRng::new(20260912);
    let mut focused = 0;
    let mut distinct = BTreeMap::new();
    let before = winning.progress();
    for _ in 0..4096 {
        for ex in winning.draw(16, &mut rng, &models[0])? {
            focused += usize::from(focus.iter().any(|f| Arc::ptr_eq(f, &ex)));
            assert_eq!(ex.value_weight, 0.);
            let key = sha256(&serde_json::to_vec(&ex.state)?);
            *distinct.entry(key).or_insert(0usize) += 1;
        }
    }
    // Native teacher control: full-search Q can disagree with the separately coupled value.
    let record: GameRecord = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
    ))
    .parse()?;
    let mut session = MicroMctsSession::new(Arc::new(models[0].clone()));
    let mut r = session.search_with_options(
        &record.initial_position(),
        8,
        None,
        MicroSearchOptions::default(),
    )?;
    r.actions.truncate(2);
    r.visits = vec![256, 256];
    r.values = vec![-0.9, 0.9];
    r.proven_value = None;
    r.proven_action_values = vec![None, None];
    r.priors = micro_softmax(&[3.2, -3.2])?;
    let before = r.priors.clone();
    let biased = actual_teaching::target(&r)?.0;
    r.priors = vec![0.5, 0.5];
    let searched = actual_teaching::target(&r)?.0;
    let deployed = micro_softmax(&[biased[0].ln() + 3.2, biased[1].ln() - 3.2])?;
    let corrected = micro_softmax(&[searched[0].ln() + 3.2, searched[1].ln() - 3.2])?;
    let mut pi: Vec<f64> = vec![0.5, 0.5];
    r.values = vec![0., 0.];
    let mut fixed_point = vec![pi.clone()];
    for _ in 0..4 {
        r.priors = micro_softmax(&[pi[0].ln() + 1.6, pi[1].ln() - 1.6])?;
        pi = actual_teaching::target(&r)?.0;
        fixed_point.push(pi.clone());
    }
    let output = json!({"teacher_rows":teacher_rows,"fresh_targets":fresh.len(),"kinds":kinds,"gradients":gradient_results,
  "recall_control":{"draws":65536,"focus_draws":focused,"distinct_input_states":distinct.len(),"counts":distinct,"before":before,"after":winning.progress(),"value_weight_zero":true,"live_replay":false},
  "native_teacher_control":{"successor_values":[0.2,-0.2],"search_q":[-0.9,0.9],"coupled_teacher":biased,"raw_plus_q_teacher":searched,
   "coupled_before":before,"deployed_after_exact_raw_fit":deployed,"deployed_after_corrected_raw_fit":corrected,
   "flat_search_q_repeated_exact_distillation_successor_values":[0.1,-0.1],"raw_policy_sequence":fixed_point},"production_writes":0});
    fs::write(&args[2], serde_json::to_vec(&output)?)?;
    Ok(())
}
