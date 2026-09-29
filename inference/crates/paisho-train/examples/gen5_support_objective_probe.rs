//! Frozen proof-policy objective trial. No games, campaign writes or model output.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, io::Read, path::Path, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn value_parameter(i: usize) -> bool {
    let start = (MICRO_INPUTS + 1) * MICRO_HIDDEN;
    (start..=start + MICRO_HIDDEN).contains(&i)
        || (MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START).contains(&i)
}
struct Row {
    id: usize,
    variants: [MicroExample; 3],
    support: Vec<bool>,
    successor_q: Vec<f64>,
    source: String,
}
fn best(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn priors(model: &MicroModel, ex: &MicroExample) -> Result<Vec<f64>> {
    let base = micro_softmax(&MicroModel::logits(&model.embed(&ex.state), &ex.actions))?;
    Ok(model.memory_priors(&ex.state, &ex.actions, &base, ex.sequence_source)?)
}
fn policy_example(row: &Row, variant: usize) -> MicroExample {
    let mut ex = row.variants[variant].clone();
    ex.value_weight = 0.;
    ex.action_values.clear();
    ex
}
fn timing(model: &MicroModel, rows: &[Row], variant: usize) -> Result<f64> {
    let started = Instant::now();
    let mut buffer = vec![];
    let mut check = 0.;
    for row in rows {
        let (loss, g) = model.loss_gradient_loop_v3_reusing(&row.variants[variant], buffer)?;
        check += loss.total(row.variants[variant].policy_weight) + g[0];
        buffer = g;
    }
    std::hint::black_box(check);
    Ok(started.elapsed().as_secs_f64())
}
fn measure(model: &MicroModel, rows: &[Row]) -> Result<Vec<Value>> {
    rows.iter().map(|row| {
        let p=priors(model,&row.variants[0])?;
        let logits=p.iter().zip(&row.successor_q).map(|(p,q)|p.max(1e-300).ln()+16.*q).collect::<Vec<_>>();
        let mass:f64=p.iter().zip(&row.support).filter(|(_,s)|**s).map(|(p,_)|p).sum();
        Ok(json!({"position":row.id,"raw_correct":row.support[best(&p)],"coupled_correct":row.support[best(&logits)],
            "winning_mass":mass,"winning_loss":-mass.max(1e-300).ln(),"raw_index":best(&p),"coupled_index":best(&logits)}))
    }).collect()
}
fn summary(before: &[Value], after: &[Value]) -> Value {
    let count = |name: &str| {
        json!({"before":before.iter().filter(|r|r[name]==true).count(),"after":after.iter().filter(|r|r[name]==true).count(),
        "gains":before.iter().zip(after).filter(|(a,b)|a[name]==false && b[name]==true).count(),
        "losses":before.iter().zip(after).filter(|(a,b)|a[name]==true && b[name]==false).count()})
    };
    json!({"raw":count("raw_correct"),"coupled":count("coupled_correct"),
        "mean_winning_mass":after.iter().map(|r|r["winning_mass"].as_f64().unwrap()).sum::<f64>()/after.len() as f64,
        "mean_winning_loss":after.iter().map(|r|r["winning_loss"].as_f64().unwrap()).sum::<f64>()/after.len() as f64})
}
fn main() -> Result<()> {
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    let args = std::env::args().collect::<Vec<_>>();
    if !(5..=6).contains(&args.len()) {
        return Err("MODEL PROVENANCE_JSON NEW_OUTPUT_JSON RATE [BATCH_UPDATES=4]".into());
    }
    let rate: f64 = args[4].parse()?;
    let updates: usize = args.get(5).map(|x| x.parse()).transpose()?.unwrap_or(4);
    if !rate.is_finite() || rate <= 0. || updates == 0 || updates > 32 {
        return Err("invalid bounded trial settings".into());
    }
    if Path::new(&args[3]).exists() {
        return Err("output already exists".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let loading = Instant::now();
    let model_bytes = fs::read(&args[1])?;
    let artifact = MicroArtifact::load(Path::new(&args[1]))?;
    let model = artifact.model()?;
    if !model.has_neural_memory() || !model.has_deep_value() {
        return Err("requires separated neural-memory model".into());
    }
    let provenance_bytes = fs::read(&args[2])?;
    let provenance: Value = serde_json::from_slice(&provenance_bytes)?;
    let mut cache = HashMap::<String, Vec<SavedMicroExample>>::new();
    let mut rows = vec![];
    let mut inputs = vec![];
    for sample in provenance["sampled"].as_array().ok_or("sampled")? {
        if sample["target_value"] != 1 {
            continue;
        }
        let teaching = &sample["first_selected_teaching"];
        if teaching.is_null() {
            continue;
        }
        let path = teaching["targets"]["path"].as_str().ok_or("targets path")?;
        if !cache.contains_key(path) {
            let bytes = fs::read(path)?;
            let sha = hash(&bytes);
            if teaching["targets"]["sha256"] != sha {
                return Err("target hash mismatch".into());
            }
            let mut decoded = vec![];
            flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut decoded)?;
            cache.insert(path.into(), serde_json::from_slice(&decoded)?);
            inputs.push(json!({"path":path,"sha256":sha}));
        }
        let saved = &cache[path][teaching["target_index"].as_u64().ok_or("target index")? as usize];
        let record: GameRecord = teaching["prefix"].as_str().ok_or("prefix")?.parse()?;
        if record.rules() != paisho_train::micro_learning::gen5::RULES {
            return Err("rule mismatch".into());
        }
        if hash(record.to_string().as_bytes())
            != sample["prefix_sha256"].as_str().ok_or("prefix hash")?
        {
            return Err("prefix hash mismatch".into());
        }
        let position = record.replay()?;
        let certificate: MicroProofCertificate =
            serde_json::from_value(teaching["certificate"].clone())?;
        certificate.verify(&position)?;
        if certificate.outcome
            != if position.to_move() == Player::Host {
                1
            } else {
                -1
            }
        {
            return Err("not a winning proof".into());
        }
        let legal = legal_actions(&position);
        let mut support = vec![];
        let mut successor_q = vec![];
        for action in &legal {
            let mut next = position.clone();
            next.apply(*action)?;
            support.push(
                certificate
                    .children
                    .iter()
                    .any(|(a, c)| a == &action.to_string() && c.outcome == certificate.outcome)
                    || next.outcome() == GameOutcome::Win(position.to_move()),
            );
            let q = match next.outcome() {
                GameOutcome::Win(w) => {
                    if w == position.to_move() {
                        1.
                    } else {
                        -1.
                    }
                }
                GameOutcome::Draw => 0.,
                _ => {
                    model.value(&model.state_features(&next))
                        * if next.to_move() == position.to_move() {
                            1.
                        } else {
                            -1.
                        }
                }
            };
            successor_q.push(q);
        }
        let mut a = saved.example_for_rules_with_trusted_q(record.rules(), true)?;
        a.policy_support = false;
        if saved.actions != legal.iter().map(ToString::to_string).collect::<Vec<_>>()
            || a.state != model.state_features(&position)
        {
            return Err("saved/native feature alignment mismatch".into());
        }
        if a.policy.iter().filter(|p| **p > 0.).count() != 1
            || a.policy.iter().zip(&support).any(|(p, s)| *p > 0. && !*s)
        {
            return Err("expected historical singleton winning target".into());
        }
        // Hold proven Q and direct value labels identical in A/B/C. This isolates
        // policy target/objective changes; it is not an old-runtime comparison.
        if a.action_values.is_empty() {
            a.action_values = vec![None; legal.len()];
        }
        for (q, s) in a.action_values.iter_mut().zip(&support) {
            if *s {
                *q = Some(1.);
            }
        }
        let n = support.iter().filter(|s| **s).count();
        let mut b = a.clone();
        b.policy = support
            .iter()
            .map(|s| if *s { 1. / n as f64 } else { 0. })
            .collect();
        let mut c = b.clone();
        c.policy_support = true;
        for ex in [&a, &b, &c] {
            ex.validate()?;
        }
        rows.push(Row {
            id: sample["position"].as_u64().ok_or("position")? as usize,
            variants: [a, b, c],
            support,
            successor_q,
            source: teaching["source_group"].as_str().unwrap_or("").into(),
        });
    }
    if rows.len() != 54 {
        return Err(format!("expected 54 primary winning proofs, got {}", rows.len()).into());
    }
    let loading_seconds = loading.elapsed().as_secs_f64();
    let before = measure(&model, &rows)?;
    let mut gradient_audit = vec![];
    let mut exact_singletons = 0;
    for (index, row) in rows.iter().enumerate() {
        let p = priors(&model, &row.variants[0])?;
        let mut gradients = vec![];
        let mut losses = vec![];
        for variant in 0..3 {
            let (loss, g) =
                model.loss_gradient_loop_v3_reusing(&policy_example(row, variant), Vec::new())?;
            if g.iter()
                .enumerate()
                .any(|(i, g)| value_parameter(i) && *g != 0.)
            {
                return Err("policy-only gradient touched value".into());
            }
            losses.push(loss.policy);
            gradients.push(g);
        }
        if row.support.iter().filter(|s| **s).count() == 1 {
            if losses[0].to_bits() != losses[2].to_bits()
                || gradients[0]
                    .iter()
                    .zip(&gradients[2])
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("singleton differs from CE".into());
            }
            exact_singletons += 1;
        }
        let norms = gradients
            .iter()
            .map(|g| g.iter().map(|x| x * x).sum::<f64>().sqrt())
            .collect::<Vec<_>>();
        let ce_vs_set = gradients[0]
            .iter()
            .zip(&gradients[2])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / (norms[0] * norms[2]).max(1e-300);
        let suppressed = (0..2)
            .map(|v| {
                p.iter()
                    .zip(&row.variants[v].policy)
                    .zip(&row.support)
                    .filter(|((p, t), s)| **s && **p > **t)
                    .count()
            })
            .collect::<Vec<_>>();
        gradient_audit.push(json!({"position":row.id,"source":row.source,"initially_raw_correct":before[index]["raw_correct"],"support":row.support.iter().filter(|s|**s).count(),"actions":p.len(),"policy_losses":losses,"gradient_norms":norms,"ce_vs_set_gradient_cosine":ce_vs_set,"winning_logits_suppressed_A_B":suppressed}));
    }
    for v in 0..3 {
        timing(&model, &rows, v)?;
    }
    let mut timings = vec![];
    for alternative in [1, 2] {
        for repeat in 0..4 {
            for variant in [0, alternative, alternative, 0] {
                timings.push(json!({"alternative":alternative,"repeat":repeat,"variant":variant,"seconds":timing(&model,&rows,variant)?}));
            }
        }
    }
    let mut trials = vec![];
    for variant in 0..3 {
        let mut current = model.clone();
        let mut history = vec![];
        for step in 1..=updates {
            let mut gradient = vec![0.; current.parameters().len()];
            let mut loss = 0.;
            for row in &rows {
                let (l, g) = current
                    .loss_gradient_loop_v3_reusing(&policy_example(row, variant), Vec::new())?;
                loss += l.policy;
                for (sum, g) in gradient.iter_mut().zip(g) {
                    *sum += g;
                }
            }
            for g in &mut gradient {
                *g /= rows.len() as f64;
            }
            let norm = gradient.iter().map(|g| g * g).sum::<f64>().sqrt();
            let clip = (10. / norm.max(1e-300)).min(1.);
            let mut next = MicroModel::from_parameters(
                current
                    .parameters()
                    .iter()
                    .zip(&gradient)
                    .map(|(w, g)| w - rate * clip * g)
                    .collect(),
            )?;
            if let Some(bank) = model.sequence_memory() {
                next = next.with_sequence_memory_owned(bank.clone());
            }
            if next
                .parameters()
                .iter()
                .zip(model.parameters())
                .enumerate()
                .any(|(i, (a, b))| value_parameter(i) && a.to_bits() != b.to_bits())
            {
                return Err("isolated update changed value parameters".into());
            }
            current = next;
            let scores = measure(&current, &rows)?;
            history.push(json!({"step":step,"mean_training_objective_before":loss/rows.len() as f64,"gradient_norm":norm,"clip":clip,"scores":scores,"summary":summary(&before,&scores)}));
        }
        trials.push(json!({"variant":variant,"history":history,"weights_written":false}));
    }
    if hash(&fs::read(&args[1])?) != hash(&model_bytes)
        || hash(&fs::read(&args[2])?) != hash(&provenance_bytes)
    {
        return Err("input changed during probe".into());
    }
    let report = json!({"model":args[1],"model_sha256":hash(&model_bytes),"provenance":args[2],"provenance_sha256":hash(&provenance_bytes),"targets":inputs,
        "rows":rows.len(),"known_winning_actions":rows.iter().map(|r|r.support.iter().filter(|s|**s).count()).sum::<usize>(),"exact_singletons":exact_singletons,
        "variants":["A historical singleton CE","B expanded verified support, uniform CE","C expanded verified support, set-mass loss"],
        "loading_and_rule_preparation_seconds_excluded":loading_seconds,"initial":before,"gradient_audit":gradient_audit,"timings_abba":timings,"trials":trials,"rate":rate,"batch_updates":updates,
        "scope":"54 training positions, no heldout strength or retention after unrelated learning; policy-only equal full batches with value frozen and shared norm clip10; timing uses identical direct value and proved Q labels; coupled ranks reuse frozen successor values, not full MCTS","campaign_started":false,"model_weights_written":false});
    fs::write(&args[3], serde_json::to_vec_pretty(&report)?)?;
    println!(
        "{}",
        json!({"rows":rows.len(),"exact_singletons":exact_singletons,"final":trials.iter().map(|v|v["history"].as_array().unwrap().last().unwrap()["summary"].clone()).collect::<Vec<_>>()})
    );
    Ok(())
}
