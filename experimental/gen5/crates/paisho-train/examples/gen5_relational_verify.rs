//! Independent rule-only reread of a post-trial union, at every saved learner/actor.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::Path, sync::Arc, time::Instant};
#[path = "gen5_structured_corpus/facts.rs"]
mod facts;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn best(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
struct Root {
    position: Position,
    origins: Vec<String>,
}
fn auxiliary(model: &MicroModel, examples: &[MicroExample]) -> Result<Value> {
    if !model.has_relational() {
        return Ok(Value::Null);
    }
    let mut squared = [0.; 10];
    let mut n = 0;
    let mut confusion = [[0usize; 4]; 10];
    let mut brier = [0.; 10];
    for e in examples {
        let outputs = model
            .structured_predictions(&e.state, &e.actions)?
            .ok_or("missing auxiliary reader")?;
        for (y, target) in outputs.iter().zip(&e.structured) {
            if let Some(t) = target {
                n += 1;
                for j in 0..10 {
                    squared[j] += (y[j] - t.counts[j]).powi(2);
                    if let Some(label) = t.events[j] {
                        let p = y[10 + j];
                        brier[j] += (p - usize::from(label) as f64).powi(2);
                        let class = match (label, p >= 0.5) {
                            (true, true) => 0,
                            (false, true) => 1,
                            (false, false) => 2,
                            (true, false) => 3,
                        };
                        confusion[j][class] += 1;
                    }
                }
            }
        }
    }
    Ok(
        json!({"count_mse":squared.map(|v|v/n as f64),"confusion_tp_fp_tn_fn":confusion,
        "event_brier":std::array::from_fn::<_,10,_>(|j|brier[j]/confusion[j].iter().sum::<usize>().max(1) as f64),
        "actions":n,"scope":"128 consumed corpus seeds; training fit and retention, not held-out improvement"}),
    )
}
fn add(roots: &mut BTreeMap<String, Root>, p: Position, origin: String) {
    roots
        .entry(facts::position_hash(&p))
        .and_modify(|r| r.origins.push(origin.clone()))
        .or_insert(Root {
            position: p,
            origins: vec![origin],
        });
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("BASELINE_RUN V7_RUN SEED_SOURCES NEW_OUTPUT".into());
    }
    let out = Path::new(&args[4]);
    fs::create_dir(out)?;
    let start = Instant::now();
    let mut roots = BTreeMap::new();
    let seeds: Vec<Value> = serde_json::from_slice(&fs::read(&args[3])?)?;
    let saved_seeds: Vec<SavedMicroExample> = serde_json::from_slice(&fs::read(
        Path::new(&args[3]).with_file_name("seed-examples.json"),
    )?)?;
    let examples = saved_seeds
        .iter()
        .map(|s| s.example_for_rules_with_trusted_q(RuleProfileId::SkudPaiShoGen5V1, true))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for s in seeds {
        let bytes = fs::read(s["psr"].as_str().ok_or("psr")?)?;
        if hash(&bytes) != s["sha256"] {
            return Err("seed source changed".into());
        }
        let r: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        add(&mut roots, r.replay()?, "seed-corpus".into());
    }
    let mut paths = vec![];
    for (label, directory) in [("baseline", &args[1]), ("v7", &args[2])] {
        let dir = Path::new(directory);
        let mut files: Vec<_> = fs::read_dir(dir)?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<_>>()?;
        files.sort();
        for p in files {
            let name = p.file_name().unwrap().to_string_lossy();
            if name.starts_with("actor-") || name.starts_with("learner-") {
                if name.ends_with(".json") {
                    paths.push((label.to_string(), p.clone()));
                }
            }
            if !name.starts_with("cycle-") || !p.join("report.json").exists() {
                continue;
            }
            let report: Value = serde_json::from_slice(&fs::read(p.join("report.json"))?)?;
            for receipt in report["receipts"].as_array().ok_or("receipts")? {
                let psr = fs::read(receipt["psr"].as_str().ok_or("PSR")?)?;
                if hash(&psr) != receipt["psr_sha256"] {
                    return Err("receipt PSR changed".into());
                }
                let record: GameRecord = std::str::from_utf8(&psr)?.parse()?;
                let targets = fs::read(receipt["targets"].as_str().ok_or("targets")?)?;
                if hash(&targets) != receipt["targets_sha256"] {
                    return Err("receipt targets changed".into());
                }
                let mut decoded = vec![];
                flate2::read::GzDecoder::new(targets.as_slice()).read_to_end(&mut decoded)?;
                let saved: Vec<SavedMicroExample> = serde_json::from_slice(&decoded)?;
                let mut p = record.initial_position();
                let mut next = 0;
                for s in saved {
                    while next < s.decision - 1 {
                        p.apply(record.actions()[next])?;
                        next += 1;
                    }
                    if p.outcome() != GameOutcome::Ongoing {
                        return Err("terminal teaching root".into());
                    }
                    add(
                        &mut roots,
                        p.clone(),
                        format!("{label}:{name}:{}", receipt["id"]),
                    );
                }
            }
        }
    }
    // One independent full legal-action terminal census. No learned labels decide correctness.
    let mut data = vec![];
    for (key, r) in roots {
        let legal = legal_actions(&r.position);
        let valid: Vec<_> = legal
            .iter()
            .map(|&a| {
                let mut q = r.position.clone();
                q.apply(a).unwrap();
                q.outcome() == GameOutcome::Win(r.position.to_move())
            })
            .collect();
        if !valid.iter().any(|v| *v) {
            continue;
        }
        data.push((key, r, legal, valid));
    }
    fs::write(out.join("roots.json"),serde_json::to_vec_pretty(&data.iter().map(|(key,r,_,v)|json!({"key":key,"origins":r.origins,"winning_actions":v.iter().filter(|v|**v).count()})).collect::<Vec<_>>())?)?;
    let mut bank: Option<Arc<SequenceBank>> = None;
    let mut readings = vec![];
    for (label, path) in paths {
        let t = Instant::now();
        let bytes = fs::read(&path)?;
        let a: MicroArtifact = serde_json::from_slice(&bytes)?;
        let model = if let Some(bank) = &bank {
            MicroModel::from_parameters(a.parameters.clone())?
                .with_sequence_memory_owned(bank.clone())
        } else {
            let model = a.model()?;
            bank = model.sequence_memory().cloned();
            model
        };
        if model.schema() != a.schema
            || model.feature_schema() != a.feature_schema
            || model
                .sequence_memory()
                .map(|b| serde_json::to_value(&b.spec).unwrap())
                != a.sequence_memory
                    .as_ref()
                    .map(|s| serde_json::to_value(s).unwrap())
        {
            return Err("model schema/bank changed".into());
        }
        let mut choices = vec![];
        for (key, r, legal, valid) in &data {
            let state = model.state_features(&r.position);
            let actions: Vec<_> = legal
                .iter()
                .map(|&a| micro_action_features(&r.position, a))
                .collect();
            let (value, prior) = model.policy_value_priors(&state, &actions, 0)?;
            choices.push(json!({"key":key,"raw":valid[best(&prior)],"winning_mass":prior.iter().zip(valid).filter(|(_,v)|**v).map(|(p,_)|p).sum::<f64>(),"value":value}));
        }
        let auxiliary = auxiliary(&model, &examples)?;
        let read = json!({"arm":label,"path":path,"identity":a.identity(),"sha256":hash(&bytes),"schema":model.schema(),"auxiliary":auxiliary,
            "raw_wins":choices.iter().filter(|c|c["raw"]==true).count(),"choices":choices,"seconds":t.elapsed().as_secs_f64()});
        println!("{} {} {}", label, path.display(), read["raw_wins"]);
        readings.push(read);
        fs::write(
            out.join("readings.json"),
            serde_json::to_vec_pretty(&readings)?,
        )?;
    }
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&json!({"winning_roots":data.len(),"models":readings.len(),
        "seconds":start.elapsed().as_secs_f64(),"scope":"post-trial union of seed and consumed fresh roots, not held-out/general strength","complete":true}))?,
    )?;
    Ok(())
}
