//! Explicit isolated V7 migration and legally regenerated R1 teaching seeds.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::{BufRead, BufReader},
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("ACTOR CORPUS NEW_OUTPUT".into());
    }
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    let start = Instant::now();
    let bytes = fs::read(&args[1])?;
    let artifact: MicroArtifact = serde_json::from_slice(&bytes)?;
    let old = artifact.model()?;
    let model = old.with_relational(20260926);
    let loading = start.elapsed().as_secs_f64();
    let corpus = Path::new(&args[2]);
    let mut examples = vec![];
    let mut sources = vec![];
    let mut groups = BTreeSet::new();
    let mut compared = 0;
    let mut populations = [0usize; 10];
    for line in BufReader::new(fs::File::open(corpus.join("rows.jsonl"))?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        if row["split"] != "train" || row["actions"].as_array().unwrap().len() > 512 {
            continue;
        }
        let psr = fs::read(corpus.join(row["psr"].as_str().unwrap()))?;
        if hash(&psr) != row["psr_sha256"] {
            return Err("PSR input changed".into());
        }
        let record: GameRecord = std::str::from_utf8(&psr)?.parse()?;
        if record.rules() != RuleProfileId::SkudPaiShoGen5V1 {
            return Err("non-Gen5 seed corpus".into());
        }
        let p = record.replay()?;
        let legal = legal_actions(&p);
        let state = model.state_features(&p);
        let features: Vec<_> = legal
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect();
        if compared < 64 {
            let a = old.policy_value_priors(&state, &features, 0)?;
            let b = model.policy_value_priors(&state, &features, 0)?;
            if a.0.to_bits() != b.0.to_bits()
                || a.1
                    .iter()
                    .zip(&b.1)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err("nonneutral migration".into());
            }
            compared += 1;
        }
        let before = MicroRelations::extract(&p, p.to_move());
        let mut targets = vec![];
        let mut winning = vec![];
        let mut mask = [false; 10];
        for &a in &legal {
            let mut q = p.clone();
            q.apply(a)?;
            let threat = micro_immediate_threat(&q, p.to_move(), usize::MAX)?;
            let target = MicroStructuredTarget::from_successor(&p, a, &q, &before, &threat);
            target.validate()?;
            for j in 0..10 {
                mask[j] |= target.events[j] == Some(true);
            }
            winning.push(q.outcome() == GameOutcome::Win(p.to_move()));
            targets.push(Some(target));
        }
        let unique = groups.insert(row["group"].as_str().unwrap().to_owned());
        // Fixed coverage selection: first legal source of each motif, then one per source.
        let rare = (0..10).any(|j| mask[j] && populations[j] < 8);
        if (!unique && !rare) || examples.len() >= 128 {
            if examples.len() >= 128 && compared == 64 {
                break;
            }
            continue;
        }
        for j in 0..10 {
            populations[j] += usize::from(mask[j]);
        }
        let n = winning.iter().filter(|v| **v).count();
        let proven = n > 0;
        let policy = if proven {
            winning
                .iter()
                .map(|v| if *v { 1. / n as f64 } else { 0. })
                .collect::<Vec<_>>()
        } else {
            vec![1. / legal.len() as f64; legal.len()]
        };
        let sample: SavedMicroExample = serde_json::from_value(json!({
            "structured":targets,"rules":record.rules().as_str(),"source_run":"r1-regenerated-seeds",
            "game_id":row["key"],"decision":record.actions().len()+1,"collector":hash(&bytes),"budget":0,
            "inherited_visits":0,"new_visits":[],"policy_raw_visits":[],"policy_pruned_visits":[],
            "tactical":if proven{json!({"schema":"paisho-mcts-proof-v1","root_value":1,
                "action_values":winning.iter().map(|v|if *v{Some(1)}else{None}).collect::<Vec<_>>(),
                "network_value":model.value(&state),"network_best_action":null})}else{Value::Null},"correction_priority":false,"actions":legal.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "state":state,"action_features":features,"policy":policy,"value":if proven{1.}else{0.},
            "policy_weight":if proven{1.}else{0.},"reason":"regulatory-consequence-corpus",
            "evidence":{"policy_support":proven,
                "observed_value":null,"observed_psr":null,"estimated_value":null,"value_weight":if proven{1.}else{0.},
                "policy_source":if proven{"verified-regulatory-win"}else{"regulatory-consequences-only"},
                "completed_action_values":[],"action_value_visits":[],"target_prior":[],"excluded_actions":[],
                "player":if p.to_move()==Player::Host{"H"}else{"G"},"actor":hash(&bytes)}
        }))?;
        sample.example_for_rules_with_trusted_q(record.rules(), true)?;
        sources.push(json!({"key":row["key"],"group":row["group"],"psr":corpus.join(row["psr"].as_str().unwrap()),"sha256":row["psr_sha256"]}));
        examples.push(sample);
    }
    let migrated = MicroArtifact::new(
        &model,
        artifact.updates,
        json!({"kind":"explicit-neutral-relational-migration","parent":hash(&bytes),"diagnostic_only":true}),
    );
    migrated.save(&out.join("actor-v7.json"))?;
    fs::write(
        out.join("seed-examples.json"),
        serde_json::to_vec(&examples)?,
    )?;
    fs::write(
        out.join("seed-sources.json"),
        serde_json::to_vec_pretty(&sources)?,
    )?;
    let result = json!({"neutral_roots":compared,"parameters":model.parameters().len(),"added":MICRO_RELATIONAL_WEIGHTS,
        "old_sha256":hash(&bytes),"new_sha256":migrated.identity(),"updates":artifact.updates,
        "examples":examples.len(),"event_root_populations":populations,"loading_seconds":loading,"seconds":start.elapsed().as_secs_f64()});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{result}");
    Ok(())
}
