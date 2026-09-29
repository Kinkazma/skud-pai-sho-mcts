//! Frozen diagnostic only: export real features/targets and independently check
//! immediate regulatory wins. Does not write models or alter a running trainer.
use paisho_ai::{
    micro_action_features, micro_state_features, MicroExample, MicroMctsSession, MicroModel,
    MicroSearchOptions,
};
use paisho_core::{legal_actions, Action, GameOutcome, GameRecord};
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    io::{Read, Write},
    path::Path,
    sync::Arc,
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: micro_capacity_probe MANIFEST OUTPUT.jsonl".into());
    }
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let model_bytes = fs::read(manifest["model"].as_str().ok_or("model")?)?;
    if format!("{:x}", Sha256::digest(&model_bytes))
        != manifest["model_sha256"].as_str().ok_or("model hash")?
    {
        return Err("model hash mismatch".into());
    }
    let artifact = MicroArtifact::load(Path::new(manifest["model"].as_str().ok_or("model")?))?;
    let model = Arc::new(artifact.model()?);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let mut output = fs::File::create(&args[2])?;
    let started = Instant::now();
    let mut count = 0;
    for row in manifest["rows"].as_array().ok_or("rows")? {
        let bytes = fs::read(row["path"].as_str().ok_or("path")?)?;
        if format!("{:x}", Sha256::digest(&bytes)) != row["sha256"].as_str().ok_or("bundle hash")? {
            return Err("bundle hash mismatch".into());
        }
        let mut decoded = String::new();
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_string(&mut decoded)?;
        let b: Value = serde_json::from_str(&decoded)?;
        if b["schema"] != "paisho-gen5-durable-lessons-v1"
            || format!(
                "{:x}",
                Sha256::digest(b["psr"].as_str().ok_or("psr")?.as_bytes())
            ) != b["psr_sha256"].as_str().ok_or("psr hash")?
        {
            return Err("bundle identity mismatch".into());
        }
        let record: GameRecord = b["psr"].as_str().ok_or("psr")?.parse()?;
        if record.rules().as_str() != b["rules"].as_str().ok_or("rules")? {
            return Err("rules mismatch".into());
        }
        let lessons: Vec<_> = b["lessons"]
            .as_array()
            .ok_or("lessons")?
            .iter()
            .filter(|l| {
                l["policy_weight"].as_f64().unwrap_or(0.) > 0.
                    && l["policy"].as_array().is_some_and(|a| !a.is_empty())
            })
            .collect();
        let mut chosen: Vec<_> = (0..4.min(lessons.len()))
            .map(|i| i * (lessons.len() - 1) / (3.min(lessons.len() - 1).max(1)))
            .collect();
        chosen.sort();
        chosen.dedup();
        for index in chosen {
            let l = lessons[index];
            let decision = l["decision"].as_u64().ok_or("decision")? as usize;
            if decision == 0 || decision > record.actions().len() + 1 {
                return Err("decision bounds".into());
            }
            let mut position = record.initial_position();
            for action in &record.actions()[..decision - 1] {
                position.apply(*action)?;
            }
            if position.outcome() != GameOutcome::Ongoing {
                return Err("terminal example".into());
            }
            let actions = legal_actions(&position);
            let ids: HashMap<_, _> = actions
                .iter()
                .enumerate()
                .map(|(i, a)| (a.to_string(), i))
                .collect();
            let mut target = vec![0.; actions.len()];
            for pair in l["policy"].as_array().ok_or("policy")? {
                let a: Action = pair[0].as_str().ok_or("action")?.parse()?;
                target[*ids.get(&a.to_string()).ok_or("illegal target")?] =
                    pair[1].as_f64().ok_or("probability")?;
            }
            let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: 0,
                state: micro_state_features(&position).to_vec(),
                actions: actions
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect(),
                policy: target,
                value: l["value"].as_f64().ok_or("value")?,
                policy_weight: l["policy_weight"].as_f64().ok_or("weight")?,
            };
            ex.validate()?;
            let embedding = model.embed(&ex.state);
            let logits = MicroModel::logits(&embedding, &ex.actions);
            let policy_pick = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .ok_or("no actions")?
                .0;
            let mut winning = vec![];
            for (i, action) in actions.iter().enumerate() {
                let mut p = position.clone();
                p.apply(*action)?;
                if p.outcome() == GameOutcome::Win(position.to_move()) {
                    winning.push(i);
                }
            }
            let mut searches = vec![];
            for budget in [256, 512] {
                let t = Instant::now();
                let r = pool.install(|| {
                    MicroMctsSession::new(model.clone()).search_with_options(
                        &position,
                        budget,
                        None,
                        MicroSearchOptions {
                            proof_search: true,
                            ..Default::default()
                        },
                    )
                })?;
                if r.actions != actions {
                    return Err("search action ordering".into());
                }
                searches.push(json!({"budget":budget,"seconds":t.elapsed().as_secs_f64(),"selected":r.selected_index,"simulations":r.simulations,"policy":r.policy_target,"proven":r.proven_value,"inference_evaluations":r.inference_evaluations}));
            }
            let parity = if count < 3 {
                let (loss, gradient) = model.loss_gradient(&ex)?;
                let mut m = model.as_ref().clone();
                m.train_batch_inline(&[&ex], 0.001, 1e-5)?;
                json!({"value":embedding.value,"logits":logits,"loss_value":loss.value,"loss_policy":loss.policy,"gradient":gradient,"after_step":m.parameters()})
            } else {
                Value::Null
            };
            writeln!(
                output,
                "{}",
                json!({"source":row["source"],"group":row["group"],"bundle":row["sha256"],"decision":decision,"reason":l["reason"],"state":ex.state.as_slice(),"actions":ex.actions,"action_names":actions.iter().map(|a|a.to_string()).collect::<Vec<_>>(),"policy":ex.policy,"value":ex.value,"policy_weight":ex.policy_weight,"policy_pick":policy_pick,"immediate_wins":winning,"searches":searches,"native_parity":parity})
            )?;
            count += 1;
        }
        eprintln!(
            "exported {count} positions in {:.1}s",
            started.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
