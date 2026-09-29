//! Frozen R1 replays -> identical pre-action features for R2 architecture trials.
//! No SGD, search, publication or production writes. One actor/bank load.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Write, path::Path, time::Instant};
#[path = "gen5_structured_corpus/facts.rs"]
mod facts;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("PLAN NEW_OUTPUT_DIRECTORY".into());
    }
    let plan_bytes = fs::read(&args[1])?;
    let plan: Value = serde_json::from_slice(&plan_bytes)?;
    let out = Path::new(&args[2]);
    fs::create_dir(out)?;
    let bytes = fs::read(plan["actor"].as_str().ok_or("actor")?)?;
    if hash(&bytes) != plan["actor_sha256"] {
        return Err("actor hash changed".into());
    }
    let artifact: MicroArtifact = serde_json::from_slice(&bytes)?;
    let start = Instant::now();
    let model = artifact.model()?;
    let loading = start.elapsed().as_secs_f64();
    let mut rows: BTreeMap<String, Value> = BTreeMap::new();
    let encoder = MicroGraphEncoder::seeded(17);
    let mut parity = vec![];
    let mut graph_positions = vec![];
    for info in plan["rows"].as_array().ok_or("rows")? {
        let b = fs::read(info["psr"].as_str().ok_or("psr")?)?;
        if hash(&b) != info["sha256"] {
            return Err("PSR hash changed".into());
        }
        let r: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let p = r.replay()?;
        if p.outcome() != GameOutcome::Ongoing {
            return Err("terminal input root".into());
        }
        let task = info["task"].as_str().ok_or("task")?;
        let key = format!("{task}:{}", facts::position_hash(&p));
        if let Some(old) = rows.get_mut(&key) {
            if old["group"] != info["group"] || old["held_out"] != info["held_out"] {
                return Err("duplicate crosses groups".into());
            }
            if task == "motif" {
                let slot = info["slot"].as_u64().ok_or("slot")? as usize;
                if !old["targets"][slot].is_null() && old["targets"][slot] != info["target"] {
                    return Err("conflicting motif labels".into());
                }
                old["targets"][slot] = info["target"].clone();
            }
            continue;
        }
        let graph = MicroPieceGraph::extract(&p, p.to_move());
        let state = model.state_features(&p);
        let mut row = json!({"key":key,"task":task,"group":info["group"],"held_out":info["held_out"],
            "psr_sha256":hash(&b),"state":state,"nodes":graph.nodes.iter().map(|n|n.features.to_vec()).collect::<Vec<_>>(),
            "edges":graph.messages.iter().map(|e|json!({"source":e.source,"destination":e.destination,"features":e.features})).collect::<Vec<_>>()});
        if task == "policy" {
            let actions = legal_actions(&p);
            let features: Vec<_> = actions
                .iter()
                .map(|&a| micro_action_features(&p, a))
                .collect();
            let embedding = model.embed(&state);
            let prior = model.memory_priors(
                &state,
                &features,
                &micro_softmax(&MicroModel::logits(&embedding, &features))?,
                0,
            )?;
            let mut winning = vec![];
            for &a in &actions {
                let mut q = p.clone();
                q.apply(a)?;
                winning.push(q.outcome() == GameOutcome::Win(p.to_move()));
            }
            let mut actual: Vec<_> = actions
                .iter()
                .zip(&winning)
                .filter(|(_, w)| **w)
                .map(|(a, _)| a.to_string())
                .collect();
            actual.sort();
            let mut expected: Vec<_> = info["winning_actions"]
                .as_array()
                .ok_or("winning set")?
                .iter()
                .map(|x| x.as_str().unwrap().to_owned())
                .collect();
            expected.sort();
            if actual != expected || actual.is_empty() {
                return Err("complete winning set changed".into());
            }
            row["actions"] = json!(features.iter().map(|x| x.to_vec()).collect::<Vec<_>>());
            row["action_names"] =
                json!(actions.iter().map(ToString::to_string).collect::<Vec<_>>());
            row["prior"] = json!(prior);
            row["winning"] = json!(winning);
        } else {
            let slot = info["slot"].as_u64().ok_or("slot")? as usize;
            let owner = if slot % 2 == 0 {
                p.to_move()
            } else {
                p.to_move().opponent()
            };
            let expected = if slot < 2 {
                HarmonyCycleGeometry::OffCentre
            } else {
                HarmonyCycleGeometry::TouchingCentre
            };
            let rel = MicroRelations::extract(&p, p.to_move());
            let label = usize::from(
                rel.cycles
                    .iter()
                    .any(|c| c.owner == owner && c.geometry == expected),
            );
            if info["target"] != label {
                return Err("motif label changed".into());
            }
            let mut targets = vec![Value::Null; 4];
            targets[slot] = json!(label);
            row["targets"] = json!(targets);
        }
        if parity.len() < 4 {
            let dy = std::array::from_fn(|i| (i as f64 - 11.) / 17.);
            let node_dy: Vec<[f64; 16]> = (0..graph.nodes.len())
                .map(|i| std::array::from_fn(|j| ((i + 3 * j) % 11) as f64 / 19.))
                .collect();
            for mode in [
                MicroGraphPropagation::IndependentPieces,
                MicroGraphPropagation::TwoHarmonyRounds,
            ] {
                parity.push(json!({"key":key,"mode":format!("{:?}",mode),"output_gradient":dy,"node_gradients":node_dy,
                    "pooled":encoder.encode(&graph,mode),"nodes":encoder.encode_nodes(&graph,mode),
                    "gradient":encoder.gradient_with_nodes(&graph,mode,&dy,&node_dy)?}));
            }
        }
        graph_positions.push(p);
        rows.insert(key, row);
    }
    let mut writer = std::io::BufWriter::new(fs::File::create(out.join("rows.jsonl"))?);
    for row in rows.values() {
        serde_json::to_writer(&mut writer, row)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    fs::write(
        out.join("parity.json"),
        serde_json::to_vec_pretty(&json!({"weights":encoder.parameters(),"cases":parity}))?,
    )?;
    let mut cost = vec![];
    for (leg, mode) in [
        MicroGraphPropagation::IndependentPieces,
        MicroGraphPropagation::TwoHarmonyRounds,
        MicroGraphPropagation::TwoHarmonyRounds,
        MicroGraphPropagation::IndependentPieces,
    ]
    .into_iter()
    .cycle()
    .take(12)
    .enumerate()
    {
        let t = Instant::now();
        let mut checksum = 0.;
        for p in &graph_positions {
            let graph = MicroPieceGraph::extract(p, p.to_move());
            checksum += encoder.encode(&graph, mode).iter().sum::<f64>();
        }
        cost.push(json!({"leg":leg,"mode":format!("{mode:?}"),"positions":graph_positions.len(),"seconds":t.elapsed().as_secs_f64(),"checksum":checksum}));
    }
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&json!({"schema":MICRO_PIECE_GRAPH_SCHEMA,
        "plan_sha256":hash(&plan_bytes),"actor_sha256":hash(&bytes),"actor_identity":artifact.identity(),"actor_updates":artifact.updates,
        "rows":rows.len(),"loading_seconds":loading,"elapsed_seconds":start.elapsed().as_secs_f64(),"graph_only_abba":cost,
        "rows_sha256":hash(&fs::read(out.join("rows.jsonl"))?),"no_search_or_training":true,"not_campaign_cost":true}))?,
    )?;
    println!(
        "{} unique task/position rows; actor/bank loaded once in {loading:.3}s",
        rows.len()
    );
    Ok(())
}
