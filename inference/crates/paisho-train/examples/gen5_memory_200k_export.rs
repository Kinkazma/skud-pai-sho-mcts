//! Read-only native export for the isolated ~200k neural-memory experiment.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Write};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

fn export(m: &MicroModel, p: &Position, excluded: u64) -> Result<Value> {
    let state = m.state_features(p);
    let legal = legal_actions(p);
    let actions: Vec<_> = legal.iter().map(|a| micro_action_features(p, *a)).collect();
    let embedding = m.embed(&state);
    let base = m.memory_priors(
        &state,
        &actions,
        &micro_softmax(&MicroModel::logits(&embedding, &actions))?,
        excluded,
    )?;
    let mut memory = [0.; 32];
    if let Some(ctx) = m.memory_context(&state, excluded)? {
        for (pattern, w) in ctx
            .patterns
            .iter()
            .zip(&m.parameters()[MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS])
        {
            for k in 0..32 {
                memory[k] += pattern[k] * w / 8.;
            }
        }
    }
    let mut next_values = vec![];
    let mut terminal = vec![];
    for a in &legal {
        let mut n = p.clone();
        n.apply(*a)?;
        let z = match n.outcome() {
            GameOutcome::Win(w) => Some(if w == p.to_move() { 1. } else { -1. }),
            GameOutcome::Draw => Some(0.),
            _ => None,
        };
        let sign = if n.to_move() == p.to_move() { 1. } else { -1. };
        next_values.push(z.unwrap_or_else(|| sign * m.embed(&m.state_features(&n)).value));
        terminal.push(z);
    }
    Ok(
        json!({"state": state, "actions": actions, "legal": legal.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "memory": memory, "base_prior": base, "base_value": embedding.value,
        "next_values": next_values, "terminal": terminal, "excluded_source": excluded}),
    )
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("FLOW_MANIFEST TEACHER_SELECTION OUTPUT_JSONL".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let spec: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let art: MicroArtifact =
        serde_json::from_slice(&fs::read(spec["models"][4]["path"].as_str().unwrap())?)?;
    assert_eq!(art.identity(), spec["models"][4]["identity"]);
    let m = art.model()?;
    let mut out = std::io::BufWriter::new(fs::File::create(&args[3])?);
    let mut count = 0;
    for s in spec["positions"].as_array().unwrap() {
        if s["kind"] != "certificate" {
            continue;
        }
        let bytes = fs::read(s["path"].as_str().unwrap())?;
        assert_eq!(hash(&bytes), s["sha256"]);
        let v: Value = serde_json::from_slice(&bytes)?;
        let Some(group) = v["human_source"].as_str() else {
            continue;
        };
        let r: GameRecord = v["prefix"].as_str().unwrap().parse()?;
        let p = r.replay()?;
        let cert: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
        cert.verify(&p)?;
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        if cert.outcome != sign {
            continue;
        }
        let mut row = export(&m, &p, 0)?;
        let legal = legal_actions(&p);
        let mut win: Vec<bool> = row["terminal"]
            .as_array()
            .unwrap()
            .iter()
            .map(|z| z.as_f64() == Some(1.))
            .collect();
        for (a, c) in cert.children {
            if c.outcome == sign {
                let a: Action = a.parse()?;
                win[legal.iter().position(|x| *x == a).unwrap()] = true;
            }
        }
        assert!(win.iter().any(|x| *x));
        row["kind"] = json!("proof");
        row["winning"] = json!(win);
        row["group"] = json!(group);
        row["source"] = s["path"].clone();
        row["source_sha256"] = s["sha256"].clone();
        serde_json::to_writer(&mut out, &row)?;
        writeln!(out)?;
        count += 1;
    }
    eprintln!("{} regulatory winning proofs verified and exported", count);
    let selection: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    for item in selection.as_array().unwrap() {
        let path = item["path"].as_str().unwrap();
        let compressed = fs::read(path)?;
        assert_eq!(hash(&compressed), item["sha256"]);
        let targets: Vec<Value> =
            serde_json::from_reader(flate2::read::GzDecoder::new(compressed.as_slice()))?;
        let target = &targets[item["row"].as_u64().unwrap() as usize];
        let record_bytes = fs::read(path.replace(".targets.json.gz", ".psr"))?;
        assert_eq!(hash(&record_bytes), item["psr_sha256"]);
        let r: GameRecord = std::str::from_utf8(&record_bytes)?.parse()?;
        let mut p = r.initial_position();
        for &a in &r.actions()[..target["decision"].as_u64().unwrap() as usize - 1] {
            p.apply(a)?;
        }
        let excluded = sequence_source(&format!(
            "{}/{}",
            target["source_run"].as_str().unwrap(),
            target["game_id"].as_str().unwrap()
        ));
        let mut row = export(&m, &p, excluded)?;
        assert_eq!(row["state"], target["state"]);
        assert_eq!(row["actions"], target["action_features"]);
        assert_eq!(row["legal"], target["actions"]);
        row["kind"] = json!("estimate");
        row["group"] = item["group"].clone();
        row["source"] = json!(path);
        row["source_sha256"] = item["sha256"].clone();
        row["decision"] = target["decision"].clone();
        row["evidence"] = target["evidence"].clone();
        serde_json::to_writer(&mut out, &row)?;
        writeln!(out)?;
    }
    out.flush()?;
    eprintln!(
        "{} full-search teacher states legally replayed and recomputed",
        selection.as_array().unwrap().len()
    );
    Ok(())
}
