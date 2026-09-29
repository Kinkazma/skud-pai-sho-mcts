//! Independent replay/feature/proof verification of one completed repair trial.
use paisho_ai::{micro_action_features, micro_spatial_state_features, MicroProofCertificate};
use paisho_core::{legal_actions, GameRecord};
use paisho_train::micro_learning::SavedMicroExample;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn same(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("gen5_repair_verify TRIAL ARCHIVE [ALIAS_DIRECTORY]".into());
    }
    let mut paths: Vec<_> = fs::read_dir(Path::new(&args[1]).join("games"))?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<_>>()?;
    paths.sort();
    let mut records = 0;
    let mut targets = 0;
    let mut terminals = 0;
    let mut roots = 0;
    for path in paths
        .into_iter()
        .filter(|p| p.extension().is_some_and(|s| s == "psr"))
    {
        let bytes = fs::read(&path)?;
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        let receipt: Value = serde_json::from_slice(&fs::read(path.with_extension("json"))?)?;
        if receipt["psr_sha256"] != hash(&bytes) {
            return Err("PSR binding".into());
        }
        let mut position = record.initial_position();
        let mut states = vec![position.clone()];
        for a in record.actions() {
            position.apply(*a)?;
            states.push(position.clone());
        }
        if receipt["outcome"] != format!("{:?}", position.outcome()) {
            return Err("outcome mismatch".into());
        }
        terminals += usize::from(position.outcome() != paisho_core::GameOutcome::Ongoing);
        records += 1;
        let target_path =
            path.with_file_name(receipt["targets_file"].as_str().ok_or("target filename")?);
        let data = fs::read(target_path)?;
        if receipt["targets_sha256"] != hash(&data) {
            return Err("target binding".into());
        }
        let mut raw = vec![];
        flate2::read::GzDecoder::new(data.as_slice()).read_to_end(&mut raw)?;
        let saved: Vec<SavedMicroExample> = serde_json::from_slice(&raw)?;
        if saved.len()
            != receipt["eligible_examples"]
                .as_u64()
                .ok_or("target count")? as usize
        {
            return Err("target count mismatch".into());
        }
        for s in saved {
            let p = states
                .get(s.decision.checked_sub(1).ok_or("decision zero")?)
                .ok_or("decision range")?;
            s.example_for_rules(record.rules())?;
            if !same(&s.state, &micro_spatial_state_features(p)) {
                return Err("spatial features differ".into());
            }
            if s.reason == "repetition-training-loss" {
                return Err("fabricated cycle defeat".into());
            }
            if !s.actions.is_empty() {
                let legal = legal_actions(p);
                if s.actions != legal.iter().map(ToString::to_string).collect::<Vec<_>>() {
                    return Err("legal order".into());
                }
                for (a, x) in legal.iter().zip(&s.action_features) {
                    if !same(x, &micro_action_features(p, *a)) {
                        return Err("action feature mismatch".into());
                    }
                }
            }
            roots += usize::from(s.tactical.as_ref().is_some_and(|t| t.root_value.is_some()));
            targets += 1;
        }
    }
    let mut proofs = 0;
    let dir = Path::new(&args[2]).join("proofs");
    if dir.exists() {
        for p in fs::read_dir(dir)? {
            let p = p?.path();
            if p.extension().map_or(true, |e| e != "json") {
                continue;
            }
            let v: Value = serde_json::from_slice(&fs::read(&p)?)?;
            let r: GameRecord = v["prefix"].as_str().ok_or("prefix")?.parse()?;
            let c: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
            c.verify(&r.replay()?)?;
            proofs += 1;
        }
    }
    let mut distinguished=0;
    if let Some(dir)=args.get(3) {
        let manifest:Value=serde_json::from_slice(&fs::read(Path::new(dir).join("aliases.json"))?)?;
        for w in manifest["witnesses"].as_array().ok_or("alias witnesses")? {
            let load=|key:&str|->Result<paisho_core::Position> {
                let r:GameRecord=fs::read_to_string(Path::new(dir).join(w[key].as_str().ok_or("alias path")?))?.parse()?;
                Ok(r.replay()?)
            };
            let a=load("prefix_a")?;let b=load("prefix_b")?;
            if paisho_ai::micro_state_features(&a)!=paisho_ai::micro_state_features(&b) || micro_spatial_state_features(&a)==micro_spatial_state_features(&b) {return Err("alias representation regression".into());}
            for row in w["same_action_input_different_immediate_result"].as_array().ok_or("alias actions")? {
                let action:paisho_core::Action=row["action"].as_str().ok_or("action")?.parse()?;
                if micro_action_features(&a,action)!=micro_action_features(&b,action) {return Err("old action alias differs".into());}
                let mut qa=a.clone();let mut qb=b.clone();qa.apply(action)?;qb.apply(action)?;
                let wa=qa.outcome()==paisho_core::GameOutcome::Win(a.to_move());let wb=qb.outcome()==paisho_core::GameOutcome::Win(b.to_move());
                if wa==wb || row["wins_a"]!=wa || row["wins_b"]!=wb {return Err("alias tactical witness differs".into());}
            }
            distinguished+=1;
        }
    }
    println!(
        "{}",
        json!({"records":records,"terminal_records":terminals,"targets":targets,"proved_root_targets":roots,"independently_verified_certificates":proofs,"exact_features":true,"verified_tactical_aliases_now_distinguishable":distinguished})
    );
    Ok(())
}
