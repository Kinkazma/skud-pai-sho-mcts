//! Read-only trace from a retained reanalysis target to its originating game.
use paisho_ai::*;
use paisho_core::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("FIFO_MANIFEST FIFO_AUDIT OUTPUT".into());
    }
    let m: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let a: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut rows = vec![];
    for x in a["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|x| x["lane"] == "Reanalysis")
    {
        let src = &m["rows"][x["source_index"].as_u64().unwrap() as usize]["source"];
        let path = src["path"]
            .as_str()
            .unwrap()
            .strip_suffix(".targets.json.gz")
            .ok_or("suffix")?;
        let receipt: Value = serde_json::from_slice(&fs::read(format!("{path}.json"))?)?;
        let b = fs::read(format!("{path}.psr"))?;
        assert_eq!(hash(&b), receipt["psr_sha256"]);
        let record: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let p = record.replay()?;
        assert_eq!(hash(record.to_string().as_bytes()), x["prefix"]);
        let state = micro_spatial_state_features(&p);
        let state_bytes: Vec<_> = state
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();
        assert_eq!(hash(&state_bytes), x["state"]);
        let pending = &receipt["case"]["before"]["pending_record"];
        if pending.is_null() {
            rows.push(json!({"row":x,"origin_missing":true}));
            continue;
        }
        let origin = pending[0].as_str().ok_or("origin")?;
        let b = fs::read(origin)?;
        assert_eq!(hash(&b), pending[1]);
        let original: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let final_p = original.replay()?;
        assert_eq!(original.setup(), record.setup());
        assert_eq!(original.rules(), record.rules());
        if !original.actions().starts_with(record.actions()) {
            assert_eq!(receipt["case"]["kind"], "archive-reanalysis");
            rows.push(json!({"row":x,"origin_missing":true,"reason":"archive-reanalysis case state is not the source trajectory"}));
            continue;
        }
        let z = match final_p.outcome() {
            GameOutcome::Win(w) => Some(if w == p.to_move() { 1. } else { -1. }),
            GameOutcome::Draw => Some(0.),
            _ => None,
        };
        rows.push(json!({"row":x,"origin":origin,"origin_sha256":hash(&b),"original_outcome":format!("{:?}",final_p.outcome()),"chooser":format!("{:?}",p.to_move()),"original_z":z,
            "prefix_hash_verified":true,"reanalysis_value":x["value"],"reason":x["reason"],"case_root":record.actions().len()==receipt["case"]["prefix_decisions"].as_u64().unwrap() as usize}));
    }
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({"rows":rows,"production_writes":0}))?,
    )?;
    Ok(())
}
