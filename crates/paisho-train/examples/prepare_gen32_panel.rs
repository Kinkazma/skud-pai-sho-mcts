//! Score-blind preparation of distinct held-out human opening continuations.
use paisho_core::{GameOutcome, GameRecord, Position, TurnPhase};
use paisho_train::gen32::RULES;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn same(a: &Position, b: &Position) -> bool {
    use paisho_core::Player::*;
    a.board() == b.board()
        && a.reserve(Host) == b.reserve(Host)
        && a.reserve(Guest) == b.reserve(Guest)
        && a.to_move() == b.to_move()
        && a.phase() == b.phase()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let input: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let out = Path::new(&a[2]);
    let count: usize = a[3].parse()?;
    fs::create_dir(out)?;
    let mut cases = vec![];
    let mut starts: Vec<Position> = vec![];
    let mut identities = HashSet::new();
    for g in input["candidates"].as_array().ok_or("candidates")? {
        if g["held_out"] != true {
            return Err("non-held-out source".into());
        }
        let identity = g["split_identity_sha256"]
            .as_str()
            .ok_or("source identity")?;
        if identities.contains(identity) {
            continue;
        }
        let source = g["path"].as_str().ok_or("source path")?;
        let bytes = fs::read(source)?;
        if g["sha256"] != hash(&bytes) {
            return Err("source hash".into());
        }
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        if record.rules() != RULES {
            return Err("rules".into());
        }
        record.replay()?;
        let target = if cases.len() % 2 == 0 { 8 } else { 16 };
        let mut prefix = GameRecord::with_rules(record.setup(), RULES);
        let mut p = prefix.initial_position();
        let mut seen = vec![p.clone()];
        let mut repeated = false;
        for action in record.actions() {
            if p.completed_turns() >= target && p.phase() == TurnPhase::Main {
                break;
            }
            if p.outcome() != GameOutcome::Ongoing {
                break;
            }
            p.apply(*action)?;
            prefix.push(*action);
            if seen.iter().any(|s| same(s, &p)) {
                repeated = true;
                break;
            }
            seen.push(p.clone());
        }
        if repeated
            || p.completed_turns() != target
            || p.phase() != TurnPhase::Main
            || p.outcome() != GameOutcome::Ongoing
            || starts.iter().any(|s| same(s, &p))
        {
            continue;
        }
        let id = cases.len();
        let path = out.join(format!("case-{id:02}.psr"));
        let text = prefix.to_string();
        fs::write(&path, &text)?;
        cases.push(json!({"id":id,"prefix_path":path.canonicalize()?,"prefix_sha256":hash(text.as_bytes()),"source_path":source,"source_sha256":g["sha256"],"split_identity_sha256":identity,"held_out":true,"decisions":prefix.actions().len(),"completed_turns":target,"to_move":format!("{:?}",p.to_move())}));
        starts.push(p);
        identities.insert(identity.to_owned());
        if cases.len() == count {
            break;
        }
    }
    if cases.len() != count {
        return Err("not enough distinct sources".into());
    }
    fs::write(
        out.join("panel.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"paisho-frozen-human-prefix-panel-v1","rules":RULES.as_str(),"selection":"held-out sources hash ordered; alternating 8/16 complete turns; distinct source identities and legal states; no model outcome used","input_sha256":hash(&fs::read(&a[1])?),"cases":cases}),
        )?,
    )?;
    println!("{} distinct held-out prefixes", cases.len());
    Ok(())
}
