//! Independent archive replay and exact-state validation of recorded cycle spans.
use paisho_core::{GameRecord, Player, Position};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::BufRead, path::Path};
fn same(a: &Position, b: &Position) -> bool {
    a.rule_profile() == b.rule_profile()
        && a.board() == b.board()
        && a.reserve(Player::Host) == b.reserve(Player::Host)
        && a.reserve(Player::Guest) == b.reserve(Player::Guest)
        && a.to_move() == b.to_move()
        && a.phase() == b.phase()
        && a.outcome() == b.outcome()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let path = Path::new(&line);
        let bytes = fs::read(path)?;
        let r: Value = serde_json::from_slice(&fs::read(path.with_extension("json"))?)?;
        assert_eq!(r["psr_sha256"], format!("{:x}", Sha256::digest(&bytes)));
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        if !r["frozen_evaluation"].is_null() {
            let mut prefix = GameRecord::with_rules(record.setup(), record.rules());
            for a in record
                .actions()
                .iter()
                .take(r["prefix_decisions"].as_u64().unwrap() as usize)
            {
                prefix.push(*a);
            }
            assert_eq!(
                r["frozen_evaluation"]["prefix"],
                format!("{:x}", Sha256::digest(prefix.to_string().as_bytes()))
            );
            assert_eq!(r["collector"], r["frozen_evaluation"]["model"]);
        }
        let mut p = record.initial_position();
        let mut states = vec![p.clone()];
        for a in record.actions() {
            p.apply(*a)?;
            states.push(p.clone());
        }
        assert_eq!(r["outcome"], format!("{:?}", p.outcome()));
        assert_eq!(
            r["decisions"].as_u64().unwrap() as usize,
            record.actions().len()
        );
        if !r["cycle"].is_null() {
            let c = &r["cycle"];
            let period = c["period"].as_u64().unwrap() as usize;
            let cycles = c["cycles"].as_u64().unwrap() as usize;
            let last = c["last_decision"].as_u64().unwrap() as usize;
            let first = c["first_decision"].as_u64().unwrap() as usize;
            assert_eq!(last, states.len() - 1);
            assert_eq!(first, last - period * cycles + 1);
            assert!((last - period * cycles + period..=last)
                .all(|i| same(&states[i], &states[i - period])));
        }
        println!(
            "{}",
            json!({"path":line,"id":r["id"],"decisions":record.actions().len(),"outcome":r["outcome"],
            "cycle_verified":!r["cycle"].is_null(),"psr_sha256":r["psr_sha256"]})
        );
    }
    Ok(())
}
