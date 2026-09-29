//! Exact replay of existing reference games, counting both sides' available data.
use paisho_core::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("JOURNAL RUN_DIRECTORY OUTPUT".into());
    }
    let root = Path::new(&args[2]);
    let mut rows = vec![];
    for line in BufReader::new(fs::File::open(&args[1])?).lines() {
        let r: Value = serde_json::from_str(&line?)?;
        if r["lane"] != "Historical" {
            continue;
        }
        let id = r["id"].as_u64().unwrap();
        let path = root.join(format!("training/games/game-{id:07}.psr"));
        let b = fs::read(&path)?;
        assert_eq!(hash(&b), r["psr"]);
        let record: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let mut p = record.initial_position();
        let seat = if r["seat"] == "Host" {
            Player::Host
        } else {
            Player::Guest
        };
        let prefix = r["prefix"].as_u64().unwrap() as usize;
        let mut own = 0;
        let mut other = 0;
        let mut own_explore = 0;
        let mut beyond_opening = 0;
        let mut opponent_immediate = 0;
        let mut own_immediate = 0;
        let mut first_opp_mate = None;
        for (i, a) in record.actions().iter().enumerate() {
            let mover = p.to_move();
            let fresh = i >= prefix;
            if fresh {
                if mover == seat {
                    own += 1;
                    if i - prefix < 40 {
                        own_explore += 1;
                        if i >= 40 {
                            beyond_opening += 1;
                        }
                    }
                } else {
                    other += 1;
                }
            }
            p.apply(*a)?;
            if fresh && p.outcome() == GameOutcome::Win(mover) {
                if mover == seat {
                    own_immediate += 1;
                } else {
                    opponent_immediate += 1;
                    first_opp_mate = Some(i + 1);
                }
            }
        }
        let result = match p.outcome() {
            GameOutcome::Win(w) => {
                if w == seat {
                    "W"
                } else {
                    "L"
                }
            }
            GameOutcome::Draw => "D",
            _ => "U",
        };
        assert_eq!(result, r["result"]);
        assert_eq!(own + other, r["decisions"].as_u64().unwrap() as usize);
        rows.push(json!({"id":id,"source":r["source"],"generation":r["generation"],"case":r["case"],"candidate_decisions":own,"opponent_decisions":other,
            "candidate_under_sampling_rule":own_explore,"sampling_beyond_absolute40":beyond_opening,"opponent_immediate_winning_decisions":opponent_immediate,
            "candidate_immediate_winning_decisions":own_immediate,"last_opponent_winning_decision":first_opp_mate,"fresh_targets":r["fresh"],"result":result,
            "prefix":prefix,"psr_sha256":r["psr"]}));
    }
    fs::write(
        &args[3],
        serde_json::to_vec(
            &json!({"rows":rows,"production_writes":0,"all_original_psrs_replayed":true}),
        )?,
    )?;
    Ok(())
}
