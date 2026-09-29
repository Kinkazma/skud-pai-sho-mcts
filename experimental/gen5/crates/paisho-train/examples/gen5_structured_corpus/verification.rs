use super::{branches::root_record, facts, hash, write_json, Result};
use paisho_core::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::BufRead,
    path::Path,
    time::Instant,
};

pub fn verify(census: &Path, out: &Path) -> Result<()> {
    let start = Instant::now();
    let roots: Vec<Value> = serde_json::from_slice(&fs::read(census.join("roots.json"))?)?;
    let mut records = BTreeMap::new();
    let mut metadata = BTreeMap::new();
    for row in roots {
        let key = row["key"].as_str().unwrap().to_owned();
        records.insert(key.clone(), root_record(census, &row)?);
        metadata.insert(key, row);
    }
    let mut verified = 0;
    let mut witnesses = 0;
    let mut negatives = 0;
    let mut cycles = 0;
    let mut seen = BTreeSet::new();
    for line in std::io::BufReader::new(fs::File::open(out.join("branches.jsonl"))?).lines() {
        let row: Value = serde_json::from_str(&line?)?;
        let r = &records[row["root"].as_str().unwrap()];
        let meta = &metadata[row["root"].as_str().unwrap()];
        if row["group"] != meta["group"] || row["held_out"] != meta["held_out"] {
            return Err("split provenance mismatch".into());
        }
        if !seen.insert((row["root"].to_string(), row["action"].to_string())) {
            return Err("duplicate action".into());
        }
        // Parse the persisted full PSR plus branch, not an unvalidated board.
        let mut branch = r.clone();
        let a: Action = row["action"].as_str().unwrap().parse()?;
        branch.push(a);
        let parsed: GameRecord = branch.to_string().parse()?;
        let p = r.replay()?;
        let q = parsed.replay()?;
        if facts::position_hash(&q) != row["position_sha256"] {
            return Err("successor mismatch".into());
        }
        if facts::result(q.outcome(), p.to_move()) != row["outcome"] {
            return Err("outcome mismatch".into());
        }
        let previous = harmonies(p.board());
        let current = harmonies(q.board());
        for (field, edges) in [
            (
                "edges_added",
                current
                    .iter()
                    .filter(|e| !previous.contains(e))
                    .collect::<Vec<_>>(),
            ),
            (
                "edges_removed",
                previous
                    .iter()
                    .filter(|e| !current.contains(e))
                    .collect::<Vec<_>>(),
            ),
        ] {
            let expected: Vec<_> = edges
                .iter()
                .map(|h| {
                    json!({"owner":h.owner.code().to_string(),
                "first":[h.first.x(),h.first.y()],"second":[h.second.x(),h.second.y()]})
                })
                .collect();
            if row[field] != json!(expected) {
                return Err("edge delta mismatch".into());
            }
        }
        let ring: Vec<_> = harmony_ring_owners_for_profile(q.board(), q.rule_profile())
            .iter()
            .map(|s| s.code().to_string())
            .collect();
        if row["ring_owners"] != json!(ring) {
            return Err("canonical ring mismatch".into());
        }
        for witness in row["cycle_witnesses"].as_array().ok_or("missing cycles")? {
            verify_cycle(&current, witness)?;
            cycles += 1;
        }
        if row["ending"] == "exhaustion" {
            let own = midline_crossing_harmony_count(q.board(), p.to_move());
            let other = midline_crossing_harmony_count(q.board(), p.to_move().opponent());
            let expected = match own.cmp(&other) {
                std::cmp::Ordering::Greater => "win",
                std::cmp::Ordering::Less => "loss",
                _ => "draw",
            };
            if expected != row["outcome"] || q.reserve(p.to_move()).basic_count() != 0 {
                return Err("exhaustion score mismatch".into());
            }
        }
        if let Some(replies) = row["threat"]["winning_replies"].as_array() {
            for reply in replies {
                let mut witness = parsed.clone();
                witness.push(reply.as_str().unwrap().parse()?);
                let replayed: GameRecord = witness.to_string().parse()?;
                if replayed.replay()?.outcome() != GameOutcome::Win(p.to_move().opponent()) {
                    return Err("invalid threat witness".into());
                }
                witnesses += 1;
            }
        }
        if ["absent", "present"]
            .iter()
            .any(|s| row["threat"]["status"] == *s)
        {
            if q.to_move() == p.to_move() || q.outcome() != GameOutcome::Ongoing {
                return Err("invalid absent scope".into());
            }
            let replies = legal_actions(&q);
            if row["threat"]["examined"] != replies.len() || row["threat"]["complete"] != true {
                return Err("incomplete enumeration".into());
            }
            let mut actual = vec![];
            for a in replies {
                let mut x = q.clone();
                x.apply(a)?;
                if x.outcome() == GameOutcome::Win(q.to_move()) {
                    actual.push(a.to_string());
                }
            }
            if row["threat"]["winning_replies"] != json!(actual) {
                return Err("reply census mismatch".into());
            }
            if actual.is_empty() {
                negatives += 1;
            }
        }
        verified += 1;
    }
    let expected: usize = records
        .values()
        .map(|r| r.replay().map(|p| legal_actions(&p).len()))
        .collect::<std::result::Result<Vec<_>, _>>()?
        .iter()
        .sum();
    if expected != verified {
        return Err("not all root actions represented".into());
    }
    write_json(
        out.join("verification.json"),
        &json!({"verified_branches":verified,"verified_winning_replies":witnesses,
        "verified_cycle_witnesses":cycles,"complete_root_action_coverage":true,"source_splits_verified":true,
        "rechecked_absences":negatives,"wall_seconds":start.elapsed().as_secs_f64(),
        "branches_sha256":hash(&fs::read(out.join("branches.jsonl"))?),"same_engine_not_independent_rules_implementation":true}),
    )
}

// Independent witness check: real harmony edges and signed crossings of the
// positive horizontal ray. No use of the prototype's cycle extractor/classifier.
pub(super) fn verify_cycle(edges: &[Harmony], witness: &Value) -> Result<()> {
    let vertices = witness["vertices"].as_array().ok_or("missing vertices")?;
    let points: Vec<(i64, i64)> = vertices
        .iter()
        .map(|v| Ok((v[0].as_i64().ok_or("x")?, v[1].as_i64().ok_or("y")?)))
        .collect::<Result<_>>()?;
    if points.len() < 4 || points.iter().collect::<BTreeSet<_>>().len() != points.len() {
        return Err("not a simple cycle witness".into());
    }
    let mut touching = false;
    let mut winding = 0;
    for (&a, &b) in points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .take(points.len())
    {
        let present = edges.iter().any(|h| {
            let x = (h.first.x() as i64, h.first.y() as i64);
            let y = (h.second.x() as i64, h.second.y() as i64);
            witness["owner"] == h.owner.code().to_string()
                && ((x == a && y == b) || (x == b && y == a))
        });
        if !present {
            return Err("cycle edge absent".into());
        }
        touching |= (a.0 == 0 && b.0 == 0 && a.1.min(b.1) <= 0 && a.1.max(b.1) >= 0)
            || (a.1 == 0 && b.1 == 0 && a.0.min(b.0) <= 0 && a.0.max(b.0) >= 0);
        if a.0 == b.0 && a.0 > 0 {
            if a.1 <= 0 && b.1 > 0 {
                winding += 1;
            }
            if b.1 <= 0 && a.1 > 0 {
                winding -= 1;
            }
        }
    }
    let expected = if touching {
        "TouchingCentre"
    } else if winding != 0 {
        "EnclosingCentre"
    } else {
        "OffCentre"
    };
    if witness["geometry"] != expected {
        return Err("cycle geometry mismatch".into());
    }
    Ok(())
}
