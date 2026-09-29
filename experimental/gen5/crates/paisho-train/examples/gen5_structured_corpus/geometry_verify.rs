use super::{
    facts::{self, Facts},
    geometry, hash, verification, write_json, Result,
};
use paisho_core::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, Write},
    path::Path,
    time::Instant,
};

pub(super) fn read(out: &Path, v: &Value) -> Result<GameRecord> {
    let b = fs::read(out.join(v["path"].as_str().ok_or("path")?))?;
    if hash(&b) != v["sha256"] {
        return Err("PSR hash mismatch".into());
    }
    let r: GameRecord = std::str::from_utf8(&b)?.parse()?;
    let p = r.replay()?;
    if r.rules() != RuleProfileId::SkudPaiShoGen5V1
        || p.outcome() != GameOutcome::Ongoing
        || facts::position_hash(&p) != v["position_sha256"]
    {
        return Err("invalid root".into());
    }
    Ok(r)
}
fn owner(v: &Value) -> Result<Player> {
    match v.as_str() {
        Some("H") => Ok(Player::Host),
        Some("G") => Ok(Player::Guest),
        _ => Err("owner".into()),
    }
}
fn validate_facts(p: &Position) -> Result<()> {
    let f = Facts::new(p);
    for c in f.json(p)["cycle_witnesses"].as_array().ok_or("cycles")? {
        verification::verify_cycle(&f.rel.edges, c)?;
    }
    Ok(())
}
pub fn verify(manifest: &str, out: &Path) -> Result<()> {
    let start = Instant::now();
    let data: Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let mut sources = BTreeMap::new();
    let mut source_cache = BTreeMap::new();
    for g in data["games"].as_array().ok_or("games")? {
        sources.insert(g["game_sha256"].as_str().unwrap().to_owned(), g);
    }
    let mut seen = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut symmetries = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut state_pairs = 0;
    let mut action_pairs = 0;
    let mut index = std::io::BufWriter::new(fs::File::create(out.join("positions.jsonl"))?);
    for name in ["states.jsonl", "pairs.jsonl"] {
        for (line_index, line) in std::io::BufReader::new(fs::File::open(out.join(name))?)
            .lines()
            .enumerate()
        {
            let v: Value = serde_json::from_str(&line?)?;
            let g = sources
                .get(v["source"].as_str().ok_or("source")?)
                .ok_or("unknown source")?;
            if v["held_out"] != g["held_out"] || v["group"] != g["split_identity_sha256"] {
                return Err("provenance mismatch".into());
            }
            let r = read(out, &v["root"])?;
            let p = r.replay()?;
            let source = g["game_sha256"].as_str().unwrap();
            if !source_cache.contains_key(source) {
                let o = &g["originals"][0];
                let b = fs::read(o["path"].as_str().unwrap())?;
                if hash(&b) != o["sha256"] {
                    return Err("source hash mismatch".into());
                }
                let old: GameRecord = std::str::from_utf8(&b)?.parse()?;
                source_cache.insert(
                    source.to_owned(),
                    old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?
                        .0,
                );
            }
            let original = &source_cache[source];
            let n = v["decision"].as_u64().ok_or("decision")? as usize;
            let base = v["source_prefix_decision"].as_u64().unwrap_or(n as u64) as usize;
            let extra: Vec<Action> = v["extension_actions"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|a| {
                            a.as_str()
                                .ok_or("extension action")?
                                .parse()
                                .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?
                .unwrap_or_default();
            if original.setup() != r.setup()
                || original.actions().get(..base) != r.actions().get(..base)
                || r.actions().get(base..) != Some(extra.as_slice())
                || n != base + extra.len()
            {
                return Err("not source prefix".into());
            }
            validate_facts(&p)?;
            let held = v["held_out"].as_bool().unwrap();
            seen.entry(facts::position_hash(&p))
                .or_default()
                .insert(held);
            symmetries
                .entry(facts::symmetry_hash(&p))
                .or_default()
                .insert(held);
            index_position(&mut index, &p, &v, name, line_index, "root")?;
            if name == "states.jsonl" {
                let s = owner(&v["owner"])?;
                let f = Facts::new(&p);
                if !f
                    .rel
                    .cycles
                    .iter()
                    .any(|c| c.owner == s && v["family"] == geometry::kind(c.geometry))
                {
                    return Err("missing current motif".into());
                }
                if v["position"] != f.json(&p) {
                    return Err("state facts mismatch".into());
                }
                if !v["acyclic_control"].is_null() {
                    let cr = read(out, &v["acyclic_control"])?;
                    let q = cr.replay()?;
                    let cn = v["control_decision"].as_u64().ok_or("control decision")? as usize;
                    if cr.setup() != original.setup()
                        || original.actions().get(..cn) != Some(cr.actions())
                    {
                        return Err("not control prefix".into());
                    }
                    // Acyclicity checked independently via E - V + components,
                    // with every piece (including isolated pieces) counted.
                    let f = Facts::new(&q);
                    let edges = f.rel.edges.iter().filter(|h| h.owner == s).count();
                    let mut vertices: BTreeSet<_> = q
                        .board()
                        .occupied()
                        .filter(|(_, t)| t.owner == s)
                        .map(|(at, _)| at)
                        .collect();
                    for h in f.rel.edges.iter().filter(|h| h.owner == s) {
                        vertices.extend([h.first, h.second]);
                    }
                    let nodes = vertices.len();
                    if edges + f.components(&q)[s.index()] != nodes
                        || q.to_move() != p.to_move()
                        || q.phase() != p.phase()
                    {
                        return Err("invalid acyclic control".into());
                    }
                    if v["control_position"] != f.json(&q) {
                        return Err("control facts mismatch".into());
                    }
                    index_position(&mut index, &q, &v, name, line_index, "acyclic_control")?;
                    seen.entry(facts::position_hash(&q))
                        .or_default()
                        .insert(held);
                    symmetries
                        .entry(facts::symmetry_hash(&q))
                        .or_default()
                        .insert(held);
                }
                state_pairs += 1;
            } else {
                let mover = owner(&v["mover"])?;
                if mover != p.to_move() {
                    return Err("mover mismatch".into());
                }
                let mut win = r.clone();
                win.push(v["winning_action"].as_str().ok_or("win")?.parse()?);
                let w: GameRecord = win.to_string().parse()?;
                let w = w.replay()?;
                if w.outcome() != GameOutcome::Win(mover)
                    || !harmony_ring_owners_for_profile(w.board(), w.rule_profile())
                        .contains(&mover)
                {
                    return Err("not winning ring".into());
                }
                let mut neg = r.clone();
                neg.push(v["nonwinning_action"].as_str().ok_or("negative")?.parse()?);
                let q: GameRecord = neg.to_string().parse()?;
                let q = q.replay()?;
                let f = Facts::new(&q);
                if q.outcome() != GameOutcome::Ongoing
                    || !f.rings.is_empty()
                    || !f.rel.cycles.iter().any(|c| {
                        c.owner == mover
                            && v["family"] == geometry::kind(c.geometry)
                            && v["new_cycle"] == geometry::created(c, &harmonies(p.board()))
                    })
                {
                    return Err("invalid cycle alternative".into());
                }
                validate_facts(&w)?;
                validate_facts(&q)?;
                index_position(&mut index, &w, &v, name, line_index, "winning_successor")?;
                index_position(&mut index, &q, &v, name, line_index, "nonwinning_successor")?;
                for p in [&w, &q] {
                    seen.entry(facts::position_hash(p))
                        .or_default()
                        .insert(held);
                    symmetries
                        .entry(facts::symmetry_hash(p))
                        .or_default()
                        .insert(held);
                }
                action_pairs += 1;
            }
        }
    }
    index.flush()?;
    write_json(
        out.join("verification.json"),
        &json!({"state_pairs":state_pairs,"action_pairs":action_pairs,
        "cross_split_exact_states":seen.values().filter(|v|v.len()>1).count(),
        "cross_split_symmetric_states":symmetries.values().filter(|v|v.len()>1).count(),"seconds":start.elapsed().as_secs_f64(),
        "states_sha256":hash(&fs::read(out.join("states.jsonl"))?),"pairs_sha256":hash(&fs::read(out.join("pairs.jsonl"))?),
        "source_prefixes_verified":true,"geometry_independently_classified":true,"same_core_rules_engine":true}),
    )
}
fn index_position(
    w: &mut impl Write,
    p: &Position,
    v: &Value,
    file: &str,
    line: usize,
    role: &str,
) -> Result<()> {
    let f = Facts::new(p);
    let mut value = json!({"file":file,"line":line,"role":role,"group":v["group"],
        "held_out":v["held_out"],"exact":facts::position_hash(p),"symmetry":facts::symmetry_hash(p),
        "position":f.json(p),"owner_graphs":f.owner_graphs(p)});
    if file == "pairs.jsonl" && role == "root" {
        let mut labels = vec![];
        for a in legal_actions(p) {
            let mut q = p.clone();
            q.apply(a)?;
            labels.push(
                json!({"action":a.to_string(),"outcome":facts::result(q.outcome(),p.to_move())}),
            );
        }
        value["all_legal_action_outcomes"] = json!(labels);
    }
    serde_json::to_writer(&mut *w, &value)?;
    writeln!(w)?;
    Ok(())
}
