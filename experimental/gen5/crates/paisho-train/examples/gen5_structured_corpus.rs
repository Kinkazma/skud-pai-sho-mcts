//! Rule-only, isolated corpus construction. No models, SGD, MCTS or publication.
#[path = "gen5_structured_corpus/branches.rs"]
mod branches;
#[path = "gen5_structured_corpus/facts.rs"]
mod facts;
#[path = "gen5_structured_corpus/geometry.rs"]
mod geometry;
#[path = "gen5_structured_corpus/geometry_neighbors.rs"]
mod geometry_neighbors;
#[path = "gen5_structured_corpus/geometry_verify.rs"]
mod geometry_verify;
#[path = "gen5_structured_corpus/legacy_index.rs"]
mod legacy_index;
#[cfg(test)]
#[path = "gen5_structured_corpus/input_tests.rs"]
mod input_tests;
#[path = "gen5_structured_corpus/verification.rs"]
mod verification;
use paisho_core::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn write_json(p: impl AsRef<Path>, v: &Value) -> Result<()> {
    fs::write(p, serde_json::to_vec_pretty(v)?)?;
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err(
            "census MANIFEST NEW_DIR | expand CENSUS NEW_DIR | verify CENSUS BRANCHES | geometry MANIFEST NEW_DIR | geometry-neighbors MANIFEST NEW_DIR | verify-geometry MANIFEST DIR | index-legacy PLAN NEW_DIR".into(),
        );
    }
    match args[1].as_str() {
        "census" => census(&args[2], Path::new(&args[3])),
        "expand" => branches::expand(Path::new(&args[2]), Path::new(&args[3])),
        "verify" => verification::verify(Path::new(&args[2]), Path::new(&args[3])),
        "geometry" => geometry::mine(&args[2], Path::new(&args[3])),
        "verify-geometry" => geometry_verify::verify(&args[2], Path::new(&args[3])),
        "geometry-neighbors" => geometry_neighbors::mine(&args[2], Path::new(&args[3])),
        "index-legacy" => legacy_index::index(&args[2], Path::new(&args[3])),
        _ => Err("unknown mode".into()),
    }
}

fn census(manifest: &str, out: &Path) -> Result<()> {
    fs::create_dir(out)?;
    fs::create_dir(out.join("roots"))?;
    let start = Instant::now();
    let bytes = fs::read(manifest)?;
    let data: Value = serde_json::from_slice(&bytes)?;
    let mut games: Vec<_> = data["games"]
        .as_array()
        .ok_or("missing games")?
        .iter()
        .collect();
    games.sort_by_key(|g| g["game_sha256"].as_str().unwrap());
    let mut counts = BTreeMap::<String, usize>::new();
    let mut sources = BTreeMap::<String, BTreeSet<String>>::new();
    let mut selected = BTreeMap::<String, BTreeSet<String>>::new();
    let mut groups = BTreeMap::<String, bool>::new();
    let mut roots = BTreeMap::<String, Value>::new();
    let mut ledger = std::io::BufWriter::new(fs::File::create(out.join("sources.jsonl"))?);
    let mut rejected = vec![];
    let mut decisions = 0;
    let mut valid = 0;
    for g in &games {
        let source = g["game_sha256"]
            .as_str()
            .ok_or("missing source")?
            .to_owned();
        let group = g["split_identity_sha256"].as_str().ok_or("missing group")?;
        let held = g["held_out"].as_bool().ok_or("missing split")?;
        if groups
            .insert(group.to_owned(), held)
            .is_some_and(|h| h != held)
        {
            return Err("source group crosses train/test".into());
        }
        let original = &g["originals"][0];
        let b = fs::read(original["path"].as_str().ok_or("missing path")?)?;
        if hash(&b) != original["sha256"] {
            return Err("source hash changed".into());
        }
        let old: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let (r, end) = match old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1) {
            Ok(r) => r,
            Err(e) => {
                rejected.push(json!({"source":source,"error":e.to_string()}));
                continue;
            }
        };
        valid += 1;
        let mut p = r.initial_position();
        let mut prefix = GameRecord::with_rules(r.setup(), r.rules());
        for (i, &a) in r.actions().iter().enumerate() {
            let mut next = p.clone();
            next.apply(a)?;
            decisions += 1;
            let f = facts::Facts::new(&next);
            let mut tags = f.geometry_tags();
            // Human players may avoid a losing last plantation. Mine legal
            // alternatives too; observed human endings alone bias this label.
            if p.reserve(p.to_move()).basic_count() == 1 {
                if let Some(last) = legal_actions(&p)
                    .into_iter()
                    .find(|a| matches!(a, Action::Plant { .. } | Action::BonusPlantBasic { .. }))
                {
                    let mut alternative = p.clone();
                    alternative.apply(last)?;
                    if facts::ending(&p, last, &alternative) != "exhaustion" {
                        return Err("last plantation failed to exhaust".into());
                    }
                    tags.push(format!(
                        "available_exhaustion_{}",
                        facts::result(alternative.outcome(), p.to_move())
                    ));
                }
            }
            let route = facts::ending(&p, a, &next);
            if route == "exhaustion" {
                tags.push(format!(
                    "exhaustion_{}",
                    facts::result(next.outcome(), p.to_move())
                ));
            } else if route == "ring" {
                tags.push("ring_ending".into());
            } else if route != "ongoing" {
                tags.push("other_ending".into());
            }
            for tag in tags {
                let key = format!("{}:{tag}", if held { "heldout" } else { "train" });
                *counts.entry(key.clone()).or_default() += 1;
                sources.entry(key.clone()).or_default().insert(group.into());
                // One root per source group per category, at most 16 groups.
                let selected_groups = selected.entry(key).or_default();
                if selected_groups.len() < 16 && selected_groups.insert(group.into()) {
                    save_root(out, &prefix, g, i, &tag, &mut roots)?;
                    // Previous decision can contain a real defensive contrast;
                    // do not invent an opponent turn when a bonus intervenes.
                    if route == "ring" && i > 0 {
                        let mut before = GameRecord::with_rules(r.setup(), r.rules());
                        for &a in &r.actions()[..i - 1] {
                            before.push(a);
                        }
                        save_root(out, &before, g, i - 1, "before_ring_reply", &mut roots)?;
                    }
                }
            }
            prefix.push(a);
            p = next;
        }
        if p != end || prefix.replay()? != end {
            return Err("full replay mismatch".into());
        }
        serde_json::to_writer(
            &mut ledger,
            &json!({"source":source,"group":group,"held_out":held,
            "original":original,"original_rules":format!("{:?}",old.rules()),
            "original_decisions":old.actions().len(),"gen5_decisions":r.actions().len(),
            "gen5_psr_sha256":hash(r.to_string().as_bytes()),"outcome":format!("{:?}",end.outcome())}),
        )?;
        writeln!(ledger)?;
        if valid % 100 == 0 {
            eprintln!("{valid} games / {decisions} legal decisions");
        }
    }
    ledger.flush()?;
    write_json(
        out.join("roots.json"),
        &json!(roots.values().collect::<Vec<_>>()),
    )?;
    write_json(
        out.join("summary.json"),
        &json!({"schema":"gen5-structured-corpus-v2-owner-separated-components",
        "manifest":manifest,"manifest_sha256":hash(&bytes),"rules":"SkudPaiShoGen5V1",
        "input_games":games.len(),"valid_games":valid,"decisions":decisions,"rejected":rejected,
        "counts":counts,"independent_groups":sources.iter().map(|(k,v)|(k,v.len())).collect::<BTreeMap<_,_>>(),
        "selected_roots":roots.len(),"wall_seconds":start.elapsed().as_secs_f64(),
        "limitations":["cycle witnesses are not all simple cycles","historical model exposure unverified",
            "no learning or strength measurement","source-group split retained; exact-state leakage still needs checking"]}),
    )?;
    Ok(())
}

fn save_root(
    out: &Path,
    r: &GameRecord,
    g: &Value,
    decision: usize,
    tag: &str,
    roots: &mut BTreeMap<String, Value>,
) -> Result<()> {
    let text = r.to_string();
    let sha = hash(text.as_bytes());
    // Include group in identity: equivalent states across groups are audited later.
    let key = hash(format!("{}:{decision}", g["split_identity_sha256"]).as_bytes());
    if let Some(row) = roots.get_mut(&key) {
        if row["sha256"] != sha {
            return Err("group/decision has different histories".into());
        }
        let tags = row["tags"].as_array_mut().unwrap();
        if !tags.contains(&json!(tag)) {
            tags.push(json!(tag));
        }
        return Ok(());
    }
    let p = r.replay()?;
    if p.outcome() != GameOutcome::Ongoing {
        return Err("selected root already terminal".into());
    }
    let path = format!("roots/{key}.psr");
    fs::write(out.join(&path), &text)?;
    let parsed: GameRecord = fs::read_to_string(out.join(&path))?.parse()?;
    if parsed != *r || parsed.replay()? != p {
        return Err("saved root roundtrip mismatch".into());
    }
    roots.insert(
        key.clone(),
        json!({"key":key,"psr":path,"sha256":sha,"source":g["game_sha256"],
        "group":g["split_identity_sha256"],"held_out":g["held_out"],"decision":decision,
        "tags":[tag],"position":facts::Facts::new(&p).json(&p)}),
    );
    Ok(())
}
