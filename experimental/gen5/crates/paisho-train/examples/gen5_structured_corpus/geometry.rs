//! Exhaustive human-prefix geometry census, bounded export by source group.
use super::{
    facts::{self, Facts},
    hash, write_json, Result,
};
use paisho_ai::{HarmonyCycleGeometry, HarmonyCycleWitness};
use paisho_core::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
    time::Instant,
};

pub(super) fn created(c: &HarmonyCycleWitness, old: &[Harmony]) -> bool {
    c.vertices
        .iter()
        .zip(c.vertices.iter().cycle().skip(1))
        .take(c.vertices.len())
        .any(|(&a, &b)| {
            !old.iter().any(|h| {
                h.owner == c.owner
                    && ((h.first == a && h.second == b) || (h.first == b && h.second == a))
            })
        })
}
pub(super) fn kind(g: HarmonyCycleGeometry) -> &'static str {
    match g {
        HarmonyCycleGeometry::EnclosingCentre => "enclosing",
        HarmonyCycleGeometry::OffCentre => "off_centre",
        HarmonyCycleGeometry::TouchingCentre => "touching",
    }
}
pub(super) fn persist(out: &Path, r: &GameRecord) -> Result<Value> {
    let text = r.to_string();
    let sha = hash(text.as_bytes());
    let path = format!("psr/{sha}.psr");
    if !out.join(&path).exists() {
        fs::write(out.join(&path), &text)?;
    }
    Ok(json!({"path":path,"sha256":sha,"position_sha256":facts::position_hash(&r.replay()?)}))
}
fn row(g: &Value) -> Value {
    json!({"source":g["game_sha256"],"group":g["split_identity_sha256"],"held_out":g["held_out"]})
}
fn admit(quota: &mut BTreeMap<String, usize>, key: String) -> bool {
    let n = quota.entry(key).or_default();
    *n += 1;
    *n <= 2
}
fn count(
    counts: &mut BTreeMap<String, usize>,
    groups: &mut BTreeMap<String, BTreeSet<String>>,
    g: &Value,
    tag: &str,
) {
    let key = format!(
        "{}:{tag}",
        if g["held_out"] == true {
            "heldout"
        } else {
            "train"
        }
    );
    *counts.entry(key.clone()).or_default() += 1;
    groups
        .entry(key)
        .or_default()
        .insert(g["split_identity_sha256"].as_str().unwrap().into());
}

pub fn mine(manifest: &str, out: &Path) -> Result<()> {
    fs::create_dir(out)?;
    fs::create_dir(out.join("psr"))?;
    let start = Instant::now();
    let bytes = fs::read(manifest)?;
    let data: Value = serde_json::from_slice(&bytes)?;
    let mut games: Vec<_> = data["games"]
        .as_array()
        .ok_or("games missing")?
        .iter()
        .collect();
    games.sort_by_key(|g| g["game_sha256"].as_str().unwrap());
    let mut statefile = std::io::BufWriter::new(fs::File::create(out.join("states.jsonl"))?);
    let mut pairfile = std::io::BufWriter::new(fs::File::create(out.join("pairs.jsonl"))?);
    let mut ledger = std::io::BufWriter::new(fs::File::create(out.join("sources.jsonl"))?);
    let mut counts = BTreeMap::new();
    let mut groups = BTreeMap::new();
    let mut quota = BTreeMap::new();
    let mut split = BTreeMap::new();
    let mut roots = 0u64;
    let mut actions = 0u64;
    let mut states = 0;
    let mut pairs = 0;
    let mut root_seen = BTreeSet::new();
    for (gi, g) in games.iter().enumerate() {
        let group = g["split_identity_sha256"].as_str().ok_or("group")?;
        let held = g["held_out"].as_bool().ok_or("split")?;
        if split
            .insert(group.to_owned(), held)
            .is_some_and(|x| x != held)
        {
            return Err("group split conflict".into());
        }
        let orig = &g["originals"][0];
        let b = fs::read(orig["path"].as_str().ok_or("path")?)?;
        if hash(&b) != orig["sha256"] {
            return Err("source hash mismatch".into());
        }
        let old: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let (r, end) = old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?;
        let mut p = r.initial_position();
        let mut prefix = GameRecord::with_rules(r.setup(), r.rules());
        let mut candidates = vec![];
        let mut selected_states = vec![];
        for decision in 0..=r.actions().len() {
            if p.outcome() != GameOutcome::Ongoing {
                break;
            }
            roots += 1;
            let f = Facts::new(&p);
            // State duplicates inside one source must not consume its quota.
            let unique = root_seen.insert((group.to_owned(), facts::position_hash(&p)));
            for owner in [Player::Host, Player::Guest] {
                let owned: Vec<_> = f.rel.cycles.iter().filter(|c| c.owner == owner).collect();
                if owned.is_empty() {
                    candidates.push((decision, p.clone(), owner));
                }
                let kinds: BTreeSet<_> = owned.iter().map(|c| kind(c.geometry)).collect();
                for k in kinds {
                    let role = if owner == p.to_move() {
                        "own"
                    } else {
                        "opponent"
                    };
                    let tag = format!("current_{role}_{k}");
                    count(&mut counts, &mut groups, g, &tag);
                    if unique && admit(&mut quota, format!("{group}:{tag}")) {
                        selected_states.push((decision, p.clone(), owner, k));
                    }
                }
            }
            let legal = legal_actions(&p);
            let mut winner = None;
            for &a in &legal {
                let mut q = p.clone();
                q.apply(a)?;
                actions += 1;
                if q.outcome() == GameOutcome::Win(p.to_move())
                    && harmony_ring_owners_for_profile(q.board(), q.rule_profile())
                        .contains(&p.to_move())
                    && winner.is_none()
                {
                    winner = Some(a);
                }
            }
            if let Some(win) = winner {
                count(&mut counts, &mut groups, g, "root_with_immediate_ring_win");
                let mut choices = BTreeMap::new();
                for &a in &legal {
                    let mut q = p.clone();
                    q.apply(a)?;
                    if q.outcome() != GameOutcome::Ongoing {
                        continue;
                    }
                    let next = Facts::new(&q);
                    if !next.rings.is_empty() {
                        continue;
                    }
                    for c in next.rel.cycles.iter().filter(|c| c.owner == p.to_move()) {
                        let k = kind(c.geometry);
                        if k == "enclosing" {
                            return Err("ongoing enclosing ring".into());
                        }
                        let is_new = created(c, &f.rel.edges);
                        choices.entry((k, is_new)).or_insert(a);
                    }
                }
                for ((k, is_new), lose) in choices {
                    let tag = format!("pair_{}_{}", k, if is_new { "created" } else { "retained" });
                    count(&mut counts, &mut groups, g, &tag);
                    if unique && admit(&mut quota, format!("{group}:{tag}")) {
                        let mut value = row(g);
                        value["family"] = json!(k);
                        value["new_cycle"] = json!(is_new);
                        value["decision"] = json!(decision);
                        value["root"] = persist(out, &prefix)?;
                        value["winning_action"] = json!(win.to_string());
                        value["nonwinning_action"] = json!(lose.to_string());
                        value["mover"] = json!(p.to_move().code().to_string());
                        serde_json::to_writer(&mut pairfile, &value)?;
                        writeln!(pairfile)?;
                        pairs += 1;
                    }
                }
            }
            if let Some(&a) = r.actions().get(decision) {
                p.apply(a)?;
                prefix.push(a);
            }
        }
        if p != end {
            return Err("replayed game mismatch".into());
        }
        // Match current motifs to legal, acyclic positions from the same game,
        // owner, player-to-move and phase. No terminal states as training inputs.
        for (decision, p, owner, k) in selected_states {
            let f = Facts::new(&p);
            let ec = f.rel.edges.iter().filter(|h| h.owner == owner).count();
            let best = candidates
                .iter()
                .filter(|(_, q, s)| {
                    *s == owner && q.to_move() == p.to_move() && q.phase() == p.phase()
                })
                .min_by_key(|(i, q, _)| {
                    let qe = harmonies(q.board())
                        .iter()
                        .filter(|h| h.owner == owner)
                        .count();
                    (
                        p.board()
                            .occupied_count()
                            .abs_diff(q.board().occupied_count())
                            * 8
                            + ec.abs_diff(qe) * 4,
                        decision.abs_diff(*i),
                        *i,
                    )
                });
            let mut positive = GameRecord::with_rules(r.setup(), r.rules());
            for &a in &r.actions()[..decision] {
                positive.push(a);
            }
            let mut value = row(g);
            value["family"] = json!(k);
            value["decision"] = json!(decision);
            value["owner"] = json!(owner.code().to_string());
            value["root"] = persist(out, &positive)?;
            value["position"] = f.json(&p);
            if let Some((index, q, _)) = best {
                let mut negative = GameRecord::with_rules(r.setup(), r.rules());
                for &a in &r.actions()[..*index] {
                    negative.push(a);
                }
                value["acyclic_control"] = persist(out, &negative)?;
                value["control_decision"] = json!(index);
                value["control_position"] = Facts::new(q).json(q);
            }
            serde_json::to_writer(&mut statefile, &value)?;
            writeln!(statefile)?;
            states += 1;
        }
        serde_json::to_writer(
            &mut ledger,
            &json!({"source":g["game_sha256"],"group":group,"held_out":held,
            "original":orig,"gen5_decisions":r.actions().len(),"original_decisions":old.actions().len()}),
        )?;
        writeln!(ledger)?;
        if (gi + 1) % 100 == 0 {
            eprintln!(
                "{} games / {roots} roots / {actions} actions / {pairs} pairs / {states} states",
                gi + 1
            );
            pairfile.flush()?;
            statefile.flush()?;
        }
    }
    statefile.flush()?;
    pairfile.flush()?;
    ledger.flush()?;
    write_json(
        out.join("summary.json"),
        &json!({"manifest":manifest,"manifest_sha256":hash(&bytes),
        "games":games.len(),"nonterminal_prefixes":roots,"legal_actions_examined":actions,
        "counts":counts,"groups":groups.iter().map(|(k,v)|(k,v.len())).collect::<BTreeMap<_,_>>(),
        "exported_state_pairs":states,"exported_action_pairs":pairs,"seconds":start.elapsed().as_secs_f64(),
        "rules":"SkudPaiShoGen5V1","no_models_or_search":true}),
    )
}
