use super::{facts, hash, write_json, Result};
use paisho_core::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
    time::Instant,
};

pub fn threat(p: &Position, mover: Player, limit: usize) -> Result<Value> {
    if p.outcome() != GameOutcome::Ongoing {
        return Ok(json!({"status":"terminal"}));
    }
    if p.to_move() == mover {
        return Ok(json!({"status":"same_player_bonus"}));
    }
    let actions = legal_actions(p);
    let mut wins = vec![];
    let mut examined = 0;
    for &a in actions.iter().take(limit) {
        let mut q = p.clone();
        q.apply(a)?;
        examined += 1;
        if q.outcome() == GameOutcome::Win(mover.opponent()) {
            wins.push(a.to_string());
        }
    }
    let complete = examined == actions.len();
    let status = if !wins.is_empty() {
        "present"
    } else if complete {
        "absent"
    } else {
        "unknown"
    };
    Ok(
        json!({"status":status,"complete":complete,"examined":examined,"legal":actions.len(),
        "winning_replies":wins,"scope":"one_opponent_decision"}),
    )
}
fn edge(h: &Harmony) -> Value {
    json!({"owner":h.owner.code().to_string(),
    "first":[h.first.x(),h.first.y()],"second":[h.second.x(),h.second.y()]})
}
pub(super) fn root_record(census: &Path, row: &Value) -> Result<GameRecord> {
    let b = fs::read(census.join(row["psr"].as_str().ok_or("psr missing")?))?;
    if hash(&b) != row["sha256"] {
        return Err("root hash mismatch".into());
    }
    Ok(std::str::from_utf8(&b)?.parse()?)
}
pub fn expand(census: &Path, out: &Path) -> Result<()> {
    fs::create_dir(out)?;
    let start = Instant::now();
    let roots_bytes = fs::read(census.join("roots.json"))?;
    let roots: Vec<Value> = serde_json::from_slice(&roots_bytes)?;
    let mut rows = std::io::BufWriter::new(fs::File::create(out.join("branches.jsonl"))?);
    let mut pairs = vec![];
    let mut counts = BTreeMap::<String, usize>::new();
    let mut groups = BTreeMap::<String, BTreeSet<String>>::new();
    let mut positions = BTreeMap::<String, BTreeSet<bool>>::new();
    let mut actions_total = 0;
    let mut replies_total = 0;
    for (ri, root) in roots.iter().enumerate() {
        let r = root_record(census, root)?;
        let p = r.replay()?;
        let mover = p.to_move();
        let before = facts::Facts::new(&p);
        let mut categories = BTreeMap::<String, Value>::new();
        let held = root["held_out"].as_bool().unwrap();
        positions
            .entry(facts::position_hash(&p))
            .or_default()
            .insert(held);
        for (ai, a) in legal_actions(&p).into_iter().enumerate() {
            let mut q = p.clone();
            q.apply(a)?;
            let after = facts::Facts::new(&q);
            let ending = facts::ending(&p, a, &q);
            let outcome = facts::result(q.outcome(), mover);
            let added: Vec<_> = after
                .rel
                .edges
                .iter()
                .filter(|e| !before.rel.edges.contains(e))
                .collect();
            let removed: Vec<_> = before
                .rel
                .edges
                .iter()
                .filter(|e| !after.rel.edges.contains(e))
                .collect();
            let own_added = added.iter().filter(|h| h.owner == mover).count();
            let t = threat(&q, mover, usize::MAX)?;
            replies_total += t["examined"].as_u64().unwrap_or(0);
            actions_total += 1;
            let mut tags = vec![];
            if ending == "exhaustion" {
                tags.push(format!("exhaustion_{outcome}"));
            }
            let own_ring = after.rings.contains(&mover);
            if own_ring {
                tags.push("own_ring".into());
            }
            for c in &after.rel.cycles {
                if c.owner == mover {
                    tags.push(format!("own_{:?}", c.geometry));
                }
            }
            if own_added > 0 {
                tags.push(
                    if outcome == "win" {
                        "connection_immediate_win"
                    } else if outcome == "ongoing" {
                        "connection_without_immediate_win"
                    } else {
                        "connection_terminal_nonwin"
                    }
                    .into(),
                );
            }
            tags.push(format!("threat_{}", t["status"].as_str().unwrap()));
            tags.sort();
            tags.dedup();
            let row = json!({"root":root["key"],"group":root["group"],"held_out":held,"action_index":ai,
                "action":a.to_string(),"mover":mover.code().to_string(),"outcome":outcome,"ending":ending,
                "to_move_after":q.to_move().code().to_string(),"phase_after":format!("{:?}",q.phase()),
                "score_before_host_guest":before.score,"score_after_host_guest":after.score,
                "score_difference_mover":after.score[mover.index()] as i64-after.score[mover.opponent().index()] as i64,
                "basic_before_mover":p.reserve(mover).basic_count(),"basic_after_mover":q.reserve(mover).basic_count(),
                "components_before_host_guest":before.components(&p),"components_after_host_guest":after.components(&q),
                "edges_added":added.iter().map(|e|edge(e)).collect::<Vec<_>>(),
                "edges_removed":removed.iter().map(|e|edge(e)).collect::<Vec<_>>(),
                "ring_owners":after.rings.iter().map(|s|s.code().to_string()).collect::<Vec<_>>(),
                "cycle_witnesses":after.json(&q)["cycle_witnesses"],"threat":t,"tags":tags,
                "position_sha256":facts::position_hash(&q)});
            positions
                .entry(row["position_sha256"].as_str().unwrap().into())
                .or_default()
                .insert(held);
            for tag in &tags {
                let key = format!("{}:{tag}", if held { "heldout" } else { "train" });
                *counts.entry(key.clone()).or_default() += 1;
                groups
                    .entry(key)
                    .or_default()
                    .insert(root["group"].as_str().unwrap().into());
                if !["own_OffCentre", "own_TouchingCentre"].contains(&tag.as_str())
                    || (after.rings.is_empty() && q.outcome() == GameOutcome::Ongoing)
                {
                    categories.entry(tag.clone()).or_insert_with(|| row.clone());
                }
            }
            serde_json::to_writer(&mut rows, &row)?;
            writeln!(rows)?;
        }
        for (name, yes, no) in [
            ("defence", "threat_absent", "threat_present"),
            (
                "connection_consequence",
                "connection_immediate_win",
                "connection_without_immediate_win",
            ),
            ("ring_vs_off_centre", "own_ring", "own_OffCentre"),
            ("ring_vs_touching", "own_ring", "own_TouchingCentre"),
        ] {
            if let (Some(a), Some(b)) = (categories.get(yes), categories.get(no)) {
                // A non-centred witness can coexist with a winning ring elsewhere.
                // It is a counterexample only if that branch has no own ring.
                if name.starts_with("ring_vs") && !b["ring_owners"].as_array().unwrap().is_empty() {
                    continue;
                }
                pairs.push(
                    json!({"family":name,"root":root["key"],"held_out":held,"group":root["group"],
                    "first_action":a["action"],"second_action":b["action"]}),
                );
            }
        }
        if (ri + 1) % 16 == 0 {
            rows.flush()?;
            eprintln!(
                "{} roots / {actions_total} actions / {replies_total} replies",
                ri + 1
            );
        }
    }
    rows.flush()?;
    write_json(out.join("contrasts.json"), &json!(pairs))?;
    write_json(
        out.join("summary.json"),
        &json!({"roots":roots.len(),"actions":actions_total,
        "opponent_replies":replies_total,"counts":counts,"independent_groups":groups.iter()
        .map(|(k,v)|(k,v.len())).collect::<BTreeMap<_,_>>(),"same_root_contrasts":pairs.len(),
        "cross_split_exact_positions":positions.values().filter(|s|s.len()>1).count(),
        "census_roots_sha256":hash(&roots_bytes),"wall_seconds":start.elapsed().as_secs_f64(),
        "no_models_loaded":true,"not_a_production_cost_benchmark":true}),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn position_identity_is_independent_of_cached_reads() {
        let p = Position::from_standard_setup_with_rules(
            StandardSetup::balanced(BasicFlower::Red3),
            RuleProfileId::SkudPaiShoGen5V1,
        );
        let before = facts::position_hash(&p);
        let _ = legal_actions(&p);
        let _ = facts::Facts::new(&p);
        assert_eq!(before, facts::position_hash(&p));
    }
    #[test]
    fn incomplete_enumeration_is_unknown_not_safe() {
        let p = Position::from_standard_setup_with_rules(
            StandardSetup::balanced(BasicFlower::Red3),
            RuleProfileId::SkudPaiShoGen5V1,
        );
        assert_eq!(
            threat(&p, p.to_move().opponent(), 0).unwrap()["status"],
            "unknown"
        );
        assert_eq!(
            threat(&p, p.to_move().opponent(), usize::MAX).unwrap()["status"],
            "absent"
        );
    }
    #[test]
    fn legal_bonus_and_terminal_are_not_safe_labels() {
        let original: GameRecord =
            include_str!("../../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let (r, end) = original
            .replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)
            .unwrap();
        let mut p = r.initial_position();
        let mut bonuses = 0;
        for &a in r.actions() {
            let mover = p.to_move();
            p.apply(a).unwrap();
            if p.outcome() == GameOutcome::Ongoing && p.to_move() == mover {
                assert_eq!(p.phase(), TurnPhase::HarmonyBonus);
                assert_eq!(threat(&p, mover, 0).unwrap()["status"], "same_player_bonus");
                bonuses += 1;
            }
        }
        assert!(bonuses > 0);
        assert_ne!(end.outcome(), GameOutcome::Ongoing);
        assert_eq!(
            threat(&end, end.to_move(), 0).unwrap()["status"],
            "terminal"
        );
    }
}
