//! Finite legal alternatives to the penultimate decision of human ring endings.
use super::{
    facts::{self, Facts},
    geometry, hash, write_json, Result,
};
use paisho_core::*;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
    time::Instant,
};

struct Choice {
    win: Action,
    negative: Action,
    distance: [usize; 5],
}
fn candidates(p: &Position, actions: &[Action]) -> Result<BTreeMap<&'static str, Choice>> {
    let mover = p.to_move();
    let mut winners = vec![];
    for (i, &a) in actions.iter().enumerate() {
        let mut q = p.clone();
        q.apply(a)?;
        if q.outcome() == GameOutcome::Win(mover)
            && harmony_ring_owners_for_profile(q.board(), q.rule_profile()).contains(&mover)
        {
            let f = Facts::new(&q);
            winners.push((i, a, q, f));
        }
    }
    let mut best = BTreeMap::<_, Choice>::new();
    if winners.is_empty() {
        return Ok(best);
    }
    let old = harmonies(p.board());
    for (j, &a) in actions.iter().enumerate() {
        let mut q = p.clone();
        q.apply(a)?;
        if q.outcome() != GameOutcome::Ongoing {
            continue;
        }
        let f = Facts::new(&q);
        if !f.rings.is_empty() {
            continue;
        }
        let kinds: BTreeSet<_> = f
            .rel
            .cycles
            .iter()
            .filter(|c| c.owner == mover && geometry::created(c, &old))
            .map(|c| geometry::kind(c.geometry))
            .collect();
        let ng = f.owner_graphs(&q)[mover.index()];
        for k in kinds {
            if k == "enclosing" {
                return Err("ongoing ring".into());
            }
            for (i, win, w, wf) in &winners {
                let wg = wf.owner_graphs(w)[mover.index()];
                let distance = [
                    w.board()
                        .occupied_count()
                        .abs_diff(q.board().occupied_count()),
                    wg[1].abs_diff(ng[1]),
                    wg[3].abs_diff(ng[3]),
                    *i,
                    j,
                ];
                if best.get(k).map_or(true, |c| distance < c.distance) {
                    best.insert(
                        k,
                        Choice {
                            win: *win,
                            negative: a,
                            distance,
                        },
                    );
                }
            }
        }
    }
    Ok(best)
}
pub fn mine(manifest: &str, out: &Path) -> Result<()> {
    fs::create_dir(out)?;
    fs::create_dir(out.join("psr"))?;
    fs::write(out.join("states.jsonl"), b"")?;
    let start = Instant::now();
    let bytes = fs::read(manifest)?;
    let d: Value = serde_json::from_slice(&bytes)?;
    let mut games: Vec<_> = d["games"].as_array().ok_or("games")?.iter().collect();
    games.sort_by_key(|g| g["game_sha256"].as_str().unwrap());
    let mut writer = std::io::BufWriter::new(fs::File::create(out.join("pairs.jsonl"))?);
    let mut quota = BTreeMap::<String, usize>::new();
    let mut groups = BTreeMap::<String, BTreeSet<String>>::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut seen = BTreeSet::new();
    let mut actions_total = 0u64;
    let mut roots = 0u64;
    let mut eligible = 0;
    let mut pairs = 0;
    for g in &games {
        let orig = &g["originals"][0];
        let b = fs::read(orig["path"].as_str().ok_or("path")?)?;
        if hash(&b) != orig["sha256"] {
            return Err("source changed".into());
        }
        let old: GameRecord = std::str::from_utf8(&b)?.parse()?;
        let (r, end) = old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?;
        if r.actions().len() < 2
            || end.outcome() == GameOutcome::Ongoing
            || harmony_ring_owners_for_profile(end.board(), end.rule_profile()).is_empty()
        {
            continue;
        }
        eligible += 1;
        let n = r.actions().len() - 2;
        let mut prefix = GameRecord::with_rules(r.setup(), r.rules());
        for &a in &r.actions()[..n] {
            prefix.push(a);
        }
        let base = prefix.replay()?;
        let group = g["split_identity_sha256"].as_str().ok_or("group")?;
        for alternative in legal_actions(&base) {
            if alternative == r.actions()[n] {
                continue;
            }
            let mut p = base.clone();
            p.apply(alternative)?;
            if p.outcome() != GameOutcome::Ongoing
                || !seen.insert((group.to_owned(), facts::position_hash(&p)))
            {
                continue;
            }
            roots += 1;
            let actions = legal_actions(&p);
            actions_total += actions.len() as u64;
            for (family, c) in candidates(&p, &actions)? {
                let key = format!(
                    "{}:{family}",
                    if g["held_out"] == true {
                        "heldout"
                    } else {
                        "train"
                    }
                );
                *counts.entry(key.clone()).or_default() += 1;
                groups.entry(key).or_default().insert(group.to_owned());
                let q = quota.entry(format!("{group}:{family}")).or_default();
                *q += 1;
                if *q > 2 {
                    continue;
                }
                let mut root = prefix.clone();
                root.push(alternative);
                let value = json!({"source":g["game_sha256"],"group":group,"held_out":g["held_out"],
                    "family":family,"new_cycle":true,"decision":n+1,"source_prefix_decision":n,
                    "extension_actions":[alternative.to_string()],"root":geometry::persist(out,&root)?,
                    "winning_action":c.win.to_string(),"nonwinning_action":c.negative.to_string(),
                    "mover":p.to_move().code().to_string(),"matching_distance":c.distance});
                serde_json::to_writer(&mut writer, &value)?;
                writeln!(writer)?;
                pairs += 1;
            }
        }
        if eligible % 50 == 0 {
            writer.flush()?;
            eprintln!("{eligible} games / {roots} legal neighbours / {actions_total} actions / {pairs} pairs / {:.1}s",start.elapsed().as_secs_f64());
        }
    }
    writer.flush()?;
    write_json(
        out.join("summary.json"),
        &json!({"manifest_sha256":hash(&bytes),"eligible_games":eligible,
        "neighbour_roots":roots,"legal_actions_examined":actions_total,"exported_pairs":pairs,"counts":counts,
        "groups":groups.iter().map(|(k,v)|(k,v.len())).collect::<BTreeMap<_,_>>(),
        "seconds":start.elapsed().as_secs_f64(),"finite_population_complete":true}),
    )
}
