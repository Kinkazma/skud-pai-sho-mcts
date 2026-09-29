//! Same held-out source positions, fixed search budgets, alternating model order.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::gen32::Artifact;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::PathBuf,
    time::Instant,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() < 5 {
        return Err("usage: probe CATALOG.gz OUTPUT MODEL...".into());
    }
    let out = PathBuf::from(&a[2]);
    fs::create_dir(&out)?;
    let mut positions = vec![];
    for line in BufReader::new(flate2::read::GzDecoder::new(fs::File::open(&a[1])?)).lines() {
        let row: serde_json::Value = serde_json::from_str(&line?)?;
        if row["source"]["held_out"] != true {
            continue;
        }
        let record: GameRecord = row["psr"].as_str().unwrap().parse()?;
        let mut prefix = GameRecord::with_rules(record.setup(), record.rules());
        let mut p = record.initial_position();
        let mut got = [false; 2];
        for (i, &act) in record.actions().iter().enumerate() {
            let phase = usize::from(p.phase() == TurnPhase::HarmonyBonus);
            if i >= record.actions().len() / 3 && !got[phase] && p.outcome() == GameOutcome::Ongoing
            {
                fs::write(
                    out.join(format!("position-{}.psr", positions.len())),
                    prefix.to_string(),
                )?;
                positions.push((p.clone(), row["game"].clone(), i, phase));
                got[phase] = true;
            }
            p.apply(act).map_err(|e| format!("{e}"))?;
            prefix.push(act);
            if got == [true; 2] {
                break;
            }
        }
        if positions.len() >= 8 {
            break;
        }
    }
    let models: Vec<_> = a[3..]
        .iter()
        .map(|p| Artifact::load(std::path::Path::new(p)).and_then(|a| a.model()))
        .collect::<Result<_, _>>()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let mut rows = vec![];
    for round in 0..2 {
        let order: Vec<_> = if round == 0 {
            (0..models.len()).collect()
        } else {
            (0..models.len()).rev().collect()
        };
        for m in order {
            for budget in [64, 256, 512] {
                for (i, (p, source, decision, phase)) in positions.iter().enumerate() {
                    let bank = models[m].policy.sequence_memory().unwrap();
                    let before = bank.telemetry();
                    let clock = Instant::now();
                    let (selected, visits) = pool.install(|| -> Result<_, String> {
                        let mut s = MctsSession::new(
                            34011 + i as u64,
                            MctsConfig {
                                simulations: budget,
                                ..Default::default()
                            },
                            &models[m],
                        )?;
                        s.set_solver(true);
                        let legal = legal_actions(p);
                        let r = s.search_until(p, &legal, None)?;
                        Ok((
                            legal[r.selected_index].to_string(),
                            r.actions.iter().map(|a| a.visits).sum::<usize>(),
                        ))
                    })?;
                    let seconds = clock.elapsed().as_secs_f64();
                    let after = bank.telemetry();
                    let ctx = bank.context(&micro_state_features(p), 0);
                    rows.push(serde_json::json!({"round":round,"model":a[3+m],"budget":budget,"position":i,"source":source,"decision":decision,"phase":phase,"seconds":seconds,"selected":selected,"visits":visits,"queries":after[0]-before[0],"cache_hits":after[1]-before[1],"scanned":after[2]-before[2],"root_neighbors":ctx.neighbors.len()}));
                }
            }
            fs::write(out.join("progress.json"), serde_json::to_vec_pretty(&rows)?)?;
        }
    }
    fs::rename(out.join("progress.json"), out.join("results.json"))?;
    Ok(())
}
