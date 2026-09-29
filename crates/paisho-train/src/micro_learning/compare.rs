//! Frozen paired Gen4/Gen3 matches. Explicit limits never become rules draws.
use super::*;
use crate::compact_learning::load_model;
use paisho_core::{legal_actions, GameOutcome, GameRecord, Player, StandardSetup, BASIC_FLOWERS};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct MicroCompareOptions {
    pub pairs: usize,
    /// Concurrent games, sharing the caller's CPU pool.
    pub workers: usize,
    pub varied_setups: bool,
    pub simulations: usize,
    pub move_ms: u64,
    pub game_seconds: f64,
    pub seconds: f64,
    pub decisions: usize,
    pub seed: u64,
}
pub fn compare_micro(
    candidate: &Path,
    reference: &Path,
    output: &Path,
    o: MicroCompareOptions,
    pool: &rayon::ThreadPool,
) -> Result<()> {
    compare_micro_profile(
        candidate,
        reference,
        output,
        o,
        pool,
        paisho_core::RuleProfileId::CURRENT,
        MicroSearchOptions::default(),
        MicroSearchOptions::default(),
        o.simulations,
    )
}
pub fn compare_micro_profile(
    candidate: &Path,
    reference: &Path,
    output: &Path,
    o: MicroCompareOptions,
    pool: &rayon::ThreadPool,
    rules: paisho_core::RuleProfileId,
    candidate_search: MicroSearchOptions,
    reference_search: MicroSearchOptions,
    reference_budget: usize,
) -> Result<()> {
    candidate_search.validate().map_err(invalid)?;
    reference_search.validate().map_err(invalid)?;
    if reference_budget == 0 {
        return Err(invalid("zero reference budget"));
    }
    if o.workers == 0
        || o.pairs == 0
        || o.simulations == 0
        || o.decisions == 0
        || !o.seconds.is_finite()
        || o.seconds <= 0.0
        || !o.game_seconds.is_finite()
        || o.game_seconds <= 0.0
    {
        return Err(invalid("invalid bounded micro comparison"));
    }
    let artifact = MicroArtifact::load(candidate)?;
    let candidate = Arc::new(artifact.model()?);
    let reference_bytes = fs::read(reference)?;
    let reference_spec: serde_json::Value = serde_json::from_slice(&reference_bytes)?;
    let random_reference = reference_spec["schema"] == "paisho-random-reference-v1";
    if random_reference && reference_spec["selection"] != "uniform-legal-decisions" {
        return Err(invalid("unsupported random reference selection"));
    }
    let micro_reference = if reference_spec["schema"] == MICRO_MODEL_SCHEMA
        || reference_spec["schema"] == MICRO_RESIDUAL_MODEL_SCHEMA
        || reference_spec["schema"] == MICRO_MEMORY_MODEL_SCHEMA
    {
        Some(Arc::new(MicroArtifact::load(reference)?.model()?))
    } else {
        None
    };
    let old = if micro_reference.is_none() && !random_reference {
        Some(load_model(reference)?.model()?)
    } else {
        None
    };
    fs::create_dir(output)?;
    save_json_new(
        &output.join("plan.json"),
        &serde_json::json!({"schema":"paisho-micro-frozen-compare-v2","rules":rules.as_str(),"options":o,"candidate_search":format!("{candidate_search:?}"),"reference_search":format!("{reference_search:?}"),"reference_budget":if random_reference{None}else{Some(reference_budget)},"candidate":artifact.identity(),"reference_sha256":sha256(&reference_bytes),"reference_kind":if random_reference{"uniform-random"}else if micro_reference.is_some(){"micro-gen4"}else{"compact-gen3"},"reference_retained":!random_reference,"shared_cpu_capacity":pool.current_num_threads(),"pairing":"same setup, exchanged seats; analyze complete pairs","promotion":false,"timing":"soft per-move deadline plus simulation ceiling; report actual time, not claimed equal realized time"}),
    )?;
    let started = Instant::now();
    let end = started + Duration::from_secs_f64(o.seconds);
    let play_game = |id| -> std::result::Result<serde_json::Value, String> {
        if Instant::now() >= end {
            return Ok(serde_json::json!({"id":id,"pair":id/2,"termination":"unplayed"}));
        }
        let t = Instant::now();
        let deadline = end.min(t + Duration::from_secs_f64(o.game_seconds));
        let setup = comparison_setup(id / 2, o.seed, o.varied_setups);
        let mut record = GameRecord::with_rules(setup, rules);
        let mut position = record.initial_position();
        let seat = if id % 2 == 0 {
            Player::Host
        } else {
            Player::Guest
        };
        let mut micro = MicroMctsSession::new(candidate.clone());
        let mut random = RandomAgent::new(o.seed.wrapping_add((id / 2) as u64));
        let mut reference_micro = micro_reference
            .as_ref()
            .map(|m| MicroMctsSession::new(m.clone()));
        let mut legacy = old
            .as_ref()
            .map(|m| {
                MctsSession::new(
                    o.seed.wrapping_add((id / 2) as u64),
                    MctsConfig {
                        simulations: reference_budget,
                        ..MctsConfig::default()
                    },
                    m,
                )
            })
            .transpose()?;
        let mut clocks = [0.0; 2];
        let mut maintenance = [0.0; 2];
        let mut calls = [0usize; 2];
        let mut counts = [0usize; 2];
        let mut simulations = [0usize; 2];
        let mut termination = "decision-limit".to_string();
        let mut error = None;
        let play = (|| -> std::result::Result<(), String> {
            for _ in 0..o.decisions {
                if Instant::now() >= deadline {
                    termination = "wall-limit".into();
                    break;
                }
                if position.outcome() != GameOutcome::Ongoing {
                    termination = "rules-terminal".into();
                    break;
                }
                let candidate_turn = position.to_move() == seat;
                let side = usize::from(!candidate_turn);
                let before = Instant::now();
                let move_deadline = if o.move_ms > 0 {
                    deadline.min(before + Duration::from_millis(o.move_ms))
                } else {
                    deadline
                };
                let attempt = pool.install(
                    || -> std::result::Result<Option<paisho_core::Action>, String> {
                        Ok(if candidate_turn {
                            let report = micro.search_with_options(
                                &position,
                                o.simulations,
                                Some(move_deadline),
                                candidate_search,
                            )?;
                            simulations[side] += report.simulations;
                            if report.simulations == 0 && Instant::now() >= deadline {
                                None
                            } else {
                                Some(report.actions[report.selected_index])
                            }
                        } else {
                            if let Some(reference) = reference_micro.as_mut() {
                                let report = reference.search_with_options(
                                    &position,
                                    reference_budget,
                                    Some(move_deadline),
                                    reference_search,
                                )?;
                                simulations[side] += report.simulations;
                                if report.simulations == 0 && Instant::now() >= deadline {
                                    None
                                } else {
                                    Some(report.actions[report.selected_index])
                                }
                            } else if random_reference {
                                let actions = legal_actions(&position);
                                if actions.is_empty() {
                                    return Err("PUCT root has no legal action".into());
                                }
                                let index = random
                                    .select_action(&position, &actions)
                                    .map_err(|e| e.to_string())?;
                                Some(actions[index])
                            } else {
                                let actions = legal_actions(&position);
                                let report = legacy.as_mut().unwrap().search_until(
                                    &position,
                                    &actions,
                                    Some(move_deadline),
                                )?;
                                simulations[side] += report.simulations;
                                Some(actions[report.selected_index])
                            }
                        })
                    },
                );
                clocks[side] += before.elapsed().as_secs_f64();
                calls[side] += 1;
                let attempt = match attempt {
                    Err(e) if e == "PUCT root has no legal action" => {
                        termination = "no-legal-action".into();
                        break;
                    }
                    other => other,
                };
                let Some(action) = attempt? else {
                    termination = "wall-limit".into();
                    break;
                };
                counts[side] += 1;
                position.apply(action).map_err(|e| e.to_string())?;
                record.push(action);
                let maintenance_start = Instant::now();
                let advanced = pool.install(|| micro.advance(action));
                maintenance[0] += maintenance_start.elapsed().as_secs_f64();
                advanced?;
                let maintenance_start = Instant::now();
                if let Some(r) = reference_micro.as_mut() {
                    pool.install(|| r.advance(action))?;
                } else if let Some(r) = &mut legacy {
                    r.advance(action);
                }
                maintenance[1] += maintenance_start.elapsed().as_secs_f64();
            }
            Ok(())
        })();
        if let Err(e) = play {
            termination = "engine-error".into();
            error = Some(e);
        } else if position.outcome() != GameOutcome::Ongoing {
            termination = "rules-terminal".into();
        }
        let score = if error.is_some() {
            None
        } else {
            match position.outcome() {
                GameOutcome::Win(p) => Some(if p == seat { 1.0 } else { 0.0 }),
                GameOutcome::Draw => Some(0.5),
                GameOutcome::Ongoing => None,
            }
        };
        let text = record.to_string();
        let path = output.join(format!("game-{id:04}.psr"));
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|e| e.to_string())?;
        use std::io::Write;
        file.write_all(text.as_bytes()).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        let row = serde_json::json!({"rules":record.rules().as_str(),"id":id,"pair":id/2,"candidate_seat":seat.code().to_string(),"termination":termination,"error":error,"score":score,"decisions":record.actions().len(),"seconds":t.elapsed().as_secs_f64(),"search_seconds":clocks,"search_calls":calls,"maintenance_seconds":maintenance,"search_decisions":counts,"simulations":simulations,"psr_sha256":sha256(text.as_bytes())});
        save_json_new(&output.join(format!("game-{id:04}.json")), &row)
            .map_err(|e| e.to_string())?;
        println!("{row}");
        Ok(row)
    };
    // Coordinators do not occupy Rayon lanes while waiting. Each search enters
    // the same pool as selfplay; no second compute pool is created.
    let next = std::sync::atomic::AtomicUsize::new(0);
    let rows = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..o.workers.min(o.pairs * 2))
            .map(|_| {
                scope.spawn(|| {
                    let mut rows = Vec::new();
                    loop {
                        let id = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if id >= o.pairs * 2 {
                            break;
                        }
                        rows.push((id, play_game(id)));
                    }
                    rows
                })
            })
            .collect();
        let mut rows: Vec<_> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("comparison coordinator panic"))
            .collect();
        rows.sort_by_key(|(id, _)| *id);
        rows.into_iter().map(|(_, row)| row).collect::<Vec<_>>()
    });
    let rows = rows
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(invalid)?;
    let wins = rows
        .iter()
        .filter(|r| r["score"].as_f64() == Some(1.0))
        .count();
    let draws = rows
        .iter()
        .filter(|r| r["score"].as_f64() == Some(0.5))
        .count();
    let losses = rows
        .iter()
        .filter(|r| r["score"].as_f64() == Some(0.0))
        .count();
    let complete_pairs = rows
        .chunks_exact(2)
        .filter(|p| p.iter().all(|r| r["score"].as_f64().is_some()))
        .count();
    save_json_new(
        &output.join("report.json"),
        &serde_json::json!({"rules":rules.as_str(),"wins":wins,"draws":draws,"losses":losses,"unknown":rows.len()-wins-draws-losses,"complete_pairs":complete_pairs,"elapsed_seconds":started.elapsed().as_secs_f64(),"games":rows,"claim":"bounded comparison only; no automatic promotion or site Elo"}),
    )?;
    Ok(())
}

/// Distinct legal standard setups: six flowers times 19 legal accent loadouts.
/// Both seats receive the same loadout; swapping agents preserves the setup.
pub(super) fn comparison_setup(pair: usize, seed: u64, varied: bool) -> StandardSetup {
    if !varied {
        return StandardSetup::balanced(BASIC_FLOWERS[pair % 6]);
    }
    let mut loadouts = Vec::new();
    for r in 0..=2 {
        for w in 0..=2 {
            for k in 0..=2 {
                for b in 0..=2 {
                    if let Ok(a) = paisho_core::AccentLoadout::new(r, w, k, b) {
                        loadouts.push(a);
                    }
                }
            }
        }
    }
    let index = (pair + (seed % 114) as usize) % 114;
    StandardSetup {
        starting_flower: BASIC_FLOWERS[index % 6],
        host_accents: loadouts[index / 6],
        guest_accents: loadouts[index / 6],
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn random_reference_plays_only_its_seat_and_records_unknown_caps() {
        let root = std::env::temp_dir().join(format!("paisho-micro-random-{}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let artifact =
            MicroArtifact::new(&MicroModel::seeded(6), 0, serde_json::json!({"test":true}));
        let path = root.join("model.json");
        artifact.save(&path).unwrap();
        let reference = root.join("random.json");
        save_json_new(&reference,&serde_json::json!({"schema":"paisho-random-reference-v1","selection":"uniform-legal-decisions"})).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let out = root.join("matches");
        let options = MicroCompareOptions {
            pairs: 1,
            workers: 1,
            varied_setups: true,
            simulations: 1,
            move_ms: 0,
            game_seconds: 30.0,
            seconds: 60.0,
            decisions: 8,
            seed: 1934,
        };
        compare_micro_profile(
            &path,
            &reference,
            &out,
            options,
            &pool,
            gen5::RULES,
            MicroSearchOptions::default(),
            MicroSearchOptions::default(),
            1,
        )
        .unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(out.join("report.json")).unwrap()).unwrap();
        assert_eq!(report["unknown"], 2);
        for id in 0..2 {
            let r: GameRecord = fs::read_to_string(out.join(format!("game-{id:04}.psr")))
                .unwrap()
                .parse()
                .unwrap();
            let mut position = r.initial_position();
            let seat = if id == 0 { Player::Host } else { Player::Guest };
            let mut random = RandomAgent::new(options.seed);
            let mut model = MicroMctsSession::new(Arc::new(artifact.model().unwrap()));
            for action in r.actions() {
                let expected = if position.to_move() == seat {
                    let s = model.search_until(&position, 1, None).unwrap();
                    s.actions[s.selected_index]
                } else {
                    let legal = legal_actions(&position);
                    legal[random.select_action(&position, &legal).unwrap()]
                };
                assert_eq!(*action, expected);
                position.apply(*action).unwrap();
                model.advance(*action).unwrap();
            }
            assert_eq!(report["games"][id]["simulations"][1], 0);
            assert!(report["games"][id]["simulations"][0].as_u64().unwrap() > 0);
        }
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn varied_standard_pairs_are_unique() {
        let starts: Vec<_> = (0..114).map(|i| comparison_setup(i, 87000, true)).collect();
        for (i, a) in starts.iter().enumerate() {
            assert!(!starts[..i].contains(a));
        }
    }
}
