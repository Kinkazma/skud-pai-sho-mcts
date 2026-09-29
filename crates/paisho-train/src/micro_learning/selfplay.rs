use super::*;
use crate::compact_selfplay::reuse::{RepetitionLoss, Repetitions};
use paisho_core::{GameOutcome, GameRecord, Player, Position, StandardSetup, BASIC_FLOWERS};
use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Arc, RwLock,
    },
    time::{Duration, Instant},
};

#[derive(Clone, Debug, Serialize)]
pub struct MicroSelfplayOptions {
    pub games: usize,
    /// Zero disables history; otherwise seconds between frozen sweeps.
    pub history_interval: f64,
    pub seconds: f64,
    pub game_seconds: f64,
    pub workers: usize,
    pub decision_limit: usize,
    pub low_budget: usize,
    pub high_budget: usize,
    pub high_fraction: f64,
    pub seed: u64,
    pub replay_capacity: usize,
    pub replay_ratio: usize,
    pub rate: f64,
    pub replay_input: Option<PathBuf>,
}
pub(super) struct Snapshot {
    pub(super) version: u64,
    pub(super) identity: String,
    model: Arc<MicroModel>,
}
struct Sample {
    saved: SavedMicroExample,
    player: Player,
    q: f64,
}
struct Played {
    error: Option<String>,
    id: usize,
    version: u64,
    record: GameRecord,
    samples: Vec<Sample>,
    seconds: f64,
    outcome: GameOutcome,
    cycle: Option<RepetitionLoss>,
    termination: String,
    simulations: usize,
    inherited: usize,
    evals: usize,
    hits: usize,
}

fn play(
    id: usize,
    snapshot: Arc<Snapshot>,
    frozen: Arc<Snapshot>,
    o: &MicroSelfplayOptions,
    end: Instant,
    pool: &rayon::ThreadPool,
    source: &str,
) -> Played {
    let started = Instant::now();
    let deadline = end.min(started + Duration::from_secs_f64(o.game_seconds));
    let mut rng = StableRng::new(o.seed.wrapping_add(id as u64));
    let setup = StandardSetup::balanced(BASIC_FLOWERS[id % BASIC_FLOWERS.len()]);
    let mut position = Position::from_standard_setup(setup);
    let mut record = GameRecord::new(setup);
    // One game in four includes the frozen human parent; swap its seat each time.
    let opponent = if id % 4 == 0 {
        frozen
    } else {
        snapshot.clone()
    };
    let candidate_seat = if id / 4 % 2 == 0 {
        Player::Host
    } else {
        Player::Guest
    };
    let same_model = Arc::ptr_eq(&snapshot.model, &opponent.model);
    let mut sessions = if same_model {
        vec![MicroMctsSession::new(snapshot.model.clone())]
    } else {
        vec![
            MicroMctsSession::new(if candidate_seat == Player::Host {
                snapshot.model.clone()
            } else {
                opponent.model.clone()
            }),
            MicroMctsSession::new(if candidate_seat == Player::Guest {
                snapshot.model.clone()
            } else {
                opponent.model.clone()
            }),
        ]
    };
    let mut repetitions = Repetitions::new(&position, 4);
    let mut cycle = None;
    let mut samples = Vec::new();
    let mut simulations = 0;
    let mut inherited = 0;
    let mut evals = 0;
    let mut hits = 0;
    let mut termination = "decision-limit".to_string();
    let result = (|| -> std::result::Result<(), String> {
        for decision in 0..o.decision_limit {
            if Instant::now() >= deadline {
                termination = "wall-limit".into();
                break;
            }
            if position.outcome() != GameOutcome::Ongoing {
                termination = "rules-terminal".into();
                break;
            }
            let player = position.to_move();
            let deep = rng.next_f64() < o.high_fraction;
            let budget = if deep { o.high_budget } else { o.low_budget };
            let report = pool.install(|| {
                sessions[if same_model { 0 } else { player.index() }].search_until(
                    &position,
                    budget,
                    Some(deadline),
                )
            });
            let report = match report {
                Ok(report) => report,
                Err(e) if e == "PUCT root has no legal action" => {
                    termination = "no-legal-action".into();
                    break;
                }
                Err(e) => return Err(e),
            };
            if report.simulations == 0 {
                termination = "wall-limit".into();
                break;
            }
            simulations += report.simulations;
            inherited += report.inherited_visits;
            evals += report.inference_evaluations;
            hits += report.inference_cache_hits;
            // Sampling visits early supplies varied standard-start selfplay. Later
            // choices use the highest visits; retained priors are never overwritten.
            let selected = if decision < 40 {
                let total: usize = report.visits.iter().sum();
                let mut draw = rng.index(total);
                let mut pick = report.selected_index;
                for (i, n) in report.visits.iter().enumerate() {
                    if draw < *n {
                        pick = i;
                        break;
                    }
                    draw -= *n;
                }
                pick
            } else {
                report.selected_index
            };
            let action = report.actions[selected];
            let policy_weight = if deep && report.simulations == budget {
                1.0
            } else {
                0.0
            };
            let collector = if player == candidate_seat {
                &snapshot.identity
            } else {
                &opponent.identity
            };
            let saved = SavedMicroExample {
                tactical: None,
                correction_priority: false,
                policy_raw_visits: vec![],
                policy_pruned_visits: vec![],
                rules: position.rule_profile().to_string(),
                source_run: source.into(),
                game_id: id.to_string(),
                decision: decision + 1,
                collector: collector.clone(),
                budget,
                inherited_visits: report.inherited_visits,
                new_visits: if policy_weight > 0.0 {
                    report.new_visits.clone()
                } else {
                    vec![]
                },
                actions: if policy_weight > 0.0 {
                    report.actions.iter().map(ToString::to_string).collect()
                } else {
                    vec![]
                },
                state: report.state.to_vec(),
                action_features: if policy_weight > 0.0 {
                    report.action_features.iter().map(|x| x.to_vec()).collect()
                } else {
                    vec![]
                },
                policy: if policy_weight > 0.0 {
                    let n: usize = report.visits.iter().sum();
                    report.visits.iter().map(|v| *v as f64 / n as f64).collect()
                } else {
                    vec![]
                },
                value: 0.0,
                policy_weight,
                reason: String::new(),
            };
            samples.push(Sample {
                saved,
                player,
                q: report.values[selected],
            });
            position.apply(action).map_err(|e| e.to_string())?;
            record.push(action);
            pool.install(|| -> std::result::Result<(), String> {
                for session in &mut sessions {
                    session.advance(action)?;
                }
                Ok(())
            })?;
            cycle = repetitions.observe(&position, player, record.actions().len());
            if cycle.is_some() {
                termination = "repetition-training-loss".into();
                break;
            }
        }
        Ok(())
    })();
    let error = result.err();
    if error.is_some() {
        termination = "engine-error".into();
    } else if position.outcome() != GameOutcome::Ongoing {
        termination = "rules-terminal".into();
    }
    Played {
        error,
        id,
        version: snapshot.version,
        record,
        samples,
        seconds: started.elapsed().as_secs_f64(),
        outcome: position.outcome(),
        cycle,
        termination,
        simulations,
        inherited,
        evals,
        hits,
    }
}
fn targets(game: &mut Played) -> Vec<SavedMicroExample> {
    if game.error.is_some() {
        return Vec::new();
    }
    game.samples
        .drain(..)
        .filter_map(|mut s| {
            if let Some(cycle) = &game.cycle {
                if s.player != cycle.loser.player() || s.saved.decision < cycle.first_decision {
                    return None;
                }
                // A behavioral loss must not imitate the repeated policy that caused it.
                s.saved.value = -1.0;
                s.saved.policy_weight = 0.0;
                s.saved.policy.clear();
                s.saved.actions.clear();
                s.saved.action_features.clear();
                s.saved.new_visits.clear();
                s.saved.reason = "repetition-training-loss".into();
            } else {
                let terminal = match game.outcome {
                    GameOutcome::Win(p) => {
                        if p == s.player {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    GameOutcome::Draw => 0.0,
                    GameOutcome::Ongoing => return None,
                };
                s.saved.value = 0.5 * s.q + 0.5 * terminal;
                s.saved.reason = "rules-terminal-q-mix".into();
            }
            Some(s.saved)
        })
        .collect()
}

pub fn run_micro_selfplay(
    model_path: &Path,
    output: &Path,
    o: MicroSelfplayOptions,
    pool: Arc<rayon::ThreadPool>,
) -> Result<()> {
    if !o.history_interval.is_finite()
        || o.history_interval < 0.0
        || o.games == 0
        || o.workers == 0
        || o.decision_limit == 0
        || o.low_budget == 0
        || o.high_budget < o.low_budget
        || !(0.0..=1.0).contains(&o.high_fraction)
        || !o.seconds.is_finite()
        || o.seconds <= 0.0
        || !o.game_seconds.is_finite()
        || o.game_seconds <= 0.0
        || !o.rate.is_finite()
        || o.rate <= 0.0
    {
        return Err(invalid("invalid bounded Gen4 selfplay options"));
    }
    let parent = MicroArtifact::load(model_path)?;
    let mut model = parent.model()?;
    fs::create_dir(output)?;
    let source = output.canonicalize()?.to_string_lossy().into_owned();
    parent.save(&output.join("parent.json"))?;
    save_json_new(
        &output.join("plan.json"),
        &serde_json::json!({"schema":"paisho-micro-selfplay-v1","rules":paisho_core::RuleProfileId::CURRENT.as_str(),"options":o,"parent":parent.identity(),"cpu_capacity":pool.current_num_threads(),"automatic_resume":false}),
    )?;
    let first = Arc::new(Snapshot {
        version: 0,
        identity: parent.identity(),
        model: Arc::new(model.clone()),
    });
    let shared = Arc::new(RwLock::new(first.clone()));
    let next = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let mut replay: VecDeque<(SavedMicroExample, Arc<MicroExample>)> = VecDeque::new();
    if let Some(path) = &o.replay_input {
        let saved = load_examples(path)?;
        for s in saved {
            let ex = s.example()?;
            replay.push_back((s, Arc::new(ex)));
            while replay.len() > o.replay_capacity {
                replay.pop_front();
            }
        }
    }
    let started = Instant::now();
    let end = started + Duration::from_secs_f64(o.seconds);
    let history_error = Arc::new(RwLock::new(None::<String>));
    let history = if o.history_interval > 0.0 {
        let (output, shared, pool, stop, interval) = (
            output.to_path_buf(),
            shared.clone(),
            pool.clone(),
            stop.clone(),
            Duration::from_secs_f64(o.history_interval),
        );
        let history_error = history_error.clone();
        Some(std::thread::spawn(move || {
            let result = super::history::run(output, shared, pool, stop.clone(), end, interval);
            if let Err(e) = &result {
                *history_error.write().unwrap() = Some(e.clone());
                stop.store(true, Ordering::Relaxed);
            }
            result
        }))
    } else {
        None
    };
    let (sender, receiver) = mpsc::sync_channel(o.workers * 2);
    let mut actors = Vec::new();
    for _ in 0..o.workers {
        let (shared, next, stop, tx, pool, o, source, frozen) = (
            shared.clone(),
            next.clone(),
            stop.clone(),
            sender.clone(),
            pool.clone(),
            o.clone(),
            source.clone(),
            first.clone(),
        );
        actors.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) && Instant::now() < end {
                let id = next.fetch_add(1, Ordering::Relaxed);
                if id >= o.games {
                    break;
                }
                let snapshot = shared.read().unwrap().clone();
                let game = play(id, snapshot, frozen.clone(), &o, end, &pool, &source);
                if tx.send(game).is_err() {
                    break;
                }
            }
        }));
    }
    drop(sender);
    let mut updates = parent.updates;
    let mut version = 0;
    let mut rng = StableRng::new(o.seed ^ 0x7265706c6179);
    let mut completed = 0;
    let mut terminal = 0;
    let mut errors = Vec::new();
    let result = (|| -> Result<()> {
        for mut game in receiver.iter() {
            if let Some(error) = &game.error {
                errors.push(error.clone());
                stop.store(true, Ordering::Relaxed);
            }
            completed += 1;
            if game.outcome != GameOutcome::Ongoing {
                terminal += 1;
            }
            let psr = game.record.to_string();
            let stem = format!("game-{:06}", game.id);
            let dir = output.join("games");
            fs::create_dir_all(&dir)?;
            let record_path = dir.join(format!("{stem}.psr"));
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&record_path)?;
            use std::io::Write;
            f.write_all(psr.as_bytes())?;
            f.sync_all()?;
            let fresh = targets(&mut game);
            let target_path = dir.join(format!("{stem}.targets.json.gz"));
            save_examples_new(&target_path, &fresh)?;
            let mut owned: Vec<Arc<MicroExample>> = fresh
                .iter()
                .map(|s| s.example().map(Arc::new))
                .collect::<Result<_>>()?;
            for _ in 0..fresh.len().saturating_mul(o.replay_ratio) {
                if !replay.is_empty() {
                    owned.push(replay[rng.index(replay.len())].1.clone());
                }
            }
            let before = updates;
            // Respect the run deadline even while draining completed actor results.
            for batch in owned.chunks(64) {
                if Instant::now() >= end {
                    break;
                }
                let refs: Vec<_> = batch.iter().map(Arc::as_ref).collect();
                pool.install(|| model.train_batch(&refs, o.rate, 1e-5))
                    .map_err(invalid)?;
                updates += 1;
            }
            for saved in fresh {
                let ex = saved.example()?;
                replay.push_back((saved, Arc::new(ex)));
                while replay.len() > o.replay_capacity {
                    replay.pop_front();
                }
            }
            if updates > before {
                version += 1;
                let artifact = MicroArtifact::new(
                    &model,
                    updates,
                    serde_json::json!({"kind":"mixed-budget-selfplay","rules":paisho_core::RuleProfileId::CURRENT.as_str(),"source_run":source,"parent":parent.identity(),"version":version,"last_game":game.id}),
                );
                artifact.save(
                    &output
                        .join("models")
                        .join(format!("model-{version:06}.json")),
                )?;
                *shared.write().unwrap() = Arc::new(Snapshot {
                    version,
                    identity: artifact.identity(),
                    model: Arc::new(model.clone()),
                });
            }
            let receipt = serde_json::json!({"rules":game.record.rules().as_str(),"id":game.id,"collector_version":game.version,"termination":game.termination,"error":game.error,"outcome":format!("{:?}",game.outcome),"cycle":game.cycle,"decisions":game.record.actions().len(),"seconds":game.seconds,"simulations":game.simulations,"inherited_visits":game.inherited,"inference_evaluations":game.evals,"inference_cache_hits":game.hits,"psr_sha256":sha256(psr.as_bytes()),"targets_file":format!("{stem}.targets.json.gz"),"targets_sha256":sha256(&fs::read(&target_path)?),"model_version":version,"updates":updates,"learned_batches":updates-before});
            save_json_new(&dir.join(format!("{stem}.json")), &receipt)?;
            let progress = serde_json::json!({"completed":completed,"terminal":terminal,"published_version":version,"updates":updates,"elapsed_seconds":started.elapsed().as_secs_f64(),"remaining_seconds":end.saturating_duration_since(Instant::now()).as_secs_f64(),"cpu_capacity":pool.current_num_threads(),"actors":o.workers,"history_error":*history_error.read().unwrap(),"errors":errors});
            let temporary = output.join("progress.tmp");
            fs::write(&temporary, serde_json::to_vec(&progress)?)?;
            fs::rename(temporary, output.join("progress.json"))?;
            println!("{receipt}");
        }
        Ok(())
    })();
    stop.store(true, Ordering::Relaxed);
    drop(receiver);
    for actor in actors {
        if actor.join().is_err() {
            errors.push("actor panic".into());
        }
    }
    if let Some(history) = history {
        match history.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => errors.push(format!("history: {e}")),
            Err(_) => errors.push("history coordinator panic".into()),
        }
    }
    result?;
    let final_artifact = MicroArtifact::new(
        &model,
        updates,
        serde_json::json!({"kind":"bounded-selfplay-final","rules":paisho_core::RuleProfileId::CURRENT.as_str(),"source_run":source,"parent":parent.identity(),"version":version}),
    );
    final_artifact.save(&output.join("model.json"))?;
    save_examples_new(
        &output.join("replay-final.json.gz"),
        &replay.into_iter().map(|(s, _)| s).collect::<Vec<_>>(),
    )?;
    save_json_new(
        &output.join("report.json"),
        &serde_json::json!({"rules":paisho_core::RuleProfileId::CURRENT.as_str(),"completed":completed,"terminal":terminal,"version":version,"updates":updates,"elapsed_seconds":started.elapsed().as_secs_f64(),"errors":errors,"automatic_resume":false,"force":"not evaluated"}),
    )?;
    if !errors.is_empty() {
        return Err(invalid(
            "Gen4 selfplay encountered actor errors; see report",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn game(outcome: GameOutcome) -> Played {
        let ex = SavedMicroExample {
            tactical: None,
            correction_priority: false,
            policy_raw_visits: vec![],
            policy_pruned_visits: vec![],
            rules: paisho_core::RuleProfileId::CURRENT.to_string(),
            source_run: "run".into(),
            game_id: "1".into(),
            decision: 30,
            collector: "a".repeat(64),
            budget: 64,
            inherited_visits: 0,
            new_visits: vec![],
            actions: vec![],
            state: vec![0.0; 128],
            action_features: vec![],
            policy: vec![],
            value: 0.0,
            policy_weight: 0.0,
            reason: String::new(),
        };
        Played {
            error: None,
            id: 1,
            version: 0,
            record: GameRecord::new(StandardSetup::balanced(BASIC_FLOWERS[0])),
            samples: vec![
                Sample {
                    saved: ex.clone(),
                    player: Player::Host,
                    q: 0.2,
                },
                Sample {
                    saved: ex,
                    player: Player::Guest,
                    q: -0.2,
                },
            ],
            seconds: 1.0,
            outcome,
            cycle: None,
            termination: "test".into(),
            simulations: 128,
            inherited: 0,
            evals: 0,
            hits: 0,
        }
    }
    #[test]
    fn truncation_has_no_fabricated_draw_and_terminal_targets_keep_perspective() {
        assert!(targets(&mut game(GameOutcome::Ongoing)).is_empty());
        let t = targets(&mut game(GameOutcome::Win(Player::Host)));
        assert_eq!(t[0].value, 0.6);
        assert_eq!(t[1].value, -0.6);
    }
    #[test]
    fn cycle_penalizes_only_closing_player_without_imitation_target() {
        let mut g = game(GameOutcome::Ongoing);
        g.cycle = Some(RepetitionLoss {
            loser: crate::compact_selfplay::Seat::Host,
            first_decision: 24,
            last_decision: 31,
            period: 6,
            cycles: 4,
        });
        let t = targets(&mut g);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].value, -1.0);
        assert_eq!(t[0].policy_weight, 0.0);
        assert!(t[0].policy.is_empty());
    }
}
