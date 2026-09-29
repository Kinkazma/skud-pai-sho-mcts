use super::*;
use collector::{play, Played};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, RwLock,
    },
    time::SystemTime,
};
#[derive(Default, Serialize, Deserialize)]
struct Counters {
    games: usize,
    terminal: usize,
    host_wins: usize,
    guest_wins: usize,
    draws: usize,
    decisions: usize,
    eligible: usize,
    fresh_used: usize,
    replay_used: usize,
    #[serde(default)]
    correction_replay_used: usize,
    seconds: f64,
}
#[derive(Default, Deserialize)]
struct Resume {
    completed: usize,
    version: u64,
    updates: u64,
    next_game_id: usize,
    next_history_index: usize,
    main_started: usize,
    historical_started: usize,
    fresh_terminal_used: usize,
    human_used: usize,
    elapsed_seconds: f64,
    learner_seconds: f64,
    archive_seconds: f64,
    publish_seconds: f64,
    replay_positions: usize,
    lanes: BTreeMap<String, Counters>,
    #[serde(default)]
    case_states: BTreeMap<usize, cases::State>,
    #[serde(default)]
    legacy_ladder: ladder::State,
    #[serde(default)]
    case_manifest_sha256: Option<String>,
}
struct LearnItem {
    example: Arc<MicroExample>,
    kind: u8,
    lane: Lane,
    terminal: bool,
}
pub fn run(mut o: Options) -> Result<()> {
    o.validate()?;
    let wall = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs_f64();
    let seconds = o
        .end_unix_seconds
        .map_or(o.seconds, |end| o.seconds.min(end - wall));
    if seconds <= 0.0 {
        return Err(invalid("Gen5 deadline already passed"));
    }
    let started = Instant::now();
    let end = started + Duration::from_secs_f64(seconds);
    cpu::set_qos(o.macos_qos, false)?;
    let (pool, main_qos) = cpu::build_pool(
        o.main_threads(),
        o.macos_qos
            .then_some(paisho_platform::ThreadQos::UserInitiated),
    )?;
    let (secondary, secondary_qos) = if o.secondary_capacity() > 0 {
        let (pool, receipts) = cpu::build_pool(
            o.secondary_capacity(),
            o.macos_qos.then_some(paisho_platform::ThreadQos::Utility),
        )?;
        (Some(Arc::new(cpu::Spare::new(pool))), receipts)
    } else {
        (None, vec![])
    };
    let parent = MicroArtifact::load(&o.model)?;
    let resume: Resume = match &o.resume_progress {
        Some(path) => serde_json::from_slice(&fs::read(path)?)?,
        None => Resume::default(),
    };
    if o.resume_progress.is_some()
        && (resume.updates != parent.updates
            || o.evaluation_anchor.is_none()
            || o.replay_index.is_none()
            || !resume.elapsed_seconds.is_finite()
            || resume.elapsed_seconds < 0.0)
    {
        return Err(invalid(
            "resume requires matching durable model, replay and original Elo anchor",
        ));
    }
    let prior_elapsed = resume.elapsed_seconds;
    let ladder = Arc::new(RwLock::new(resume.legacy_ladder));
    if o.legacy_replay {
        let hash = sha256(&fs::read(&o.reference)?);
        let mut state = ladder.write().unwrap();
        if state.stage > 3 || (!state.reference.is_empty() && state.reference != hash) {
            return Err(invalid("legacy ladder reference changed"));
        }
        state.reference = hash;
    }
    let mut model = parent.model()?;
    let reference =
        if o.legacy_replay || o.historical || o.history_interval > 0.0 || o.calibration_reference_budget.is_some() {
            Some(crate::compact_learning::load_model(&o.reference)?)
        } else {
            None
        };
    if o.output.exists() {
        let entries = fs::read_dir(&o.output)?.collect::<std::result::Result<Vec<_>, _>>()?;
        let entries: Vec<_> = entries.into_iter().filter(|entry| entry.file_name() != ".DS_Store").collect();
        if entries.len() != 1 || entries[0].file_name() != "history" || !entries[0].path().is_dir()
        {
            return Err(invalid("output already exists except for restored history"));
        }
    } else {
        fs::create_dir(&o.output)?;
    }
    let source = o.output.canonicalize()?.to_string_lossy().into_owned();
    let out = PathBuf::from(&source);
    for d in ["games", "models", "history", "milestones"] {
        if d == "history" {
            fs::create_dir_all(out.join(d))?;
        } else {
            fs::create_dir(out.join(d))?;
        }
    }
    parent.save(&out.join("parent.json"))?;
    if let Some(r) = &reference {
        save_json_new(&out.join("reference-gen3-1.json"), r)?;
    }
    save_json_new(
        &out.join("plan.json"),
        &serde_json::json!({"schema":"paisho-gen5-runtime-v1","rules":RULES.as_str(),"options":o,"parent":parent.identity(),"reference_sha256":if reference.is_some(){Some(sha256(&fs::read(&o.reference)?))}else{None},"main_cpu_capacity":o.main_threads(),"historical_cpu_capacity":if o.historical || o.history_interval>0.0{o.secondary_capacity()}else{0},"secondary_cpu_capacity":o.secondary_capacity(),"qos_main":main_qos,"qos_secondary":secondary_qos,"physical_core_affinity":false,"absolute_end_unix_seconds":wall+seconds,"automatic_resume":false,"replay_ratio_includes_human_draws":true,"historical_quota":"at most one fresh historical start per 19 main starts; independent soft quota"}),
    )?;
    let human = if o.learn && o.human_fraction > 0.0 {
        memory::human(o.human_dataset.as_ref().unwrap(), &out, &pool)?
    } else {
        vec![]
    };
    let mut memory = memory::Memory::new(&o);
    if let Some(path) = &o.replay_index {
        memory.load(path)?;
    }
    if o.resume_progress.is_some() && memory.len() != resume.replay_positions {
        return Err(invalid(
            "restored replay position count differs from pause receipt",
        ));
    }
    let human_cases = if o.case_curriculum.is_some() {
        Some(Arc::new(cases::load(
            o.human_dataset.as_ref().unwrap(),
            o.seed,
            &out,
        )?))
    } else {
        None
    };
    let case_manifest_sha256 = human_cases
        .as_ref()
        .map(|_| sha256(&fs::read(out.join("human-cases.json")).unwrap()));
    if resume.case_manifest_sha256.is_some() && resume.case_manifest_sha256 != case_manifest_sha256
    {
        return Err(invalid("resumed human cases changed"));
    }
    let mut case_states = resume.case_states;
    let mut durable_memory = match &o.case_curriculum {
        Some(c) => Some(durable::Archive::open(&c.archive)?),
        None => None,
    };
    let proofs = durable_memory
        .as_ref()
        .map(|d| d.proofs.clone())
        .unwrap_or_default();
    let first = Arc::new(Snapshot {
        artifact: None,
        version: resume.version,
        identity: parent.identity(),
        model: Arc::new(model.clone()),
        path: out.join("parent.json"),
    });
    let evaluation_anchor = match &o.evaluation_anchor {
        Some(path) => {
            let anchor = MicroArtifact::load(path)?;
            anchor.save(&out.join("evaluation-anchor.json"))?;
            Arc::new(Snapshot {
                artifact: None,
                version: 0,
                identity: anchor.identity(),
                model: Arc::new(anchor.model()?),
                path: out.join("evaluation-anchor.json"),
            })
        }
        None => first.clone(),
    };
    let calibration_model = if o.calibration_reference_budget.is_some() {
        Some(Arc::new(reference.as_ref().unwrap().model()?))
    } else {
        None
    };
    let shared = Arc::new(RwLock::new(first.clone()));
    let champions = Arc::new(RwLock::new(Vec::<history::Ranked>::new()));
    let stop = Arc::new(AtomicBool::new(false));
    o.stop_signal = Some(stop.clone());
    // One cheap monitor; game loops read only the atomic flag.
    let monitor_stop = stop.clone();
    let stop_path = out.join("stop-request.json");
    let stop_monitor = std::thread::spawn(move || {
        while !monitor_stop.load(Ordering::Relaxed) && Instant::now() < end {
            if stop_path.exists() { monitor_stop.store(true, Ordering::Relaxed); break; }
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    let next = Arc::new(AtomicUsize::new(resume.next_game_id));
    let main_started = Arc::new(AtomicUsize::new(resume.main_started));
    let historical_started = Arc::new(AtomicUsize::new(resume.historical_started));
    let async_error = Arc::new(RwLock::new(None::<String>));
    // Independent bounded queues: a slow historical game can only fill its own
    // queue. Main producers never await a historical game or evaluation.
    let (main_tx, main_rx) = mpsc::sync_channel::<Played>(o.actors * 2);
    let (history_tx, history_rx) = mpsc::sync_channel::<Played>(2);
    let mut actors = vec![];
    let replay_reference = if o.legacy_replay { Some(Arc::new(reference.as_ref().unwrap().model()?)) } else { None };
    for actor in 0..o.actors {
        let actor_reference = replay_reference.clone();
        let actor_ladder = ladder.clone();
        let calibration_model = calibration_model.clone();
        let game_pool = if calibration_model.is_some() {
            cpu::Executor::direct(cpu::build_pool(1, None)?.0)
        } else if o.reuse_idle_secondary && actor % o.threads >= o.main_threads() {
            cpu::Executor::adaptive(pool.clone(), secondary.as_ref().unwrap().clone())
        } else {
            cpu::Executor::direct(pool.clone())
        };
        let actor_error = async_error.clone();
        let actor_cases = human_cases.clone();
        let actor_state = case_states.get(&actor).cloned().unwrap_or(cases::State {
            ticket: actor,
            ..Default::default()
        });
        let actor_proofs = proofs.clone();
        let (shared, champions, stop, next, count, tx, pool, o, source) = (
            shared.clone(),
            champions.clone(),
            stop.clone(),
            next.clone(),
            main_started.clone(),
            main_tx.clone(),
            game_pool,
            o.clone(),
            source.clone(),
        );
        actors.push(std::thread::spawn(move || {
            if let Err(e) = cpu::set_qos(
                o.macos_qos,
                o.reuse_idle_secondary && actor % o.threads >= o.main_threads(),
            ) {
                *actor_error.write().unwrap() = Some(e.to_string());
                stop.store(true, Ordering::Relaxed);
                return;
            }
            if let Some(cases) = actor_cases {
                case_actor::run(
                    actor,
                    actor_reference,
                    actor_ladder,
                    actor_state,
                    cases,
                    shared,
                    actor_proofs,
                    next,
                    count,
                    stop,
                    tx,
                    pool,
                    o,
                    source,
                    end,
                );
                return;
            }
            let mut rng = StableRng::new(o.seed ^ 0x6163746f72 ^ (actor as u64));
            while !stop.load(Ordering::Relaxed) && Instant::now() < end {
                let id = next.fetch_add(1, Ordering::Relaxed);
                if id >= o.games {
                    break;
                }
                count.fetch_add(1, Ordering::Relaxed);
                let snapshot = shared.read().unwrap().clone();
                let opponent = if rng.next_f64() < o.checkpoint_fraction / 0.95 {
                    let choices = champions.read().unwrap();
                    if choices.is_empty() {
                        snapshot.clone()
                    } else {
                        choices[rng.index(choices.len())].snapshot.clone()
                    }
                } else {
                    snapshot.clone()
                };
                let legacy = calibration_model
                    .as_ref()
                    .map(|m| (m.as_ref(), o.calibration_reference_budget.unwrap()));
                let game = play(id, snapshot, opponent, legacy, &o, end, &pool, &source);
                if tx.send(game).is_err() {
                    break;
                }
            }
        }));
    }
    let (reanalysis_tx, reanalysis_rx) = mpsc::sync_channel(1);
    let reanalysis_worker = o.case_curriculum.as_ref().map(|_| {
        reanalysis::spawn(
            o.clone(),
            reanalysis_rx,
            main_tx.clone(),
            shared.clone(),
            next.clone(),
            stop.clone(),
            cpu::Executor::direct(pool.clone()),
            proofs.clone(),
            source.clone(),
            end,
        )
    });
    drop(main_tx);
    let history_worker=secondary.clone().filter(|_|o.historical || o.history_interval>0.0).map(|spare|{
        let pool=spare.pool.clone();
        let (o,shared,champions,stop,next,count,historical_started,source,initial,async_error)=(o.clone(),shared.clone(),champions.clone(),stop.clone(),next.clone(),main_started.clone(),historical_started.clone(),source.clone(),evaluation_anchor.clone(),async_error.clone());
        std::thread::spawn(move || {
            let result=(||->Result<()>{
                cpu::set_qos(o.macos_qos,true)?;
                let old=reference.unwrap().model()?;let mut last=Instant::now();let mut sweeps=resume.next_history_index;
                while !stop.load(Ordering::Relaxed) && Instant::now()<end {
                    if o.history_interval>0.0 && last.elapsed().as_secs_f64()>=o.history_interval {
                        if !spare.can_admit(o.history_seconds,o.threads,o.historical_capacity_fraction) {
                            std::thread::sleep(Duration::from_millis(50)); continue;
                        }
                        let _reservation=spare.reserve();
                        let snapshot=shared.read().unwrap().clone();
                        if let Some(ranked)=history::assess(snapshot,&initial,&o,end,&pool,sweeps)?{
                            let mut best=champions.write().unwrap();
                            if !best.iter().any(|r|r.snapshot.identity==ranked.snapshot.identity){best.push(ranked);best.sort_by(|a,b|b.score.total_cmp(&a.score));best.truncate(3);}
                            atomic_json(&o.output.join("checkpoint-pool.json"),&best.iter().map(|r|serde_json::json!({"version":r.snapshot.version,"identity":r.snapshot.identity,"score":r.score,"path":r.snapshot.path})).collect::<Vec<_>>())?;
                        }sweeps+=1;last=Instant::now();continue;
                    }
                    let hist=historical_started.load(Ordering::Relaxed);
                    let budget=[32,64,128][hist%3];
                    let cap=o.historical_seconds.iter().find(|(b,_)|*b==budget).unwrap().1;
                    let unlimited=o.historical_unlimited_budgets.contains(&budget);
                    if o.historical && hist<count.load(Ordering::Relaxed)/19 && spare.can_admit(if unlimited {0.02} else {cap},o.threads,o.historical_capacity_fraction) {
                        let id=next.fetch_add(1,Ordering::Relaxed);if id>=o.games{break;}
                        historical_started.fetch_add(1,Ordering::Relaxed);
                        let snapshot=shared.read().unwrap().clone();
                        let game=if unlimited {
                            play(id,snapshot.clone(),snapshot,Some((&old,budget)),&o,end,
                                &cpu::Executor::budgeted(spare.clone(),o.threads,o.historical_capacity_fraction,end),&source)
                        } else {let _reservation=spare.reserve();play(id,snapshot.clone(),snapshot,Some((&old,budget)),&o,end,&cpu::Executor::direct(pool.clone()),&source)};
                        if history_tx.send(game).is_err(){break;}
                    }else{
                        if next.load(Ordering::Relaxed)>=o.games{break;}
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }Ok(())
            })();
            if let Err(e)=result{*async_error.write().unwrap()=Some(e.to_string());stop.store(true,Ordering::Relaxed);}
        })
    });
    let mut rng = StableRng::new(o.seed ^ 0x6c6561726e);
    let mut updates = parent.updates;
    let mut version = resume.version;
    let mut counters = resume.lanes;
    let mut completed = resume.completed;
    let mut errors = vec![];
    let mut learner_seconds = resume.learner_seconds;
    let mut archive_seconds = resume.archive_seconds;
    let mut publish_seconds = resume.publish_seconds;
    let mut human_used = resume.human_used;
    let mut fresh_terminal_used = resume.fresh_terminal_used;
    let (ready_rx, archiver) = archive::spawn(
        main_rx,
        history_rx,
        out.clone(),
        o.actors * 2,
        o.archive_workers,
        o.case_curriculum.as_ref().map(|c| c.archive.clone()),
        o.checkpoint_seconds > 0.0,
        proofs.clone(),
        stop.clone(),
        async_error.clone(),
    );
    let mut published_files: Vec<(std::sync::Weak<Snapshot>, PathBuf)> = vec![];
    let mut pruned_targets = 0usize;
    let mut pruned_models = 0usize;
    let mut checkpoint = checkpoint::Checkpoint::new(o.checkpoint_seconds);
    let mut pending_deletions = vec![];
    let mut last_progress: Option<serde_json::Value> = None;
    let mut last_live = Instant::now();
    let mut consolidation_seconds = 0.0;
    let mut durable_seconds = 0.0;
    let result = (|| -> Result<()> {
        for ready in ready_rx.iter() {
            let archive::Ready {
                mut game,
                durable_path,
                durable_seconds: saved_seconds,
                owned,
                corrections,
                target_path,
                target_hash,
                psr_hash,
                archive_seconds: seconds,
            } = ready;
            archive_seconds += seconds;
            durable_seconds += saved_seconds;
            completed += 1;
            if let Some(error) = &game.error {
                errors.push(error.clone());
                stop.store(true, Ordering::Relaxed);
            }
            let lane = format!("{:?}", game.lane);
            let terminal = game.outcome != GameOutcome::Ongoing;
            let fresh_count = owned.len();
            let stem = format!("game-{:07}", game.id);
            let dir = out.join("games");
            let count = counters.entry(lane.clone()).or_default();
            count.games += 1;
            count.terminal += usize::from(terminal);
            match game.outcome {
                GameOutcome::Win(Player::Host) => count.host_wins += 1,
                GameOutcome::Win(Player::Guest) => count.guest_wins += 1,
                GameOutcome::Draw => count.draws += 1,
                GameOutcome::Ongoing => {}
            }
            count.decisions += game.record.actions().len() - game.prefix_decisions;
            count.eligible += owned.len();
            count.seconds += game.seconds;
            let revisit_due = (game.lane == Lane::Selfplay || o.legacy_replay && game.lane == Lane::Historical) && count.games % 20 == 0;
            let mut items: Vec<_> = owned
                .iter()
                .map(|ex| LearnItem {
                    example: ex.clone(),
                    kind: 0,
                    lane: game.lane,
                    terminal,
                })
                .collect();
            if let Some(path) = durable_path {
                durable_memory.as_mut().unwrap().add(path);
            }
            let rehearsal_budget = if o.learn {
                o.case_curriculum.as_ref().map_or(0, |c| {
                    (owned.len().saturating_mul(o.replay_ratio) as f64 * c.durable_fraction).round()
                        as usize
                })
            } else {
                0
            };
            let consolidation_started = Instant::now();
            let rehearsal = match &mut durable_memory {
                Some(d) => d.rehearse(rehearsal_budget, &mut rng, &model)?,
                None => vec![],
            };
            consolidation_seconds += consolidation_started.elapsed().as_secs_f64();
            if revisit_due {
                if let Some(task) = durable_memory.as_mut().and_then(|d| d.revisit.take()) {
                    let _ = reanalysis_tx.try_send(task);
                }
            }
            let durable_draws = rehearsal.len();
            items.extend(rehearsal.into_iter().map(|example| LearnItem {
                example,
                kind: 4,
                lane: Lane::Selfplay,
                terminal: false,
            }));
            if o.learn {
                for _ in 0..owned
                    .len()
                    .saturating_mul(o.replay_ratio)
                    .saturating_sub(durable_draws)
                {
                    if !human.is_empty() && rng.next_f64() < o.human_fraction {
                        items.push(LearnItem {
                            example: human[rng.index(human.len())].clone(),
                            kind: 2,
                            lane: Lane::Selfplay,
                            terminal: true,
                        });
                    } else {
                        let prefer = rng.next_f64() < o.historical_replay_fraction;
                        let correction = o.correction_replay_fraction > 0.0
                            && memory.correction_len() > 0
                            && rng.next_f64() < o.correction_replay_fraction;
                        let prioritized = if correction {
                            memory.draw_correction(&mut rng)
                        } else {
                            None
                        };
                        let kind = if prioritized.is_some() { 3 } else { 1 };
                        if let Some((ex, replay_lane)) =
                            prioritized.or_else(|| memory.draw(&mut rng, prefer))
                        {
                            items.push(LearnItem {
                                example: ex,
                                kind,
                                lane: replay_lane,
                                terminal: false,
                            });
                        }
                    }
                }
                shuffle(&mut items, &mut rng);
            }
            let before = updates;
            let mut fresh_used = 0;
            let mut replay_used = 0;
            let mut correction_replay_used = 0;
            let mut human_game_used = 0;
            let t = Instant::now();
            if o.learn {
                for batch in items.chunks(64) {
                    if Instant::now() >= end || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let refs: Vec<_> = batch.iter().map(|item| item.example.as_ref()).collect();
                    if o.inline_learning {
                        model.train_batch_inline(&refs, o.rate, 1e-5)
                    } else {
                        pool.install(|| model.train_batch(&refs, o.rate, 1e-5))
                    }
                    .map_err(invalid)?;
                    updates += 1;
                    for item in batch {
                        match item.kind {
                            0 => {
                                fresh_used += 1;
                                if item.terminal {
                                    fresh_terminal_used += 1;
                                }
                                counters
                                    .entry(format!("{:?}", item.lane))
                                    .or_default()
                                    .fresh_used += 1;
                            }
                            1 | 3 | 4 => {
                                replay_used += 1;
                                if item.kind == 3 {
                                    correction_replay_used += 1;
                                    counters
                                        .entry(format!("{:?}", item.lane))
                                        .or_default()
                                        .correction_replay_used += 1;
                                }
                                counters
                                    .entry(format!("{:?}", item.lane))
                                    .or_default()
                                    .replay_used += 1;
                            }
                            _ => {
                                human_used += 1;
                                human_game_used += 1;
                            }
                        }
                    }
                }
            }
            learner_seconds += t.elapsed().as_secs_f64();
            let correction_candidates = corrections.iter().filter(|v| **v).count();
            memory.add(owned, &target_path, &target_hash, game.lane, &corrections);
            // Only dense targets already copied into the independent compact
            // archive may be removed; PSRs, receipts and inherited sources stay.
            for (path, lane) in memory.take_obsolete() {
                if o.case_curriculum.is_some()
                    && (lane != Lane::Historical || o.legacy_replay)
                    && path.parent() == Some(out.join("games").as_path())
                    && path.exists()
                {
                    if o.checkpoint_seconds > 0.0 {
                        pending_deletions.push(path);
                    } else {
                        fs::remove_file(path)?;
                        pruned_targets += 1;
                    }
                }
            }
            let t = Instant::now();
            if updates > before {
                version += 1;
                let artifact = MicroArtifact::new(
                    &model,
                    updates,
                    serde_json::json!({"kind":"gen5-selfplay","rules":RULES.as_str(),"parent":parent.identity(),"source_run":source,"version":version,"last_game":game.id,"search_mode":o.mode}),
                );
                let path = out.join("models").join(format!("model-{version:07}.json"));
                if o.checkpoint_seconds == 0.0 {
                    artifact.save(&path)?;
                }
                let published = Arc::new(Snapshot {
                    artifact: Some(Arc::new(artifact.clone())),
                    version,
                    identity: artifact.identity(),
                    model: Arc::new(model.clone()),
                    path: path.clone(),
                });
                *shared.write().unwrap() = published.clone();
                if o.case_curriculum.is_some() && o.checkpoint_seconds == 0.0 && version % 100 != 0
                {
                    published_files.push((Arc::downgrade(&published), path));
                }
                if published_files.len() >= 32 {
                    let mut keep = vec![];
                    for (weak, path) in published_files.drain(..) {
                        if weak.strong_count() == 0 {
                            fs::remove_file(path)?;
                            pruned_models += 1;
                        } else {
                            keep.push((weak, path));
                        }
                    }
                    published_files = keep;
                }
            }
            publish_seconds += t.elapsed().as_secs_f64();
            let fully_learned =
                o.learn && fresh_used == fresh_count && (fresh_count == 0 || updates > before);
            if let Some(attempt) = game
                .case
                .as_ref()
                .filter(|a| a.kind != "archive-reanalysis")
            {
                let mut state = if fully_learned {
                    attempt.after.clone()
                } else {
                    attempt.before.clone()
                };
                // Reanalysis records carry the same next case state, never new attempts.
                if state.rotate_reason.is_some() && state.pending.is_empty() {
                    state.advance(o.actors);
                }
                case_states.insert(attempt.actor, state);
            }
            if o.legacy_replay && fully_learned { ladder.write().unwrap().observe(&game, version); }
            let mut receipt = serde_json::json!({"rules":RULES.as_str(),"id":game.id,"lane":lane,"collector_version":game.snapshot.version,"collector":game.snapshot.identity,"opponent":game.opponent,"reference_budget":game.reference_budget,"candidate_seat":format!("{:?}",game.candidate_seat),"termination":game.termination,"outcome":format!("{:?}",game.outcome),"error":game.error,"cycle":game.cycle,"decisions":game.record.actions().len(),"seconds":game.seconds,"cap_seconds":game.cap_seconds,"campaign_censored":game.campaign_censored,"search_seconds":game.search_seconds,"pool_wait_seconds":game.pool_wait_seconds,"maintenance_seconds":game.maintenance_seconds,"sample_seconds":game.sample_seconds,"simulations":game.simulations,"inherited_visits":game.inherited,"inference_evaluations":game.evals,"inference_cache_hits":game.hits,"forced_playouts":game.forced_playouts,"pruned_policy_visits":game.pruned_policy_visits,"policy_searches":game.policy_searches,"policy_coverage_sum":game.policy_coverage_sum,"psr_sha256":psr_hash,"targets_file":format!("{stem}.targets.json.gz"),"targets_sha256":target_hash,"eligible_examples":fresh_count,"fresh_used":fresh_used,"replay_used":replay_used,"correction_replay_used":correction_replay_used,"correction_candidates":correction_candidates,"human_used":human_game_used,"model_version":version,"updates":updates,"learned_batches":updates-before,"search_mode":o.mode});
            receipt.as_object_mut().unwrap().extend(serde_json::json!({"case":game.case,"reanalysis":game.reanalysis,"prefix_decisions":game.prefix_decisions,"continuation_decisions":game.record.actions().len()-game.prefix_decisions,"durable_draws":durable_draws,"fully_learned":fully_learned,"durable_save_seconds":saved_seconds}).as_object().unwrap().clone());
            if o.checkpoint_seconds > 0.0 {
                durable::write_pending(&dir.join(format!("{stem}.json")), &receipt)?;
            } else {
                save_json_new(&dir.join(format!("{stem}.json")), &receipt)?;
            }
            let (proof_disk_count, proof_cache_entries) = {
                let c = proofs.read().unwrap();
                (c.disk_count, c.len())
            };
            let progress = serde_json::json!({"legacy_ladder":*ladder.read().unwrap(),"next_game_id":next.load(Ordering::Relaxed),"case_states":case_states,"case_manifest_sha256":case_manifest_sha256,"dense_targets_pruned":pruned_targets,"unreferenced_models_pruned":pruned_models,"consolidation_seconds":consolidation_seconds,"durable_save_seconds":durable_seconds,"durable_bundles":durable_memory.as_ref().map(|d|d.len()),"durable_draws":durable_memory.as_ref().map(|d|d.draws),"durable_proofs":proof_disk_count,"proof_cache_entries":proof_cache_entries,"completed":completed,"last_game_id":game.id,"lanes":counters,"version":version,"updates":updates,"elapsed_seconds":prior_elapsed+started.elapsed().as_secs_f64(),"remaining_seconds":end.saturating_duration_since(Instant::now()).as_secs_f64(),"main_started":main_started.load(Ordering::Relaxed),"historical_started":historical_started.load(Ordering::Relaxed),"main_cpu_capacity":o.main_threads(),"historical_cpu_capacity":if history_worker.is_some(){o.secondary_capacity()}else{0},"secondary_cpu_capacity":o.secondary_capacity(),"secondary_usage":secondary.as_ref().map(|s|s.telemetry()),"replay_positions":memory.len(),"correction_pool_positions":memory.correction_len(),"correction_reference_bytes":memory.correction_reference_bytes(),"replay_bytes":memory.bytes,"replay_evicted":memory.evicted,"human_examples":human.len(),"human_used":human_used,"fresh_terminal_used":fresh_terminal_used,"learner_seconds":learner_seconds,"archive_seconds":archive_seconds,"publish_seconds":publish_seconds,"async_error":*async_error.read().unwrap(),"errors":errors});
            println!("{receipt}");
            if o.checkpoint_seconds > 0.0 {
                let mut progress = progress;
                if checkpoint.due() {
                    checkpoint.commit(&out, &shared.read().unwrap(), &memory, &mut progress)?;
                    for path in pending_deletions.drain(..) {
                        fs::remove_file(path)?;
                        pruned_targets += 1;
                    }
                }
                progress["durable_version"] = checkpoint.version.into();
                progress["checkpoint_seconds"] = o.checkpoint_seconds.into();
                if last_progress.is_none() || last_live.elapsed().as_secs_f64() >= 1.0 {
                    durable::write_pending(&out.join("progress.json"), &progress)?;
                    last_live = Instant::now();
                }
                last_progress = Some(progress);
            } else {
                durable::write(&out.join("progress.json"), &progress)?;
            }
            if let Some(ack) = game.ack.take() {
                let _ = ack.send(case_actor::Feedback {
                    snapshot: shared.read().unwrap().clone(),
                    fully_learned,
                });
            }
        }
        Ok(())
    })();
    stop.store(true, Ordering::Relaxed);
    let _ = stop_monitor.join();
    drop(ready_rx);
    if archiver.join().is_err() {
        errors.push("archive worker panic".into());
    }
    for actor in actors {
        if actor.join().is_err() {
            errors.push("main actor panic".into());
        }
    }
    drop(reanalysis_tx);
    if let Some(worker) = reanalysis_worker {
        if worker.join().is_err() {
            errors.push("reanalysis worker panic".into());
        }
    }
    if let Some(worker) = history_worker {
        if worker.join().is_err() {
            errors.push("historical worker panic".into());
        }
    }
    if let Some(error) = async_error.read().unwrap().clone() {
        errors.push(error);
    }
    result?;
    if let Some(mut progress) = last_progress {
        checkpoint.commit(&out, &shared.read().unwrap(), &memory, &mut progress)?;
        durable::write(&out.join("progress.json"), &progress)?;
        for path in pending_deletions {
            fs::remove_file(path)?;
        }
    }
    MicroArtifact::new(&model,updates,serde_json::json!({"kind":"gen5-final","rules":RULES.as_str(),"parent":parent.identity(),"source_run":source,"version":version,"search_mode":o.mode})).save(&out.join("model.json"))?;
    memory.save(&out.join("replay-final.index.json"))?;
    save_json_new(
        &out.join("report.json"),
        &serde_json::json!({"rules":RULES.as_str(),"completed":completed,"lanes":counters,"version":version,"updates":updates,"elapsed_seconds":prior_elapsed+started.elapsed().as_secs_f64(),"fresh_terminal_used":fresh_terminal_used,"human_used":human_used,"learner_seconds":learner_seconds,"archive_seconds":archive_seconds,"publish_seconds":publish_seconds,"replay_positions":memory.len(),"correction_pool_positions":memory.correction_len(),"correction_reference_bytes":memory.correction_reference_bytes(),"replay_bytes":memory.bytes,"errors":errors,"automatic_resume":false,"secondary_usage":secondary.as_ref().map(|s|s.telemetry())}),
    )?;
    if !errors.is_empty() {
        return Err(invalid(errors.join("; ")));
    }
    Ok(())
}
