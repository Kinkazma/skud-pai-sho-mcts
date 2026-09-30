use super::*;
use collector::{play, Played};
#[path = "runtime_resume_weights.rs"]
mod resume_weights;
#[path = "runtime_transaction_log.rs"]
mod transaction_log;
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, RwLock,
    },
    time::SystemTime,
};
fn validate_publication_transfer_resume(enabled: bool, saved: &serde_json::Value) -> Result<()> {
    if saved["publication_transfer"] == true && !enabled {
        return Err(invalid("learned publication transfer cannot resume under the older protocol"));
    }
    Ok(())
}
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
    #[serde(default)]
    initial_weight_migration: serde_json::Value,
    #[serde(default)]
    protection: serde_json::Value,
    #[serde(default)]
    publication_guard: serde_json::Value,
    #[serde(default)]
    publication_work_clock: Option<publication_cadence::WorkClock>,
    #[serde(default)]
    frozen_evaluations: serde_json::Value,
    #[serde(default)]
    opponent_ladders: BTreeMap<String, ladder::State>,
    #[serde(default)]
    curriculum_states: BTreeMap<String, BTreeMap<usize, cases::State>>,
    #[serde(default)]
    recall_quotas: recall::Quotas,
    #[serde(default)]
    durable_recall: serde_json::Value,
    #[serde(default)]
    learner_rng: Option<u64>,
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
    fresh_index: Option<usize>,
}
pub fn run(mut o: Options) -> Result<()> {
    if o.learning_loop_v2 {
        o.reanalysis_cache = Some(Arc::new(ReanalysisCache::new(64 * 1024 * 1024)));
    }
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
    let started = paisho_platform::training_time::now();
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
    // This immutable starting artifact is shared by every publication in the run.
    let parent_identity = parent.identity();
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
    resume.recall_quotas.validate()?;
    if resume.publication_work_clock.is_some() && o.publication_work_budget.is_none() {
        return Err(invalid("work-clock resume requires an explicit publication work budget"));
    }
    let mut publication_work_clock = if o.publication_work_budget.is_some() {
        let clock=resume.publication_work_clock.clone().unwrap_or_default();
        clock.validate()?;
        Some(clock)
    } else { None };
    let mut recall_quotas = resume.recall_quotas;
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
    // Read the SAVED protocol before enable_v3 mutates the restored Guard state.
    // A real V3 resume keeps its in-progress shadow weights.
    let migrate_legacy_weights = resume_weights::migration_required(
        o.learning_loop_v3, o.resume_progress.is_some(), &resume.publication_guard,
    )?;
    let mut weight_migration = resume.initial_weight_migration.clone();
    let mut model = parent.model()?;
    if o.learning_loop_v3 && !o.learning_loop_v2 {
        return Err(invalid("transactional learning requires the V2 learning protocol"));
    }
    if resume.publication_guard["learning_loop_v3"] == true && !o.learning_loop_v3 {
        return Err(invalid("transactional publication cannot resume under the older protocol"));
    }
    validate_publication_transfer_resume(o.publication_transfer, &resume.publication_guard)?;
    if o.neural_memory && (!model.has_neural_memory() || !o.learning_loop_v2) {
        return Err(invalid("neural memory requires explicit migration and V34 learning protections"));
    }
    if o.structural_repair && !model.has_spatial() {
        return Err(invalid(
            "structural repair requires the explicitly migrated V4 model",
        ));
    }
    let reference = if o.legacy_replay
        || o.historical
        || o.history_interval > 0.0
        || o.calibration_reference_budget.is_some()
    {
        Some(crate::compact_learning::load_model(&o.reference)?)
    } else {
        None
    };
    if o.output.exists() {
        let entries = fs::read_dir(&o.output)?.collect::<std::result::Result<Vec<_>, _>>()?;
        let entries: Vec<_> = entries
            .into_iter()
            .filter(|entry| entry.file_name() != ".DS_Store")
            .collect();
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
    if let Some(path) = paisho_platform::training_time::path() {
        atomic_json(&out.join("resident-clock.json"), &serde_json::json!({
            "schema":"paisho-resident-clock-v1", "pid":std::process::id(),
            "path":path, "remaining_time_preserved":true,
        }))?;
    }
    parent.save(&out.join("parent.json"))?;
    if let Some(r) = &reference {
        save_json_new(&out.join("reference-gen3-1.json"), r)?;
    }
    save_json_new(
        &out.join("plan.json"),
        &serde_json::json!({"schema":"paisho-gen5-runtime-v1","rules":RULES.as_str(),"options":o,"parent":parent_identity,"reference_sha256":if reference.is_some(){Some(sha256(&fs::read(&o.reference)?))}else{None},"main_cpu_capacity":o.main_threads(),"historical_cpu_capacity":if o.historical || o.history_interval>0.0{o.secondary_capacity()}else{0},"secondary_cpu_capacity":o.secondary_capacity(),"qos_main":main_qos,"qos_secondary":secondary_qos,"physical_core_affinity":false,"absolute_end_unix_seconds":wall+seconds,"automatic_resume":false,"replay_ratio_includes_human_draws":true,"historical_quota":"at most one fresh historical start per 19 main starts; independent soft quota"}),
    )?;
    let human = if o.learn && o.human_fraction > 0.0 {
        memory::human(
            o.human_dataset.as_ref().unwrap(),
            &out,
            &pool,
            model.has_spatial(),
        )?
    } else {
        vec![]
    };
    let mut memory = memory::Memory::new(&o);
    if let Some(path) = &o.replay_index {
        pool.install(|| {
            memory
                .load_for_model(path, model.has_spatial())
                .map_err(|e| e.to_string())
        })
        .map_err(invalid)?;
    }
    if o.resume_progress.is_some() && memory.len() != resume.replay_positions {
        return Err(invalid(
            "restored replay position count differs from pause receipt",
        ));
    }
    // Import uses the complete pool. Partition only after import, and release
    // that pool before creating the replacement workers: no extra CPU budget.
    let search_threads=o.main_threads()-o.learner_threads;
    let search_pools = if o.search_pool_shards > 1 || o.learner_threads > 0 {
        drop(pool);
        let pools = cpu::build_search_pools(
            search_threads,
            o.search_pool_shards,
            o.macos_qos
                .then_some(paisho_platform::ThreadQos::UserInitiated),
        )?;
        atomic_json(
            &out.join("search-pools.json"),
            &serde_json::json!({"shards":pools.len(),"threads_per_shard":search_threads/pools.len(),"search_threads":search_threads,"learner_threads":o.learner_threads,"total_threads":o.main_threads(),"actors":o.actors}),
        )?;
        pools
    } else {
        vec![pool]
    };
    let pool = search_pools[0].clone();
    let learning_pools=if o.learner_threads > 0 {
        vec![cpu::build_pool(o.learner_threads,o.macos_qos.then_some(paisho_platform::ThreadQos::UserInitiated))?.0]
    } else { search_pools.clone() };
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
    let mut curriculum_states = resume.curriculum_states;
    let opponents = Arc::new(references::load_all(&o.opponents)?);
    let mut opponent_ladders = Vec::new();
    for model in opponents.iter() {
        let key = &model.spec.generation;
        let mut state = resume
            .opponent_ladders
            .get(key)
            .cloned()
            .unwrap_or_else(|| {
                if key == "3.1" {
                    ladder.read().unwrap().clone()
                } else {
                    ladder::State::default()
                }
            });
        if !state.reference.is_empty() && state.reference != model.spec.sha256 {
            return Err(invalid("curriculum opponent identity changed"));
        }
        if state.stage >= ladder::BUDGETS.len() {
            return Err(invalid("invalid opponent stage"));
        }
        if o.learning_loop_repair && resume.frozen_evaluations.is_null() {
            state.current.clear();
        }
        state.max_stage = ladder::BUDGETS.len() - 1;
        state.reference = model.spec.sha256.clone();
        opponent_ladders.push(Arc::new(RwLock::new(state)));
    }
    let opponent_ladders = Arc::new(opponent_ladders);
    if !opponents.is_empty() {
        curriculum_states
            .entry("3.1".into())
            .or_insert_with(|| case_states.clone());
        atomic_json(
            &out.join("opponents.json"),
            &serde_json::json!({"selfplay_fraction":0.8,"per_reference_fraction":0.04,"models":o.opponents,"resident":true,"private_game_trees":true}),
        )?;
    }
    let mut durable_memory = match &o.case_curriculum {
        Some(c) => Some(durable::Archive::open(&c.archive)?),
        None => None,
    };
    if let Some(d) = &mut durable_memory {
        d.policy_consolidation(o.learning_loop_repair);
        for source in &o.recall_archive_sources {
            d.add_read_only(source)?;
        }
        d.restore(&resume.durable_recall)?;
        d.cache_enabled = o.structural_repair;
        d.proof_recall = o.proof_recall;
    }
    let proofs = durable_memory
        .as_ref()
        .map(|d| d.proofs.clone())
        .unwrap_or_default();
    let first = Arc::new(Snapshot {
        artifact: None,
        version: resume.version,
        identity: parent_identity.clone(),
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
    let identity_workers = cpu::Ordered::new(&learning_pools);
    let mut publication = if o.learning_loop_repair {
        Some(publication::Guard::open(
            o.publication_guard.as_ref().unwrap(),
            &out,
            o.value_policy_strength,
            first.clone(),
            &resume.publication_guard,
        )?)
    } else {
        None
    };
    let mut protection = if o.learning_loop_v2 {
        let guard = publication.as_mut().unwrap();
        guard.enable_v2(o.publication_validation.as_ref().unwrap())?;
        if o.learning_loop_v3 { guard.enable_v3()?; }
        guard.enable_parallel(&learning_pools);
        if o.publication_transfer { guard.enable_transfer()?; }
        let initial_working = resume_weights::working_snapshot(
            migrate_legacy_weights, &first, &guard.accepted(), parent.updates,
            resume.version, out.join("v2-to-v3-start.json"),
        )?;
        if migrate_legacy_weights {
            // Preserve the original parent.json and all consumed counters. This
            // separately identified artifact changes only the initial weights.
            let artifact = initial_working.artifact.as_ref()
                .ok_or_else(|| invalid("missing protocol migration artifact"))?;
            artifact.save(&initial_working.path)?;
            weight_migration = artifact.provenance.clone();
            model = initial_working.model.as_ref().clone();
            *shared.write().unwrap() = initial_working;
        }
        let d = durable_memory.as_mut().unwrap();
        d.trusted_action_values = o.learning_loop_v3;
        d.enable_parallel(&learning_pools);
        d.enable_coverage(
            &out,
            &resume.durable_recall,
            &model,
            &o.case_curriculum.as_ref().unwrap().archive,
        )?;
        d.seed_values(guard.reference_examples());
        let mut protected = protection::Protection::new(&model, guard.reference_examples())?;
        protected.enable_parallel(&learning_pools);
        if o.learning_loop_v3 {
            protected.enable_loop_v3();
            protected.enable_validation_value(guard.diagnostic_validation_examples()?)?;
        }
        protected.restore(&resume.protection)?;
        if o.learning_loop_v3 {
            protected.adopt_accepted(&guard.accepted().model)?;
            d.focus_policy(guard.correction_examples()?);
        }
        Some(protected)
    } else {
        None
    };
    let actor_shared = Arc::new(RwLock::new(
        publication
            .as_ref()
            .map_or_else(|| first.clone(), |g| g.accepted()),
    ));
    let evaluations = if o.learning_loop_repair {
        Some(Arc::new(evaluation::Evaluations::open(
            &out,
            human_cases.as_ref().unwrap(),
            &o,
            &resume.frozen_evaluations,
            &model,
        )?))
    } else {
        None
    };
    // Open the observation journal before producers start, so an I/O error
    // cannot leave running producers behind an early return.
    let mut publication_transactions = if o.learning_loop_v3 {
        Some(transaction_log::Log::open(&out.join("publication-transactions.jsonl"),
            transaction_log::Counts::capture(resume.completed,resume.version,parent.updates,&resume.lanes,&recall_quotas),
            o.resume_progress.is_some(),migrate_legacy_weights)?)
    } else {None};
    let champions = Arc::new(RwLock::new(Vec::<history::Ranked>::new()));
    let stop = Arc::new(AtomicBool::new(false));
    o.stop_signal = Some(stop.clone());
    // One cheap monitor; game loops read only the atomic flag.
    let monitor_stop = stop.clone();
    let stop_path = out.join("stop-request.json");
    let stop_monitor = std::thread::spawn(move || {
        while !monitor_stop.load(Ordering::Relaxed) && paisho_platform::training_time::now() < end {
            if stop_path.exists() {
                monitor_stop.store(true, Ordering::Relaxed);
                break;
            }
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
    let replay_reference = if o.legacy_replay {
        Some(Arc::new(reference.as_ref().unwrap().model()?))
    } else {
        None
    };
    // The throughput clock starts only after the replay, models and guards load.
    // Campaign deadlines remain independent of this reporting origin.
    let computation_start_elapsed_seconds = prior_elapsed + paisho_platform::training_time::elapsed(started).as_secs_f64();
    atomic_json(&out.join("computation-start.json"), &serde_json::json!({
        "elapsed_seconds": computation_start_elapsed_seconds,
        "initial_lanes": resume.lanes,
        "pid": std::process::id(),
    }))?;
    for actor in 0..o.actors {
        let actor_reference = replay_reference.clone();
        let actor_opponents = opponents.clone();
        let actor_opponent_ladders = opponent_ladders.clone();
        let actor_evaluations = evaluations.clone();
        let actor_curriculum_states = (0..6)
            .map(|group| {
                curriculum_states
                    .get(&ensemble_actor::key(group))
                    .and_then(|m| m.get(&actor))
                    .cloned()
                    .unwrap_or(cases::State {
                        ticket: actor + group * o.actors,
                        ..Default::default()
                    })
            })
            .collect();
        let actor_ladder = ladder.clone();
        let calibration_model = calibration_model.clone();
        let game_pool = if calibration_model.is_some() {
            cpu::Executor::direct(cpu::build_pool(1, None)?.0)
        } else if o.search_pool_shards > 1 {
            cpu::Executor::direct(search_pools[actor % search_pools.len()].clone())
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
            actor_shared.clone(),
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
                if !actor_opponents.is_empty() {
                    ensemble_actor::run(
                        actor,
                        actor_opponents,
                        actor_opponent_ladders,
                        actor_evaluations,
                        actor_curriculum_states,
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
            while !stop.load(Ordering::Relaxed) && paisho_platform::training_time::now() < end {
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
    let teacher_shared=if o.learning_loop_v3 {actor_shared.clone()} else {shared.clone()};
    let reanalysis_worker = o.case_curriculum.as_ref().map(|_| {
        reanalysis::spawn(
            o.clone(),
            reanalysis_rx,
            main_tx.clone(),
            teacher_shared.clone(),
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
        let (o,shared,champions,stop,next,count,historical_started,source,initial,async_error)=(o.clone(),teacher_shared.clone(),champions.clone(),stop.clone(),next.clone(),main_started.clone(),historical_started.clone(),source.clone(),evaluation_anchor.clone(),async_error.clone());
        std::thread::spawn(move || {
            let result=(||->Result<()>{
                cpu::set_qos(o.macos_qos,true)?;
                let old=reference.unwrap().model()?;let mut last=paisho_platform::training_time::now();let mut sweeps=resume.next_history_index;
                while !stop.load(Ordering::Relaxed) && paisho_platform::training_time::now()<end {
                    if o.history_interval>0.0 && paisho_platform::training_time::elapsed(last).as_secs_f64()>=o.history_interval {
                        if !spare.can_admit(o.history_seconds,o.threads,o.historical_capacity_fraction) {
                            std::thread::sleep(Duration::from_millis(50)); continue;
                        }
                        let _reservation=spare.reserve();
                        let snapshot=shared.read().unwrap().clone();
                        if let Some(ranked)=history::assess(snapshot,&initial,&o,end,&pool,sweeps)?{
                            let mut best=champions.write().unwrap();
                            if !best.iter().any(|r|r.snapshot.identity==ranked.snapshot.identity){best.push(ranked);best.sort_by(|a,b|b.score.total_cmp(&a.score));best.truncate(3);}
                            atomic_json(&o.output.join("checkpoint-pool.json"),&best.iter().map(|r|serde_json::json!({"version":r.snapshot.version,"identity":r.snapshot.identity,"score":r.score,"path":r.snapshot.path})).collect::<Vec<_>>())?;
                        }sweeps+=1;last=paisho_platform::training_time::now();continue;
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
    let mut rng = StableRng::new(resume.learner_rng.unwrap_or(o.seed ^ 0x6c6561726e));
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
        o.learning_loop_v3,
        o.publication_transfer,
        proofs.clone(),
        stop.clone(),
        async_error.clone(),
    );
    let mut published_files: Vec<(std::sync::Weak<Snapshot>, PathBuf)> = vec![];
    let mut pruned_targets = 0usize;
    let mut pruned_models = 0usize;
    let mut checkpoint = if o.legacy_replay {
        checkpoint::Checkpoint::with_retention(o.checkpoint_seconds, o.checkpoint_keep)
    } else {
        checkpoint::Checkpoint::new(o.checkpoint_seconds)
    };
    let mut pending_deletions = vec![];
    let mut last_progress: Option<serde_json::Value> = None;
    let mut last_live = paisho_platform::training_time::now();
    let mut consolidation_seconds = 0.0;
    let mut durable_seconds = 0.0;
    let result = (|| -> Result<()> {
        for ready in ready_rx.iter() {
            let archive::Ready {
                evidence_counts,
                registered_proofs,
                mut game,
                durable_path,
                durable_seconds: saved_seconds,
                owned,
                lessons,
                proof_targets,
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
            if game.evaluation.is_some() {
                let group = game
                    .case
                    .as_ref()
                    .and_then(|a| a.opponent_generation.as_ref())
                    .ok_or_else(|| invalid("measurement missing reference"))?;
                let index = o
                    .opponents
                    .iter()
                    .position(|s| &s.generation == group)
                    .ok_or_else(|| invalid("measurement unknown reference"))?;
                evaluations
                    .as_ref()
                    .ok_or_else(|| invalid("measurement outside repaired protocol"))?
                    .observe(index, &game, &opponent_ladders[index])?;
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
            let revisit_due = (game.lane == Lane::Selfplay
                || o.legacy_replay && game.lane == Lane::Historical)
                && count.games % 20 == 0;
            let mut items: Vec<_> = owned
                .iter()
                .enumerate()
                .map(|(index, ex)| LearnItem {
                    example: ex.clone(),
                    kind: 0,
                    lane: game.lane,
                    terminal,
                    fresh_index: Some(index),
                })
                .collect();
            if let Some(path) = durable_path {
                durable_memory.as_mut().unwrap().add(path);
            }
            if let Some(d) = &mut durable_memory {
                for path in registered_proofs {
                    d.register_persisted_proof(path);
                }
            }
            let learning_open = o.learn && paisho_platform::training_time::now() < end && !stop.load(Ordering::Relaxed);
            if learning_open {
                if let Some(p) = &mut protection {
                    p.observe(&owned);
                }
            }
            let quotas = if learning_open && o.structural_repair {
                recall_quotas.allocate(
                    owned.len(),
                    o.replay_ratio,
                    o.recall_fraction,
                    if human.is_empty() {
                        0.0
                    } else {
                        o.human_fraction
                    },
                )
            } else {
                (0, 0)
            };
            let rehearsal_budget = if learning_open && o.structural_repair {
                quotas.0
            } else if learning_open {
                o.case_curriculum.as_ref().map_or(0, |c| {
                    (owned.len().saturating_mul(o.replay_ratio) as f64 * c.durable_fraction).round()
                        as usize
                })
            } else {
                0
            };
            let consolidation_started = paisho_platform::training_time::now();
            let rehearsal = match &mut durable_memory {
                Some(d) if o.structural_repair => {
                    d.rehearse_cached(rehearsal_budget, &mut rng, &model)?
                }
                Some(d) => d.rehearse(rehearsal_budget, &mut rng, &model)?,
                None => vec![],
            };
            consolidation_seconds += paisho_platform::training_time::elapsed(consolidation_started).as_secs_f64();
            if revisit_due {
                if let Some(task) = durable_memory.as_mut().and_then(|d| d.revisit.take()) {
                    let _ = reanalysis_tx.try_send(task);
                }
            }
            let durable_draws = rehearsal.len();
            if o.structural_repair && durable_draws != rehearsal_budget {
                return Err(invalid(
                    "durable archive cannot supply the requested recall quota",
                ));
            }
            let winning_draws = durable_memory.as_ref().map_or(0, |d| d.last_winning_draws);
            items.extend(
                rehearsal
                    .into_iter()
                    .enumerate()
                    .map(|(i, example)| LearnItem {
                        example,
                        kind: if i < winning_draws { 5 } else { 4 },
                        lane: Lane::Selfplay,
                        terminal: false,
                        fresh_index: None,
                    }),
            );
            if learning_open {
                let mut human_due = quotas.1;
                for _ in 0..owned
                    .len()
                    .saturating_mul(o.replay_ratio)
                    .saturating_sub(durable_draws)
                {
                    if !human.is_empty()
                        && (if o.structural_repair {
                            human_due > 0
                        } else {
                            rng.next_f64() < o.human_fraction
                        })
                    {
                        human_due = human_due.saturating_sub(1);
                        items.push(LearnItem {
                            example: human[rng.index(human.len())].clone(),
                            kind: 2,
                            lane: Lane::Selfplay,
                            terminal: true,
                            fresh_index: None,
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
                        if let Some((ex, replay_lane)) = prioritized
                            .or_else(|| memory.draw(&mut rng, prefer))
                            .or_else(|| {
                                if o.structural_repair && !owned.is_empty() {
                                    Some((owned[rng.index(owned.len())].clone(), game.lane))
                                } else {
                                    None
                                }
                            })
                        {
                            items.push(LearnItem {
                                example: ex,
                                kind,
                                lane: replay_lane,
                                terminal: false,
                                fresh_index: None,
                            });
                        }
                    }
                }
                shuffle(&mut items, &mut rng);
            }
            let before = updates;
            let mut durable_used = 0;
            let mut winning_used = 0;
            let mut fresh_used = 0;
            let mut replay_used = 0;
            let mut correction_replay_used = 0;
            let mut human_game_used = 0;
            let t = paisho_platform::training_time::now();
            if o.learn {
                for batch in items.chunks(64) {
                    if paisho_platform::training_time::now() >= end || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let refs: Vec<_> = batch.iter().map(|item| item.example.as_ref()).collect();
                    let rate = if o.structural_repair {
                        o.rate * batch.len() as f64 / 64.0
                    } else {
                        o.rate
                    };
                    if let Some(p) = &mut protection {
                        let shared = batch.iter().map(|item|item.example.clone()).collect::<Vec<_>>();
                        p.train_shared(&mut model, &shared, rate, 1e-5)?;
                    } else {
                        if o.inline_learning {
                            model.train_batch_inline(&refs, rate, 1e-5)
                        } else {
                            pool.install(|| model.train_batch(&refs, rate, 1e-5))
                        }
                        .map_err(invalid)?;
                    }
                    updates += 1;
                    if let Some(clock)=&mut publication_work_clock {
                        clock.consumed(batch.len(), &target_hash, batch.iter().filter_map(|item|item.fresh_index));
                    }
                    for item in batch {
                        if o.structural_repair {
                            recall_quotas.consumed_examples += 1;
                        }
                        if item.kind == 4 || item.kind == 5 {
                            if item.kind == 5 {
                                winning_used += 1;
                            }
                            durable_used += 1;
                            if o.structural_repair {
                                recall_quotas.consumed_recall += 1;
                            }
                        }
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
                            1 | 3 | 4 | 5 => {
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
            if learning_open && o.structural_repair {
                let consumed = fresh_used + replay_used + human_game_used;
                if let Some(d) = &mut durable_memory {
                    for item in &items[consumed..] {
                        if item.kind == 4 || item.kind == 5 {
                            d.defer(item.example.clone(), item.kind == 5);
                        }
                    }
                }
                recall_quotas.settle(
                    items.len().saturating_sub(consumed),
                    durable_draws - durable_used,
                    o.recall_fraction,
                );
            }
            learner_seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
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
            let t = paisho_platform::training_time::now();
            if updates > before {
                let publication_due=learning_open && publication.as_ref().is_some_and(|g| {
                    match (&publication_work_clock,&o.publication_work_budget) {
                        (Some(clock),Some(budget))=>clock.due(budget),
                        _=>g.due(),
                    }
                });
                let work_actor_before=publication_due.then(||publication.as_ref().unwrap().accepted().identity.clone());
                let transaction_before=if publication_due && publication_transactions.is_some() {
                    Some((publication.as_ref().unwrap().accepted(),
                        prior_elapsed+paisho_platform::training_time::elapsed(started).as_secs_f64()-computation_start_elapsed_seconds))
                } else {None};
                let mut transaction_rebased=false;
                if publication_due {
                    if o.learning_loop_v3 {
                        let feedback=publication.as_mut().unwrap().observe_candidate(&model)?;
                        durable_memory.as_mut().unwrap().focus_policy(feedback);
                    }
                    if let Some(p) = &mut protection {
                        if o.publication_transfer {
                            p.consolidate_for_publication(&mut model)?;
                            publication.as_mut().unwrap()
                                .set_publication_fresh(p.publication_fresh_contract()?)?;
                        } else if o.diagnostic_consolidation_capture {
                            p.consolidate_diagnostic(&mut model, &out.join("consolidation-audit"),
                                version + 1, updates, game.id, &publication.as_ref().unwrap().accepted())?;
                        } else {
                            p.consolidate(&mut model)?;
                        }
                    }
                }
                version += 1;
                let artifact = MicroArtifact::new(
                    &model,
                    updates,
                    serde_json::json!({"kind":"gen5-selfplay","rules":RULES.as_str(),"parent":parent_identity,"source_run":source,"version":version,"last_game":game.id,"search_mode":o.mode}),
                );
                let path = out.join("models").join(format!("model-{version:07}.json"));
                if o.checkpoint_seconds == 0.0 {
                    artifact.save(&path)?;
                }
                let identity = artifact_identity::identity(&artifact, &identity_workers);
                let mut published = Arc::new(Snapshot {
                    artifact: Some(Arc::new(artifact)),
                    version,
                    identity,
                    model: Arc::new(model.clone()),
                    path: path.clone(),
                });
                *shared.write().unwrap() = published.clone();
                let transaction_candidate=transaction_before.as_ref().map(|_|published.clone());
                if let Some(guard) = &mut publication {
                    // Use one deadline decision for the entire V3 transaction.
                    // Hashing an artifact must not open a publication window
                    // that skipped consolidation earlier in this receipt.
                    let checked=if o.learning_loop_v3 {
                        if publication_due {guard.consider(published.clone(),true)?} else {None}
                    } else {guard.consider(published.clone(),false)?};
                    if let Some(focus) = checked {
                        durable_memory.as_mut().unwrap().focus_policy(focus);
                        *actor_shared.write().unwrap() = guard.accepted();
                        if o.learning_loop_v3 {
                            let accepted=guard.accepted();
                            protection.as_mut().unwrap().adopt_accepted(&accepted.model)?;
                            // Start the next transaction from the accepted joint
                            // weights, including any policy repair. A rejected
                            // shadow model must not silently become the learner's
                            // next baseline. Preserve consumed update counters.
                            model=accepted.model.as_ref().clone();
                            transaction_rebased=true;
                            let final_artifact=MicroArtifact::new(&model,updates,
                                serde_json::json!({"kind":"gen5-transaction-boundary",
                                    "accepted":accepted.identity,"candidate":published.identity,
                                    "version":version,"updates_consumed":updates}));
                            if o.checkpoint_seconds==0. {final_artifact.save(&path)?;}
                            published=Arc::new(Snapshot {identity:artifact_identity::identity(&final_artifact,&identity_workers),
                                artifact:Some(Arc::new(final_artifact)),version,
                                model:Arc::new(model.clone()),path:path.clone()});
                            *shared.write().unwrap()=published.clone();
                        }
                    }
                } else {
                    *actor_shared.write().unwrap() = published.clone();
                }
                if let (Some(log),Some((before_actor,control_start)),Some(candidate)) =
                    (&mut publication_transactions,transaction_before,transaction_candidate) {
                    let guard=publication.as_ref().unwrap();
                    log.boundary(game.id,control_start,
                        prior_elapsed+paisho_platform::training_time::elapsed(started).as_secs_f64()-computation_start_elapsed_seconds,
                        transaction_log::Counts::capture(completed,version,updates,&counters,&recall_quotas),
                        &before_actor,&candidate,&guard.accepted(),&published,
                        &guard.progress(),&protection.as_ref().unwrap().progress(),transaction_rebased)?;
                }
                if o.case_curriculum.is_some() && o.checkpoint_seconds == 0.0 && version % 100 != 0
                {
                    published_files.push((Arc::downgrade(&published), path));
                }
                if transaction_rebased {
                    if let (Some(clock),Some(before))=(&mut publication_work_clock,work_actor_before) {
                        clock.completed(before!=publication.as_ref().unwrap().accepted().identity);
                    }
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
            publish_seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
            let fully_learned =
                o.learn && fresh_used == fresh_count && (fresh_count == 0 || updates > before);
            if let Some(attempt) = game
                .case
                .as_ref()
                .filter(|a| a.kind != "archive-reanalysis" && a.kind != "frozen-measurement")
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
                if let Some(group) = &attempt.opponent_generation {
                    curriculum_states
                        .entry(group.clone())
                        .or_default()
                        .insert(attempt.actor, state);
                } else {
                    case_states.insert(attempt.actor, state);
                }
            }
            if o.structural_repair && fully_learned {
                if let Some(d) = &mut durable_memory {
                    for (key, group, ex, proof) in lessons {
                        d.admit(key, group, ex, proof);
                    }
                }
                if o.publication_transfer {
                    publication.as_mut().unwrap().observe_consumed(
                        &game, &proof_targets, fully_learned, &source, updates,
                    )?;
                }
            }
            if fully_learned && !o.learning_loop_repair {
                if opponents.is_empty() && o.legacy_replay {
                    ladder.write().unwrap().observe(&game, version);
                } else if let Some(group) = game
                    .case
                    .as_ref()
                    .and_then(|a| a.opponent_generation.as_ref())
                {
                    if let Some(index) = o.opponents.iter().position(|s| s.generation == *group) {
                        opponent_ladders[index]
                            .write()
                            .unwrap()
                            .observe(&game, version);
                    }
                }
            }
            checkpoint.observe(&game, &o.opponents);
            let mut receipt = serde_json::json!({"rules":RULES.as_str(),"id":game.id,"lane":lane,"collector_version":game.snapshot.version,"collector":game.snapshot.identity,"opponent":game.opponent,"reference_budget":game.reference_budget,"candidate_seat":format!("{:?}",game.candidate_seat),"termination":game.termination,"outcome":format!("{:?}",game.outcome),"error":game.error,"cycle":game.cycle,"decisions":game.record.actions().len(),"seconds":game.seconds,"cap_seconds":game.cap_seconds,"campaign_censored":game.campaign_censored,"search_seconds":game.search_seconds,"pool_wait_seconds":game.pool_wait_seconds,"maintenance_seconds":game.maintenance_seconds,"sample_seconds":game.sample_seconds,"simulations":game.simulations,"inherited_visits":game.inherited,"inference_evaluations":game.evals,"inference_cache_hits":game.hits,"forced_playouts":game.forced_playouts,"pruned_policy_visits":game.pruned_policy_visits,"policy_searches":game.policy_searches,"policy_coverage_sum":game.policy_coverage_sum,"psr_sha256":psr_hash,"targets_file":format!("{stem}.targets.json.gz"),"targets_sha256":target_hash,"eligible_examples":fresh_count,"fresh_used":fresh_used,"replay_used":replay_used,"correction_replay_used":correction_replay_used,"correction_candidates":correction_candidates,"human_used":human_game_used,"model_version":version,"updates":updates,"learned_batches":updates-before,"search_mode":o.mode});
            receipt.as_object_mut().unwrap().extend(serde_json::json!({"tactical_evaluations":game.tactical_evaluations,"case":game.case,"reanalysis":game.reanalysis,"prefix_decisions":game.prefix_decisions,"continuation_decisions":game.record.actions().len()-game.prefix_decisions,"durable_draws":durable_used,"durable_scheduled":durable_draws,"structural_repair":o.structural_repair,"recall_fraction_all_examples":if o.structural_repair{Some(o.recall_fraction)}else{None},"learning_normalization":if o.structural_repair{"64-examples"}else{"per-batch"},"fully_learned":fully_learned,"durable_save_seconds":saved_seconds}).as_object().unwrap().clone());
            receipt["fresh_evidence"] = evidence_counts;
            receipt["learning_loop_repair"] = o.learning_loop_repair.into();
            receipt["learning_loop_v2"] = o.learning_loop_v2.into();
            if o.publication_transfer { receipt["publication_transfer"] = true.into(); }
            receipt["neural_memory_parameters"] = if model.has_neural_memory(){MICRO_NEURAL_MEMORY_WEIGHTS}else{0}.into();
            receipt["reused_search"] = game.reused_search.into();
            receipt["search_evidence_simulations"] = game.search_evidence_simulations.into();
            receipt["search_cache_key"] = serde_json::to_value(&game.search_cache_key)?;
            receipt["value_policy_strength"] = o.value_policy_strength.into();
            receipt["minimum_search_depth"] = o.minimum_search_depth.into();
            receipt["measurement"] = game.measurement.into();
            receipt["frozen_evaluation"] = serde_json::to_value(&game.evaluation)?;
            receipt["actor_version"] = actor_shared.read().unwrap().version.into();
            receipt["proved_winning_recall_used"] = winning_used.into();
            receipt["reference_identity"] = game
                .case
                .as_ref()
                .filter(|_| game.reference_budget.is_some())
                .and_then(|a| a.opponent_generation.as_ref())
                .and_then(|g| o.opponents.iter().find(|s| s.generation == *g))
                .map(|s| s.sha256.clone())
                .into();
            if o.checkpoint_seconds > 0.0 {
                durable::write_pending(&dir.join(format!("{stem}.json")), &receipt)?;
            } else {
                save_json_new(&dir.join(format!("{stem}.json")), &receipt)?;
            }
            let (proof_disk_count, proof_cache_entries) = {
                let c = proofs.read().unwrap();
                (c.disk_count, c.len())
            };
            let mut progress = serde_json::json!({"legacy_ladder":*ladder.read().unwrap(),"next_game_id":next.load(Ordering::Relaxed),"case_states":case_states,"case_manifest_sha256":case_manifest_sha256,"dense_targets_pruned":pruned_targets,"unreferenced_models_pruned":pruned_models,"consolidation_seconds":consolidation_seconds,"durable_save_seconds":durable_seconds,"durable_bundles":durable_memory.as_ref().map(|d|d.len()),"durable_draws":durable_memory.as_ref().map(|d|d.draws),"durable_proofs":proof_disk_count,"proof_cache_entries":proof_cache_entries,"completed":completed,"last_game_id":game.id,"lanes":counters,"version":version,"updates":updates,"elapsed_seconds":prior_elapsed+paisho_platform::training_time::elapsed(started).as_secs_f64(),"computation_start_elapsed_seconds":computation_start_elapsed_seconds,"paused_seconds":paisho_platform::training_time::paused().as_secs_f64(),"remaining_seconds":end.saturating_duration_since(paisho_platform::training_time::now()).as_secs_f64(),"main_started":main_started.load(Ordering::Relaxed),"historical_started":historical_started.load(Ordering::Relaxed),"main_cpu_capacity":o.main_threads(),"historical_cpu_capacity":if history_worker.is_some(){o.secondary_capacity()}else{0},"secondary_cpu_capacity":o.secondary_capacity(),"secondary_usage":secondary.as_ref().map(|s|s.telemetry()),"replay_positions":memory.len(),"correction_pool_positions":memory.correction_len(),"correction_reference_bytes":memory.correction_reference_bytes(),"replay_bytes":memory.bytes,"replay_evicted":memory.evicted,"human_examples":human.len(),"human_used":human_used,"fresh_terminal_used":fresh_terminal_used,"learner_seconds":learner_seconds,"archive_seconds":archive_seconds,"publish_seconds":publish_seconds,"async_error":*async_error.read().unwrap(),"errors":errors});
            if !opponents.is_empty() {
                progress["curriculum_states"] = serde_json::to_value(&curriculum_states)?;
                progress["opponent_ladders"] = serde_json::to_value(
                    o.opponents
                        .iter()
                        .zip(opponent_ladders.iter())
                        .map(|(s, l)| (s.generation.clone(), l.read().unwrap().clone()))
                        .collect::<BTreeMap<_, _>>(),
                )?;
            }
            if let Some(e) = &evaluations {
                progress["frozen_evaluations"] = e.progress();
            }
            if let Some(g) = &publication {
                progress["publication_guard"] = g.progress();
            }
            progress["learning_loop_repair"] = o.learning_loop_repair.into();
            progress["learning_loop_v2"] = o.learning_loop_v2.into();
            progress["learning_loop_v3"] = o.learning_loop_v3.into();
            if o.publication_transfer { progress["publication_transfer"] = true.into(); }
            progress["initial_weight_migration"] = weight_migration.clone();
            progress["neural_memory_parameters"] = if model.has_neural_memory(){MICRO_NEURAL_MEMORY_WEIGHTS}else{0}.into();
            if let Some(bank)=model.sequence_memory() {
                progress["sequence_cache"] = serde_json::json!({"queries_hits_candidates":bank.telemetry(),"shard_entries":bank.cache_occupancy()});
            }
            if let Some(cache) = &o.reanalysis_cache {
                progress["reanalysis_cache"] = cache.progress();
            }
            if let Some(p) = &protection {
                progress["protection"] = p.progress();
            }
            progress["checkpoint_models_retained"] = checkpoint.retained().into();
            progress["checkpoint_models_pruned"] = checkpoint.pruned().into();
            progress["recall_quotas"] = serde_json::to_value(&recall_quotas)?;
            if let Some(clock)=&publication_work_clock {
                progress["publication_work_clock"]=serde_json::to_value(clock)?;
                progress["publication_work_budget"]=serde_json::to_value(&o.publication_work_budget)?;
            }
            progress["durable_recall"] = durable_memory
                .as_ref()
                .map_or(serde_json::Value::Null, |d| d.progress());
            progress["learner_rng"] = rng.state().into();
            println!("{receipt}");
            if o.checkpoint_seconds > 0.0 {
                let mut progress = progress;
                if checkpoint.due() {
                    if let Some(p) = &mut protection {
                        p.checkpoint(&out)?;
                        progress["protection"] = p.progress();
                    }
                    if let Some(d) = &mut durable_memory {
                        d.checkpoint(&out)?;
                        progress["durable_recall"] = d.progress();
                    }
                    checkpoint.commit(&out, &shared.read().unwrap(), &memory, &mut progress)?;
                    if let Some(e) = &evaluations {
                        e.committed()?;
                    }
                    if let Some(g) = &mut publication {
                        g.committed()?;
                    }
                    if let Some(p) = &mut protection {
                        p.committed()?;
                    }
                    if let Some(d) = &mut durable_memory {
                        d.committed()?;
                    }
                    for path in pending_deletions.drain(..) {
                        fs::remove_file(path)?;
                        pruned_targets += 1;
                    }
                }
                progress["durable_version"] = checkpoint.version.into();
                progress["checkpoint_seconds"] = o.checkpoint_seconds.into();
                if last_progress.is_none() || paisho_platform::training_time::elapsed(last_live).as_secs_f64() >= 1.0 {
                    durable::write_pending(&out.join("progress.json"), &progress)?;
                    last_live = paisho_platform::training_time::now();
                }
                last_progress = Some(progress);
            } else {
                durable::write(&out.join("progress.json"), &progress)?;
            }
            if let Some(ack) = game.ack.take() {
                let _ = ack.send(case_actor::Feedback {
                    snapshot: actor_shared.read().unwrap().clone(),
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
        if let Some(p) = &mut protection {
            p.checkpoint(&out)?;
            progress["protection"] = p.progress();
        }
        if let Some(d) = &mut durable_memory {
            d.checkpoint(&out)?;
            progress["durable_recall"] = d.progress();
        }
        checkpoint.commit(&out, &shared.read().unwrap(), &memory, &mut progress)?;
        durable::write(&out.join("progress.json"), &progress)?;
        if let Some(e) = &evaluations {
            e.committed()?;
        }
        if let Some(g) = &mut publication {
            g.committed()?;
        }
        if let Some(p) = &mut protection {
            p.committed()?;
        }
        if let Some(d) = &mut durable_memory {
            d.committed()?;
        }
        for path in pending_deletions {
            fs::remove_file(path)?;
        }
    }
    MicroArtifact::new(&model,updates,serde_json::json!({"kind":"gen5-final","rules":RULES.as_str(),"parent":parent_identity,"source_run":source,"version":version,"search_mode":o.mode})).save(&out.join("model.json"))?;
    memory.save(&out.join("replay-final.index.json"))?;
    if let Some(log)=&mut publication_transactions {
        log.finish(prior_elapsed+paisho_platform::training_time::elapsed(started).as_secs_f64()-computation_start_elapsed_seconds,
            transaction_log::Counts::capture(completed,version,updates,&counters,&recall_quotas),
            shared.read().unwrap().as_ref(),&publication.as_ref().unwrap().accepted())?;
    }
    save_json_new(
        &out.join("report.json"),
        &serde_json::json!({"rules":RULES.as_str(),"completed":completed,"lanes":counters,"version":version,"updates":updates,"elapsed_seconds":prior_elapsed+paisho_platform::training_time::elapsed(started).as_secs_f64(),"computation_start_elapsed_seconds":computation_start_elapsed_seconds,"fresh_terminal_used":fresh_terminal_used,"human_used":human_used,"learner_seconds":learner_seconds,"archive_seconds":archive_seconds,"publish_seconds":publish_seconds,"replay_positions":memory.len(),"correction_pool_positions":memory.correction_len(),"correction_reference_bytes":memory.correction_reference_bytes(),"replay_bytes":memory.bytes,"errors":errors,"automatic_resume":false,"secondary_usage":secondary.as_ref().map(|s|s.telemetry())}),
    )?;
    if !errors.is_empty() {
        return Err(invalid(errors.join("; ")));
    }
    Ok(())
}

#[cfg(test)]
mod publication_transfer_resume_tests {
    use super::*;
    #[test]
    fn publication_transfer_cannot_silently_drop_restored_obligations() {
        let saved = serde_json::json!({"publication_transfer":true});
        assert!(validate_publication_transfer_resume(false, &saved).is_err());
        assert!(validate_publication_transfer_resume(true, &saved).is_ok());
        assert!(validate_publication_transfer_resume(false, &serde_json::Value::Null).is_ok());
        assert!(validate_publication_transfer_resume(true, &serde_json::Value::Null).is_ok());
    }
}
