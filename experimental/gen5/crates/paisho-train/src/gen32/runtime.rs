use super::*;
mod lessons;
use lessons::Lessons;
use std::{
    collections::{HashSet, VecDeque},
    io::Write,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc, Mutex, RwLock,
    },
    time::{Duration, Instant},
};
#[derive(Clone, Serialize, Deserialize)]
struct Row {
    path: PathBuf,
    sha256: String,
    index: usize,
}
struct Entry {
    row: Row,
    example: Arc<MicroExample>,
    bytes: usize,
}
fn size(e: &MicroExample) -> usize {
    std::mem::size_of::<MicroExample>() + e.actions.capacity() * 256 + e.policy.capacity() * 8
}
fn checkpoint(
    out: &Path,
    parent: &Artifact,
    model: &Gen32Model,
    updates: u64,
    replay: &VecDeque<Entry>,
    lessons: &Lessons,
    progress: &mut serde_json::Value,
    retired: &mut HashSet<PathBuf>,
) -> Result<()> {
    let previous: Option<serde_json::Value> = fs::read(out.join("checkpoint.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok());
    let model_path = out.join("models").join(format!("model-{updates}.json"));
    parent.updated(model, updates).save(&model_path)?;
    let index = out.join(format!("replay-{updates}.json"));
    atomic(
        &index,
        &serde_json::json!({"rules":RULES.as_str(),"rows":replay.iter().map(|e|&e.row).collect::<Vec<_>>(),"corrections":lessons.catalog,"correction_cursor":lessons.cursor,"correction_draws":lessons.draws,"correction_reread":lessons.reread}),
    )?;
    progress["model"] = model_path.to_string_lossy().into_owned().into();
    progress["replay_index"] = index.to_string_lossy().into_owned().into();
    fs::File::open(out.join("models"))?.sync_all()?;
    fs::File::open(out.join("games"))?.sync_all()?;
    atomic(&out.join("checkpoint.json"), progress)?;
    fs::File::open(out)?.sync_all()?;
    // Only current-run targets that actually left the FIFO are eligible. Older
    // runs keep their own resumable checkpoints; every original PSR is retained.
    let active: HashSet<_> = replay.iter().map(|e| e.row.path.clone()).collect();
    let mut removed = 0;
    for path in retired.drain() {
        if !active.contains(&path) && path.parent() == Some(out.join("games").as_path()) {
            if path.exists() {
                fs::remove_file(path)?;
                removed += 1;
            }
        }
    }
    if removed > 0 {
        let mut ledger = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(out.join("retention.jsonl"))?;
        writeln!(
            ledger,
            "{}",
            serde_json::json!({"checkpoint_updates":updates,"dense_target_files_removed":removed,"original_psrs_retained":true})
        )?;
        ledger.sync_all()?;
    }

    if let Some(old) = previous.and_then(|v| v["replay_index"].as_str().map(PathBuf::from)) {
        if old != index && old.parent() == Some(out) {
            fs::remove_file(old)?;
        }
    }
    Ok(())
}
pub fn run(o: Options) -> Result<()> {
    o.validate()?;
    let parent = Artifact::load(&o.model)?;
    let mut model = parent.model()?;
    // Load and validate the frozen opponent once, shared by all actors in RAM.
    let reference = if let Some(path) = &o.historical_reference {
        let bytes = fs::read(path)?;
        if Some(sha256(&bytes)) != o.historical_reference_sha256 {
            return Err(invalid("historical reference hash mismatch"));
        }
        let artifact: ModelArtifact = serde_json::from_slice(&bytes)?;
        Some(Arc::new(artifact.model()?))
    } else {
        None
    };
    fs::create_dir_all(&o.output)?;
    let out = o.output.canonicalize()?;
    if out.join("progress.json").exists() {
        return Err(invalid("output already contains a run"));
    }
    fs::create_dir_all(out.join("games"))?;
    fs::create_dir_all(out.join("models"))?;
    let mut replay: VecDeque<Entry> = VecDeque::new();
    let mut bytes = 0;
    let mut retired = HashSet::new();
    let mut lessons = Lessons::default();
    if let Some(path) = &o.replay_index {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        if v["rules"] != RULES.as_str() {
            return Err(invalid("replay rules mismatch"));
        }
        lessons = Lessons::restore(&v)?;
        let rows: Vec<Row> = serde_json::from_value(v["rows"].clone())?;
        let mut last = PathBuf::new();
        let mut examples = vec![];
        for row in rows {
            if row.path != last {
                let b = fs::read(&row.path)?;
                if sha256(&b) != row.sha256 {
                    return Err(invalid("replay hash mismatch"));
                }
                examples = crate::micro_learning::load_examples(&row.path)?;
                last = row.path.clone();
            }
            let e = Arc::new(
                examples
                    .get(row.index)
                    .ok_or_else(|| invalid("replay index"))?
                    .example()?,
            );
            let n = size(&e);
            bytes += n;
            replay.push_back(Entry {
                row,
                example: e,
                bytes: n,
            });
        }
        if replay.len() > o.replay_capacity || bytes > o.replay_max_bytes {
            return Err(invalid("replay requires more capacity"));
        }
    }
    let begin = Instant::now();
    let end = begin + Duration::from_secs_f64(o.seconds);
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(if o.parallel_candidates { o.threads } else { 1 })
            .build()?,
    );
    let shared = Arc::new(RwLock::new((parent.updates, Arc::new(model.clone()))));
    let stop = Arc::new(AtomicBool::new(false));
    let next = Arc::new(AtomicUsize::new(o.actors.min(o.games)));
    let errors = Arc::new(Mutex::new(Vec::<String>::new()));
    let (tx, rx) = mpsc::sync_channel(o.actors * 2);
    let mut actors = vec![];
    for actor in 0..o.actors {
        let (o, pool, shared, stop, next, tx, errors, out) = (
            o.clone(),
            if o.parallel_candidates {
                pool.clone()
            } else {
                Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build()?)
            },
            shared.clone(),
            stop.clone(),
            next.clone(),
            tx.clone(),
            errors.clone(),
            out.clone(),
        );
        let reference = reference.clone();
        actors.push(std::thread::spawn(move || {
            let mut ordinal = 0;
            while Instant::now() < end && !stop.load(Ordering::Relaxed) {
                let id = if ordinal == 0 {
                    actor
                } else {
                    next.fetch_add(1, Ordering::Relaxed)
                };
                if id >= o.games {
                    break;
                }
                let (version, m) = shared.read().unwrap().clone();
                let play = || {
                    game::play(
                        id,
                        actor,
                        ordinal,
                        version,
                        &m,
                        &o,
                        end,
                        &stop,
                        &pool,
                        &out.to_string_lossy(),
                        if o.historical_game(actor, ordinal) {
                            reference.as_deref().map(|m| m as &dyn MctsEvaluator)
                        } else {
                            None
                        },
                    )
                };
                let result = if o.parallel_candidates {
                    play()
                } else {
                    pool.install(play)
                };
                match result {
                    Ok(g) => {
                        if tx.send(g).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        errors.lock().unwrap().push(e);
                        stop.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                ordinal += 1;
            }
        }));
    }
    drop(tx);
    let rx = Arc::new(Mutex::new(rx));
    let (tx, ready) = mpsc::sync_channel(o.actors * 2);
    let mut archives = vec![];
    for _ in 0..o.archive_workers {
        let (rx, tx, out, errors, stop) = (
            rx.clone(),
            tx.clone(),
            out.clone(),
            errors.clone(),
            stop.clone(),
        );
        archives.push(std::thread::spawn(move || loop {
            let g = match rx.lock().unwrap().recv() {
                Ok(g) => g,
                Err(_) => break,
            };
            let result = (|| -> Result<_> {
                let stem = format!("game-{:07}", g.receipt["id"].as_u64().unwrap());
                let psr = g.record.to_string();
                let psr_path = out.join("games").join(format!("{stem}.psr"));
                fs::write(&psr_path, psr.as_bytes())?;
                fs::File::open(&psr_path)?.sync_all()?;
                let path = out.join("games").join(format!("{stem}.targets.json.gz"));
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                serde_json::to_writer(&mut encoder, &g.targets)?;
                let b = encoder.finish()?;
                let hash = sha256(&b);
                fs::write(&path, &b)?;
                fs::File::open(&path)?.sync_all()?;
                let corrections: Vec<_> = g
                    .targets
                    .iter()
                    .filter(|s| s.correction_priority)
                    .take(4)
                    .collect();
                let mut correction_rows = Vec::new();
                if !corrections.is_empty() {
                    let p = out
                        .join("games")
                        .join(format!("{stem}.corrections.json.gz"));
                    let mut e =
                        flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                    serde_json::to_writer(&mut e, &corrections)?;
                    let data = e.finish()?;
                    fs::write(&p, &data)?;
                    fs::File::open(&p)?.sync_all()?;
                    let sha = sha256(&data);
                    correction_rows = (0..corrections.len())
                        .map(|index| Row {
                            path: p.clone(),
                            sha256: sha.clone(),
                            index,
                        })
                        .collect();
                }
                let mut receipt = g.receipt;
                receipt["durable_corrections"] = correction_rows.len().into();
                receipt["psr_sha256"] = sha256(psr.as_bytes()).into();
                receipt["targets_sha256"] = hash.clone().into();
                atomic(&out.join("games").join(format!("{stem}.json")), &receipt)?;
                Ok((g.targets, path, hash, receipt, correction_rows))
            })();
            match result {
                Ok(g) => {
                    if tx.send(g).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    errors.lock().unwrap().push(e.to_string());
                    stop.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }));
    }
    drop(tx);
    drop(rx);
    let mut rng = StableRng::new(o.seed);
    let mut updates = parent.updates;
    let mut completed = 0;
    let mut terminals = 0;
    let mut fresh = 0;
    let mut reread = 0;
    let mut progress = serde_json::json!({"generation":parent.generation,"rules":RULES.as_str(),"updates":updates,"completed":0,"terminal":0,"fresh":0,"replay_used":0});
    checkpoint(
        &out,
        &parent,
        &model,
        updates,
        &replay,
        &lessons,
        &mut progress,
        &mut retired,
    )?;
    let mut last = Instant::now();
    let mut last_status = Instant::now();
    let mut learner_seconds = 0.;
    let mut recent = VecDeque::new();
    let mut journal = fs::OpenOptions::new()
        .create_new(true)
        .append(true)
        .open(out.join("receipts.jsonl"))?;
    loop {
        if out.join("stop-request.json").exists()
            || out.join("pause-request.json").exists()
            || Instant::now() >= end
        {
            stop.store(true, Ordering::Relaxed);
        }
        match ready.recv_timeout(Duration::from_millis(100)) {
            Ok((saved, path, hash, mut receipt, correction_rows)) => {
                completed += 1;
                terminals += usize::from(receipt["termination"] == "rules-terminal");
                let started = Instant::now();
                let before = updates;
                lessons.add_saved(correction_rows, &saved)?;
                if o.learn && Instant::now() < end && !stop.load(Ordering::Relaxed) {
                    for (index, s) in saved.iter().enumerate() {
                        let e = Arc::new(s.example()?);
                        model.train(&e, o.rate).map_err(invalid)?;
                        updates += 1;
                        fresh += 1;
                        let n = size(&e);
                        bytes += n;
                        replay.push_back(Entry {
                            row: Row {
                                path: path.clone(),
                                sha256: hash.clone(),
                                index,
                            },
                            example: e,
                            bytes: n,
                        });
                        while replay.len() > o.replay_capacity || bytes > o.replay_max_bytes {
                            let gone = replay.pop_front().unwrap();
                            bytes -= gone.bytes;
                            retired.insert(gone.row.path);
                        }
                        for _ in 0..o.replay_ratio {
                            if !replay.is_empty() {
                                let lesson = if o.correction_replay {
                                    lessons.sample(&mut rng)?
                                } else {
                                    None
                                };
                                let example = lesson.unwrap_or_else(|| {
                                    replay[rng.index(replay.len())].example.clone()
                                });
                                model.train(&example, o.rate).map_err(invalid)?;
                                updates += 1;
                                reread += 1;
                            }
                        }
                        if Instant::now() >= end {
                            break;
                        }
                    }
                    *shared.write().unwrap() = (updates, Arc::new(model.clone()));
                }
                learner_seconds += started.elapsed().as_secs_f64();
                receipt["updates"] = updates.into();
                receipt["learned"] = (updates - before).into();
                receipt["published_unix_seconds"] = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs_f64()
                    .into();
                writeln!(journal, "{}", serde_json::to_string(&receipt)?)?;
                recent.push_front(receipt.clone());
                recent.truncate(8);
                progress = serde_json::json!({"generation":parent.generation,"rules":RULES.as_str(),"updates":updates,"completed":completed,"terminal":terminals,"fresh":fresh,"replay_used":reread,"durable_corrections":lessons.catalog.len(),"correction_replays":lessons.reread,"replay_positions":replay.len(),"replay_bytes":bytes,"elapsed_seconds":begin.elapsed().as_secs_f64(),"remaining_seconds":end.saturating_duration_since(Instant::now()).as_secs_f64(),"learner_seconds":learner_seconds,"threads":o.threads,"actors":o.actors,"archive_workers":o.archive_workers,"recent_games":recent,"last_game":receipt});
                if last.elapsed().as_secs_f64() >= o.checkpoint_seconds {
                    journal.sync_all()?;
                    checkpoint(
                        &out,
                        &parent,
                        &model,
                        updates,
                        &replay,
                        &lessons,
                        &mut progress,
                        &mut retired,
                    )?;
                    last = Instant::now();
                }
                if last_status.elapsed().as_secs_f64() >= 1. {
                    atomic(&out.join("progress.json"), &progress)?;
                    last_status = Instant::now();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    stop.store(true, Ordering::Relaxed);
    for h in actors.into_iter().chain(archives) {
        if h.join().is_err() {
            errors.lock().unwrap().push("worker panicked".into());
        }
    }
    journal.sync_all()?;
    checkpoint(
        &out,
        &parent,
        &model,
        updates,
        &replay,
        &lessons,
        &mut progress,
        &mut retired,
    )?;
    progress["errors"] = serde_json::to_value(&*errors.lock().unwrap())?;
    atomic(&out.join("progress.json"), &progress)?;
    atomic(&out.join("summary.json"), &progress)?;
    parent
        .updated(&model, updates)
        .save(&out.join("model.json"))?;
    if !errors.lock().unwrap().is_empty() {
        return Err(invalid("Gen3 worker error; see summary"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pruning_preserves_psrs_active_replay_and_previous_campaigns() {
        let root = std::env::temp_dir().join(format!("gen32-retention-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let out = root.join("current");
        fs::create_dir_all(out.join("games")).unwrap();
        fs::create_dir_all(out.join("models")).unwrap();
        let parent_path = root.join("parent.json");
        atomic(&parent_path, &ModelArtifact::legacy()).unwrap();
        let initial = root.join("initial.json");
        bootstrap(&parent_path, &initial, None).unwrap();
        let parent = Artifact::load(&initial).unwrap();
        let model = parent.model().unwrap();
        let active = out.join("games/active.targets.json.gz");
        let expired = out.join("games/expired.targets.json.gz");
        let old = root.join("previous.targets.json.gz");
        let psr = out.join("games/expired.psr");
        for p in [&active, &expired, &old, &psr] {
            fs::write(p, b"preserved source").unwrap();
        }
        let e = Arc::new(MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: 0,
            state: [0.; 128].to_vec(),
            actions: vec![],
            policy: vec![],
            value: 0.,
            policy_weight: 0.,
        });
        let replay = VecDeque::from([Entry {
            row: Row {
                path: active.clone(),
                sha256: sha256(b"preserved source"),
                index: 0,
            },
            example: e.clone(),
            bytes: size(&e),
        }]);
        let mut retired = HashSet::from([active.clone(), expired.clone(), old.clone()]);
        let mut progress = serde_json::json!({"updates":1});
        checkpoint(
            &out,
            &parent,
            &model,
            1,
            &replay,
            &Lessons::default(),
            &mut progress,
            &mut retired,
        )
        .unwrap();
        assert!(active.exists());
        assert!(!expired.exists());
        assert!(old.exists());
        assert!(psr.exists());
        assert!(retired.is_empty());
        assert!(Path::new(progress["replay_index"].as_str().unwrap()).exists());
        fs::remove_dir_all(root).unwrap();
    }
}
