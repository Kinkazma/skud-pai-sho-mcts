//! Archive one finished game while the learner consumes the preceding game.
//! PSR/targets are durable before Ready is sent; model publication stays causal.
use super::*;
use collector::{targets, Played};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, RwLock,
};
pub(super) struct Ready {
    pub durable_seconds: f64,
    pub durable_path: Option<PathBuf>,
    pub game: Played,
    pub owned: Vec<Arc<MicroExample>>,
    pub corrections: Vec<bool>,
    pub target_path: PathBuf,
    pub target_hash: String,
    pub psr_hash: String,
    pub archive_seconds: f64,
}
pub(super) fn spawn(
    main: mpsc::Receiver<Played>,
    history: mpsc::Receiver<Played>,
    out: PathBuf,
    capacity: usize,
    workers: usize,
    durable_dir: Option<PathBuf>,
    buffered: bool,
    proofs: durable::Proofs,
    stop: Arc<AtomicBool>,
    error: Arc<RwLock<Option<String>>>,
) -> (mpsc::Receiver<Ready>, std::thread::JoinHandle<()>) {
    let (tx, rx) = mpsc::sync_channel(capacity);
    let receivers = Arc::new(std::sync::Mutex::new((main, history)));
    let persistence = Arc::new(std::sync::Mutex::new(()));
    let worker = std::thread::spawn(move || {
        let mut handles = vec![];
        for _ in 0..workers {
            let (receivers, persistence, tx, out, durable_dir, proofs, stop, error) = (
                receivers.clone(),
                persistence.clone(),
                tx.clone(),
                out.clone(),
                durable_dir.clone(),
                proofs.clone(),
                stop.clone(),
                error.clone(),
            );
            handles.push(std::thread::spawn(move || {
                let result = (|| -> Result<()> {
                    let mut closed = [false; 2];
                    let mut turn = false;
                    while !closed.iter().all(|v| *v) {
                        let mut received = None;
                        turn = !turn;
                        {
                            let receivers = receivers.lock().unwrap();
                            for lane in [usize::from(turn), usize::from(!turn)] {
                                if closed[lane] {
                                    continue;
                                }
                                let rx = if lane == 0 {
                                    &receivers.0
                                } else {
                                    &receivers.1
                                };
                                match rx.try_recv() {
                                    Ok(game) => {
                                        received = Some(game);
                                        break;
                                    }
                                    Err(mpsc::TryRecvError::Disconnected) => closed[lane] = true,
                                    Err(mpsc::TryRecvError::Empty) => {}
                                }
                            }
                        }
                        let Some(mut game) = received else {
                            std::thread::sleep(Duration::from_millis(1));
                            continue;
                        };
                        if let Some(e) = &game.error {
                            *error.write().unwrap() = Some(e.clone());
                            stop.store(true, Ordering::Relaxed);
                        }
                        let t = Instant::now();
                        let fresh = targets(&mut game);
                        let stem = format!("game-{:07}", game.id);
                        let dir = out.join("games");
                        let psr = game.record.to_string();
                        use std::io::Write;
                        let mut f = fs::OpenOptions::new()
                            .create_new(true)
                            .write(true)
                            .open(dir.join(format!("{stem}.psr")))?;
                        f.write_all(psr.as_bytes())?;
                        if durable_dir.is_some() {
                            paisho_platform::sync_before_batch_commit(&f)?;
                        } else {
                            f.sync_all()?;
                        }
                        let target_path = dir.join(format!("{stem}.targets.json.gz"));
                        save_examples_batch(&target_path, &fresh, durable_dir.is_some())?;
                        if durable_dir.is_some() && !buffered {
                            fs::File::open(&dir)?.sync_all()?;
                        }
                        let target_hash = sha256(&fs::read(&target_path)?);
                        let psr_hash = sha256(psr.as_bytes());
                        let owned = fresh
                            .iter()
                            .map(|s| s.example_for_rules(RULES).map(Arc::new))
                            .collect::<Result<Vec<_>>>()?;
                        let saving = Instant::now();
                        let durable_path = match &durable_dir {
                            Some(dir) => {
                                // Revision replacement and proof registration remain ordered.
                                let _guard = persistence.lock().unwrap();
                                durable::persist_mode(dir, &game, &fresh, &proofs, buffered)?
                            }
                            None => None,
                        };
                        let ready = Ready {
                            durable_seconds: saving.elapsed().as_secs_f64(),
                            durable_path,
                            corrections: fresh.iter().map(|s| s.correction_priority).collect(),
                            game,
                            owned,
                            target_path,
                            target_hash,
                            psr_hash,
                            archive_seconds: t.elapsed().as_secs_f64(),
                        };
                        if tx.send(ready).is_err() {
                            break;
                        }
                    }
                    Ok(())
                })();
                if let Err(e) = result {
                    *error.write().unwrap() = Some(e.to_string());
                    stop.store(true, Ordering::Relaxed);
                }
            }));
        }
        drop(tx);
        for handle in handles {
            if handle.join().is_err() {
                *error.write().unwrap() = Some("archive writer panic".into());
                stop.store(true, Ordering::Relaxed);
            }
        }
    });
    (rx, worker)
}
