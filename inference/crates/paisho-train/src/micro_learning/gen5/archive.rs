//! Archive one finished game while the learner consumes the preceding game.
//! PSR/targets are durable before Ready is sent; model publication stays causal.
use super::*;
use collector::{targets, Played};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, RwLock,
};
pub(super) struct Ready {
    pub evidence_counts: serde_json::Value,
    pub registered_proofs: Vec<PathBuf>,
    pub durable_seconds: f64,
    pub durable_path: Option<PathBuf>,
    pub game: Played,
    pub owned: Vec<Arc<MicroExample>>,
    pub lessons: Vec<(String,String,Arc<MicroExample>,bool)>,
    /// Native decision provenance retained only for publication-transfer proofs.
    /// Admission remains the learner's responsibility after all fresh SGD.
    pub proof_targets: Vec<SavedMicroExample>,
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
    trusted_action_values: bool,
    publication_transfer: bool,
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
                        let t = paisho_platform::training_time::now();
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
                        let owned: Vec<Arc<MicroExample>> = fresh
                            .iter()
                            .map(|s| s.example_for_rules_with_trusted_q(RULES,trusted_action_values).map(Arc::new))
                            .collect::<Result<Vec<_>>>()?;
                        let saving = paisho_platform::training_time::now();
                        let durable_path = match &durable_dir {
                            Some(dir) => {
                                // Revision replacement and proof registration remain ordered.
                                let _guard = persistence.lock().unwrap();
                                durable::persist_mode(dir, &game, &fresh, &proofs, buffered)?
                            }
                            None => None,
                        };
                        let lessons = if game.snapshot.model.has_spatial() { fresh.iter().zip(&owned).filter(|(s,_)|s.correction_priority || game.reanalysis || game.loop_repair && s.tactical.as_ref().is_some_and(|t|t.root_value.is_some())).map(|(s,ex)| {
                            let key=sha256(cases::prefix(&game.record,s.decision-1).to_string().as_bytes());
                            let group=game.case.as_ref().map_or_else(||psr_hash.clone(),|c|c.human_source.clone());
                            (key,group,ex.clone(),s.tactical.as_ref().is_some_and(|t|t.root_value.is_some()))
                        }).collect() }else{vec![]};
                        let registered_proofs=if game.loop_repair && durable_path.is_some() {
                            durable_dir.as_ref().map_or_else(Vec::new,|dir|game.certificates.iter().map(|(decision,_)| {
                                let key=sha256(cases::prefix(&game.record,decision-1).to_string().as_bytes());
                                dir.join("proofs").join(format!("{key}.json"))
                            }).collect())
                        } else {vec![]};
                        let mut by_teacher=std::collections::BTreeMap::<String,usize>::new();
                        let mut by_player=std::collections::BTreeMap::<String,usize>::new();
                        for s in &fresh {if let Some(e)=&s.evidence{*by_teacher.entry(e.policy_source.clone()).or_default()+=1;*by_player.entry(e.player.clone()).or_default()+=1;}}
                        let evidence_counts=serde_json::json!({"teachers":by_teacher,"players":by_player,"observed":fresh.iter().filter(|s|s.evidence.as_ref().is_some_and(|e|e.observed_value.is_some())).count(),"policy_only":fresh.iter().filter(|s|s.evidence.as_ref().is_some_and(|e|e.value_weight==0.)).count(),"value_only":fresh.iter().filter(|s|s.policy_weight==0.).count()});
                        let proof_targets = transfer_targets(publication_transfer, &game.certificates, &fresh);
                        let ready = Ready {
                            evidence_counts,
                            registered_proofs,
                            lessons,
                            proof_targets,
                            durable_seconds: paisho_platform::training_time::elapsed(saving).as_secs_f64(),
                            durable_path,
                            corrections: fresh.iter().map(|s| s.correction_priority).collect(),
                            game,
                            owned,
                            target_path,
                            target_hash,
                            psr_hash,
                            archive_seconds: paisho_platform::training_time::elapsed(t).as_secs_f64(),
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

fn transfer_targets(enabled: bool, certificates: &[(usize, MicroProofCertificate)],
    fresh: &[SavedMicroExample]) -> Vec<SavedMicroExample>
{
    if !enabled { return vec![]; }
    fresh.iter().filter(|s| s.tactical.as_ref().is_some_and(|t| t.root_value == Some(1))
        && certificates.iter().any(|(decision, _)| *decision == s.decision))
        .cloned().collect()
}

#[cfg(test)]
mod transfer_target_tests {
    use super::*;
    #[test]
    fn publication_transfer_keeps_only_actual_winning_proof_targets_when_enabled() {
        let mut win = super::super::super::tactics::fixture();
        win.decision = 1;
        let mut loss = win.clone(); loss.decision = 2;
        loss.tactical.as_mut().unwrap().root_value = Some(-1);
        let mut missing = win.clone(); missing.decision = 3;
        let fresh = vec![win.clone(), loss, missing];
        let certificates = vec![(1, MicroProofCertificate { outcome: 1, children: vec![] }),
            (2, MicroProofCertificate { outcome: -1, children: vec![] })];
        assert!(transfer_targets(false, &certificates, &fresh).is_empty());
        let selected = transfer_targets(true, &certificates, &fresh);
        assert_eq!(selected.len(), 1);
        assert_eq!(serde_json::to_value(&selected[0]).unwrap(), serde_json::to_value(&win).unwrap());
    }
}
