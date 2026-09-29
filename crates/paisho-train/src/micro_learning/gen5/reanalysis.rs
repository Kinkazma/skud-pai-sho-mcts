//! Bounded old-lesson reanalysis queue; collectors never wait for this lane.
use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, RwLock,
};
pub(super) struct Request {
    pub prefix: GameRecord,
    pub origin: cases::Attempt,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn spawn(
    o: Options,
    requests: mpsc::Receiver<Request>,
    output: mpsc::SyncSender<collector::Played>,
    current: Arc<RwLock<Arc<Snapshot>>>,
    next: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    pool: cpu::Executor,
    proofs: durable::Proofs,
    source: String,
    end: Instant,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        while !stop.load(Ordering::Relaxed)
            && Instant::now() < end
            && next.load(Ordering::Relaxed) < o.games
        {
            let task = match requests.recv_timeout(Duration::from_millis(50)) {
                Ok(task) => task,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(_) => break,
            };
            let id = next.fetch_add(1, Ordering::Relaxed);
            if id >= o.games {
                break;
            }
            let snapshot = current.read().unwrap().clone();
            let mut options = o.clone();
            options.budgets = vec![(o.case_curriculum.as_ref().unwrap().reanalysis_budget, 1.0)];
            let mut game = collector::play_from(
                id,
                snapshot.clone(),
                snapshot,
                None,
                &options,
                end,
                &pool,
                &source,
                Some(&task.prefix),
                true,
                Some(&proofs),
            );
            game.case = Some(cases::Attempt {
                kind: "archive-reanalysis".into(),
                ..task.origin
            });
            if output.send(game).is_err() {
                break;
            }
        }
    })
}
