//! Per-case causal feedback; no wait on another actor or on the historical lane.
use super::*;
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc, RwLock,
};
pub(super) struct Feedback {
    pub snapshot: Arc<Snapshot>,
    pub fully_learned: bool,
}
fn submit(
    mut game: collector::Played,
    tx: &mpsc::SyncSender<collector::Played>,
    stop: &AtomicBool,
    end: Instant,
) -> Option<Arc<Snapshot>> {
    let (ack, rx) = mpsc::sync_channel(1);
    game.ack = Some(ack);
    tx.send(game).ok()?;
    while !stop.load(Ordering::Relaxed) && Instant::now() < end {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(f) => return f.fully_learned.then_some(f.snapshot),
            Err(mpsc::RecvTimeoutError::Timeout) => (),
            Err(_) => return None,
        }
    }
    None
}
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    actor: usize,
    legacy: Option<Arc<CompactValueModel>>,
    ladder: ladder::Shared,
    mut state: cases::State,
    cases: Arc<Vec<cases::Case>>,
    shared: Arc<RwLock<Arc<Snapshot>>>,
    proofs: durable::Proofs,
    next: Arc<AtomicUsize>,
    count: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    tx: mpsc::SyncSender<collector::Played>,
    pool: cpu::Executor,
    o: Options,
    source: String,
    end: Instant,
) {
    let config = o.case_curriculum.as_ref().unwrap();

    while !stop.load(Ordering::Relaxed) && Instant::now() < end {
        if state.rotate_reason.is_some() && state.pending.is_empty() {
            state.advance(o.actors);
        }
        let current_budget = ladder.read().unwrap().budget();
        if legacy.is_some() {
            if state.legacy_budget.is_some_and(|b| b != current_budget) {
                state.advance(o.actors);
            }
            state.legacy_budget = Some(current_budget);
        }
        let case = &cases[state.ticket % cases.len()];
        let id = next.fetch_add(1, Ordering::Relaxed);
        if id >= o.games {
            break;
        }
        let snapshot = shared.read().unwrap().clone();
        let mut after = state.clone();
        let mut options = o.clone();
        if legacy.is_some() {
            options.candidate_seat = Some(match state.focus.as_deref() {
                Some("H") => Player::Host,
                Some("G") => Player::Guest,
                _ if state.ticket % 2 == 0 => Player::Host,
                _ => Player::Guest,
            });
        }
        let pending = if let Some(index) = state.pending.first() {
            let result = (|| -> Result<GameRecord> {
                let (path, expected) = state
                    .pending_record
                    .as_ref()
                    .ok_or_else(|| invalid("reanalysis source missing"))?;
                let bytes = fs::read(path)?;
                if sha256(&bytes) != *expected {
                    return Err(invalid("pending reanalysis source changed"));
                }
                let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
                if *index > record.actions().len() {
                    return Err(invalid("pending reanalysis index missing"));
                }
                Ok(cases::prefix(&record, *index))
            })();
            match result {
                Ok(record) => Some(record),
                Err(e) => {
                    let mut game = collector::play_from(
                        id,
                        snapshot.clone(),
                        snapshot,
                        None,
                        &options,
                        Instant::now(),
                        &pool,
                        &source,
                        Some(&case.record),
                        false,
                        Some(&proofs),
                    );
                    game.error = Some(e.to_string());
                    let _ = tx.send(game);
                    return;
                }
            }
        } else {
            None
        };
        let reanalysis = pending.is_some();
        if reanalysis {
            options.budgets = vec![(config.reanalysis_budget, 1.0)];
            after.pending.remove(0);
        } else {
            count.fetch_add(1, Ordering::Relaxed);
        }
        let mut game = collector::play_from(
            id,
            snapshot.clone(),
            snapshot,
            if reanalysis {
                None
            } else {
                legacy.as_ref().map(|m| (m.as_ref(), current_budget))
            },
            &options,
            end,
            &pool,
            &source,
            Some(pending.as_ref().unwrap_or(&case.record)),
            reanalysis,
            Some(&proofs),
        );
        if !reanalysis {
            let hash = sha256(game.record.to_string().as_bytes());
            after.observe(game.outcome, &hash, config);
            after.pending = collector::correction_prefixes(&game, config.reanalysis_positions)
                .iter()
                .map(|p| p.actions().len())
                .collect();
            after.pending_record = Some((
                Path::new(&source)
                    .join("games")
                    .join(format!("game-{id:07}.psr")),
                hash,
            ));
        }
        game.case = Some(cases::Attempt {
            actor,
            case: case.identity.clone(),
            human_source: case.source.clone(),
            zone: case.zone,
            prefix_decisions: case.record.actions().len(),
            before: state.clone(),
            after: after.clone(),
            kind: if reanalysis {
                "reanalysis"
            } else {
                "continuation"
            }
            .into(),
        });
        if submit(game, &tx, &stop, end).is_none() {
            break;
        }
        state = after;
    }
}
