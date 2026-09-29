//! Eight main workers and a borrowable secondary pool. Historical work claims
//! the secondary pool between searches; Gen5 falls back to main without waiting.
use super::*;
mod ordered;
pub(super) use ordered::Ordered;
use paisho_platform::{set_current_thread_qos, ThreadQos};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    mpsc, RwLock, RwLockWriteGuard,
};
#[derive(Serialize)]
pub(super) struct QosReceipt {
    worker: usize,
    requested: Option<u32>,
    observed: Option<u32>,
}
pub(super) fn build_pool(
    threads: usize,
    qos: Option<ThreadQos>,
) -> Result<(Arc<rayon::ThreadPool>, Vec<QosReceipt>)> {
    let (tx, rx) = mpsc::channel();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .start_handler(move |worker| {
            let result = qos
                .map(set_current_thread_qos)
                .transpose()
                .map(Option::flatten)
                .map_err(|e| e.to_string());
            let _ = tx.send((worker, result));
        })
        .build()?;
    let mut receipts = Vec::new();
    for _ in 0..threads {
        let (worker, observed) = rx.recv()?;
        receipts.push(QosReceipt {
            worker,
            requested: qos.map(ThreadQos::code),
            observed: observed.map_err(invalid)?,
        });
    }
    receipts.sort_by_key(|r| r.worker);
    Ok((Arc::new(pool), receipts))
}
pub(super) fn set_qos(enabled: bool, background: bool) -> Result<()> {
    if enabled {
        set_current_thread_qos(if background {
            ThreadQos::Utility
        } else {
            ThreadQos::UserInitiated
        })?;
    }
    Ok(())
}
pub(super) fn build_search_pools(threads:usize,shards:usize,qos:Option<ThreadQos>)->Result<Vec<Arc<rayon::ThreadPool>>> {
    if shards==0 || threads==0 || threads%shards!=0 {return Err(invalid("invalid search pool partition"));}
    (0..shards).map(|_|build_pool(threads/shards,qos).map(|(pool,_)|pool)).collect()
}
pub(super) struct Spare {
    pub pool: Arc<rayon::ThreadPool>,
    reserved: AtomicBool,
    gate: RwLock<()>,
    borrowed: AtomicUsize,
    fallback: AtomicUsize,
    reservations: AtomicUsize,
    drain_ns: AtomicU64,
    reserved_ns: AtomicU64,
    active_since: std::sync::Mutex<Option<Instant>>,
    started: Instant,
}
impl Spare {
    pub fn new(pool: Arc<rayon::ThreadPool>) -> Self {
        Self {
            pool,
            reserved: AtomicBool::new(false),
            gate: RwLock::new(()),
            borrowed: AtomicUsize::new(0),
            fallback: AtomicUsize::new(0),
            reservations: AtomicUsize::new(0),
            drain_ns: AtomicU64::new(0),
            reserved_ns: AtomicU64::new(0),
            active_since: std::sync::Mutex::new(None),
            started: paisho_platform::training_time::now(),
        }
    }
    pub fn reserve(&self) -> Reservation<'_> {
        self.reserved.store(true, Ordering::SeqCst);
        let t = paisho_platform::training_time::now();
        *self.active_since.lock().unwrap() = Some(t);
        let guard = self.gate.write().unwrap();
        self.drain_ns.fetch_add(
            paisho_platform::training_time::elapsed(t).as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        self.reservations.fetch_add(1, Ordering::Relaxed);
        Reservation {
            spare: self,
            _guard: guard,
        }
    }
    pub fn reserved_seconds(&self) -> f64 {
        let active = self.active_since.lock().unwrap();
        self.reserved_ns.load(Ordering::Relaxed) as f64 / 1e9
            + active.map_or(0.0, |t| paisho_platform::training_time::elapsed(t).as_secs_f64())
    }
    pub fn can_admit(&self, seconds: f64, total_threads: usize, fraction: f64) -> bool {
        allowance(
            paisho_platform::training_time::elapsed(self.started).as_secs_f64(),
            self.reserved_seconds(),
            seconds,
            total_threads,
            self.pool.current_num_threads(),
            fraction,
        )
    }
    pub fn telemetry(&self) -> serde_json::Value {
        serde_json::json!({"borrowed_gen5_jobs":self.borrowed.load(Ordering::Relaxed),"gen5_fallback_jobs":self.fallback.load(Ordering::Relaxed),"historical_reservations":self.reservations.load(Ordering::Relaxed),"reservation_drain_seconds":self.drain_ns.load(Ordering::Relaxed) as f64/1e9,"physical_core_affinity":false,"reserved_seconds":self.reserved_seconds(),"elapsed_seconds":paisho_platform::training_time::elapsed(self.started).as_secs_f64()})
    }
}
pub(super) struct Reservation<'a> {
    spare: &'a Spare,
    _guard: RwLockWriteGuard<'a, ()>,
}
impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        if let Some(t) = self.spare.active_since.lock().unwrap().take() {
            self.spare.reserved_ns.fetch_add(
                paisho_platform::training_time::elapsed(t).as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
        }
        self.spare.reserved.store(false, Ordering::SeqCst);
    }
}
#[derive(Clone)]
pub(super) struct Executor {
    main: Arc<rayon::ThreadPool>,
    spare: Option<Arc<Spare>>,
    budgeted: Option<(Arc<Spare>, usize, f64, Instant)>,
}
impl Executor {
    pub fn direct(pool: Arc<rayon::ThreadPool>) -> Self {
        Self {
            main: pool,
            spare: None,
            budgeted: None,
        }
    }
    pub fn adaptive(main: Arc<rayon::ThreadPool>, spare: Arc<Spare>) -> Self {
        Self {
            main,
            spare: Some(spare),
            budgeted: None,
        }
    }
    /// Keep the game/tree alive while returning the secondary pool between jobs.
    /// Admission may overshoot by one indivisible job; actual cost is charged and
    /// the next job waits for credit. Waiting holds no pool reservation.
    pub fn budgeted(spare: Arc<Spare>, threads: usize, fraction: f64, end: Instant) -> Self {
        Self {
            main: spare.pool.clone(),
            spare: None,
            budgeted: Some((spare, threads, fraction, end)),
        }
    }
    pub fn install<F: FnOnce() -> R + Send, R: Send>(&self, f: F) -> R {
        if let Some((spare, threads, fraction, end)) = &self.budgeted {
            while paisho_platform::training_time::now() < *end && !spare.can_admit(0.02, *threads, *fraction) {
                std::thread::sleep(Duration::from_millis(5));
            }
            let _reservation = spare.reserve();
            return spare.pool.install(f);
        }
        if let Some(spare) = &self.spare {
            if !spare.reserved.load(Ordering::SeqCst) {
                if let Ok(_guard) = spare.gate.try_read() {
                    if !spare.reserved.load(Ordering::SeqCst) {
                        spare.borrowed.fetch_add(1, Ordering::Relaxed);
                        return spare.pool.install(f);
                    }
                }
            }
            spare.fallback.fetch_add(1, Ordering::Relaxed);
        }
        self.main.install(f)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_keeps_the_total_cpu_budget_and_rejects_partial_shards() {
        let pools=build_search_pools(4,4,None).unwrap();
        assert_eq!(pools.iter().map(|p|p.current_num_threads()).sum::<usize>(),4);
        for p in pools {assert_eq!(p.install(rayon::current_num_threads),1);}
        assert!(build_search_pools(4,3,None).is_err());
        assert!(build_search_pools(4,0,None).is_err());
    }
    #[test]
    fn budgeted_jobs_release_the_pool_between_calls() {
        let (main, _) = build_pool(1, None).unwrap();
        let (other, _) = build_pool(1, None).unwrap();
        let spare = Arc::new(Spare::new(other));
        let borrowed = Executor::adaptive(main, spare.clone());
        let historical = Executor::budgeted(
            spare.clone(),
            10,
            0.075,
            paisho_platform::training_time::now() + Duration::from_secs(2),
        );
        historical.install(|| assert!(spare.reserved.load(Ordering::SeqCst)));
        assert!(!spare.reserved.load(Ordering::SeqCst));
        assert_eq!(borrowed.install(|| 7), 7);
        assert_eq!(spare.telemetry()["historical_reservations"], 1);
        assert_eq!(spare.telemetry()["borrowed_gen5_jobs"], 1);
    }

    #[test]
    fn historical_priority_falls_back_without_waiting_and_releases_idle_capacity() {
        let (main, _) = build_pool(1, None).unwrap();
        let (other, _) = build_pool(2, None).unwrap();
        let spare = Arc::new(Spare::new(other));
        let executor = Executor::adaptive(main, spare.clone());
        assert_eq!(executor.install(rayon::current_num_threads), 2);
        {
            let _reservation = spare.reserve();
            assert_eq!(executor.install(rayon::current_num_threads), 1);
        }
        assert_eq!(executor.install(rayon::current_num_threads), 2);
        assert_eq!(spare.telemetry()["borrowed_gen5_jobs"], 2);
        assert_eq!(spare.telemetry()["gen5_fallback_jobs"], 1);
    }
    #[test]
    fn qos_receipts_cover_every_worker() {
        let (_, receipts) = build_pool(2, Some(ThreadQos::Background)).unwrap();
        assert_eq!(receipts.len(), 2);
        #[cfg(target_os = "macos")]
        assert!(receipts
            .iter()
            .all(|r| r.observed == Some(ThreadQos::Background.code())));
    }
}

// Admission reserves the complete game/evaluation cap before starting. Idle
// historical capacity is borrowed by Gen5; no partial game is started for quota.
fn allowance(
    elapsed: f64,
    spent: f64,
    next: f64,
    total: usize,
    secondary: usize,
    fraction: f64,
) -> bool {
    fraction > 0.0 && spent + next + 0.05 <= elapsed * total as f64 * fraction / secondary as f64
}
#[cfg(test)]
mod allowance_tests {
    #[test]
    fn caps_collection_and_evaluation_together_without_spending_future_capacity() {
        assert!(!super::allowance(0., 0., 8., 10, 2, 0.075));
        assert!(super::allowance(100., 20., 8., 10, 2, 0.075));
        assert!(!super::allowance(100., 30., 8., 10, 2, 0.075));
        assert!(!super::allowance(1000., 200., 180., 10, 2, 0.075));
        assert!(!super::allowance(1000., 0., 1., 10, 2, 0.0));
    }
}
