//! Independent frozen reads share the existing search workers and caller.
//! Ordered results do not wait for unused dispatches behind busy searches.
use std::sync::Arc;

#[derive(Clone)]
pub(in crate::micro_learning::gen5) struct Ordered {
    pools: Vec<Arc<rayon::ThreadPool>>,
}
impl Ordered {
    pub fn new(pools: &[Arc<rayon::ThreadPool>]) -> Self {
        assert!(!pools.is_empty());
        Self {
            pools: pools.to_vec(),
        }
    }
    /// Owned frozen inputs let idle shards finish the work without waiting for
    /// an unused dispatch queued behind a busy search. Pending dispatches retain
    /// only the empty ticket after completion, not the model or replay examples.
    pub fn map_owned<T, R, F, C>(&self, rows: Vec<T>, cost: C, f: F) -> Vec<R>
    where
        T: Send + Sync + 'static,
        R: Send + 'static,
        F: Fn(&T) -> R + Send + Sync + 'static,
        C: Fn(&T) -> usize,
    {
        let out = Vec::with_capacity(rows.len());
        self.fold_owned(rows, cost, f, out, |out, value| out.push(value))
    }
    /// Consume completed results in original input order while other workers
    /// continue. The caller helps compute whenever no ordered result is ready.
    pub fn fold_owned<T, R, F, C, U, A>(
        &self,
        rows: Vec<T>,
        cost: C,
        f: F,
        mut out: U,
        mut add: A,
    ) -> U
    where
        T: Send + Sync + 'static,
        R: Send + 'static,
        F: Fn(&T) -> R + Send + Sync + 'static,
        C: Fn(&T) -> usize,
        A: FnMut(&mut U, R),
    {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc, Mutex,
        };
        if rows.len() < 2 {
            for row in &rows {
                add(&mut out, f(row));
            }
            return out;
        }
        let n = rows.len();
        let costs = rows.iter().map(cost).collect::<Vec<_>>();
        let mut order = (0..n).collect::<Vec<_>>();
        order.sort_by(|&a, &b| costs[b].cmp(&costs[a]).then_with(|| a.cmp(&b)));
        let work = Arc::new(Mutex::new(Some(Arc::new((
            rows,
            order,
            f,
            AtomicUsize::new(0),
        )))));
        let (tx, rx) = mpsc::channel();
        for pool in &self.pools {
            for _ in 0..pool.current_num_threads() {
                let ticket = work.clone();
                let tx = tx.clone();
                pool.spawn_fifo(move || {
                    let active = ticket.lock().unwrap().clone();
                    let Some(active) = active else {
                        return;
                    };
                    loop {
                        let at = active.3.fetch_add(1, Ordering::Relaxed);
                        if at >= active.0.len() {
                            break;
                        }
                        let i = active.1[at];
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            (active.2)(&active.0[i])
                        }));
                        if tx.send((i, result)).is_err() {
                            break;
                        }
                    }
                });
            }
        }
        let mut result: Vec<_> = (0..n).map(|_| None).collect();
        let mut received = 0;
        let mut next = 0;
        let mut failure = None;
        let active = work.lock().unwrap().as_ref().unwrap().clone();
        loop {
            while let Ok((i, value)) = rx.try_recv() {
                result[i] = Some(value);
                received += 1;
            }
            while next < n && result[next].is_some() {
                let value = result[next].take().unwrap();
                next += 1;
                if failure.is_none() {
                    failure = match value {
                        Ok(value) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            add(&mut out, value)
                        }))
                        .err(),
                        Err(panic) => Some(panic),
                    };
                }
            }
            if received == n {
                break;
            }
            // All computing uses existing threads. In particular, a fully busy
            // search pool cannot prevent this caller from finishing its reads.
            let at = active.3.fetch_add(1, Ordering::Relaxed);
            if at < active.0.len() {
                let i = active.1[at];
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (active.2)(&active.0[i])
                }));
                tx.send((i, result))
                    .expect("owned read receiver disappeared");
            } else {
                let (i, value) = rx.recv().expect("owned read workers disappeared");
                result[i] = Some(value);
                received += 1;
            }
        }
        drop(active);
        drop(tx);
        // All actual computations have returned. A queued worker must not keep
        // an old replay batch resident while it waits for its search to finish.
        work.lock().unwrap().take();
        if let Some(panic) = failure {
            std::panic::resume_unwind(panic);
        }
        out
    }
}

#[cfg(test)]
mod owned_tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;
    #[test]
    fn streamed_fold_keeps_float_order_and_recovers_from_callback_panic() {
        let p = Ordered::new(&[Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        )]);
        let rows = (0..64).collect::<Vec<_>>();
        let values = [1e16, 1., -1e16, -0., 0.125, -0.5];
        let expected = rows.iter().fold(-0.0, |s, i| s + values[i % values.len()]);
        // The reducer stays on the caller; its state need not be Send.
        let state = std::rc::Rc::new(std::cell::Cell::new(-0.0f64));
        let value = p.fold_owned(
            rows,
            |i| 63 - i,
            move |i| values[i % values.len()],
            state,
            |s, v| s.set(s.get() + v),
        );
        assert_eq!(value.get().to_bits(), expected.to_bits());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.fold_owned(
                vec![0, 1, 2, 3],
                |i| *i,
                |i| *i,
                (),
                |_, i| assert_ne!(i, 1)
            )))
            .is_err()
        );
        assert_eq!(p.map_owned(vec![1, 2, 3], |i| *i, |i| i + 1), vec![2, 3, 4]);
    }
    #[test]
    fn caller_can_finish_while_every_pool_is_occupied() {
        let pools = (0..2)
            .map(|_| {
                Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let mut releases = vec![];
        for p in &pools {
            let (tx, rx) = mpsc::channel();
            let (ready, waiting) = mpsc::channel();
            p.spawn(move || {
                ready.send(()).unwrap();
                rx.recv().unwrap();
            });
            waiting.recv().unwrap();
            releases.push(tx);
        }
        let p = Ordered::new(&pools);
        let (done, result) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            done.send(p.map_owned((0..64).collect(), |n| *n, |n| n * 3))
                .unwrap()
        });
        let received = result.recv_timeout(Duration::from_secs(5));
        for tx in releases {
            tx.send(()).unwrap();
        }
        thread.join().unwrap();
        assert_eq!(
            received.expect("caller waited for occupied pools"),
            (0..64).map(|n| n * 3).collect::<Vec<_>>()
        );
    }
    #[test]
    fn completed_reads_do_not_wait_for_unused_busy_shard() {
        let pools = (0..2)
            .map(|_| {
                Arc::new(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let (ready_tx, ready_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        pools[0].spawn(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
        });
        ready_rx.recv().unwrap();
        let marker = Arc::new(91usize);
        let weak = Arc::downgrade(&marker);
        let ordered = Ordered::new(&pools);
        let (done_tx, done_rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let values = ordered.map_owned(
                (0..64).map(|i| (i, marker.clone())).collect(),
                |r| r.0 % 9,
                |r| r.0 * r.0 + *r.1,
            );
            drop(marker);
            done_tx.send(values).unwrap();
        });
        let received = done_rx.recv_timeout(Duration::from_secs(5));
        let released_batch = received.is_ok() && pools[1].install(|| weak.upgrade().is_none());
        // Always release the blocked search, including on failure, before panic.
        release_tx.send(()).unwrap();
        handle.join().unwrap();
        assert_eq!(
            received.expect("unused shard blocked a completed lot"),
            (0..64).map(|i| i * i + 91).collect::<Vec<_>>()
        );
        assert!(released_batch, "completed batch retained by queued ticket");
    }
    #[test]
    fn panicking_read_returns_panic_without_poisoning_other_results() {
        let p = Ordered::new(&[Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .build()
                .unwrap(),
        )]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.map_owned(
                vec![1, 2, 3, 4],
                |_| 1,
                |n| {
                    assert_ne!(*n, 2);
                    *n
                }
            )))
            .is_err()
        );
        assert_eq!(
            p.map_owned(vec![4, 3, 2, 1], |n| *n, |n| n + 1),
            vec![5, 4, 3, 2]
        );
    }
}
