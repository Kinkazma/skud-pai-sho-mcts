//! Bounded usage evidence. Two recent 8192-query blocks; no lifetime popularity.
use super::*;
use std::sync::atomic::AtomicU64;
pub(super) type Uses = Vec<[AtomicU64; 2]>;
pub(super) fn new_uses(n: usize) -> Uses {
    (0..n)
        .map(|_| [AtomicU64::new(0), AtomicU64::new(0)])
        .collect()
}
#[derive(Serialize, Deserialize)]
pub struct SequenceUsage {
    pub source: u64,
    pub decision: u32,
    pub uses: u32,
}
impl SequenceBank {
    pub(super) fn record_uses(&self, context: &SequenceContext, sequence: usize) {
        let epoch = (sequence / 8192 + 1) as u64;
        for &i in &context.neighbors {
            let cell = &self.uses[i][epoch as usize % 2];
            let _ = cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                let old_epoch = old >> 32;
                if old_epoch > epoch {
                    Some(old)
                } else {
                    let count = if old_epoch == epoch {
                        ((old & 0xffffffff) + 1).min(0xffffffff)
                    } else {
                        1
                    };
                    Some((epoch << 32) | count)
                }
            });
        }
    }
    pub fn recent_usage(&self) -> Vec<SequenceUsage> {
        let epoch = (self.queries.load(Ordering::Relaxed) / 8192 + 1) as u64;
        self.uses
            .iter()
            .zip(&self.entries)
            .filter_map(|(cells, e)| {
                let count: u64 = cells
                    .iter()
                    .map(|c| {
                        let x = c.load(Ordering::Relaxed);
                        let t = x >> 32;
                        if t > 0 && t <= epoch && epoch - t <= 1 {
                            x & 0xffffffff
                        } else {
                            0
                        }
                    })
                    .sum();
                (count > 0).then_some(SequenceUsage {
                    source: e.source,
                    decision: e.decision,
                    uses: count.min(u32::MAX as u64) as u32,
                })
            })
            .collect()
    }
}
