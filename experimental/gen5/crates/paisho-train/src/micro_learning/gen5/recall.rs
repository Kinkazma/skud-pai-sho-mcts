//! Persistent fractional quotas. The denominator is every learning example,
//! with human sampling still measured over the original replay budget.
use super::*;
#[derive(Clone, Default, Serialize, Deserialize)]
pub(super) struct Quotas {
    recall_credit: f64,
    human_credit: f64,
    pub scheduled_recall: usize,
    pub consumed_recall: usize,
    pub consumed_examples: usize,
}
impl Quotas {
    pub fn allocate(
        &mut self,
        fresh: usize,
        ratio: usize,
        recall: f64,
        human: f64,
    ) -> (usize, usize) {
        self.recall_credit += fresh.saturating_mul(ratio + 1) as f64 * recall;
        self.human_credit += fresh.saturating_mul(ratio) as f64 * human;
        let humans = self.human_credit.floor().max(0.0) as usize;
        let durable = (self.recall_credit.floor().max(0.0) as usize)
            .min(fresh.saturating_mul(ratio).saturating_sub(humans));
        self.recall_credit -= durable as f64;
        self.human_credit -= humans as f64;
        self.scheduled_recall += durable;
        (durable, humans)
    }
    pub fn settle(&mut self, skipped: usize, unused_recall: usize, fraction: f64) {
        self.recall_credit += unused_recall as f64 - fraction * skipped as f64;
    }
    pub fn validate(&self) -> Result<()> {
        if !self.recall_credit.is_finite()
            || self.recall_credit.abs() > 1_000_000.0
            || !self.human_credit.is_finite()
            || !(0.0..1.0).contains(&self.human_credit)
        {
            return Err(invalid("invalid persisted recall quota"));
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn singleton_reanalysis_gets_half_all_examples_and_resume_keeps_fraction() {
        let mut q = Quotas::default();
        assert_eq!(q.allocate(1, 4, 0.5, 0.02), (2, 0));
        let mut q: Quotas = serde_json::from_slice(&serde_json::to_vec(&q).unwrap()).unwrap();
        assert_eq!(q.allocate(1, 4, 0.5, 0.02), (3, 0));
        for _ in 0..98 {
            q.allocate(1, 4, 0.5, 0.02);
        }
        assert_eq!(q.scheduled_recall, 250);
        q.validate().unwrap();
    }
    #[test]
    fn quotas_do_not_depend_on_receipt_partition() {
        let mut a = Quotas::default();
        let mut b = Quotas::default();
        let total = a.allocate(500, 4, 0.5, 0.02);
        let parts: Vec<_> = (0..100).map(|_| b.allocate(5, 4, 0.5, 0.02)).collect();
        assert_eq!(total.0, parts.iter().map(|p| p.0).sum::<usize>());
        // Floating fractions may differ by <1 sample, carried into the next receipt.
        assert!((total.1 as isize - parts.iter().map(|p| p.1).sum::<usize>() as isize).abs() <= 1);
    }
    #[test]
    fn partial_receipt_carries_actual_recall_deficit_through_resume() {
        let mut q = Quotas::default();
        assert_eq!(q.allocate(20, 4, 0.5, 0.02).0, 50);
        // Only 64 of 100 examples learned; 20 of the 50 recalls remain unused.
        q.settle(36, 20, 0.5);
        let mut q: Quotas = serde_json::from_slice(&serde_json::to_vec(&q).unwrap()).unwrap();
        assert_eq!(q.allocate(20, 4, 0.5, 0.02).0, 52);
        // Thirty actually learned recalls + fifty-two = half of 64 + 100.
        assert_eq!(30 + 52, (64 + 100) / 2);
        q.validate().unwrap();
    }
}
