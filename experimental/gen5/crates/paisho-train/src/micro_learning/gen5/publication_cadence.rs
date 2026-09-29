//! Publication work clock. Fresh means a consumed row of a target artifact,
//! distinct within this block; replay/human presentations never count as fresh.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkBudget {
    pub presentations: usize,
    pub fresh_examples: usize,
}
impl WorkBudget {
    pub(super) fn validate(&self) -> Result<()> {
        if self.fresh_examples == 0 || self.fresh_examples > 1_000_000
            || self.presentations < self.fresh_examples || self.presentations > 100_000_000 {
            return Err(invalid("invalid publication work budget"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WorkClock {
    pub presentations: usize,
    pub fresh_examples: usize,
    // Compact bitsets preserve row identity through partial receipts and resume.
    fresh_rows: BTreeMap<String, Vec<u64>>,
    pub completed_checks: usize,
    pub presentations_since_actor_change: usize,
    pub last_presentations: usize,
    pub last_fresh_examples: usize,
}
impl WorkClock {
    pub fn validate(&self) -> Result<()> {
        if self.fresh_rows.keys().any(|s| s.len()!=64 || !s.bytes().all(|b|b.is_ascii_hexdigit()))
            || self.fresh_rows.values().flatten().map(|v|v.count_ones() as usize).sum::<usize>() != self.fresh_examples
            || self.fresh_examples > self.presentations
            || self.presentations > self.presentations_since_actor_change {
            return Err(invalid("inconsistent publication work clock"));
        }
        Ok(())
    }
    pub fn consumed(&mut self, n: usize, source: &str, fresh: impl Iterator<Item=usize>) {
        self.presentations += n;
        self.presentations_since_actor_change += n;
        for index in fresh {
            let words=self.fresh_rows.entry(source.into()).or_default();
            let word=index/64;
            if words.len()<=word { words.resize(word+1,0); }
            let bit=1u64<<(index%64);
            if words[word]&bit==0 { words[word]|=bit; self.fresh_examples+=1; }
        }
    }
    pub fn due(&self, budget: &WorkBudget) -> bool {
        self.presentations>=budget.presentations && self.fresh_examples>=budget.fresh_examples
    }
    pub fn completed(&mut self, actor_changed: bool) {
        self.completed_checks+=1;
        self.last_presentations=self.presentations;
        self.last_fresh_examples=self.fresh_examples;
        self.presentations=0;
        self.fresh_examples=0;
        self.fresh_rows.clear();
        if actor_changed { self.presentations_since_actor_change=0; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn source()->String { "ab".repeat(32) }
    #[test]
    fn both_thresholds_require_consumed_work_not_wall_time_or_replay() {
        let mut c=WorkClock::default();let b=WorkBudget{presentations:10,fresh_examples:3};
        c.consumed(10,&source(),std::iter::empty());assert!(!c.due(&b));
        c.consumed(3,&source(),[0,1,2].into_iter());assert!(c.due(&b));c.validate().unwrap();
    }
    #[test]
    fn duplicate_rows_partial_receipts_and_resume_preserve_clock_exactly() {
        let mut a=WorkClock::default();a.consumed(6,&source(),[0,1,64,1].into_iter());
        assert_eq!(a.fresh_examples,3);
        let mut b:WorkClock=serde_json::from_slice(&serde_json::to_vec(&a).unwrap()).unwrap();
        for c in [&mut a,&mut b] { c.consumed(4,&source(),[0,65,66].into_iter());c.validate().unwrap(); }
        assert_eq!(a,b);assert_eq!(a.fresh_examples,5);
    }
    #[test]
    fn rejection_starts_new_block_but_keeps_actor_lag() {
        let mut c=WorkClock::default();c.consumed(10,&source(),[0].into_iter());c.completed(false);
        assert_eq!(c.presentations,0);assert_eq!(c.fresh_examples,0);assert_eq!(c.presentations_since_actor_change,10);
        c.consumed(5,&source(),[0].into_iter());c.completed(true);assert_eq!(c.presentations_since_actor_change,0);c.validate().unwrap();
    }
    #[test]
    fn thresholds_and_corrupted_resume_are_rejected() {
        assert!(WorkBudget{presentations:5,fresh_examples:6}.validate().is_err());
        assert!(WorkBudget{presentations:0,fresh_examples:0}.validate().is_err());
        let c=WorkClock{fresh_examples:1,..Default::default()};assert!(c.validate().is_err());
    }
}
