//! Class-balanced verified value recall inside the existing ordinary recall quota.
use super::*;
use std::collections::VecDeque;
#[derive(Default)]
pub(super) struct Balanced {
    rows: [VecDeque<(String, Arc<MicroExample>)>; 3],
    cursor: [usize; 3],
    draws: usize,
}
impl Balanced {
    pub fn admit(&mut self, key: String, ex: Arc<MicroExample>) {
        let k = (ex.value as i8 + 1) as usize;
        if k >= 3 || ![-1., 0., 1.].contains(&ex.value) {
            return;
        }
        if self.rows[k].iter().any(|(s, _)| s == &key) {
            return;
        }
        if self.rows[k].len() == 64 {
            self.rows[k].pop_front();
        }
        self.rows[k].push_back((key, ex));
    }
    pub fn draw(&mut self) -> Option<Arc<MicroExample>> {
        for _ in 0..3 {
            let k = self.draws % 3;
            self.draws += 1;
            if self.rows[k].is_empty() {
                continue;
            }
            let i = self.cursor[k] % self.rows[k].len();
            self.cursor[k] += 1;
            let mut ex = self.rows[k][i].1.as_ref().clone();
            ex.policy_weight = 0.;
            ex.value_weight = 1.;
            ex.actions.clear();
            ex.policy.clear();
            ex.action_values.clear();
            return Some(Arc::new(ex));
        }
        None
    }
    pub fn restore(&mut self, v: &serde_json::Value) {
        self.draws = v["draws"].as_u64().unwrap_or(0) as usize;
        for i in 0..3 {
            self.cursor[i] = v["cursor"][i].as_u64().unwrap_or(0) as usize;
        }
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"draws":self.draws,"cursor":self.cursor,"classes":self.rows.iter().map(VecDeque::len).collect::<Vec<_>>(),"capacity_per_class":64})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frequent_wins_cannot_dominate_the_verified_value_quota() {
        let mut b = Balanced::default();
        for (z, n) in [(-1., 3), (0., 1), (1., 50)] {
            for i in 0..n {
                let e = Arc::new(MicroExample { policy_support: false, action_values: vec![Some(z as f64)],
                    state: vec![0.; MICRO_INPUTS],
                    actions: vec![[0.;32]],
                    policy: vec![1.],
                    value: z,
                    policy_weight: 0.,
                    value_weight: 0.,
                    sequence_source: 0,
                });
                b.admit(format!("{z}:{i}"), e);
            }
        }
        let mut counts = [0; 3];
        for _ in 0..300 {
            let e = b.draw().unwrap();
            e.validate().unwrap();
            assert!(e.action_values.is_empty());
            assert_eq!(e.value_weight, 1.);
            assert_eq!(e.policy_weight, 0.);
            counts[(e.value as i8 + 1) as usize] += 1;
        }
        assert_eq!(counts, [100, 100, 100]);
    }
}
