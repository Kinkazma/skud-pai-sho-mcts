//! Proof-backed learning corrections, separate from heuristic search disagreement.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TacticalEvidence {
    pub schema: String,
    /// All outcomes are relative to the player choosing at this position.
    pub root_value: Option<i8>,
    pub action_values: Vec<Option<i8>>,
    pub network_value: f64,
    pub network_best_action: Option<usize>,
}
impl TacticalEvidence {
    pub(crate) fn from_report(r: &MicroSearchReport, policy: bool) -> Option<Self> {
        if r.proven_value.is_none()
            && (!policy || !r.proven_action_values.iter().any(Option::is_some))
        {
            return None;
        }
        Some(Self {
            schema: "paisho-mcts-proof-v1".into(),
            root_value: r.proven_value,
            action_values: if policy {
                r.proven_action_values.clone()
            } else {
                vec![]
            },
            network_value: r.network_value,
            network_best_action: if policy {
                r.priors
                    .iter()
                    .enumerate()
                    .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
                    .map(|(i, _)| i)
            } else {
                None
            },
        })
    }
    fn allowed(&self, i: usize) -> bool {
        if self.action_values.contains(&Some(1)) {
            self.action_values[i] == Some(1)
        } else if self.root_value == Some(0) {
            self.action_values[i] == Some(0)
        } else {
            self.action_values.iter().all(|v| *v == Some(-1)) || self.action_values[i] != Some(-1)
        }
    }
    pub(crate) fn informative(&self) -> bool {
        self.root_value
            .is_some_and(|v| (v as f64 - self.network_value).abs() >= 0.5)
            || self
                .network_best_action
                .is_some_and(|i| i < self.action_values.len() && !self.allowed(i))
    }
    pub(crate) fn validate(&self, s: &SavedMicroExample) -> Result<()> {
        if self.schema != "paisho-mcts-proof-v1"
            || !self.network_value.is_finite()
            || !(-1.0..=1.0).contains(&self.network_value)
            || self.root_value.is_some_and(|v| !(-1..=1).contains(&v))
            || self
                .action_values
                .iter()
                .flatten()
                .any(|v| !(-1..=1).contains(v))
            || self.action_values.len() != s.actions.len()
            || s.policy.len() != self.action_values.len()
            || self
                .network_best_action
                .is_some_and(|i| i >= self.action_values.len())
            || (self.root_value.is_none() && !self.action_values.iter().any(Option::is_some))
        {
            return Err(invalid("invalid tactical evidence identity or alignment"));
        }
        if self.root_value.is_some_and(|v| s.value != v as f64) {
            return Err(invalid("proven value must replace network/game mixture"));
        }
        if s.policy
            .iter()
            .enumerate()
            .any(|(i, p)| !self.allowed(i) && *p != 0.0)
        {
            return Err(invalid("tactical target retains a refuted action"));
        }
        Ok(())
    }
}

/// Bound each game's influence; retain evenly spaced informative corrections.
pub(crate) fn prioritize(samples: &mut [SavedMicroExample]) {
    let indices: Vec<_> = samples
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            s.tactical
                .as_ref()
                .is_some_and(TacticalEvidence::informative)
                .then_some(i)
        })
        .collect();
    let count = indices.len().min(4);
    for j in 0..count {
        samples[indices[j * indices.len() / count]].correction_priority = true;
    }
}

#[cfg(test)]
pub(crate) fn fixture() -> SavedMicroExample {
    SavedMicroExample { evidence: None,
        rules: gen5::RULES.to_string(),
        source_run: "test".into(),
        game_id: "1".into(),
        decision: 1,
        collector: "a".repeat(64),
        budget: 64,
        inherited_visits: 0,
        new_visits: vec![1, 99],
        policy_raw_visits: vec![1, 99],
        policy_pruned_visits: vec![0, 99],
        actions: vec!["a".into(), "b".into()],
        state: vec![0.0; 128],
        action_features: vec![vec![0.0; 32]; 2],
        policy: vec![1.0, 0.0],
        value: 1.0,
        policy_weight: 1.0,
        reason: "search-proven-value".into(),
        correction_priority: false,
        tactical: Some(TacticalEvidence {
            schema: "paisho-mcts-proof-v1".into(),
            root_value: Some(1),
            action_values: vec![Some(1), None],
            network_value: 0.0,
            network_best_action: Some(1),
        }),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_exact_value_mask_and_archive_compatibility() {
        let mut s = fixture();
        assert!(s.example_for_rules(super::super::gen5::RULES).is_ok());
        s.value = 0.5;
        assert!(s.example_for_rules(super::super::gen5::RULES).is_err());
        s.value = 1.0;
        s.policy = vec![0.5, 0.5];
        assert!(s.example_for_rules(super::super::gen5::RULES).is_err());
        s.policy = vec![1.0, 0.0];
        s.correction_priority = true;
        let restored: SavedMicroExample =
            serde_json::from_value(serde_json::to_value(&s).unwrap()).unwrap();
        assert!(restored
            .example_for_rules(super::super::gen5::RULES)
            .is_ok());
        s.tactical = None;
        assert!(s.example_for_rules(super::super::gen5::RULES).is_err());
        s.correction_priority = false;
        let old = serde_json::to_value(&s).unwrap();
        assert!(old.get("tactical").is_none());
        assert!(old.get("correction_priority").is_none());
    }
    #[test]
    fn priority_requires_correction_and_cannot_be_dominated_by_one_long_game() {
        let mut all = vec![fixture(); 101];
        prioritize(&mut all);
        assert_eq!(all.iter().filter(|s| s.correction_priority).count(), 4);
        let mut uninformative = fixture();
        let t = uninformative.tactical.as_mut().unwrap();
        t.network_value = 0.9;
        t.network_best_action = Some(0);
        assert!(!t.informative());
        prioritize(std::slice::from_mut(&mut uninformative));
        assert!(!uninformative.correction_priority);
    }
}
