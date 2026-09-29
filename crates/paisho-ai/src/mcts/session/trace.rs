//! Explicit read-only diagnostics: no legal enumeration, inference or search.
use super::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MctsPathStage {
    Expanded,
    PreparedOnly,
    RankedOnly,
    UnrankedOnly,
    ActionsNotGenerated,
    Absent,
    AncestorUnexpanded,
    NoTree,
}
#[derive(Clone, Debug, PartialEq)]
pub struct MctsPathStep {
    pub action: Action,
    pub stage: MctsPathStage,
    pub chooser: Option<Player>,
    pub visits: usize,
    pub parent_visited_children: usize,
    /// One-based position among remaining actions, next expansion first.
    pub pending_rank: Option<usize>,
    /// Converted to the requested perspective, including same-player bonuses.
    pub mean_value: Option<f64>,
    pub cached_leaf_value: Option<f64>,
    pub proof: Option<GameOutcome>,
}
impl MctsSession<'_> {
    /// Distinguishes ranked/evaluated candidates from actual visited branches.
    /// Evidence describes the currently retained tree, not discarded history.
    pub fn trace_path(&self, path: &[Action], perspective: Player) -> Vec<MctsPathStep> {
        let sign = if self.perspective == Some(perspective) {
            1.
        } else {
            -1.
        };
        let mut node = self.root.as_ref();
        path.iter()
            .enumerate()
            .map(|(i, action)| {
                let mut step = MctsPathStep {
                    action: *action,
                    stage: if i == 0 {
                        MctsPathStage::NoTree
                    } else {
                        MctsPathStage::AncestorUnexpanded
                    },
                    chooser: None,
                    visits: 0,
                    parent_visited_children: 0,
                    pending_rank: None,
                    mean_value: None,
                    cached_leaf_value: None,
                    proof: None,
                };
                let Some(parent) = node else {
                    return step;
                };
                step.chooser = Some(parent.position.to_move());
                step.parent_visited_children = parent.children.len();
                step.pending_rank = parent
                    .ranked_unexpanded
                    .iter()
                    .rev()
                    .position(|a| a == action)
                    .map(|i| i + 1);
                if let Some(child) = parent.children.iter().find(|c| c.action == *action) {
                    step.stage = MctsPathStage::Expanded;
                    step.visits = child.node.visits;
                    step.mean_value = (child.node.visits > 0)
                        .then(|| sign * child.node.value_sum / child.node.visits as f64);
                    step.cached_leaf_value = child.node.cached_leaf.map(|v| sign * v as f64);
                    step.proof = child.node.proof;
                    node = Some(&child.node);
                } else {
                    let prepared = parent.prepared.iter().find(|c| c.action == *action);
                    step.stage = if prepared.is_some() {
                        MctsPathStage::PreparedOnly
                    } else if parent.ranked_unexpanded.contains(action) {
                        MctsPathStage::RankedOnly
                    } else if parent.unranked.contains(action) {
                        MctsPathStage::UnrankedOnly
                    } else if !parent.actions_ready {
                        MctsPathStage::ActionsNotGenerated
                    } else {
                        MctsPathStage::Absent
                    };
                    step.cached_leaf_value =
                        prepared.and_then(|c| c.leaf_value).map(|v| sign * v as f64);
                    node = None;
                }
                step
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trace_separates_prepared_from_visited_and_preserves_search() {
        let p = paisho_core::Position::from_standard_setup(paisho_core::StandardSetup::balanced(
            paisho_core::BasicFlower::Red3,
        ));
        let actions = paisho_core::legal_actions(&p);
        let model = crate::CompactValueModel::default();
        let mut s = MctsSession::new(17, MctsConfig::default(), &model).unwrap();
        assert_eq!(
            s.trace_path(&actions[..1], p.to_move())[0].stage,
            MctsPathStage::NoTree
        );
        let mut root = Node::new(p.clone());
        let mut child = Node::new(p.clone());
        child.visits = 7;
        child.value_sum = 3.5;
        root.children.push(Child {
            action: actions[0],
            policy_bias: 0.,
            node: child,
        });
        root.prepared.push(PreparedCandidate {
            action: actions[1],
            position: p.clone(),
            leaf_value: Some(0.25),
        });
        s.perspective = Some(p.to_move());
        s.root = Some(root);
        let before = s.retained_bytes();
        let visited = s.trace_path(&actions[..1], p.to_move());
        assert_eq!(visited[0].visits, 7);
        assert_eq!(visited[0].mean_value, Some(0.5));
        let prepared = s.trace_path(&actions[1..3], p.to_move());
        assert_eq!(prepared[0].stage, MctsPathStage::PreparedOnly);
        assert_eq!(prepared[0].visits, 0);
        assert_eq!(prepared[0].cached_leaf_value, Some(0.25));
        assert_eq!(prepared[1].stage, MctsPathStage::AncestorUnexpanded);
        assert_eq!(before, s.retained_bytes());
        assert_eq!(
            s.trace_path(&actions[..1], p.to_move().opponent())[0].mean_value,
            Some(-0.5)
        );
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    #[test]
    fn reading_trace_does_not_consume_rng_or_change_next_search() {
        let p = paisho_core::Position::from_standard_setup(paisho_core::StandardSetup::balanced(
            paisho_core::BasicFlower::Red3,
        ));
        let legal = paisho_core::legal_actions(&p);
        let model = crate::CompactValueModel::default();
        let config = MctsConfig {
            simulations: 8,
            ..Default::default()
        };
        let mut a = MctsSession::new(731, config, &model).unwrap();
        let mut b = MctsSession::new(731, config, &model).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(
                a.search_until(&p, &legal, None).unwrap(),
                b.search_until(&p, &legal, None).unwrap()
            );
            for action in &legal {
                a.trace_path(&[*action], p.to_move());
            }
            assert_eq!(
                a.search_until(&p, &legal, None).unwrap(),
                b.search_until(&p, &legal, None).unwrap()
            );
        });
    }
}
