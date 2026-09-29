//! Private, per-game search memory. No evaluator replacement is exposed.
use super::*;
mod trace;
pub use trace::{MctsPathStage, MctsPathStep};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MctsReuseStatistics {
    pub inherited_root_visits: usize,
    pub reused_candidate_positions: usize,
    pub reused_leaf_values: usize,
    /// Resident tree estimate including vector capacities; excludes search scratch.
    pub tree_bytes_before_limit: usize,
    pub memory_limit_reset: bool,
}

/// A single retained tree tied to one immutable evaluator and configuration.
///
/// Call `advance` after *every* actual action, including the opponent's actions
/// and same-player harmony bonuses. A position mismatch safely discards memory.
/// Values in the tree retain their original player perspective; reports convert
/// them to the current player without assuming that every edge changes player.
/// As with ordinary zero-sum MCTS, evaluator values must be antisymmetric between
/// players. A game owns its evaluator snapshot for this session's entire lifetime.
/// The caller must not change it through interior mutability.
///
/// Each search pays `config.simulations` NEW visits (subject to a soft deadline).
/// Reported root action visits also include inherited visits. This is a different
/// search policy from restarting a tree; strength and real time must be measured.
pub struct MctsSession<'a> {
    config: MctsConfig,
    evaluator: &'a dyn MctsEvaluator,
    rng: StableRng,
    root: Option<Node>,
    perspective: Option<Player>,
    last_reuse: MctsReuseStatistics,
    memory_limit_bytes: usize,
    solver: bool,
    capture_coverage: bool,
    last_proofs: Vec<Option<GameOutcome>>,
    last_estimates: Vec<f64>,
    last_policy: Vec<f64>,
}

impl<'a> MctsSession<'a> {
    pub fn new(
        seed: u64,
        config: MctsConfig,
        evaluator: &'a dyn MctsEvaluator,
    ) -> Result<Self, String> {
        config.validate().map_err(|error| error.to_string())?;
        if config.independent_trees != 1 {
            return Err("retained MCTS sessions require one independent tree".into());
        }
        Ok(Self {
            config,
            evaluator,
            rng: StableRng::new(seed),
            root: None,
            perspective: None,
            last_reuse: MctsReuseStatistics::default(),
            memory_limit_bytes: 256 * 1024 * 1024,
            solver: false,
            capture_coverage: false,
            last_proofs: Vec::new(),
            last_estimates: Vec::new(),
            last_policy: Vec::new(),
        })
    }

    /// Enable sound terminal minimax propagation. Changing this clears the tree.
    pub fn set_solver(&mut self, enabled: bool) {
        if self.solver != enabled {
            self.clear();
            self.solver = enabled;
        }
    }
    /// Experimental internal-node move ordering. One capture can take the next
    /// ordinary widening slot per node. No extra visits or material penalty.
    pub fn set_capture_coverage(&mut self, enabled: bool) {
        if self.capture_coverage != enabled {
            self.clear();
            self.capture_coverage = enabled;
        }
    }
    /// Exact outcomes per supplied root action; missing means unknown, never loss.
    pub fn action_proofs(&self) -> &[Option<GameOutcome>] {
        &self.last_proofs
    }
    /// Current-player estimates with solver bounds; raw report sums stay intact.
    pub fn action_value_estimates(&self) -> &[f64] {
        &self.last_estimates
    }
    /// Visit policy restricted to recommended actions when proofs give a safe
    /// alternative. SearchReport remains the audit of actual visits.
    pub fn learning_policy(&self) -> &[f64] {
        &self.last_policy
    }

    /// Bound retained memory between searches. Search scratch and at most one
    /// search's growth are transient; this is not a process-RSS limit.
    pub fn set_memory_limit_bytes(&mut self, bytes: usize) {
        self.memory_limit_bytes = bytes;
        if self.retained_bytes() > bytes {
            self.clear();
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.root
            .as_ref()
            .map_or(0, |root| std::mem::size_of::<Node>() + root.heap_bytes())
    }

    pub fn clear(&mut self) {
        self.root = None;
        self.last_proofs.clear();
        self.last_estimates.clear();
        self.last_policy.clear();
        self.perspective = None;
        self.last_reuse = MctsReuseStatistics::default();
    }

    pub fn retained_visits(&self) -> usize {
        self.root.as_ref().map_or(0, |root| root.visits)
    }

    pub fn reuse_statistics(&self) -> MctsReuseStatistics {
        self.last_reuse
    }

    /// Keep exactly the played branch and release siblings. If the branch has
    /// only been ranked, retain its already-applied position and leaf value.
    /// Returns true when an expanded subtree was retained; false otherwise.
    pub fn advance(&mut self, action: Action) -> bool {
        let Some(mut root) = self.root.take() else {
            return false;
        };
        if let Some(index) = root
            .children
            .iter()
            .position(|child| child.action == action)
        {
            self.root = Some(root.children.swap_remove(index).node);
            return true;
        }
        if let Some(index) = root
            .prepared
            .iter()
            .position(|candidate| candidate.action == action)
        {
            let prepared = root.prepared.swap_remove(index);
            let mut next = Node::new(prepared.position);
            next.cache_candidates = true;
            next.solver = self.solver;
            next.capture_coverage = self.capture_coverage;
            next.cached_leaf = prepared.leaf_value;
            self.root = Some(next);
        } else {
            self.perspective = None;
        }
        false
    }

    pub fn search_until(
        &mut self,
        position: &Position,
        root_actions: &[Action],
        deadline: Option<std::time::Instant>,
    ) -> Result<SearchReport, String> {
        if root_actions.is_empty() {
            return Err("MCTS requires legal root actions".into());
        }
        // Root actions are engine-generated, exactly as for MctsAgent. Do not
        // regenerate the same list merely to validate an internal caller.
        if self
            .root
            .as_ref()
            .is_some_and(|root| root.position != *position)
        {
            self.clear();
        }
        let perspective = *self.perspective.get_or_insert(position.to_move());
        let mut metrics = TreeMetrics::default();
        if self.root.is_none() {
            let ordering = order_actions_cached(
                position,
                root_actions,
                perspective,
                self.config.heuristic_weights,
                &mut self.rng,
                self.evaluator,
                true,
            )?;
            let mut root = Node::with_ordered_actions(position.clone(), ordering.actions);
            root.cache_candidates = true;
            root.solver = self.solver;
            root.capture_coverage = self.capture_coverage;
            root.prepared = ordering.prepared;
            root.policy_bias = ordering.policy_bias;
            self.root = Some(root);
            metrics.evaluated_actions = root_actions.len();
            metrics.generated_nodes = 1;
            metrics.generated_actions = root_actions.len();
            metrics.action_ranking_worker_indices = ordering.worker_indices;
        }
        let root = self.root.as_mut().unwrap();
        root.apply_root_policy(root_actions, self.evaluator)?;
        let inherited_root_visits = root.visits;
        let mut simulation_rng = StableRng::new(self.rng.next_u64());
        let mut simulations = 0;
        for _ in 0..self.config.simulations {
            if self.solver && root.proof.is_some() {
                break;
            }
            if simulations > 0 && deadline.is_some_and(|limit| std::time::Instant::now() >= limit) {
                break;
            }
            if let Err(error) = simulate(
                root,
                perspective,
                self.config,
                &mut simulation_rng,
                &mut metrics,
                0,
                self.evaluator,
            ) {
                // A failed evaluation may have consumed an unexpanded batch.
                // Do not let a retry inherit this incomplete mutation.
                self.clear();
                return Err(error);
            }
            simulations += 1;
        }
        let sign = if perspective == position.to_move() {
            1.0
        } else {
            -1.0
        };
        let actions: Vec<_> = root_actions
            .iter()
            .map(|&action| {
                let child = root.children.iter().find(|child| child.action == action);
                ActionStatistics {
                    action,
                    visits: child.map_or(0, |child| child.node.visits),
                    value_sum: child.map_or(0.0, |child| sign * child.node.value_sum),
                }
            })
            .collect();
        let proofs: Vec<_> = root_actions
            .iter()
            .map(|action| {
                if self.solver {
                    root.children
                        .iter()
                        .find(|c| c.action == *action)
                        .and_then(|c| c.node.proof)
                } else {
                    None
                }
            })
            .collect();
        let estimates: Vec<_> = root_actions
            .iter()
            .map(|action| {
                root.children
                    .iter()
                    .find(|c| c.action == *action)
                    .map_or(0., |c| {
                        let mean = if c.node.visits == 0 {
                            0.
                        } else {
                            c.node.value_sum / c.node.visits as f64
                        };
                        sign * c.node.bounded_value(mean, perspective)
                    })
            })
            .collect();
        let eligible = solver::root_eligible(&proofs, &estimates, position.to_move());
        let selected_index = actions
            .iter()
            .enumerate()
            .filter(|(i, _)| eligible[*i])
            .max_by(|(li, left), (ri, right)| {
                solver::rank(proofs[*li], position.to_move())
                    .cmp(&solver::rank(proofs[*ri], position.to_move()))
                    .then_with(|| left.visits.cmp(&right.visits))
                    .then_with(|| estimates[*li].total_cmp(&estimates[*ri]))
                    .then_with(|| ri.cmp(li))
            })
            .unwrap()
            .0;
        let total: usize = actions
            .iter()
            .enumerate()
            .filter(|(i, _)| eligible[*i])
            .map(|(_, a)| a.visits)
            .sum();
        let policy: Vec<_> = actions
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if total == 0 {
                    f64::from(i == selected_index)
                } else if eligible[i] {
                    a.visits as f64 / total as f64
                } else {
                    0.
                }
            })
            .collect();
        let tree_bytes_before_limit = self.retained_bytes();
        let memory_limit_reset = tree_bytes_before_limit > self.memory_limit_bytes;
        if memory_limit_reset {
            self.clear();
        }
        self.last_proofs = proofs;
        self.last_estimates = estimates;
        self.last_policy = policy;
        self.last_reuse = MctsReuseStatistics {
            tree_bytes_before_limit,
            memory_limit_reset,
            inherited_root_visits,
            reused_candidate_positions: metrics.reused_candidate_positions,
            reused_leaf_values: metrics.reused_leaf_values,
        };
        Ok(SearchReport {
            selected_index,
            simulations,
            trees: 1,
            workers: 1,
            worker_capacity: 1,
            action_ranking_workers: metrics.action_ranking_worker_indices.len().max(1),
            action_ranking_worker_capacity: rayon::current_num_threads(),
            evaluated_actions: metrics.evaluated_actions,
            expanded_nodes: metrics.expanded_nodes,
            generated_nodes: metrics.generated_nodes,
            generated_actions: metrics.generated_actions,
            maximum_depth: metrics.maximum_depth,
            rollout_steps: metrics.rollout_steps,
            actions,
        })
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod draw_floor_tests {
    use super::*;
    #[test]
    fn retained_root_recommends_and_teaches_draw_but_keeps_exploring_unknown() {
        let p = Position::from_standard_setup(paisho_core::StandardSetup::balanced(
            paisho_core::BasicFlower::Red3,
        ));
        let actions = legal_actions(&p);
        let model = crate::CompactValueModel::default();
        for solver in [false, true] {
            let mut session = MctsSession::new(
                17,
                MctsConfig {
                    simulations: 1,
                    maximum_tree_depth: 1,
                    ..Default::default()
                },
                &model,
            )
            .unwrap();
            session.set_solver(solver);
            let mut root = Node::with_ordered_actions(p.clone(), vec![]);
            root.solver = solver;
            root.visits = 100;
            for (i, (visits, sum, proof)) in [(1, 0., Some(GameOutcome::Draw)), (99, -89.1, None)]
                .into_iter()
                .enumerate()
            {
                let mut n = Node::new(p.clone());
                n.solver = solver;
                n.visits = visits;
                n.value_sum = sum;
                n.proof = proof;
                root.children.push(Child {
                    action: actions[i],
                    node: n,
                    policy_bias: 0.,
                });
            }
            session.root = Some(root);
            session.perspective = Some(p.to_move());
            let report = session.search_until(&p, &actions[..2], None).unwrap();
            if solver {
                assert_eq!(report.selected_index, 0);
                assert_eq!(report.actions[0].visits, 1);
                assert_eq!(report.actions[1].visits, 100);
                assert!(report.actions[1].mean_value() < 0.);
                assert_eq!(session.learning_policy(), &[1., 0.]);
                assert_eq!(session.action_proofs(), &[Some(GameOutcome::Draw), None]);
                assert_eq!(session.action_value_estimates()[0], 0.);
            } else {
                assert_eq!(report.selected_index, 1);
                assert!(session.learning_policy()[1] > 0.9);
            }
        }
    }
}
