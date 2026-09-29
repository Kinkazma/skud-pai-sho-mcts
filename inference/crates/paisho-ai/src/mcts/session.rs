//! Private, per-game search memory. No evaluator replacement is exposed.
use super::*;

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
    last_proofs: Vec<Option<GameOutcome>>,
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
            last_proofs: Vec::new(),
        })
    }

    /// Enable sound terminal minimax propagation. Changing this clears the tree.
    pub fn set_solver(&mut self, enabled: bool) {
        if self.solver != enabled {
            self.clear();
            self.solver = enabled;
        }
    }
    /// Exact outcomes per supplied root action; missing means unknown, never loss.
    pub fn action_proofs(&self) -> &[Option<GameOutcome>] {
        &self.last_proofs
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
            if simulations > 0 && deadline.is_some_and(|limit| paisho_platform::training_time::now() >= limit) {
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
        let selected_index = actions
            .iter()
            .enumerate()
            .max_by(|(li, left), (ri, right)| {
                solver::rank(proofs[*li], position.to_move())
                    .cmp(&solver::rank(proofs[*ri], position.to_move()))
                    .then_with(|| left.visits.cmp(&right.visits))
                    .then_with(|| left.mean_value().total_cmp(&right.mean_value()))
                    .then_with(|| ri.cmp(li))
            })
            .unwrap()
            .0;
        let tree_bytes_before_limit = self.retained_bytes();
        let memory_limit_reset = tree_bytes_before_limit > self.memory_limit_bytes;
        if memory_limit_reset {
            self.clear();
        }
        self.last_proofs = proofs;
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
