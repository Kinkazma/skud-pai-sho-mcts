use core::fmt;

mod session;
mod solver;
pub use session::{MctsReuseStatistics, MctsSession};

use paisho_core::{legal_actions, Action, GameOutcome, Player, Position};
use rayon::prelude::*;

use crate::{evaluate_position, Agent, AgentError, AgentTelemetry, HeuristicWeights, StableRng};

/// Candidate ordering and leaf evaluation for one immutable search snapshot.
/// Implementations preserve input order and bounded finite values. Existing
/// CPU/Metal ordering backends keep the legacy CPU leaf by default; learned
/// evaluators override `evaluate_leaf` as well as `evaluate`.
pub trait MctsEvaluator: Sync {
    /// Optional chooser-relative bounded progressive bias. Evaluated once per
    /// prepared batch, retained with the tree; zero keeps historical UCT exact.
    fn policy_bias(
        &self,
        _position: &Position,
        _actions: &[Action],
    ) -> Result<Option<Vec<f64>>, String> {
        Ok(None)
    }

    /// Root-only retrieval override. Cached branches retain their values; only
    /// chooser-relative progressive biases change when a node becomes real root.
    fn root_policy_bias(
        &self,
        _position: &Position,
        _actions: &[Action],
    ) -> Result<Option<Vec<f64>>, String> {
        Ok(None)
    }

    /// Opt in only when ordering values equal leaf values for the same snapshot.
    fn ordering_matches_leaf(&self) -> bool {
        false
    }

    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String>;

    fn evaluate_leaf(
        &self,
        position: &Position,
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<f32, String> {
        Ok(evaluate_position(position, perspective, weights))
    }
}

pub struct CpuMctsEvaluator;
impl MctsEvaluator for CpuMctsEvaluator {
    fn ordering_matches_leaf(&self) -> bool {
        true
    }

    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        Ok(if positions.len() >= PARALLEL_ACTION_RANKING_THRESHOLD {
            positions
                .par_iter()
                .map(|p| evaluate_position(p, perspective, weights))
                .collect()
        } else {
            positions
                .iter()
                .map(|p| evaluate_position(p, perspective, weights))
                .collect()
        })
    }
}

pub const EXHAUSTIVE_ACTION_RANKING: usize = usize::MAX;
// Below this measured crossover, Rayon coordination is not reliably profitable.
pub(crate) const PARALLEL_ACTION_RANKING_THRESHOLD: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    pub simulations: usize,
    /// Logical independent root trees. This is part of the agent definition
    /// and deliberately does not change with the host's CPU count.
    pub independent_trees: usize,
    pub maximum_tree_depth: usize,
    /// Previously unseen internal actions ranked together when progressive
    /// widening requests another candidate.
    pub action_rank_batch_size: usize,
    /// Multiplier applied to `sqrt(visits + 1)` when deciding how many
    /// children may be expanded at the played root.
    pub root_widening_factor: f32,
    /// Multiplier applied to `sqrt(visits + 1)` below the played root.
    pub progressive_widening_factor: f32,
    pub rollout_depth: usize,
    pub exploration: f32,
    pub heuristic_weights: HeuristicWeights,
}

impl MctsConfig {
    pub fn validate(self) -> Result<Self, MctsConfigError> {
        if self.simulations == 0 {
            return Err(MctsConfigError::ZeroSimulations);
        }
        if self.independent_trees == 0 {
            return Err(MctsConfigError::ZeroIndependentTrees);
        }
        if self.maximum_tree_depth == 0 {
            return Err(MctsConfigError::ZeroTreeDepth);
        }
        if self.action_rank_batch_size == 0 {
            return Err(MctsConfigError::ZeroActionRankBatch);
        }
        if !valid_widening_factor(self.root_widening_factor)
            || !valid_widening_factor(self.progressive_widening_factor)
        {
            return Err(MctsConfigError::InvalidProgressiveWideningFactor);
        }
        if !self.exploration.is_finite() || self.exploration < 0.0 {
            return Err(MctsConfigError::InvalidExploration);
        }
        if !self.heuristic_weights.all_finite() {
            return Err(MctsConfigError::NonFiniteHeuristicWeight);
        }
        Ok(self)
    }
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 1_024,
            independent_trees: 1,
            maximum_tree_depth: 96,
            action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
            root_widening_factor: 1.0,
            progressive_widening_factor: 1.0,
            rollout_depth: 0,
            exploration: core::f32::consts::SQRT_2,
            heuristic_weights: HeuristicWeights::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MctsConfigError {
    ZeroSimulations,
    ZeroIndependentTrees,
    ZeroTreeDepth,
    ZeroActionRankBatch,
    InvalidProgressiveWideningFactor,
    InvalidExploration,
    NonFiniteHeuristicWeight,
}

impl fmt::Display for MctsConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::ZeroSimulations => "MCTS requires at least one simulation",
            Self::ZeroIndependentTrees => "MCTS requires at least one independent tree",
            Self::ZeroTreeDepth => "MCTS requires a positive tree depth",
            Self::ZeroActionRankBatch => "MCTS requires a positive action-ranking batch size",
            Self::InvalidProgressiveWideningFactor => {
                "MCTS progressive-widening factor must be finite and positive"
            }
            Self::InvalidExploration => "MCTS exploration must be finite and non-negative",
            Self::NonFiniteHeuristicWeight => "MCTS heuristic weights must all be finite",
        };
        formatter.write_str(text)
    }
}

impl std::error::Error for MctsConfigError {}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActionStatistics {
    pub action: Action,
    pub visits: usize,
    pub value_sum: f64,
}

impl ActionStatistics {
    pub fn mean_value(self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchReport {
    pub selected_index: usize,
    pub simulations: usize,
    pub trees: usize,
    /// Number of distinct Rayon workers that actually executed tree jobs.
    pub workers: usize,
    pub worker_capacity: usize,
    /// Number of distinct Rayon workers that evaluated action-ordering candidates.
    pub action_ranking_workers: usize,
    pub action_ranking_worker_capacity: usize,
    pub evaluated_actions: usize,
    pub expanded_nodes: usize,
    /// Positions whose action list was actually generated (including roots).
    pub generated_nodes: usize,
    /// Actions in those materialized lists; deferred leaf lists cost nothing.
    pub generated_actions: usize,
    pub maximum_depth: usize,
    pub rollout_steps: usize,
    pub actions: Vec<ActionStatistics>,
}

impl SearchReport {
    pub fn mean_branching_factor(&self) -> f64 {
        if self.generated_nodes == 0 {
            0.0
        } else {
            self.generated_actions as f64 / self.generated_nodes as f64
        }
    }

    pub fn visited_root_actions(&self) -> usize {
        self.actions
            .iter()
            .filter(|action| action.visits > 0)
            .count()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MctsAgent {
    config: MctsConfig,
    rng: StableRng,
    last_report: Option<SearchReport>,
    decisions: usize,
    total_simulations: usize,
    total_evaluated_actions: usize,
    total_expanded_nodes: usize,
    total_generated_nodes: usize,
    total_generated_actions: usize,
    maximum_search_depth: usize,
    maximum_search_trees: usize,
    maximum_search_workers: usize,
    maximum_search_worker_capacity: usize,
    maximum_action_ranking_workers: usize,
    maximum_action_ranking_worker_capacity: usize,
    total_rollout_steps: usize,
}

impl MctsAgent {
    pub fn new(seed: u64, config: MctsConfig) -> Result<Self, MctsConfigError> {
        Ok(Self {
            config: config.validate()?,
            rng: StableRng::new(seed),
            last_report: None,
            decisions: 0,
            total_simulations: 0,
            total_evaluated_actions: 0,
            total_expanded_nodes: 0,
            total_generated_nodes: 0,
            total_generated_actions: 0,
            maximum_search_depth: 0,
            maximum_search_trees: 0,
            maximum_search_workers: 0,
            maximum_search_worker_capacity: 0,
            maximum_action_ranking_workers: 0,
            maximum_action_ranking_worker_capacity: 0,
            total_rollout_steps: 0,
        })
    }

    pub const fn config(&self) -> MctsConfig {
        self.config
    }

    pub fn last_report(&self) -> Option<&SearchReport> {
        self.last_report.as_ref()
    }

    pub fn search(&mut self, position: &Position, root_actions: &[Action]) -> SearchReport {
        self.search_with_evaluator(position, root_actions, &CpuMctsEvaluator)
            .expect("CPU MCTS evaluation is infallible")
    }

    pub fn search_with_evaluator(
        &mut self,
        position: &Position,
        root_actions: &[Action],
        evaluator: &dyn MctsEvaluator,
    ) -> Result<SearchReport, String> {
        self.search_with_evaluator_until(position, root_actions, evaluator, None)
    }

    /// Optional soft wall deadline, checked between simulations. Root ordering
    /// and at least one simulation per logical tree complete even after expiry.
    /// Reports the actual simulation count; ordinary fixed-budget search is unchanged.
    pub fn search_with_evaluator_until(
        &mut self,
        position: &Position,
        root_actions: &[Action],
        evaluator: &dyn MctsEvaluator,
        deadline: Option<std::time::Instant>,
    ) -> Result<SearchReport, String> {
        if root_actions.is_empty() {
            return Err("MCTS requires legal root actions".into());
        }
        assert!(
            !root_actions.is_empty(),
            "MCTS needs at least one legal action"
        );
        let perspective = position.to_move();
        let root_ordering = order_actions(
            position,
            root_actions,
            perspective,
            self.config.heuristic_weights,
            &mut self.rng,
            evaluator,
        )?;
        let ordered_root_actions = root_ordering.actions;
        let trees = self.config.independent_trees.min(self.config.simulations);
        let worker_capacity = rayon::current_num_threads().min(trees);
        let action_ranking_worker_capacity = rayon::current_num_threads();
        let base = self.config.simulations / trees;
        let remainder = self.config.simulations % trees;
        let jobs: Vec<_> = (0..trees)
            .map(|tree| WorkerJob {
                simulations: base + usize::from(tree < remainder),
                seed: self.rng.next_u64(),
            })
            .collect();

        let local_results: Vec<_> = jobs
            .into_par_iter()
            .map(|job| {
                run_local_tree(
                    position,
                    &ordered_root_actions,
                    &root_ordering.policy_bias,
                    perspective,
                    self.config,
                    job,
                    evaluator,
                    deadline,
                )
            })
            .collect();

        let mut actions: Vec<_> = root_actions
            .iter()
            .copied()
            .map(|action| ActionStatistics {
                action,
                visits: 0,
                value_sum: 0.0,
            })
            .collect();
        let mut evaluated_actions = root_actions.len();
        let mut expanded_nodes = 0;
        let mut generated_nodes = 0;
        let mut generated_actions = 0;
        let mut maximum_depth = 0;
        let mut rollout_steps = 0;
        let mut completed_simulations = 0;
        let mut worker_indices = Vec::new();
        let mut action_ranking_worker_indices = root_ordering.worker_indices;
        for local in local_results {
            let local = local?;
            completed_simulations += local.simulations;
            if let Some(index) = local.worker_index {
                if !worker_indices.contains(&index) {
                    worker_indices.push(index);
                }
            }
            evaluated_actions += local.metrics.evaluated_actions;
            expanded_nodes += local.metrics.expanded_nodes;
            generated_nodes += local.metrics.generated_nodes;
            generated_actions += local.metrics.generated_actions;
            maximum_depth = maximum_depth.max(local.metrics.maximum_depth);
            rollout_steps += local.metrics.rollout_steps;
            merge_worker_indices(
                &mut action_ranking_worker_indices,
                local.metrics.action_ranking_worker_indices,
            );
            for statistic in local.actions {
                let index = root_actions
                    .iter()
                    .position(|action| *action == statistic.action)
                    .expect("local roots retain a supplied root action");
                actions[index].visits += statistic.visits;
                actions[index].value_sum += statistic.value_sum;
            }
        }

        let selected_index = actions
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.visits
                    .cmp(&right.visits)
                    .then_with(|| left.mean_value().total_cmp(&right.mean_value()))
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("root actions are non-empty");
        Ok(SearchReport {
            selected_index,
            simulations: completed_simulations,
            trees,
            workers: worker_indices.len().max(1),
            worker_capacity,
            action_ranking_workers: action_ranking_worker_indices.len().max(1),
            action_ranking_worker_capacity,
            evaluated_actions,
            expanded_nodes,
            generated_nodes,
            generated_actions,
            maximum_depth,
            rollout_steps,
            actions,
        })
    }
}

impl Agent for MctsAgent {
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        let report = self.search(position, legal_actions);
        let selected = report.selected_index;
        self.decisions += 1;
        self.total_simulations += report.simulations;
        self.total_evaluated_actions += report.evaluated_actions;
        self.total_expanded_nodes += report.expanded_nodes;
        self.total_generated_nodes += report.generated_nodes;
        self.total_generated_actions += report.generated_actions;
        self.maximum_search_depth = self.maximum_search_depth.max(report.maximum_depth);
        self.maximum_search_trees = self.maximum_search_trees.max(report.trees);
        self.maximum_search_workers = self.maximum_search_workers.max(report.workers);
        self.maximum_search_worker_capacity = self
            .maximum_search_worker_capacity
            .max(report.worker_capacity);
        self.maximum_action_ranking_workers = self
            .maximum_action_ranking_workers
            .max(report.action_ranking_workers);
        self.maximum_action_ranking_worker_capacity = self
            .maximum_action_ranking_worker_capacity
            .max(report.action_ranking_worker_capacity);
        self.total_rollout_steps += report.rollout_steps;
        self.last_report = Some(report);
        Ok(selected)
    }

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry {
            decisions: self.decisions,
            simulations: self.total_simulations,
            evaluated_actions: self.total_evaluated_actions,
            expanded_nodes: self.total_expanded_nodes,
            generated_nodes: self.total_generated_nodes,
            generated_actions: self.total_generated_actions,
            maximum_search_depth: self.maximum_search_depth,
            maximum_search_trees: self.maximum_search_trees,
            maximum_search_workers: self.maximum_search_workers,
            maximum_search_worker_capacity: self.maximum_search_worker_capacity,
            maximum_action_ranking_workers: self.maximum_action_ranking_workers,
            maximum_action_ranking_worker_capacity: self.maximum_action_ranking_worker_capacity,
            rollout_steps: self.total_rollout_steps,
        }
    }

    fn reset_telemetry(&mut self) {
        self.last_report = None;
        self.decisions = 0;
        self.total_simulations = 0;
        self.total_evaluated_actions = 0;
        self.total_expanded_nodes = 0;
        self.total_generated_nodes = 0;
        self.total_generated_actions = 0;
        self.maximum_search_depth = 0;
        self.maximum_search_trees = 0;
        self.maximum_search_workers = 0;
        self.maximum_search_worker_capacity = 0;
        self.maximum_action_ranking_workers = 0;
        self.maximum_action_ranking_worker_capacity = 0;
        self.total_rollout_steps = 0;
    }
}

#[derive(Clone, Copy)]
struct WorkerJob {
    simulations: usize,
    seed: u64,
}

fn run_local_tree(
    position: &Position,
    root_actions: &[Action],
    policy_bias: &[(Action, f64)],
    perspective: Player,
    config: MctsConfig,
    job: WorkerJob,
    evaluator: &dyn MctsEvaluator,
    deadline: Option<std::time::Instant>,
) -> Result<LocalTreeResult, String> {
    let mut rng = StableRng::new(job.seed);
    let mut root = Node::with_ordered_actions(position.clone(), root_actions.to_vec());
    root.policy_bias = policy_bias.to_vec();
    root.apply_root_policy(root_actions, evaluator)?;
    let mut metrics = TreeMetrics {
        generated_nodes: 1,
        generated_actions: root_actions.len(),
        ..TreeMetrics::default()
    };
    let mut simulations = 0;
    for _ in 0..job.simulations {
        if simulations > 0 && deadline.is_some_and(|limit| paisho_platform::training_time::now() >= limit) {
            break;
        }
        simulate(
            &mut root,
            perspective,
            config,
            &mut rng,
            &mut metrics,
            0,
            evaluator,
        )?;
        simulations += 1;
    }
    let actions = root
        .children
        .into_iter()
        .map(|child| ActionStatistics {
            action: child.action,
            visits: child.node.visits,
            value_sum: child.node.value_sum,
        })
        .collect();
    Ok(LocalTreeResult {
        simulations,
        actions,
        metrics,
        worker_index: rayon::current_thread_index(),
    })
}

struct LocalTreeResult {
    simulations: usize,
    actions: Vec<ActionStatistics>,
    metrics: TreeMetrics,
    worker_index: Option<usize>,
}

#[derive(Default)]
struct TreeMetrics {
    reused_candidate_positions: usize,
    reused_leaf_values: usize,
    evaluated_actions: usize,
    expanded_nodes: usize,
    generated_nodes: usize,
    generated_actions: usize,
    maximum_depth: usize,
    rollout_steps: usize,
    action_ranking_worker_indices: Vec<usize>,
}

struct PreparedCandidate {
    action: Action,
    position: Position,
    leaf_value: Option<f32>,
}

struct Node {
    position: Position,
    visits: usize,
    value_sum: f64,
    actions_ready: bool,
    unranked: Vec<Action>,
    ranked_unexpanded: Vec<Action>,
    children: Vec<Child>,
    cache_candidates: bool,
    prepared: Vec<PreparedCandidate>,
    cached_leaf: Option<f32>,
    policy_bias: Vec<(Action, f64)>,
    solver: bool,
    proof: Option<GameOutcome>,
}

impl Node {
    fn new(position: Position) -> Self {
        let proof = solver::terminal(&position);
        Self {
            position,
            solver: false,
            proof,
            visits: 0,
            value_sum: 0.0,
            actions_ready: false,
            unranked: Vec::new(),
            ranked_unexpanded: Vec::new(),
            children: Vec::new(),
            cache_candidates: false,
            prepared: Vec::new(),
            cached_leaf: None,
            policy_bias: Vec::new(),
        }
    }

    fn with_ordered_actions(position: Position, actions: Vec<Action>) -> Self {
        let proof = solver::terminal(&position);
        Self {
            position,
            solver: false,
            proof,
            visits: 0,
            value_sum: 0.0,
            actions_ready: true,
            unranked: Vec::new(),
            ranked_unexpanded: actions,
            children: Vec::new(),
            cache_candidates: false,
            prepared: Vec::new(),
            cached_leaf: None,
            policy_bias: Vec::new(),
        }
    }

    fn heap_bytes(&self) -> usize {
        self.policy_bias.capacity() * std::mem::size_of::<(Action, f64)>()
            + self.unranked.capacity() * std::mem::size_of::<Action>()
            + self.ranked_unexpanded.capacity() * std::mem::size_of::<Action>()
            + self.prepared.capacity() * std::mem::size_of::<PreparedCandidate>()
            + self.children.capacity() * std::mem::size_of::<Child>()
            + self
                .children
                .iter()
                .map(|child| child.node.heap_bytes())
                .sum::<usize>()
    }

    fn apply_root_policy(
        &mut self,
        actions: &[Action],
        evaluator: &dyn MctsEvaluator,
    ) -> Result<(), String> {
        if let Some(bias) = evaluator.root_policy_bias(&self.position, actions)? {
            if bias.len() != actions.len() || bias.iter().any(|b| !b.is_finite() || b.abs() > 1.0) {
                return Err("invalid root policy bias".into());
            }
            self.policy_bias = actions.iter().copied().zip(bias).collect();
            for child in &mut self.children {
                child.policy_bias = self
                    .policy_bias
                    .iter()
                    .find(|(a, _)| *a == child.action)
                    .map_or(0., |(_, b)| *b);
            }
        }
        Ok(())
    }
    fn action_count(&self) -> usize {
        self.unranked.len() + self.ranked_unexpanded.len()
    }

    fn ensure_actions(&mut self, metrics: &mut TreeMetrics) {
        if !self.actions_ready {
            self.unranked = legal_actions(&self.position);
            self.actions_ready = true;
            metrics.generated_nodes += 1;
            metrics.generated_actions += self.action_count();
        }
    }

    fn has_unexpanded(&self) -> bool {
        !self.ranked_unexpanded.is_empty() || !self.unranked.is_empty()
    }

    fn prepare_action_batch(
        &mut self,
        perspective: Player,
        weights: HeuristicWeights,
        batch_limit: usize,
        rng: &mut StableRng,
        evaluator: &dyn MctsEvaluator,
    ) -> Result<BatchPreparation, String> {
        if !self.ranked_unexpanded.is_empty() || self.unranked.is_empty() {
            return Ok(BatchPreparation::default());
        }
        let batch_size = self.unranked.len().min(batch_limit);
        let mut batch = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let index = rng.index(self.unranked.len());
            batch.push(self.unranked.swap_remove(index));
        }
        let ordering = order_actions_cached(
            &self.position,
            &batch,
            perspective,
            weights,
            rng,
            evaluator,
            self.cache_candidates,
        )?;
        self.policy_bias.extend(ordering.policy_bias);
        self.prepared = ordering.prepared;
        self.ranked_unexpanded = ordering.actions;
        Ok(BatchPreparation {
            evaluated_actions: batch_size,
            worker_indices: ordering.worker_indices,
        })
    }
}

#[derive(Default)]
struct BatchPreparation {
    evaluated_actions: usize,
    worker_indices: Vec<usize>,
}

struct ActionOrdering {
    policy_bias: Vec<(Action, f64)>,
    actions: Vec<Action>,
    prepared: Vec<PreparedCandidate>,
    worker_indices: Vec<usize>,
}

fn order_actions(
    position: &Position,
    actions: &[Action],
    perspective: Player,
    weights: HeuristicWeights,
    rng: &mut StableRng,
    evaluator: &dyn MctsEvaluator,
) -> Result<ActionOrdering, String> {
    order_actions_cached(
        position,
        actions,
        perspective,
        weights,
        rng,
        evaluator,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn order_actions_cached(
    position: &Position,
    actions: &[Action],
    perspective: Player,
    weights: HeuristicWeights,
    rng: &mut StableRng,
    evaluator: &dyn MctsEvaluator,
    cache: bool,
) -> Result<ActionOrdering, String> {
    let maximize = position.to_move() == perspective;
    let mut ranked: Vec<_> = actions
        .iter()
        .copied()
        .enumerate()
        .map(|(index, action)| (action, 0.0, rng.next_u64(), None, index))
        .collect();
    let prepare = |action: &Action| {
        let mut candidate = position.clone();
        candidate
            .apply(*action)
            .expect("MCTS ranks only engine-generated actions");
        (candidate, rayon::current_thread_index())
    };
    let prepared: Vec<_> = if actions.len() >= PARALLEL_ACTION_RANKING_THRESHOLD {
        actions.par_iter().map(prepare).collect()
    } else {
        actions.iter().map(prepare).collect()
    };
    let (positions, workers): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    let values = evaluator.evaluate(&positions, perspective, weights)?;
    if values.len() != ranked.len() || values.iter().any(|v| !v.is_finite() || v.abs() > 1.0) {
        return Err("invalid MCTS evaluator response".into());
    }
    for ((entry, value), worker) in ranked.iter_mut().zip(values.iter().copied()).zip(workers) {
        entry.1 = value;
        entry.3 = worker;
    }
    let bias = evaluator.policy_bias(position, actions)?;
    if bias.as_ref().is_some_and(|b| {
        b.len() != actions.len() || b.iter().any(|v| !v.is_finite() || v.abs() > 1.0)
    }) {
        return Err("invalid MCTS policy bias".into());
    }
    let policy_bias: Vec<_> = bias.as_ref().map_or_else(Vec::new, |b| {
        actions.iter().copied().zip(b.iter().copied()).collect()
    });
    if let Some(b) = bias {
        for entry in &mut ranked {
            let sign = if maximize { 1.0 } else { -1.0 };
            // Exact terminal scores remain above/below any policy adjustment.
            entry.1 += sign * 0.05 * b[entry.4] as f32 * (1.0 - entry.1.abs());
        }
    }
    ranked.sort_by(|left, right| {
        let value_order = if maximize {
            left.1.total_cmp(&right.1)
        } else {
            right.1.total_cmp(&left.1)
        };
        value_order.then_with(|| left.2.cmp(&right.2))
    });
    // Store only candidates likely to be expanded. The remaining order is kept;
    // an uncached candidate is applied normally if widening eventually uses it.
    let prepared = if cache {
        let mut keep = vec![false; actions.len()];
        for entry in ranked.iter().rev().take(64) {
            keep[entry.4] = true;
        }
        actions
            .iter()
            .copied()
            .zip(positions)
            .zip(values.iter().copied())
            .enumerate()
            .filter(|(index, _)| keep[*index])
            .map(|(_, ((action, position), value))| PreparedCandidate {
                action,
                position,
                leaf_value: evaluator.ordering_matches_leaf().then_some(value),
            })
            .collect()
    } else {
        Vec::new()
    };
    let mut worker_indices = Vec::new();
    for worker in ranked.iter().filter_map(|entry| entry.3) {
        push_worker_index(&mut worker_indices, worker);
    }
    Ok(ActionOrdering {
        policy_bias,
        prepared,
        actions: ranked
            .into_iter()
            .map(|(action, _, _, _, _)| action)
            .collect(),
        worker_indices,
    })
}

fn push_worker_index(indices: &mut Vec<usize>, worker: usize) {
    if !indices.contains(&worker) {
        indices.push(worker);
    }
}

fn merge_worker_indices(indices: &mut Vec<usize>, additional: Vec<usize>) {
    for worker in additional {
        push_worker_index(indices, worker);
    }
}

struct Child {
    policy_bias: f64,
    action: Action,
    node: Node,
}

fn simulate(
    node: &mut Node,
    perspective: Player,
    config: MctsConfig,
    rng: &mut StableRng,
    metrics: &mut TreeMetrics,
    depth: usize,
    evaluator: &dyn MctsEvaluator,
) -> Result<f64, String> {
    metrics.maximum_depth = metrics.maximum_depth.max(depth);
    if node.solver {
        if let Some(proof) = node.proof {
            return Ok(solver::value(proof, perspective));
        }
    }
    // A newly expanded child only needs a value. Generate its legal moves if
    // a later simulation actually searches through it, preserving order/RNG.
    if node.position.outcome() == GameOutcome::Ongoing && depth < config.maximum_tree_depth {
        node.ensure_actions(metrics);
    }
    let value =
        if node.position.outcome() != GameOutcome::Ongoing || depth >= config.maximum_tree_depth {
            node_leaf_value(
                node,
                perspective,
                config.heuristic_weights,
                evaluator,
                metrics,
            )? as f64
        } else if should_expand(
            node,
            if depth == 0 {
                config.root_widening_factor
            } else {
                config.progressive_widening_factor
            },
        ) {
            let preparation = node.prepare_action_batch(
                perspective,
                config.heuristic_weights,
                config.action_rank_batch_size,
                rng,
                evaluator,
            )?;
            metrics.evaluated_actions += preparation.evaluated_actions;
            merge_worker_indices(
                &mut metrics.action_ranking_worker_indices,
                preparation.worker_indices,
            );
            let action = node
                .ranked_unexpanded
                .pop()
                .expect("progressive widening prepares an action before expansion");
            let prepared = node
                .prepared
                .iter()
                .position(|candidate| candidate.action == action)
                .map(|index| node.prepared.swap_remove(index));
            let (position, cached_leaf) = if let Some(candidate) = prepared {
                metrics.reused_candidate_positions += 1;
                (candidate.position, candidate.leaf_value)
            } else {
                let mut position = node.position.clone();
                position
                    .apply(action)
                    .expect("MCTS expands only engine-generated actions");
                (position, None)
            };
            let mut child = Node::new(position);
            child.cache_candidates = node.cache_candidates;
            child.solver = node.solver;
            child.cached_leaf = cached_leaf;
            metrics.expanded_nodes += 1;
            metrics.maximum_depth = metrics.maximum_depth.max(depth + 1);
            let rollout = if config.rollout_depth == 0
                && child.cached_leaf.is_some()
                && child.position.outcome() == GameOutcome::Ongoing
            {
                metrics.reused_leaf_values += 1;
                RolloutResult {
                    value: child.cached_leaf.unwrap(),
                    steps: 0,
                }
            } else {
                rollout(
                    child.position.clone(),
                    perspective,
                    config.rollout_depth,
                    config.heuristic_weights,
                    rng,
                    evaluator,
                )?
            };
            if child.cache_candidates && config.rollout_depth == 0 {
                child.cached_leaf = Some(rollout.value);
            }
            metrics.rollout_steps += rollout.steps;
            let value = rollout.value as f64;
            child.visits = 1;
            child.value_sum = value;
            node.children.push(Child {
                policy_bias: node
                    .policy_bias
                    .iter()
                    .find(|(a, _)| *a == action)
                    .map_or(0.0, |(_, b)| *b),
                action,
                node: child,
            });
            value
        } else if node.children.is_empty() {
            node_leaf_value(
                node,
                perspective,
                config.heuristic_weights,
                evaluator,
                metrics,
            )? as f64
        } else {
            let selected = select_child(node, perspective, config.exploration);
            simulate(
                &mut node.children[selected].node,
                perspective,
                config,
                rng,
                metrics,
                depth + 1,
                evaluator,
            )?
        };

    if node.solver {
        node.refresh_proof();
    }
    node.visits += 1;
    node.value_sum += value;
    Ok(value)
}

fn node_leaf_value(
    node: &mut Node,
    perspective: Player,
    weights: HeuristicWeights,
    evaluator: &dyn MctsEvaluator,
    metrics: &mut TreeMetrics,
) -> Result<f32, String> {
    if node.cache_candidates && node.position.outcome() == GameOutcome::Ongoing {
        if let Some(value) = node.cached_leaf {
            metrics.reused_leaf_values += 1;
            return Ok(value);
        }
        let value = leaf_value(&node.position, perspective, weights, evaluator)?;
        node.cached_leaf = Some(value);
        Ok(value)
    } else {
        leaf_value(&node.position, perspective, weights, evaluator)
    }
}

fn should_expand(node: &Node, factor: f32) -> bool {
    node.has_unexpanded()
        && (node.children.len() < progressive_widening_limit(node.visits, factor)
            || (node.solver && node.children.iter().all(|c| c.node.proof.is_some())))
}

fn progressive_widening_limit(visits: usize, factor: f32) -> usize {
    (f64::from(factor) * (visits.saturating_add(1) as f64).sqrt())
        .ceil()
        .max(1.0) as usize
}

fn valid_widening_factor(factor: f32) -> bool {
    factor.is_finite() && factor > 0.0
}

fn select_child(node: &Node, perspective: Player, exploration: f32) -> usize {
    let maximize = node.position.to_move() == perspective;
    let parent_log = (node.visits.max(1) as f64).ln();
    let unresolved = node.solver
        && node.children.iter().any(|c| c.node.proof.is_none())
        && !node
            .children
            .iter()
            .any(|c| c.node.proof == Some(GameOutcome::Win(node.position.to_move())));
    node.children
        .iter()
        .enumerate()
        .filter(|(_, child)| !unresolved || child.node.proof.is_none())
        .max_by(|(left_index, left), (right_index, right)| {
            let left_score = uct_score(left, maximize, parent_log, exploration);
            let right_score = uct_score(right, maximize, parent_log, exploration);
            let proof_order = if node.solver {
                solver::rank(left.node.proof, node.position.to_move())
                    .cmp(&solver::rank(right.node.proof, node.position.to_move()))
            } else {
                std::cmp::Ordering::Equal
            };
            proof_order
                .then_with(|| left_score.total_cmp(&right_score))
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
        .expect("selection requires at least one child")
}

fn uct_score(child: &Child, maximize: bool, parent_log: f64, exploration: f32) -> f64 {
    let mean = child.node.value_sum / child.node.visits as f64;
    let exploitation = if maximize { mean } else { -mean };
    exploitation
        + exploration as f64 * (parent_log / child.node.visits as f64).sqrt()
        + child.policy_bias / (1.0 + child.node.visits as f64)
}

fn rollout(
    mut position: Position,
    perspective: Player,
    maximum_depth: usize,
    weights: HeuristicWeights,
    rng: &mut StableRng,
    evaluator: &dyn MctsEvaluator,
) -> Result<RolloutResult, String> {
    let mut steps = 0;
    for _ in 0..maximum_depth {
        if position.outcome() != GameOutcome::Ongoing {
            break;
        }
        let actions = legal_actions(&position);
        if actions.is_empty() {
            break;
        }
        let action = actions[rng.index(actions.len())];
        position
            .apply(action)
            .expect("rollouts use only engine-generated actions");
        steps += 1;
    }
    Ok(RolloutResult {
        value: leaf_value(&position, perspective, weights, evaluator)?,
        steps,
    })
}

fn leaf_value(
    position: &Position,
    perspective: Player,
    weights: HeuristicWeights,
    evaluator: &dyn MctsEvaluator,
) -> Result<f32, String> {
    // Learned weights cannot override a result established by the rules.
    match position.outcome() {
        GameOutcome::Win(winner) => return Ok(if winner == perspective { 1.0 } else { -1.0 }),
        GameOutcome::Draw => return Ok(0.0),
        GameOutcome::Ongoing => {}
    }
    let value = evaluator.evaluate_leaf(position, perspective, weights)?;
    if !value.is_finite() || value.abs() > 1.0 {
        return Err("invalid MCTS leaf evaluator response".into());
    }
    Ok(value)
}

struct RolloutResult {
    value: f32,
    steps: usize,
}

#[cfg(test)]
mod tests {
    use paisho_core::{legal_actions, BasicFlower, Position, StandardSetup};

    use super::{
        progressive_widening_limit, select_child, Child, MctsConfig, MctsConfigError, Node,
        EXHAUSTIVE_ACTION_RANKING,
    };
    use crate::{HeuristicWeights, StableRng};

    fn node_with_statistics(position: Position, visits: usize, value_sum: f64) -> Node {
        Node {
            position,
            solver: false,
            proof: None,
            visits,
            value_sum,
            actions_ready: true,
            unranked: Vec::new(),
            ranked_unexpanded: Vec::new(),
            children: Vec::new(),
            cache_candidates: false,
            prepared: Vec::new(),
            cached_leaf: None,
            policy_bias: Vec::new(),
        }
    }

    #[test]
    fn internal_action_batches_partition_all_candidates_without_eager_refill() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let expected = legal_actions(&position);
        let batch_size = 8;
        assert!(expected.len() > batch_size * 2);
        let mut node = Node::new(position.clone());
        node.ensure_actions(&mut super::TreeMetrics::default());
        let mut rng = StableRng::new(7);

        let first = node
            .prepare_action_batch(
                position.to_move(),
                HeuristicWeights::default(),
                batch_size,
                &mut rng,
                &super::CpuMctsEvaluator,
            )
            .unwrap();
        assert_eq!(first.evaluated_actions, batch_size);
        assert!(first.worker_indices.is_empty());
        assert_eq!(node.ranked_unexpanded.len(), batch_size);
        assert_eq!(node.unranked.len(), expected.len() - batch_size);
        let deferred = node
            .prepare_action_batch(
                position.to_move(),
                HeuristicWeights::default(),
                batch_size,
                &mut rng,
                &super::CpuMctsEvaluator,
            )
            .unwrap();
        assert_eq!(deferred.evaluated_actions, 0);
        assert!(deferred.worker_indices.is_empty());

        let mut seen = Vec::new();
        let mut evaluated = batch_size;
        while node.has_unexpanded() {
            while let Some(action) = node.ranked_unexpanded.pop() {
                assert!(expected.contains(&action));
                assert!(!seen.contains(&action), "candidate was duplicated");
                seen.push(action);
            }
            evaluated += node
                .prepare_action_batch(
                    position.to_move(),
                    HeuristicWeights::default(),
                    batch_size,
                    &mut rng,
                    &super::CpuMctsEvaluator,
                )
                .unwrap()
                .evaluated_actions;
        }

        assert_eq!(evaluated, expected.len());
        assert_eq!(seen.len(), expected.len());
        assert!(expected.iter().all(|action| seen.contains(action)));
    }

    #[test]
    fn deferred_actions_are_generated_once_and_preserve_engine_order() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let expected = legal_actions(&position);
        let mut node = Node::new(position);
        let mut metrics = super::TreeMetrics::default();
        assert!(!node.actions_ready);
        assert_eq!(node.action_count(), 0);
        node.ensure_actions(&mut metrics);
        assert_eq!(node.unranked, expected);
        assert_eq!(metrics.generated_nodes, 1);
        assert_eq!(metrics.generated_actions, expected.len());
        node.ensure_actions(&mut metrics);
        assert_eq!(metrics.generated_nodes, 1);
        assert_eq!(metrics.generated_actions, expected.len());
    }

    #[test]
    fn one_simulation_does_not_generate_unused_child_actions() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let actions = legal_actions(&position);
        let mut agent = super::MctsAgent::new(
            12,
            MctsConfig {
                simulations: 1,
                ..MctsConfig::default()
            },
        )
        .unwrap();
        let report = agent.search(&position, &actions);
        assert_eq!(report.simulations, 1);
        assert_eq!(report.expanded_nodes, 1);
        assert_eq!(report.generated_nodes, 1);
        assert_eq!(report.generated_actions, actions.len());
        assert_eq!(report.actions.iter().map(|a| a.visits).sum::<usize>(), 1);
    }

    #[test]
    fn selection_maximizes_for_root_and_minimizes_for_opponent() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let actions = legal_actions(&position);
        let mut parent = node_with_statistics(position.clone(), 20, 0.0);
        parent.children = vec![
            Child {
                policy_bias: 0.0,
                action: actions[0],
                node: node_with_statistics(position.clone(), 10, 6.0),
            },
            Child {
                policy_bias: 0.0,
                action: actions[1],
                node: node_with_statistics(position, 10, -2.0),
            },
        ];

        let root_player = parent.position.to_move();
        assert_eq!(select_child(&parent, root_player, 0.0), 0);
        assert_eq!(select_child(&parent, root_player.opponent(), 0.0), 1);
    }

    #[test]
    fn exploration_prefers_the_less_visited_equal_value_child() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let actions = legal_actions(&position);
        let mut parent = node_with_statistics(position.clone(), 5, 0.0);
        parent.children = vec![
            Child {
                policy_bias: 0.0,
                action: actions[0],
                node: node_with_statistics(position.clone(), 1, 0.0),
            },
            Child {
                policy_bias: 0.0,
                action: actions[1],
                node: node_with_statistics(position, 4, 0.0),
            },
        ];

        assert_eq!(select_child(&parent, parent.position.to_move(), 1.0), 0);
    }

    #[test]
    fn default_preserves_exhaustive_internal_action_ranking() {
        assert_eq!(
            MctsConfig::default().action_rank_batch_size,
            EXHAUSTIVE_ACTION_RANKING
        );
    }

    #[test]
    fn default_preserves_one_coherent_search_tree() {
        assert_eq!(MctsConfig::default().independent_trees, 1);
    }

    #[test]
    fn default_preserves_the_measured_uniform_widening_policy() {
        let config = MctsConfig::default();
        assert_eq!(config.root_widening_factor, 1.0);
        assert_eq!(config.progressive_widening_factor, 1.0);
        assert_eq!(progressive_widening_limit(127, 1.0), 12);
        assert_eq!(progressive_widening_limit(127, 3.0), 34);
    }

    #[test]
    fn configuration_rejects_zero_independent_trees() {
        let config = MctsConfig {
            independent_trees: 0,
            heuristic_weights: HeuristicWeights::default(),
            ..MctsConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(MctsConfigError::ZeroIndependentTrees)
        );
    }

    #[test]
    fn configuration_rejects_zero_action_ranking_batch() {
        let config = MctsConfig {
            action_rank_batch_size: 0,
            ..MctsConfig::default()
        };
        assert_eq!(config.validate(), Err(MctsConfigError::ZeroActionRankBatch));
    }

    #[test]
    fn configuration_rejects_invalid_progressive_widening_factors() {
        for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            for config in [
                MctsConfig {
                    root_widening_factor: invalid,
                    ..MctsConfig::default()
                },
                MctsConfig {
                    progressive_widening_factor: invalid,
                    ..MctsConfig::default()
                },
            ] {
                assert_eq!(
                    config.validate(),
                    Err(MctsConfigError::InvalidProgressiveWideningFactor)
                );
            }
        }
    }

    #[test]
    fn configuration_rejects_non_finite_heuristic_weights() {
        let config = MctsConfig {
            heuristic_weights: HeuristicWeights {
                harmony: f32::NAN,
                ..HeuristicWeights::default()
            },
            ..MctsConfig::default()
        };
        assert_eq!(
            config.validate(),
            Err(MctsConfigError::NonFiniteHeuristicWeight)
        );
    }
}
