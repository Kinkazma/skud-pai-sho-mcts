//! Retained PUCT with model-scoped, exact-state inference reuse. Search statistics
//! remain local to each tree edge; transpositions share only immutable inference.
use super::*;
mod certificates;
mod exploration;
mod forced;
mod proofs;
pub use certificates::MicroProofCertificate;
pub use exploration::{MicroSearchMode, MicroSearchOptions};
use paisho_core::{legal_actions, Action, GameOutcome, Player, Position, STANDARD_TILE_KINDS};
use rayon::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, OnceLock,
};
use std::time::Instant;

struct Policy {
    actions: Vec<Action>,
    features: Arc<Vec<[f64; MICRO_ACTION_INPUTS]>>,
    priors: Vec<f64>,
    log_priors: OnceLock<Vec<f64>>,
}
impl Policy {
    fn bytes(&self) -> usize {
        std::mem::size_of::<Policy>() + self.actions.capacity() * std::mem::size_of::<Action>()
            + self.features.capacity() * std::mem::size_of::<[f64; MICRO_ACTION_INPUTS]>()
            // Reserve the possible Gumbel log-prior cache conservatively even
            // before materialization; PUCT never pays its logarithms.
            + self.priors.capacity() * 16
    }
}
struct Inference {
    accounting: Option<Arc<AtomicUsize>>,
    resident: AtomicBool,
    position: Position,
    state: [f64; MICRO_INPUTS],
    embedding: MicroEmbedding,
    policy: OnceLock<Result<Arc<Policy>, String>>,
}
impl Inference {
    fn value(&self) -> f64 {
        match self.position.outcome() {
            GameOutcome::Win(p) => {
                if p == self.position.to_move() {
                    1.0
                } else {
                    -1.0
                }
            }
            GameOutcome::Draw => 0.0,
            GameOutcome::Ongoing => self.embedding.value,
        }
    }
    fn policy(&self) -> Result<Arc<Policy>, String> {
        self.policy
            .get_or_init(|| {
                let actions = legal_actions(&self.position);
                // Fuse extraction/scoring for both policy schemas. Cache the
                // resulting priors per node and amortize pool dispatch over at
                // least 256 actions per job; independent games share the pool.
                let score = |action: &Action| {
                    let features = micro_action_features(&self.position, *action);
                    let logit = MicroModel::logit(&self.embedding, &features);
                    (features, logit)
                };
                let rows: Vec<_> = if actions.len() >= 512 {
                    actions.par_iter().with_min_len(256).map(score).collect()
                } else {
                    actions.iter().map(score).collect()
                };
                let (features, logits): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
                let priors = if logits.is_empty() {
                    Vec::new()
                } else {
                    micro_softmax(&logits)?
                };
                let policy = Arc::new(Policy {
                    actions,
                    features: Arc::new(features),
                    priors,
                    log_priors: OnceLock::new(),
                });
                if self.resident.load(Ordering::Relaxed) {
                    if let Some(counter) = &self.accounting {
                        counter.fetch_add(policy.bytes(), Ordering::Relaxed);
                    }
                }
                Ok(policy)
            })
            .clone()
    }
    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self
                .policy
                .get()
                .and_then(|p| p.as_ref().ok())
                .map_or(0, |p| p.bytes())
    }
}
struct Node {
    memory_policy: Option<Arc<Policy>>,
    inference: Arc<Inference>,
    visits: usize,
    value_sum: f64,
    /// Root-forced visits of the incoming edge, retained with that edge.
    forced_visits: usize,
    proof: proofs::Proof,
    children: Vec<Option<Box<Node>>>,
}
impl Node {
    fn new(inference: Arc<Inference>) -> Self {
        Self {
            memory_policy: None,
            proof: proofs::Proof::new(inference.position.outcome()),
            inference,
            visits: 0,
            value_sum: 0.0,
            forced_visits: 0,
            children: Vec::new(),
        }
    }
    fn bytes(&self) -> usize {
        self.inference.bytes()
            + self.memory_policy.as_ref().map_or(0, |p| p.bytes())
            + self.children.capacity() * std::mem::size_of::<Option<Box<Node>>>()
            + self
                .children
                .iter()
                .flatten()
                .map(|n| std::mem::size_of::<Node>() + n.bytes())
                .sum::<usize>()
    }
}
struct Cache {
    entries: HashMap<u64, Vec<Arc<Inference>>>,
    fifo: VecDeque<u64>,
    count: usize,
    limit: usize,
    hits: usize,
    evaluations: usize,
    bytes: Arc<AtomicUsize>,
}
impl Cache {
    fn key(position: &Position) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        position.to_move().hash(&mut h);
        position.completed_turns().hash(&mut h);
        (position.phase() == paisho_core::TurnPhase::HarmonyBonus).hash(&mut h);
        for tile in position.board().occupied() {
            tile.hash(&mut h);
        }
        for p in [Player::Host, Player::Guest] {
            for k in STANDARD_TILE_KINDS {
                position.reserve(p).count(k).hash(&mut h);
            }
        }
        h.finish()
    }
    fn get(&mut self, position: Position, model: &MicroModel) -> Arc<Inference> {
        let key = Self::key(&position);
        if let Some(value) = self
            .entries
            .get(&key)
            .and_then(|bucket| bucket.iter().find(|v| v.position == position))
        {
            self.hits += 1;
            return value.clone();
        }
        let state = micro_state_features(&position);
        let embedding = model.embed(&state);
        let inference = Arc::new(Inference {
            accounting: (self.limit > 0).then(|| self.bytes.clone()),
            resident: AtomicBool::new(self.limit > 0),
            position,
            state,
            embedding,
            policy: OnceLock::new(),
        });
        self.evaluations += 1;
        if self.limit > 0 {
            while self.count >= self.limit {
                if let Some(key) = self.fifo.pop_front() {
                    if let Some(bucket) = self.entries.remove(&key) {
                        self.count -= bucket.len();
                        for old in bucket {
                            old.resident.store(false, Ordering::Relaxed);
                            self.bytes.fetch_sub(old.bytes(), Ordering::Relaxed);
                        }
                    }
                } else {
                    break;
                }
            }
            if !self.entries.contains_key(&key) {
                self.fifo.push_back(key);
            }
            self.entries.entry(key).or_default().push(inference.clone());
            self.count += 1;
            self.bytes.fetch_add(inference.bytes(), Ordering::Relaxed);
        }
        inference
    }
    fn clear(&mut self) {
        for inference in self.entries.values().flatten() {
            inference.resident.store(false, Ordering::Relaxed);
        }
        self.bytes.store(0, Ordering::Relaxed);
        self.entries.clear();
        self.fifo.clear();
        self.count = 0;
    }
}

#[derive(Clone, Debug)]
pub struct MicroSearchReport {
    pub selected_index: usize,
    /// Explicit improved policy (Gumbel) or visit policy (PUCT).
    pub policy_target: Vec<f64>,
    pub search_priors: Vec<f64>,
    pub actions: Vec<Action>,
    pub priors: Vec<f64>,
    /// Total visits include inherited search, `new_visits` are this decision only.
    pub visits: Vec<usize>,
    pub new_visits: Vec<usize>,
    pub new_forced_visits: Vec<usize>,
    /// Visits removed from the target only; underlying search stays unchanged.
    pub pruned_visits: Vec<usize>,
    /// Exact outcomes relative to the root player, never network predictions.
    pub proven_value: Option<i8>,
    pub proven_action_values: Vec<Option<i8>>,
    pub network_value: f64,
    pub values: Vec<f64>,
    pub simulations: usize,
    pub inherited_visits: usize,
    pub inference_cache_hits: usize,
    pub inference_evaluations: usize,
    pub retained_bytes: usize,
    pub memory_reset: bool,
    pub state: [f64; MICRO_INPUTS],
    pub action_features: Arc<Vec<[f64; MICRO_ACTION_INPUTS]>>,
}
impl MicroSearchReport {
    /// Using total visits is intentional: inherited visits belong to the same
    /// immutable model. Both counts are exposed so archives can audit this choice.
    pub fn example(&self, terminal_value: f64, policy_weight: f64) -> Result<MicroExample, String> {
        let total: usize = self.visits.iter().sum();
        if total == 0 || (self.simulations == 0 && self.proven_value.is_none()) {
            return Err("no root visits for policy target".into());
        }
        let ex = MicroExample {
            sequence_source: 0,
            state: self.state,
            actions: self.action_features.to_vec(),
            policy: self.policy_target.clone(),
            value: terminal_value,
            policy_weight,
        };
        ex.validate()?;
        Ok(ex)
    }
}

pub struct MicroMctsSession {
    model: Arc<MicroModel>,
    root: Option<Node>,
    cache: Cache,
    exploration: f64,
    maximum_depth: usize,
    memory_limit: usize,
}
impl MicroMctsSession {
    pub fn new(model: Arc<MicroModel>) -> Self {
        Self {
            model,
            root: None,
            cache: Cache {
                entries: HashMap::new(),
                fifo: VecDeque::new(),
                count: 0,
                limit: 65536,
                hits: 0,
                evaluations: 0,
                bytes: Arc::new(AtomicUsize::new(0)),
            },
            exploration: 1.5,
            maximum_depth: 96,
            memory_limit: 512 * 1024 * 1024,
        }
    }
    pub fn set_limits(&mut self, cache_entries: usize, memory_bytes: usize) {
        self.cache.limit = cache_entries;
        self.memory_limit = memory_bytes;
        self.enforce_memory();
        if self.cache.count > cache_entries {
            self.cache.clear();
        }
    }
    pub fn retained_visits(&self) -> usize {
        self.root.as_ref().map_or(0, |n| n.visits)
    }
    pub fn retained_bytes(&self) -> usize {
        // Conservative estimate: shared entries can be counted twice. The bound
        // applies between searches; one search's scratch/growth is transient.
        self.root
            .as_ref()
            .map_or(0, |n| std::mem::size_of::<Node>() + n.bytes())
            + self.cache.bytes.load(Ordering::Relaxed)
    }
    fn enforce_memory(&mut self) -> bool {
        if self.retained_bytes() > self.memory_limit {
            self.root = None;
            self.cache.clear();
            true
        } else {
            false
        }
    }
    /// Called after EVERY actual action. Preserve all descendants on the played
    /// branch; discarded siblings' inference remains in the bounded cache.
    pub fn advance(&mut self, action: Action) -> Result<bool, String> {
        let Some(mut root) = self.root.take() else {
            return Ok(false);
        };
        // An unsearched leaf has no branches to recover. Do not generate and
        // score all its legal actions merely to advance the actual played move.
        if let Some(policy) = root.inference.policy.get() {
            let policy = policy.as_ref().map_err(Clone::clone)?;
            let index = policy
                .actions
                .iter()
                .position(|a| *a == action)
                .ok_or("advance requires a legal action")?;
            if let Some(child) = root.children.get_mut(index).and_then(Option::take) {
                self.root = Some(*child);
                return Ok(true);
            }
        }
        let mut position = root.inference.position.clone();
        position.apply(action).map_err(|e| e.to_string())?;
        self.root = Some(Node::new(self.cache.get(position, &self.model)));
        Ok(false)
    }
    pub fn search_until(
        &mut self,
        position: &Position,
        simulations: usize,
        deadline: Option<Instant>,
    ) -> Result<MicroSearchReport, String> {
        self.search_with_options(
            position,
            simulations,
            deadline,
            MicroSearchOptions::default(),
        )
    }
    pub fn search_with_options(
        &mut self,
        position: &Position,
        simulations: usize,
        deadline: Option<Instant>,
        options: MicroSearchOptions,
    ) -> Result<MicroSearchReport, String> {
        options.validate()?;
        if simulations == 0 || position.outcome() != GameOutcome::Ongoing {
            return Err("PUCT requires a live position and positive budget".into());
        }
        let hits = self.cache.hits;
        let evaluations = self.cache.evaluations;
        if self
            .root
            .as_ref()
            .map_or(true, |r| r.inference.position != *position)
        {
            self.root = Some(Node::new(self.cache.get(position.clone(), &self.model)));
        }
        let root = self.root.as_mut().unwrap();
        let base = root.inference.policy()?;
        if root.memory_policy.is_none() && self.model.sequence_memory().is_some() && !base.actions.is_empty() {
            let priors = self.model.memory_priors(&root.inference.state, &base.features, &base.priors, 0)?;
            root.memory_policy = Some(Arc::new(Policy {actions:base.actions.clone(),features:base.features.clone(),priors,log_priors:OnceLock::new()}));
        }
        let policy = root.memory_policy.clone().unwrap_or(base);
        if policy.actions.is_empty() {
            return Err("PUCT root has no legal action".into());
        }
        let inherited_visits = root.visits;
        let before: Vec<_> = (0..policy.actions.len())
            .map(|i| {
                root.children
                    .get(i)
                    .and_then(Option::as_ref)
                    .map_or(0, |c| c.visits)
            })
            .collect();
        let search_priors = exploration::noisy_priors(&policy.priors, options);
        let before_forced: Vec<_> = if options.forced_playout_strength > 0.0 {
            (0..policy.actions.len())
                .map(|i| {
                    root.children
                        .get(i)
                        .and_then(Option::as_ref)
                        .map_or(0, |c| c.forced_visits)
                })
                .collect()
        } else {
            vec![]
        };
        let gumbel = (options.mode == MicroSearchMode::Gumbel)
            .then(|| exploration::GumbelRoot::new(&policy.priors, simulations, options));
        let mut done = 0;
        for step in 0..simulations {
            if deadline.is_some_and(|t| Instant::now() >= t)
                || (options.proof_search && root.proof.outcome.is_some())
            {
                break;
            }
            let forced = gumbel
                .as_ref()
                .map(|g| g.select(root, &policy.priors, &before, Some(step)));
            simulate_mode(
                root,
                &mut self.cache,
                &self.model,
                self.exploration,
                self.maximum_depth,
                0,
                forced,
                Some(&search_priors),
                options.mode,
                options.forced_playout_strength,
                options.proof_search,
            )?;
            done += 1;
        }
        let visits: Vec<_> = (0..policy.actions.len())
            .map(|i| {
                root.children
                    .get(i)
                    .and_then(Option::as_ref)
                    .map_or(0, |c| c.visits)
            })
            .collect();
        let proven_value = options
            .proof_search
            .then(|| root.proof.value(position.to_move()))
            .flatten();
        let proven_action_values: Vec<_> = if options.proof_search {
            (0..policy.actions.len())
                .map(|i| {
                    root.children
                        .get(i)
                        .and_then(Option::as_ref)
                        .and_then(|c| c.proof.value(position.to_move()))
                })
                .collect()
        } else {
            vec![]
        };
        let values: Vec<_> = (0..policy.actions.len())
            .map(|i| {
                root.children
                    .get(i)
                    .and_then(Option::as_ref)
                    .map_or(root.inference.value(), |c| {
                        if options.proof_search {
                            if let Some(v) = c.proof.value(position.to_move()) {
                                return v as f64;
                            }
                        }
                        let sign = if c.inference.position.to_move() == position.to_move() {
                            1.0
                        } else {
                            -1.0
                        };
                        sign * c.value_sum / c.visits.max(1) as f64
                    })
            })
            .collect();
        let mut selected_index = if let Some(g) = &gumbel {
            g.select(root, &policy.priors, &before, None)
        } else {
            (0..policy.actions.len())
                .max_by(|a, b| {
                    visits[*a]
                        .cmp(&visits[*b])
                        .then_with(|| values[*a].total_cmp(&values[*b]))
                        .then_with(|| policy.priors[*a].total_cmp(&policy.priors[*b]))
                        .then_with(|| b.cmp(a))
                })
                .unwrap()
        };
        let allowed = proofs::allowed(&proven_action_values, proven_value);
        if !allowed.is_empty() && !allowed[selected_index] {
            selected_index = (0..visits.len())
                .filter(|i| allowed[*i])
                .max_by(|a, b| {
                    visits[*a]
                        .cmp(&visits[*b])
                        .then_with(|| values[*a].total_cmp(&values[*b]))
                        .then_with(|| policy.priors[*a].total_cmp(&policy.priors[*b]))
                        .then_with(|| b.cmp(a))
                })
                .unwrap();
        }
        let corrected_visits = (options.forced_playout_strength > 0.0).then(|| {
            forced::prune(
                &visits,
                &values,
                &search_priors,
                selected_index,
                self.exploration * ((root.visits + 1) as f64).sqrt(),
                options.forced_playout_strength,
            )
        });
        let target_visits = corrected_visits.as_deref().unwrap_or(&visits);
        let mut policy_target = if gumbel.is_some() {
            exploration::improved_policy(root, &policy.priors)
        } else {
            let total: usize = target_visits.iter().sum();
            target_visits
                .iter()
                .map(|n| *n as f64 / total.max(1) as f64)
                .collect()
        };
        proofs::mask(&mut policy_target, &allowed, selected_index);
        if policy_target.iter().sum::<f64>() == 0.0 {
            policy_target[selected_index] = 1.0;
        }
        let mut report = MicroSearchReport {
            proven_value,
            proven_action_values,
            network_value: root.inference.embedding.value,
            policy_target,
            search_priors,
            selected_index,
            actions: policy.actions.clone(),
            priors: policy.priors.clone(),
            new_visits: visits.iter().zip(before).map(|(a, b)| a - b).collect(),
            new_forced_visits: before_forced
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    root.children
                        .get(i)
                        .and_then(Option::as_ref)
                        .map_or(0, |c| c.forced_visits)
                        - b
                })
                .collect(),
            pruned_visits: if corrected_visits.is_some() {
                visits
                    .iter()
                    .zip(target_visits)
                    .enumerate()
                    .map(|(i, (a, b))| {
                        if !allowed.is_empty() && !allowed[i] {
                            *a
                        } else {
                            a - b
                        }
                    })
                    .collect()
            } else {
                vec![]
            },
            visits,
            values,
            simulations: done,
            inherited_visits,
            inference_cache_hits: self.cache.hits - hits,
            inference_evaluations: self.cache.evaluations - evaluations,
            retained_bytes: 0,
            memory_reset: false,
            state: root.inference.state,
            action_features: policy.features.clone(),
        };
        report.retained_bytes = self.retained_bytes();
        if report.retained_bytes > self.memory_limit {
            self.root = None;
            self.cache.clear();
            report.memory_reset = true;
        }
        Ok(report)
    }
}

#[cfg(test)]
fn simulate(
    node: &mut Node,
    cache: &mut Cache,
    model: &MicroModel,
    cpuct: f64,
    max_depth: usize,
    depth: usize,
) -> Result<f64, String> {
    simulate_mode(
        node,
        cache,
        model,
        cpuct,
        max_depth,
        depth,
        None,
        None,
        MicroSearchMode::Puct,
        0.0,
        false,
    )
}
fn simulate_mode(
    node: &mut Node,
    cache: &mut Cache,
    model: &MicroModel,
    cpuct: f64,
    max_depth: usize,
    depth: usize,
    forced: Option<usize>,
    root_priors: Option<&[f64]>,
    mode: MicroSearchMode,
    forced_strength: f64,
    proof_search: bool,
) -> Result<f64, String> {
    let value = if proof_search && node.proof.outcome.is_some() {
        node.proof.value(node.inference.position.to_move()).unwrap() as f64
    } else if node.inference.position.outcome() != GameOutcome::Ongoing || depth >= max_depth {
        node.inference.value()
    } else {
        let policy = node.inference.policy()?;
        if policy.actions.is_empty() {
            // No rule outcome is invented: a blocked search leaf uses its value.
            let value = node.inference.value();
            node.visits += 1;
            node.value_sum += value;
            return Ok(value);
        }
        if node.children.is_empty() {
            node.children.resize_with(policy.actions.len(), || None);
        }
        let scale = cpuct * ((node.visits + 1) as f64).sqrt();
        let player = node.inference.position.to_move();
        let first_play_value = node.inference.value();
        let root_total = if forced_strength > 0.0 {
            node.children.iter().flatten().map(|c| c.visits).sum()
        } else {
            0
        };
        let is_forced = |i: usize| {
            forced_strength > 0.0
                && forced::needs_visit(
                    node.children
                        .get(i)
                        .and_then(Option::as_ref)
                        .map_or(0, |c| c.visits),
                    root_priors.unwrap_or(&policy.priors)[i],
                    root_total,
                    forced_strength,
                )
        };
        let score = |i: usize| {
            if proof_search
                && node.children[i]
                    .as_ref()
                    .and_then(|c| c.proof.value(player))
                    == Some(-1)
            {
                return f64::NEG_INFINITY;
            }
            if is_forced(i) {
                return f64::INFINITY;
            }
            let (q, n) =
                node.children
                    .get(i)
                    .and_then(Option::as_ref)
                    .map_or((first_play_value, 0), |c| {
                        let sign = if c.inference.position.to_move() == player {
                            1.0
                        } else {
                            -1.0
                        };
                        (sign * c.value_sum / c.visits.max(1) as f64, c.visits)
                    });
            q + scale * root_priors.unwrap_or(&policy.priors)[i] / (1 + n) as f64
        };
        let mut i = if let Some(forced) = forced {
            forced
        } else if mode == MicroSearchMode::Gumbel {
            exploration::interior(node, &policy.priors)
        } else {
            // Evaluate each PUCT candidate only once, lowest-index tie break.
            let mut i = 0;
            let mut best = score(0);
            for candidate in 1..policy.actions.len() {
                let value = score(candidate);
                if value.total_cmp(&best).is_gt() {
                    i = candidate;
                    best = value;
                }
            }
            i
        };
        // Forced exploration/Gumbel must not override a proven losing edge.
        if proof_search && score(i) == f64::NEG_INFINITY {
            if let Some(safe) = (0..policy.actions.len())
                .filter(|j| score(*j) != f64::NEG_INFINITY)
                .max_by(|a, b| score(*a).total_cmp(&score(*b)).then_with(|| b.cmp(a)))
            {
                i = safe;
            }
        }
        let previously_proven = node.children[i]
            .as_ref()
            .is_some_and(|c| c.proof.outcome.is_some());
        let forced_playout = forced.is_none() && is_forced(i);
        let sampled = if let Some(child) = node.children.get_mut(i).and_then(Option::as_mut) {
            child.forced_visits += usize::from(forced_playout);
            let sign = if child.inference.position.to_move() == player {
                1.0
            } else {
                -1.0
            };
            sign * simulate_mode(
                child,
                cache,
                model,
                cpuct,
                max_depth,
                depth + 1,
                None,
                None,
                mode,
                0.0,
                proof_search,
            )?
        } else {
            let mut next = node.inference.position.clone();
            next.apply(policy.actions[i]).map_err(|e| e.to_string())?;
            let mut child = Node::new(cache.get(next, model));
            let child_value = child.inference.value();
            child.visits = 1;
            child.value_sum = child_value;
            let sign = if child.inference.position.to_move() == player {
                1.0
            } else {
                -1.0
            };
            node.children[i] = Some(Box::new(child));
            sign * child_value
        };
        if !previously_proven {
            if let Some(outcome) = node.children[i].as_ref().and_then(|c| c.proof.outcome) {
                node.proof
                    .child_solved(outcome, player, policy.actions.len());
            }
        }
        if proof_search {
            node.proof.value(player).map_or(sampled, |v| v as f64)
        } else {
            sampled
        }
    };
    node.visits += 1;
    node.value_sum += value;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use paisho_core::GameRecord;
    #[test]
    fn cached_memory_matches_full_audit_after_lazy_policy_eviction_and_clear() {
        let model = Arc::new(MicroModel::seeded(8));
        let mut session = MicroMctsSession::new(model);
        session.set_limits(32, 512 * 1024 * 1024);
        let mut position = Position::from_standard_setup(paisho_core::StandardSetup::balanced(
            paisho_core::BASIC_FLOWERS[0],
        ));
        for _ in 0..12 {
            let r = session
                .search_with_options(
                    &position,
                    64,
                    None,
                    MicroSearchOptions {
                        mode: MicroSearchMode::Gumbel,
                        ..Default::default()
                    },
                )
                .unwrap();
            let scanned: usize = session
                .cache
                .entries
                .values()
                .flatten()
                .map(|x| x.bytes())
                .sum();
            assert_eq!(session.cache.bytes.load(Ordering::Relaxed), scanned);
            position.apply(r.actions[r.selected_index]).unwrap();
            session.advance(r.actions[r.selected_index]).unwrap();
            if position.outcome() != GameOutcome::Ongoing {
                break;
            }
        }
        session.set_limits(0, usize::MAX);
        assert_eq!(session.cache.bytes.load(Ordering::Relaxed), 0);
        if position.outcome() == GameOutcome::Ongoing {
            session.search_until(&position, 8, None).unwrap();
        }
        assert_eq!(session.cache.bytes.load(Ordering::Relaxed), 0);
    }
    #[test]
    fn blocked_leaf_keeps_prediction_and_does_not_fabricate_outcome() {
        let model = MicroModel::seeded(4);
        let p = Position::from_standard_setup(paisho_core::StandardSetup::balanced(
            paisho_core::BASIC_FLOWERS[0],
        ));
        let state = micro_state_features(&p);
        let embedding = model.embed(&state);
        let expected = embedding.value;
        let inference = Arc::new(Inference {
            accounting: None,
            resident: AtomicBool::new(false),
            position: p,
            state,
            embedding,
            policy: OnceLock::new(),
        });
        inference
            .policy
            .set(Ok(Arc::new(Policy {
                actions: vec![],
                features: Arc::new(vec![]),
                priors: vec![],
                log_priors: OnceLock::new(),
            })))
            .ok()
            .unwrap();
        let mut node = Node::new(inference);
        let mut session = MicroMctsSession::new(Arc::new(model.clone()));
        assert_eq!(
            simulate(&mut node, &mut session.cache, &model, 1.5, 96, 0).unwrap(),
            expected
        );
        assert_eq!(node.visits, 1);
        assert_eq!(node.inference.position.outcome(), GameOutcome::Ongoing);
        session.root = Some(node);
        let root = session.root.as_ref().unwrap().inference.position.clone();
        assert_eq!(
            session.search_until(&root, 64, None).unwrap_err(),
            "PUCT root has no legal action"
        );
    }
    #[test]
    fn actual_bonus_and_player_switch_edges_back_up_without_automatic_negation() {
        let record: GameRecord = include_str!("../../tests/fixtures/site_bot_v1_ring_finish.psr")
            .parse()
            .unwrap();
        let mut position = record.initial_position();
        let mut saw_same = false;
        let mut saw_switch = false;
        let mut saw_terminal = false;
        for action in record.actions() {
            let mut w = vec![0.0; MICRO_PARAMETERS];
            w[VALUE_B] = 0.3_f64.atanh();
            let mut s = MicroMctsSession::new(Arc::new(MicroModel::from_parameters(w).unwrap()));
            let inference = s.cache.get(position.clone(), &s.model);
            // Force the real recorded legal edge, isolating backup semantics
            // from the expressivity of the policy's local action descriptor.
            let actions = legal_actions(&position);
            let index = actions.iter().position(|a| a == action).unwrap();
            let features = Arc::new(
                actions
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect(),
            );
            let mut priors = vec![0.0; actions.len()];
            priors[index] = 1.0;
            assert!(inference
                .policy
                .set(Ok(Arc::new(Policy {
                    actions,
                    features,
                    priors,
                    log_priors: OnceLock::new(),
                })))
                .is_ok());
            s.cache
                .bytes
                .fetch_add(inference.policy().unwrap().bytes(), Ordering::Relaxed);
            s.root = Some(Node::new(inference));
            let r = s.search_until(&position, 1, None).unwrap();
            assert_eq!(r.actions[r.selected_index], *action);
            let mut child = position.clone();
            child.apply(*action).unwrap();
            let same = child.to_move() == position.to_move();
            let value = match child.outcome() {
                GameOutcome::Ongoing => {
                    if same {
                        0.3
                    } else {
                        -0.3
                    }
                }
                GameOutcome::Win(p) => {
                    if p == position.to_move() {
                        1.0
                    } else {
                        -1.0
                    }
                }
                GameOutcome::Draw => 0.0,
            };
            assert!((r.values[index] - value).abs() < 1e-12);
            saw_same |= same;
            saw_switch |= !same;
            saw_terminal |= child.outcome() != GameOutcome::Ongoing;
            position = child;
        }
        assert!(saw_same && saw_switch && saw_terminal);
    }
}
