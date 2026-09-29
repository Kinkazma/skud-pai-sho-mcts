//! Retained PUCT with model-scoped, exact-state inference reuse. Search statistics
//! remain local to each tree edge; transpositions share only immutable inference.
use super::*;
mod certificates;
mod exploration;
mod forced;
mod guard;
mod proofs;
mod depth_probe;
pub use depth_probe::MicroDepthTrialStats;
#[cfg(test)]
mod solved_tests;
#[cfg(test)]
mod consistency_audit_tests;
#[cfg(test)]
mod successor_reuse_tests;
#[cfg(test)]
mod root_fpu_tests;
#[cfg(test)]
mod lazy_embedding_tests;
#[cfg(test)]
mod base_policy_measure_tests;
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
    #[cfg(test)]
    eager_embedding: bool,
    accounting: Option<Arc<AtomicUsize>>,
    resident: AtomicBool,
    position: Position,
    state: Vec<f64>,
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
    fn policy(&self, model: &MicroModel) -> Result<Arc<Policy>, String> {
        self.policy
            .get_or_init(|| {
                let complete;
                #[cfg(test)]
                let embedding=if self.eager_embedding {&self.embedding} else {complete=model.embed_policy_for_search(&self.state,&self.embedding);&complete};
                #[cfg(not(test))]
                let embedding={complete=model.embed_policy_for_search(&self.state,&self.embedding);&complete};
                let actions = legal_actions(&self.position);
                // Fuse extraction/scoring for both policy schemas. Cache the
                // resulting priors per node and amortize pool dispatch over at
                // least 256 actions per job; independent games share the pool.
                let score = |action: &Action| {
                    let features = micro_action_features(&self.position, *action);
                    let logit = MicroModel::logit(embedding, &features);
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
            + self.state.capacity() * 8
            + self.embedding.residual.as_ref().map_or(0, |r| r.bytes())
            + self
                .policy
                .get()
                .and_then(|p| p.as_ref().ok())
                .map_or(0, |p| p.bytes())
    }
}
struct Node {
    tactical_scanned: bool,
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
            tactical_scanned: false,
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
    #[cfg(test)]
    eager_embedding: bool,
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
        let state = model.state_features(&position);
        let embedding = {
            #[cfg(test)]
            if self.eager_embedding {model.embed(&state)} else {model.embed_value_for_search(&state)}
            #[cfg(not(test))]
            model.embed_value_for_search(&state)
        };
        let inference = Arc::new(Inference {
            #[cfg(test)]
            eager_embedding: self.eager_embedding,
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
    /// Exact root policy before value coupling, when computed during this search.
    /// Repeated reads of an already prepared root may leave this absent.
    pub raw_priors: Option<Arc<Vec<f64>>>,
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
    pub tactical_evaluations: usize,
    pub inherited_visits: usize,
    pub inference_cache_hits: usize,
    pub inference_evaluations: usize,
    pub retained_bytes: usize,
    pub memory_reset: bool,
    pub state: Vec<f64>,
    pub action_features: Arc<Vec<[f64; MICRO_ACTION_INPUTS]>>,
}
impl MicroSearchReport {
    /// Using total visits is intentional: inherited visits belong to the same
    /// immutable model. Both counts are exposed so archives can audit this choice.
    pub fn example(&self, terminal_value: f64, policy_weight: f64) -> Result<MicroExample, String> {
        let total: usize = self.visits.iter().sum();
        if self.proven_value.is_none() && (total == 0 || self.simulations == 0) {
            return Err("no root visits for policy target".into());
        }
        let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight:1.,
            sequence_source: 0,
            state: self.state.clone(),
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
    root_value_strength: f64,
    // Diagnostic-only opt-in; no node layout change and no extra inference.
    root_successor_fpu: bool,
    root_successor_values: Option<Vec<f64>>,
    #[cfg(test)]
    recompute_root_successors: bool,
    maximum_depth: usize,
    minimum_search_depth: usize,
    diagnostic_depth_floor: Option<usize>,
    diagnostic_depth_expansion_budget: Option<usize>,
    diagnostic_depth_stats: MicroDepthTrialStats,
    memory_limit: usize,
}
impl MicroMctsSession {
    pub fn new(model: Arc<MicroModel>) -> Self {
        Self {
            model,
            root: None,
            cache: Cache {
                #[cfg(test)]
                eager_embedding: false,
                entries: HashMap::new(),
                fifo: VecDeque::new(),
                count: 0,
                limit: 65536,
                hits: 0,
                evaluations: 0,
                bytes: Arc::new(AtomicUsize::new(0)),
            },
            exploration: 1.5,
            root_value_strength: 0.,
            root_successor_fpu: false,
            root_successor_values: None,
            #[cfg(test)]
            recompute_root_successors: false,
            maximum_depth: 96,
            minimum_search_depth: 0,
            diagnostic_depth_floor: None,
            diagnostic_depth_expansion_budget: None,
            diagnostic_depth_stats: MicroDepthTrialStats::default(),
            memory_limit: 512 * 1024 * 1024,
        }
    }
    /// Diagnostic only: use already computed, root-player successor values as
    /// the PUCT first-play estimate. No values become visits, proofs or targets.
    /// Missing values (beta=0 or already solved roots) retain the historical FPU.
    pub fn set_root_successor_fpu(&mut self, enabled: bool) {
        if self.root_successor_fpu != enabled {
            self.root = None;
            self.root_successor_values = None;
            self.root_successor_fpu = enabled;
        }
    }
    /// Values from the actual coupling pass, retained only by this opt-in root.
    /// This diagnostic field is separate from MicroSearchReport::values.
    pub fn root_successor_fpu_values(&self) -> Option<&[f64]> {
        self.root_successor_values.as_deref()
    }
    /// Reweight root priors by successor value; zero preserves the original search.
    pub fn set_root_value_strength(&mut self, beta: f64) -> Result<(), String> {
        if !beta.is_finite() || !(0.0..=16.0).contains(&beta) {
            return Err("invalid diagnostic root coupling".into());
        }
        if self.root_value_strength != beta { self.root = None; self.root_successor_values = None; }
        self.root_value_strength = beta;
        Ok(())
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
            + self.root_successor_values.as_ref().map_or(0, |q| q.capacity() * 8)
    }
    fn enforce_memory(&mut self) -> bool {
        if self.retained_bytes() > self.memory_limit {
            self.root = None;
            self.root_successor_values = None;
            self.cache.clear();
            true
        } else {
            false
        }
    }
    /// Called after EVERY actual action. Preserve all descendants on the played
    /// branch; discarded siblings' inference remains in the bounded cache.
    pub fn advance(&mut self, action: Action) -> Result<bool, String> {
        self.root_successor_values = None;
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
        if (self.minimum_search_depth > 0 || self.diagnostic_depth_floor.is_some()) && options.mode != MicroSearchMode::Puct {
            return Err("minimum search depth requires PUCT".into());
        }
        self.diagnostic_depth_stats = MicroDepthTrialStats::default();
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
            self.root_successor_values = None;
        }
        let root = self.root.as_mut().unwrap();
        let mut early_successors=vec![];
        let keep_successors=true;
        #[cfg(test)]
        let keep_successors=keep_successors && !self.recompute_root_successors;
        let early_guard = if self.root_value_strength != 0. && self.model.has_spatial() && options.proof_search {
            Some(if keep_successors {
                guard::discover_with_successors(root,&mut self.cache,&self.model,deadline,&mut early_successors)?
            } else {guard::discover(root, &mut self.cache, &self.model, deadline)?})
        } else {None};
        let base = root.inference.policy(&self.model)?;
        let mut raw_priors = None;
        if root.memory_policy.is_none()
            && (self.model.sequence_memory().is_some() || self.model.has_neural_memory() || self.root_value_strength != 0.)
            && !base.actions.is_empty()
        {
            self.root_successor_values = None;
            let mut priors =
                self.model
                    .memory_priors(&root.inference.state, &base.features, &base.priors, 0)?;
            raw_priors = Some(Arc::new(priors.clone()));
            if self.root_value_strength != 0. && root.proof.outcome.is_none() {
                let player = root.inference.position.to_move();
                let mut logits = Vec::with_capacity(base.actions.len());
                let mut successor_values = self.root_successor_fpu.then(|| Vec::with_capacity(base.actions.len()));
                for (i,(action, prior)) in base.actions.iter().zip(&priors).enumerate() {
                    // Discovery just constructed these exact boards. Keep its
                    // work local: cache admission/evaluation still happens here,
                    // in the original order, with the original search counters.
                    let next=if let Some(p)=early_successors.get_mut(i).and_then(Option::take) {p} else {
                        let mut p=root.inference.position.clone();p.apply(*action).map_err(|e|format!("{e:?}"))?;p
                    };
                    let v = self.cache.get(next, &self.model);
                    let q = if v.position.to_move() == player { v.value() } else { -v.value() };
                    if let Some(values) = &mut successor_values { values.push(q); }
                    logits.push(prior.max(1e-300).ln() + self.root_value_strength * q);
                }
                priors = micro_softmax(&logits)?;
                self.root_successor_values = successor_values;
            }
            root.memory_policy = Some(Arc::new(Policy {
                actions: base.actions.clone(),
                features: base.features.clone(),
                priors,
                log_priors: OnceLock::new(),
            }));
        }
        drop(early_successors);
        let policy = root.memory_policy.clone().unwrap_or(base);
        if policy.actions.is_empty() {
            return Err("PUCT root has no legal action".into());
        }
        let tactical_evaluations = if let Some(n)=early_guard {n} else if self.model.has_spatial() && options.proof_search {
            guard::discover(root, &mut self.cache, &self.model, deadline)?
        } else {
            0
        };
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
        let mut floor = if self.minimum_search_depth > 0 {
            Some(depth_probe::Floor::new(self.minimum_search_depth,
                simulations.checked_mul(self.minimum_search_depth).ok_or("depth expansion budget overflow")?))
        } else {
            self.diagnostic_depth_floor.map(|minimum| depth_probe::Floor::new(minimum, self.diagnostic_depth_expansion_budget.unwrap_or(simulations)))
        };
        let mut done = 0;
        for step in 0..simulations {
            if deadline.is_some_and(|t| paisho_platform::training_time::now() >= t)
                || floor.as_ref().is_some_and(|f| f.remaining == 0)
                || (options.proof_search && root.proof.outcome.is_some())
            {
                break;
            }
            let forced = gumbel.as_ref().map(|g| {
                g.select(
                    root,
                    &policy.priors,
                    &before,
                    Some(step),
                    options.proof_search && self.model.has_spatial(),
                )
            });
            simulate_mode_with_root_fpu(
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
                self.root_successor_values.as_deref(),
                floor.as_mut(),
            )?;
            done += 1;
        }
        if let Some(floor) = floor { self.diagnostic_depth_stats = floor.stats; }
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
                        let mean = c.value_sum / c.visits.max(1) as f64;
                        sign * if options.proof_search && self.model.has_spatial() {
                            c.proof.bound(mean)
                        } else {
                            mean
                        }
                    })
            })
            .collect();
        let mut selected_index = if let Some(g) = &gumbel {
            g.select(
                root,
                &policy.priors,
                &before,
                None,
                options.proof_search && self.model.has_spatial(),
            )
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
        let allowed = if options.proof_search && self.model.has_spatial() {
            proofs::decision_allowed(&proven_action_values, proven_value, &values, &visits)
        } else {
            proofs::allowed(&proven_action_values, proven_value)
        };
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
            exploration::improved_policy(
                root,
                &policy.priors,
                options.proof_search && self.model.has_spatial(),
            )
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
            raw_priors,
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
            tactical_evaluations,
            inherited_visits,
            inference_cache_hits: self.cache.hits - hits,
            inference_evaluations: self.cache.evaluations - evaluations,
            retained_bytes: 0,
            memory_reset: false,
            state: root.inference.state.clone(),
            action_features: policy.features.clone(),
        };
        report.retained_bytes = self.retained_bytes();
        if report.retained_bytes > self.memory_limit {
            self.root = None;
            self.root_successor_values = None;
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
#[cfg(test)]
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
    simulate_mode_with_root_fpu(node, cache, model, cpuct, max_depth, depth, forced,
        root_priors, mode, forced_strength, proof_search, None, None)
}
fn simulate_mode_with_root_fpu(
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
    root_fpu: Option<&[f64]>,
    mut floor: Option<&mut depth_probe::Floor>,
) -> Result<f64, String> {
    let value = if proof_search && node.proof.outcome.is_some() {
        if let Some(f) = floor.as_deref_mut() { f.leaf(depth, true); }
        node.proof.value(node.inference.position.to_move()).unwrap() as f64
    } else if node.inference.position.outcome() != GameOutcome::Ongoing || depth >= max_depth
        || floor.as_ref().is_some_and(|f| f.remaining == 0) {
        if let Some(f) = floor.as_deref_mut() { f.leaf(depth, node.inference.position.outcome() != GameOutcome::Ongoing); }
        node.inference.value()
    } else {
        let policy = node.inference.policy(model)?;
        if policy.actions.is_empty() {
            if let Some(f) = floor.as_deref_mut() { f.leaf(depth, false); }
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
                && node.children[i].as_ref().is_some_and(|c| {
                    if model.has_spatial() {
                        c.proof.outcome.is_some()
                    } else {
                        c.proof.value(player) == Some(-1)
                    }
                })
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
                    .map_or((root_fpu.map_or(first_play_value, |q| q[i]), 0), |c| {
                        let sign = if c.inference.position.to_move() == player {
                            1.0
                        } else {
                            -1.0
                        };
                        let mean = c.value_sum / c.visits.max(1) as f64;
                        let q = if proof_search && model.has_spatial() {
                            c.proof.bound(mean)
                        } else {
                            mean
                        };
                        (sign * q, c.visits)
                    });
            q + scale * root_priors.unwrap_or(&policy.priors)[i] / (1 + n) as f64
        };
        let mut i = if let Some(forced) = forced {
            forced
        } else if mode == MicroSearchMode::Gumbel {
            exploration::interior(node, &policy.priors, proof_search && model.has_spatial())
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
        // Forced exploration/Gumbel cannot reallocate work to solved V4 edges.
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
            sign * simulate_mode_with_root_fpu(
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
                None,
                floor.as_deref_mut(),
            )?
        } else {
            let mut next = node.inference.position.clone();
            next.apply(policy.actions[i]).map_err(|e| e.to_string())?;
            let mut child = Node::new(cache.get(next, model));
            if let Some(f) = floor.as_deref_mut() { f.remaining -= 1; f.stats.expansions += 1; }
            let prolong = floor.as_ref().is_some_and(|f| depth + 1 < f.minimum && f.remaining > 0)
                && depth + 1 < max_depth && child.inference.position.outcome() == GameOutcome::Ongoing;
            let child_value = if prolong {
                simulate_mode_with_root_fpu(&mut child, cache, model, cpuct, max_depth, depth + 1,
                    None, None, mode, 0., proof_search, None, floor.as_deref_mut())?
            } else {
                let value = child.inference.value();
                child.visits = 1; child.value_sum = value;
                if let Some(f) = floor.as_deref_mut() { f.leaf(depth + 1, child.inference.position.outcome() != GameOutcome::Ongoing); }
                value
            };
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
            node.proof.value(player).map_or_else(
                || {
                    if model.has_spatial() {
                        node.proof.bound(sampled)
                    } else {
                        sampled
                    }
                },
                |v| v as f64,
            )
        } else {
            sampled
        }
    };
    let value = if proof_search && model.has_spatial() {
        node.proof.bound(value)
    } else {
        value
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
    fn exported_raw_prior_precedes_value_coupling_and_retained_roots_use_fallback() {
        let model=Arc::new(MicroModel::seeded(19).with_neural_memory(21));
        let position=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BASIC_FLOWERS[0]));
        let mut session=MicroMctsSession::new(model.clone());
        session.set_root_value_strength(16.).unwrap();
        let report=session.search_with_options(&position,8,None,MicroSearchOptions::default()).unwrap();
        let base=micro_softmax(&MicroModel::logits(&model.embed(&report.state),&report.action_features)).unwrap();
        let raw=model.memory_priors(&report.state,&report.action_features,&base,0).unwrap();
        assert_eq!(report.raw_priors.as_ref().unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),raw.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert!(raw.iter().zip(&report.priors).any(|(a,b)|a.to_bits()!=b.to_bits()));
        let retained=session.search_with_options(&position,8,None,MicroSearchOptions::default()).unwrap();
        assert!(retained.raw_priors.is_none());
    }
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
            #[cfg(test)]
            eager_embedding: true,
            accounting: None,
            resident: AtomicBool::new(false),
            position: p,
            state: state.to_vec(),
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
                .fetch_add(inference.policy(&s.model).unwrap().bytes(), Ordering::Relaxed);
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
