use core::fmt;
use std::path::Path;
use std::sync::OnceLock;

use paisho_core::RuleProfileId;
use rayon::prelude::*;

use crate::{
    PolicyTargetKindV1, ReplayDecisionV1, ReplayDigestV1, ReplayShardReferenceV1, ReplayShardV1,
    ReplaySnapshotV1, ReplaySnapshotV1Error, ReplayTrainingExampleV1, ReplayValidationError,
};

#[derive(Clone, Copy, Debug)]
struct ExampleLocation {
    shard: usize,
    game: usize,
    decision: usize,
}

#[derive(Clone, Debug)]
pub struct ReplayDatasetV1 {
    snapshot_digest: ReplayDigestV1,
    shards: Vec<ReplayShardV1>,
    locations: Vec<ExampleLocation>,
    materialized: OnceLock<Vec<ReplayTrainingExampleV1>>,
}

impl ReplayDatasetV1 {
    pub fn from_snapshot(
        snapshot: &ReplaySnapshotV1,
        directory: &Path,
    ) -> Result<Self, ReplayDatasetV1Error> {
        Self::from_snapshot_selected(snapshot, directory, |_| true)
    }

    /// Select only stochastic behavior produced by the exact network snapshot
    /// being optimized. MCTS, teachers, historical opponents and one-hot smoke
    /// fixtures remain readable but cannot silently enter policy-gradient data.
    pub fn from_snapshot_for_behavior(
        snapshot: &ReplaySnapshotV1,
        directory: &Path,
        producer: ReplayDigestV1,
    ) -> Result<Self, ReplayDatasetV1Error> {
        Self::from_snapshot_selected(snapshot, directory, |decision| {
            decision.policy().kind() == PolicyTargetKindV1::Behavior
                && decision.policy().producer() == producer
        })
    }

    fn from_snapshot_selected(
        snapshot: &ReplaySnapshotV1,
        directory: &Path,
        include: impl Fn(&ReplayDecisionV1) -> bool,
    ) -> Result<Self, ReplayDatasetV1Error> {
        let shards = snapshot
            .load_verified_shards(directory)
            .map_err(ReplayDatasetV1Error::Snapshot)?;
        Self::from_shards_selected(snapshot.digest(), shards, include)
    }

    /// Consume the already validated shard just produced by live actors. The
    /// snapshot identity is derived from its actual contents, never caller-supplied.
    /// Disk recovery continues to use the fully verified snapshot constructors.
    pub fn from_shard_for_behavior(
        shard: ReplayShardV1,
        file_name: &str,
        producer: ReplayDigestV1,
    ) -> Result<Self, ReplayDatasetV1Error> {
        let reference = ReplayShardReferenceV1::from_shard(file_name, &shard)
            .map_err(ReplayDatasetV1Error::Snapshot)?;
        let snapshot =
            ReplaySnapshotV1::new(vec![reference]).map_err(ReplayDatasetV1Error::Snapshot)?;
        Self::from_shards_selected(snapshot.digest(), vec![shard], |decision| {
            decision.policy().kind() == PolicyTargetKindV1::Behavior
                && decision.policy().producer() == producer
        })
    }

    fn from_shards_selected(
        snapshot_digest: ReplayDigestV1,
        shards: Vec<ReplayShardV1>,
        include: impl Fn(&ReplayDecisionV1) -> bool,
    ) -> Result<Self, ReplayDatasetV1Error> {
        let mut locations = Vec::new();
        for (shard_index, shard) in shards.iter().enumerate() {
            for (game_index, game) in shard.games().iter().enumerate() {
                for (decision_index, decision) in game.decisions().iter().enumerate() {
                    if !include(decision) {
                        continue;
                    }
                    locations.push(ExampleLocation {
                        shard: shard_index,
                        game: game_index,
                        decision: decision_index,
                    });
                }
            }
        }
        if locations.is_empty() {
            return Err(ReplayDatasetV1Error::Empty);
        }
        u64::try_from(locations.len()).map_err(|_| ReplayDatasetV1Error::TooManyExamples)?;
        Ok(Self {
            snapshot_digest,
            shards,
            locations,
            materialized: OnceLock::new(),
        })
    }

    pub const fn snapshot_digest(&self) -> ReplayDigestV1 {
        self.snapshot_digest
    }

    /// All shards were verified against one snapshot profile at construction.
    /// A dataset is non-empty, so its first shard always carries that profile.
    pub fn rule_profile(&self) -> RuleProfileId {
        self.shards[0].rule_profile()
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Whether every selected decision already carries the frozen actor value
    /// needed by terminal PPO. Legacy or mixed corpora return false and keep
    /// using the inference compatibility path.
    pub fn has_complete_behavior_values(&self) -> bool {
        self.locations.iter().all(|location| {
            self.shards[location.shard].games()[location.game].decisions()[location.decision]
                .behavior_value()
                .is_some()
        })
    }

    /// Materialize the selected dataset in canonical shard/game/decision order.
    /// This walks each game once, unlike random-access sampling which may need
    /// to reconstruct several independent prefixes of the same game.
    pub fn materialize_all(&self) -> Result<Vec<ReplayTrainingExampleV1>, ReplayValidationError> {
        if let Some(materialized) = self.materialized.get() {
            return Ok(materialized.clone());
        }
        self.materialize_all_uncached()
    }

    /// Reconstruct every selected example once, grouped by game and spread
    /// across the global Rayon pool. The cache is only an execution aid: it is
    /// derived from the verified immutable shards and is never checkpointed.
    pub fn preload(&self) -> Result<(), ReplayValidationError> {
        if self.materialized.get().is_none() {
            let materialized = self.materialize_all_uncached()?;
            let _ = self.materialized.set(materialized);
        }
        Ok(())
    }

    fn materialize_all_uncached(
        &self,
    ) -> Result<Vec<ReplayTrainingExampleV1>, ReplayValidationError> {
        let mut groups = Vec::<Vec<ExampleLocation>>::new();
        for &location in &self.locations {
            match groups.last_mut() {
                Some(group)
                    if group[0].shard == location.shard && group[0].game == location.game =>
                {
                    group.push(location);
                }
                _ => groups.push(vec![location]),
            }
        }
        let chunks = groups
            .into_par_iter()
            .map(|group| {
                let first = group[0];
                let game = &self.shards[first.shard].games()[first.game];
                let materialized = game.materialize_training_examples()?;
                Ok(group
                    .into_iter()
                    .map(|location| materialized[location.decision].clone())
                    .collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, ReplayValidationError>>()?;
        let examples = chunks.into_iter().flatten().collect::<Vec<_>>();
        debug_assert_eq!(examples.len(), self.locations.len());
        Ok(examples)
    }

    fn materialize_indices(
        &self,
        indices: &[usize],
    ) -> Result<Vec<ReplayTrainingExampleV1>, ReplayValidationError> {
        if let Some(materialized) = self.materialized.get() {
            return Ok(indices
                .par_iter()
                .map(|&index| materialized[index].clone())
                .collect());
        }
        indices
            .par_iter()
            .map(|&index| {
                let location = self.locations[index];
                self.shards[location.shard].games()[location.game]
                    .materialize_training_example(location.decision)
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplaySamplerStateV1 {
    snapshot_digest: ReplayDigestV1,
    seed: u64,
    next_replay_index: u64,
}

impl ReplaySamplerStateV1 {
    pub const fn new(snapshot_digest: ReplayDigestV1, seed: u64, next_replay_index: u64) -> Self {
        Self {
            snapshot_digest,
            seed,
            next_replay_index,
        }
    }

    pub const fn snapshot_digest(self) -> ReplayDigestV1 {
        self.snapshot_digest
    }

    pub const fn seed(self) -> u64 {
        self.seed
    }

    pub const fn next_replay_index(self) -> u64 {
        self.next_replay_index
    }
}

#[derive(Debug)]
pub struct ReplaySamplerV1<'dataset> {
    dataset: &'dataset ReplayDatasetV1,
    state: ReplaySamplerStateV1,
    cached_epoch: Option<u64>,
    order: Vec<usize>,
}

impl<'dataset> ReplaySamplerV1<'dataset> {
    pub fn new(dataset: &'dataset ReplayDatasetV1, seed: u64) -> Self {
        Self {
            dataset,
            state: ReplaySamplerStateV1::new(dataset.snapshot_digest(), seed, 0),
            cached_epoch: None,
            order: Vec::new(),
        }
    }

    pub fn resume(
        dataset: &'dataset ReplayDatasetV1,
        state: ReplaySamplerStateV1,
    ) -> Result<Self, ReplaySamplerV1Error> {
        if state.snapshot_digest != dataset.snapshot_digest() {
            return Err(ReplaySamplerV1Error::SnapshotMismatch {
                expected: state.snapshot_digest,
                actual: dataset.snapshot_digest(),
            });
        }
        Ok(Self {
            dataset,
            state,
            cached_epoch: None,
            order: Vec::new(),
        })
    }

    pub const fn state(&self) -> ReplaySamplerStateV1 {
        self.state
    }

    pub fn prepare_batch(
        &mut self,
        batch_size: usize,
    ) -> Result<ReplaySampleBatchV1, ReplaySamplerV1Error> {
        if batch_size == 0 {
            return Err(ReplaySamplerV1Error::ZeroBatchSize);
        }
        let batch_size_u64 =
            u64::try_from(batch_size).map_err(|_| ReplaySamplerV1Error::BatchTooLarge)?;
        let end = self
            .state
            .next_replay_index
            .checked_add(batch_size_u64)
            .ok_or(ReplaySamplerV1Error::ReplayIndexOverflow)?;
        let dataset_size = self.dataset.len() as u64;
        let mut indices = Vec::with_capacity(batch_size);
        for replay_index in self.state.next_replay_index..end {
            let epoch = replay_index / dataset_size;
            let offset = (replay_index % dataset_size) as usize;
            self.prepare_epoch(epoch);
            indices.push(self.order[offset]);
        }
        let examples = self
            .dataset
            .materialize_indices(&indices)
            .map_err(ReplaySamplerV1Error::Validation)?;
        Ok(ReplaySampleBatchV1 {
            sampler_state: self.state,
            next_replay_index: end,
            examples,
        })
    }

    pub fn commit_batch(
        &mut self,
        batch: &ReplaySampleBatchV1,
    ) -> Result<(), ReplaySamplerV1Error> {
        if batch.sampler_state != self.state {
            return Err(ReplaySamplerV1Error::PreparedBatchStateMismatch {
                expected: self.state,
                actual: batch.sampler_state,
            });
        }
        self.state.next_replay_index = batch.next_replay_index;
        Ok(())
    }

    fn prepare_epoch(&mut self, epoch: u64) {
        if self.cached_epoch == Some(epoch) {
            return;
        }
        self.order = (0..self.dataset.len()).collect();
        let mut rng = SamplerRng::new(self.state.seed ^ epoch.wrapping_mul(0xD1B5_4A32_D192_ED03));
        for upper in (1..self.order.len()).rev() {
            let other = rng.bounded(upper + 1);
            self.order.swap(upper, other);
        }
        self.cached_epoch = Some(epoch);
    }
}

#[derive(Clone, Debug)]
pub struct ReplaySampleBatchV1 {
    sampler_state: ReplaySamplerStateV1,
    next_replay_index: u64,
    examples: Vec<ReplayTrainingExampleV1>,
}

impl ReplaySampleBatchV1 {
    pub const fn snapshot_digest(&self) -> ReplayDigestV1 {
        self.sampler_state.snapshot_digest
    }

    pub const fn start_replay_index(&self) -> u64 {
        self.sampler_state.next_replay_index
    }

    pub const fn next_replay_index(&self) -> u64 {
        self.next_replay_index
    }

    pub fn examples(&self) -> &[ReplayTrainingExampleV1] {
        &self.examples
    }

    pub fn into_examples(self) -> Vec<ReplayTrainingExampleV1> {
        self.examples
    }
}

struct SamplerRng {
    state: u64,
}

impl SamplerRng {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn bounded(&mut self, upper: usize) -> usize {
        let bound = upper as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

#[derive(Debug)]
pub enum ReplayDatasetV1Error {
    Empty,
    TooManyExamples,
    Snapshot(ReplaySnapshotV1Error),
}

impl fmt::Display for ReplayDatasetV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("a replay dataset cannot be empty"),
            Self::TooManyExamples => formatter.write_str("replay dataset exceeds UInt64 indexing"),
            Self::Snapshot(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for ReplayDatasetV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Snapshot(source) => Some(source),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ReplaySamplerV1Error {
    SnapshotMismatch {
        expected: ReplayDigestV1,
        actual: ReplayDigestV1,
    },
    ZeroBatchSize,
    BatchTooLarge,
    ReplayIndexOverflow,
    PreparedBatchStateMismatch {
        expected: ReplaySamplerStateV1,
        actual: ReplaySamplerStateV1,
    },
    Validation(ReplayValidationError),
}

impl fmt::Display for ReplaySamplerV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SnapshotMismatch { expected, actual } => write!(
                formatter,
                "sampler snapshot {expected} does not match dataset snapshot {actual}"
            ),
            Self::ZeroBatchSize => formatter.write_str("replay batch size must be positive"),
            Self::BatchTooLarge => formatter.write_str("replay batch size exceeds UInt64"),
            Self::ReplayIndexOverflow => formatter.write_str("replay sampler index overflow"),
            Self::PreparedBatchStateMismatch { expected, actual } => write!(
                formatter,
                "prepared replay batch starts from sampler state {actual:?}, but the current state is {expected:?}"
            ),
            Self::Validation(source) => source.fmt(formatter),
        }
    }
}

impl std::error::Error for ReplaySamplerV1Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Validation(source) => Some(source),
            _ => None,
        }
    }
}
