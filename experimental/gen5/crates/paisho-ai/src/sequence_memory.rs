//! Immutable, portable sequence bank. Stable handcrafted keys, learned reader.
//! The bank is shared across actors; queries never touch the filesystem.
mod index;
mod usage;
pub use usage::SequenceUsage;
mod retention;
pub use retention::*;
mod codec;
mod spatial;
pub use index::*;
pub use spatial::SequenceGeometry;

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
pub const SEQUENCE_KEY: usize = 64;
pub const SEQUENCE_CHANNELS: usize = 12;
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SequenceMemorySpec {
    pub path: String,
    pub sha256: String,
}
#[derive(Clone, Debug, PartialEq)]
pub struct SequenceEntry {
    pub key: [f32; SEQUENCE_KEY],
    pub patterns: [[i8; 32]; 4],
    pub source: u64,
    pub game: u32,
    pub decision: u32,
    pub end_decision: u32,
    pub outcome: i8,
    pub phase: u8,
}
#[derive(Clone, Debug, Default)]
pub struct SequenceContext {
    pub patterns: Vec<[f64; 32]>,
    pub neighbors: Vec<usize>,
    pub confidence: f64,
}
impl SequenceContext {
    pub fn features(&self, a: &[f64; 32]) -> [f64; SEQUENCE_CHANNELS] {
        let mut x = [0.; SEQUENCE_CHANNELS];
        for (out, p) in x.iter_mut().zip(&self.patterns) {
            *out = a.iter().zip(p).map(|(a, b)| a * b).sum::<f64>() / 8.;
        }
        x
    }
}
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct QueryKey {
    vector: [u32; 64],
    phase: u8,
    excluded: u64,
    geometry: Option<SequenceGeometry>,
}
#[derive(Debug, Default)]
struct QueryCache {
    rows: HashMap<QueryKey, Arc<SequenceContext>>,
    fifo: VecDeque<QueryKey>,
}
#[derive(Debug)]
pub struct SequenceBank {
    pub spec: SequenceMemorySpec,
    pub entries: Vec<SequenceEntry>,
    pub centroids: Vec<[f32; 64]>,
    pub buckets: Vec<Vec<u32>>,
    pub games: usize,
    pub human_games: usize,
    pub probes: usize,
    pub neighbors: usize,
    spatial: Option<Vec<SequenceGeometry>>,
    source_ids: HashSet<u64>,
    cache: Vec<Mutex<QueryCache>>,
    uses: usage::Uses,
    pub queries: AtomicUsize,
    pub cache_hits: AtomicUsize,
    pub candidates: AtomicUsize,
}
impl PartialEq for SequenceBank {
    fn eq(&self, b: &Self) -> bool {
        self.probes == b.probes
            && self.neighbors == b.neighbors
            && if self.spec.sha256.len() == 64 && b.spec.sha256.len() == 64 {
                self.spec == b.spec
            } else {
                self.entries == b.entries
                    && self.centroids == b.centroids
                    && self.buckets == b.buckets
                    && self.spatial == b.spatial
            }
    }
}
pub fn sequence_source(source: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let h = Sha256::digest(source.as_bytes());
    u64::from_le_bytes(h[..8].try_into().unwrap())
}
pub fn sequence_key(x: &[f64; 128]) -> [f32; 64] {
    // 32 relative strategic features + 32 reserve/spatial/context features.
    let mut key = [0.; 64];
    for i in 0..32 {
        key[i] = x[2 * i].clamp(-4., 4.) as f32;
        key[32 + i] = x[64 + 2 * i].clamp(-4., 4.) as f32;
    }
    let norm = key.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in &mut key {
        *x /= norm;
    }
    key
}
impl SequenceBank {
    pub fn context(&self, state: &[f64], excluded: u64) -> Arc<SequenceContext> {
        self.try_context(state, excluded)
            .expect("validated sequence state")
    }
    pub fn try_context(
        &self,
        state: &[f64],
        excluded: u64,
    ) -> Result<Arc<SequenceContext>, String> {
        if ![128, crate::MICRO_SPATIAL_INPUTS].contains(&state.len())
            || state.iter().any(|x| !x.is_finite())
        {
            return Err("invalid sequence state".into());
        }
        let key = sequence_key(state[..128].try_into().unwrap());
        let geometry = if self.has_spatial() {
            Some(SequenceGeometry::from_state(state)?)
        } else {
            None
        };
        let phase = u8::from(state[125] > 0.5);
        let q = QueryKey {
            vector: key.map(f32::to_bits),
            phase,
            // Zero already means no exclusion in nearest_index. An absent
            // source has exactly the same eligible entries, across all weights.
            excluded: if excluded != 0 && self.source_ids.contains(&excluded) { excluded } else { 0 },
            geometry,
        };
        // Low bits of two normalized floats can coincide in many positions.
        // Hash the full immutable key to use all of the existing cache capacity.
        use std::hash::{Hash, Hasher};
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        q.hash(&mut hash);
        let shard = hash.finish() as usize % self.cache.len();
        let sequence = self.queries.fetch_add(1, Ordering::Relaxed);
        if let Some(c) = self.cache[shard].lock().unwrap().rows.get(&q).cloned() {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
            self.record_uses(&c, sequence);
            return Ok(c);
        }
        let (neighbors, candidates) = self.nearest_index(
            &key,
            geometry.as_ref(),
            phase,
            excluded,
            self.probes,
            self.neighbors,
        );
        self.candidates.fetch_add(candidates, Ordering::Relaxed);
        let mut patterns = vec![[0.; 32]; 12];
        let mut weights = [0.; 3];
        let mut confidence = 0.;
        for (distance, i) in &neighbors {
            let e = &self.entries[*i];
            let group = (e.outcome + 1) as usize;
            let weight = (-8. * f64::from(*distance)).exp();
            confidence += weight;
            weights[group] += weight;
            for t in 0..4 {
                for k in 0..32 {
                    patterns[group * 4 + t][k] += weight * f64::from(e.patterns[t][k]) / 127.;
                }
            }
        }
        let total = weights.iter().sum::<f64>().max(1e-12);
        let relevance = confidence / self.neighbors.max(1) as f64;
        // Normalise across outcomes, preserving frequency without prescribing signs.
        for p in &mut patterns {
            for x in p {
                *x *= relevance / total;
            }
        }
        let c = Arc::new(SequenceContext {
            patterns,
            neighbors: neighbors.iter().map(|(_, i)| *i).collect(),
            confidence: confidence / self.neighbors.max(1) as f64,
        });
        self.record_uses(&c, sequence);
        let mut cache = self.cache[shard].lock().unwrap();
        if cache.rows.len() >= 16384 {
            if let Some(old) = cache.fifo.pop_front() {
                cache.rows.remove(&old);
            }
        }
        if !cache.rows.contains_key(&q) {
            cache.fifo.push_back(q.clone());
            cache.rows.insert(q, c.clone());
        }
        Ok(c)
    }
    pub fn telemetry(&self) -> [usize; 3] {
        [
            self.queries.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
            self.candidates.load(Ordering::Relaxed),
        ]
    }
    /// Best-effort diagnostic only; never wait for an active lookup to report.
    pub fn cache_occupancy(&self) -> Vec<Option<usize>> {
        self.cache.iter().map(|c|c.try_lock().ok().map(|c|c.rows.len())).collect()
    }
}
