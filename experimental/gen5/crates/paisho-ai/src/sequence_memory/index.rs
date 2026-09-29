use super::*;
use rayon::prelude::*;
pub fn sequence_distance(a: &[f32; 64], b: &[f32; 64]) -> f32 {
    {
        let mut sum = [0f32; 4];
        for i in (0..64).step_by(4) {
            for lane in 0..4 {
                let d = a[i + lane] - b[i + lane];
                sum[lane] += d * d;
            }
        }
        sum.iter().sum()
    }
}
impl SequenceBank {
    pub fn build(
        entries: Vec<SequenceEntry>,
        games: usize,
        human_games: usize,
        clusters: usize,
    ) -> Self {
        Self::build_index(entries, None, games, human_games, clusters)
    }
    pub(super) fn build_index(
        entries: Vec<SequenceEntry>,
        spatial: Option<Vec<SequenceGeometry>>,
        games: usize,
        human_games: usize,
        clusters: usize,
    ) -> Self {
        assert!(!entries.is_empty() && games <= 50000 && human_games <= games);
        let n = clusters.min(entries.len()).max(1);
        let samples: Vec<_> = entries
            .iter()
            .step_by((entries.len() / 8192).max(1))
            .take(8192)
            .collect();
        let mut centroids: Vec<_> = (0..n).map(|i| entries[i * entries.len() / n].key).collect();
        for _ in 0..4 {
            let assigned: Vec<_> = samples
                .par_iter()
                .map(|e| closest(&e.key, &centroids))
                .collect();
            let mut sums = vec![[0f64; 64]; n];
            let mut counts = vec![0usize; n];
            for (e, b) in samples.iter().zip(assigned) {
                counts[b] += 1;
                for k in 0..64 {
                    sums[b][k] += f64::from(e.key[k]);
                }
            }
            for b in 0..n {
                if counts[b] > 0 {
                    for k in 0..64 {
                        centroids[b][k] = (sums[b][k] / counts[b] as f64) as f32;
                    }
                }
            }
        }
        let assignments: Vec<_> = entries
            .par_iter()
            .map(|e| closest(&e.key, &centroids))
            .collect();
        let mut buckets = vec![vec![]; n];
        for (i, b) in assignments.into_iter().enumerate() {
            buckets[b].push(i as u32);
        }
        let (entries, buckets, spatial) = pack(entries, buckets, spatial);
        Self {
            uses: usage::new_uses(entries.len()),
            source_ids: entries.iter().map(|e|e.source).collect(),
            spec: SequenceMemorySpec {
                path: String::new(),
                sha256: String::new(),
            },
            entries,
            centroids,
            buckets,
            games,
            human_games,
            probes: 4,
            neighbors: 8,
            spatial,
            cache: (0..16).map(|_| Mutex::new(QueryCache::default())).collect(),
            queries: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            candidates: AtomicUsize::new(0),
        }
    }
    pub fn nearest(
        &self,
        key: &[f32; 64],
        phase: u8,
        excluded: u64,
        probes: usize,
        k: usize,
    ) -> (Vec<(f32, usize)>, usize) {
        self.nearest_index(key, None, phase, excluded, probes, k)
    }
    pub(super) fn nearest_index(
        &self,
        key: &[f32; 64],
        geometry: Option<&SequenceGeometry>,
        phase: u8,
        excluded: u64,
        probes: usize,
        k: usize,
    ) -> (Vec<(f32, usize)>, usize) {
        if k == 0 || probes == 0 {
            return (vec![], 0);
        }
        let mut coarse: Vec<_> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (sequence_distance(key, c), i))
            .collect();
        let order = |a: &(f32, usize), b: &(f32, usize)| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
        if probes < coarse.len() {
            coarse.select_nth_unstable_by(probes, order);
            coarse.truncate(probes);
        }
        coarse.sort_unstable_by(order);
        let mut best: Vec<(f32, usize)> = vec![];
        let mut inspected = 0;
        for (_, b) in coarse.into_iter().take(probes) {
            for id in &self.buckets[b] {
                let i = *id as usize;
                let e = &self.entries[i];
                if e.phase != phase || excluded != 0 && e.source == excluded {
                    continue;
                }
                inspected += 1;
                let mut d = sequence_distance(key, &e.key);
                if best.len() == k && d >= best.last().unwrap().0 {
                    continue;
                }
                if let Some(g) = geometry {
                    d += 0.5 * g.distance(&self.spatial.as_ref().expect("spatial bank")[i]);
                }
                // At most one segment per source: neighboring slices are not independent evidence.
                if let Some(j) = best
                    .iter()
                    .position(|(_, j)| self.entries[*j].source == e.source)
                {
                    if best[j].0 <= d {
                        continue;
                    }
                    best.remove(j);
                }
                if best.len() < k || d < best.last().unwrap().0 {
                    let pos = best.partition_point(|(distance, _)| *distance <= d);
                    best.insert(pos, (d, i));
                    best.truncate(k);
                }
            }
        }
        (best, inspected)
    }
}
fn closest(key: &[f32; 64], centroids: &[[f32; 64]]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (sequence_distance(key, c), i))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .unwrap()
        .1
}

// Store each coarse bucket contiguously: distance scans touch sequential keys.
pub(super) fn pack(
    entries: Vec<SequenceEntry>,
    buckets: Vec<Vec<u32>>,
    spatial: Option<Vec<SequenceGeometry>>,
) -> (
    Vec<SequenceEntry>,
    Vec<Vec<u32>>,
    Option<Vec<SequenceGeometry>>,
) {
    let mut rows = Vec::with_capacity(entries.len());
    let mut packed = Vec::with_capacity(buckets.len());
    let mut maps = spatial.as_ref().map(|s| Vec::with_capacity(s.len()));
    for bucket in buckets {
        let mut ids = Vec::with_capacity(bucket.len());
        for id in bucket {
            ids.push(rows.len() as u32);
            rows.push(entries[id as usize].clone());
            if let Some(maps) = &mut maps {
                maps.push(spatial.as_ref().unwrap()[id as usize]);
            }
        }
        packed.push(ids);
    }
    (rows, packed, maps)
}
