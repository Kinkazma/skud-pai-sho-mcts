use super::*;
use std::io::{Read, Write};
impl SequenceBank {
    /// Fixed endian compact keys/patterns. Container has a caller-verified SHA256.
    pub fn write_to(&self, w: &mut impl Write) -> std::io::Result<()> {
        w.write_all(if self.has_spatial() {
            b"PSSEQ002"
        } else {
            b"PSSEQ001"
        })?;
        for n in [
            self.games,
            self.human_games,
            self.entries.len(),
            self.centroids.len(),
        ] {
            w.write_all(&(n as u32).to_le_bytes())?;
        }
        for (i, e) in self.entries.iter().enumerate() {
            for x in e.key {
                w.write_all(&((x.clamp(-1., 1.) * 32767.).round() as i16).to_le_bytes())?;
            }
            for pattern in e.patterns {
                for x in pattern {
                    w.write_all(&[x as u8])?;
                }
            }
            w.write_all(&e.source.to_le_bytes())?;
            for x in [e.game, e.decision, e.end_decision] {
                w.write_all(&x.to_le_bytes())?;
            }
            w.write_all(&[e.outcome as u8, e.phase])?;
            if let Some(g) = self.geometry(i) {
                for plane in g.0 {
                    for word in plane {
                        w.write_all(&word.to_le_bytes())?;
                    }
                }
            }
        }
        for c in &self.centroids {
            for x in c {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        for bucket in &self.buckets {
            w.write_all(&(bucket.len() as u32).to_le_bytes())?;
            for i in bucket {
                w.write_all(&i.to_le_bytes())?;
            }
        }
        Ok(())
    }
    pub fn read_from(r: &mut impl Read, spec: SequenceMemorySpec) -> Result<Self, String> {
        fn bytes<const N: usize>(r: &mut impl Read) -> Result<[u8; N], String> {
            let mut b = [0; N];
            r.read_exact(&mut b).map_err(|e| e.to_string())?;
            Ok(b)
        }
        fn u32read(r: &mut impl Read) -> Result<usize, String> {
            Ok(u32::from_le_bytes(bytes(r)?) as usize)
        }
        let spatial_format = match &bytes::<8>(r)? {
            b"PSSEQ001" => false,
            b"PSSEQ002" => true,
            _ => return Err("sequence bank format".into()),
        };
        let games = u32read(r)?;
        let human_games = u32read(r)?;
        let count = u32read(r)?;
        let n = u32read(r)?;
        if games > 50000
            || human_games > games
            || count == 0
            || count > 10000000
            || n == 0
            || n > 4096
        {
            return Err("sequence bank bounds".into());
        }
        let mut entries = Vec::with_capacity(count);
        let mut spatial = spatial_format.then(|| Vec::with_capacity(count));
        for _ in 0..count {
            let mut key = [0.; 64];
            for x in &mut key {
                *x = f32::from(i16::from_le_bytes(bytes(r)?)) / 32767.;
            }
            let mut patterns = [[0i8; 32]; 4];
            for p in &mut patterns {
                for x in p {
                    *x = bytes::<1>(r)?[0] as i8;
                }
            }
            let source = u64::from_le_bytes(bytes(r)?);
            let game = u32read(r)? as u32;
            let decision = u32read(r)? as u32;
            let end_decision = u32read(r)? as u32;
            let outcome = bytes::<1>(r)?[0] as i8;
            let phase = bytes::<1>(r)?[0];
            if !(-1..=1).contains(&outcome)
                || phase > 1
                || game as usize >= games
                || end_decision <= decision
            {
                return Err("invalid sequence entry".into());
            }
            entries.push(SequenceEntry {
                key,
                patterns,
                source,
                game,
                decision,
                end_decision,
                outcome,
                phase,
            });
            if let Some(maps) = &mut spatial {
                let mut planes = [[0u64; 5]; 5];
                for plane in &mut planes {
                    for word in plane {
                        *word = u64::from_le_bytes(bytes(r)?);
                    }
                }
                let g = SequenceGeometry(planes);
                if !g.valid() {
                    return Err("invalid sequence geometry".into());
                }
                maps.push(g);
            }
        }
        let mut centroids = vec![[0.; 64]; n];
        for c in &mut centroids {
            for x in c {
                *x = f32::from_le_bytes(bytes(r)?);
                if !x.is_finite() {
                    return Err("nonfinite sequence centroid".into());
                }
            }
        }
        let mut buckets = vec![];
        let mut seen = vec![false; count];
        for _ in 0..n {
            let size = u32read(r)?;
            if size > count {
                return Err("sequence bucket size".into());
            }
            let mut bucket = Vec::with_capacity(size);
            for _ in 0..size {
                let i = u32read(r)?;
                if i >= count || seen[i] {
                    return Err("duplicate or invalid sequence index".into());
                }
                seen[i] = true;
                bucket.push(i as u32);
            }
            buckets.push(bucket);
        }
        if seen.iter().any(|x| !*x) {
            return Err("missing sequence entries".into());
        }
        let mut trailing = [0u8; 1];
        if r.read(&mut trailing).map_err(|e| e.to_string())? != 0 {
            return Err("trailing sequence data".into());
        }
        let (entries, buckets, spatial) = super::index::pack(entries, buckets, spatial);
        Ok(Self {
            uses: usage::new_uses(entries.len()),
            source_ids: entries.iter().map(|e|e.source).collect(),
            spec,
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
        })
    }
}
