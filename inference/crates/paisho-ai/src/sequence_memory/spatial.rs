//! Exact owner/kind geometry in five bit planes (200 bytes per lookup anchor).
//! Coarse strategic buckets stay shared with V1; all their eligible candidates
//! are reranked by categorical occupied-square disagreement, not tile ordinals.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SequenceGeometry(pub(super) [[u64; 5]; 5]);
impl SequenceGeometry {
    pub fn from_state(state: &[f64]) -> Result<Self, String> {
        if state.len() != crate::MICRO_SPATIAL_INPUTS {
            return Err("spatial sequence retrieval requires 417 state inputs".into());
        }
        let mut planes = [[0u64; 5]; 5];
        for (i, x) in state[128..].iter().enumerate() {
            let v = x * 12.0;
            if !v.is_finite() || v.abs() > 12.0 || (v - v.round()).abs() > 1e-9 {
                return Err("invalid spatial sequence tile code".into());
            }
            let signed = v.round() as i8;
            let code = if signed < 0 { 12 - signed } else { signed } as u8;
            for (bit, plane) in planes.iter_mut().enumerate() {
                plane[i / 64] |= u64::from((code >> bit) & 1) << (i % 64);
            }
        }
        Ok(Self(planes))
    }
    /// Mismatching cells / cells occupied in either position, in [0, 1].
    /// Moving a tile changes its old and new squares; owner/type differences
    /// count equally. Empty regions cannot dilute a small tactical difference.
    pub fn distance(&self, other: &Self) -> f32 {
        let mut different = 0;
        let mut occupied = 0;
        for word in 0..5 {
            let mut xor = 0;
            let mut union = 0;
            for bit in 0..5 {
                xor |= self.0[bit][word] ^ other.0[bit][word];
                union |= self.0[bit][word] | other.0[bit][word];
            }
            different += xor.count_ones();
            occupied += union.count_ones();
        }
        different as f32 / occupied.max(1) as f32
    }
    pub(super) fn valid(&self) -> bool {
        (0..5).all(|bit| self.0[bit][4] >> 33 == 0)
            && (0..289).all(|i| {
                (0..5)
                    .map(|bit| ((self.0[bit][i / 64] >> (i % 64)) & 1) << bit)
                    .sum::<u64>()
                    <= 24
            })
    }
}

impl SequenceBank {
    pub fn has_spatial(&self) -> bool {
        self.spatial.is_some()
    }
    pub fn geometry(&self, i: usize) -> Option<&SequenceGeometry> {
        self.spatial.as_ref().and_then(|s| s.get(i))
    }
    pub fn build_spatial(
        entries: Vec<SequenceEntry>,
        geometry: Vec<SequenceGeometry>,
        games: usize,
        human_games: usize,
        clusters: usize,
    ) -> Self {
        assert_eq!(entries.len(), geometry.len());
        Self::build_index(entries, Some(geometry), games, human_games, clusters)
    }
    pub fn nearest_spatial(
        &self,
        key: &[f32; 64],
        geometry: &SequenceGeometry,
        phase: u8,
        excluded: u64,
        probes: usize,
        k: usize,
    ) -> (Vec<(f32, usize)>, usize) {
        assert!(self.has_spatial());
        self.nearest_index(key, Some(geometry), phase, excluded, probes, k)
    }
}
