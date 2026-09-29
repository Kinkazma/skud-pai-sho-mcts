/// Small deterministic SplitMix64 generator. Its algorithm is part of the
/// reproducibility contract; it is not intended for cryptographic use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StableRng {
    state: u64,
}

impl StableRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        (self.next_u64() >> 11) as f64 * SCALE
    }

    /// Uniformly samples `0..upper_bound` without modulo bias.
    pub fn index(&mut self, upper_bound: usize) -> usize {
        assert!(upper_bound > 0, "an index needs a non-empty range");
        let bound = upper_bound as u64;
        let rejection_zone = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= rejection_zone {
                return (value % bound) as usize;
            }
        }
    }
}
