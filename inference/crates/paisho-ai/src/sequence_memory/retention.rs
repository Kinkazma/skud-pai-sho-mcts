//! Rarity is a revocable status, not a certificate that a motif caused a win.
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MotifEvidence {
    pub human: bool,
    pub associated_win: bool,
    /// Distinct source games in the motif's neighborhood, not repeated slices.
    pub source_games: usize,
    /// Successful retrievals in the most recent measurement window.
    pub recent_uses: usize,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RarityPolicy {
    pub max_source_games: usize,
    pub max_recent_uses: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryProtection {
    Human,
    RareWin,
    Ordinary,
}
impl RarityPolicy {
    pub fn protection(self, evidence: MotifEvidence) -> MemoryProtection {
        if evidence.human {
            MemoryProtection::Human
        } else if evidence.associated_win
            && evidence.source_games > 0
            && evidence.source_games <= self.max_source_games
            && evidence.recent_uses <= self.max_recent_uses
        {
            MemoryProtection::RareWin
        } else {
            MemoryProtection::Ordinary
        }
    }
}
