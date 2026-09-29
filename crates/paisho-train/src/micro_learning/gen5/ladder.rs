//! Training curriculum indicator, never a frozen strength/Elo evaluation.
use super::*;
use std::sync::RwLock;
pub(super) const BUDGETS: [usize; 4] = [8, 32, 64, 128];
pub(super) type Shared = Arc<RwLock<State>>;
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct State {
    pub stage: usize,
    pub reference: String,
    pub batches: Vec<Batch>,
    pub current: Vec<Entry>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Entry {
    case: String,
    host: bool,
    win: bool,
    draw: bool,
    unknown: bool,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Batch {
    pub budget: usize,
    pub wins: usize,
    pub draws: usize,
    pub unknown: usize,
    pub games: usize,
    pub promoted: bool,
    pub version: u64,
}
impl State {
    pub fn budget(&self) -> usize {
        BUDGETS[self.stage.min(3)]
    }
    pub fn observe(&mut self, game: &collector::Played, version: u64) {
        let Some(c) = &game.case else {
            return;
        };
        if game.reanalysis
            || game.error.is_some()
            || game.termination == "user-stop"
            || game.termination == "wall-limit" && game.campaign_censored
            || game.reference_budget != Some(self.budget())
            || c.before.attempts != 0
            || self.current.iter().any(|e| e.case == c.case)
        {
            return;
        }
        self.record(c.case.clone(), game.candidate_seat, game.outcome, version);
    }
    fn record(&mut self, case: String, seat: Player, outcome: GameOutcome, version: u64) {
        if self.current.iter().any(|e| e.case == case) {
            return;
        }
        let host = seat == Player::Host;
        // Exactly 50 fresh case attempts per seat; retries never count twice.
        if self.current.iter().filter(|e| e.host == host).count() >= 50 {
            return;
        }
        self.current.push(Entry {
            case,
            host,
            win: outcome == GameOutcome::Win(seat),
            draw: outcome == GameOutcome::Draw,
            unknown: outcome == GameOutcome::Ongoing,
        });
        if self.current.len() == 100 {
            let wins = self.current.iter().filter(|e| e.win).count();
            let promoted = wins >= 60 && self.stage < 3;
            self.batches.push(Batch {
                budget: self.budget(),
                wins,
                draws: self.current.iter().filter(|e| e.draw).count(),
                unknown: self.current.iter().filter(|e| e.unknown).count(),
                games: 100,
                promoted,
                version,
            });
            self.current.clear();
            if promoted {
                self.stage += 1;
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gate_is_sixty_actual_wins_with_fifty_cases_per_seat() {
        let mut s = State::default();
        for wins in [59, 60, 60, 60, 100] {
            let stage = s.stage;
            for i in 0..100 {
                let seat = if i % 2 == 0 {
                    Player::Host
                } else {
                    Player::Guest
                };
                let result = if i < wins {
                    GameOutcome::Win(seat)
                } else if i % 2 == 0 {
                    GameOutcome::Draw
                } else {
                    GameOutcome::Ongoing
                };
                s.record(format!("case-{i}"), seat, result, 1000);
                if i < 99 {
                    assert_eq!(s.stage, stage);
                }
            }
            assert_eq!(
                s.stage,
                if wins >= 60 {
                    (stage + 1).min(3)
                } else {
                    stage
                }
            );
            assert_eq!(s.batches.last().unwrap().wins, wins);
            assert!(s.current.is_empty());
        }
        assert_eq!(s.budget(), 128);
        let encoded = serde_json::to_vec(&s).unwrap();
        assert_eq!(
            serde_json::from_slice::<State>(&encoded)
                .unwrap()
                .batches
                .len(),
            5
        );
    }
    #[test]
    fn repeated_case_and_one_sided_wins_cannot_promote() {
        let mut s = State::default();
        for _ in 0..100 {
            s.record(
                "same".into(),
                Player::Host,
                GameOutcome::Win(Player::Host),
                1,
            );
        }
        assert_eq!(s.current.len(), 1);
        for i in 0..100 {
            s.record(
                format!("{i}"),
                Player::Host,
                GameOutcome::Win(Player::Host),
                1,
            );
        }
        assert_eq!(s.current.len(), 50);
        assert_eq!(s.budget(), 8);
    }
}
