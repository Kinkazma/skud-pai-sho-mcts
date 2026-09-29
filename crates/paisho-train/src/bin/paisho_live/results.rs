//! Count cached terminal outcomes only; never reconstruct boards or run inference.
use paisho_core::{GameOutcome, Player};
use paisho_replay::ReplayDigestV1;
use paisho_train::{live_actors::LiveActorsReport, CurriculumTierV1};
use serde_json::{json, Value};

pub fn summarize(
    report: &LiveActorsReport,
    producer: ReplayDigestV1,
    generation: u64,
    opponent: CurriculumTierV1,
) -> Value {
    let mut wins = 0;
    let mut losses = 0;
    let mut draws = 0;
    let mut self_play_decisive = 0;
    let mut self_play_draws = 0;
    let mut external_games = 0;
    for result in &report.retained {
        let game = &result.game;
        let host = game.host_agent() == producer;
        let guest = game.guest_agent() == producer;
        if host != guest {
            external_games += 1;
        }
        match game.outcome() {
            GameOutcome::Draw if host && guest => self_play_draws += 1,
            GameOutcome::Draw if host || guest => draws += 1,
            GameOutcome::Win(_) if host && guest => self_play_decisive += 1,
            GameOutcome::Win(side) if host || guest => {
                if (side == Player::Host && host) || (side == Player::Guest && guest) {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }
            _ => {}
        }
    }
    json!({"generation":generation,"opponent":opponent,"wins":wins,"losses":losses,
        "draws":draws,"self_play_decisive":self_play_decisive,
        "games":external_games,"total_games":report.retained.len(),"self_play_draws":self_play_draws,"attempts":report.attempts,
        "scope":"retained-terminal-games"})
}
