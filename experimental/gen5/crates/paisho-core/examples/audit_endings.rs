//! Read-only terminal-board audit of PSRs supplied one path per stdin line.
use paisho_core::{
    harmony_ring_owners_for_profile, midline_crossing_harmony_count, GameOutcome, GameRecord,
    Player,
};
use std::io::BufRead;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::io::stdin().lock().lines() {
        let record: GameRecord = std::fs::read_to_string(path?)?.parse()?;
        let p = record.replay()?;
        let rings = harmony_ring_owners_for_profile(p.board(), record.rules());
        let host_ring = rings.contains(&Player::Host);
        let guest_ring = rings.contains(&Player::Guest);
        let host_basic = p.reserve(Player::Host).basic_count();
        let guest_basic = p.reserve(Player::Guest).basic_count();
        let outcome = match p.outcome() {
            GameOutcome::Win(Player::Host) => "host",
            GameOutcome::Win(Player::Guest) => "guest",
            GameOutcome::Draw => "draw",
            GameOutcome::Ongoing => "ongoing",
        };
        let category = if outcome == "ongoing" {
            "unfinished"
        } else if !rings.is_empty() && (host_basic == 0 || guest_basic == 0) {
            "ring-and-exhaustion"
        } else if !rings.is_empty() {
            "harmony-ring"
        } else if host_basic == 0 || guest_basic == 0 {
            "basic-exhaustion"
        } else {
            "other-terminal"
        };
        println!("{{\"category\":\"{category}\",\"outcome\":\"{outcome}\",\"host_ring\":{host_ring},\"guest_ring\":{guest_ring},\"host_basic_remaining\":{host_basic},\"guest_basic_remaining\":{guest_basic},\"host_midline\":{},\"guest_midline\":{},\"decisions\":{}}}",midline_crossing_harmony_count(p.board(),Player::Host),midline_crossing_harmony_count(p.board(),Player::Guest),record.actions().len());
    }
    Ok(())
}
