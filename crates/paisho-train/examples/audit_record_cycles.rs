//! Retrospective exact-state loop audit; never adjudicates or changes a PSR.
use paisho_core::{GameRecord, Player, Position};
use std::io::BufRead;
fn same(a: &Position, b: &Position) -> bool {
    a.rule_profile() == b.rule_profile()
        && a.board() == b.board()
        && a.reserve(Player::Host) == b.reserve(Player::Host)
        && a.reserve(Player::Guest) == b.reserve(Player::Guest)
        && a.to_move() == b.to_move()
        && a.phase() == b.phase()
        && a.outcome() == b.outcome()
}
fn cycle(states: &[Position], end: usize, maximum: usize) -> Option<usize> {
    (2..=maximum).find(|&period| {
        end >= 6 * period
            && (end - 6 * period + period..=end).all(|i| same(&states[i], &states[i - period]))
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    for path in std::io::stdin().lock().lines() {
        let path = path?;
        let record: GameRecord = std::fs::read_to_string(&path)?.parse()?;
        let mut position = record.initial_position();
        let mut states = vec![position.clone()];
        for &action in record.actions() {
            position.apply(action)?;
            states.push(position.clone());
        }
        let end = states.len() - 1;
        let first = (0..=end).find_map(|i| cycle(&states, i, 32).map(|p| (i, p)));
        let first4 = (0..=end).find_map(|i| cycle(&states, i, 4).map(|p| (i, p)));
        println!(
            "{}",
            serde_json::json!({"path":path,"decisions":end,"outcome":format!("{:?}",position.outcome()),"first_six_repeats_period_le32":first,"first_six_repeats_period_le4":first4,"trailing_period_le32":cycle(&states,end,32),"trailing_period_le4":cycle(&states,end,4)})
        );
    }
    Ok(())
}
