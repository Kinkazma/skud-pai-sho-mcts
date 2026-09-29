//! Bounded ring-only adversarial diagnostic; never treats budget exhaustion as safety.
use paisho_core::{
    harmony_ring_owners_for_profile, legal_actions, GameOutcome, GameRecord, Player, Position,
};
use serde_json::json;
use std::{fs, path::Path};
fn solve(p: &Position, attacker: Player, depth: usize, left: &mut usize) -> Option<bool> {
    if *left == 0 {
        return None;
    }
    *left -= 1;
    match p.outcome() {
        GameOutcome::Win(w) => {
            return Some(
                w == attacker
                    && harmony_ring_owners_for_profile(p.board(), p.rule_profile())
                        .contains(&attacker),
            )
        }
        GameOutcome::Draw => return Some(false),
        GameOutcome::Ongoing => {}
    }
    if depth == 0 {
        return Some(false);
    }
    let actions = legal_actions(p);
    if actions.is_empty() {
        return Some(false);
    }
    let own = p.to_move() == attacker;
    let mut unknown = false;
    for a in actions {
        let mut c = p.clone();
        c.apply(a).unwrap();
        match solve(&c, attacker, depth - 1, left) {
            Some(v) if v == own => return Some(own),
            None => unknown = true,
            _ => {}
        }
    }
    if unknown {
        None
    } else {
        Some(!own)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let dir = Path::new(&args[1]);
    let mut rows = vec![];
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json")
            || !path
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("partie-")
        {
            continue;
        }
        let meta: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
        let record: GameRecord = meta["moves"].as_str().unwrap().parse()?;
        let human = if meta["human_side"] == "host" {
            Player::Host
        } else {
            Player::Guest
        };
        let mut positions = vec![record.initial_position()];
        for a in record.actions() {
            let mut p = positions.last().unwrap().clone();
            p.apply(*a)?;
            positions.push(p);
        }
        for (prefix, p) in positions.iter().enumerate() {
            if p.outcome() != GameOutcome::Ongoing || prefix < positions.len().saturating_sub(12) {
                continue;
            }
            let mut horizons = vec![];
            for depth in 1..=3 {
                let mut left = 30000;
                let forced = solve(p, human, depth, &mut left);
                horizons.push(json!({"depth":depth,"human_forced_ring":forced,"nodes":30000-left}));
                if forced == Some(true) {
                    break;
                }
            }
            let mut counterfactual = serde_json::Value::Null;
            if p.to_move() != human {
                if let Some(next) = positions.get(prefix + 1) {
                    if solve(next, human, 2, &mut 30000) == Some(true) {
                        let mut remaining = 100000;
                        let mut safe = None;
                        let mut examined = 0;
                        for action in legal_actions(p) {
                            let mut alternative = p.clone();
                            alternative.apply(action)?;
                            examined += 1;
                            if solve(&alternative, human, 2, &mut remaining) == Some(false) {
                                safe = Some(action.to_string());
                                break;
                            }
                            if remaining == 0 {
                                break;
                            }
                        }
                        counterfactual = json!({"actual_allows_forced_ring_within_two_decisions":true,"alternative_avoiding_that_horizon":safe,"examined":examined,"nodes":100000-remaining,"budget_exhausted":remaining==0});
                    }
                }
            }
            let row = json!({"file":path.file_name().unwrap().to_string_lossy(),"agent":meta["agent"],"human":meta["human_side"],"prefix":prefix,"to_move":format!("{:?}",p.to_move()),"phase":format!("{:?}",p.phase()),"next_recorded":record.actions().get(prefix).map(ToString::to_string),"horizons":horizons,"counterfactual":counterfactual});
            println!("{}", row);
            rows.push(row);
        }
    }
    fs::write(&args[2], serde_json::to_vec_pretty(&rows)?)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use paisho_core::{BasicFlower, StandardSetup};
    #[test]
    fn budget_exhaustion_stays_unknown() {
        let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        assert_eq!(solve(&p, Player::Host, 3, &mut 0), None);
        assert_eq!(solve(&p, Player::Host, 0, &mut 1), Some(false));
    }

    #[test]
    fn wheel_bonus_is_two_same_player_decisions_and_defense_is_verified() {
        let record: GameRecord = include_str!("../../../benchmarks/results/gen32-ring-foresight-2026-09-11/human/partie-20260910T220624Z-a370f0c673ddb3b4fffdeec0c7243d5f.psr").parse().unwrap();
        let mut before = record.initial_position();
        for a in &record.actions()[..20] {
            before.apply(*a).unwrap();
        }
        let mut actual = before.clone();
        actual.apply(record.actions()[20]).unwrap();
        assert_eq!(actual.to_move(), Player::Guest);
        assert_eq!(solve(&actual, Player::Guest, 1, &mut 30000), Some(false));
        assert_eq!(solve(&actual, Player::Guest, 2, &mut 30000), Some(true));
        actual.apply(record.actions()[21]).unwrap();
        assert_eq!(actual.to_move(), Player::Guest);
        assert_eq!(solve(&actual, Player::Guest, 1, &mut 30000), Some(true));
        let defense = legal_actions(&before)
            .into_iter()
            .find(|a| a.to_string() == "arrange -1,6 -5,7")
            .unwrap();
        before.apply(defense).unwrap();
        assert_eq!(solve(&before, Player::Guest, 2, &mut 30000), Some(false));
    }
}
