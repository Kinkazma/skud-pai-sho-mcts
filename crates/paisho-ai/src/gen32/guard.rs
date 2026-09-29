//! Bounded tactical audit. Absence of a win within a horizon is not a game proof.
use crate::SearchReport;
use paisho_core::{legal_actions, visit_harmonies, Action, GameOutcome, Player, Position};
use std::time::Instant;
#[derive(Clone, Debug, Default)]
pub struct Gen32GuardReport {
    pub selected: usize,
    pub refuted: Vec<usize>,
    pub safe: Vec<usize>,
    pub visited: usize,
    pub exhausted: bool,
}
struct Budget {
    left: usize,
    visited: usize,
    deadline: Option<Instant>,
}
impl Budget {
    fn take(&mut self) -> bool {
        if self.left == 0 || self.deadline.is_some_and(|d| Instant::now() >= d) {
            self.left = 0;
            return false;
        }
        self.left -= 1;
        self.visited += 1;
        true
    }
}
fn opponent(p: Player) -> Player {
    if p == Player::Host {
        Player::Guest
    } else {
        Player::Host
    }
}
/// Tri-state AND/OR with real chooser semantics, including same-player bonuses.
fn solve(p: &Position, attacker: Player, depth: usize, b: &mut Budget) -> Option<bool> {
    if !b.take() {
        return None;
    }
    match p.outcome() {
        GameOutcome::Win(w) => return Some(w == attacker),
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
    let attack = p.to_move() == attacker;
    let mut unknown = false;
    for a in actions {
        let mut child = p.clone();
        child.apply(a).expect("legal tactical action");
        match solve(&child, attacker, depth - 1, b) {
            Some(w) if w == attack => return Some(attack),
            None => {
                unknown = true;
                if b.left == 0 {
                    break;
                }
            }
            _ => {}
        }
    }
    if unknown {
        None
    } else {
        Some(!attack)
    }
}
/// Inspect the chosen action, then alternatives only when its opponent win is
/// proved. Each query is independently capped; all queries share a hard node cap.
/// The cheap harmony gate is a cost heuristic, not a completeness guarantee.
pub fn gen32_tactical_guard(
    p: &Position,
    actions: &[Action],
    report: &SearchReport,
    maximum_positions: usize,
    deadline: Option<Instant>,
) -> Gen32GuardReport {
    let mut out = Gen32GuardReport {
        selected: report.selected_index,
        ..Default::default()
    };
    if maximum_positions == 0 {
        return out;
    }
    let enemy = opponent(p.to_move());
    let mut harmonies = 0;
    visit_harmonies(p.board(), |h| {
        if h.owner == enemy {
            harmonies += 1;
        }
    });
    if harmonies < 3 {
        return out;
    }
    let mut order: Vec<_> = (0..actions.len()).collect();
    order.sort_by(|&a, &b| {
        report.actions[b]
            .visits
            .cmp(&report.actions[a].visits)
            .then_with(|| {
                report.actions[b]
                    .mean_value()
                    .total_cmp(&report.actions[a].mean_value())
            })
            .then(a.cmp(&b))
    });
    order.retain(|i| *i != report.selected_index);
    order.insert(0, report.selected_index);
    for index in order {
        if out.visited >= maximum_positions {
            out.exhausted = true;
            break;
        }
        let mut next = p.clone();
        next.apply(actions[index]).expect("legal root action");
        // Complete our pending bonus, then the opponent's movement and bonus.
        let depth = if next.to_move() == enemy { 2 } else { 3 };
        let mut b = Budget {
            left: (maximum_positions - out.visited).min(4096),
            visited: 0,
            deadline,
        };
        let verdict = solve(&next, enemy, depth, &mut b);
        out.visited += b.visited;
        match verdict {
            Some(true) => out.refuted.push(index),
            Some(false) => {
                out.safe.push(index);
                out.selected = index;
                break;
            }
            None => {
                out.exhausted = true;
            }
        }
        // No alternative search unless the original action is actually refuted.
        if index == report.selected_index && verdict != Some(true) {
            break;
        }
        if deadline.is_some_and(|d| Instant::now() >= d) {
            break;
        }
    }
    // No verified alternative: preserve search choice; don't invent a defence.
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use paisho_core::{GameRecord, RuleProfileId};
    fn p() -> Position {
        let r: GameRecord = include_str!("../../../../benchmarks/results/gen32-ring-foresight-2026-09-11/human/partie-20260910T220624Z-a370f0c673ddb3b4fffdeec0c7243d5f.psr").parse().unwrap();
        let mut p =
            GameRecord::with_rules(r.setup(), RuleProfileId::SkudPaiSho2022V2).initial_position();
        for a in &r.actions()[..21] {
            p.apply(*a).unwrap();
        }
        p
    }
    #[test]
    fn expired_deadline_is_unknown() {
        let mut b = Budget {
            left: 100,
            visited: 0,
            deadline: Some(Instant::now()),
        };
        assert_eq!(solve(&p(), Player::Guest, 2, &mut b), None);
        assert_eq!(b.visited, 0);
    }
    #[test]
    fn full_bonus_turn_and_exhaustion_are_distinguished() {
        let p = p();
        let mut budget = Budget {
            left: 100000,
            visited: 0,
            deadline: None,
        };
        assert_eq!(solve(&p, Player::Guest, 1, &mut budget), Some(false));
        assert_eq!(solve(&p, Player::Guest, 2, &mut budget), Some(true));
        budget.left = 0;
        assert_eq!(solve(&p, Player::Guest, 2, &mut budget), None);
    }
}
