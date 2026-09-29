use paisho_core::{legal_actions, GameOutcome, Player, Position};

/// Exact result of a bounded adversarial search. `NotProven` deliberately
/// means only that this horizon contains no proven forced win.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TacticalVerdict {
    ForcedWin,
    NotProven,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TacticalProof {
    pub verdict: TacticalVerdict,
    pub visited_positions: u64,
    pub terminal_positions: u64,
}

impl TacticalProof {
    pub const fn is_forced_win(self) -> bool {
        matches!(self.verdict, TacticalVerdict::ForcedWin)
    }
}

/// Proves whether `attacker` can force a terminal win within at most
/// `maximum_decisions` legal decisions. A Harmony Bonus is one decision and
/// naturally remains controlled by the same player.
pub fn prove_forced_win(
    position: &Position,
    attacker: Player,
    maximum_decisions: usize,
) -> TacticalProof {
    let mut statistics = ProofStatistics::default();
    let forced = search(position, attacker, maximum_decisions, &mut statistics);
    TacticalProof {
        verdict: if forced {
            TacticalVerdict::ForcedWin
        } else {
            TacticalVerdict::NotProven
        },
        visited_positions: statistics.visited_positions,
        terminal_positions: statistics.terminal_positions,
    }
}

#[derive(Default)]
struct ProofStatistics {
    visited_positions: u64,
    terminal_positions: u64,
}

fn search(
    position: &Position,
    attacker: Player,
    decisions_left: usize,
    statistics: &mut ProofStatistics,
) -> bool {
    statistics.visited_positions += 1;
    match position.outcome() {
        GameOutcome::Win(winner) => {
            statistics.terminal_positions += 1;
            return winner == attacker;
        }
        GameOutcome::Draw => {
            statistics.terminal_positions += 1;
            return false;
        }
        GameOutcome::Ongoing => {}
    }
    if decisions_left == 0 {
        return false;
    }

    let actions = legal_actions(position);
    if actions.is_empty() {
        return false;
    }
    let attacker_controls_node = position.to_move() == attacker;
    for action in actions {
        let mut child = position.clone();
        child
            .apply(action)
            .expect("the tactical oracle applies only engine-generated actions");
        let child_is_forced = search(&child, attacker, decisions_left - 1, statistics);
        if attacker_controls_node == child_is_forced {
            return attacker_controls_node;
        }
    }
    !attacker_controls_node
}

#[cfg(test)]
mod tests {
    use paisho_core::{BasicFlower, Position, StandardSetup};

    use super::*;

    #[test]
    fn an_ongoing_position_is_not_proven_at_zero_depth() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let proof = prove_forced_win(&position, position.to_move(), 0);
        assert_eq!(proof.verdict, TacticalVerdict::NotProven);
        assert_eq!(proof.visited_positions, 1);
        assert_eq!(proof.terminal_positions, 0);
    }
}
