//! Sound outcomes are separate from heuristic values and Monte Carlo averages.
use super::*;
pub(super) fn terminal(p: &Position) -> Option<GameOutcome> {
    match p.outcome() {
        GameOutcome::Ongoing => None,
        outcome => Some(outcome),
    }
}
pub(super) fn value(outcome: GameOutcome, player: Player) -> f64 {
    match outcome {
        GameOutcome::Win(w) if w == player => 1.,
        GameOutcome::Win(_) => -1.,
        _ => 0.,
    }
}
pub(super) fn rank(proof: Option<GameOutcome>, player: Player) -> i8 {
    match proof {
        Some(GameOutcome::Win(w)) if w == player => 2,
        Some(GameOutcome::Win(_)) => -1,
        _ => 0,
    }
}
impl Node {
    /// A proven option bounds the chooser's value without solving the node.
    /// All numeric values use the retained tree's fixed player perspective.
    pub(super) fn bounded_value(&self, estimate: f64, perspective: Player) -> f64 {
        if !self.solver {
            return estimate;
        }
        if let Some(proof) = self.proof {
            return value(proof, perspective);
        }
        if self
            .children
            .iter()
            .any(|c| c.node.proof == Some(GameOutcome::Draw))
        {
            if self.position.to_move() == perspective {
                estimate.max(0.)
            } else {
                estimate.min(0.)
            }
        } else {
            estimate
        }
    }
    pub(super) fn refresh_proof(&mut self) {
        if self.proof.is_some() {
            return;
        }
        let chooser = self.position.to_move();
        if self
            .children
            .iter()
            .any(|c| c.node.proof == Some(GameOutcome::Win(chooser)))
        {
            self.proof = Some(GameOutcome::Win(chooser));
        } else if self.actions_ready
            && !self.has_unexpanded()
            && !self.children.is_empty()
            && self.children.iter().all(|c| c.node.proof.is_some())
        {
            self.proof = if self
                .children
                .iter()
                .any(|c| c.node.proof == Some(GameOutcome::Draw))
            {
                Some(GameOutcome::Draw)
            } else {
                self.children[0].node.proof
            };
        }
    }
}

/// Recommendation/learning eligibility, not exploration eligibility. Unknown
/// replies remain searchable. Negative estimates are not relabelled as proofs.
pub(super) fn root_eligible(
    proofs: &[Option<GameOutcome>],
    estimates: &[f64],
    player: Player,
) -> Vec<bool> {
    let win = proofs.contains(&Some(GameOutcome::Win(player)));
    let draw = proofs.contains(&Some(GameOutcome::Draw));
    let all_lost = proofs
        .iter()
        .all(|p| *p == Some(GameOutcome::Win(player.opponent())));
    proofs
        .iter()
        .zip(estimates)
        .map(|(p, v)| {
            if win {
                *p == Some(GameOutcome::Win(player))
            } else if draw {
                *p == Some(GameOutcome::Draw) || p.is_none() && *v > 0.
            } else {
                all_lost || *p != Some(GameOutcome::Win(player.opponent()))
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    use paisho_core::{BasicFlower, StandardSetup};
    fn parent() -> Node {
        Node::new(Position::from_standard_setup(StandardSetup::balanced(
            BasicFlower::Red3,
        )))
    }
    fn child(outcome: Option<GameOutcome>) -> Child {
        let mut n = parent();
        let action = legal_actions(&n.position)[0];
        n.proof = outcome;
        Child {
            action,
            node: n,
            policy_bias: 0.,
        }
    }
    #[test]
    fn unknown_and_unexpanded_are_not_losses() {
        let mut n = parent();
        let player = n.position.to_move();
        let opponent = if player == Player::Host {
            Player::Guest
        } else {
            Player::Host
        };
        n.children.push(child(Some(GameOutcome::Win(opponent))));
        n.refresh_proof();
        assert_eq!(n.proof, None);
        n.actions_ready = true;
        n.children.push(child(None));
        n.refresh_proof();
        assert_eq!(n.proof, None);
        n.children.pop();
        n.refresh_proof();
        assert_eq!(n.proof, Some(GameOutcome::Win(opponent)));
    }
    #[test]
    fn chooser_win_overrides_unknown_siblings_and_averages() {
        let mut n = parent();
        let p = n.position.to_move();
        n.children.push(child(None));
        n.children.push(child(Some(GameOutcome::Win(p))));
        n.value_sum = -100.;
        n.refresh_proof();
        assert_eq!(n.proof, Some(GameOutcome::Win(p)));
        assert_eq!(n.value_sum, -100.);
    }
    #[test]
    fn solved_draw_does_not_steal_visits_from_unresolved_branches() {
        let mut n = parent();
        n.solver = true;
        n.visits = 100;
        let mut draw = child(Some(GameOutcome::Draw));
        draw.node.visits = 1;
        let mut unknown = child(None);
        unknown.node.visits = 99;
        unknown.node.value_sum = -90.;
        n.children = vec![draw, unknown];
        assert_eq!(select_child(&n, n.position.to_move(), 1.4), 1);
        n.solver = false;
        assert_eq!(select_child(&n, n.position.to_move(), 1.4), 0);
    }
    #[test]
    fn draw_requires_every_reply_resolved() {
        let mut n = parent();
        n.actions_ready = true;
        n.children.push(child(Some(GameOutcome::Draw)));
        n.children.push(child(None));
        n.refresh_proof();
        assert_eq!(n.proof, None);
        n.children.pop();
        n.refresh_proof();
        assert_eq!(n.proof, Some(GameOutcome::Draw));
    }
    #[test]
    fn known_draw_bounds_value_without_forging_a_proof_or_changing_raw_statistics() {
        let mut n = parent();
        n.solver = true;
        n.value_sum = -72.;
        n.visits = 100;
        n.children = vec![child(Some(GameOutcome::Draw)), child(None)];
        let chooser = n.position.to_move();
        assert_eq!(n.bounded_value(-0.72, chooser), 0.);
        assert_eq!(n.bounded_value(0.72, chooser.opponent()), 0.);
        assert_eq!(n.bounded_value(0.3, chooser), 0.3);
        assert_eq!(n.bounded_value(-0.3, chooser.opponent()), -0.3);
        assert_eq!(n.proof, None);
        assert_eq!(n.value_sum, -72.);
        n.solver = false;
        assert_eq!(n.bounded_value(-0.72, chooser), -0.72);
    }
    #[test]
    fn recommendation_distinguishes_draw_floor_from_exploration_and_exact_win() {
        let p = Player::Host;
        let proofs = [
            Some(GameOutcome::Draw),
            None,
            None,
            None,
            Some(GameOutcome::Win(Player::Guest)),
        ];
        assert_eq!(
            root_eligible(&proofs, &[0., -0.8, 0., 0.2, -1.], p),
            vec![true, false, false, true, false]
        );
        let wins = [Some(GameOutcome::Draw), Some(GameOutcome::Win(p)), None];
        assert_eq!(
            root_eligible(&wins, &[0., -0.9, 0.9], p),
            vec![false, true, false]
        );
        assert_eq!(
            root_eligible(&[None, None], &[-0.8, -0.9], p),
            vec![true, true]
        );
    }
}
