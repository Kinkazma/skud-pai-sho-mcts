//! Incremental minimax proofs from regulatory terminals, with no extra expansion.
//! Absolute winners are essential: a bonus edge may retain the same player.
use super::*;

pub(super) struct Proof {
    pub outcome: Option<GameOutcome>,
    solved_children: usize,
    has_draw: bool,
}
impl Proof {
    pub fn new(outcome: GameOutcome) -> Self {
        Self {
            outcome: (outcome != GameOutcome::Ongoing).then_some(outcome),
            solved_children: 0,
            has_draw: false,
        }
    }
    /// Called exactly once when each legal child first becomes solved.
    pub fn child_solved(&mut self, outcome: GameOutcome, chooser: Player, legal: usize) {
        if self.outcome.is_some() || outcome == GameOutcome::Ongoing {
            return;
        }
        self.solved_children += 1;
        self.has_draw |= outcome == GameOutcome::Draw;
        if outcome == GameOutcome::Win(chooser) {
            self.outcome = Some(outcome);
        } else if legal > 0 && self.solved_children == legal {
            self.outcome = Some(if self.has_draw {
                GameOutcome::Draw
            } else {
                GameOutcome::Win(chooser.opponent())
            });
        }
    }
    pub fn value(&self, player: Player) -> Option<i8> {
        self.outcome.map(|o| match o {
            GameOutcome::Win(p) => {
                if p == player {
                    1
                } else {
                    -1
                }
            }
            GameOutcome::Draw => 0,
            GameOutcome::Ongoing => unreachable!(),
        })
    }
    /// A drawing response proves a lower bound, not the outcome of an unsolved node.
    pub fn bound(&self, estimate: f64) -> f64 {
        if self.has_draw {
            estimate.max(0.0)
        } else {
            estimate
        }
    }
}
/// Allocation and the final decision have different purposes. Once draws stop
/// consuming visits, their old visit counts cannot measure their desirability.
/// Keep searching unknown replies, but play a proved draw if no visited unknown
/// has a positive estimate. A positive estimate is never labelled as a proof.
pub(super) fn decision_allowed(
    proofs: &[Option<i8>],
    root: Option<i8>,
    values: &[f64],
    visits: &[usize],
) -> Vec<bool> {
    if root.is_some() || !proofs.contains(&Some(0)) || proofs.contains(&Some(1)) {
        return allowed(proofs, root);
    }
    let promising = |i: usize| proofs[i].is_none() && visits[i] > 0 && values[i] > 0.0;
    let has_promising = (0..proofs.len()).any(promising);
    (0..proofs.len())
        .map(|i| {
            if has_promising {
                promising(i)
            } else {
                proofs[i] == Some(0)
            }
        })
        .collect()
}
/// Prefer proven wins; accept a draw only when the whole node is solved.
/// A known loss can be ignored only while another alternative remains.
pub(super) fn allowed(values: &[Option<i8>], root: Option<i8>) -> Vec<bool> {
    let wins = values.contains(&Some(1));
    let all_loss = !values.is_empty() && values.iter().all(|v| *v == Some(-1));
    values
        .iter()
        .map(|v| {
            if wins {
                *v == Some(1)
            } else if root == Some(0) {
                *v == Some(0)
            } else {
                all_loss || *v != Some(-1)
            }
        })
        .collect()
}
pub(super) fn mask(policy: &mut [f64], allowed: &[bool], fallback: usize) {
    if allowed.is_empty() || allowed.iter().all(|v| *v) {
        return;
    }
    for (p, keep) in policy.iter_mut().zip(allowed) {
        if !keep {
            *p = 0.0;
        }
    }
    let mass: f64 = policy.iter().sum();
    if mass > 0.0 {
        for p in policy {
            *p /= mass;
        }
    } else {
        policy[fallback] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_regulatory_win_overrides_retained_visits_and_is_reusable() {
        let record: paisho_core::GameRecord =
            include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let mut position = record.initial_position();
        for a in &record.actions()[..record.actions().len() - 1] {
            position.apply(*a).unwrap();
        }
        let winning = *record.actions().last().unwrap();
        let mut session = MicroMctsSession::new(Arc::new(MicroModel::seeded(3)));
        let inference = session.cache.get(position.clone(), &session.model);
        let policy = inference.policy(&session.model).unwrap();
        let win_index = policy.actions.iter().position(|a| *a == winning).unwrap();
        let mut root = Node::new(inference);
        root.children.resize_with(policy.actions.len(), || None);
        let rival = policy
            .actions
            .iter()
            .enumerate()
            .find_map(|(i, a)| {
                let mut next = position.clone();
                next.apply(*a).unwrap();
                (i != win_index && next.outcome() == GameOutcome::Ongoing).then_some((i, next))
            })
            .unwrap();
        let mut rival_child = Node::new(session.cache.get(rival.1, &session.model));
        rival_child.visits = 10000;
        rival_child.value_sum = 9000.0;
        root.children[rival.0] = Some(Box::new(rival_child));
        root.visits = 10000;
        simulate_mode(
            &mut root,
            &mut session.cache,
            &session.model,
            1.5,
            96,
            0,
            Some(win_index),
            None,
            MicroSearchMode::Puct,
            0.0,
            true,
        )
        .unwrap();
        assert_eq!(root.proof.value(position.to_move()), Some(1));
        session.root = Some(root);
        let r = session
            .search_with_options(
                &position,
                64,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    forced_playout_strength: 2.0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(r.simulations, 0); // solved retained root, not a timeout
        assert_eq!(r.selected_index, win_index);
        assert_eq!(r.visits[rival.0], 10000);
        assert_eq!(r.policy_target[win_index], 1.0);
        assert_eq!(r.pruned_visits[rival.0], 10000);
        assert!(r.example(1.0, 1.0).is_ok());
        position.apply(r.actions[r.selected_index]).unwrap();
        assert_ne!(position.outcome(), GameOutcome::Ongoing);
        assert!(session.advance(winning).unwrap());
        assert_eq!(
            session.root.as_ref().unwrap().proof.outcome,
            Some(position.outcome())
        );
    }
    #[test]
    fn a_cooperative_line_is_not_a_forced_win_and_unknown_is_not_draw() {
        let mut p = Proof::new(GameOutcome::Ongoing);
        p.child_solved(GameOutcome::Win(Player::Host), Player::Guest, 3);
        assert_eq!(p.outcome, None);
        p.child_solved(GameOutcome::Draw, Player::Guest, 3);
        assert_eq!(p.outcome, None);
        p.child_solved(GameOutcome::Win(Player::Host), Player::Guest, 3);
        assert_eq!(p.outcome, Some(GameOutcome::Draw));
        assert_eq!(Proof::new(GameOutcome::Ongoing).outcome, None);
    }
    #[test]
    fn proof_uses_actual_chooser_and_all_replies_for_a_loss() {
        for chooser in [Player::Host, Player::Guest] {
            let mut win = Proof::new(GameOutcome::Ongoing);
            win.child_solved(GameOutcome::Win(chooser), chooser, 999);
            assert_eq!(win.value(chooser), Some(1));
            assert_eq!(win.value(chooser.opponent()), Some(-1));
            let mut loss = Proof::new(GameOutcome::Ongoing);
            loss.child_solved(GameOutcome::Win(chooser.opponent()), chooser, 2);
            assert_eq!(loss.outcome, None);
            loss.child_solved(GameOutcome::Win(chooser.opponent()), chooser, 2);
            assert_eq!(loss.value(chooser), Some(-1));
        }
    }
    #[test]
    fn proof_masks_loss_and_forcing_cannot_dilute_a_winning_target() {
        let v = [Some(-1), None, Some(1)];
        let mut target = [0.8, 0.19, 0.01];
        mask(&mut target, &allowed(&v, Some(1)), 2);
        assert_eq!(target, [0.0, 0.0, 1.0]);
        let mut target = [1.0, 0.0];
        mask(&mut target, &allowed(&[Some(-1), None], None), 1);
        assert_eq!(target, [0.0, 1.0]);
        assert_eq!(allowed(&[Some(-1), Some(-1)], Some(-1)), vec![true, true]);
    }
}
