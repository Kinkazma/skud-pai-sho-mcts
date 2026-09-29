//! One internal capture ordering override per node, using existing legal lists.
//! Deliberately opt-in: improves coverage, not a proof that captures are good.
use super::*;
impl Node {
    pub(super) fn take_expansion_action(&mut self, depth: usize) -> Action {
        let next = *self.ranked_unexpanded.last().expect("prepared expansion");
        if self.capture_coverage && depth > 0 && !self.capture_probe_done {
            // An already evaluated immediate win retains priority over the probe.
            let win = self.prepared.iter().any(|c| {
                c.action == next
                    && c.position.outcome() == GameOutcome::Win(self.position.to_move())
            });
            if !win {
                self.capture_probe_done = true;
                let index = self.ranked_unexpanded.iter().rposition(|a| match a {
                    Action::Arrange { to, .. } => self
                        .position
                        .board()
                        .get(*to)
                        .is_some_and(|t| t.owner != self.position.to_move()),
                    _ => false,
                });
                if let Some(i) = index {
                    return self.ranked_unexpanded.remove(i);
                }
            }
        }
        self.ranked_unexpanded.pop().unwrap()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Node, Action, Action) {
        let record: paisho_core::GameRecord =
            include_str!("../../tests/fixtures/gen34-r5-capture.psr")
                .parse()
                .unwrap();
        let mut p = record.initial_position();
        for &a in &record.actions()[..6] {
            p.apply(a).unwrap();
        }
        let capture = record.actions()[6];
        let other = legal_actions(&p)
            .into_iter()
            .find(|a| !matches!(a, Action::Arrange { to, .. } if p.board().get(*to).is_some()))
            .unwrap();
        (
            Node::with_ordered_actions(p, vec![capture, other]),
            capture,
            other,
        )
    }
    #[test]
    fn only_one_capture_override_and_root_untouched() {
        let (mut n, capture, other) = fixture();
        assert_eq!(n.take_expansion_action(1), other);
        let (mut n, _, _) = fixture();
        n.capture_coverage = true;
        assert_eq!(n.take_expansion_action(0), other);
        let (mut n, _, _) = fixture();
        n.capture_coverage = true;
        assert_eq!(n.take_expansion_action(1), capture);
        assert!(n.capture_probe_done);
        assert_eq!(n.take_expansion_action(1), other);
    }
    #[test]
    fn capture_does_not_replace_known_terminal_win() {
        let (mut n, _, other) = fixture();
        n.capture_coverage = true;
        // Use an actual terminal position for the same chooser from the record.
        let record: paisho_core::GameRecord =
            include_str!("../../tests/fixtures/gen34-r5-capture.psr")
                .parse()
                .unwrap();
        let terminal = record.replay().unwrap();
        assert_eq!(terminal.outcome(), GameOutcome::Win(n.position.to_move()));
        n.prepared.push(PreparedCandidate {
            action: other,
            position: terminal,
            leaf_value: Some(-1.),
        });
        assert_eq!(n.take_expansion_action(1), other);
        assert!(!n.capture_probe_done);
    }
}
