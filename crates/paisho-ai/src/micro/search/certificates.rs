//! Replayable minimax certificates, independent of neural weights and visits.
use super::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicroProofCertificate {
    /// Absolute outcome: Host=1, Guest=-1, draw=0.
    pub outcome: i8,
    pub children: Vec<(String, MicroProofCertificate)>,
}
fn code(outcome: GameOutcome) -> i8 {
    match outcome {
        GameOutcome::Win(Player::Host) => 1,
        GameOutcome::Win(Player::Guest) => -1,
        GameOutcome::Draw => 0,
        GameOutcome::Ongoing => unreachable!(),
    }
}
impl MicroProofCertificate {
    pub fn verify(&self, position: &Position) -> Result<GameOutcome, String> {
        self.check(position, &mut 100_000, 0)
    }
    fn check(
        &self,
        position: &Position,
        remaining: &mut usize,
        depth: usize,
    ) -> Result<GameOutcome, String> {
        if *remaining == 0 || depth > 256 || !(-1..=1).contains(&self.outcome) {
            return Err("invalid or oversized minimax certificate".into());
        }
        *remaining -= 1;
        let expected = match self.outcome {
            1 => GameOutcome::Win(Player::Host),
            -1 => GameOutcome::Win(Player::Guest),
            _ => GameOutcome::Draw,
        };
        if position.outcome() != GameOutcome::Ongoing {
            return if self.children.is_empty() && position.outcome() == expected {
                Ok(expected)
            } else {
                Err("terminal certificate mismatch".into())
            };
        }
        let legal = legal_actions(position);
        let indices: HashMap<_, _> = legal
            .iter()
            .copied()
            .enumerate()
            .map(|(i, a)| (a, i))
            .collect();
        let mut seen = vec![false; legal.len()];
        let mut results = vec![];
        for (name, child) in &self.children {
            let action: Action = name
                .parse()
                .map_err(|e| format!("invalid certificate action: {e}"))?;
            let index = *indices.get(&action).ok_or("illegal certificate action")?;
            if seen[index] {
                return Err("duplicate certificate action".into());
            }
            seen[index] = true;
            let mut next = position.clone();
            next.apply(legal[index]).map_err(|e| e.to_string())?;
            results.push(child.check(&next, remaining, depth + 1)?);
        }
        let chooser_win = GameOutcome::Win(position.to_move());
        let actual = if results.contains(&chooser_win) {
            chooser_win
        } else if !legal.is_empty() && seen.iter().all(|s| *s) {
            if results.contains(&GameOutcome::Draw) {
                GameOutcome::Draw
            } else {
                GameOutcome::Win(position.to_move().opponent())
            }
        } else {
            return Err("certificate does not cover adversarial replies".into());
        };
        if actual != expected {
            return Err("certificate minimax outcome mismatch".into());
        }
        Ok(actual)
    }
}
fn extract(node: &Node, left: &mut usize, depth: usize) -> Option<MicroProofCertificate> {
    // Keep nested certificates readable under serde_json's default recursion limit.
    if *left == 0 || depth > 32 {
        return None;
    }
    *left -= 1;
    let outcome = node.proof.outcome?;
    let mut certificate = MicroProofCertificate {
        outcome: code(outcome),
        children: vec![],
    };
    if node.inference.position.outcome() != GameOutcome::Ongoing {
        return Some(certificate);
    }
    let policy = node.inference.policy().ok()?;
    let win = outcome == GameOutcome::Win(node.inference.position.to_move());
    for (i, action) in policy.actions.iter().enumerate() {
        let child = node.children.get(i).and_then(Option::as_ref);
        if win && child.map_or(true, |c| c.proof.outcome != Some(outcome)) {
            continue;
        }
        certificate
            .children
            .push((action.to_string(), extract(child?, left, depth + 1)?));
        if win {
            break;
        }
    }
    Some(certificate)
}
impl MicroMctsSession {
    /// Export only genuinely solved trees; do not expand to build the certificate.
    pub fn certificate(&self, max_nodes: usize) -> Option<MicroProofCertificate> {
        let mut left = max_nodes;
        extract(self.root.as_ref()?, &mut left, 0)
    }
    /// Verify all replies before importing a proof under the exact supplied state.
    pub fn install_certificate(
        &mut self,
        position: &Position,
        certificate: &MicroProofCertificate,
    ) -> Result<(), String> {
        certificate.verify(position)?;
        fn build(
            p: Position,
            c: &MicroProofCertificate,
            cache: &mut Cache,
            model: &MicroModel,
        ) -> Result<Node, String> {
            let mut node = Node::new(cache.get(p.clone(), model));
            node.proof = proofs::Proof::new(match c.outcome {
                1 => GameOutcome::Win(Player::Host),
                -1 => GameOutcome::Win(Player::Guest),
                _ => GameOutcome::Draw,
            });
            if p.outcome() == GameOutcome::Ongoing {
                let policy = node.inference.policy()?;
                node.children.resize_with(policy.actions.len(), || None);
                let indices: HashMap<_, _> = policy
                    .actions
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, a)| (a, i))
                    .collect();
                for (name, child) in &c.children {
                    let action: Action = name
                        .parse()
                        .map_err(|e| format!("invalid certificate action: {e}"))?;
                    let i = *indices.get(&action).ok_or("certificate action missing")?;
                    let mut next = p.clone();
                    next.apply(policy.actions[i]).map_err(|e| e.to_string())?;
                    node.children[i] = Some(Box::new(build(next, child, cache, model)?));
                }
            }
            Ok(node)
        }
        self.root = Some(build(
            position.clone(),
            certificate,
            &mut self.cache,
            &self.model,
        )?);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Position, MicroProofCertificate, Action) {
        let record: paisho_core::GameRecord =
            include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let mut p = record.initial_position();
        for a in &record.actions()[..record.actions().len() - 1] {
            p.apply(*a).unwrap();
        }
        let action = *record.actions().last().unwrap();
        let mut next = p.clone();
        next.apply(action).unwrap();
        let c = MicroProofCertificate {
            outcome: code(next.outcome()),
            children: vec![(
                action.to_string(),
                MicroProofCertificate {
                    outcome: code(next.outcome()),
                    children: vec![],
                },
            )],
        };
        (p, c, action)
    }
    #[test]
    fn exact_certificate_survives_unrelated_weights_and_restarts() {
        let (p, c, action) = fixture();
        assert_eq!(c.verify(&p).unwrap(), GameOutcome::Win(p.to_move()));
        for seed in [1, 1001] {
            let mut s = MicroMctsSession::new(Arc::new(MicroModel::seeded(seed)));
            s.install_certificate(&p, &c).unwrap();
            let r = s
                .search_with_options(
                    &p,
                    256,
                    None,
                    MicroSearchOptions {
                        proof_search: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(r.actions[r.selected_index], action);
            assert_eq!(r.simulations, 0);
            assert_eq!(r.policy_target.iter().sum::<f64>(), 1.0);
            let restored = s.certificate(100).unwrap();
            assert_eq!(restored.verify(&p).unwrap(), c.verify(&p).unwrap());
        }
    }
    #[test]
    fn invented_terminal_and_wrong_prefix_are_rejected() {
        let (p, mut c, _) = fixture();
        c.children[0].1.outcome *= -1;
        assert!(c.verify(&p).is_err());
        let (_, c, _) = fixture();
        let start = Position::from_standard_setup(paisho_core::StandardSetup::balanced(
            paisho_core::BASIC_FLOWERS[0],
        ));
        assert!(c.verify(&start).is_err());
        let fake = MicroProofCertificate {
            outcome: 1,
            children: vec![],
        };
        assert!(fake.verify(&p).is_err());
    }
}
