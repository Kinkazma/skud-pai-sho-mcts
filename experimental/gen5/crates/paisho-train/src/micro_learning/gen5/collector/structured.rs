//! Consequences of the already-played action, without another search or reply census.
use super::*;
use paisho_core::{Action, Position};

pub(super) fn attach(
    saved: &mut SavedMicroExample,
    p: &Position,
    a: Action,
    q: &Position,
    certificate: Option<&MicroProofCertificate>,
) -> std::result::Result<(), String> {
    let name = a.to_string();
    if saved.actions.is_empty() {
        saved.actions.push(name.clone());
        saved
            .action_features
            .push(micro_action_features(p, a).to_vec());
        saved.policy.push(1.); // policy_weight stays zero: no imitation of an unproved move.
        if let Some(t) = &mut saved.tactical {
            t.action_values = vec![t.root_value];
            t.network_best_action = None;
        }
    }
    let i = saved
        .actions
        .iter()
        .position(|s| s == &name)
        .ok_or("structured played action missing")?;
    if saved.structured.is_empty() {
        saved.structured = vec![None; saved.actions.len()];
    }
    let mut threat = micro_immediate_threat(q, p.to_move(), 0)?;
    if threat.label().is_none() && matches!(threat, MicroImmediateThreat::Unknown { .. }) {
        if let Some(child) =
            certificate.and_then(|c| c.children.iter().find(|(s, _)| s == &name).map(|(_, c)| c))
        {
            // Minimax outcome alone is not an immediate threat: replay each proposed witness.
            for (reply, _) in child.children.iter().take(4) {
                let reply: Action = reply
                    .parse()
                    .map_err(|e: paisho_core::ActionNotationError| e.to_string())?;
                if let Ok(t) = micro_verify_threat_witness(q, p.to_move(), reply) {
                    if t.label() == Some(true) {
                        threat = t;
                        break;
                    }
                }
            }
        }
    }
    let before = MicroRelations::extract(p, p.to_move());
    let target = MicroStructuredTarget::from_successor(p, a, q, &before, &threat);
    target.validate()?;
    saved.structured[i] = Some(target);
    Ok(())
}

/// A real next decision can certify the previous action's remaining threat.
/// Bonus decisions by the same player are deliberately excluded.
pub(super) fn observed_reply(
    game: &mut Played,
    p: &Position,
    reply: Action,
    q: &Position,
    decision: usize,
) {
    if q.outcome() != GameOutcome::Win(p.to_move()) {
        return;
    }
    let Some(previous_action) = game.record.actions().last().map(ToString::to_string) else {
        return;
    };
    if let Some(previous) = game
        .samples
        .iter_mut()
        .rev()
        .find(|s| s.saved.decision + 1 == decision && s.player != p.to_move())
    {
        if let Some(i) = previous
            .saved
            .actions
            .iter()
            .position(|a| a == &previous_action)
        {
            if let Some(Some(t)) = previous.saved.structured.get_mut(i) {
                t.record_threat(&MicroImmediateThreat::Present {
                    witness: reply,
                    examined: 1,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn observed_actions_from_both_seats_teach_consequences_without_policy_imitation() {
        let source: GameRecord = include_str!(
            "../../../../examples/gen5_structured_corpus/fixtures/cross_owner_lotus.psr"
        )
        .parse()
        .unwrap();
        let record = source.replay_prefix_with_rules(RULES).unwrap().0;
        let model = MicroModel::seeded(17).with_relational(19);
        let mut p = record.initial_position();
        let mut seats = [0; 2];
        for (i, &a) in record.actions().iter().enumerate() {
            let mut s = crate::micro_learning::tactics::fixture();
            s.actions.clear();
            s.action_features.clear();
            s.policy.clear();
            s.new_visits.clear();
            s.policy_raw_visits.clear();
            s.policy_pruned_visits.clear();
            s.tactical = None;
            s.correction_priority = false;
            s.policy_weight = 0.;
            s.value = 0.;
            s.state = model.state_features(&p);
            s.decision = i + 1;
            let mut q = p.clone();
            q.apply(a).unwrap();
            attach(&mut s, &p, a, &q, None).unwrap();
            let ex = s.example_for_rules(RULES).unwrap();
            assert_eq!(ex.policy_weight, 0.);
            assert_eq!(ex.actions.len(), 1);
            assert!(ex.structured[0].is_some());
            assert_eq!(ex.structured[0].as_ref().unwrap().events[5], None);
            seats[usize::from(p.to_move() == Player::Guest)] += 1;
            p = q;
        }
        assert!(seats.iter().all(|n| *n > 0));
    }
}
