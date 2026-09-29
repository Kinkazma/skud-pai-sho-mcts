//! V4 real-root tactical discovery, independent of the learned prior.
//! Only regulatory terminal successors become proofs. Work is separately counted.
use super::*;
pub(super) fn discover(
    root: &mut Node,
    cache: &mut Cache,
    model: &MicroModel,
    deadline: Option<Instant>,
) -> Result<usize, String> {
    discover_inner(root,cache,model,deadline,None)
}
pub(super) fn discover_with_successors(
    root:&mut Node,cache:&mut Cache,model:&MicroModel,deadline:Option<Instant>,
    successors:&mut Vec<Option<Position>>,
)->Result<usize,String> {
    discover_inner(root,cache,model,deadline,Some(successors))
}
fn discover_inner(
    root:&mut Node,cache:&mut Cache,model:&MicroModel,deadline:Option<Instant>,
    mut successors:Option<&mut Vec<Option<Position>>>,
)->Result<usize,String> {
    if root.tactical_scanned || root.proof.outcome.is_some() {
        return Ok(0);
    }
    let policy = root.inference.policy(model)?;
    if let Some(rows)=successors.as_mut() {rows.resize_with(policy.actions.len(),||None);}
    root.children.resize_with(policy.actions.len(), || None);
    let chooser = root.inference.position.to_move();
    let mut checks = 0;
    // All legal immediate successors, without spending visits on prior ranking.
    for (i, action) in policy.actions.iter().enumerate() {
        if deadline.is_some_and(|t| paisho_platform::training_time::now() >= t) {
            return Ok(checks);
        }
        if root.children[i]
            .as_ref()
            .is_some_and(|c| c.proof.outcome.is_some())
        {
            if let Some(rows)=successors.as_mut() {rows[i]=Some(root.children[i].as_ref().unwrap().inference.position.clone());}
            continue;
        }
        // The retained child's immutable position is already the exact result
        // of this action. Count the same check without replaying that action.
        if let Some(child) = root.children[i].as_ref() {
            debug_assert_eq!(child.inference.position.outcome(), GameOutcome::Ongoing);
            checks += 1;
            if let Some(rows)=successors.as_mut() {rows[i]=Some(child.inference.position.clone());}
            continue;
        }
        let mut next = root.inference.position.clone();
        next.apply(*action).map_err(|e| e.to_string())?;
        checks += 1;
        if next.outcome() != GameOutcome::Ongoing {
            let outcome = next.outcome();
            root.children[i] = Some(Box::new(Node::new(cache.get(next, model))));
            if let Some(rows)=successors.as_mut() {rows[i]=Some(root.children[i].as_ref().unwrap().inference.position.clone());}
            root.proof
                .child_solved(outcome, chooser, policy.actions.len());
            if outcome == GameOutcome::Win(chooser) {
                root.tactical_scanned = true;
                return Ok(checks);
            }
        } else if let Some(rows)=successors.as_mut() {
            rows[i]=Some(next);
        }
    }
    root.tactical_scanned = true;
    Ok(checks)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_one_simulation_root_finds_and_certifies_a_win_with_arbitrary_prior() {
        let old: paisho_core::GameRecord =
            include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let (record, _) = old
            .replay_prefix_with_rules(paisho_core::RuleProfileId::SkudPaiShoGen5V1)
            .unwrap();
        let mut position = record.initial_position();
        for a in &record.actions()[..record.actions().len() - 1] {
            position.apply(*a).unwrap();
        }
        let model = MicroModel::seeded(321).with_spatial_policy();
        let mut session = MicroMctsSession::new(Arc::new(model));
        let report = session
            .search_with_options(
                &position,
                1,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    dirichlet_fraction: 0.9,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(report.proven_value, Some(1));
        assert!(report.tactical_evaluations > 0);
        assert_eq!(report.simulations, 0);
        assert!(report.example(1.0, 1.0).is_ok());
        session.certificate(100).unwrap().verify(&position).unwrap();
        position
            .apply(report.actions[report.selected_index])
            .unwrap();
        assert_ne!(position.outcome(), GameOutcome::Ongoing);
    }
}
