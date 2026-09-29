//! Opponent decisions are observations; only a regulatory win is imitation-ready.
use super::*;
use paisho_core::{Action, Position};
#[allow(clippy::too_many_arguments)]
pub(super) fn opponent(
    p: &Position,
    action: Action,
    actions: &[Action],
    snapshot: &Snapshot,
    actor: &str,
    budget: usize,
    id: usize,
    source: &str,
    decision: usize,
) -> std::result::Result<(Sample, Option<MicroProofCertificate>), String> {
    let state = snapshot.model.state_features(p);
    let embedding = snapshot.model.embed(&state);
    let mut next = p.clone();
    next.apply(action).map_err(|e| e.to_string())?;
    let win = next.outcome() == GameOutcome::Win(p.to_move());
    let index = actions
        .iter()
        .position(|a| *a == action)
        .ok_or("observed illegal action")?;
    let mut policy = vec![];
    let mut tactical = None;
    let mut certificate = None;
    let mut features = vec![];
    if win {
        let absolute = if p.to_move() == Player::Host { 1 } else { -1 };
        let c = MicroProofCertificate {
            outcome: absolute,
            children: vec![(
                action.to_string(),
                MicroProofCertificate {
                    outcome: absolute,
                    children: vec![],
                },
            )],
        };
        c.verify(p)?;
        certificate = Some(c);
        policy = vec![0.; actions.len()];
        policy[index] = 1.;
        let mut values = vec![None; actions.len()];
        values[index] = Some(1);
        features = actions
            .iter()
            .map(|a| micro_action_features(p, *a).to_vec())
            .collect();
        tactical = Some(TacticalEvidence {
            schema: "paisho-mcts-proof-v1".into(),
            root_value: Some(1),
            action_values: values,
            network_value: embedding.value,
            network_best_action: None,
        });
    }
    Ok((
        Sample {
            player: p.to_move(),
            q: embedding.value,
            saved: SavedMicroExample {
                evidence: Some(TargetEvidence { policy_support:false,
                    policy_coordinates: String::new(),
                    search_prior: vec![],
                    coupling_strength: None,
                    observed_value: None,
                    observed_psr: None,
                    estimated_value: Some(embedding.value),
                    value_weight: 1.,
                    policy_source: if win {
                        "verified-regulatory-win"
                    } else {
                        "observed-action-value-only"
                    }
                    .into(),
                    completed_action_values: vec![],
                    action_value_visits: vec![],
                    target_prior: vec![],
                    excluded_actions: vec![],
                    player: p.to_move().code().to_string(),
                    actor: actor.into(),
                }),
                rules: RULES.to_string(),
                source_run: source.into(),
                game_id: id.to_string(),
                decision,
                collector: snapshot.identity.clone(),
                budget,
                inherited_visits: 0,
                new_visits: vec![],
                policy_raw_visits: vec![],
                policy_pruned_visits: vec![],
                tactical,
                correction_priority: false,
                actions: if win {
                    actions.iter().map(ToString::to_string).collect()
                } else {
                    vec![]
                },
                state,
                action_features: features,
                policy,
                value: if win { 1. } else { 0. },
                policy_weight: if win { 1. } else { 0. },
                reason: String::new(),
            },
        },
        certificate,
    ))
}
