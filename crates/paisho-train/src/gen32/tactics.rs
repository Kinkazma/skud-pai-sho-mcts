use super::*;
use crate::micro_learning::TacticalEvidence;

/// A known draw is a minimax lower bound, not an exact value while other moves
/// remain unresolved. Do not teach a negative value below that certified bound.
pub(super) fn respect_draw_floor(value: f64, evidence: Option<&TacticalEvidence>) -> f64 {
    if evidence.is_some_and(|t| t.action_values.contains(&Some(0))) {
        value.max(0.)
    } else {
        value
    }
}
/// Preserve raw visits; the corrected policy is explicitly a tactical target,
/// not an invented MCTS visit distribution.
pub(super) fn correct(
    s: &mut SavedMicroExample,
    model: &Gen32Model,
    p: &Position,
    proofs: &[Option<GameOutcome>],
    guard: &Gen32GuardReport,
    selected: usize,
) -> std::result::Result<(), String> {
    let player = p.to_move();
    let mut values: Vec<_> = proofs
        .iter()
        .map(|v| match v {
            Some(GameOutcome::Win(w)) => Some(if *w == player { 1 } else { -1 }),
            Some(GameOutcome::Draw) => Some(0),
            _ => None,
        })
        .collect();
    for &i in &guard.refuted {
        if values[i] != Some(1) {
            values[i] = Some(-1);
        }
    }
    if !values.iter().any(Option::is_some) {
        return Ok(());
    }
    let root = if values.contains(&Some(1)) {
        Some(1)
    } else if values.iter().all(Option::is_some) {
        values.iter().flatten().max().copied()
    } else {
        None
    };
    let allow = |i: usize| {
        if root == Some(1) {
            values[i] == Some(1)
        } else if root == Some(0) {
            values[i] == Some(0)
        } else {
            root == Some(-1) || values[i] != Some(-1)
        }
    };
    for i in 0..s.policy.len() {
        if !allow(i) {
            s.policy[i] = 0.;
        }
    }
    let sum: f64 = s.policy.iter().sum();
    if sum > 0. {
        for x in &mut s.policy {
            *x /= sum;
        }
    } else if allow(selected) {
        s.policy[selected] = 1.;
    } else {
        let n = (0..s.policy.len()).filter(|&i| allow(i)).count();
        for i in 0..s.policy.len() {
            s.policy[i] = if allow(i) { 1. / n as f64 } else { 0. };
        }
    }
    let state = micro_state_features(p);
    let embedding = model.policy.embed(&state);
    let best = s
        .action_features
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let a: [f64; 32] = a.as_slice().try_into().unwrap();
            (i, MicroModel::logit(&embedding, &a))
        })
        .max_by(|(i, a), (j, b)| a.total_cmp(b).then_with(|| j.cmp(i)))
        .map(|(i, _)| i);
    let evidence = TacticalEvidence {
        schema: "paisho-mcts-proof-v1".into(),
        root_value: root,
        action_values: values,
        network_value: model.value_at(p, player) as f64,
        network_best_action: best,
    };
    if let Some(v) = root {
        s.value = v as f64;
    }
    s.correction_priority = evidence.informative();
    s.tactical = Some(evidence);
    Ok(())
}

#[cfg(test)]
mod draw_tests {
    use super::*;
    #[test]
    fn completed_game_target_respects_draw_bound_without_inventing_exact_root_value() {
        let evidence = TacticalEvidence {
            schema: "paisho-mcts-proof-v1".into(),
            root_value: None,
            action_values: vec![Some(0), None],
            network_value: -0.8,
            network_best_action: Some(1),
        };
        assert_eq!(respect_draw_floor(-0.45, Some(&evidence)), 0.);
        assert_eq!(respect_draw_floor(0.25, Some(&evidence)), 0.25);
        assert_eq!(respect_draw_floor(-0.45, None), -0.45);
        assert_eq!(evidence.root_value, None);
    }
}
