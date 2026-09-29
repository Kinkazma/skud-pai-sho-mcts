//! Search estimates, regulatory proofs and unknown action values stay distinct.
use super::*;

/// Build an explicitly verified winning support once, outside neural forwards.
/// Certificate children remain valid even for multi-move wins; additional
/// actions enter only when applying the rules produces an immediate win.
pub fn verified_winning_policy(
    position: &paisho_core::Position,
    legal: &[paisho_core::Action],
    certificate: &MicroProofCertificate,
) -> Result<Vec<f64>> {
    certificate.verify(position).map_err(invalid)?;
    let outcome = if position.to_move() == paisho_core::Player::Host {1} else {-1};
    if certificate.outcome != outcome {
        return Err(invalid("winning support requires a winning certificate"));
    }
    let mut policy=vec![0.;legal.len()];
    for (i,action) in legal.iter().enumerate() {
        let certified=certificate.children.iter().any(|(a,c)|a==&action.to_string() && c.outcome==outcome);
        let immediate=if certified {false} else {
            let mut next=position.clone();next.apply(*action)?;
            next.outcome()==paisho_core::GameOutcome::Win(position.to_move())
        };
        if certified || immediate {policy[i]=1.;}
    }
    let count=policy.iter().sum::<f64>();
    if count==0. {return Err(invalid("verified winning support is empty"));}
    for p in &mut policy {*p/=count;}
    Ok(policy)
}

pub(super) fn attach_winning_support(
    saved: &mut SavedMicroExample,
    position: &paisho_core::Position,
    certificate: &MicroProofCertificate,
) -> Result<()> {
    let legal=paisho_core::legal_actions(position);
    if saved.actions!=legal.iter().map(ToString::to_string).collect::<Vec<_>>() {
        return Err(invalid("winning support action alignment mismatch"));
    }
    let policy=verified_winning_policy(position,&legal,certificate)?;
    let tactical=saved.tactical.as_mut().ok_or_else(||invalid("winning support missing tactical provenance"))?;
    if tactical.root_value!=Some(1) || tactical.action_values.len()!=legal.len() {
        return Err(invalid("winning support missing verified root"));
    }
    for (i,p) in policy.iter().enumerate() {
        if *p>0. {
            if tactical.action_values[i].is_some_and(|q|q!=1) {
                return Err(invalid("regulatory win contradicts recorded action proof"));
            }
            tactical.action_values[i]=Some(1);
        }
    }
    let evidence=saved.evidence.as_mut().ok_or_else(||invalid("winning support missing teacher provenance"))?;
    evidence.policy_support=true;
    saved.policy=policy;
    saved.policy_raw_visits.clear();saved.policy_pruned_visits.clear();
    Ok(())
}

/// The input support has already been verified by the rules/certificate path.
/// Supervise these same wins consistently; never derive support from a Q value.
pub fn complete_verified_winning_values(
    values: &mut Vec<Option<f64>>,
    policy: &[f64],
) -> Result<()> {
    if policy.is_empty() || policy.iter().any(|p| !p.is_finite() || *p<0.)
        || (policy.iter().sum::<f64>()-1.).abs()>1e-8 {
        return Err(invalid("invalid verified winning support"));
    }
    if values.is_empty() {values.resize(policy.len(),None);}
    if values.len()!=policy.len() {return Err(invalid("verified support/Q shape mismatch"));}
    for (q,p) in values.iter_mut().zip(policy) {
        if *p>0. {
            if q.is_some_and(|q|q!=1.) {return Err(invalid("winning support contradicts action proof"));}
            *q=Some(1.);
        }
    }
    Ok(())
}
pub(super) fn action_value_targets(
    n: usize,
    evidence: Option<&TargetEvidence>,
    proof: Option<&TacticalEvidence>,
) -> Vec<Option<f64>> {
    let mut out = vec![None; n];
    if let Some(e) = evidence.filter(|e| {
        e.policy_source == "full-search-estimate" && e.completed_action_values.len() == n
    }) {
        for (i, q) in e.completed_action_values.iter().enumerate() {
            if !e.excluded_actions.get(i).copied().unwrap_or(false) {
                out[i] = Some(*q);
            }
        }
    }
    if let Some(p) = proof.filter(|p| p.action_values.len() == n) {
        for (to, known) in out.iter_mut().zip(&p.action_values) {
            if let Some(v) = known {
                *to = Some(*v as f64);
            }
        }
    }
    if out.iter().all(Option::is_none) {
        out.clear();
    }
    out
}
/// Conservative V3 auxiliary labels. Imputed completion remains in policy
/// evidence but is not supervision for an individually unexplored action.
/// Legacy new visits are sufficient positive evidence; zero with inherited
/// visits is unknown because the old archive has no per-action retained counts.
pub(super) fn action_value_targets_v3(
    n: usize,
    evidence: Option<&TargetEvidence>,
    proof: Option<&TacticalEvidence>,
    legacy_new_visits: &[usize],
) -> Vec<Option<f64>> {
    let mut out=vec![None;n];
    if let Some(e)=evidence.filter(|e|e.policy_source=="full-search-estimate" && e.completed_action_values.len()==n) {
        let visits=if e.action_value_visits.len()==n {&e.action_value_visits[..]} else {legacy_new_visits};
        if visits.len()==n {
            for (i,q) in e.completed_action_values.iter().enumerate() {
                if visits[i]>0 && !e.excluded_actions.get(i).copied().unwrap_or(false) {out[i]=Some(*q);}
            }
        }
    }
    if let Some(p)=proof.filter(|p|p.action_values.len()==n) {
        for (to,known) in out.iter_mut().zip(&p.action_values) {
            if let Some(v)=known {*to=Some(*v as f64);}
        }
    }
    if out.iter().all(Option::is_none) {out.clear();}
    out
}
pub fn certificate_action_values(
    p: &paisho_core::Position,
    legal: &[paisho_core::Action],
    c: &MicroProofCertificate,
) -> Result<Vec<Option<f64>>> {
    let mut values = vec![None; legal.len()];
    let sign = if p.to_move() == paisho_core::Player::Host {
        1.
    } else {
        -1.
    };
    for (a, child) in &c.children {
        let a: paisho_core::Action = a.parse()?;
        let i = legal
            .iter()
            .position(|x| *x == a)
            .ok_or_else(|| invalid("certificate Q action absent"))?;
        values[i] = Some(sign * child.outcome as f64);
    }
    Ok(values)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verified_support_expands_rules_wins_and_requires_explicit_v3_metadata() {
        let record:paisho_core::GameRecord=include_str!(concat!(env!("CARGO_MANIFEST_DIR"),"/tests/fixtures/gen5-multiple-proved-wins.psr")).parse().unwrap();
        let position=record.replay().unwrap();let legal=paisho_core::legal_actions(&position);
        let immediate:Vec<_>=legal.iter().map(|a|{let mut p=position.clone();p.apply(*a).unwrap();p.outcome()==paisho_core::GameOutcome::Win(position.to_move())}).collect();
        let first=immediate.iter().position(|v|*v).unwrap();
        assert_eq!(immediate.iter().filter(|v|**v).count(),4);
        let outcome=if position.to_move()==paisho_core::Player::Host {1}else{-1};
        let certificate=MicroProofCertificate {outcome,children:vec![(legal[first].to_string(),MicroProofCertificate {outcome,children:vec![]})]};
        let immutable=serde_json::to_vec(&certificate).unwrap();
        let model=MicroModel::seeded(39).with_spatial_policy();
        let mut saved=super::super::tactics::fixture();
        saved.rules=record.rules().to_string();saved.value=1.;
        saved.state=model.state_features(&position);
        saved.actions=legal.iter().map(ToString::to_string).collect();
        saved.action_features=legal.iter().map(|a|micro_action_features(&position,*a).to_vec()).collect();
        saved.policy=vec![0.;legal.len()];saved.policy[first]=1.;
        saved.policy_raw_visits.clear();saved.policy_pruned_visits.clear();saved.new_visits.clear();
        saved.tactical=Some(TacticalEvidence {schema:"paisho-mcts-proof-v1".into(),root_value:Some(1),action_values:(0..legal.len()).map(|i|(i==first).then_some(1)).collect(),network_value:0.,network_best_action:None});
        saved.evidence=Some(TargetEvidence {policy_support:false,policy_coordinates:String::new(),search_prior:vec![],coupling_strength:None,observed_value:None,observed_psr:None,estimated_value:Some(0.),value_weight:1.,policy_source:"verified-search".into(),completed_action_values:vec![],action_value_visits:vec![],target_prior:vec![],excluded_actions:vec![],player:position.to_move().code().to_string(),actor:"a".repeat(64)});
        assert!(!saved.example_for_rules_with_trusted_q(record.rules(),true).unwrap().policy_support);
        attach_winning_support(&mut saved,&position,&certificate).unwrap();
        assert_eq!(immutable,serde_json::to_vec(&certificate).unwrap());
        assert!(saved.policy.iter().zip(&immediate).all(|(p,win)|(*p>0.)==*win));
        let restored:SavedMicroExample=serde_json::from_slice(&serde_json::to_vec(&saved).unwrap()).unwrap();
        assert!(!restored.example_for_rules(record.rules()).unwrap().policy_support);
        assert!(restored.example_for_rules_with_trusted_q(record.rules(),true).unwrap().policy_support);
        let fresh=restored.example_for_rules_with_trusted_q(record.rules(),true).unwrap();
        let mut recalled_q=certificate_action_values(&position,&legal,&certificate).unwrap();
        assert_eq!(recalled_q.iter().flatten().count(),1);
        complete_verified_winning_values(&mut recalled_q,&fresh.policy).unwrap();
        assert_eq!(recalled_q,fresh.action_values);
        assert_eq!(recalled_q.iter().flatten().count(),4);
        let mut unproved=restored;
        unproved.evidence.as_mut().unwrap().policy_source="full-search-estimate".into();
        assert!(unproved.example_for_rules_with_trusted_q(record.rules(),true).is_err());
    }
    #[test]
    fn unknown_actions_never_become_loss_targets_and_proofs_override_estimates() {
        let t = TacticalEvidence {
            schema: "paisho-mcts-proof-v1".into(),
            root_value: Some(1),
            action_values: vec![Some(1), None, Some(-1)],
            network_value: 0.,
            network_best_action: Some(1),
        };
        assert_eq!(
            action_value_targets(3, None, Some(&t)),
            vec![Some(1.), None, Some(-1.)]
        );
        assert!(action_value_targets(3, None, None).is_empty());
        let e = TargetEvidence { policy_support:false,
            policy_coordinates: "raw-policy-v1".into(),
            search_prior: vec![],
            coupling_strength: None,
            observed_value: None,
            observed_psr: None,
            estimated_value: Some(0.),
            value_weight: 0.25,
            policy_source: "full-search-estimate".into(),
            completed_action_values: vec![0.1, 0.2, 0.3],
            action_value_visits: vec![],
            target_prior: vec![1. / 3.; 3],
            excluded_actions: vec![false, false, true],
            player: "H".into(),
            actor: "actor".into(),
        };
        assert_eq!(
            action_value_targets(3, Some(&e), Some(&t)),
            vec![Some(1.), Some(0.2), Some(-1.)]
        );
        assert_eq!(action_value_targets_v3(3,Some(&e),Some(&t),&[0,0,0]),vec![Some(1.),None,Some(-1.)]);
        assert_eq!(action_value_targets_v3(3,Some(&e),Some(&t),&[0,1,0]),vec![Some(1.),Some(0.2),Some(-1.)]);
        assert!(action_value_targets_v3(3,Some(&e),None,&[]).is_empty());
        let mut complete=e.clone();complete.action_value_visits=vec![0,12,0];
        assert_eq!(action_value_targets_v3(3,Some(&complete),None,&[10,0,0]),vec![None,Some(0.2),None]);
        let old_json=serde_json::to_value(&e).unwrap();
        assert!(old_json.get("action_value_visits").is_none());
        let restored:TargetEvidence=serde_json::from_value(old_json).unwrap();
        assert!(restored.action_value_visits.is_empty());
        complete.validate().unwrap();
        complete.action_value_visits.pop();
        assert!(complete.validate().is_err());
    }
    #[test]
    fn v3_dense_conversion_preserves_teacher_and_uses_only_positive_visit_evidence() {
        let evidence=TargetEvidence { policy_support:false,
            policy_coordinates:"raw-policy-v1".into(), search_prior:vec![], coupling_strength:None,
            observed_value:None, observed_psr:None, estimated_value:Some(0.2),value_weight:0.25,
            policy_source:"full-search-estimate".into(), completed_action_values:vec![0.1,0.2,0.3],
            action_value_visits:vec![], target_prior:vec![1./3.;3], excluded_actions:vec![],
            player:"H".into(), actor:"actor".into(),
        };
        let policy=micro_softmax(&[0.1,0.2,0.3]).unwrap();
        let mut saved=SavedMicroExample { structured: Vec::new(),
            evidence:Some(evidence), rules:paisho_core::RuleProfileId::CURRENT.to_string(),
            source_run:"test".into(), game_id:"1".into(),decision:1,collector:"a".repeat(64),budget:8,
            inherited_visits:10,new_visits:vec![0,1,0],policy_raw_visits:vec![],policy_pruned_visits:vec![],
            tactical:None,correction_priority:false,actions:vec!["a".into(),"b".into(),"c".into()],
            state:vec![0.;417],action_features:vec![vec![0.;32];3],policy,value:0.2,policy_weight:1.,reason:"test".into(),
        };
        let bytes=serde_json::to_vec(&saved).unwrap();
        let old=saved.example().unwrap();
        let corrected=saved.example_for_rules_with_trusted_q(paisho_core::RuleProfileId::CURRENT,true).unwrap();
        assert_eq!(old.action_values,vec![Some(0.1),Some(0.2),Some(0.3)]);
        assert_eq!(corrected.action_values,vec![None,Some(0.2),None]);
        assert_eq!(old.state,corrected.state);assert_eq!(old.actions,corrected.actions);
        assert_eq!(old.policy,corrected.policy);assert_eq!(old.value,corrected.value);
        assert_eq!(bytes,serde_json::to_vec(&saved).unwrap());
        saved.evidence.as_mut().unwrap().action_value_visits=vec![10,1,0];
        let restored:SavedMicroExample=serde_json::from_slice(&serde_json::to_vec(&saved).unwrap()).unwrap();
        let corrected=restored.example_for_rules_with_trusted_q(paisho_core::RuleProfileId::CURRENT,true).unwrap();
        assert_eq!(corrected.action_values,vec![Some(0.1),Some(0.2),None]);
    }
}
