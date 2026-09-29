//! Minimum-depth PUCT, tree inspection and isolated expansion-budget experiments.
//! Campaign floors preserve the requested root rollouts with up to budget*floor
//! new nodes. Zero preserves historical search; the maximum remains independent.
use super::*;

#[derive(Clone,Debug,Default,serde::Serialize)]
pub struct MicroDepthTrialStats {
    pub expansions: usize,
    pub leaves_by_depth: Vec<usize>,
    pub terminal_or_proved_leaves: usize,
    pub unresolved_below_floor: usize,
}
pub(super) struct Floor {
    pub minimum: usize,
    pub remaining: usize,
    pub stats: MicroDepthTrialStats,
}
impl Floor {
    pub fn new(minimum: usize, budget: usize) -> Self {
        Self {minimum,remaining:budget,stats:MicroDepthTrialStats::default()}
    }
    pub fn leaf(&mut self, depth: usize, solved: bool) {
        if self.stats.leaves_by_depth.len()<=depth {self.stats.leaves_by_depth.resize(depth+1,0);}
        self.stats.leaves_by_depth[depth]+=1;
        self.stats.terminal_or_proved_leaves+=usize::from(solved);
        self.stats.unresolved_below_floor+=usize::from(!solved && depth<self.minimum);
    }
}


impl MicroMctsSession {
    /// Extend unresolved new leaves to this minimum number of decisions from
    /// the current root. This is NOT a maximum: retained branches can go deeper.
    /// Each search permits simulations*minimum new nodes, preserving its root
    /// rollout budget. Proofs, terminal positions and deadlines may finish early.
    /// Zero restores historical search. Changing the protocol clears the tree,
    /// while changing only the rollout budget preserves useful retained visits.
    pub fn set_minimum_search_depth(&mut self, minimum: usize) -> Result<(), String> {
        if minimum > self.maximum_depth { return Err("minimum search depth exceeds maximum".into()); }
        if self.minimum_search_depth != minimum || self.diagnostic_depth_floor.is_some() {
            self.root = None;
            self.root_successor_values = None;
        }
        self.minimum_search_depth = minimum;
        self.diagnostic_depth_floor = None;
        self.diagnostic_depth_expansion_budget = None;
        Ok(())
    }
    pub fn minimum_search_depth(&self) -> usize { self.minimum_search_depth }

    /// Isolated PUCT trial. None preserves production; Some(0) instruments the
    /// unchanged algorithm. Positive floors extend newly expanded leaves along
    /// current interior PUCT, spending at most `simulations` new nodes in total.
    /// Thus the root rollout count may be LOWER than the supplied node budget.
    /// Terminal/proved leaves and budget exhaustion can stop before the floor.
    #[doc(hidden)]
    pub fn diagnostic_set_depth_floor(&mut self, floor: Option<usize>) -> Result<(),String> {
        if self.minimum_search_depth > 0 {return Err("disable campaign floor before diagnostic override".into());}
        if floor.is_some_and(|f|f>self.maximum_depth) {return Err("floor exceeds depth cap".into());}
        if floor!=self.diagnostic_depth_floor {
            self.root=None; self.root_successor_values=None; self.diagnostic_depth_floor=floor;
        }
        Ok(())
    }
    /// Optional separate expansion cap for equal-root-rollout comparisons.
    /// None uses the supplied simulation budget as the shared new-node cap.
    /// This deliberately permits additional computation, which must be reported.
    #[doc(hidden)]
    pub fn diagnostic_set_depth_expansion_budget(&mut self, budget: Option<usize>) -> Result<(),String> {
        if budget==Some(0) {return Err("positive expansion budget required".into());}
        if budget!=self.diagnostic_depth_expansion_budget {
            self.root=None;self.root_successor_values=None;self.diagnostic_depth_expansion_budget=budget;
        }
        Ok(())
    }
    #[doc(hidden)]
    pub fn diagnostic_depth_trial_stats(&self) -> &MicroDepthTrialStats { &self.diagnostic_depth_stats }

    #[doc(hidden)]
    pub fn diagnostic_maximum_depth(&self) -> usize { self.maximum_depth }

    /// Changing the cap invalidates old search statistics, preserving only
    /// model/state inference. This setter is not exposed by campaign options.
    #[doc(hidden)]
    pub fn diagnostic_set_maximum_depth(&mut self, depth: usize) -> Result<(),String> {
        if depth==0 || depth>256 || self.minimum_search_depth>depth || self.diagnostic_depth_floor.is_some_and(|f|f>depth) { return Err("diagnostic depth must be 1..=256 and at least the minimum".into()); }
        if depth!=self.maximum_depth {
            self.root=None; self.root_successor_values=None; self.maximum_depth=depth;
        }
        Ok(())
    }

    /// Export a child proof without changing retained search statistics.
    #[doc(hidden)]
    pub fn diagnostic_action_certificate(&self, index: usize) -> Option<MicroProofCertificate> {
        let node=self.root.as_ref()?.children.get(index)?.as_ref()?;
        certificates::extract(node,&self.model,&mut 100_000,0)
    }

    /// Counts distinct visited nodes by depth, root at zero. This is a retained
    /// tree census, not a distribution of simulation lengths. Cold sessions
    /// isolate one search; reused sessions also include previous searches.
    #[doc(hidden)]
    pub fn diagnostic_visited_nodes_by_depth(&self) -> Vec<usize> {
        let mut result=vec![];
        let mut stack:Vec<_>=self.root.as_ref().into_iter().map(|n|(n,0)).collect();
        while let Some((node,depth))=stack.pop() {
            if node.visits>0 {
                if result.len()<=depth {result.resize(depth+1,0);}
                result[depth]+=1;
            }
            for child in node.children.iter().flatten() {stack.push((child.as_ref(),depth+1));}
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inspection_does_not_change_search_and_cap_changes_clear_the_tree() {
        let p=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3));
        let model=Arc::new(MicroModel::seeded(19).with_spatial_policy());
        let mut a=MicroMctsSession::new(model.clone());let mut b=MicroMctsSession::new(model);
        assert_eq!(a.diagnostic_maximum_depth(),96);
        let r=a.search_until(&p,128,None).unwrap();
        assert_eq!(a.diagnostic_visited_nodes_by_depth()[0],1);
        assert_eq!(a.retained_visits(),128);
        let s=b.search_until(&p,128,None).unwrap();
        assert_eq!(r.visits,s.visits);assert_eq!(r.values,s.values);assert_eq!(r.policy_target,s.policy_target);
        let r=a.search_until(&p,128,None).unwrap();let s=b.search_until(&p,128,None).unwrap();
        assert_eq!(r.visits,s.visits);assert_eq!(r.values,s.values);
        a.diagnostic_set_maximum_depth(4).unwrap();assert_eq!(a.retained_visits(),0);
        a.search_until(&p,128,None).unwrap();assert!(a.diagnostic_visited_nodes_by_depth().len()<=5);
        assert!(a.diagnostic_set_maximum_depth(0).is_err());
    }
    #[test]
    fn zero_floor_is_bit_exact_and_positive_floors_spend_one_shared_budget() {
        let p=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3));
        let model=Arc::new(MicroModel::seeded(19).with_spatial_policy());
        for budget in [32,64,128,256,512] {
            let mut ordinary=MicroMctsSession::new(model.clone());
            let mut measured=MicroMctsSession::new(model.clone());
            measured.diagnostic_set_depth_floor(Some(0)).unwrap();
            for _ in 0..2 {
                let a=ordinary.search_until(&p,budget,None).unwrap();
                let b=measured.search_until(&p,budget,None).unwrap();
                assert_eq!(a.visits,b.visits); assert_eq!(a.selected_index,b.selected_index);
                assert_eq!(a.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                assert_eq!(a.policy_target.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),b.policy_target.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                assert_eq!(measured.diagnostic_depth_trial_stats().leaves_by_depth.iter().sum::<usize>(),b.simulations);
            }
            for minimum in [5,6,8,12,16,32] {
                let mut s=MicroMctsSession::new(model.clone());s.diagnostic_set_depth_floor(Some(minimum)).unwrap();
                let report=s.search_until(&p,budget,None).unwrap();let stats=s.diagnostic_depth_trial_stats();
                assert_eq!(stats.expansions,budget); assert!(report.simulations<=budget);
                assert_eq!(stats.leaves_by_depth.iter().sum::<usize>(),report.simulations);
                assert!(stats.unresolved_below_floor<=1); // Only last credit can truncate.
                assert!(stats.leaves_by_depth.len()>minimum);
                assert_eq!(s.retained_visits(),report.simulations);
                assert_eq!(report.visits.iter().sum::<usize>(),report.simulations);
                assert!(report.values.iter().all(|v|v.is_finite() && v.abs()<=1.));
            }
        }
    }

    #[test]
    fn prolongation_preserves_exact_terminal_proofs() {
        let model=MicroModel::seeded(23).with_spatial_policy();
        let record:paisho_core::GameRecord=include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr").parse().unwrap();
        let (record,_)=record.replay_prefix_with_rules(paisho_core::RuleProfileId::SkudPaiShoGen5V1).unwrap();
        let mut p=record.initial_position();
        for &action in &record.actions()[..record.actions().len()-1] {p.apply(action).unwrap();}
        let player=p.to_move();let action=*record.actions().last().unwrap();
        let session=MicroMctsSession::new(Arc::new(model.clone()));let mut cache=session.cache;
        let mut root=Node::new(cache.get(p.clone(),&model));
        let index=root.inference.policy(&model).unwrap().actions.iter().position(|&a|a==action).unwrap();
        let mut floor=depth_probe::Floor::new(32,32);
        let value=simulate_mode_with_root_fpu(&mut root,&mut cache,&model,1.5,96,0,Some(index),None,MicroSearchMode::Puct,0.,true,None,Some(&mut floor)).unwrap();
        p.apply(action).unwrap();assert_eq!(p.outcome(),GameOutcome::Win(player));
        assert_eq!(value,1.);assert_eq!(root.proof.value(player),Some(1));
        assert_eq!(floor.stats.expansions,1);assert_eq!(floor.stats.terminal_or_proved_leaves,1);
    }

    #[test]
    fn separate_expansion_allowance_preserves_requested_root_rollouts() {
        let p=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3));
        let model=Arc::new(MicroModel::seeded(19).with_spatial_policy());
        for floor in [5,6,8,12,16,32] {
            let mut s=MicroMctsSession::new(model.clone());s.diagnostic_set_depth_floor(Some(floor)).unwrap();
            s.diagnostic_set_depth_expansion_budget(Some(32*floor)).unwrap();
            let r=s.search_until(&p,32,None).unwrap();let stats=s.diagnostic_depth_trial_stats();
            assert_eq!(r.simulations,32);assert_eq!(r.visits.iter().sum::<usize>(),32);
            assert!(stats.expansions>32 && stats.expansions<=32*floor);
            assert_eq!(stats.unresolved_below_floor,0);
            assert_eq!(stats.leaves_by_depth.iter().sum::<usize>(),32);
        }
    }

    #[test]
    fn campaign_floor_matches_tested_algorithm_at_both_training_budgets() {
        let p=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3));
        let model=Arc::new(MicroModel::seeded(19).with_spatial_policy());
        for budget in [256,512] {
            let mut campaign=MicroMctsSession::new(model.clone());
            campaign.set_minimum_search_depth(5).unwrap();
            let mut trial=MicroMctsSession::new(model.clone());
            trial.diagnostic_set_depth_floor(Some(5)).unwrap();
            trial.diagnostic_set_depth_expansion_budget(Some(budget*5)).unwrap();
            let a=campaign.search_until(&p,budget,None).unwrap();
            let b=trial.search_until(&p,budget,None).unwrap();
            assert_eq!(format!("{a:?}"),format!("{b:?}"));
            assert_eq!(a.simulations,budget);
            let stats=campaign.diagnostic_depth_trial_stats();
            assert_eq!(stats.unresolved_below_floor,0);
            assert!(stats.expansions<=budget*5);
            assert_eq!(campaign.diagnostic_maximum_depth(),96);
            let next=campaign.search_until(&p,768-budget,None).unwrap();
            assert_eq!(next.inherited_visits,budget);
            assert_eq!(next.simulations,768-budget);
            assert!(campaign.diagnostic_set_maximum_depth(4).is_err());
            campaign.set_minimum_search_depth(6).unwrap();
            assert_eq!(campaign.retained_visits(),0);
        }
    }

    #[test]
    fn campaign_minimum_does_not_cap_deeper_retained_paths() {
        let p=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3));
        let template=MicroModel::seeded(19).with_spatial_policy();
        let neutral=MicroModel::from_parameters(vec![0.;template.parameters().len()]).unwrap();
        let mut s=MicroMctsSession::new(Arc::new(neutral));
        // Equal values and no exploration concentrate this controlled search on
        // the retained branch, allowing it to pass the minimum deterministically.
        s.exploration=0.;
        s.set_minimum_search_depth(5).unwrap();
        s.search_with_options(&p,16,None,MicroSearchOptions {proof_search:false,..Default::default()}).unwrap();
        assert!(s.diagnostic_depth_trial_stats().leaves_by_depth.iter().skip(6).sum::<usize>()>0);
        assert_eq!(s.diagnostic_maximum_depth(),96);
    }

    #[test]
    fn campaign_zero_floor_and_terminal_deadline_exceptions_remain_valid() {
        let model=Arc::new(MicroModel::seeded(23).with_spatial_policy());
        let p=Position::from_standard_setup(paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3));
        let mut old=MicroMctsSession::new(model.clone());
        let mut zero=MicroMctsSession::new(model.clone());
        zero.set_minimum_search_depth(0).unwrap();
        assert_eq!(format!("{:?}",old.search_until(&p,64,None).unwrap()),format!("{:?}",zero.search_until(&p,64,None).unwrap()));
        zero.set_minimum_search_depth(5).unwrap();
        assert_eq!(zero.search_until(&p,512,Some(paisho_platform::training_time::now())).unwrap().simulations,0);
        let r:paisho_core::GameRecord=include_str!("../../../tests/fixtures/site_bot_v1_ring_finish.psr").parse().unwrap();
        let (r,_)=r.replay_prefix_with_rules(paisho_core::RuleProfileId::SkudPaiShoGen5V1).unwrap();
        let mut p=r.initial_position();
        for &a in &r.actions()[..r.actions().len()-1] {p.apply(a).unwrap();}
        let mut s=MicroMctsSession::new(model);
        s.set_minimum_search_depth(5).unwrap();
        let report=s.search_with_options(&p,512,None,MicroSearchOptions {proof_search:true,..Default::default()}).unwrap();
        assert_eq!(report.proven_value,Some(1));
        let winner=p.to_move();p.apply(report.actions[report.selected_index]).unwrap();
        assert_eq!(p.outcome(),GameOutcome::Win(winner));
    }

}
