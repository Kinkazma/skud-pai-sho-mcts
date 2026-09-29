//! Read-only KL chain decomposition; never changes a Guard criterion.
use super::*;
#[derive(Clone, Serialize)]
struct Components {
    direct: f64,
    between_support_and_rest: f64,
    within_support_weighted: f64,
    within_rest_weighted: f64,
    identity_residual: f64,
    old_mass: [f64;2],
    new_mass: [f64;2],
    clamped_log_entries: usize,
    probability_chain_rule_valid: bool,
}
fn decompose(a:&[f64],b:&[f64],valid:&[bool])->Result<Components> {
    if a.len()!=b.len() || a.len()!=valid.len() || a.is_empty() || !valid.iter().any(|v|*v)
        || a.iter().chain(b).any(|p|!p.is_finite() || *p<0.) {
        return Err(invalid("invalid diagnostic KL distributions/support"));
    }
    let mut old=[0.;2];let mut new=[0.;2];let mut clamped=0;
    for ((p,q),v) in a.iter().zip(b).zip(valid) {
        let group=usize::from(!*v);old[group]+=p;new[group]+=q;
        if *p>0. && (*p<1e-300 || *q<1e-300) {clamped+=1;}
    }
    let direct=a.iter().zip(b).filter(|(p,_)|**p>0.)
        .map(|(p,q)|p*(p.max(1e-300).ln()-q.max(1e-300).ln())).sum::<f64>();
    let mut between=0.;let mut within=[0.;2];
    for group in 0..2 {
        if old[group]>0. {between+=old[group]*(old[group].max(1e-300).ln()-new[group].max(1e-300).ln());}
    }
    for ((p,q),v) in a.iter().zip(b).zip(valid).filter(|((p,_),_)|**p>0.) {
        let group=usize::from(!*v);
        within[group]+=p*((p.max(1e-300).ln()-old[group].max(1e-300).ln())
            -(q.max(1e-300).ln()-new[group].max(1e-300).ln()));
    }
    let reconstructed=between+within[0]+within[1];
    let valid_probability=clamped==0 && (old.iter().sum::<f64>()-1.).abs()<1e-12
        && (new.iter().sum::<f64>()-1.).abs()<1e-12
        && (0..2).all(|i|old[i]==0. || new[i]>0.);
    Ok(Components {direct,between_support_and_rest:between,within_support_weighted:within[0],
        within_rest_weighted:within[1],identity_residual:direct-reconstructed,
        old_mass:old,new_mass:new,clamped_log_entries:clamped,probability_chain_rule_valid:valid_probability})
}
fn aggregate(rows:&[(usize,bool,Components)], select:impl Fn(bool)->bool, denominator:usize)->serde_json::Value {
    let included:Vec<_>=rows.iter().filter(|(_,raw,_)|select(*raw)).collect();
    let d=denominator.max(1) as f64;
    serde_json::json!({"rows":included.len(),"denominator_all_winning_roots":denominator,
        "direct_contribution":included.iter().map(|(_,_,c)|c.direct).sum::<f64>()/d,
        "between_contribution":included.iter().map(|(_,_,c)|c.between_support_and_rest).sum::<f64>()/d,
        "within_support_contribution":included.iter().map(|(_,_,c)|c.within_support_weighted).sum::<f64>()/d,
        "within_rest_contribution":included.iter().map(|(_,_,c)|c.within_rest_weighted).sum::<f64>()/d})
}
impl Guard {
    /// The same policy-axis arithmetic as finite interpolation; no repair/write.
    pub fn diagnostic_policy_axis_model(&self,candidate:&MicroModel,fraction:f64)->Result<MicroModel> {
        if ![1.,0.5,0.25,0.125].contains(&fraction)
            || candidate.parameters().len()!=self.accepted.model.parameters().len() {
            return Err(invalid("unsupported diagnostic policy axis"));
        }
        if fraction==1. {return Ok(candidate.clone());}
        let weights=self.accepted.model.parameters().iter().zip(candidate.parameters()).enumerate()
            .map(|(i,(old,new))|if branches::value_parameter(i) {*new} else {old+fraction*(new-old)}).collect();
        let mut model=MicroModel::from_parameters(weights).map_err(invalid)?;
        if let Some(bank)=candidate.sequence_memory() {model=model.with_sequence_memory_owned(bank.clone());}
        if model.parameters().iter().zip(candidate.parameters()).enumerate()
            .any(|(i,(a,b))|branches::value_parameter(i) && a.to_bits()!=b.to_bits()) {
            return Err(invalid("diagnostic changed fixed value parameters"));
        }
        Ok(model)
    }
    pub fn diagnostic_kl_decomposition(&self,model:&MicroModel)->Result<serde_json::Value> {
        let started=Instant::now();let mut panels=vec![];let mut arithmetic_seconds=0.;
        for (panel,p) in std::iter::once(self).chain(self.validation.as_deref()).enumerate() {
            let score=p.evaluate(model)?;
            let t=Instant::now();let mut rows=vec![];let mut details=vec![];
            for (index,witness) in p.rows.iter().enumerate() {
                let old=&p.score.priors[index];let new=&score.priors[index];
                if old.is_empty() {continue;}
                if witness.example.value!=1. {return Err(invalid("KL probe expected winning roots only"));}
                let c=decompose(old,new,&witness.valid)?;
                if c.identity_residual.abs()>2e-11*(1.+c.direct.abs()) {
                    return Err(invalid("diagnostic KL chain identity failed"));
                }
                details.push(serde_json::json!({"row":index,"old_raw":p.score.raw[index],"new_raw":score.raw[index],
                    "old_coupled":p.score.coupled[index],"new_coupled":score.coupled[index],
                    "proved_actions":witness.valid.iter().filter(|v|**v).count(),"actions":old.len(),"components":&c}));
                rows.push((index,p.score.raw[index],c));
            }
            let sum=rows.iter().fold(0.,|sum,(_,_,c)|sum+c.direct);
            let reproduced=sum.max(0.)/rows.len().max(1) as f64;
            let native=repair::kl(&p.score,&score);
            if reproduced.to_bits()!=native.to_bits() {return Err(invalid("decomposition direct KL differs from native Guard"));}
            let all=aggregate(&rows,|_|true,rows.len());
            let old_raw_right=aggregate(&rows,|raw|raw,rows.len());
            let old_raw_wrong=aggregate(&rows,|raw|!raw,rows.len());
            let lost=|old:&[bool],new:&[bool]|old.iter().zip(new).enumerate().filter(|(_, (a,b))|**a && !**b).map(|(i,_)|i).collect::<Vec<_>>();
            arithmetic_seconds+=t.elapsed().as_secs_f64();
            panels.push(serde_json::json!({"panel":panel,"native_kl":native,"native_kl_bit_exact":true,
                "passes_current_kl":native.is_finite() && native<=0.001,
                "all":all,"old_raw_right_contribution":old_raw_right,"old_raw_wrong_contribution":old_raw_wrong,
                "all_probability_chain_rules_valid":rows.iter().all(|(_,_,c)|c.probability_chain_rule_valid),
                "max_identity_residual":rows.iter().map(|(_,_,c)|c.identity_residual.abs()).fold(0.,f64::max),
                "raw_lost":lost(&p.score.raw,&score.raw),"coupled_lost":lost(&p.score.coupled,&score.coupled),
                "value_mse_before":p.score.value_mse,"value_mse_after":score.value_mse,
                "passes_current_value":score.value_mse.is_finite() && score.value_mse<=p.score.value_mse+1e-9,
                "rows":details}));
        }
        Ok(serde_json::json!({"panels":panels,"measure_and_decompose_seconds":started.elapsed().as_secs_f64(),
            "decomposition_arithmetic_seconds":arithmetic_seconds,"kl_threshold_unchanged":0.001,
            "scope":"raw-prior KL on all known winning roots, including previously wrong raw choices; no repair or admission"}))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn kl_decomposition_only_support_swap_is_safe_but_not_free_under_current_guard() {
        let d=decompose(&[0.45,0.45,0.1],&[0.81,0.09,0.1],&[true,true,false]).unwrap();
        assert!(d.direct>0.4);assert!(d.between_support_and_rest.abs()<1e-14);
        assert!(d.within_rest_weighted.abs()<1e-14);assert!(d.identity_residual.abs()<1e-14);
        assert!(d.probability_chain_rule_valid);
    }
    #[test] fn kl_decomposition_separates_support_mass_from_unknown_action_redistribution() {
        let mass=decompose(&[0.45,0.45,0.1],&[0.2,0.2,0.6],&[true,true,false]).unwrap();
        assert!(mass.direct>0.5);assert!(mass.within_support_weighted.abs()<1e-14);
        assert!(mass.within_rest_weighted.abs()<1e-14);
        let unknown=decompose(&[0.6,0.2,0.2],&[0.6,0.35,0.05],&[true,false,false]).unwrap();
        assert!(unknown.within_rest_weighted>0.);assert!(unknown.between_support_and_rest.abs()<1e-14);
    }
    #[test] fn kl_decomposition_internal_redistribution_still_needs_the_choice_guard() {
        let a=[0.6,0.01,0.01,0.01,0.37];let b=[0.1575,0.1575,0.1575,0.1575,0.37];
        let valid=[true,true,true,true,false];let d=decompose(&a,&b,&valid).unwrap();
        let best=|p:&[f64]|(0..p.len()).max_by(|&i,&j|p[i].total_cmp(&p[j])).unwrap();
        assert!(valid[best(&a)]);assert!(!valid[best(&b)]);
        assert!(d.between_support_and_rest.abs()<1e-14);assert!(d.within_rest_weighted.abs()<1e-14);
        assert!(d.within_support_weighted>0.);
    }
    #[test] fn kl_decomposition_marks_clamps_and_handles_all_proved_support() {
        let clamp=decompose(&[0.5,0.5],&[0.,1.],&[true,false]).unwrap();
        assert!(!clamp.probability_chain_rule_valid);assert_eq!(clamp.clamped_log_entries,1);
        assert!(clamp.identity_residual.abs()<1e-10);
        let all=decompose(&[0.5,0.5],&[0.75,0.25],&[true,true]).unwrap();
        assert!(all.between_support_and_rest.abs()<1e-14);assert_eq!(all.within_rest_weighted,0.);
        assert!(all.identity_residual.abs()<1e-14);
    }
}
