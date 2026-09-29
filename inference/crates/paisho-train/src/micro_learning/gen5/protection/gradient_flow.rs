//! Read-only diagnostic of one full minibatch. Never called by production SGD.
//! Two additional backwards per row; native train_shared remains untouched.
use super::*;
use paisho_ai::{MICRO_HIDDEN, MICRO_INPUTS, MICRO_VALUE_TRUNK, MICRO_NEURAL_MEMORY_START,
    MICRO_NEURAL_MEMORY_PARAMETERS, MICRO_RESIDUAL_PARAMETERS, MICRO_MEMORY_PARAMETERS};
use std::collections::BTreeMap;

// Exact predicate used by publication/branches.rs, copied into this diagnostic
// because that private implementation is not exported outside publication.
fn value_coordinate(i: usize) -> bool {
    let start = (MICRO_INPUTS + 1) * MICRO_HIDDEN;
    (start..=start + MICRO_HIDDEN).contains(&i)
        || (MICRO_VALUE_TRUNK..MICRO_NEURAL_MEMORY_START).contains(&i)
}
fn route(i: usize) -> &'static str {
    if value_coordinate(i) { "direct_value" }
    else if (MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS).contains(&i) { "sequence_reader" }
    else if i < MICRO_NEURAL_MEMORY_START { "base_policy_side_logit" }
    else {
        let j=i-MICRO_NEURAL_MEMORY_START;
        if j<199030 { "neural_shared" }
        else if (j<199290 && (j-199030)%2==0) || j==199290 { "neural_policy_head" }
        else { "neural_q_head" }
    }
}
fn bits(values: &[f64]) -> String {
    let bytes=values.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>();
    sha256(&bytes)
}
fn metric(model: &MicroModel, e: &MicroExample, loss: MicroLoss) -> [f64;4] {
    let policy=loss.policy*e.policy_weight;
    let delta=model.value(&e.state)-e.value;
    let direct=0.5*delta*delta*e.value_weight;
    [policy,direct,loss.value-direct,loss.total(e.policy_weight)]
}
fn mean_metric(sum: [f64;4], n: usize) -> serde_json::Value {
    if n==0 { return serde_json::Value::Null; }
    serde_json::json!({"rows":n,"weighted_policy":sum[0]/n as f64,
        "direct_value":sum[1]/n as f64,"auxiliary_q_by_loss_subtraction":sum[2]/n as f64,
        "total":sum[3]/n as f64})
}
fn directional(u: &[f64], d: &[f64], factor: f64, fresh: usize) -> serde_json::Value {
    let norm=dot(d,d).sqrt();
    if fresh==0 {return serde_json::json!({"norm":norm,"joint_fresh_dot":null,"first_order_policy_change":null,"cosine":null});}
    let inner=dot(u,d);let unorm=dot(u,u).sqrt();
    serde_json::json!({"norm":norm,"joint_fresh_dot":inner,"first_order_policy_change":-factor*inner,
        "cosine":if norm>0. && unorm>0. {Some(inner/(norm*unorm))}else{None}})
}

pub(crate) struct Pending {
    expected: MicroModel,
    fresh: Vec<Arc<MicroExample>>,
    report: serde_json::Value,
}
impl Pending {
    /// Called after the real SGD timer has stopped. Fail before reading any
    /// post-step loss if the predicted native optimizer step is not bit exact.
    pub(crate) fn finish(mut self, actual: &MicroModel) -> Result<serde_json::Value> {
        if actual.parameters().len()!=self.expected.parameters().len()
            || actual.parameters().iter().zip(self.expected.parameters()).any(|(a,b)|a.to_bits()!=b.to_bits()) {
            return Err(invalid("gradient-flow prediction differs from actual train_shared bits"));
        }
        match (actual.sequence_memory(),self.expected.sequence_memory()) {
            (None,None)=>{},(Some(a),Some(b)) if Arc::ptr_eq(a,b)=>{},
            _=>return Err(invalid("gradient-flow sequence bank changed")),
        }
        let mut sum=[0.;4];
        for e in &self.fresh {
            let loss=actual.loss_loop_v3(e).map_err(invalid)?;
            let values=metric(actual,e,loss);
            if values.iter().any(|x|!x.is_finite()) {return Err(invalid("non-finite post-SGD fresh metric"));}
            for (a,b) in sum.iter_mut().zip(values) {*a+=b;}
        }
        self.report["actual_step_bits_exact"]=true.into();
        self.report["actual_after_bits"]=bits(actual.parameters()).into();
        self.report["fresh_after"]=mean_metric(sum,self.fresh.len());
        Ok(self.report)
    }
}
impl Protection {
    /// Exact optimizer anchor contract; no forward, backward or state mutation.
    pub(crate) fn diagnostic_gradient_anchor_signature(&self) -> serde_json::Value {
        let (refs, gram) = if self.policy_gradient_constraint {(self.gradients.as_slice(), self.gram.as_slice())}
            else {(&self.gradients[1..], self.value_gram.as_slice())};
        serde_json::json!({"anchor_bits":bits(self.anchor.parameters()),"loop_v3":self.loop_v3,
            "policy_gradient_constraint":self.policy_gradient_constraint,
            "reference_gradient_bits":refs.iter().map(|r|bits(r)).collect::<Vec<_>>(),"gram":gram})
    }
    pub(crate) fn diagnostic_gradient_flow(
        &self, model: &MicroModel, batch: &[Arc<MicroExample>], kinds: &[u8], rate: f64, l2: f64,
    ) -> Result<Pending> {
        if !self.loop_v3 || !model.has_neural_memory() || batch.len()!=64 || kinds.len()!=64
            || kinds.iter().any(|k|*k>5) || !rate.is_finite() || rate<=0. || !l2.is_finite() || l2<0. {
            return Err(invalid("gradient-flow requires V3 neural full64 with native provenance/rate"));
        }
        let n=model.parameters().len();
        let fresh=batch.iter().zip(kinds).filter(|(_,k)|**k==0).map(|(e,_)|e.clone()).collect::<Vec<_>>();
        let mut full=vec![0.;n];
        let mut policy_by_kind=vec![vec![0.;n];6];
        let mut joint_fresh=vec![0.;n];
        let mut fresh_before=[0.;4];
        let mut counts=[0usize;6];let mut actions=[0usize;6];let mut known_q=[0usize;6];
        let mut active_q=[0usize;6];let mut q_budget=[0.;6];let mut policy_weight=[0.;6];let mut supports=[0usize;6];
        let inputs=batch.iter().cloned().zip(kinds.iter().copied()).collect::<Vec<_>>();
        let frozen=model.clone();
        let compute=move |(e,k): &(Arc<MicroExample>,u8)| -> std::result::Result<_,String> {
            let (loss,g)=frozen.loss_gradient_loop_v3_reusing(e,Vec::new())?;
            let mut p=e.as_ref().clone();p.value_weight=0.;
            let (_,pg)=frozen.loss_gradient_reusing(&p,Vec::new())?;
            let before=if *k==0 {Some(metric(&frozen,e,loss))}else{None};
            Ok((g,pg,before,*k))
        };
        let mut failure=None;
        let mut reduce=|(),part:std::result::Result<(Vec<f64>,Vec<f64>,Option<[f64;4]>,u8),String>| {
            if failure.is_some(){return;}
            let (g,p,before,kind)=match part {Ok(x)=>x,Err(e)=>{failure=Some(e);return;}};
            // EXACT same full-gradient reduction as native full64 train_shared.
            for (sum,v) in full.iter_mut().zip(&g) {*sum+=v*(1./64.);}
            for (sum,v) in policy_by_kind[kind as usize].iter_mut().zip(&p) {*sum+=v*(1./64.);}
            if let Some(values)=before {
                for (sum,v) in joint_fresh.iter_mut().zip(&p) {*sum+=v/fresh.len() as f64;}
                for (sum,v) in fresh_before.iter_mut().zip(values) {*sum+=v;}
            }
        };
        match &self.parallel {
            Some(pool)=>pool.fold_owned(inputs,|item|item.0.actions.len(),compute,(),|_,part|reduce((),part)),
            None=>for input in &inputs {reduce((),compute(input));},
        }
        if let Some(e)=failure {return Err(invalid(e));}
        for (e,&kind) in batch.iter().zip(kinds) {
            let k=kind as usize;let known=e.action_values.iter().flatten().count();
            counts[k]+=1;actions[k]+=e.actions.len();known_q[k]+=known;
            if e.value_weight>0. && known>0 {active_q[k]+=known;q_budget[k]+=0.25*e.value_weight;}
            policy_weight[k]+=e.policy_weight;supports[k]+=usize::from(e.policy_support);
        }
        let mut pf=vec![0.;n];let mut pr=vec![0.;n];let mut v=vec![0.;n];let mut q=vec![0.;n];
        for i in 0..n {
            if value_coordinate(i) {v[i]=full[i];}
            else {
                pf[i]=policy_by_kind[0][i];
                for group in policy_by_kind.iter().skip(1) {pr[i]+=group[i];}
                q[i]=(full[i]-pf[i])-pr[i];
            }
        }
        let mut regularized=full.clone();
        let l2_direction=model.parameters().iter().map(|w|l2*w).collect::<Vec<_>>();
        for (g,w) in regularized.iter_mut().zip(model.parameters()) {*g+=l2*w;}
        let (refs,gram)=if self.policy_gradient_constraint {(self.gradients.as_slice(),self.gram.as_slice())}
            else {(&self.gradients[1..],self.value_gram.as_slice())};
        let projected=projection::homogeneous(&regularized,refs,gram).ok_or_else(||invalid("gradient-flow native projection failed"))?;
        let norm=dot(&projected,&projected).sqrt();
        if !norm.is_finite(){return Err(invalid("gradient-flow non-finite projected norm"));}
        let clip=if norm>10. {10./norm}else{1.};let factor=rate*clip;
        let expected=weights(model,model.parameters().iter().zip(&projected).map(|(w,g)|w-rate*clip*g).collect())?;
        let correction=projected.iter().zip(&regularized).map(|(a,b)|a-b).collect::<Vec<_>>();
        let residual=(0..n).map(|i|full[i]-(((pf[i]+pr[i])+v[i])+q[i])).collect::<Vec<_>>();
        let fresh_n=fresh.len();
        let mut directions=BTreeMap::new();
        for (name,d) in [("incoming_full",&full),("fresh_policy",&pf),("nonfresh_policy",&pr),
            ("auxiliary_q_residual",&q),("direct_value",&v),("l2",&l2_direction),
            ("after_l2",&regularized),("projection_delta",&correction),("actual_projected",&projected),
            ("decomposition_rounding_residual",&residual)] {
            directions.insert(name,directional(&joint_fresh,d,factor,fresh_n));
        }
        let mut routes=BTreeMap::new();
        for name in ["direct_value","sequence_reader","base_policy_side_logit","neural_shared","neural_policy_head","neural_q_head"] {
            let masked=q.iter().enumerate().map(|(i,v)|if route(i)==name {*v}else{0.}).collect::<Vec<_>>();
            routes.insert(name,serde_json::json!({"coordinates":(0..n).filter(|i|route(*i)==name).count(),
                "q_direction":directional(&joint_fresh,&masked,factor,fresh_n)}));
        }
        let mut groups=vec![];
        for k in 0..6 {
            for (i,x) in policy_by_kind[k].iter_mut().enumerate() {if value_coordinate(i){*x=0.;}}
            groups.push(serde_json::json!({"kind":k,"rows":counts[k],"action_rows":actions[k],"known_q_labels":known_q[k],
                "active_q_labels":active_q[k],"sum_auxiliary_scale_before_known_label_division":q_budget[k],
                "sum_policy_weight":policy_weight[k],"support_examples":supports[k],
                "policy_direction":directional(&joint_fresh,&policy_by_kind[k],factor,fresh_n)}));
        }
        let examples=batch.iter().map(|e|resume_example::ResumeExample::from_with_trusted_q(e,true)).collect::<Vec<_>>();
        let report=serde_json::json!({"schema":"paisho-gen5-first-full64-gradient-flow-v1","rows":64,"fresh_rows":fresh_n,
            "additional_backward_calls":128,"parameters":n,"before_bits":bits(model.parameters()),"predicted_after_bits":bits(expected.parameters()),
            "examples_sha256":sha256(&serde_json::to_vec(&examples)?),"kinds":kinds,"groups":groups,
            "anchor_bits":bits(self.anchor.parameters()),"reference_gradient_bits":refs.iter().map(|r|bits(r)).collect::<Vec<_>>(),
            "gram":gram,"policy_gradient_constraint":self.policy_gradient_constraint,"rate":rate,"l2":l2,
            "projected_norm":norm,"clip_scale":clip,"joint_fresh_policy_gradient_norm":dot(&joint_fresh,&joint_fresh).sqrt(),
            "directions":directions,"q_routes":routes,"fresh_before":mean_metric(fresh_before,fresh_n),
            "max_abs_decomposition_rounding_residual":residual.iter().fold(0_f64,|a,x|a.max(x.abs())),
            "q_subtraction_input_norm_sum":dot(&full,&full).sqrt()+dot(&pf,&pf).sqrt()+dot(&pr,&pr).sqrt(),
            "scope":"read-only first full64; fresh means kind0 in this batch; Q is full-P residual outside V, near-roundoff claims require a Q-only check; no ablation, no extra criterion; own sequence_source; forward policy/Q losses are diagnostic, not MCTS strength"});
        Ok(Pending{expected,fresh,report})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_neural_masks_partition_exact_schema() {
        let mut c=BTreeMap::new();
        for i in 0..MICRO_NEURAL_MEMORY_PARAMETERS {*c.entry(route(i)).or_insert(0usize)+=1;}
        assert_eq!(c["direct_value"],77282);
        assert_eq!(c["sequence_reader"],12);
        assert_eq!(c["neural_shared"],199030);
        assert_eq!(c["neural_policy_head"],131);
        assert_eq!(c["neural_q_head"],131);
        assert_eq!(c.values().sum::<usize>(),MICRO_NEURAL_MEMORY_PARAMETERS);
    }
    #[test]
    fn directional_sign_and_missing_fresh_are_explicit() {
        assert_eq!(directional(&[2.],&[-3.],0.1,1)["first_order_policy_change"],0.6000000000000001);
        assert!(directional(&[0.],&[1.],0.1,0)["joint_fresh_dot"].is_null());
    }
    #[test]
    fn diagnostic_predicts_real_full64_without_changing_protection() {
        let model=MicroModel::seeded(912).with_neural_memory(381);
        let ex=Arc::new(MicroExample {state:vec![0.;417],actions:vec![[0.;32];2],policy:vec![1.,0.],
            action_values:vec![Some(1.),None],policy_support:true,sequence_source:0,
            value:1.,value_weight:1.,policy_weight:1.});
        let mut reference=ex.as_ref().clone();reference.actions.clear();reference.policy.clear();reference.action_values.clear();reference.policy_weight=0.;
        let mut p=Protection::new(&model,vec![Arc::new(reference)]).unwrap();p.enable_loop_v3();
        let batch=vec![ex;64];let kinds=(0..64).map(|i|if i%2==0 {0}else{4}).collect::<Vec<_>>();
        let before=p.progress();let pending=p.diagnostic_gradient_flow(&model,&batch,&kinds,0.01,1e-5).unwrap();
        assert_eq!(p.progress(),before);
        let mut actual=model.clone();p.train_shared(&mut actual,&batch,0.01,1e-5).unwrap();
        let report=pending.finish(&actual).unwrap();
        assert_eq!(report["actual_step_bits_exact"],true);assert_eq!(report["additional_backward_calls"],128);
        assert_eq!(report["fresh_rows"],32);
    }
}
