//! Isolated attempt to retain fresh proved decisions under the ORIGINAL Guard.
//! No publication, feedback admission, change of anchor or production checkpoint.
use super::*;
const MAX_CHECKS:usize=12;
const MAX_STEPS:usize=4;
const EPS:f64=1e-6;
#[derive(Clone,Copy,PartialEq,Eq,Serialize)]
#[serde(rename_all="snake_case")]
enum ProjectionSchedule { Once, RelinearizeAfterAccepted }
impl ProjectionSchedule {
    fn families(self,has_seed:bool)->&'static [(&'static str,i32)] {
        if self==Self::RelinearizeAfterAccepted && has_seed {&[("projected",6)]}
        else {&[("direct",6),("projected",6)]}
    }
    fn can_project(self,budget:&Budget,enabled:bool,blocked:bool)->bool {
        enabled && blocked && budget.checks<MAX_CHECKS
            && (!budget.projection_used || self==Self::RelinearizeAfterAccepted)
    }
}
#[derive(Clone)]
pub(crate) struct GainTarget {
    pub id:usize,
    pub example:Arc<MicroExample>,
    pub coupled_offsets:Vec<f64>,
}
#[derive(Clone,Copy,Serialize,Deserialize)]
pub(super) struct GainGoals { pub raw:bool, pub coupled:bool }
#[derive(Clone,Serialize)]
struct Term { good:usize, bad:Option<usize>, hinge:f64 }
#[derive(Clone,Serialize)]
struct TargetScore {
    id:usize,raw:bool,coupled:bool,mass:f64,hinge:f64,good:usize,bad:Option<usize>,
    terms:Vec<Term>,
}
#[derive(Default,Serialize)]
struct Budget { checks:usize, admitted:usize, projection_used:bool }
impl Budget {
    fn check(&mut self)->bool {if self.checks>=MAX_CHECKS {false} else {self.checks+=1;true}}
    fn admit(&mut self)->bool {if self.admitted>=MAX_STEPS {false} else {self.admitted+=1;true}}
    fn project_after_kl_rejection(&self,enabled:bool,kl_blocked:bool)->bool {
        enabled && !self.projection_used && kl_blocked && self.checks<MAX_CHECKS
    }
}
fn best(p:&[f64],keep:impl Fn(usize)->bool)->Option<usize> {
    (0..p.len()).filter(|&i|keep(i)).max_by(|&a,&b|p[a].total_cmp(&p[b]).then_with(||b.cmp(&a)))
}
fn prior(model:&MicroModel,ex:&MicroExample)->Result<Vec<f64>> {
    let base=micro_softmax(&MicroModel::logits(&model.embed(&ex.state),&ex.actions)).map_err(invalid)?;
    model.memory_priors(&ex.state,&ex.actions,&base,ex.sequence_source).map_err(invalid)
}
fn target_scores(model:&MicroModel,targets:&[GainTarget])->Result<Vec<TargetScore>> {
    target_scores_goals(model,targets,None)
}
fn target_scores_goals(model:&MicroModel,targets:&[GainTarget],goals:Option<&[GainGoals]>)->Result<Vec<TargetScore>> {
    targets.iter().enumerate().map(|(i,t)| {
        let p=prior(model,&t.example)?;let w=|i|t.example.policy[i]>0.;
        if p.len()!=t.coupled_offsets.len() || p.iter().any(|p|!p.is_finite()) {return Err(invalid("invalid fresh target priors/offsets"));}
        let logits:Vec<_>=p.iter().map(|p|p.max(1e-300).ln()).collect();
        let coupled:Vec<_>=logits.iter().zip(&t.coupled_offsets).map(|(p,q)|p+q).collect();
        let good=best(&logits,w).ok_or_else(||invalid("fresh target lacks verified support"))?;
        let bad=best(&logits,|i|!w(i));
        if goals.is_none() && (p[good]<=1e-300 || bad.is_some_and(|i|p[i]<=1e-300)) {return Err(invalid("clamped fresh margin cannot be linearized"));}
        let goal=goals.map_or(GainGoals {raw:true,coupled:false},|g|g[i]);
        let mut terms=vec![];
        for values in [goal.raw.then_some(&logits),goal.coupled.then_some(&coupled)].into_iter().flatten() {
            let good=best(values,w).ok_or_else(||invalid("fresh target lacks verified support"))?;
            let bad=best(values,|i|!w(i));
            if p[good]<=1e-300 || bad.is_some_and(|i|p[i]<=1e-300) {return Err(invalid("clamped fresh margin cannot be linearized"));}
            terms.push(Term {good,bad,hinge:bad.map_or(0.,|b|(values[b]-values[good]+EPS).max(0.))});
        }
        let hinge=terms.iter().map(|t|t.hinge).sum::<f64>()/terms.len().max(1) as f64;
        Ok(TargetScore {id:t.id,raw:w(best(&logits,|_|true).unwrap()),coupled:w(best(&coupled,|_|true).unwrap()),
            mass:p.iter().enumerate().filter(|(i,_)|w(*i)).map(|(_,p)|p).sum(),
            hinge,good,bad,terms})
    }).collect()
}
fn merit(rows:&[TargetScore])->f64 {rows.iter().map(|r|r.hinge).sum::<f64>()/rows.len().max(1) as f64}
fn targets_retained(old:&[TargetScore],new:&[TargetScore])->bool {
    old.len()==new.len() && old.iter().zip(new).all(|(a,b)|a.id==b.id && (!a.raw||b.raw) && (!a.coupled||b.coupled))
}
fn improves(guard_ok:bool,old:&[TargetScore],new:&[TargetScore])->bool {
    guard_ok && targets_retained(old,new) && merit(new).is_finite() && merit(new)<merit(old)
}
pub(super) fn margin_gradient(model:&MicroModel,source:&MicroExample,good:usize,bad:usize)->Result<Vec<f64>> {
    let mut ex=source.clone();ex.structured.clear();ex.policy_support=false;ex.value_weight=0.;ex.policy_weight=1.;ex.action_values.clear();
    ex.policy.fill(0.);ex.policy[good]=1.;
    let mut a=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).map_err(invalid)?.1;
    ex.policy[good]=0.;ex.policy[bad]=1.;
    let b=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).map_err(invalid)?.1;
    for (i,(a,b)) in a.iter_mut().zip(b).enumerate() {*a=if branches::value_parameter(i) {0.} else {*a-b};}
    Ok(a)
}
fn targets_gradient(model:&MicroModel,targets:&[GainTarget],scores:&[TargetScore])->Result<Vec<f64>> {
    let mut sum=vec![0.;model.parameters().len()];
    for (t,s) in targets.iter().zip(scores).filter(|(_,s)|s.hinge>0.) {
        for term in s.terms.iter().filter(|t|t.hinge>0.) {
            if let Some(bad)=term.bad {for (a,g) in sum.iter_mut().zip(margin_gradient(model,&t.example,term.good,bad)?) {*a+=g/s.terms.len().max(1) as f64;}}
        }
    }
    for g in &mut sum {*g/=targets.len() as f64;} Ok(sum)
}
// CE with the OLD prior is the differentiable part of KL(old || current).
// Never use the winning-support objective or auxiliary Q here.
fn kl_gradient_one(model:&MicroModel,source:&MicroExample,old:&[f64],current:&[f64])->std::result::Result<Vec<f64>,String> {
    if old.len()!=source.actions.len() || old.len()!=current.len()
        || old.iter().zip(current).any(|(p,q)|*p>0. && (*p<1e-300 || *q<1e-300)) {
        return Err("cannot linearize clamped or mismatched KL".into());
    }
    let mut ex=source.clone();ex.structured.clear();ex.policy=old.to_vec();ex.policy_support=false;
    ex.value_weight=0.;ex.policy_weight=1.;ex.action_values.clear();
    let mut g=model.loss_gradient_loop_v3_reusing(&ex,Vec::new())?.1;
    for (i,g) in g.iter_mut().enumerate() {if branches::value_parameter(i) {*g=0.;}}
    Ok(g)
}
pub(super) fn kl_gradient(panel:&Guard,model:&MicroModel,current:&Score)->Result<(Vec<f64>,usize)> {
    let indices:Vec<_>=panel.score.priors.iter().enumerate().filter(|(_,p)|!p.is_empty()).map(|(i,_)|i).collect();
    let mut sum=vec![0.;model.parameters().len()];let model=model.clone();
    let old=panel.score.priors.clone();let new=current.priors.clone();
    let rows=panel.rows.clone();let f=move |i:&usize|kl_gradient_one(&model,&rows[*i].example,&old[*i],&new[*i]);
    for chunk in indices.chunks(16) {
        let parts:Vec<_>=match &panel.parallel {Some(p)=>p.map_owned(chunk.to_vec(),|_|1,f.clone()),None=>chunk.iter().map(&f).collect()};
        for part in parts {for (a,g) in sum.iter_mut().zip(part.map_err(invalid)?) {*a+=g;}}
    }
    for g in &mut sum {*g/=indices.len().max(1) as f64;} Ok((sum,indices.len()))
}
fn worst_old_margin(guard:&Guard,model:&MicroModel,current:&[Score],rejected:&[Score])->Result<Option<(Vec<f64>,f64)>> {
    let mut worst:Option<(Arc<MicroExample>,usize,usize,f64,f64)>=None;
    for ((p,now),bad) in std::iter::once(guard).chain(guard.validation.as_deref()).zip(current).zip(rejected) {
        for (row,witness) in p.rows.iter().enumerate() {
            for coupled in [false,true] {
                let (old_right,new_wrong)=if coupled {(p.score.coupled[row],!bad.coupled[row])} else {(p.score.raw[row],!bad.raw[row])};
                if !old_right || !new_wrong {continue;}
                let raw_now;let raw_bad;
                let (a,b): (&Vec<f64>,&Vec<f64>)=if coupled {(&now.coupled_logits[row],&bad.coupled_logits[row])} else {
                    raw_now=now.priors[row].iter().map(|p|p.max(1e-300).ln()).collect::<Vec<_>>();
                    raw_bad=bad.priors[row].iter().map(|p|p.max(1e-300).ln()).collect::<Vec<_>>();(&raw_now,&raw_bad)
                };
                let good=best(a,|i|witness.valid[i]).ok_or_else(||invalid("old proof lacks winning support"))?;
                let competitor=best(b,|i|!witness.valid[i]).ok_or_else(||invalid("old failed choice lacks competitor"))?;
                if now.priors[row][good]<=1e-300 || now.priors[row][competitor]<=1e-300 {return Err(invalid("clamped old margin cannot be linearized"));}
                let bad_good=best(b,|i|witness.valid[i]).unwrap();let violation=b[competitor]-b[bad_good]+EPS;
                if worst.as_ref().map_or(true,|(_,_,_,_,v)|violation>*v) {
                    worst=Some((witness.example.clone(),good,competitor,a[competitor]-a[good]+EPS,violation));
                }
            }
        }
    }
    worst.map(|(ex,good,bad,rhs,_)|Ok((margin_gradient(model,&ex,good,bad)?,rhs))).transpose()
}
pub(super) fn shifted(model:&MicroModel,step:&[f64],scale:f64)->Result<MicroModel> {
    let w=model.parameters().iter().zip(step).enumerate().map(|(i,(w,d))|if branches::value_parameter(i) {*w} else {w-scale*d}).collect();
    let mut next=MicroModel::from_parameters(w).map_err(invalid)?;
    if let Some(bank)=model.sequence_memory() {next=next.with_sequence_memory_owned(bank.clone());}Ok(next)
}
fn fresh_loss(model:&MicroModel,rows:&[Arc<MicroExample>],parallel:Option<&cpu::Ordered>)->Result<f64> {
    let n=rows.len().max(1);let model=model.clone();
    let f=move |e:&Arc<MicroExample>|model.loss_loop_v3(e).map(|l|l.total(e.policy_weight)/n as f64);
    let parts:Vec<_>=match parallel {Some(p)=>p.map_owned(rows.to_vec(),|e|e.actions.len(),f),None=>rows.iter().map(f).collect()};
    let mut sum=0.;for part in parts {sum+=part.map_err(invalid)?;}if !sum.is_finite() {return Err(invalid("non-finite fresh transfer gate"));}Ok(sum)
}
impl Guard {
    pub fn diagnostic_fresh_gain_probe(&self,start:&MicroModel,targets:&[GainTarget],allow_projection:bool)->Result<(Vec<MicroModel>,serde_json::Value)> {
        if !start.has_deep_value() || targets.len()!=2 {return Err(invalid("fresh gain probe requires deep model and exactly two verified targets"));}
        self.gain_probe_inner(start,targets,None,None,allow_projection,ProjectionSchedule::Once)
    }
    pub(super) fn diagnostic_fresh_gain_transfer(&self,start:&MicroModel,targets:&[GainTarget],goals:&[GainGoals],fresh:&[Arc<MicroExample>],allow_projection:bool)->Result<(Vec<MicroModel>,serde_json::Value)> {
        self.gain_transfer_with_schedule(start,targets,goals,fresh,allow_projection,ProjectionSchedule::Once)
    }
    pub(super) fn diagnostic_fresh_gain_transfer_relinearized(&self,start:&MicroModel,targets:&[GainTarget],goals:&[GainGoals],fresh:&[Arc<MicroExample>])->Result<(Vec<MicroModel>,serde_json::Value)> {
        self.gain_transfer_with_schedule(start,targets,goals,fresh,true,ProjectionSchedule::RelinearizeAfterAccepted)
    }
    fn gain_transfer_with_schedule(&self,start:&MicroModel,targets:&[GainTarget],goals:&[GainGoals],fresh:&[Arc<MicroExample>],allow_projection:bool,schedule:ProjectionSchedule)->Result<(Vec<MicroModel>,serde_json::Value)> {
        if !start.has_deep_value() || !(1..=8).contains(&targets.len()) || goals.len()!=targets.len() || fresh.len()!=64
            || targets.iter().map(|t|t.id).collect::<std::collections::BTreeSet<_>>().len()!=targets.len() {
            return Err(invalid("gain transfer requires 1..8 unique certified targets, explicit goals and 64 fixed fresh rows"));
        }
        self.gain_probe_inner(start,targets,Some(goals),Some(fresh),allow_projection,schedule)
    }
    fn gain_probe_inner(&self,start:&MicroModel,targets:&[GainTarget],goals:Option<&[GainGoals]>,fresh:Option<&[Arc<MicroExample>]>,allow_projection:bool,schedule:ProjectionSchedule)->Result<(Vec<MicroModel>,serde_json::Value)> {
        let beginning=Instant::now();let state_before=self.progress();let mut budget=Budget::default();
        assert!(budget.check());let (mut scores,why)=self.checked_v3(start)?;
        if !why.is_empty() {return Err(invalid(format!("fresh gain start is not admissible: {why:?}")));}
        let mut model=start.clone();let mut target_score=target_scores_goals(&model,targets,goals)?;
        let fresh_ceiling=fresh.map(|rows|fresh_loss(start,rows,self.parallel.as_ref())).transpose()?;
        let mut fresh_seconds=0.;
        let mut trajectory=vec![model.clone()];let mut attempts=vec![];
        // The first rejected point only identifies constraint sources. Every
        // derivative and right-hand side below is rebuilt at the current model.
        // It is never an accepted anchor, a cached gradient, or a KL reset.
        let mut projection_seed:Option<Vec<Score>>=None;
        let mut projection_history=vec![];let mut projection_count=0;
        let mut gradient_seconds=0.;let mut kl_gradient_seconds=0.;let mut kl_gradient_rows=0;
        let mut projection_seconds=0.;let mut projection_constraints=0;let mut old_margin_gradient_seconds=0.;let mut projection_detail=serde_json::Value::Null;let mut stopped_reason=None::<String>;
        while budget.admitted<MAX_STEPS && budget.checks<MAX_CHECKS && merit(&target_score)>0. {
            let t=Instant::now();let gradient=match targets_gradient(&model,targets,&target_score) {Ok(g)=>g,Err(e)=>{stopped_reason=Some(e.to_string());break;}};
            gradient_seconds+=t.elapsed().as_secs_f64();
            let Some(direct)=margin_step::diagnostic_displacement(&gradient,merit(&target_score)) else {break;};
            let mut chosen=None;
            let mut blocked=projection_seed.clone();
            let reuse_sources=blocked.is_some();
            'families: for &(kind,halves) in schedule.families(reuse_sources) {
                if chosen.is_some() || budget.checks>=MAX_CHECKS {break;}
                let step=if kind=="direct" {direct.clone()} else {
                    if !schedule.can_project(&budget,allow_projection,blocked.is_some()) {break;}
                    budget.projection_used=true;projection_count+=1;
                    assert!(projection_count<=MAX_STEPS);
                    if schedule==ProjectionSchedule::RelinearizeAfterAccepted && projection_seed.is_none() {
                        projection_seed=blocked.clone();
                    }
                    let rejected=blocked.as_ref().unwrap();
                    let mut refs=vec![];let mut rhs=vec![];let t=Instant::now();
                    for ((p,now),bad) in std::iter::once(self).chain(self.validation.as_deref()).zip(&scores).zip(rejected) {
                        if repair::kl(&p.score,bad)>repair::MAX_KL {
                            let (g,n)=match kl_gradient(p,&model,now) {Ok(v)=>v,Err(e)=>{stopped_reason=Some(e.to_string());break 'families;}};refs.push(g);kl_gradient_rows+=n;
                            rhs.push(repair::kl(&p.score,now)-repair::MAX_KL);
                        }
                    }
                    kl_gradient_seconds+=t.elapsed().as_secs_f64();
                    if refs.is_empty() {break;}
                    let margin_t=Instant::now();
                    match worst_old_margin(self,&model,&scores,rejected) {Ok(Some((g,b)))=>{refs.push(g);rhs.push(b);},Ok(None)=>{},Err(e)=>{stopped_reason=Some(e.to_string());break 'families;}}
                    old_margin_gradient_seconds+=margin_t.elapsed().as_secs_f64();
                    projection_constraints=refs.len();assert!(projection_constraints<=3);
                    projection_detail=serde_json::json!({"constraints":refs.len(),"trigger_check":budget.checks,"rhs":rhs,
                        "linearization_number":projection_count,"admitted_before":budget.admitted,"reuse_constraint_sources":reuse_sources,
                        "current_kl":std::iter::once(self).chain(self.validation.as_deref()).zip(&scores).map(|(p,s)|repair::kl(&p.score,s)).collect::<Vec<_>>(),
                        "normal_norms":refs.iter().map(|g|g.iter().map(|x|x*x).sum::<f64>().sqrt()).collect::<Vec<_>>(),
                        "direct_dot_normals":refs.iter().map(|g|g.iter().zip(&direct).map(|(a,b)|a*b).sum::<f64>()).collect::<Vec<_>>()});
                    let t=Instant::now();let projected=super::super::protection::diagnostic_affine(&direct,&refs,&rhs);
                    projection_seconds+=t.elapsed().as_secs_f64();
                    let Some(step)=projected else {stopped_reason=Some("affine projection unavailable".into());break;};
                    let norm=step.iter().map(|x|x*x).sum::<f64>().sqrt();
                    projection_detail["step_norm"]=serde_json::json!(norm);
                    projection_detail["distance_from_direct"]=serde_json::json!(step.iter().zip(&direct).map(|(a,b)|(a-b).powi(2)).sum::<f64>().sqrt());
                    projection_detail["target_directional_decrease"]=serde_json::json!(gradient.iter().zip(&step).map(|(a,b)|a*b).sum::<f64>());
                    projection_history.push(projection_detail.clone());
                    if !reuse_sources && step.iter().zip(&direct).all(|(a,b)|a==b) {stopped_reason=Some("projection did not change the already-tested direction".into());break;}
                    if !norm.is_finite() || norm>0.02 || norm==0. {stopped_reason=Some("projected displacement outside finite bound".into());break;}step
                };
                for halve in 0..halves {
                    if budget.checks>=MAX_CHECKS {break;}
                    let scale=0.5_f64.powi(halve);
                    let trial=match shifted(&model,&step,scale) {Ok(m)=>m,Err(e)=>{attempts.push(serde_json::json!({"kind":kind,"scale":scale,"error":e.to_string(),"accepted":false,"guard_called":false}));continue;}};
                    assert!(budget.check());
                    let t=Instant::now();let (measured,reasons)=match self.checked_v3(&trial) {Ok(v)=>v,Err(e)=>{attempts.push(serde_json::json!({"kind":kind,"check":budget.checks,"scale":scale,"error":e.to_string(),"accepted":false,"guard_called":true}));continue;}};let guard_seconds=t.elapsed().as_secs_f64();
                    let targets_after=match target_scores_goals(&trial,targets,goals) {Ok(v)=>v,Err(e)=>{attempts.push(serde_json::json!({"kind":kind,"check":budget.checks,"scale":scale,"error":e.to_string(),"accepted":false,"guard_called":true}));continue;}};
                    let fresh_t=Instant::now();
                    let fresh_after=match fresh.map(|rows|fresh_loss(&trial,rows,self.parallel.as_ref())).transpose() {
                        Ok(value)=>value,
                        Err(error)=>{
                            fresh_seconds+=fresh_t.elapsed().as_secs_f64();
                            attempts.push(serde_json::json!({"kind":kind,"check":budget.checks,"scale":scale,
                                "error":error.to_string(),"accepted":false,"guard_called":true,"fresh_gate_failed":true}));
                            continue;
                        }
                    };
                    fresh_seconds+=fresh_t.elapsed().as_secs_f64();
                    let fresh_ok=fresh_after.zip(fresh_ceiling).map_or(true,|(now,ceiling)|now.is_finite() && now<=ceiling+1e-12);
                    let accepted=improves(reasons.is_empty(),&target_score,&targets_after) && fresh_ok;
                    attempts.push(serde_json::json!({"kind":kind,"scale":scale,"check":budget.checks,"guard_seconds":guard_seconds,
                        "accepted":accepted,"fresh_after":fresh_after,"fresh_not_worse_than_start":fresh_ok,"guard_reasons":reasons,"targets":targets_after,"merit":merit(&targets_after),
                        "new_choices_retained":targets_retained(&target_score,&targets_after),"scores":measured,
                        "kl":std::iter::once(self).chain(self.validation.as_deref()).zip(&measured).map(|(p,s)|repair::kl(&p.score,s)).collect::<Vec<_>>() }));
                    if accepted {chosen=Some((trial,measured,targets_after));break;}
                    if kind=="direct" && blocked.is_none() && std::iter::once(self).chain(self.validation.as_deref()).zip(&measured)
                        .any(|(p,s)|repair::kl(&p.score,s)>repair::MAX_KL) {blocked=Some(measured);}
                    // Spend the same finite budget on a different direction as soon
                    // as KL blocks it. Direct-only mode keeps its six scales.
                    if kind=="direct" && budget.project_after_kl_rejection(allow_projection,blocked.is_some()) {break;}
                }
            }
            let Some((next,next_scores,next_targets))=chosen else {break;};
            assert!(budget.admit());model=next;scores=next_scores;target_score=next_targets;trajectory.push(model.clone());
        }
        let mut trajectory_value_bits=vec![];
        for (step,saved) in trajectory.iter().enumerate() {
            if saved.parameters().len()!=start.parameters().len() {return Err(invalid("fresh gain trajectory shape changed"));}
            let mut compared=0;let mut mismatches=0;
            for (i,(old,new)) in start.parameters().iter().zip(saved.parameters()).enumerate() {
                if branches::value_parameter(i) {compared+=1;if old.to_bits()!=new.to_bits() {mismatches+=1;}}
            }
            if mismatches!=0 {return Err(invalid("fresh gain trajectory changed value parameter bits"));}
            trajectory_value_bits.push(serde_json::json!({"step":step,"value_parameters_compared":compared,"value_bit_mismatches":mismatches,"all_value_bits_equal_start":true}));
        }
        if self.progress()!=state_before {return Err(invalid("fresh gain probe changed original Guard state"));}
        Ok((trajectory,serde_json::json!({"allow_projection":allow_projection,"projection_schedule_mode":schedule,
            "projection_schedule":if schedule==ProjectionSchedule::Once {"first KL rejection while projection is unused; direct-only keeps six scales"} else {"after the first admitted projection, relinearize its constraint sources at each accepted point before six bounded projected trials"},"budget":budget,"max_admitted":MAX_STEPS,"max_guard_checks_including_start":MAX_CHECKS,
            "fresh_ceiling_fixed_before_trial":fresh_ceiling,"fresh_gate_rows":fresh.map_or(0,|r|r.len()),"fresh_gate_seconds":fresh_seconds,"goals":goals,
            "initial_anchor_identity":self.accepted.identity,"final_scores":{"primary":&scores[0],"validation":scores.get(1)},
            "final_kl":std::iter::once(self).chain(self.validation.as_deref()).zip(&scores).map(|(p,s)|repair::kl(&p.score,s)).collect::<Vec<_>>(),
            "trajectory_value_bits":trajectory_value_bits,"final_targets":target_score,"final_merit":merit(&target_score),"attempts":attempts,
            "target_gradient_seconds":gradient_seconds,"kl_gradient_seconds":kl_gradient_seconds,"kl_gradient_rows":kl_gradient_rows,
            "projection_seconds":projection_seconds,"projection_constraints":projection_constraints,"projection_linearizations":projection_count,"projection_history":projection_history,
            "projection_detail":projection_detail,"old_margin_gradient_seconds":old_margin_gradient_seconds,"total_seconds":beginning.elapsed().as_secs_f64(),
            "stopped_reason":stopped_reason,"value_frozen":true,"guard_anchor_never_reset":true,"guard_state_unchanged":true,"scope":"diagnostic only; every retained step passes the original Guard, retained new choices and strictly decreases fixed target hinge"})))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn row()->MicroExample {MicroExample { structured: Vec::new(),state:vec![0.1;128],actions:vec![[0.2;32],[-0.1;32],[0.7;32]],
        policy:vec![1./3.;3],policy_support:true,action_values:vec![Some(1.);3],sequence_source:0,value:1.,policy_weight:1.,value_weight:1.}}
    #[test] fn fresh_gain_kl_gradient_is_classical_ce_not_support_or_q() {
        let old=MicroModel::seeded(33);let ex=row();let p=prior(&old,&ex).unwrap();
        let mut w=old.parameters().to_vec();w[5185]+=0.15;let model=MicroModel::from_parameters(w).unwrap();
        let q=prior(&model,&ex).unwrap();let g=kl_gradient_one(&model,&ex,&p,&q).unwrap();
        let loss=|m:&MicroModel| {let q=prior(m,&ex).unwrap();p.iter().zip(q).map(|(p,q)|p*(p.ln()-q.ln())).sum::<f64>()};
        for i in [4161,4200,5185,5200] {
            let eps=1e-5;let mut a=model.parameters().to_vec();let mut b=a.clone();a[i]+=eps;b[i]-=eps;
            let numeric=(loss(&MicroModel::from_parameters(a).unwrap())-loss(&MicroModel::from_parameters(b).unwrap()))/(2.*eps);
            assert!((g[i]-numeric).abs()<2e-8,"KL derivative {i}: {} vs {numeric}",g[i]);
        }
        assert!(g.iter().enumerate().all(|(i,g)|!branches::value_parameter(i)||*g==0.));
        assert!(kl_gradient_one(&model,&ex,&p,&[0.,0.5,0.5]).is_err());
    }
    #[test] fn fresh_gain_margin_gradient_has_the_finite_difference_and_descent_sign() {
        let model=MicroModel::seeded(43);let ex=row();let g=margin_gradient(&model,&ex,0,2).unwrap();
        let margin=|m:&MicroModel| {let p=prior(m,&ex).unwrap();p[2].ln()-p[0].ln()};
        for i in [4161,4200,5185,5200] {
            assert!(!branches::value_parameter(i));
            let eps=1e-5;let mut a=model.parameters().to_vec();let mut b=a.clone();a[i]+=eps;b[i]-=eps;
            let numeric=(margin(&MicroModel::from_parameters(a).unwrap())-margin(&MicroModel::from_parameters(b).unwrap()))/(2.*eps);
            assert!((g[i]-numeric).abs()<2e-8,"margin derivative {i}: {} vs {numeric}",g[i]);
        }
        assert!(g.iter().any(|g|g.abs()>1e-8));
        assert!(margin(&shifted(&model,&g,1e-5).unwrap())<margin(&model));
        assert!(g.iter().enumerate().all(|(i,g)|!branches::value_parameter(i)||*g==0.));
    }
    #[test] fn fresh_gain_shift_copies_every_value_bit_even_for_a_nonzero_value_direction() {
        let model=MicroModel::seeded(71).with_neural_memory(19);
        let next=shifted(&model,&vec![0.001;model.parameters().len()],0.5).unwrap();
        assert!(model.parameters().iter().zip(next.parameters()).enumerate()
            .all(|(i,(a,b))|!branches::value_parameter(i)||a.to_bits()==b.to_bits()));
        assert!(model.parameters().iter().zip(next.parameters()).enumerate()
            .any(|(i,(a,b))|!branches::value_parameter(i)&&a.to_bits()!=b.to_bits()));
    }
    #[test] fn relational_policy_relay_keeps_shared_encoder_and_all_values_exact() {
        let model=MicroModel::seeded(71).with_relational(19);let mut w=model.parameters().to_vec();
        let start=w.len()-33;for (i,v) in w[start..].iter_mut().enumerate(){*v=(i as f64+1.).sin()*0.1;}
        let model=MicroModel::from_parameters(w).unwrap();
        let next=shifted(&model,&vec![0.001;model.parameters().len()],0.5).unwrap();
        let source:GameRecord=include_str!("../../../../examples/gen5_structured_corpus/fixtures/cross_owner_lotus.psr").parse().unwrap();
        let record=source.replay_prefix_with_rules(RULES).unwrap().0;let mut p=record.initial_position();
        for &a in record.actions(){let state=model.state_features(&p);assert_eq!(model.value(&state).to_bits(),next.value(&state).to_bits());p.apply(a).unwrap();}
        assert!(model.parameters().iter().zip(next.parameters()).enumerate().all(|(i,(a,b))|!branches::value_parameter(i)||a.to_bits()==b.to_bits()));
    }
    #[test] fn fresh_gain_projection_has_budget_immediately_after_the_first_kl_rejection() {
        let mut b=Budget::default();assert!(b.check()); // Initial admissibility.
        let mut on_scales=vec![];
        for halve in 0..6 {
            assert!(b.check());on_scales.push(0.5_f64.powi(halve));
            // The recorded frontier first trial rejects on KL. Previously,
            // all six direct scales ran before the projection family.
            if b.project_after_kl_rejection(true,true) {break;}
        }
        assert_eq!(on_scales,vec![1.]);assert_eq!(b.checks,2);
        assert!(b.project_after_kl_rejection(true,true));
        assert!(!b.project_after_kl_rejection(false,true));
        assert!(!b.project_after_kl_rejection(true,false));
        b.projection_used=true;assert!(b.check());assert_eq!(b.checks,3);
        assert!(!b.project_after_kl_rejection(true,true)); // Later direct steps keep all scales.
        while b.check() {}assert_eq!(b.checks,12);
        b.projection_used=false;assert!(!b.project_after_kl_rejection(true,true));
        let mut off=Budget::default();assert!(off.check());
        let mut direct_scales=vec![];
        for halve in 0..6 {
            assert!(off.check());direct_scales.push(0.5_f64.powi(halve));
            if off.project_after_kl_rejection(false,true) {break;}
        }
        assert_eq!(direct_scales,vec![1.,0.5,0.25,0.125,0.0625,0.03125]);assert_eq!(off.checks,7);
    }
    #[test] fn gain_relinearization_avoids_the_second_known_blocked_direct_family() {
        // Controlled scheduling counterexample: the first direct trial violates
        // KL, its half projected step is admitted, and the next direct family
        // would spend six checks outside KL. A freshly linearized direction is
        // admissible immediately. This tests allocation, not model efficacy.
        let run=|schedule:ProjectionSchedule| {
            let mut b=Budget::default();assert!(b.check());
            assert!(b.check());assert!(b.project_after_kl_rejection(true,true));
            b.projection_used=true;assert!(b.check());assert!(b.check());assert!(b.admit());
            let mut directions=vec![];
            for &(kind,halves) in schedule.families(true) {
                if kind=="projected" && !schedule.can_project(&b,true,true) {break;}
                for _ in 0..halves {
                    if !b.check() {break;}directions.push(kind);
                    if kind=="projected" {assert!(b.admit());return (b,directions);}
                }
            }
            (b,directions)
        };
        let (old,directs)=run(ProjectionSchedule::Once);
        assert_eq!(directs,vec!["direct";6]);assert_eq!((old.checks,old.admitted),(10,1));
        let (new,directions)=run(ProjectionSchedule::RelinearizeAfterAccepted);
        assert_eq!(directions,vec!["projected"]);assert_eq!((new.checks,new.admitted),(5,2));
        assert!(old.checks<=MAX_CHECKS && new.checks<=MAX_CHECKS);
    }
    #[test] fn gain_relinearization_preserves_one_shared_budget_and_historical_once() {
        let schedule=ProjectionSchedule::RelinearizeAfterAccepted;
        assert_eq!(schedule.families(false),ProjectionSchedule::Once.families(false));
        let mut b=Budget::default();assert!(b.check());b.projection_used=true;
        assert!(!ProjectionSchedule::Once.can_project(&b,true,true));
        assert!(!schedule.can_project(&b,false,true));assert!(!schedule.can_project(&b,true,false));
        for _ in 0..MAX_STEPS {
            assert!(schedule.can_project(&b,true,true));assert!(b.check());assert!(b.admit());
        }
        assert!(!b.admit());assert_eq!(b.admitted,4);
        while b.check() {}assert_eq!(b.checks,12);assert!(!schedule.can_project(&b,true,true));
    }
    #[test] fn fresh_gain_budget_cannot_exceed_twelve_checks_or_four_admissions() {
        let mut b=Budget::default();for _ in 0..12 {assert!(b.check());}assert!(!b.check());assert_eq!(b.checks,12);
        for _ in 0..4 {assert!(b.admit());}assert!(!b.admit());assert_eq!(b.admitted,4);
    }
    #[test] fn fresh_gain_improvement_never_admits_an_invalid_or_forgotten_choice() {
        let a=TargetScore {id:18,raw:false,coupled:true,mass:0.1,hinge:1.,good:0,bad:Some(1),terms:vec![]};
        let mut b=a.clone();b.hinge=0.5;
        assert!(!improves(false,&[a.clone()],&[b.clone()]));assert!(improves(true,&[a.clone()],&[b.clone()]));
        b.coupled=false;assert!(!improves(true,&[a.clone()],&[b]));assert!(!improves(true,&[a.clone()],&[a]));
    }
    #[test] fn gain_transfer_legacy_raw_objective_keeps_its_exact_margin() {
        let model=MicroModel::seeded(31);let mut ex=row();ex.policy=vec![1.,0.,0.];ex.action_values.clear();
        let t=GainTarget {id:0,example:Arc::new(ex),coupled_offsets:vec![0.;3]};
        let p=prior(&model,&t.example).unwrap();let old_hinge=(p[1].max(p[2]).ln()-p[0].ln()+EPS).max(0.);
        let score=target_scores(&model,&[t.clone()]).unwrap();
        assert_eq!(old_hinge.to_bits(),score[0].hinge.to_bits());
        let explicit=target_scores_goals(&model,&[t],Some(&[GainGoals {raw:true,coupled:false}])).unwrap();
        assert_eq!(score[0].hinge.to_bits(),explicit[0].hinge.to_bits());
    }
    #[test] fn gain_transfer_coupled_only_has_the_correct_partial_gradient() {
        let model=MicroModel::seeded(43);let mut ex=row();ex.policy=vec![1.,0.,0.];ex.action_values.clear();
        let t=GainTarget {id:0,example:Arc::new(ex),coupled_offsets:vec![0.,3.,6.]};
        let goals=[GainGoals {raw:false,coupled:true}];
        let score=target_scores_goals(&model,&[t.clone()],Some(&goals)).unwrap();
        assert!(score[0].hinge>0.);let g=targets_gradient(&model,&[t.clone()],&score).unwrap();
        for i in [4161,4200,5185,5200] {
            let eps=1e-5;let mut a=model.parameters().to_vec();let mut b=a.clone();a[i]+=eps;b[i]-=eps;
            let loss=|w|merit(&target_scores_goals(&MicroModel::from_parameters(w).unwrap(),&[t.clone()],Some(&goals)).unwrap());
            let numeric=(loss(a)-loss(b))/(2.*eps);
            assert!((g[i]-numeric).abs()<2e-8,"coupled partial {i}: {} vs {numeric}",g[i]);
        }
    }
    #[test] fn gain_transfer_ineligible_rows_have_no_objective_but_still_protect_choices() {
        let model=MicroModel::seeded(41);let mut ex=row();ex.policy=vec![1.,0.,0.];ex.action_values.clear();
        let t=GainTarget {id:0,example:Arc::new(ex),coupled_offsets:vec![0.,3.,6.]};
        let score=target_scores_goals(&model,&[t],Some(&[GainGoals {raw:false,coupled:false}])).unwrap();
        assert_eq!(score[0].hinge,0.);assert!(score[0].terms.is_empty());
        let mut before=score[0].clone();before.coupled=true;
        assert!(!targets_retained(&[before],&score));
    }

}
