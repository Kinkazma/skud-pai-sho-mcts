//! Diagnostic continuous choice merit, reusing the priors already read by losses.
//! Final finite choices, references, gradient halfspace and resource bounds stay fixed.
use super::*;

const INTERIOR: f64 = 1e-6; // Same interior target as lost_choice_constraint.
#[derive(Clone, Serialize)]
struct Gap {
    good: usize,
    bad: Option<usize>,
    raw_gap: Option<f64>,
    selected_supported: bool,
}
#[derive(Clone)]
pub(super) struct Margins { gaps: Vec<Gap> }
fn penalty(gap: Option<f64>) -> f64 {
    gap.map_or(0.,|g|(g + INTERIOR).max(0.))
}
impl Margins {

    /// Two largest gaps among the OLD mask; no outcome-conditioned row IDs or
    /// new cutoff. A satisfied second constraint retains its actual slack.
    fn critical(&self,old:&[bool]) -> Result<Vec<usize>> {
        if self.gaps.len()!=old.len() {return Err(invalid("joint choice mask mismatch"));}
        let mut indices=self.gaps.iter().zip(old).enumerate()
            .filter(|(_, (gap,protected))|**protected && gap.raw_gap.is_some()).map(|(i,_)|i).collect::<Vec<_>>();
        indices.sort_by(|&a,&b|self.gaps[b].raw_gap.unwrap().total_cmp(&self.gaps[a].raw_gap.unwrap()).then_with(||a.cmp(&b)));
        indices.truncate(2);Ok(indices)
    }
    pub fn constraints(&self,model:&MicroModel,rows:&[Arc<MicroExample>],old:&[bool]) -> Result<Vec<(Vec<f64>,f64,serde_json::Value)>> {
        let references=rows.iter().filter(|r|r.value==1. && !r.policy.is_empty()).collect::<Vec<_>>();
        if references.len()!=self.gaps.len() {return Err(invalid("joint choice references changed"));}
        let mut out=Vec::new();
        for row in self.critical(old)? {
            let selected=&self.gaps[row];let good=selected.good;let bad=selected.bad.unwrap();
            // Exact same gradient difference and target as lost_choice_constraint.
            let mut example=references[row].as_ref().clone();
            example.value_weight=0.;example.policy_weight=1.;example.policy.fill(0.);example.policy[good]=1.;
            let (_,mut gradient)=model.loss_gradient(&example).map_err(invalid)?;
            example.policy[good]=0.;example.policy[bad]=1.;
            let (_,bad_gradient)=model.loss_gradient(&example).map_err(invalid)?;
            for (a,b) in gradient.iter_mut().zip(bad_gradient) {*a-=b;}
            let gap=selected.raw_gap.unwrap()+INTERIOR;
            out.push((gradient,gap,serde_json::json!({"winning_row":row,"good":good,"bad":bad,
                "raw_gap":selected.raw_gap,"rhs":gap,"was_still_correct":selected.selected_supported})));
        }
        Ok(out)
    }

    pub fn merit(&self, old: &[bool]) -> Result<f64> {
        if self.gaps.len()!=old.len() {return Err(invalid("continuous choice mask mismatch"));}
        Ok(self.gaps.iter().zip(old).filter(|(_,p)|**p).map(|(g,_)|penalty(g.raw_gap)).sum())
    }
    pub fn trace(&self, old: &[bool]) -> serde_json::Value {
        serde_json::json!({"merit":self.merit(old).expect("validated old choice mask"),
            "protected":self.gaps.iter().zip(old).enumerate().filter(|(_,(_,p))|**p)
                .map(|(row,(g,_))|serde_json::json!({"winning_row":row,"good":g.good,"bad":g.bad,
                    "raw_gap":g.raw_gap,"penalty":penalty(g.raw_gap),"selected_supported":g.selected_supported}))
                .collect::<Vec<_>>()})
    }
}
fn gap(p: &[f64], policy: &[f64]) -> Result<Gap> {
    if p.len()!=policy.len() || p.is_empty() || p.iter().any(|v|!v.is_finite()||*v<0.) {
        return Err(invalid("invalid continuous choice probabilities"));
    }
    let good=(0..p.len()).filter(|&i|policy[i]>0.)
        .max_by(|&a,&b|p[a].total_cmp(&p[b]).then_with(||b.cmp(&a)))
        .ok_or_else(||invalid("protected policy has no supported action"))?;
    let bad=(0..p.len()).filter(|&i|policy[i]<=0.)
        .max_by(|&a,&b|p[a].total_cmp(&p[b]).then_with(||b.cmp(&a)));
    Ok(Gap {good,bad,raw_gap:bad.map(|i|p[i].max(1e-300).ln()-p[good].max(1e-300).ln()),
        selected_supported:policy[best(p)]>0.})
}
/// Off delegates literally to the old reader. On adds only scalar gap accounting
/// to the SAME value/prior reads, ordered fold and loss formulas; no extra forward.
pub(super) fn read(
    model:&MicroModel, rows:&[Arc<MicroExample>], parallel:Option<&cpu::Ordered>, enabled:bool,
) -> Result<(([f64;4],Vec<bool>),Option<Margins>)> {
    if !enabled {return Ok((losses(model,rows,parallel)?,None));}
    let model=model.clone();
    let score=move |e:&Arc<MicroExample>| -> std::result::Result<_,String> {
        let k=(e.value as i8+1) as usize+1;
        let error=0.5*(model.value(&e.state)-e.value).powi(2);
        let policy=if e.value==1. && !e.policy.is_empty() {
            let p=prior(&model,e).map_err(|e|e.to_string())?;
            let mass:f64=p.iter().zip(&e.policy).filter(|(_,t)|**t>0.).map(|(p,_)|p).sum();
            Some((mass.max(1e-300).ln(),e.policy[best(&p)]>0.,gap(&p,&e.policy).map_err(|e|e.to_string())?))
        } else {None};
        Ok((k,error,policy))
    };
    let values:Vec<_>=match parallel {Some(p)=>p.map_owned(rows.to_vec(),|e|e.actions.len(),score),None=>rows.iter().map(score).collect()};
    let mut loss=[0.;4];let mut ns=[0usize;4];let mut choices=vec![];let mut gaps=vec![];
    for value in values {
        let (k,error,policy)=value.map_err(invalid)?;ns[k]+=1;loss[k]+=error;
        if let Some((ln,good,g))=policy {loss[0]-=ln;ns[0]+=1;choices.push(good);gaps.push(g);}
    }
    for k in 0..4 {loss[k]/=ns[k].max(1) as f64;}
    Ok(((loss,choices),Some(Margins {gaps})))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuous_merit_sees_progress_before_a_lost_choice_crosses_its_boundary() {
        let old=[true];let a=Margins {gaps:vec![gap(&[0.4,0.6],&[1.,0.]).unwrap()]};
        let b=Margins {gaps:vec![gap(&[0.45,0.55],&[1.,0.]).unwrap()]};
        assert!(!a.gaps[0].selected_supported&&!b.gaps[0].selected_supported);
        let discrete=violation_merit(&[0.;4],&[0.;4],&[false],&old,true);
        assert_eq!(discrete,1.);assert!(b.merit(&old).unwrap()<a.merit(&old).unwrap());
        assert_eq!(a.merit(&[false]).unwrap(),0.);
    }
    #[test]
    fn continuous_merit_preserves_tie_order_and_only_the_old_protected_mask() {
        let tie=gap(&[0.5,0.5],&[0.,1.]).unwrap();
        assert!(!tie.selected_supported);assert_eq!(penalty(tie.raw_gap),INTERIOR);
        let fixed=gap(&[0.499,0.501],&[0.,1.]).unwrap();
        assert!(fixed.selected_supported);assert_eq!(penalty(fixed.raw_gap),0.);
        assert_eq!(penalty(gap(&[0.5,0.5],&[0.5,0.5]).unwrap().raw_gap),0.);
        for g in [-1e-10,0.,1e-10] {assert!((penalty(Some(g))-(INTERIOR+g)).abs()<1e-20);}
    }
    #[test]
    fn diagnostic_gap_reader_keeps_native_losses_and_choices_bit_exact() {
        let model=MicroModel::seeded(37);
        let mut state=vec![0.;128];state[0]=0.5;
        let mut action=[0.;32];action[1]=0.75;
        let rows=vec![Arc::new(MicroExample {state,actions:vec![[0.;32],action],policy:vec![0.5,0.5],
            action_values:vec![],policy_support:true,policy_weight:1.,value_weight:1.,value:1.,sequence_source:0}),
            Arc::new(MicroExample {state:vec![0.;128],actions:vec![],policy:vec![],action_values:vec![],
                policy_support:false,policy_weight:0.,value_weight:1.,value:-1.,sequence_source:0})];
        let original=losses(&model,&rows,None).unwrap();let (on,trace)=read(&model,&rows,None,true).unwrap();
        let (off,empty)=read(&model,&rows,None,false).unwrap();
        for (a,b) in original.0.iter().zip(on.0.iter()) {assert_eq!(a.to_bits(),b.to_bits());}
        for (a,b) in original.0.iter().zip(off.0.iter()) {assert_eq!(a.to_bits(),b.to_bits());}
        assert_eq!(original.1,on.1);assert_eq!(original.1,off.1);assert!(empty.is_none());
        assert_eq!(trace.unwrap().gaps.len(),1);
    }

    #[test]
    fn joint_choices_keep_a_near_boundary_correct_row_in_the_two_halfspaces() {
        let make=|value,correct| Gap {good:1,bad:Some(0),raw_gap:Some(value),selected_supported:correct};
        let margins=Margins {gaps:vec![make(-0.1,true),make(-1.1107174299196387e-6,true),
            make(0.0007234330802641242,false),make(1.,false)]};
        assert_eq!(margins.critical(&[true,true,true,false]).unwrap(),vec![2,1]);
        assert!(margins.gaps[1].raw_gap.unwrap()+INTERIOR<0.);
        let ties=Margins {gaps:vec![make(-0.2,true),make(-0.2,true),make(-0.2,true)]};
        assert_eq!(ties.critical(&[true,true,true]).unwrap(),vec![0,1]);
        assert_eq!(ties.critical(&[false,true,false]).unwrap(),vec![1]);
        assert!(ties.critical(&[false,false,false]).unwrap().is_empty());
    }
    #[test]
    fn joint_choice_gradient_matches_the_existing_worst_choice_halfspace() {
        let model=MicroModel::seeded(57);let mut a=[0.;32];a[1]=0.75;
        let mut example=MicroExample {state:vec![0.;128],actions:vec![[0.;32],a],policy:vec![1.,0.],
            action_values:vec![],policy_support:true,policy_weight:1.,value_weight:1.,value:1.,sequence_source:0};
        let p=prior(&model,&example).unwrap();let selected=best(&p);example.policy.fill(0.);example.policy[1-selected]=1.;
        let rows=vec![Arc::new(example)];let old=[true];
        let (expected,rhs)=lost_choice_constraint(&model,&rows,&old).unwrap().unwrap();
        let (_,read)=read(&model,&rows,None,true).unwrap();let result=read.unwrap().constraints(&model,&rows,&old).unwrap();
        assert_eq!(result.len(),1);assert_eq!(result[0].1.to_bits(),rhs.to_bits());
        assert_eq!(result[0].0.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),expected.iter().map(|x|x.to_bits()).collect::<Vec<_>>());
    }
}
