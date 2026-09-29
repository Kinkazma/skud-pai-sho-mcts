//! Test the existing eight fractions BEFORE selecting a model. No gradients.
use super::*;
const FRACTIONS:[f64;8]=[0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625];

fn compose_weights(old:&[f64],new:&[f64],policy:f64,value:f64)->Vec<f64> {
    old.iter().zip(new).enumerate().map(|(i,(a,b))| {
        let f=if branches::value_parameter(i) {value}else{policy};
        if f==1. {*b}else if f==0. {*a}else{a+f*(b-a)}
    }).collect()
}
fn compose(old:&MicroModel,new:&MicroModel,policy:f64,value:f64)->Result<MicroModel> {
    if old.parameters().len()!=new.parameters().len() {return Err(invalid("fraction model shape mismatch"));}
    let mut m=MicroModel::from_parameters(compose_weights(old.parameters(),new.parameters(),policy,value)).map_err(invalid)?;
    if let Some(bank)=old.sequence_memory() {m=m.with_sequence_memory_owned(bank.clone());}Ok(m)
}
fn selected_fresh(receipts:&[serde_json::Value])->Result<(Vec<Arc<MicroExample>>,Vec<Input>)> {
    let mut rows=vec![];let mut inputs=vec![];
    for r in receipts {
        let input=Input {path:r["targets"].as_str().ok_or_else(||invalid("fresh receipt target missing"))?.into(),
            sha256:r["targets_sha256"].as_str().ok_or_else(||invalid("fresh receipt SHA missing"))?.into()};
        let saved=decode_examples(&input.bytes()?)?;
        let stride=r["stride"].as_u64().ok_or_else(||invalid("fresh stride missing"))? as usize;
        if stride==0||r["all_fresh"].as_u64()!=Some(saved.len() as u64) {return Err(invalid("fresh receipt count/stride differs"));}
        let selected=saved.iter().step_by(stride).map(|s|s.example_for_rules_with_trusted_q(RULES,true).map(Arc::new)).collect::<Result<Vec<_>>>()?;
        if r["selected_fresh"].as_u64()!=Some(selected.len() as u64) {return Err(invalid("fresh selected count differs"));}
        rows.extend(selected);inputs.push(input);
    }
    if rows.len()!=99 {return Err(invalid("dynamic fraction diagnostic requires all99 selected cycle1 examples"));}
    Ok((rows,inputs))
}
fn fresh_loss(guard:&Guard,m:&MicroModel,rows:&[Arc<MicroExample>])->Result<f64> {
    let count=rows.len() as f64;let model=m.clone();
    let f=move |e:&Arc<MicroExample>|model.loss_loop_v3(e).map(|loss|loss.total(e.policy_weight)/count);
    let parts=match &guard.parallel {Some(p)=>p.map_owned(rows.to_vec(),|e|e.actions.len(),f),None=>rows.iter().map(f).collect()};
    let mut total=0.;for p in parts {total+=p.map_err(invalid)?;}if !total.is_finite() {return Err(invalid("nonfinite fresh loss"));}Ok(total)
}
fn losses_against(old:&Score,new:&Score)->bool {
    old.raw.len()!=new.raw.len()||old.coupled.len()!=new.coupled.len()
        ||old.raw.iter().zip(&new.raw).any(|(a,b)|*a&&!*b)||old.coupled.iter().zip(&new.coupled).any(|(a,b)|*a&&!*b)
}
fn gain(initial:f64,candidate:f64,trial:f64)->serde_json::Value {
    let available=initial-candidate;let retained=initial-trial;
    serde_json::json!({"initial":initial,"actor001":candidate,"trial":trial,"gain_vs_initial":retained,"loss_delta_vs_actor001":trial-candidate,
        "fraction_of_actor001_gain":if available>1e-12 {Some(retained/available)}else{None},"actor001_gain_denominator":available})
}
fn bits_changed(old:&MicroModel,new:&MicroModel,value:bool)->usize {
    old.parameters().iter().zip(new.parameters()).enumerate().filter(|(i,(a,b))|
        branches::value_parameter(*i)==value&&a.to_bits()!=b.to_bits()).count()
}
fn guard_choice_changes(old:&[Score],new:&[Score])->serde_json::Value {
    serde_json::json!(old.iter().zip(new).map(|(a,b)| {
        let counts=|old:&[bool],new:&[bool]|serde_json::json!({"before":old.iter().filter(|x|**x).count(),"after":new.iter().filter(|x|**x).count(),
            "gained_rows":old.iter().zip(new).enumerate().filter_map(|(i,(a,b))|(!*a&&*b).then_some(i)).collect::<Vec<_>>(),
            "lost_rows":old.iter().zip(new).enumerate().filter_map(|(i,(a,b))|(*a&&!*b).then_some(i)).collect::<Vec<_>>()});
        serde_json::json!({"raw":counts(&a.raw,&b.raw),"coupled":counts(&a.coupled,&b.coupled)})
    }).collect::<Vec<_>>())
}

pub(super) fn run(guard:&Guard,panel:&ActivePanel,loaded:&[Loaded],all_reads:&read_cache::Reads,
    initial:&MicroModel,candidate:&MicroModel,receipts:&[serde_json::Value],initial_updates:u64,candidate_updates:u64,out:&Path)->Result<serde_json::Value> {
    let state_before=guard.progress();let rows=loaded.iter().map(|l|l.witness.clone()).collect::<Vec<_>>();
    let (fresh,inputs)=selected_fresh(receipts)?;
    let (old_scores,old_reasons)=guard.checked_v3(initial)?;
    let (candidate_scores,candidate_reasons)=guard.checked_v3(candidate)?;
    if !old_reasons.is_empty()||!candidate_reasons.is_empty() {return Err(invalid("fraction endpoints fail original Guard"));}
    let initial_all=all_reads.evaluate(&rows,initial,guard.beta,guard.parallel.as_ref())?;
    let candidate_all=all_reads.evaluate(&rows,candidate,guard.beta,guard.parallel.as_ref())?;
    let initial_fresh=fresh_loss(guard,initial,&fresh)?;let candidate_fresh=fresh_loss(guard,candidate,&fresh)?;
    let mut table=vec![];let mut first_family=[None,None];let mut chosen:Option<(usize,MicroModel)>=None;
    for (family,name) in ["joint","full_policy_reduced_value"].into_iter().enumerate() {
        for fraction in FRACTIONS {
            let policy=if family==0 {fraction}else{1.};let value=fraction;
            let m=compose(initial,candidate,policy,value)?;
            let t=Instant::now();let (scores,reasons)=guard.checked_v3(&m)?;
            let dynamic=panel.measure(&m,guard)?;let dynamic_losses=panel.losses(&dynamic)?;
            let all=all_reads.evaluate(&rows,&m,guard.beta,guard.parallel.as_ref())?;
            let keep_current=!losses_against(&candidate_all,&all);
            let admitted=reasons.is_empty()&&dynamic_losses.is_empty()&&keep_current;
            let check_seconds=t.elapsed().as_secs_f64();let t=Instant::now();let loss=fresh_loss(guard,&m,&fresh)?;
            let loss_seconds=t.elapsed().as_secs_f64();let index=table.len();
            if admitted {
                if first_family[family].is_none() {first_family[family]=Some(index);}
                if chosen.is_none() {chosen=Some((index,m.clone()));}
            }
            table.push(serde_json::json!({"index":index,"family":name,"policy_fraction":policy,"value_fraction":value,"admitted":admitted,
                "old_guard_reasons":reasons,"dynamic_losses":dynamic_losses,"actor001_choices_on_six_retained":keep_current,
                "all_six_vs_actor001":paired_choices(loaded,&candidate_all,&all),"all_six_vs_actor000":paired_choices(loaded,&initial_all,&all),
                "old_panel_choices_vs_actor001":guard_choice_changes(&candidate_scores,&scores),
                "fresh":gain(initial_fresh,candidate_fresh,loss),"value_mse":old_scores.iter().zip(&candidate_scores).zip(&scores)
                    .map(|((a,b),c)|gain(a.value_mse,b.value_mse,c.value_mse)).collect::<Vec<_>>(),
                "parameter_sha256":parameter_sha(&m),"changed_policy_coefficients_vs_initial":bits_changed(initial,&m,false),
                "changed_value_coefficients_vs_initial":bits_changed(initial,&m,true),"finite_check_seconds":check_seconds,"fresh_loss_seconds":loss_seconds}));
        }
    }
    // Family order defines the choice before the results are known. We still
    // measure all16 to expose the tradeoff, never choose its best loss afterward.
    let (selected_index,selected)=chosen.map_or((None,initial.clone()),|(i,m)|(Some(i),m));
    let (final_scores,reasons)=guard.checked_v3(&selected)?;
    let final_dynamic=panel.measure(&selected,guard)?;
    let final_all=all_reads.evaluate(&rows,&selected,guard.beta,guard.parallel.as_ref())?;
    if !reasons.is_empty()||!panel.losses(&final_dynamic)?.is_empty() {return Err(invalid("fraction selection/fallback fails original or dynamic Guard"));}
    if selected_index.is_some()&&losses_against(&candidate_all,&final_all) {return Err(invalid("selected fraction lost an actor001 success"));}
    if selected_index.is_none()&&!same_parameters(&selected,initial) {return Err(invalid("fraction fallback not exact actor000"));}
    let artifact=MicroArtifact::new(&selected,if selected_index.is_some(){candidate_updates}else{initial_updates},serde_json::json!({"diagnostic_only":true,"dynamic_fraction_selection":true,
        "selected_index":selected_index,"order":"joint1/2..1/256 then P1 V1/2..1/256 then exact initial fallback"}));
    let path=out.join("diagnostic-fraction-selected.json");fs::write(&path,serde_json::to_vec_pretty(&artifact)?)?;
    for input in &inputs {input.bytes()?;}
    if guard.progress()!=state_before {return Err(invalid("fraction probe changed initial Guard state"));}
    Ok(serde_json::json!({"schema":"paisho-gen5-dynamic-fraction-selection-v1","selected_fresh_count":fresh.len(),"selected_fresh_inputs":inputs,
        "initial_fresh_loss":initial_fresh,"actor001_fresh_loss":candidate_fresh,"initial_scores":old_scores,"actor001_scores":candidate_scores,
        "rows":table,"first_admitted_joint":first_family[0],"first_admitted_full_policy":first_family[1],"selected_index":selected_index,
        "selected_model":path,"selected_parameter_sha256":parameter_sha(&selected),"fallback_initial_exact":selected_index.is_none(),
        "final_old_panel_choices_vs_actor001":guard_choice_changes(&candidate_scores,&final_scores),
        "final_fresh_loss":selected_index.map_or(initial_fresh,|i|table[i]["fresh"]["trial"].as_f64().unwrap()),
        "final_scores":final_scores,"all_six_final_vs_actor001":paired_choices(loaded,&candidate_all,&final_all),
        "all_six_final_vs_actor000":paired_choices(loaded,&initial_all,&final_all),"all_16_evaluated":true,"optimizer_steps":0,
        "limits":["fixed eight fractions may miss other admissible mixtures","nonzero changed coefficients alone do not prove retained learning",
            "policy coefficients fixed does not fix priors because neural reader consumes value","fresh99 loss is descriptive, not an added acceptance criterion",
            "all sixteen rows measured but selection order fixed before observation","timings descriptive, not an ABBA campaign benchmark"]}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn dynamic_fraction_endpoints_copy_parameter_bits() {
        let a=vec![-0.;292363];let b=vec![0.25;292363];
        assert!(compose_weights(&a,&b,0.,0.).iter().zip(&a).all(|(a,b)|a.to_bits()==b.to_bits()));
        let mixed=compose_weights(&a,&b,1.,0.5);
        for (i,w) in mixed.iter().enumerate() {assert_eq!(w.to_bits(),if branches::value_parameter(i) {0.125_f64.to_bits()}else{0.25_f64.to_bits()});}
    }
    #[test] fn dynamic_fraction_order_is_joint_then_value_only_reduction() {
        let schedule=(0..2).flat_map(|family|FRACTIONS.map(move|f|(if family==0{f}else{1.},f))).collect::<Vec<_>>();
        assert_eq!(schedule.len(),16);assert_eq!(schedule[0],(0.5,0.5));assert_eq!(schedule[7],(1./256.,1./256.));
        assert_eq!(schedule[8],(1.,0.5));assert_eq!(schedule[15],(1.,1./256.));
    }
}
