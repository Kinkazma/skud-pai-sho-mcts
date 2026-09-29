//! Isolated native conversion/gradient audit and resident ABBA kernel timing.
//! Reads immutable model/teacher inputs; never starts games or writes model weights.
use paisho_ai::*;
use paisho_train::micro_learning::{MicroArtifact,SavedMicroExample};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,path::Path,time::Instant};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
fn value_coordinate(i:usize)->bool {(4128..4161).contains(&i)||(15822..93071).contains(&i)}
fn run(m:&MicroModel,rows:&[(MicroExample,MicroExample)],variant:usize)->Result<f64> {
    let start=Instant::now();let mut buffer=vec![];let mut check=0.;
    for (old,new) in rows {
        let (loss,g)=match variant {
            0=>m.loss_gradient_reusing(old,buffer)?,
            1=>m.loss_gradient_detached_value_reusing(old,buffer)?,
            2=>m.loss_gradient_loop_v3_reusing(new,buffer)?,
            3=>m.loss_gradient_all_actions_auxiliary_reusing(new,buffer)?,
            _=>unreachable!(),
        };
        check+=loss.total(old.policy_weight)+g[0];buffer=g;
    }
    std::hint::black_box(check);Ok(start.elapsed().as_secs_f64())
}
fn main()->Result<()> {
    std::env::set_var("VECLIB_MAXIMUM_THREADS","1");
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=4 {return Err("MODEL SAMPLE_JSON OUTPUT_JSON".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let model_bytes=fs::read(&args[1])?;
    let artifact=MicroArtifact::load(Path::new(&args[1]))?;
    let model=artifact.model()?;
    if !model.has_neural_memory() {return Err("probe requires active neural architecture".into());}
    let teachers_bytes=fs::read(&args[2])?;
    let saved:Vec<SavedMicroExample>=serde_json::from_slice(&teachers_bytes)?;
    let loading_complete=Instant::now();
    let mut rows=vec![];let mut old_targets=0;let mut new_targets=0;
    let mut serialized_bytes=0;let mut serialized_provenance_bytes=0;
    for s in &saved {
        let old=s.example_for_rules(paisho_train::micro_learning::gen5::RULES)?;
        let new=s.example_for_rules_with_trusted_q(paisho_train::micro_learning::gen5::RULES,true)?;
        assert_eq!(old.state,new.state);assert_eq!(old.actions,new.actions);assert_eq!(old.policy,new.policy);
        assert_eq!(old.value.to_bits(),new.value.to_bits());assert_eq!(old.value_weight.to_bits(),new.value_weight.to_bits());
        assert_eq!(old.policy_weight.to_bits(),new.policy_weight.to_bits());assert_eq!(old.sequence_source,new.sequence_source);
        old_targets+=old.action_values.iter().flatten().count();new_targets+=new.action_values.iter().flatten().count();
        for (i,q) in new.action_values.iter().enumerate() {
            if let Some(q)=q {assert_eq!(Some(q.to_bits()),old.action_values.get(i).copied().flatten().map(f64::to_bits));}
        }
        let bytes=serde_json::to_vec(s)?;serialized_bytes+=bytes.len();
        // Only a serialization size bound. These are explicitly synthetic
        // all-positive counts, never persisted as actual evidence or learned.
        let mut tagged=s.clone();
        if let Some(e)=&mut tagged.evidence {
            if !e.completed_action_values.is_empty() {e.action_value_visits=vec![512;e.completed_action_values.len()];}
        }
        serialized_provenance_bytes+=serde_json::to_vec(&tagged)?.len();
        if old.policy_weight>0. && !old.actions.is_empty() {rows.push((old,new));}
    }
    let conversion_seconds=loading_complete.elapsed().as_secs_f64();
    // Spread the controlled native checks across the already frozen sample.
    let count=16.min(rows.len());
    let selected=(0..count).map(|i|rows[i*(rows.len()-1)/(count-1).max(1)].clone()).collect::<Vec<_>>();
    let mut leaked_rows=0;let mut validated_coefficients=0;
    for (old,new) in &selected {
        let (joint,gj)=model.loss_gradient(old)?;
        let (detached,gd)=model.loss_gradient_detached_value(old)?;
        assert_eq!(joint.value.to_bits(),detached.value.to_bits());assert_eq!(joint.policy.to_bits(),detached.policy.to_bits());
        let mut direct=old.clone();direct.structured.clear();direct.actions.clear();direct.policy.clear();direct.action_values.clear();direct.policy_weight=0.;
        let (_,gv)=model.loss_gradient(&direct)?;
        for (i,((j,d),v)) in gj.iter().zip(&gd).zip(&gv).enumerate() {
            assert_eq!(d.to_bits(),if value_coordinate(i) {v.to_bits()}else{j.to_bits()});
            validated_coefficients+=1;
        }
        leaked_rows+=usize::from(gj.iter().zip(&gd).enumerate().any(|(i,(a,b))|value_coordinate(i)&&a!=b));
        let v3=model.loss_gradient_loop_v3_reusing(new,vec![f64::NAN;gd.len()+17])?;
        let finite=model.loss_loop_v3(new)?;
        assert_eq!(v3.0.value.to_bits(),finite.value.to_bits());assert_eq!(v3.0.policy.to_bits(),finite.policy.to_bits());
    }
    // All variants warm up the same immutable model and sequence contexts before
    // timing. The 4 legs remain in explicit ABBA order; no games or loading inside.
    for variant in [0,1,2,3] {run(&model,&selected,variant)?;}
    let mut timings=vec![];
    for b in [1,2,3] {for repeat in 0..4 {for variant in [0,b,b,0] {
        timings.push(json!({"alternative":b,"repeat":repeat,"variant":variant,"seconds":run(&model,&selected,variant)?}));
    }}}
    let summary=(1..=3).map(|b|{
        let a=timings.iter().filter(|r|r["alternative"]==b&&r["variant"]==0).map(|r|r["seconds"].as_f64().unwrap()).sum::<f64>()/8.;
        let t=timings.iter().filter(|r|r["alternative"]==b&&r["variant"]==b).map(|r|r["seconds"].as_f64().unwrap()).sum::<f64>()/8.;
        json!({"variant":b,"baseline_mean_seconds":a,"variant_mean_seconds":t,"ratio":t/a,"delta_ms_per_example":(t-a)*1000./selected.len() as f64})
    }).collect::<Vec<Value>>();
    let report=json!({"model":args[1],"model_sha256":hash(&model_bytes),"teachers":args[2],"teachers_sha256":hash(&teachers_bytes),
        "rows":saved.len(),"old_auxiliary_targets":old_targets,"trusted_auxiliary_targets":new_targets,
        "conversion_and_size_audit_seconds":conversion_seconds,"serialized_bytes":serialized_bytes,
        "synthetic_three_digit_visit_serialized_bytes":serialized_provenance_bytes,
        "native_gradient_rows":selected.len(),"gradient_coefficients_verified":validated_coefficients,"rows_with_joint_value_side_gradient":leaked_rows,
        "actions_per_selected_row":selected.iter().map(|r|r.0.actions.len()).collect::<Vec<_>>(),
        "variants":{"0":"legacy targets, joint gradient","1":"same targets, detached V context only","2":"trusted Q mask, detached V context, historical known-Q normalization","3":"experimental trusted Q mask, detached V context, all-action Q normalization"},
        "timings":timings,"summary":summary,"campaign_started":false,"weights_written":false,
        "limitations":"resident single-thread kernel timings only; sample conversion consistency and partial gradient routing, not retained learning or whole campaign strength/throughput"});
    fs::write(&args[3],serde_json::to_vec_pretty(&report)?)?;
    println!("{}",serde_json::to_string_pretty(&report["summary"])?);Ok(())
}
