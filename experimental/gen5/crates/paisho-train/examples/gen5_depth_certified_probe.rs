//! Frozen evaluation on previously certified roots; reference proofs are NEVER
//! installed in the search. All arms are graded against the same verified union.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{gen5,MicroArtifact};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeSet,fs,io::Write,path::Path,sync::Arc,time::Instant};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
fn main()->Result<()> {
    let a:Vec<_>=std::env::args().collect();if a.len()!=5 {return Err("MODEL CONFIG FROZEN_REFERENCE_PLAN NEW_OUTPUT_DIR".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let out=Path::new(&a[4]);fs::create_dir(out)?;let start=Instant::now();
    let bytes=fs::read(&a[1])?;let artifact:MicroArtifact=serde_json::from_slice(&bytes)?;let model=Arc::new(artifact.model()?);
    let loading=start.elapsed().as_secs_f64();let config:gen5::Options=serde_json::from_slice(&fs::read(&a[2])?)?;
    let options=config.search(17,false)?;
    let plan_bytes=fs::read(&a[3])?;let plan:Value=serde_json::from_slice(&plan_bytes)?;
    let mut file=std::io::BufWriter::new(fs::File::create(out.join("searches.jsonl"))?);
    let mut roots_file=std::io::BufWriter::new(fs::File::create(out.join("roots.jsonl"))?);
    let mut searches=0;let mut search_seconds=0.;let mut nonimmediate=0;let mut verified=0;
    for (index,input) in plan["positions"].as_array().ok_or("positions missing")?.iter().enumerate() {
        let bytes=fs::read(input["path"].as_str().ok_or("path missing")?)?;
        if hash(&bytes)!=input["sha256"] {return Err("reference hash mismatch".into());}
        let source:Value=serde_json::from_slice(&bytes)?;let record:GameRecord=source["prefix"].as_str().unwrap().parse()?;
        let p=record.replay()?;let certificate:MicroProofCertificate=serde_json::from_value(source["certificate"].clone())?;
        if certificate.verify(&p)?!=GameOutcome::Win(p.to_move()) {return Err("reference not a proved win".into());}
        let key=hash(record.to_string().as_bytes());let mut winning=BTreeSet::new();let mut immediate=vec![];
        for &action in &legal_actions(&p) {let mut next=p.clone();next.apply(action)?;
            if next.outcome()==GameOutcome::Win(p.to_move()) {immediate.push(action.to_string());winning.insert(action.to_string());}}
        for (action,child) in &certificate.children {let mut next=p.clone();next.apply(action.parse()?)?;
            if child.verify(&next)?==GameOutcome::Win(p.to_move()) {winning.insert(action.clone());}}
        nonimmediate+=usize::from(immediate.is_empty());let mut emitted=vec![];let mut order=vec![];
        for b in [32,64,128,256,512] {for f in [0,6,8,16,32] {order.push((b,f));}}
        if index%2==1 {order.reverse();}let len=order.len();order.rotate_left(index%len);
        for (budget,floor) in order {
            let mut s=MicroMctsSession::new(model.clone());s.set_root_value_strength(config.value_policy_strength)?;
            s.diagnostic_set_depth_floor(Some(floor))?;
            let t=Instant::now();let r=s.search_with_options(&p,budget,None,options)?;let seconds=t.elapsed().as_secs_f64();search_seconds+=seconds;
            if s.diagnostic_depth_trial_stats().expansions>budget {return Err("work cap violated".into());}
            let mut proofs=vec![];
            for (i,&value) in r.proven_action_values.iter().enumerate() {if value==Some(1) {
                let proof=s.diagnostic_action_certificate(i).ok_or("child proof missing")?;
                let mut next=p.clone();next.apply(r.actions[i])?;
                if proof.verify(&next)?!=GameOutcome::Win(p.to_move()) {return Err("search proof mismatch".into());}
                winning.insert(r.actions[i].to_string());verified+=1;
                proofs.push(json!({"action":r.actions[i].to_string(),"certificate":proof}));
            }}
            emitted.push(json!({"key":key,"index":index,"budget":budget,"floor":floor,"immediate_control":!immediate.is_empty(),
                "root_rollouts":r.simulations,"stats":s.diagnostic_depth_trial_stats(),"seconds":seconds,
                "inference_evaluations":r.inference_evaluations,"tactical_evaluations":r.tactical_evaluations,
                "selected":r.actions[r.selected_index].to_string(),"proven_value":r.proven_value,"proofs":proofs,
                "legal":r.actions.len(),"visited_root_actions":r.visits.iter().filter(|&&n|n>0).count()}));searches+=1;
        }
        for mut row in emitted {row["selected_in_verified_support"]=json!(winning.contains(row["selected"].as_str().unwrap()));
            serde_json::to_writer(&mut file,&row)?;writeln!(file)?;}
        serde_json::to_writer(&mut roots_file,&json!({"key":key,"index":index,"reference":input,"immediate_wins":immediate,"verified_wins_union":winning}))?;writeln!(roots_file)?;
        file.flush()?;roots_file.flush()?;
        if index%25==24 {eprintln!("{} roots, {searches} searches",index+1);}
    }
    fs::write(out.join("summary.json"),serde_json::to_vec_pretty(&json!({"model_sha256":hash(&bytes),"model_identity":artifact.identity(),
        "reference_plan_sha256":hash(&plan_bytes),"loading_seconds":loading,"native_search_seconds":search_seconds,"wall_seconds":start.elapsed().as_secs_f64(),
        "searches":searches,"roots":searches/25,"non_immediate_roots":nonimmediate,"verified_new_certificates":verified,
        "reference_certificates_installed":0,"learning_updates":0,"exploration":false,"cold_search":true}))?)?;
    Ok(())
}
