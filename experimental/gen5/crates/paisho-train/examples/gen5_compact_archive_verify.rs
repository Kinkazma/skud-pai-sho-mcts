//! Read-only verification of compact lessons, including targets evicted from FIFO.
use paisho_ai::MicroProofCertificate;
use paisho_core::{legal_actions,GameOutcome,GameRecord};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,io::Read,path::Path};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn main()->Result<()> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=2 {return Err("TRIAL ARCHIVE".into());}
    let mut records=0;let mut lessons=0;let mut certificates=0;let mut improved_targets=0;let mut legacy_teachers_without_mask=0;
    for entry in fs::read_dir(&args[1])? {
        let path=entry?.path();
        let Some(name)=path.file_name().and_then(|s|s.to_str()).and_then(|s|s.strip_suffix(".json.gz")) else {continue;};
        let bytes=fs::read(&path)?;if name!=hash(&bytes) {return Err("bundle hash".into());}
        let mut raw=vec![];flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut raw)?;
        let b:Value=serde_json::from_slice(&raw)?;
        if b["schema"]!="paisho-gen5-durable-lessons-v1" || b["rules"]!="skud-pai-sho-gen5-v1" {return Err("bundle schema/rules".into());}
        let psr=b["psr"].as_str().ok_or("PSR missing")?;
        if b["psr_sha256"]!=hash(psr.as_bytes()) {return Err("bundle PSR hash".into());}
        let id=b["game_id"].as_u64().ok_or("game id")?;
        let base=Path::new(&args[0]).join("games").join(format!("game-{id:07}"));
        if fs::read(base.with_extension("psr"))?!=psr.as_bytes() {return Err("receipt PSR binding".into());}
        let r:Value=serde_json::from_slice(&fs::read(base.with_extension("json"))?)?;
        if r["collector"]!=b["source"] {return Err("collector binding".into());}
        let record:GameRecord=psr.parse()?;let mut position=record.initial_position();let mut states=vec![position.clone()];
        for action in record.actions() {position.apply(*action)?;states.push(position.clone());}
        let saved=b["lessons"].as_array().ok_or("lessons")?;
        if r["eligible_examples"].as_u64()!=Some(saved.len() as u64) {return Err("lesson count".into());}
        for lesson in saved {
            let d=lesson["decision"].as_u64().ok_or("decision")? as usize;
            let p=states.get(d.checked_sub(1).ok_or("decision zero")?).ok_or("decision bounds")?;
            let actions=legal_actions(p).iter().map(ToString::to_string).collect::<Vec<_>>();
            let mut mass=0.0;
            for pair in lesson["policy"].as_array().ok_or("policy")? {
                let a=pair[0].as_str().ok_or("action")?;let q=pair[1].as_f64().ok_or("probability")?;
                if !actions.iter().any(|x|x==a) || !q.is_finite() || q<=0.0 {return Err("illegal policy target".into());}mass+=q;
            }
            if mass!=0.0 && (mass-1.0).abs()>1e-8 {return Err("policy mass".into());}
            let value=lesson["value"].as_f64().ok_or("value")?;
            if !value.is_finite() || value.abs()>1.0 {return Err("invalid value".into());}
            if lesson["reason"]=="rules-terminal-z" {
                let expected=match position.outcome() {GameOutcome::Win(w)=>if w==p.to_move(){1.0}else{-1.0},GameOutcome::Draw=>0.0,GameOutcome::Ongoing=>return Err("invented terminal target".into())};
                if value!=expected {return Err("terminal perspective".into());}
            }
            if lesson["evidence"]["policy_source"]=="full-search-estimate" && lesson["policy_weight"].as_f64().unwrap_or(0.)>0. {
                let e=&lesson["evidence"];
                if let Some(mask)=e["excluded_actions"].as_array() {
                    let q:Vec<f64>=serde_json::from_value(e["completed_action_values"].clone())?;
                    let prior:Vec<f64>=serde_json::from_value(e["target_prior"].clone())?;
                    if q.len()!=actions.len() || prior.len()!=actions.len() || mask.len()!=actions.len(){return Err("teacher legal alignment".into());}
                    let allowed:Vec<_>=(0..actions.len()).filter(|&i|mask[i]==false).collect();
                    let logits:Vec<_>=allowed.iter().map(|&i|prior[i].max(1e-300).ln()+q[i]).collect();
                    let probs=paisho_ai::micro_softmax(&logits)?;let mut expected=vec![0.;actions.len()];for (i,p) in allowed.into_iter().zip(probs){expected[i]=p;}
                    let mut saved=vec![0.;actions.len()];for pair in lesson["policy"].as_array().unwrap(){let i=actions.iter().position(|a|Some(a.as_str())==pair[0].as_str()).ok_or("teacher action")?;saved[i]=pair[1].as_f64().ok_or("teacher probability")?;}
                    if expected.iter().zip(saved).any(|(a,b)|(a-b).abs()>1e-10){return Err("compact improved policy mismatch".into());}improved_targets+=1;
                }else{legacy_teachers_without_mask+=1;}
            }
            if lesson["reason"]=="repetition-training-loss" {return Err("invented cycle defeat".into());}
            lessons+=1;
        }
        for pair in b["proofs"].as_array().ok_or("proofs")? {
            let d=pair[0].as_u64().ok_or("proof decision")? as usize;
            let c:MicroProofCertificate=serde_json::from_value(pair[1].clone())?;
            c.verify(states.get(d.checked_sub(1).ok_or("proof zero")?).ok_or("proof bounds")?)?;certificates+=1;
        }
        records+=1;
    }
    println!("{}",json!({"compact_bundles":records,"lessons":lessons,"certificates":certificates,"legal_policies_and_terminal_perspectives":true,"improved_targets_recomputed":improved_targets,"legacy_teachers_without_mask":legacy_teachers_without_mask}));Ok(())
}
