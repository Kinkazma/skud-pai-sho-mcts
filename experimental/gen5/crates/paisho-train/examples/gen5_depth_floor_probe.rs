//! Bounded-depth diagnostic: same saved actor, human roots and campaign PUCT.
//! Budgets count NEW nodes; floor variants can perform fewer root simulations.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{gen5,MicroArtifact};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::Write,path::Path,sync::Arc,time::Instant};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
fn main()->Result<()> {
    let a:Vec<_>=std::env::args().collect();if a.len()!=7 {return Err("MODEL CONFIG DATASET ORIGINAL_ROWS NEW_OUTPUT_DIR SEED_OFFSET".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let out=Path::new(&a[5]);fs::create_dir(out)?;let seed_offset:u64=a[6].parse()?;
    let start=Instant::now();let bytes=fs::read(&a[1])?;
    let artifact:MicroArtifact=serde_json::from_slice(&bytes)?;let model=Arc::new(artifact.model()?);
    let loading=start.elapsed().as_secs_f64();let config:gen5::Options=serde_json::from_slice(&fs::read(&a[2])?)?;
    let dataset:Value=serde_json::from_slice(&fs::read(&a[3])?)?;
    let mut selected:BTreeMap<String,(String,bool)>=BTreeMap::new();
    for line in fs::read_to_string(&a[4])?.lines() {let row:Value=serde_json::from_str(line)?;
        selected.insert(row["source"].as_str().unwrap().into(),(row["route"].as_str().unwrap().into(),row["held_out"].as_bool().unwrap()));}
    let mut output=std::io::BufWriter::new(fs::File::create(out.join("searches.jsonl"))?);
    let mut roots_file=std::io::BufWriter::new(fs::File::create(out.join("roots.jsonl"))?);
    let mut roots=0;let mut searches=0;let mut search_seconds=0.;let mut certificates=0;
    for (game_index,(source,(route,held))) in selected.iter().enumerate() {
        let g=dataset["games"].as_array().unwrap().iter().find(|g|g["game_sha256"]==source.as_str()).ok_or("source missing")?;
        let original=&g["originals"][0];let b=fs::read(original["path"].as_str().unwrap())?;
        if hash(&b)!=original["sha256"] {return Err("source hash mismatch".into());}
        let old:GameRecord=std::str::from_utf8(&b)?.parse()?;
        let (record,_)=old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?;
        let n=record.actions().len();let wanted:BTreeSet<_>=[n.saturating_sub(32),n-12,n-4,n-1].into_iter().collect();
        let mut sessions=BTreeMap::new();let mut order=vec![];
        for budget in [32,64,128,256,512] {for floor in [0,6,8,16,32] {
            let mut s=MicroMctsSession::new(model.clone());s.set_root_value_strength(config.value_policy_strength)?;
            s.diagnostic_set_depth_floor(Some(floor))?;sessions.insert((budget,floor),s);order.push((budget,floor));}}
        // Balance order across independent games, keeping each arm's retained tree.
        if game_index%2==1 {order.reverse();}let len=order.len();order.rotate_left(game_index%len);
        let mut p=record.initial_position();
        for (decision,&played) in record.actions().iter().enumerate() {
            if wanted.contains(&decision) {
                let key=format!("{source}:{decision}");let seed=95717+game_index as u64*1000+decision as u64+seed_offset;
                let options=config.search(seed,true)?;
                let mut immediate=vec![];let legal=legal_actions(&p);
                for &action in &legal {let mut next=p.clone();next.apply(action)?;
                    if next.outcome()==GameOutcome::Win(p.to_move()) {immediate.push(action.to_string());}}
                let mut winning:BTreeSet<String>=immediate.iter().cloned().collect();
                let mut emitted=vec![];
                for &(budget,floor) in &order {
                    let s=sessions.get_mut(&(budget,floor)).unwrap();let inherited=s.retained_visits();let t=Instant::now();
                    let r=s.search_with_options(&p,budget,None,options)?;let seconds=t.elapsed().as_secs_f64();search_seconds+=seconds;
                    let stats=s.diagnostic_depth_trial_stats();
                    if stats.expansions>budget || stats.leaves_by_depth.iter().sum::<usize>()!=r.simulations {return Err("expansion/rollout accounting mismatch".into());}
                    let selected=r.actions[r.selected_index].to_string();let mut proofs=vec![];
                    for (i,&v) in r.proven_action_values.iter().enumerate() {if v==Some(1) {
                        let certificate=s.diagnostic_action_certificate(i).ok_or("missing winning child certificate")?;
                        let mut next=p.clone();next.apply(r.actions[i])?;
                        if certificate.verify(&next)?!=GameOutcome::Win(p.to_move()) {return Err("invalid winning certificate".into());}
                        winning.insert(r.actions[i].to_string());certificates+=1;
                        proofs.push(json!({"action":r.actions[i].to_string(),"certificate":certificate}));
                    }}
                    emitted.push(json!({"key":key,"source":source,"route":route,"held_out":held,"decision":decision,"seed":seed,
                        "budget":budget,"floor":floor,"root_rollouts":r.simulations,"stats":stats,"seconds":seconds,
                        "inherited":inherited,"inference_evaluations":r.inference_evaluations,"tactical_evaluations":r.tactical_evaluations,
                        "legal":r.actions.len(),"visited_root_actions":r.visits.iter().filter(|&&v|v>0).count(),
                        "selected":selected,"proven_value":r.proven_value,"proofs":proofs,
                        "selected_index":r.selected_index,"actions":r.actions.iter().map(|a|a.to_string()).collect::<Vec<_>>(),
                        "values_bits":r.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"visits":r.visits,
                        "policy_bits":r.policy_target.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}));
                    searches+=1;
                }
                for mut row in emitted {row["selected_in_verified_support"]=json!(winning.contains(row["selected"].as_str().unwrap()));
                    serde_json::to_writer(&mut output,&row)?;writeln!(output)?;}
                serde_json::to_writer(&mut roots_file,&json!({"key":key,"source":source,"decision":decision,"route":route,"held_out":held,
                    "immediate_wins":immediate,"verified_wins_union":winning,"legal":legal.len()}))?;writeln!(roots_file)?;
                roots+=1;output.flush()?;roots_file.flush()?;
            }
            p.apply(played)?;for s in sessions.values_mut() {s.advance(played)?;}
        }
        eprintln!("{} / {} games, {roots} roots, {searches} searches",game_index+1,selected.len());
    }
    fs::write(out.join("summary.json"),serde_json::to_vec_pretty(&json!({"model_sha256":hash(&bytes),"model_identity":artifact.identity(),
        "loading_seconds":loading,"wall_seconds":start.elapsed().as_secs_f64(),"native_search_seconds":search_seconds,
        "roots":roots,"searches":searches,"verified_certificates":certificates,"seed_offset":seed_offset,
        "budgets":[32,64,128,256,512],"floors":[0,6,8,16,32],"maximum_depth":96,
        "limits":["new-node cap, not equal root rollout count or equal wall time","same human sources as auxiliary study plus earlier prefix","verified-support miss means unknown, not proved loss","no training/publication or campaign activation"]}))?)?;
    Ok(())
}
