//! Isolated rule relations / auxiliary-target export and campaign-budget depth census.
//! Uses saved production weights read-only, exact current teaching coordinates,
//! real human continuations, and Options::search for campaign exploration.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{gen5,MicroArtifact};
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::BTreeMap,fs,io::Write,path::Path,sync::Arc,time::Instant};
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(bytes:&[u8])->String {format!("{:x}",Sha256::digest(bytes))}
fn sign(outcome:GameOutcome,player:Player)->Option<f64> {
    match outcome {GameOutcome::Win(w)=>Some(if w==player {1.} else {-1.}),GameOutcome::Draw=>Some(0.),_=>None}
}
fn ending(r:&GameRecord,p:&Position)->&'static str {
    if p.outcome()==GameOutcome::Ongoing {return "unresolved";}
    if matches!(r.actions().last(),Some(Action::Plant{..}|Action::BonusPlantBasic{..}))
        && [Player::Host,Player::Guest].into_iter().any(|p0|p.reserve(p0).basic_count()==0) {return "exhaustion";}
    if !harmony_ring_owners_for_profile(p.board(),p.rule_profile()).is_empty() {"ring"} else {"other"}
}
fn session(model:&Arc<MicroModel>,cap:usize,beta:f64)->MicroMctsSession {
    let mut s=MicroMctsSession::new(model.clone());s.set_root_value_strength(beta).unwrap();
    s.diagnostic_set_maximum_depth(cap).unwrap();s
}
fn main()->Result<()> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=5 {return Err("MODEL CONFIG HUMAN_DATASET NEW_OUTPUT_DIR".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let out=Path::new(&args[4]);fs::create_dir(out)?;
    let start=Instant::now();let bytes=fs::read(&args[1])?;
    let artifact:MicroArtifact=serde_json::from_slice(&bytes)?;
    let model=Arc::new(artifact.model()?);let loading=start.elapsed().as_secs_f64();
    let config:gen5::Options=serde_json::from_slice(&fs::read(&args[2])?)?;
    let data:Value=serde_json::from_slice(&fs::read(&args[3])?)?;
    let mut games:Vec<_>=data["games"].as_array().ok_or("games missing")?.iter().collect();
    games.sort_by_key(|g|g["game_sha256"].as_str().unwrap());
    let mut counts:BTreeMap<(String,bool),usize>=BTreeMap::new();let mut chosen=vec![];let mut scanned=0;
    // Selection depends only on verified ending, existing split, source hash and
    // length. It never consults model predictions. Balance both victory routes.
    for g in games {
        let held=g["held_out"].as_bool().unwrap();
        let original=&g["originals"][0];let b=fs::read(original["path"].as_str().unwrap())?;
        if hash(&b)!=original["sha256"] {return Err("human source hash changed".into());}
        let old:GameRecord=std::str::from_utf8(&b)?.parse()?;
        let (r,p)=old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?;scanned+=1;
        let route=ending(&r,&p);
        if r.actions().len()<16 || !matches!(p.outcome(),GameOutcome::Win(_)) || !["ring","exhaustion"].contains(&route) {continue;}
        let count=counts.entry((route.into(),held)).or_default();
        if *count>=if held {4} else {12} {continue;}
        *count+=1;chosen.push((g.clone(),r,route.to_owned()));
        if counts.len()==4 && counts.iter().all(|((_,h),n)|*n>=if *h {4}else{12}) {break;}
    }
    if counts.len()!=4 {return Err("both ending routes and splits required".into());}
    let mut rows=std::io::BufWriter::new(fs::File::create(out.join("rows.jsonl"))?);
    let mut depth=std::io::BufWriter::new(fs::File::create(out.join("depth.jsonl"))?);
    let mut produced=0;let mut depth_searches=0;let mut native_seconds=0.;let mut features_seconds=0.;
    let mut profile_counts:BTreeMap<String,usize>=BTreeMap::new();
    for (game_index,(g,r,route)) in chosen.iter().enumerate() {
        let profile=profile_counts.entry(route.clone()).or_default();let profile_game=*profile<6;*profile+=1;
        let mut sessions:BTreeMap<(usize,usize),MicroMctsSession>=BTreeMap::new();
        if profile_game {for budget in [256,512] {for cap in [4,96,192] {sessions.insert((budget,cap),session(&model,cap,config.value_policy_strength));}}}
        let n=r.actions().len();let wanted=[n-12,n-4,n-1];let mut p=r.initial_position();
        for (decision,&played) in r.actions().iter().enumerate() {
            if wanted.contains(&decision) {
                let key=hash(format!("{}:{decision}",g["game_sha256"]).as_bytes());
                let seed=95717+game_index as u64*1000+decision as u64;
                let options=config.search(seed,true)?;
                let mut reports=BTreeMap::new();
                for (&(budget,cap),s) in &mut sessions {
                    let inherited=s.retained_visits();let t=Instant::now();
                    let report=s.search_with_options(&p,budget,None,options)?;let seconds=t.elapsed().as_secs_f64();
                    native_seconds+=seconds;depth_searches+=1;
                    let nodes=s.diagnostic_visited_nodes_by_depth();
                    let mut selected_next=p.clone();selected_next.apply(report.actions[report.selected_index])?;
                    let census=json!({"key":key,"source":g["game_sha256"],"decision":decision,"route":route,
                        "budget":budget,"cap":cap,"retained_visits_before":inherited,"new_simulations":report.simulations,
                        "tactical_evaluations":report.tactical_evaluations,"legal":report.actions.len(),"seconds":seconds,
                        "visited_nodes_by_depth":nodes,"selected":report.actions[report.selected_index].to_string(),
                        "selected_terminal":sign(selected_next.outcome(),p.to_move()),
                        "values_bits":report.values.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),
                        "visits":report.visits,"policy_bits":report.policy_target.iter().map(|v|v.to_bits()).collect::<Vec<_>>()});
                    serde_json::to_writer(&mut depth,&census)?;writeln!(depth)?;
                    reports.insert((budget,cap),report);
                }
                let teacher_budget=if game_index%2==0 {256}else{512};
                let report=match reports.remove(&(teacher_budget,96)) {
                    Some(r)=>r,None=>{let t=Instant::now();let r=session(&model,96,config.value_policy_strength).search_with_options(&p,teacher_budget,None,options)?;native_seconds+=t.elapsed().as_secs_f64();r}
                };
                let t=Instant::now();let state=model.state_features(&p);let e=model.embed(&state);
                let features:Vec<_>=report.actions.iter().map(|&a|micro_action_features(&p,a)).collect();
                let prior=model.memory_priors(&state,&features,&micro_softmax(&MicroModel::logits(&e,&features))?,0)?;
                let (target,_) =gen5::diagnostic_teaching_target(&report,&prior)?;
                let rel=MicroRelations::extract(&p,p.to_move());
                let mut auxiliary=vec![];let mut next_values=vec![];let mut terminals=vec![];let mut q_targets=vec![];
                for (i,&a) in report.actions.iter().enumerate() {
                    let mut next=p.clone();next.apply(a)?;
                    let z=sign(next.outcome(),p.to_move());terminals.push(z);
                    next_values.push(z.unwrap_or_else(||model.embed(&model.state_features(&next)).value*if next.to_move()==p.to_move(){1.}else{-1.}));
                    q_targets.push(z.or_else(||report.proven_action_values[i].map(|v|v as f64)).or_else(||(report.visits[i]>0).then_some(report.values[i])));
                    auxiliary.push(micro_relation_targets(&p,a)?.to_vec());
                }
                let tokens:Vec<_>=features.iter().map(|a|rel.action_tokens(a).iter().map(|v|v.to_vec()).collect::<Vec<_>>()).collect();
                let row=json!({"key":key,"source":g["game_sha256"],"split_identity":g["split_identity_sha256"],
                    "held_out":g["held_out"],"route":route,"decision":decision,"remaining_recorded_decisions":n-decision,
                    "teacher_budget":teacher_budget,"state":state,"actions":features,"prior":prior,"next_values":next_values,
                    "terminal":terminals,"q_targets":q_targets,"policy_target":target,"proven":report.proven_value,
                    "global":rel.global,"edge_tokens":tokens,"auxiliary":auxiliary,
                    "edges":rel.edges.iter().map(|h|json!({"first":[h.first.x(),h.first.y()],"second":[h.second.x(),h.second.y()],"owner":h.owner.code().to_string(),"midline":harmony_crosses_midline(*h)})).collect::<Vec<_>>(),
                    "cycles":rel.cycles.iter().map(|c|json!({"owner":c.owner.code().to_string(),"geometry":format!("{:?}",c.geometry),"vertices":c.vertices.iter().map(|v|[v.x(),v.y()]).collect::<Vec<_>>()})).collect::<Vec<_>>()});
                features_seconds+=t.elapsed().as_secs_f64();
                serde_json::to_writer(&mut rows,&row)?;writeln!(rows)?;rows.flush()?;depth.flush()?;
                produced+=1;
                if produced%12==0 {eprintln!("{produced} roots exported; {depth_searches} searches profiled");}
            }
            p.apply(played)?;for s in sessions.values_mut() {s.advance(played)?;}
        }
    }
    fs::write(out.join("summary.json"),serde_json::to_vec_pretty(&json!({"schema":MICRO_RELATION_SCHEMA,
        "model_sha256":hash(&bytes),"model_identity":artifact.identity(),"model_updates":artifact.updates,
        "loading_seconds":loading,"wall_seconds":start.elapsed().as_secs_f64(),"native_search_seconds":native_seconds,
        "export_feature_and_target_seconds":features_seconds,"roots":produced,"depth_searches":depth_searches,
        "source_counts":counts.iter().map(|((r,h),n)|json!({"route":r,"held_out":h,"games":n})).collect::<Vec<_>>(),
        "scanned_records":scanned,"target_names":MICRO_RELATION_TARGET_NAMES,
        "configuration":{"mode":config.mode,"beta":config.value_policy_strength,"forced_playout_strength":config.forced_playout_strength,"dirichlet_fraction":config.dirichlet_fraction,"budgets":[256,512]},
        "limitations":["controlled saved-model replay; not live campaign throughput","held-out for this experiment, prior historical exposure unverified","tree census is not simulation-length histogram","no publication or campaign activation"]}))?)?;
    Ok(())
}
