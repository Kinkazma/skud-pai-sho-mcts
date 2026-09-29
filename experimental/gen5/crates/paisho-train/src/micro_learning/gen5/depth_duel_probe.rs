//! Isolated paired Gen5 depth-floor versus frozen Gen3.5 games. No learner,
//! archive admission, publication, process controller or production output path.
use super::*;
use crate::compact_selfplay::reuse::Repetitions;
use paisho_core::{legal_actions,StandardSetup,BasicFlower};
use serde_json::{json,Value};
use std::io::Write;

#[doc(hidden)]
pub fn run(plan_path:&Path,out:&Path)->Result<Value> {
    let plan_bytes=fs::read(plan_path)?;let plan:Value=serde_json::from_slice(&plan_bytes)?;
    if plan["schema"]!="gen5-depth-duel-v1" {return Err(invalid("unknown depth duel schema"));}
    fs::create_dir(out)?;fs::create_dir(out.join("games"))?;
    let load=Instant::now();let model_bytes=fs::read(plan["model"].as_str().ok_or_else(||invalid("model missing"))?)?;
    if sha256(&model_bytes)!=plan["model_sha256"] {return Err(invalid("V5 changed"));}
    let artifact:MicroArtifact=serde_json::from_slice(&model_bytes)?;let model=Arc::new(artifact.model()?);
    let config_bytes=fs::read(plan["config"].as_str().ok_or_else(||invalid("config missing"))?)?;
    if sha256(&config_bytes)!=plan["config_sha256"] {return Err(invalid("config changed"));}
    let config:Options=serde_json::from_slice(&config_bytes)?;
    let spec:OpponentSpec=serde_json::from_value(plan["opponent"].clone())?;
    if spec.generation!="3.5" {return Err(invalid("requires Gen3.5"));}
    let opponent=Frozen::load(&spec)?;let load_seconds=load.elapsed().as_secs_f64();
    let maximum=plan["maximum_new_decisions"].as_u64().ok_or_else(||invalid("decision limit missing"))? as usize;
    if maximum==0 || maximum>800 {return Err(invalid("decision limit must be 1..=800"));}
    let mut starts:Vec<(String,GameRecord)>=vec![];
    for flower in [BasicFlower::Red3,BasicFlower::White5] {
        starts.push((format!("initial-{}",flower.code()),GameRecord::with_rules(StandardSetup::balanced(flower),RULES)));
    }
    for item in plan["human_starts"].as_array().ok_or_else(||invalid("starts missing"))? {
        let bytes=fs::read(item["path"].as_str().ok_or_else(||invalid("start path missing"))?)?;
        if sha256(&bytes)!=item["sha256"] {return Err(invalid("start source changed"));}
        let original:GameRecord=std::str::from_utf8(&bytes)?.parse()?;
        let (r,_)=original.replay_prefix_with_rules(RULES)?;
        let decisions=item["decisions"].as_u64().ok_or_else(||invalid("start decisions missing"))? as usize;
        if decisions>=r.actions().len() {return Err(invalid("start outside source"));}
        starts.push((item["label"].as_str().unwrap().into(),cases::prefix(&r,decisions)));
    }
    if starts.len()!=6 {return Err(invalid("six starts required"));}
    let began=Instant::now();let mut games=vec![];let mut journal=std::io::BufWriter::new(fs::File::create(out.join("games.jsonl"))?);
    let mut decisions=std::io::BufWriter::new(fs::File::create(out.join("decisions.jsonl"))?);
    for budget in [32,64,128,256] {for (start_index,(label,initial)) in starts.iter().enumerate() {for seat in [Player::Host,Player::Guest] {
        let mut floors=[0,16,32];floors.rotate_left((start_index+usize::from(seat==Player::Guest))%3);
        for floor in floors {
            let id=games.len();let seed=97191+1000*start_index as u64+u64::from(seat==Player::Guest);
            let mut record=initial.clone();let prefix=record.actions().len();let (mut p,mut repetitions)=cases::context(&record)?;
            let mut v5=MicroMctsSession::new(model.clone());v5.set_root_value_strength(config.value_policy_strength).map_err(invalid)?;
            v5.diagnostic_set_depth_floor(Some(floor)).map_err(invalid)?;
            let expansion_cap=budget*floor.max(1);
            v5.diagnostic_set_depth_expansion_budget(Some(expansion_cap)).map_err(invalid)?;
            let mut v35=MctsSession::new(seed,MctsConfig{simulations:budget,..Default::default()},&opponent).map_err(invalid)?;
            v35.set_solver(spec.solver);let options=config.search(seed,false).map_err(invalid)?;
            let t=Instant::now();let mut v5_seconds=0.;let mut v35_seconds=0.;let mut termination="decision-limit";
            for decision in 0..maximum {
                if p.outcome()!=GameOutcome::Ongoing {termination="rules-terminal";break;}
                let mover=p.to_move();let legal=legal_actions(&p);
                if legal.is_empty() {termination="unresolved-no-legal-action";break;}
                let t=Instant::now();
                let (action,metrics)=if mover==seat {
                    let r=v5.search_with_options(&p,budget,None,options).map_err(invalid)?;let seconds=t.elapsed().as_secs_f64();v5_seconds+=seconds;
                    if v5.diagnostic_depth_trial_stats().expansions>expansion_cap {return Err(invalid("V5 expansion cap violated"));}
                    (r.actions[r.selected_index],json!({"engine":"V5","seconds":seconds,"rollouts":r.simulations,"new_node_budget":expansion_cap,
                        "floor_stats":v5.diagnostic_depth_trial_stats(),"inference_evaluations":r.inference_evaluations,
                        "tactical_evaluations":r.tactical_evaluations,"visited_root_actions":r.visits.iter().filter(|&&n|n>0).count()}))
                } else {
                    let r=v35.search_until(&p,&legal,None).map_err(invalid)?;let seconds=t.elapsed().as_secs_f64();v35_seconds+=seconds;
                    (legal[r.selected_index],json!({"engine":"Gen3.5","seconds":seconds,"rollouts":r.simulations,"expanded_nodes":r.expanded_nodes,
                        "evaluated_actions":r.evaluated_actions,"maximum_depth":r.maximum_depth,"visited_root_actions":r.visited_root_actions()}))
                };
                serde_json::to_writer(&mut decisions,&json!({"game":id,"decision":decision,"mover":mover.code().to_string(),"action":action.to_string(),"metrics":metrics}))?;writeln!(decisions)?;
                p.apply(action)?;record.push(action);v5.advance(action).map_err(invalid)?;v35.advance(action);
                if repetitions.observe(&p,mover,record.actions().len()).is_some() {termination="unresolved-repetition";break;}
            }
            if p.outcome()!=GameOutcome::Ongoing {termination="rules-terminal";}
            let outcome=match p.outcome() {GameOutcome::Win(w) if w==seat=>"win",GameOutcome::Win(_)=>"loss",GameOutcome::Draw=>"draw",_=>"unresolved"};
            if record.replay()?.outcome()!=p.outcome() {return Err(invalid("saved game replay mismatch"));}
            let psr=record.to_string();fs::write(out.join("games").join(format!("{id:03}.psr")),&psr)?;
            let game=json!({"id":id,"budget":budget,"floor":floor,"start":label,"start_index":start_index,"seat":seat.code().to_string(),"seed":seed,
                "prefix_decisions":prefix,"new_decisions":record.actions().len()-prefix,"outcome":outcome,"termination":termination,
                "seconds":t.elapsed().as_secs_f64(),"v5_seconds":v5_seconds,"gen35_seconds":v35_seconds,"psr_sha256":sha256(psr.as_bytes())});
            serde_json::to_writer(&mut journal,&game)?;writeln!(journal)?;journal.flush()?;decisions.flush()?;games.push(game);
            fs::write(out.join("progress.json"),serde_json::to_vec_pretty(&json!({"complete":false,"games":games.len(),"planned":144,"active_seconds":began.elapsed().as_secs_f64(),"last":games.last()}))?)?;
            eprintln!("{}/144 budget {budget} floor {floor} {label} {} {outcome}",games.len(),seat.code());
        }
    }}}
    let report=json!({"complete":true,"plan_sha256":sha256(&plan_bytes),"model_sha256":sha256(&model_bytes),"opponent":spec,
        "loading_seconds":load_seconds,"active_seconds":began.elapsed().as_secs_f64(),"games":games,"learning_updates":0,
        "reference_loads":1,"v5_loads":1,"maximum_new_decisions":maximum,"exploration":false,
        "limits":["small paired development test, not Elo","same maximum root rollout budget; floor arms explicitly permit extra node expansions; distinct search kernels","unfinished games remain unresolved, not draws or regulatory defeats","weights fixed; no learning or publication tested"]});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;Ok(report)
}
