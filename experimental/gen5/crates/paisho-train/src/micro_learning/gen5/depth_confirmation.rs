//! Large frozen depth-floor comparison. One resident model/bank pair, private
//! deterministic game trees, independent output files, no production writer.
use super::*;
use paisho_core::legal_actions;
use rayon::prelude::*;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

struct Start { label: String, record: GameRecord, seed: u64 }

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Job { id: String, start: usize, budget: usize, floor: usize, guest: bool }

fn play(job: &Job, start: &Start, model: &Arc<MicroModel>, config: &Options,
        opponent: &Frozen, maximum: usize, out: &Path) -> Result<Value> {
    if ![32,64,128,256].contains(&job.budget) || ![0,5,6,8,12,16,32].contains(&job.floor)
        || job.id.is_empty() || !job.id.bytes().all(|c| c.is_ascii_alphanumeric() || c==b'-') {
        return Err(invalid("invalid diagnostic job"));
    }
    let dir=out.join("games").join(&job.id); fs::create_dir(&dir)?;
    let seat=if job.guest {Player::Guest} else {Player::Host};
    let seed=start.seed+u64::from(job.guest);
    let mut record=start.record.clone(); let prefix=record.actions().len();
    let (mut p,mut repetitions)=cases::context(&record)?;
    let mut v5=MicroMctsSession::new(model.clone());
    v5.set_root_value_strength(config.value_policy_strength).map_err(invalid)?;
    v5.diagnostic_set_depth_floor(Some(job.floor)).map_err(invalid)?;
    let cap=job.budget*job.floor.max(1);
    v5.diagnostic_set_depth_expansion_budget(Some(cap)).map_err(invalid)?;
    let mut v35=MctsSession::new(seed,MctsConfig{simulations:job.budget,..Default::default()},opponent).map_err(invalid)?;
    v35.set_solver(opponent.spec.solver);
    let options=config.search(seed,false).map_err(invalid)?;
    let began=Instant::now(); let mut v5_seconds=0.; let mut v35_seconds=0.;
    let mut termination="decision-limit"; let mut rows=Vec::new();
    for decision in 0..maximum {
        if p.outcome()!=GameOutcome::Ongoing {termination="rules-terminal";break;}
        let mover=p.to_move(); let legal=legal_actions(&p);
        if legal.is_empty() {termination="unresolved-no-legal-action";break;}
        let t=Instant::now();
        let (action,metrics)=if mover==seat {
            let r=v5.search_with_options(&p,job.budget,None,options).map_err(invalid)?;
            let seconds=t.elapsed().as_secs_f64(); v5_seconds+=seconds;
            if r.simulations>job.budget || v5.diagnostic_depth_trial_stats().expansions>cap {
                return Err(invalid("V5 budget violated"));
            }
            (r.actions[r.selected_index],json!({"engine":"V5","seconds":seconds,"rollouts":r.simulations,
                "new_node_budget":cap,"floor_stats":v5.diagnostic_depth_trial_stats(),
                "inference_evaluations":r.inference_evaluations,"tactical_evaluations":r.tactical_evaluations,
                "visited_root_actions":r.visits.iter().filter(|&&n|n>0).count()}))
        } else {
            let r=v35.search_until(&p,&legal,None).map_err(invalid)?;
            let seconds=t.elapsed().as_secs_f64(); v35_seconds+=seconds;
            if r.simulations>job.budget {return Err(invalid("Gen3.5 budget violated"));}
            (legal[r.selected_index],json!({"engine":"Gen3.5","seconds":seconds,"rollouts":r.simulations,
                "expanded_nodes":r.expanded_nodes,"evaluated_actions":r.evaluated_actions,
                "maximum_depth":r.maximum_depth,"visited_root_actions":r.visited_root_actions()}))
        };
        rows.push(json!({"decision":decision,"mover":mover.code().to_string(),"action":action.to_string(),"metrics":metrics}));
        p.apply(action)?; record.push(action); v5.advance(action).map_err(invalid)?; v35.advance(action);
        if repetitions.observe(&p,mover,record.actions().len()).is_some() {termination="unresolved-repetition";break;}
    }
    if p.outcome()!=GameOutcome::Ongoing {termination="rules-terminal";}
    let outcome=match p.outcome() {GameOutcome::Win(w) if w==seat=>"win",GameOutcome::Win(_)=>"loss",
        GameOutcome::Draw=>"draw",_=>"unresolved"};
    if record.replay()?.outcome()!=p.outcome() {return Err(invalid("saved game replay mismatch"));}
    let psr=record.to_string(); fs::write(dir.join("game.psr"),&psr)?;
    let mut decisions=std::io::BufWriter::new(fs::File::create(dir.join("decisions.jsonl"))?);
    for row in rows {serde_json::to_writer(&mut decisions,&row)?;writeln!(decisions)?;}
    decisions.flush()?;
    let game=json!({"id":job.id,"budget":job.budget,"floor":job.floor,"start":start.label,
        "start_index":job.start,"seat":seat.code().to_string(),"seed":seed,"prefix_decisions":prefix,
        "new_decisions":record.actions().len()-prefix,"outcome":outcome,"termination":termination,
        "seconds":began.elapsed().as_secs_f64(),"v5_seconds":v5_seconds,"gen35_seconds":v35_seconds,
        "psr_sha256":sha256(psr.as_bytes())});
    fs::write(dir.join("result.json.tmp"),serde_json::to_vec(&game)?)?;
    fs::rename(dir.join("result.json.tmp"),dir.join("result.json"))?; Ok(game)
}

fn starts(plan: &Value, out: &Path) -> Result<Vec<Start>> {
    let mut starts=Vec::new(); let mut metadata=Vec::new(); let mut skipped=Vec::new();
    let mut seen=std::collections::HashSet::new();
    for item in plan["starts"].as_array().ok_or_else(||invalid("starts missing"))? {
        let result=(|| -> Result<GameRecord> {
            if let Some(index)=item["initial_index"].as_u64() {
                let setup=super::super::compare::comparison_setup(index as usize,0,true);
                if [paisho_core::BasicFlower::Red3,paisho_core::BasicFlower::White5].iter()
                    .any(|&f|setup==paisho_core::StandardSetup::balanced(f)) {
                    return Err(invalid("pilot opening excluded"));
                }
                return Ok(GameRecord::with_rules(setup,RULES));
            }
            let bytes=fs::read(item["path"].as_str().ok_or_else(||invalid("start path missing"))?)?;
            if sha256(&bytes)!=item["sha256"] {return Err(invalid("source changed"));}
            let original:GameRecord=std::str::from_utf8(&bytes)?.parse()?;
            let (r,_)=original.replay_prefix_with_rules(RULES)?;
            let n=if let Some(n)=item["decisions"].as_u64() {n as usize} else {
                r.actions().len()*item["prefix_percent"].as_u64().ok_or_else(||invalid("prefix percentage missing"))? as usize/100
            };
            if n>=r.actions().len() || n==0 {return Err(invalid("empty or terminal prefix"));}
            Ok(cases::prefix(&r,n))
        })();
        // Exclusions happen before model search and are fully enumerated.
        let record=match result {Ok(r)=>r,Err(e)=>{skipped.push(json!({"item":item,"reason":e.to_string()}));continue;}};
        let (_,p)=record.replay_prefix_with_rules(RULES)?;
        if p.outcome()!=GameOutcome::Ongoing {skipped.push(json!({"item":item,"reason":"already terminal"}));continue;}
        let psr=record.to_string(); let hash=sha256(psr.as_bytes());
        if !seen.insert(hash.clone()) {skipped.push(json!({"item":item,"reason":"duplicate exact prefix"}));continue;}
        let index=starts.len(); let label=item["label"].as_str().ok_or_else(||invalid("label missing"))?.to_string();
        let seed=item["seed"].as_u64().ok_or_else(||invalid("seed missing"))?;
        fs::write(out.join("starts").join(format!("{index:04}.psr")),&psr)?;
        metadata.push(json!({"index":index,"label":label,"seed":seed,"prefix_decisions":record.actions().len(),
            "prefix_sha256":hash,"origin":item}));
        starts.push(Start{label,record,seed});
    }
    fs::write(out.join("population.json"),serde_json::to_vec_pretty(&json!({"starts":metadata,"excluded":skipped}))?)?;
    Ok(starts)
}

/// JSON-lines controller protocol. `jobs` command executes one bounded batch;
/// `finish` closes the resident process. No timing deadline affects actions.
#[doc(hidden)]
pub fn run(plan_path:&Path,out:&Path)->Result<()> {
    let bytes=fs::read(plan_path)?; let plan:Value=serde_json::from_slice(&bytes)?;
    if plan["schema"]!="gen5-depth-confirmation-v1" {return Err(invalid("unknown confirmation schema"));}
    fs::create_dir(out)?;fs::create_dir(out.join("games"))?;fs::create_dir(out.join("starts"))?;
    fs::write(out.join("plan.json"),&bytes)?;
    let load=Instant::now();let model_bytes=fs::read(plan["model"].as_str().ok_or_else(||invalid("model missing"))?)?;
    if sha256(&model_bytes)!=plan["model_sha256"] {return Err(invalid("V5 changed"));}
    let artifact:MicroArtifact=serde_json::from_slice(&model_bytes)?;let model=Arc::new(artifact.model()?);
    let config_bytes=fs::read(plan["config"].as_str().ok_or_else(||invalid("config missing"))?)?;
    if sha256(&config_bytes)!=plan["config_sha256"] {return Err(invalid("config changed"));}
    let config:Options=serde_json::from_slice(&config_bytes)?;
    let spec:OpponentSpec=serde_json::from_value(plan["opponent"].clone())?;
    if spec.generation!="3.5" {return Err(invalid("requires Gen3.5"));}
    let opponent=Frozen::load(&spec)?;let model_load_seconds=load.elapsed().as_secs_f64();
    let maximum=plan["maximum_new_decisions"].as_u64().ok_or_else(||invalid("decision limit missing"))? as usize;
    if maximum==0 || maximum>800 {return Err(invalid("decision limit must be 1..=800"));}
    let starts=starts(&plan,out)?;
    let workers=plan["workers"].as_u64().ok_or_else(||invalid("workers missing"))? as usize;
    if !(1..=10).contains(&workers) {return Err(invalid("workers must be 1..=10"));}
    let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build()?;
    let serial=rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    println!("{}",json!({"ready":true,"population":starts.len(),"loading_seconds":model_load_seconds}));
    std::io::stdout().flush()?;
    let mut game_count=0;let mut active_seconds=0.;
    for line in std::io::stdin().lock().lines() {
        let command:Value=serde_json::from_str(&line?)?;
        if command["finish"]==true {break;}
        let jobs:Vec<Job>=serde_json::from_value(command["jobs"].clone())?;
        if jobs.iter().any(|j|j.start>=starts.len()) {return Err(invalid("invalid start index"));}
        let t=Instant::now();let selected=if command["serial"]==true {&serial} else {&pool};
        let results:Vec<std::result::Result<Value,String>>=selected.install(|| jobs.par_iter().map(|job|
            play(job,&starts[job.start],&model,&config,&opponent,maximum,out).map_err(|e|e.to_string())).collect());
        let batch_seconds=t.elapsed().as_secs_f64();active_seconds+=batch_seconds;
        let results=results.into_iter().collect::<std::result::Result<Vec<_>,_>>().map_err(invalid)?;
        game_count+=results.len();
        println!("{}",json!({"batch_complete":true,"games":results,"batch_seconds":batch_seconds,"active_seconds":active_seconds}));
        std::io::stdout().flush()?;
    }
    let report=json!({"complete":true,"plan_sha256":sha256(&bytes),"model_sha256":sha256(&model_bytes),
        "reference_loads":1,"v5_loads":1,"loading_seconds":model_load_seconds,"active_seconds":active_seconds,
        "games":game_count,"workers":workers,"learning_updates":0,"exploration":false});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("{}",report);Ok(())
}
