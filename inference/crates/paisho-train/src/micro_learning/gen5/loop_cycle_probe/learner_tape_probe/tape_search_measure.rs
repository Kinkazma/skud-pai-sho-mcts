//! Same frozen tape roots, actual cold MCTS and source0 successor-value ranking.
//! No learning or proof installation. The full prior cache is read-only evidence.
use super::*;
use paisho_core::{Action,Position,legal_actions};
use serde_json::{json,Value};
use std::collections::{BTreeMap,BTreeSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema:String,source_plan:Input,tape_report:Input,tape_manifest:Input,proofs:Vec<Input>,
    models:Vec<ModelInput>,budgets:Vec<usize>,seed:u64,max_kernel_seconds:u64,
    cache_report:Option<Input>,relay_report:Option<Input>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInput {label:String,input:Input,parameter_bits_sha256:String}
struct Root {key:String,position:Position,actions:Vec<Action>,fixture:Value,example:Arc<MicroExample>}
fn best(x:&[f64])->usize {(0..x.len()).max_by(|&a,&b|x[a].total_cmp(&x[b]).then_with(||b.cmp(&a))).unwrap()}
fn same_input(a:&Input,b:&Input)->Result<bool> {Ok(a.sha256==b.sha256 && fs::canonicalize(&a.path)?==fs::canonicalize(&b.path)?)}
fn append(file:&mut fs::File,value:&Value)->Result<()> {
    serde_json::to_writer(&mut *file,value)?;file.write_all(b"\n")?;file.flush()?;Ok(())
}
fn complete(simulations:usize,inherited:usize,budget:usize,proof:Option<i8>)->bool {
    inherited==0 && simulations<=budget && (simulations==budget || proof.is_some())
}
fn finite_coupled(values:&[f64],logits:&[f64])->Result<()> {
    if values.iter().chain(logits).any(|v|!v.is_finite()) {return Err(invalid("nonfinite coupled inference"));}Ok(())
}
fn options_json(o:&MicroSearchOptions,beta:f64)->Value {
    json!({"mode":format!("{:?}",o.mode),"seed":o.seed,"dirichlet_fraction":o.dirichlet_fraction,
        "dirichlet_total":o.dirichlet_total,"gumbel_scale":o.gumbel_scale,"considered_actions":o.considered_actions,
        "forced_playout_strength":o.forced_playout_strength,"proof_search":o.proof_search,
        "root_value_strength":beta,"root_successor_fpu":false,"exploration":1.5,"maximum_depth":96,
        "cache_entries":65536,"memory_limit_bytes":536870912,"root_context_source":0})
}
// Same source0 operator as gen5_flow_recheck_panel, with the native independent
// value forward instead of also constructing a needless successor policy.
fn coupled(model:&MicroModel,root:&Root,beta:f64)->Result<Value> {
    let state=model.state_features(&root.position);
    let features=root.actions.iter().map(|&a|micro_action_features(&root.position,a)).collect::<Vec<_>>();
    let base=micro_softmax(&MicroModel::logits(&model.embed(&state),&features)).map_err(invalid)?;
    let raw=model.memory_priors(&state,&features,&base,0).map_err(invalid)?;
    let player=root.position.to_move();let mut values=Vec::with_capacity(raw.len());
    for &action in &root.actions {
        let mut child=root.position.clone();child.apply(action)?;
        let value=match child.outcome() {
            GameOutcome::Win(w)=>if w==player {1.}else{-1.},GameOutcome::Draw=>0.,
            GameOutcome::Ongoing=>model.value(&model.state_features(&child))*if child.to_move()==player {1.}else{-1.},
        };values.push(value);
    }
    let logits=raw.iter().zip(&values).map(|(p,q)|p.max(1e-300).ln()+beta*q).collect::<Vec<_>>();
    finite_coupled(&values,&logits)?;
    let selected=best(&logits);
    Ok(json!({"status":"complete","selected_index":selected,"selected":root.actions[selected].to_string(),
        "raw_selected_index":best(&raw),"raw_priors":raw,"coupled_logits":logits,"successor_values":values}))
}
fn search(model:Arc<MicroModel>,root:&Root,budget:usize,options:MicroSearchOptions,beta:f64,remaining:f64)->Result<(Value,f64)> {
    let mut session=MicroMctsSession::new(model);session.set_root_value_strength(beta).map_err(invalid)?;
    session.set_root_successor_fpu(false);session.set_limits(65536,536870912);
    if session.retained_visits()!=0 {return Err(invalid("nonempty diagnostic tree"));}
    let deadline=paisho_platform::training_time::now()+Duration::from_secs_f64(remaining.max(0.000001));
    let started=Instant::now();let result=session.search_with_options(&root.position,budget,Some(deadline),options);
    let seconds=started.elapsed().as_secs_f64();let r=match result {
        Ok(r)=>r,Err(e)=>return Ok((json!({"status":"native_error","error":e}),seconds)),
    };
    let valid=r.inherited_visits==0 && r.simulations<=budget;
    let done=complete(r.simulations,r.inherited_visits,budget,r.proven_value);
    let status=if !valid {"cold_tree_or_budget_error"}else if done {"complete"}else{"incomplete_deadline"};
    let exported=Instant::now();let certificate=session.certificate(10000);let export_seconds=exported.elapsed().as_secs_f64();
    let actions=r.actions.iter().map(ToString::to_string).collect::<Vec<_>>();
    if actions!=root.actions.iter().map(ToString::to_string).collect::<Vec<_>>() {return Err(invalid("native legal order changed"));}
    Ok((json!({"status":status,"selected_index":r.selected_index,"selected":r.actions[r.selected_index].to_string(),
        "completion_reason":if r.simulations==budget {"budget"}else if r.proven_value.is_some(){"root_proven"}else{"deadline"},
        "simulations":r.simulations,"inherited_visits":r.inherited_visits,"root_visits":session.retained_visits(),
        "inference_evaluations":r.inference_evaluations,"inference_cache_hits":r.inference_cache_hits,
        "tactical_evaluations":r.tactical_evaluations,"memory_reset":r.memory_reset,"retained_bytes":r.retained_bytes,
        "proven_value":r.proven_value,"proven_action_values":r.proven_action_values,"network_value":r.network_value,
        "actions":actions,"visits":r.visits,"new_visits":r.new_visits,"values":r.values,
        "raw_priors":r.raw_priors.as_deref(),"priors":r.priors,"search_priors":r.search_priors,
        "certificate":certificate,"certificate_export_seconds":export_seconds}),seconds))
}
fn query_key(meta:&Value)->Result<String> {Ok(sha256(&serde_json::to_vec(meta)?))}
pub fn run(plan_path:&Path,out:&Path)->Result<Value> {
    if out.exists() {return Err(invalid("search transfer output must be new"));}
    let started=Instant::now();let bytes=fs::read(plan_path)?;let plan:Plan=serde_json::from_slice(&bytes)?;
    if plan.schema!="paisho-gen5-tape-search-plan-v1" || plan.budgets!=[8,256] || plan.seed!=37
        || plan.max_kernel_seconds!=180 || plan.proofs.len()!=64 || plan.models.len()!=if plan.relay_report.is_some(){13}else{9} {
        return Err(invalid("fixed search transfer protocol changed"));
    }
    let source:TapePlan=serde_json::from_slice(&plan.source_plan.bytes()?)?;let report=json(&plan.tape_report)?;
    let tape=json(&plan.tape_manifest)?;
    if source.blocks.len()!=4 || report["schema"]!="paisho-gen5-learner-tape-result-v1"
        || report["plan_sha256"]!=plan.source_plan.sha256 || tape["plan_sha256"]!=plan.source_plan.sha256
        || !same_input(&input(&report["tape"])?,&plan.tape_manifest)? || report["same_tape_all_arms"]!=true
        || report["reset_replay_every_sgd_and_boundary_exact"]!=true {return Err(invalid("tape binding/verification changed"));}
    let proof_rows=tape["proof_cohort"].as_array().filter(|v|v.len()==64).ok_or_else(||invalid("same64 proofs missing"))?;
    let o:Options=serde_json::from_slice(&source.config.bytes()?)?;
    let options=o.search(plan.seed,false).map_err(invalid)?;
    if o.mode!="puct" || !o.proof_search || o.value_policy_strength!=16. {return Err(invalid("production search contract changed"));}
    let contract=options_json(&options,o.value_policy_strength);
    let relay=plan.relay_report.as_ref().map(json).transpose()?;
    if let Some(r)=&relay {if r["schema"]!="paisho-gen5-tape-policy-relay-result-v1"
        || !same_input(&input(&r["source_report"])?,&plan.tape_report)?
        || !same_input(&input(&r["source_plan"])?,&plan.source_plan)?
        || r["results"].as_array().map(Vec::len)!=Some(4) {return Err(invalid("relay provenance changed"));}}
    let engine=sha256(&fs::read(std::env::current_exe()?)?);
    let mut roots=vec![];let mut keys=BTreeSet::new();
    for (pi,p) in proof_rows.iter().enumerate() {
        let fixture=json(&plan.proofs[pi])?;let record:GameRecord=fixture["prefix"].as_str().ok_or_else(||invalid("proof prefix"))?.parse()?;
        let position=record.replay()?;let key=sha256(record.to_string().as_bytes());
        if p["key"]!=key || !keys.insert(key.clone()) || position.rule_profile()!=RULES
            || fixture["rules"]!=RULES.as_str() || position.outcome()!=GameOutcome::Ongoing {return Err(invalid("frozen proof root changed"));}
        let example=restore(&serde_json::from_value(p["example"].clone())?)?;
        let actions=legal_actions(&position);
        if example.actions!=actions.iter().map(|&a|micro_action_features(&position,a)).collect::<Vec<_>>() {
            return Err(invalid("proof example legal action features differ"));
        }
        roots.push(Root{key,position,actions,fixture,example});
    }
    let input_seconds=started.elapsed().as_secs_f64();let load_started=Instant::now();
    let mut loaded:BTreeMap<String,Arc<MicroModel>>=BTreeMap::new();let mut role_models=vec![];
    let mut first:Option<MicroModel>=None;let mut bank_spec=None;
    for (mi,m) in plan.models.iter().enumerate() {
        let (label,path,expected_bits)=if mi==0 {("initial-actor".into(),source.initial_actor.path.clone(),None)}
        else if mi<9 {let bi=(mi-1)/2;let role=if (mi-1)%2==0 {"working"}else{"actor"};let b=&report["arms"]["persistent"]["boundaries"][bi];
            (format!("persistent-{bi}-{role}"),PathBuf::from(b[role].as_str().ok_or_else(||invalid("boundary path"))?),b[format!("{role}_bits")].as_str().map(str::to_owned))}
        else {let bi=mi-9;let b=&relay.as_ref().unwrap()["results"][bi];
            if b["block"]!=bi || b["before_bits"]!=report["arms"]["persistent"]["boundaries"][bi]["actor_bits"] {return Err(invalid("relay seed changed"));}
            let published=input(&b["final_actor"])?;if !same_input(&published,&m.input)? {return Err(invalid("relay final file changed"));}
            (format!("relay-{bi}-actor"),published.path,b["final_actor_bits"].as_str().map(str::to_owned))};
        if m.label!=label || fs::canonicalize(path)?!=fs::canonicalize(&m.input.path)?
            || expected_bits.as_ref().is_some_and(|b|b!=&m.parameter_bits_sha256)
            || (mi==0 && !same_input(&m.input,&source.initial_actor)?) {return Err(invalid("frozen model role changed"));}
        let artifact:MicroArtifact=serde_json::from_slice(&m.input.bytes()?)?;
        let spec=serde_json::to_value(&artifact.sequence_memory)?;
        if let Some(old)=&bank_spec {if old!=&spec {return Err(invalid("model bank descriptor changed"));}}else{bank_spec=Some(spec);}
        if sha256(&artifact.parameters.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>())!=m.parameter_bits_sha256 {return Err(invalid("model coefficient bits changed"));}
        if !loaded.contains_key(&m.parameter_bits_sha256) {
            let (_,model)=load_model(&m.input,first.as_ref())?;
            if model.parameters().len()!=292363 {return Err(invalid("V5 architecture changed"));}
            if let Some(base)=&first {if !Arc::ptr_eq(base.sequence_memory().ok_or_else(||invalid("bank missing"))?,model.sequence_memory().ok_or_else(||invalid("bank missing"))?) {return Err(invalid("bank not shared"));}}
            else {first=Some(model.clone());}
            loaded.insert(m.parameter_bits_sha256.clone(),Arc::new(model));
        }
        role_models.push(loaded[&m.parameter_bits_sha256].clone());
    }
    let load_seconds=load_started.elapsed().as_secs_f64();
    for root in &roots {if role_models[0].state_features(&root.position)!=root.example.state {return Err(invalid("root state/recorded features differ"));}}
    let mut cache:BTreeMap<String,(Input,Value)>=BTreeMap::new();
    if let Some(src)=&plan.cache_report {
        let old=json(src)?;
        if old["schema"]!="paisho-gen5-tape-search-result-v1" || old["engine_binary_sha256"]!=engine
            || !same_input(&input(&old["tape_report"])?,&plan.tape_report)? || old["contract"]!=contract {
            return Err(invalid("cache belongs to different engine/tape/options"));
        }
        for q in old["queries"].as_array().ok_or_else(||invalid("cache query list"))? {
            let src=input(&q["raw"])?;let raw=json(&src)?;let key=query_key(&raw["metadata"])?;
            if raw["key"]!=key || raw["metadata"]["engine_binary_sha256"]!=engine {return Err(invalid("cache key changed"));}
            if raw["result"]["status"]=="complete" {cache.insert(key,(src,raw));}
        }
    }
    fs::create_dir_all(out)?;fs::create_dir(out.join("queries"))?;
    let mut journal=fs::OpenOptions::new().write(true).create_new(true).open(out.join("queries.jsonl"))?;
    let mut queries=vec![];let mut raw_rows=vec![];let mut kernel=0.;let mut coupling_seconds=0.;let mut search_seconds=0.;
    let mut fresh_queries=0;let mut errors=0;let mut exhausted=false;
    'roots: for (pi,root) in roots.iter().enumerate() {
        for offset in 0..plan.models.len() {let mi=(pi+offset)%plan.models.len();let model=&plan.models[mi];
            for budget in [0,8,256] {
                let metadata=json!({"contract_version":1,"engine_binary_sha256":engine,"parameters":model.parameter_bits_sha256,
                    "bank":bank_spec,"root":root.key,"rules":RULES.as_str(),"options":contract,"budget":budget,
                    "kind":if budget==0 {"coupled"}else{"search"}});let key=query_key(&metadata)?;
                let reused=cache.contains_key(&key);
                if !reused {
                    if kernel>=plan.max_kernel_seconds as f64 {exhausted=true;break 'roots;}
                    let (result,seconds)=if budget==0 {let t=Instant::now();let r=coupled(&role_models[mi],root,o.value_policy_strength)?;(r,t.elapsed().as_secs_f64())}
                        else {search(role_models[mi].clone(),root,budget,options,o.value_policy_strength,plan.max_kernel_seconds as f64-kernel)?};
                    kernel+=seconds;if budget==0 {coupling_seconds+=seconds;}else{search_seconds+=seconds;}
                    if result["status"]!="complete" {errors+=1;}
                    let raw=json!({"metadata":metadata,"key":key,"kernel_seconds":seconds,"result":result});
                    let src=write_json(&out.join("queries").join(format!("{key}.json")),&raw)?;
                    cache.insert(key.clone(),(src,raw));fresh_queries+=1;
                }
                let (src,raw)=&cache[&key];let q=json!({"position":pi,"model":mi,"label":model.label,"budget":budget,"raw":src,"reused":reused});
                append(&mut journal,&q)?;queries.push(q);raw_rows.push((pi,mi,budget,raw["result"].clone()));
                // Preserve the partial report/cache without reusing a failed or
                // deadline-truncated query through a later same-bit model role.
                if raw["result"]["status"]!="complete" {exhausted=raw["result"]["status"]=="incomplete_deadline";break 'roots;}
            }
        }
    }
    // All reference labels and new certificates are consulted AFTER the choices.
    let grading=Instant::now();let mut graded=vec![];let mut root_metrics=vec![];
    for (pi,root) in roots.iter().enumerate() {
        let cert:MicroProofCertificate=serde_json::from_value(root.fixture["certificate"].clone())?;
        let known=action_values::verified_winning_policy(&root.position,&root.actions,&cert)?;
        if !root.example.policy_support || known.iter().zip(&root.example.policy).any(|(a,b)|(*a>0.)!=(*b>0.)) {return Err(invalid("reference proof support changed"));}
        let immediate=root.actions.iter().map(|&a|{let mut p=root.position.clone();p.apply(a).unwrap();p.outcome()==GameOutcome::Win(root.position.to_move())}).collect::<Vec<_>>();
        let mut union=known.iter().map(|v|*v>0.).collect::<Vec<_>>();let mut new_certificates=0;let mut certificate_keys=BTreeSet::new();
        for (_,_,budget,r) in raw_rows.iter().filter(|(p,_,_,_)|*p==pi) {
            if *budget==0 || r["certificate"].is_null() {continue;}
            if !certificate_keys.insert(sha256(&serde_json::to_vec(&r["certificate"])?)) {continue;}
            let c:MicroProofCertificate=serde_json::from_value(r["certificate"].clone())?;
            if c.verify(&root.position).map_err(invalid)?==GameOutcome::Win(root.position.to_move()) {
                let support=action_values::verified_winning_policy(&root.position,&root.actions,&c)?;
                for (a,b) in union.iter_mut().zip(support) {*a|=b>0.;}new_certificates+=1;
            }
        }
        for (_,mi,budget,r) in raw_rows.iter().filter(|(p,_,_,_)|*p==pi) {
            let selected=r["selected_index"].as_u64().map(|v|v as usize);
            if let Some(i)=selected {if i>=root.actions.len() || r["selected"]!=root.actions[i].to_string() {return Err(invalid("cached selected action changed"));}}
            graded.push(json!({"position":pi,"key":root.key,"model":mi,"label":plan.models[*mi].label,"budget":budget,
                "status":r["status"],"selected":r["selected"],"known_support_selected":selected.map(|i|known[i]>0.),
                "symmetric_verified_selected":selected.map(|i|union[i]),"immediate_selected":selected.map(|i|immediate[i])}));
        }
        root_metrics.push(json!({"position":pi,"key":root.key,"first_block":proof_rows[pi]["first_block"],
            "immediate_actions":immediate.iter().filter(|v|**v).count(),"known_support_actions":known.iter().filter(|v|**v>0.).count(),
            "symmetric_support_actions":union.iter().filter(|v|**v).count(),"verified_unique_winning_search_certificates":new_certificates}));
    }
    let mut scores=vec![];
    for mi in 0..plan.models.len() {for budget in [0,8,256] {for stratum in ["all","immediate","nonimmediate"] {
        let rows=graded.iter().filter(|r|r["model"]==mi && r["budget"]==budget && (stratum=="all" ||
            (root_metrics[r["position"].as_u64().unwrap() as usize]["immediate_actions"].as_u64().unwrap()>0)==(stratum=="immediate"))).collect::<Vec<_>>();
        let done=rows.iter().filter(|r|r["status"]=="complete").collect::<Vec<_>>();
        scores.push(json!({"model":mi,"label":plan.models[mi].label,"budget":budget,"stratum":stratum,"observed":rows.len(),"complete":done.len(),
            "known_support_wins":done.iter().filter(|r|r["known_support_selected"]==true).count(),
            "symmetric_verified_wins":done.iter().filter(|r|r["symmetric_verified_selected"]==true).count()}));
    }}}
    let planned=plan.models.len()*64*3;let done=errors==0 && !exhausted && queries.len()==planned;
    let result=json!({"schema":"paisho-gen5-tape-search-result-v1","plan_sha256":sha256(&bytes),"engine_binary_sha256":engine,
        "source_plan":plan.source_plan,"tape_report":plan.tape_report,"tape_manifest":plan.tape_manifest,"relay_report":plan.relay_report,
        "contract":contract,"complete":done,"errors":errors,"kernel_limit_reached":exhausted,"planned_queries":planned,"completed_query_records":queries.len(),
        "models":plan.models.iter().map(|m|json!({"label":m.label,"input":m.input,"bits":m.parameter_bits_sha256})).collect::<Vec<_>>(),
        "unique_models":loaded.len(),"queries":queries,"graded":graded,"roots":root_metrics,"scores":scores,
        "fresh_queries":fresh_queries,"bank_loads":1,"shared_bank":true,"installed_reference_certificates":0,"new_games":0,"learning_updates":0,
        "input_seconds":input_seconds,"model_and_bank_load_seconds":load_seconds,"native_kernel_seconds":kernel,
        "coupled_kernel_seconds":coupling_seconds,"search_kernel_seconds":search_seconds,"external_grading_seconds":grading.elapsed().as_secs_f64(),
        "seconds_before_final_write":started.elapsed().as_secs_f64(),"counter_scope":"inference_evaluations = new state/value encodings, not all memory/policy forwards",
        "unknown_support_is_not_a_proven_loss":true,"cached_timing_not_added_to_this_run":true});
    write_json(&out.join("report.json"),&result)?;Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_exact_budget_or_root_proof_completes_a_cold_query() {
        assert!(complete(8,0,8,None));assert!(complete(2,0,8,Some(1)));assert!(complete(0,0,8,Some(0)));
        assert!(!complete(7,0,8,None));assert!(!complete(8,1,8,None));assert!(!complete(9,0,8,Some(1)));
    }
    #[test]
    fn nonfinite_coupling_cannot_be_reported_complete() {
        assert!(finite_coupled(&[0.,1.],&[-1000.,2.]).is_ok());
        assert!(finite_coupled(&[f64::NAN],&[0.]).is_err());
        assert!(finite_coupled(&[0.],&[f64::INFINITY]).is_err());
    }
    #[test]
    fn cache_key_changes_for_any_computational_coordinate() {
        let a=json!({"root":"a","model":"b","options":{},"budget":8,"engine":"c"});let key=query_key(&a).unwrap();
        for k in ["root","model","options","budget","engine"] {let mut b=a.clone();b[k]=json!("changed");assert_ne!(key,query_key(&b).unwrap());}
    }
    #[test]
    fn coupled_value_only_forward_matches_existing_eager_formula() {
        let record:GameRecord=include_str!(concat!(env!("CARGO_MANIFEST_DIR"),"/../paisho-ai/tests/fixtures/micro-alias-0-a.psr")).parse().unwrap();
        let position=cases::prefix(&record,1).replay().unwrap();let actions=legal_actions(&position);
        let model=MicroModel::seeded(217).with_neural_memory(91);
        let state=model.state_features(&position);let features=actions.iter().map(|&a|micro_action_features(&position,a)).collect::<Vec<_>>();
        let example=Arc::new(MicroExample{policy_support:false,state:state.clone(),actions:features.clone(),policy:vec![1./actions.len() as f64;actions.len()],
            action_values:vec![],sequence_source:0,value:0.,value_weight:0.,policy_weight:1.});
        let root=Root{key:String::new(),position,actions,fixture:Value::Null,example};
        let actual=coupled(&model,&root,16.).unwrap();
        let base=micro_softmax(&MicroModel::logits(&model.embed(&state),&features)).unwrap();
        let prior=model.memory_priors(&state,&features,&base,0).unwrap();let mut expected=vec![];
        for (&a,p) in root.actions.iter().zip(prior) {
            let mut n=root.position.clone();n.apply(a).unwrap();let player=root.position.to_move();
            let q=match n.outcome(){GameOutcome::Win(w)=>if w==player{1.}else{-1.},GameOutcome::Draw=>0.,
                GameOutcome::Ongoing=>model.embed(&model.state_features(&n)).value*if n.to_move()==player{1.}else{-1.}};
            expected.push(p.max(1e-300).ln()+16.*q);
        }
        let logits:Vec<f64>=serde_json::from_value(actual["coupled_logits"].clone()).unwrap();
        assert_eq!(logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert_eq!(actual["selected_index"],best(&expected));
    }
}
