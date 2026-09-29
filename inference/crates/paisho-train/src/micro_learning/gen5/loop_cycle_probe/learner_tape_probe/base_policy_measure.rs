//! Read only the actual interior policy on frozen tape examples. Never query a
//! sequence bank, the neural reader, auxiliary Q, successor values, or MCTS.
use super::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    tape_plan: Input,
    tape_report: Input,
    tape_manifest: Input,
    models: Vec<ModelInput>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelInput {
    label: String,
    input: Input,
    parameter_bits_sha256: String,
}
#[derive(Clone, Serialize)]
struct BaseRead {
    logits: Vec<f64>,
    priors: Vec<f64>,
    weighted_policy: f64,
    argmax: Option<usize>,
    support_mass: Option<f64>,
    support_correct: Option<bool>,
    support_margin: Option<f64>,
}
fn maximum(p: &[f64]) -> Option<usize> {
    (0..p.len()).max_by(|&a,&b|p[a].total_cmp(&p[b]).then_with(||b.cmp(&a)))
}
fn log_sum_exp(logits: &[f64]) -> f64 {
    let max=logits.iter().copied().fold(f64::NEG_INFINITY,f64::max);
    max+logits.iter().map(|x|(x-max).exp()).sum::<f64>().ln()
}
fn from_logits(ex: &MicroExample, logits: Vec<f64>) -> Result<BaseRead> {
    ex.validate().map_err(invalid)?;
    if logits.len()!=ex.actions.len() || logits.iter().any(|v|!v.is_finite()) {
        return Err(invalid("base logits/action mismatch"));
    }
    if logits.is_empty() {
        return Ok(BaseRead{logits,priors:vec![],weighted_policy:0.,argmax:None,
            support_mass:None,support_correct:None,support_margin:None});
    }
    let priors=micro_softmax(&logits).map_err(invalid)?;
    let argmax=maximum(&priors).unwrap();
    let support=ex.policy.iter().enumerate().filter(|(_,v)|**v>0.).map(|(i,_)|i).collect::<Vec<_>>();
    let log_z=log_sum_exp(&logits);
    let loss=if ex.policy_support && support.len()>1 {
        log_z-log_sum_exp(&support.iter().map(|&i|logits[i]).collect::<Vec<_>>())
    } else {ex.policy.iter().zip(&logits).map(|(t,l)|t*(log_z-l)).sum()};
    let (mass,correct,margin)=if ex.policy_support && ex.policy_weight>0. {
        let mass=support.iter().map(|&i|priors[i]).sum::<f64>();
        let inside=support.iter().map(|&i|logits[i]).fold(f64::NEG_INFINITY,f64::max);
        let outside=logits.iter().zip(&ex.policy).filter(|(_,p)|**p==0.).map(|(l,_)|*l).reduce(f64::max);
        (Some(mass),Some(ex.policy[argmax]>0.),outside.map(|v|inside-v))
    } else {(None,None,None)};
    if !loss.is_finite() || !mass.unwrap_or(0.).is_finite() {
        return Err(invalid("nonfinite base policy measurement"));
    }
    Ok(BaseRead{logits,priors,weighted_policy:ex.policy_weight*loss,argmax:Some(argmax),
        support_mass:mass,support_correct:correct,support_margin:margin})
}
fn read_base(model: &MicroModel, ex: &MicroExample) -> Result<BaseRead> {
    if ex.actions.is_empty() {return from_logits(ex,vec![]);}
    from_logits(ex,model.diagnostic_interior_policy_logits(&ex.state,&ex.actions))
}
fn number(value: &Value, key: &str) -> Result<f64> {
    value[key].as_f64().filter(|v|v.is_finite()).ok_or_else(||invalid(format!("missing finite {key}")))
}
fn same_descriptor(a: &Input,b: &Input)->bool {a.path==b.path && a.sha256==b.sha256}
fn same_snapshot_path(expected: &Path, requested: &Input) -> Result<bool> {
    let expected=fs::canonicalize(expected)?;
    if expected!=fs::canonicalize(&requested.path)? {return Ok(false);}
    // Equivalence of names never bypasses immutable artifact authentication.
    Ok(sha256(&fs::read(expected)?)==requested.sha256)
}
fn expected_labels()->Vec<String> {
    let mut labels=vec!["initial-actor".into()];
    for arm in ["recorded-reset","persistent"] {for block in 0..4 {for role in ["actor","working"] {
        labels.push(format!("{arm}-{block}-{role}"));
    }}}
    labels
}
fn aggregate(base: &[BaseRead], complete: &[Value], entries: &[Value]) -> Result<Value> {
    if base.len()!=complete.len() {return Err(invalid("base/complete row mismatch"));}
    let mut sum=0.;let mut base_loss=0.;let mut full_loss=0.;
    let mut support_weight=0.;let mut base_ok=0.;let mut full_ok=0.;let mut played_ok=0.;
    let mut base_mass=0.;let mut full_mass=0.;let mut played_mass=0.;
    let mut full_only=0.;let mut base_only=0.;
    for e in entries {
        let i=e["union_index"].as_u64().ok_or_else(||invalid("view index"))? as usize;
        let n=e["weight_numerator"].as_u64().ok_or_else(||invalid("view numerator"))?;
        let d=e["weight_denominator"].as_u64().filter(|d|*d>0).ok_or_else(||invalid("view denominator"))?;
        let w=n as f64/d as f64;
        if i>=base.len() || e["weight"].as_f64().map(f64::to_bits)!=Some(w.to_bits()) {
            return Err(invalid("view weights changed"));
        }
        let b=&base[i];let f=&complete[i];sum+=w;
        base_loss+=w*b.weighted_policy;full_loss+=w*number(f,"policy")?;
        match (b.support_correct,f["raw_in_support"].as_bool(),f["played_raw_in_support"].as_bool()) {
            (Some(bc),Some(fc),Some(pc))=>{
                support_weight+=w;base_ok+=w*f64::from(u8::from(bc));full_ok+=w*f64::from(u8::from(fc));
                played_ok+=w*f64::from(u8::from(pc));
                base_mass+=w*b.support_mass.unwrap();full_mass+=w*number(f,"support_mass")?;
                played_mass+=w*number(f,"played_raw_support_mass")?;
                if fc && !bc {full_only+=w;} if bc && !fc {base_only+=w;}
            },
            (None,None,None)=>{},
            _=>return Err(invalid("complete/base support eligibility differs")),
        }
    }
    if !(sum>0.) {return Err(invalid("empty weighted group"));}
    Ok(json!({"rows":entries.len(),"weight_sum":sum,"base_policy_weighted":base_loss,
        "complete_policy_weighted":full_loss,"support_weight":support_weight,
        "base_success_weight":base_ok,"complete_success_weight":full_ok,"played_complete_success_weight":played_ok,
        "base_support_mass_weighted":base_mass,"complete_support_mass_weighted":full_mass,
        "played_complete_support_mass_weighted":played_mass,"complete_only_success_weight":full_only,
        "base_only_success_weight":base_only,"policy_values_are_weighted_sums_not_group_conditional_means":true}))
}
pub fn run(plan_path: &Path,out: &Path)->Result<Value> {
    let started=Instant::now();let bytes=fs::read(plan_path)?;
    let plan:Plan=serde_json::from_slice(&bytes)?;
    if plan.schema!="paisho-gen5-tape-base-policy-plan-v1" || out.exists()
        || plan.models.iter().map(|m|m.label.clone()).collect::<Vec<_>>()!=expected_labels() {
        return Err(invalid("invalid fixed base policy plan/output"));
    }
    let tape_plan:TapePlan=serde_json::from_slice(&plan.tape_plan.bytes()?)?;
    let report=json(&plan.tape_report)?;let tape=json(&plan.tape_manifest)?;
    if report["schema"]!="paisho-gen5-learner-tape-result-v1" || tape["schema"]!="paisho-gen5-native-learning-tape-v1"
        || report["plan_sha256"]!=plan.tape_plan.sha256 || tape["plan_sha256"]!=plan.tape_plan.sha256
        || !same_descriptor(&input(&report["tape"])?,&plan.tape_manifest)
        || report["same_tape_all_arms"]!=true || report["reset_replay_every_sgd_and_boundary_exact"]!=true
        || report["first_block_common_sgd_exact"]!=true || tape_plan.blocks.len()!=4 {
        return Err(invalid("unverified or different tape/report"));
    }
    let proof_rows=tape["proof_cohort"].as_array().filter(|r|r.len()==64).ok_or_else(||invalid("expected same 64 frozen proofs"))?;
    let mut proofs=vec![];let mut keys=BTreeSet::new();
    for p in proof_rows {
        if !keys.insert(p["key"].as_str().ok_or_else(||invalid("proof key"))?.to_string()) {
            return Err(invalid("duplicate proof key"));
        }
        let e=restore(&serde_json::from_value(p["example"].clone())?)?;
        if !e.policy_support || e.policy_weight<=0. || e.actions.is_empty() {return Err(invalid("proof support omitted"));}
        proofs.push(e);
    }
    let mut cohorts=vec![];
    for (i,block) in tape_plan.blocks.iter().enumerate() {
        let manifest=json(&block.retention)?;
        let examples_input=input(&manifest["union_saved"])?;
        let saved:Vec<SavedMicroExample>=serde_json::from_slice(&examples_input.bytes()?)?;
        let examples=saved.iter().map(|s|s.example_for_rules_with_trusted_q(RULES,true)).collect::<Result<Vec<_>>>()?;
        if manifest["event"]!=i || manifest["rows"]!=examples.len() || manifest["heldout"]!=false
            || manifest["admission_criterion"]!=false || manifest["already_learned"]!=true
            || manifest["union"].as_array().map(Vec::len)!=Some(examples.len()) {
            return Err(invalid("retention population changed"));
        }
        for (j,e) in examples.iter().enumerate() {
            let state_bits=e.state.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>();
            if manifest["union"][j]["state_sha256"]!=sha256(&state_bits) {return Err(invalid("retention row state changed"));}
        }
        cohorts.push((manifest,examples,examples_input));
    }
    if cohorts.iter().map(|(_,e,_)|e.len()).sum::<usize>()!=999 {return Err(invalid("expected same 999 rows"));}
    fs::create_dir_all(out)?;fs::create_dir(out.join("unique-models"))?;
    let complete_all=report["measurements"].as_array().ok_or_else(||invalid("tape reads missing"))?;
    let mut cache:BTreeMap<String,(Input,Vec<Vec<BaseRead>>,Vec<BaseRead>)>=BTreeMap::new();
    let mut models=vec![];let mut bank_spec=None;let mut architecture=None;let mut kernel=0.;let mut calculated=0usize;
    let mut embeddings=0usize;let mut action_logits=0usize;
    for (mi,request) in plan.models.iter().enumerate() {
        let expected_path=if mi==0 {tape_plan.initial_actor.path.clone()} else {
            let parts=request.label.rsplitn(3,'-').collect::<Vec<_>>();
            let role=parts[0];let bi=parts[1].parse::<usize>()?;let arm=parts[2];
            let b=&report["arms"][arm]["boundaries"][bi];
            if b[format!("{role}_bits")]!=request.parameter_bits_sha256 {return Err(invalid("requested boundary bits differ"));}
            PathBuf::from(b[role].as_str().ok_or_else(||invalid("boundary path"))?)
        };
        if !same_snapshot_path(&expected_path,&request.input)?
            || (mi==0 && request.input.sha256!=tape_plan.initial_actor.sha256) {
            return Err(invalid("snapshot role/path changed"));
        }
        let artifact:MicroArtifact=serde_json::from_slice(&request.input.bytes()?)?;
        let this_bank=serde_json::to_value(&artifact.sequence_memory)?;
        if let Some(bank)=&bank_spec {if bank!=&this_bank {return Err(invalid("snapshot bank descriptor differs"));}}
        else {bank_spec=Some(this_bank);}
        let this_architecture=(artifact.schema.clone(),artifact.feature_schema.clone());
        if let Some(arch)=&architecture {if arch!=&this_architecture {return Err(invalid("snapshot architecture differs"));}}
        else {architecture=Some(this_architecture);}
        let coefficient_bits=artifact.parameters.iter().flat_map(|v|v.to_bits().to_le_bytes()).collect::<Vec<_>>();
        if sha256(&coefficient_bits)!=request.parameter_bits_sha256 {return Err(invalid("snapshot parameter bits changed"));}
        let matching=complete_all.iter().filter(|m|m["model"]==request.label).collect::<Vec<_>>();
        if matching.len()!=1 || matching[0]["bits"]!=request.parameter_bits_sha256 {return Err(invalid("complete reads not same model"));}
        let complete=matching[0];
        let reused=cache.contains_key(&request.parameter_bits_sha256);
        if !reused {
            let candidate=MicroModel::from_parameters(artifact.parameters).map_err(invalid)?;
            if candidate.schema()!=artifact.schema || candidate.feature_schema()!=artifact.feature_schema || !candidate.has_spatial() {
                return Err(invalid("snapshot architecture changed"));
            }
            let t=Instant::now();let mut rows=vec![];
            for (_,es,_) in &cohorts {rows.push(es.iter().map(|e|read_base(&candidate,e)).collect::<Result<Vec<_>>>()?);}
            let pr=proofs.iter().map(|e|read_base(&candidate,e)).collect::<Result<Vec<_>>>()?;
            kernel+=t.elapsed().as_secs_f64();calculated+=1063;
            for (_,es,_) in &cohorts {for e in es {embeddings+=usize::from(!e.actions.is_empty());action_logits+=e.actions.len();}}
            for e in &proofs {embeddings+=usize::from(!e.actions.is_empty());action_logits+=e.actions.len();}
            let f=write_json(&out.join("unique-models").join(format!("{}.json",request.parameter_bits_sha256)),
                &json!({"parameter_bits_sha256":request.parameter_bits_sha256,"retention":rows,"proofs":pr,
                "action_order":"exact frozen example action order; indices are not globally interchangeable"}))?;
            cache.insert(request.parameter_bits_sha256.clone(),(f,rows,pr));
        }
        let (source,rows,pr)=&cache[&request.parameter_bits_sha256];let mut panels=vec![];
        for (ci,(manifest,es,input)) in cohorts.iter().enumerate() {
            let old=&complete["retention"][ci];
            if old["block"]!=ci {return Err(invalid("complete retention block reordered"));}
            let full=old["rows"].as_array().filter(|r|r.len()==es.len()).ok_or_else(||invalid("complete retention rows"))?;
            let mut views=vec![];
            for view in manifest["views"].as_array().ok_or_else(||invalid("retention views"))? {
                let entries=view["entries"].as_array().ok_or_else(||invalid("view entries"))?;
                let a=aggregate(&rows[ci],full,entries)?;
                if (number(&a,"weight_sum")?-1.).abs()>1e-12 {return Err(invalid("view not normalized"));}
                let mut grouped=vec![];
                for field in ["source_sha256","lane","value_class","stratum"] {
                    let mut groups=BTreeMap::<String,Vec<Value>>::new();
                    for e in entries {let j=e["union_index"].as_u64().unwrap() as usize;
                        let metadata=&manifest["union"][j];
                        groups.entry(metadata[field].as_str().ok_or_else(||invalid("row group"))?.into()).or_default().push(e.clone());
                    }
                    let group=groups.iter().map(|(g,e)|Ok(json!({"group":g,"metrics":aggregate(&rows[ci],full,e)?}))).collect::<Result<Vec<_>>>()?;
                    grouped.push(json!({"field":field,"groups":group}));
                }
                views.push(json!({"name":view["name"],"metrics":a,"grouped":grouped}));
            }
            panels.push(json!({"block":ci,"examples":input,"manifest":tape_plan.blocks[ci].retention,
                "base_rows":source,"base_rows_index":ci,"complete_rows":full,"views":views}));
        }
        let old_proofs=complete["proofs"].as_array().filter(|p|p.len()==64).ok_or_else(||invalid("complete proof rows"))?;
        for (i,p) in old_proofs.iter().enumerate() {if p["key"]!=proof_rows[i]["key"] || p["first_block"]!=proof_rows[i]["first_block"] {
            return Err(invalid("complete proof selection reordered"));
        }}
        let full=old_proofs.iter().map(|p|p["metric"].clone()).collect::<Vec<_>>();
        let entries=(0..64).map(|i|json!({"union_index":i,"weight_numerator":1,"weight_denominator":64,"weight":1./64.})).collect::<Vec<_>>();
        let proof_aggregate=aggregate(pr,&full,&entries)?;
        models.push(json!({"label":request.label,"model":request.input,"bits":request.parameter_bits_sha256,
            "base_reads":source,"reused_base_reads":reused,"retention":panels,"proofs_complete":old_proofs,
            "proofs_aggregate":proof_aggregate}));
    }
    // Reauthenticate bound files after reads; no artifact is rewritten.
    plan.tape_plan.bytes()?;plan.tape_report.bytes()?;plan.tape_manifest.bytes()?;
    for b in &tape_plan.blocks {let m=json(&b.retention)?;input(&m["union_saved"])?.bytes()?;}
    for m in &plan.models {m.input.bytes()?;}
    let result=json!({"schema":"paisho-gen5-tape-base-policy-result-v1","plan_sha256":sha256(&bytes),
        "tape_plan":plan.tape_plan,"tape_report":plan.tape_report,"tape_manifest":plan.tape_manifest,
        "models":models,"retention_rows":999,"proof_rows":64,"unique_parameter_sets":cache.len(),
        "computed_base_rows":calculated,"computed_policy_embeddings":embeddings,"computed_action_logits":action_logits,
        "base_kernel_seconds":kernel,"seconds_before_final_write":started.elapsed().as_secs_f64(),
        "bank_descriptors_equal":true,"bank_loads":0,"neural_forwards":0,"auxiliary_q_reads":0,"new_searches":0,
        "scope":"same fixed consumed examples and proof cohort; base includes the existing residual16 policy head; source-aware full-policy reads reused from authenticated tape, source0 full CE unavailable; no strength conclusion"});
    write_json(&out.join("report.json"),&result)?;Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn example(support:bool)->MicroExample {MicroExample{policy_support:support,action_values:vec![],sequence_source:91,
        value_weight:0.,state:vec![0.;128],actions:vec![[0.;32];3],policy:vec![0.5,0.5,0.],value:0.,policy_weight:0.7}}
    #[test]
    fn snapshot_path_equivalence_always_requires_same_file_and_sha() {
        let dir=std::env::temp_dir().join(format!("gen5-base-path-{}-{}",std::process::id(),std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        fs::create_dir_all(dir.join("child")).unwrap();
        let path=dir.join("model.json");fs::write(&path,b"frozen model").unwrap();
        let source=Input{path:path.clone(),sha256:sha256(b"frozen model")};
        assert!(same_snapshot_path(&dir.join("child/../model.json"),&source).unwrap());
        let other=dir.join("other.json");fs::write(&other,b"frozen model").unwrap();
        assert!(!same_snapshot_path(&other,&source).unwrap());
        let bad=Input{path,sha256:sha256(b"different")};
        assert!(!same_snapshot_path(&source.path,&bad).unwrap());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn base_support_ignores_distribution_inside_verified_set() {
        let ex=example(true);let a=from_logits(&ex,vec![0.8f64.ln(),0.1f64.ln(),0.1f64.ln()]).unwrap();
        let b=from_logits(&ex,vec![0.1f64.ln(),0.8f64.ln(),0.1f64.ln()]).unwrap();
        assert!((a.weighted_policy-b.weighted_policy).abs()<1e-15);assert_eq!(a.support_correct,Some(true));
        assert!((a.weighted_policy+0.7*0.9f64.ln()).abs()<1e-15);
    }
    #[test]
    fn base_ce_singleton_and_ties_keep_original_convention() {
        let mut ex=example(false);ex.policy=vec![0.,1.,0.];
        let a=from_logits(&ex,vec![0.,0.,0.]).unwrap();ex.policy_support=true;
        let b=from_logits(&ex,vec![0.,0.,0.]).unwrap();assert_eq!(a.weighted_policy.to_bits(),b.weighted_policy.to_bits());
        assert_eq!(a.argmax,Some(0));assert_eq!(b.support_correct,Some(false));
    }
    #[test]
    fn base_large_logits_and_empty_value_rows_are_defined() {
        let mut ex=example(true);let r=from_logits(&ex,vec![1000.,999.,998.]).unwrap();assert!(r.weighted_policy.is_finite());
        ex.actions.clear();ex.policy.clear();ex.policy_weight=0.;let r=from_logits(&ex,vec![]).unwrap();
        assert_eq!(r.weighted_policy,0.);assert_eq!(r.argmax,None);
    }
    #[test]
    fn base_scalar_matches_native_loss_without_root_readers() {
        let model=MicroModel::seeded(412).with_residual_policy(73);
        let mut ex=example(false);ex.actions[0][0]=1.;ex.actions[1][1]=0.75;ex.actions[2][2]=-0.5;
        ex.state[3]=0.6;
        for support in [false,true] {
            ex.policy_support=support;
            let read=read_base(&model,&ex).unwrap();let native=model.loss_loop_v3(&ex).unwrap();
            assert_eq!(read.weighted_policy.to_bits(),(native.policy*ex.policy_weight).to_bits());
        }
    }
}
