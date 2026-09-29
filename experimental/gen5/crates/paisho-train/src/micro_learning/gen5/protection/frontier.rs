//! Reproduce one frozen consolidation transaction; only its numerical target varies.
use super::*;

#[derive(Clone, Deserialize, Serialize)]
struct Input { path: PathBuf, sha256: String }
impl Input {
    fn bytes(&self) -> Result<Vec<u8>> {
        let bytes = fs::read(&self.path)?;
        if sha256(&bytes) != self.sha256 { return Err(invalid(format!("changed frontier input {}", self.path.display()))); }
        Ok(bytes)
    }
}
#[derive(Deserialize)]
struct ProofInput {
    #[serde(flatten)]
    input: Input,
    original_panel_position: usize,
    fresh_fifo_index: usize,
    prefix_sha256: String,
}
#[derive(Deserialize)]
struct Manifest {
    anchor: Input,
    candidate: Input,
    fresh_saved_examples: Input,
    native_config: Input,
    original_report: Input,
    proof_positions: Vec<ProofInput>,
    fresh_count: usize,
}
#[derive(Deserialize)]
struct ProofRecord { rules: String, prefix: String, certificate: MicroProofCertificate }
struct Proof {
    id: usize,
    position: paisho_core::Position,
    actions: Vec<paisho_core::Action>,
    valid: Vec<bool>,
    example: MicroExample,
}

fn parameters_sha(model: &MicroModel) -> String {
    sha256(&model.parameters().iter().flat_map(|w| w.to_bits().to_le_bytes()).collect::<Vec<_>>())
}
fn bits_equal(a: &MicroModel, b: &MicroModel) -> bool {
    a.parameters().len() == b.parameters().len()
        && a.parameters().iter().zip(b.parameters()).all(|(a,b)| a.to_bits() == b.to_bits())
}
fn numbers_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a.as_array(), b.as_array()) {
        (Some(a),Some(b)) => a.len()==b.len() && a.iter().zip(b).all(|(a,b)| numbers_equal(a,b)),
        _ => a.as_f64().zip(b.as_f64()).is_some_and(|(a,b)| a.to_bits()==b.to_bits()),
    }
}

fn proof(input: &ProofInput, saved: &[SavedMicroExample], model: &MicroModel) -> Result<Proof> {
    let p: ProofRecord = serde_json::from_slice(&input.input.bytes()?)?;
    if p.rules != RULES.as_str() || sha256(p.prefix.as_bytes()) != input.prefix_sha256 {
        return Err(invalid("frontier proof identity/rules changed"));
    }
    let record: GameRecord = p.prefix.parse()?;
    let position = record.replay()?;
    p.certificate.verify(&position).map_err(invalid)?;
    let sign = if position.to_move() == Player::Host {1} else {-1};
    if position.outcome()!=GameOutcome::Ongoing || p.certificate.outcome * sign != 1 {
        return Err(invalid("frontier probe must be a nonterminal proved win"));
    }
    let actions = paisho_core::legal_actions(&position);
    let mut valid = Vec::with_capacity(actions.len());
    for action in &actions {
        let certificate = p.certificate.children.iter()
            .any(|(a,c)| a==&action.to_string() && c.outcome==p.certificate.outcome);
        let mut next = position.clone(); next.apply(*action)?;
        valid.push(certificate || next.outcome()==GameOutcome::Win(position.to_move()));
    }
    let row = saved.get(input.fresh_fifo_index).ok_or_else(||invalid("frontier proof missing from fresh FIFO"))?;
    let state = model.state_features(&position);
    let action_features: Vec<_> = actions.iter().map(|a| micro_action_features(&position,*a)).collect();
    if row.state.len()!=state.len() || row.state.iter().zip(&state).any(|(a,b)| a.to_bits()!=b.to_bits())
        || row.actions != actions.iter().map(ToString::to_string).collect::<Vec<_>>()
        || row.action_features.iter().map(Vec::as_slice).ne(action_features.iter().map(|a| a.as_slice()))
    { return Err(invalid("frontier proof and recorded fresh state/actions differ")); }
    let n = valid.iter().filter(|v| **v).count();
    if n==0 {return Err(invalid("frontier proof has no verified winning support"));}
    let example = MicroExample { structured: Vec::new(), state, actions:action_features,
        policy:valid.iter().map(|v| if *v {1./n as f64} else {0.}).collect(),
        value:1., policy_weight:1., value_weight:0., sequence_source:0,
        action_values:vec![], policy_support:false };
    Ok(Proof {id:input.original_panel_position,position,actions,valid,example})
}
fn proof_choices(model: &MicroModel, proofs: &[Proof], beta: f64) -> Result<serde_json::Value> {
    let mut rows = vec![];
    for proof in proofs {
        let raw = prior(model,&proof.example)?;
        let mut coupled:Vec<_> = raw.iter().map(|p|p.max(1e-300).ln()).collect();
        for (i,action) in proof.actions.iter().enumerate() {
            let mut next=proof.position.clone(); next.apply(*action)?;
            let value=match next.outcome() {
                GameOutcome::Win(p) => if p==proof.position.to_move() {1.} else {-1.},
                GameOutcome::Draw => 0.,
                GameOutcome::Ongoing => {
                    let v=model.value(&model.state_features(&next));
                    if next.to_move()==proof.position.to_move() {v} else {-v}
                }
            };
            coupled[i] += beta * value;
        }
        let a=best(&raw);let b=best(&coupled);
        rows.push(serde_json::json!({"position":proof.id,"raw_verified_win":proof.valid[a],
            "coupled_verified_win":proof.valid[b],"raw_action":proof.actions[a].to_string(),
            "coupled_action":proof.actions[b].to_string(),"raw_probability":raw[a],
            "verified_policy_mass":raw.iter().zip(&proof.valid).filter(|(_,v)|**v).map(|(p,_)|p).sum::<f64>(),
            "winning_support_actions":proof.valid.iter().filter(|v|**v).count(),
            "value":model.value(&proof.example.state),"sequence_source":0}));
    }
    Ok(serde_json::Value::Array(rows))
}

/// MANIFEST NEW_OUTPUT. No search, SGD, replay admission, publication or campaign.
/// A fresh output folder prevents writes to any frozen model used as input.
pub fn run(manifest_path: &Path, out: &Path) -> Result<serde_json::Value> {
    run_mode(manifest_path,out,false)
}
/// Same frontier inputs, baseline/+validation-value ABBA, then the real Guard.
/// Guard artifacts stay in NEW_OUTPUT; no diagnostic artifact is resume-ready.
pub fn run_validation_probe(manifest_path: &Path, out: &Path) -> Result<serde_json::Value> {
    run_mode(manifest_path,out,true)
}
fn run_mode(manifest_path: &Path, out: &Path, validation_probe: bool) -> Result<serde_json::Value> {
    if out.exists() {return Err(invalid("frontier output must not already exist"));}
    let loading=Instant::now();
    let manifest_bytes=fs::read(manifest_path)?;
    let m:Manifest=serde_json::from_slice(&manifest_bytes)?;
    if m.fresh_count!=64 || m.proof_positions.len()!=2
        || m.proof_positions.iter().map(|p|p.original_panel_position).collect::<Vec<_>>() != [18,25]
    {return Err(invalid("frontier fixture must contain the exact last64 and probes18/25"));}
    let a:MicroArtifact=serde_json::from_slice(&m.anchor.bytes()?)?;
    let anchor=a.model()?;
    let b:MicroArtifact=serde_json::from_slice(&m.candidate.bytes()?)?;
    if serde_json::to_value(&a.sequence_memory)? != serde_json::to_value(&b.sequence_memory)? {
        return Err(invalid("frontier anchor/candidate sequence memory dependencies differ"));
    }
    let mut candidate=MicroModel::from_parameters(b.parameters.clone()).map_err(invalid)?;
    if candidate.schema()!=b.schema || candidate.feature_schema()!=b.feature_schema {
        return Err(invalid("frontier candidate model schema changed"));
    }
    if let Some(bank)=anchor.sequence_memory() {candidate=candidate.with_sequence_memory_owned(bank.clone());}
    let saved:Vec<SavedMicroExample>=serde_json::from_slice(&m.fresh_saved_examples.bytes()?)?;
    if saved.len()!=64 {return Err(invalid("frontier fresh FIFO length changed"));}
    let fresh=saved.iter().map(|e|e.example_for_rules_with_trusted_q(RULES,true).map(Arc::new)).collect::<Result<Vec<_>>>()?;
    let o:Options=serde_json::from_slice(&m.native_config.bytes()?)?;
    let original:serde_json::Value=serde_json::from_slice(&m.original_report.bytes()?)?;
    let cycle=original["cycles"].as_array().and_then(|c|c.iter().find(|c|c["cycle"].as_u64()==Some(12)))
        .ok_or_else(||invalid("frontier original cycle12 is missing"))?;
    let guard_path=o.publication_guard.as_ref().ok_or_else(||invalid("frontier missing guard"))?;
    let validation_path=o.publication_validation.as_ref().ok_or_else(||invalid("frontier missing validation"))?;
    let panel_inputs=[Input {path:guard_path.clone(),sha256:cycle["publication"]["manifest"].as_str().ok_or_else(||invalid("missing original guard hash"))?.into()},
        Input {path:validation_path.clone(),sha256:cycle["publication"]["validation_manifest"].as_str().ok_or_else(||invalid("missing original validation hash"))?.into()}];
    for p in &panel_inputs {p.bytes()?;}
    let proofs=m.proof_positions.iter().map(|p|proof(p,&saved,&anchor)).collect::<Result<Vec<_>>>()?;
    fs::create_dir(out)?;
    let initial=Arc::new(Snapshot {identity:a.identity(),artifact:Some(Arc::new(a.clone())),
        model:Arc::new(anchor.clone()),version:11,path:m.anchor.path.clone()});
    let mut guard=publication::Guard::open(guard_path,&out.join("guard"),o.value_policy_strength,initial.clone(),&serde_json::Value::Null)?;
    guard.enable_v2(validation_path)?;guard.enable_v3()?;
    let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(10).build()?);
    guard.enable_parallel(&[pool.clone()]);
    let references=guard.reference_examples();
    if references.len()!=57 {return Err(invalid("frontier protection panel length changed"));}
    let validation_rows = if validation_probe {guard.diagnostic_validation_examples()?} else {vec![]};
    if validation_probe && validation_rows.len()!=339 {return Err(invalid("frontier validation panel length changed"));}
    let initial_scores=guard.diagnostic_measure(&anchor)?;
    let candidate_scores=guard.diagnostic_measure(&candidate)?;
    let initial_choices=proof_choices(&anchor,&proofs,o.value_policy_strength)?;
    let candidate_choices=proof_choices(&candidate,&proofs,o.value_policy_strength)?;
    let loading_seconds=loading.elapsed().as_secs_f64();
    let mut rows=vec![];
    let mut previous:Vec<(f64,bool,Vec<u64>,Vec<u64>,serde_json::Value)>=vec![];
    let mut guard_previous:Vec<(bool,Vec<u64>)>=vec![];
    let variants=if validation_probe {[(0.5,false),(0.5,true),(0.5,true),(0.5,false)]}
        else {[(1.,false),(0.5,false),(0.5,false),(1.,false)]};
    for (index,(fraction,extra_value)) in variants.into_iter().enumerate() {
        let mut protection=Protection::new(&anchor,references.clone())?;
        protection.enable_loop_v3();protection.enable_parallel(&[pool.clone()]);protection.observe(&fresh);
        let extra_setup=Instant::now();
        if extra_value {protection.diagnostic_enable_validation_value(validation_rows.clone())?;}
        let extra_setup_seconds=if extra_value {extra_setup.elapsed().as_secs_f64()} else {0.};
        if extra_value {
            let panel=protection.validation_value.as_ref().unwrap();
            for (model,score) in [(&anchor,&initial_scores),(&candidate,&candidate_scores)] {
                if panel.loss(model,protection.parallel.as_ref())?.to_bits()
                    !=score["validation"]["value_mse"].as_f64().ok_or_else(||invalid("missing Guard MSE"))?.to_bits()
                {return Err(invalid("diagnostic value objective differs from actual Guard MSE"));}
            }
        }
        protection.updates=cycle["protection"]["updates"].as_u64().unwrap_or(0) as usize;
        let updates=protection.updates;
        let mut applied=candidate.clone();let mut captured=None;
        let started=Instant::now();
        protection.consolidate_target(&mut applied,fraction,|m|captured=Some(m.clone()))?;
        let seconds=started.elapsed().as_secs_f64();
        let attempted=captured.ok_or_else(||invalid("frontier attempted candidate not captured"))?;
        // Deliberately outside consolidation timing, measured even after refusal.
        let fresh_attempted=fresh_loss_mode(&attempted,&fresh,protection.parallel.as_ref(),true)?;
        let fresh_applied=fresh_loss_mode(&applied,&fresh,protection.parallel.as_ref(),true)?;
        let after_choices=proof_choices(&attempted,&proofs,o.value_policy_strength)?;
        let applied_choices=proof_choices(&applied,&proofs,o.value_policy_strength)?;
        let attempted_scores=guard.diagnostic_measure(&attempted)?;
        let applied_scores=guard.diagnostic_measure(&applied)?;
        if extra_value {
            let panel=protection.validation_value.as_ref().unwrap();
            for (model,score) in [(&attempted,&attempted_scores),(&applied,&applied_scores)] {
                if panel.loss(model,protection.parallel.as_ref())?.to_bits()
                    !=score["validation"]["value_mse"].as_f64().ok_or_else(||invalid("missing final Guard MSE"))?.to_bits()
                {return Err(invalid("final diagnostic objective differs from actual Guard MSE"));}
            }
        }
        if protection.updates!=updates || protection.fresh.len()!=64 {return Err(invalid("frontier consolidation changed presentations/fresh FIFO"));}
        let accepted=protection.last["accepted"].as_bool()==Some(true);
        if !accepted && (!bits_equal(&applied,&anchor)||!bits_equal(&protection.anchor,&anchor)) {
            return Err(invalid("frontier failed transaction did not restore exact anchor weights"));
        }
        let attempted_bits=attempted.parameters().iter().map(|w|w.to_bits()).collect::<Vec<_>>();
        let applied_bits=applied.parameters().iter().map(|w|w.to_bits()).collect::<Vec<_>>();
        if let Some((_,_,a,b,last))=previous.iter().find(|(f,e,_,_,_)|*f==fraction && *e==extra_value) {
            if a!=&attempted_bits || b!=&applied_bits || last!=&protection.last {return Err(invalid("frontier repeated variant differs"));}
        } else {previous.push((fraction,extra_value,attempted_bits,applied_bits,protection.last.clone()));}
        let attempted_path=out.join(format!("attempted-{index:02}.json"));
        let applied_path=out.join(format!("applied-{index:02}.json"));
        for (model,path,label) in [(&attempted,&attempted_path,"private-attempt"),(&applied,&applied_path,"applied-after-consolidation")] {
            MicroArtifact::new(model,b.updates,serde_json::json!({"diagnostic_only":true,"label":label,"target_tolerance_fraction":fraction})).save(path)?;
        }
        let publication = if validation_probe {
            let guard_setup=Instant::now();
            let mut final_guard=publication::Guard::open(guard_path,&out.join(format!("publication-{index:02}")),
                o.value_policy_strength,initial.clone(),&serde_json::Value::Null)?;
            final_guard.enable_v2(validation_path)?;final_guard.enable_v3()?;
            final_guard.enable_parallel(&[pool.clone()]);
            let guard_setup_seconds=guard_setup.elapsed().as_secs_f64();
            let artifact=Arc::new(MicroArtifact::new(&applied,b.updates,
                serde_json::json!({"diagnostic_only":true,"validation_value":extra_value})));
            let snapshot=Arc::new(Snapshot {identity:artifact.identity(),artifact:Some(artifact),
                model:Arc::new(applied.clone()),version:12,path:applied_path.clone()});
            let t=Instant::now();let focus=final_guard.consider(snapshot,true)?;
            let seconds=t.elapsed().as_secs_f64();
            let final_snapshot=final_guard.accepted();let final_model=&final_snapshot.model;
            let final_scores=guard.diagnostic_measure(final_model)?;
            let final_choices=proof_choices(final_model,&proofs,o.value_policy_strength)?;
            let final_fresh=fresh_loss_mode(final_model,&fresh,protection.parallel.as_ref(),true)?;
            let final_bits=final_model.parameters().iter().map(|w|w.to_bits()).collect::<Vec<_>>();
            if let Some((_,bits))=guard_previous.iter().find(|(extra,_)|*extra==extra_value) {
                if bits!=&final_bits {return Err(invalid("repeated final Guard model differs"));}
            } else {guard_previous.push((extra_value,final_bits));}
            let preserved=guard_preserved_choices(&initial_scores,&final_scores);
            if !preserved {return Err(invalid("real Guard accepted a lost old raw/coupled choice"));}
            let value_retained=["primary","validation"].into_iter().all(|panel| {
                initial_scores[panel]["value_mse"].as_f64().zip(final_scores[panel]["value_mse"].as_f64())
                    .is_some_and(|(a,b)|b.is_finite() && b<=a+FINITE_LOSS_TOLERANCE)
            });
            if !value_retained {return Err(invalid("real Guard accepted an increased balanced value MSE"));}
            let final_path=out.join(format!("after-publication-{index:02}.json"));
            MicroArtifact::new(final_model,b.updates,serde_json::json!({"diagnostic_only":true,
                "label":"actual-Guard-output","validation_value":extra_value})).save(&final_path)?;
            serde_json::json!({"seconds":seconds,"setup_seconds_excluded":guard_setup_seconds,
                "progress":final_guard.progress(),"scores":final_scores,"new_proof_choices":final_choices,
                "fresh_measured":final_fresh,"model":final_path,"parameter_sha256":parameters_sha(final_model),
                "changed_from_anchor":!bits_equal(final_model,&anchor),"all_old_raw_and_coupled_choices_retained":preserved,
                "corrective_rows_returned_not_learned":focus.as_ref().map_or(0,Vec::len),
                "criteria_unmodified":true,"balanced_value_mse_retained_on_both_panels":value_retained,
                "production_publication":false})
        } else {serde_json::Value::Null};
        let old=&cycle["protection"]["last"];
        rows.push(serde_json::json!({"target_tolerance_fraction":fraction,"diagnostic_validation_value":extra_value,
            "extra_anchor_gradient_setup_seconds":extra_setup_seconds,"publication":publication,
            "seconds":seconds,"progress":protection.progress(),
            "fresh_attempted_measured":fresh_attempted,"fresh_applied_measured":fresh_applied,"fresh_after_always_measured_outside_timing":true,
            "attempted_choices":after_choices,"applied_choices":applied_choices,
            "attempted_scores":attempted_scores,"applied_scores":applied_scores,
            "attempted_model":attempted_path,"applied_model":applied_path,
            "attempted_parameter_sha256":parameters_sha(&attempted),"applied_parameter_sha256":parameters_sha(&applied),
            "input_fresh_losses_match_original_bits":numbers_equal(&protection.last["fresh_before"],&old["fresh_before"]) && numbers_equal(&protection.last["fresh_anchor"],&old["fresh_anchor"]),
            "original_reference_before_matches_bits":numbers_equal(&protection.last["reference_before"],&old["reference_before"]),
            "original_reference_after_matches_bits":numbers_equal(&protection.last["reference_after"],&old["reference_after"]),
            "original_decision_matches":protection.last["accepted"]==old["accepted"],
            "presentations_and_fresh_fifo_preserved":true,"failed_rollback_exact":!accepted}));
    }
    for input in [&m.anchor,&m.candidate,&m.fresh_saved_examples,&m.native_config,&m.original_report].into_iter().chain(panel_inputs.iter()) {input.bytes()?;}
    for p in &m.proof_positions {p.input.bytes()?;}
    if fs::read(manifest_path)?!=manifest_bytes {return Err(invalid("frontier manifest changed during trial"));}
    let result=serde_json::json!({"schema":"paisho-gen5-consolidation-frontier-v1","manifest":manifest_path,
        "manifest_sha256":sha256(&manifest_bytes),"anchor":m.anchor,"candidate":m.candidate,"panels":panel_inputs,
        "fresh_saved_examples":m.fresh_saved_examples,"fresh_examples":fresh.len(),"protection_references":references.len(),
        "original_cycle":cycle["protection"]["last"],"anchor_scores":initial_scores,"candidate_scores":candidate_scores,
        "anchor_new_proof_choices":initial_choices,"candidate_new_proof_choices":candidate_choices,
        "runs":rows,"variant_order":if validation_probe {"baseline,validation-value,validation-value,baseline"} else {"outer,interior,interior,outer"},
        "diagnostic_validation_value_probe":validation_probe,"repeated_parameters_and_decisions_exact":true,
        "loading_and_initial_scores_seconds_excluded":loading_seconds,"input_hashes_unchanged":true,
        "same_native_v3_conversion_and_criteria":true,"acceptance_tolerance":FINITE_LOSS_TOLERANCE,
        "scope":if validation_probe {"frozen cycle12 diagnostic: extra aggregate value constraint then actual Guard only in private output; no SGD, search, replay admission, production publication or campaign; measure final fresh loss and new proof choices after Guard"} else {"frozen cycle12 consolidation only; no SGD, search, replay admission, publication or campaign; fresh final attempted/applied losses actually measured even on refusal"}});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}

fn guard_preserved_choices(before: &serde_json::Value, after: &serde_json::Value) -> bool {
    ["primary","validation"].into_iter().all(|panel|["raw","coupled"].into_iter().all(|kind| {
        match (before[panel][kind].as_array(),after[panel][kind].as_array()) {
            (Some(a),Some(b)) => a.len()==b.len() && a.iter().zip(b).all(|(a,b)|a.as_bool()==Some(false) || b.as_bool()==Some(true)),
            _ => false,
        }
    }))
}

/// Read-only policy-axis diagnosis of the accepted-value339 consolidation candidate.
/// FRONTIER_MANIFEST VALIDATION339_REPORT NEW_OUTPUT. Never repairs or publishes.
pub fn run_kl_probe(manifest_path:&Path, report_path:&Path, out:&Path)->Result<serde_json::Value> {
    if out.exists() {return Err(invalid("KL probe output must be new"));}
    let loading=Instant::now();let manifest_bytes=fs::read(manifest_path)?;
    let m:Manifest=serde_json::from_slice(&manifest_bytes)?;
    if m.fresh_count!=64 || m.proof_positions.len()!=2
        || m.proof_positions.iter().map(|p|p.original_panel_position).collect::<Vec<_>>() != [18,25] {
        return Err(invalid("KL probe requires exact frontier64 and proofs18/25"));
    }
    let report_bytes=fs::read(report_path)?;let report:serde_json::Value=serde_json::from_slice(&report_bytes)?;
    if report["manifest_sha256"].as_str()!=Some(sha256(&manifest_bytes).as_str()) {
        return Err(invalid("KL probe and validation339 report have different frontier inputs"));
    }
    let run=&report["runs"][1];
    if run["diagnostic_validation_value"]!=true || run["progress"]["last"]["accepted"]!=true {
        return Err(invalid("KL probe requires the first accepted validation339 leg"));
    }
    let candidate_path=PathBuf::from(run["applied_model"].as_str().ok_or_else(||invalid("missing validation339 applied model"))?);
    let candidate_bytes=fs::read(&candidate_path)?;
    let artifact:MicroArtifact=serde_json::from_slice(&candidate_bytes)?;
    let a:MicroArtifact=serde_json::from_slice(&m.anchor.bytes()?)?;let anchor=a.model()?;
    if serde_json::to_value(&a.sequence_memory)?!=serde_json::to_value(&artifact.sequence_memory)? {
        return Err(invalid("KL probe bank dependencies differ"));
    }
    let mut candidate=MicroModel::from_parameters(artifact.parameters.clone()).map_err(invalid)?;
    if candidate.schema()!=artifact.schema || candidate.feature_schema()!=artifact.feature_schema {
        return Err(invalid("KL probe candidate schema differs"));
    }
    if let Some(bank)=anchor.sequence_memory() {candidate=candidate.with_sequence_memory_owned(bank.clone());}
    if run["applied_parameter_sha256"].as_str()!=Some(parameters_sha(&candidate).as_str()) {
        return Err(invalid("KL probe candidate parameters differ from native339 report"));
    }
    let saved:Vec<SavedMicroExample>=serde_json::from_slice(&m.fresh_saved_examples.bytes()?)?;
    if saved.len()!=64 {return Err(invalid("KL probe fresh FIFO length differs"));}
    let fresh=saved.iter().map(|e|e.example_for_rules_with_trusted_q(RULES,true).map(Arc::new)).collect::<Result<Vec<_>>>()?;
    let proofs=m.proof_positions.iter().map(|p|proof(p,&saved,&anchor)).collect::<Result<Vec<_>>>()?;
    let o:Options=serde_json::from_slice(&m.native_config.bytes()?)?;
    let panel_inputs:Vec<Input>=serde_json::from_value(report["panels"].clone())?;
    if panel_inputs.len()!=2 || o.publication_guard.as_ref()!=Some(&panel_inputs[0].path)
        || o.publication_validation.as_ref()!=Some(&panel_inputs[1].path) {
        return Err(invalid("KL probe publication panels differ"));
    }
    for p in &panel_inputs {p.bytes()?;}
    fs::create_dir(out)?;
    let snapshot=Arc::new(Snapshot {identity:a.identity(),artifact:Some(Arc::new(a)),
        model:Arc::new(anchor.clone()),version:11,path:m.anchor.path.clone()});
    let mut guard=publication::Guard::open(&panel_inputs[0].path,&out.join("guard"),o.value_policy_strength,snapshot,&serde_json::Value::Null)?;
    guard.enable_v2(&panel_inputs[1].path)?;guard.enable_v3()?;
    let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(10).build()?);
    guard.enable_parallel(&[pool.clone()]);let parallel=cpu::Ordered::new(&[pool]);
    let anchor_choices=proof_choices(&anchor,&proofs,o.value_policy_strength)?;
    let anchor_fresh=fresh_loss_mode(&anchor,&fresh,Some(&parallel),true)?;
    let loading_seconds=loading.elapsed().as_secs_f64();let mut rows=vec![];
    let mut value_mses:Option<Vec<u64>>=None;
    for fraction in [1.,0.5,0.25,0.125] {
        let model=guard.diagnostic_policy_axis_model(&candidate,fraction)?;
        let decomposition=guard.diagnostic_kl_decomposition(&model)?;
        let mse=decomposition["panels"].as_array().unwrap().iter()
            .map(|p|p["value_mse_after"].as_f64().unwrap().to_bits()).collect::<Vec<_>>();
        if let Some(expected)=&value_mses {if expected!=&mse {return Err(invalid("fixed-value KL axis changed panel MSE bits"));}}
        else {value_mses=Some(mse);}
        let choices=proof_choices(&model,&proofs,o.value_policy_strength)?;
        let fresh=fresh_loss_mode(&model,&fresh,Some(&parallel),true)?;
        rows.push(serde_json::json!({"policy_fraction":fraction,"value_fraction":1.,"value_mse_bits_identical":true,
            "parameter_sha256":parameters_sha(&model),"decomposition":decomposition,
            "new_proof_choices":choices,"fresh_loss_measured":fresh}));
    }
    for p in [&m.anchor,&m.candidate,&m.fresh_saved_examples,&m.native_config,&m.original_report].into_iter().chain(panel_inputs.iter()) {p.bytes()?;}
    for p in &m.proof_positions {p.input.bytes()?;}
    if fs::read(&candidate_path)?!=candidate_bytes || fs::read(manifest_path)?!=manifest_bytes || fs::read(report_path)?!=report_bytes {
        return Err(invalid("KL probe inputs changed during measurement"));
    }
    let result=serde_json::json!({"schema":"paisho-gen5-policy-kl-decomposition-v1",
        "frontier_manifest":manifest_path,"frontier_sha256":sha256(&manifest_bytes),
        "validation339_report":report_path,"validation339_report_sha256":sha256(&report_bytes),
        "candidate":candidate_path,"candidate_file_sha256":sha256(&candidate_bytes),"anchor":m.anchor,"panels":panel_inputs,
        "anchor_new_choices":anchor_choices,"anchor_fresh":anchor_fresh,"loading_seconds_excluded":loading_seconds,
        "runs":rows,"all_input_hashes_unchanged":true,"learning_or_repair":false,"production_publication":false,
        "scope":"known winning roots; Bernoulli support/rest plus old-mass-weighted conditional terms; clamp flags distinguish probability KL from clamped-log arithmetic; no criterion is changed"});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}

/// Bounded private policy adjustment after the original Guard admitted P1/8.
pub fn run_gain_probe(manifest_path:&Path, report_path:&Path, out:&Path)->Result<serde_json::Value> {
    if out.exists() {return Err(invalid("Fresh gain probe output must be new"));}
    let loading=Instant::now();let manifest_bytes=fs::read(manifest_path)?;
    let m:Manifest=serde_json::from_slice(&manifest_bytes)?;
    if m.fresh_count!=64 || m.proof_positions.len()!=2
        || m.proof_positions.iter().map(|p|p.original_panel_position).collect::<Vec<_>>() != [18,25] {
        return Err(invalid("Fresh gain probe requires exact frontier64 and proofs18/25"));
    }
    let report_bytes=fs::read(report_path)?;let report:serde_json::Value=serde_json::from_slice(&report_bytes)?;
    if report["manifest_sha256"].as_str()!=Some(sha256(&manifest_bytes).as_str()) {
        return Err(invalid("Fresh gain probe and validation339 report have different frontier inputs"));
    }
    let run=&report["runs"][1];
    if run["diagnostic_validation_value"]!=true || run["progress"]["last"]["accepted"]!=true {
        return Err(invalid("Fresh gain probe requires the first accepted validation339 leg"));
    }
    let candidate_path=PathBuf::from(run["publication"]["model"].as_str().ok_or_else(||invalid("missing validation339 applied model"))?);
    let candidate_bytes=fs::read(&candidate_path)?;
    let artifact:MicroArtifact=serde_json::from_slice(&candidate_bytes)?;
    let a:MicroArtifact=serde_json::from_slice(&m.anchor.bytes()?)?;let anchor=a.model()?;
    if serde_json::to_value(&a.sequence_memory)?!=serde_json::to_value(&artifact.sequence_memory)? {
        return Err(invalid("Fresh gain probe bank dependencies differ"));
    }
    let mut candidate=MicroModel::from_parameters(artifact.parameters.clone()).map_err(invalid)?;
    if candidate.schema()!=artifact.schema || candidate.feature_schema()!=artifact.feature_schema {
        return Err(invalid("Fresh gain probe candidate schema differs"));
    }
    if let Some(bank)=anchor.sequence_memory() {candidate=candidate.with_sequence_memory_owned(bank.clone());}
    if run["publication"]["parameter_sha256"].as_str()!=Some(parameters_sha(&candidate).as_str()) {
        return Err(invalid("Fresh gain probe candidate parameters differ from native339 report"));
    }
    if run["publication"]["progress"]["last_decision"]!="accepted-transaction"
        || run["publication"]["progress"]["repair"]["value_fraction"].as_f64()!=Some(1.)
        || run["publication"]["progress"]["repair"]["policy_fraction"].as_f64()!=Some(0.125) {
        return Err(invalid("fresh gain start was not actually admitted by the Guard"));
    }
    let saved:Vec<SavedMicroExample>=serde_json::from_slice(&m.fresh_saved_examples.bytes()?)?;
    if saved.len()!=64 {return Err(invalid("Fresh gain probe fresh FIFO length differs"));}
    let fresh=saved.iter().map(|e|e.example_for_rules_with_trusted_q(RULES,true).map(Arc::new)).collect::<Result<Vec<_>>>()?;
    let proofs=m.proof_positions.iter().map(|p|proof(p,&saved,&anchor)).collect::<Result<Vec<_>>>()?;
    let o:Options=serde_json::from_slice(&m.native_config.bytes()?)?;
    let panel_inputs:Vec<Input>=serde_json::from_value(report["panels"].clone())?;
    if panel_inputs.len()!=2 || o.publication_guard.as_ref()!=Some(&panel_inputs[0].path)
        || o.publication_validation.as_ref()!=Some(&panel_inputs[1].path) {
        return Err(invalid("Fresh gain probe publication panels differ"));
    }
    for p in &panel_inputs {p.bytes()?;}
    fs::create_dir(out)?;
    let snapshot=Arc::new(Snapshot {identity:a.identity(),artifact:Some(Arc::new(a)),
        model:Arc::new(anchor.clone()),version:11,path:m.anchor.path.clone()});
    let mut guard=publication::Guard::open(&panel_inputs[0].path,&out.join("guard"),o.value_policy_strength,snapshot,&serde_json::Value::Null)?;
    guard.enable_v2(&panel_inputs[1].path)?;guard.enable_v3()?;
    let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(10).build()?);
    guard.enable_parallel(&[pool.clone()]);let parallel=cpu::Ordered::new(&[pool]);
    let anchor_choices=proof_choices(&anchor,&proofs,o.value_policy_strength)?;
    let anchor_fresh=fresh_loss_mode(&anchor,&fresh,Some(&parallel),true)?;
    // Validate that these two proofs were actually learned by the pre-consolidation
    // shadow, not selected after inspecting a result of this diagnostic.
    let pre_artifact:MicroArtifact=serde_json::from_slice(&m.candidate.bytes()?)?;
    let mut pre_model=MicroModel::from_parameters(pre_artifact.parameters).map_err(invalid)?;
    if let Some(bank)=anchor.sequence_memory() {pre_model=pre_model.with_sequence_memory_owned(bank.clone());}
    let pre_choices=proof_choices(&pre_model,&proofs,o.value_policy_strength)?;
    for (old,new) in anchor_choices.as_array().unwrap().iter().zip(pre_choices.as_array().unwrap()) {
        if old["position"]!=new["position"] || old["raw_verified_win"]!=false || new["raw_verified_win"]!=true {
            return Err(invalid("fresh gain proofs were not new raw gains before consolidation"));
        }
    }
    let mut targets=vec![];
    for proof in &proofs {
        let mut ex=proof.example.clone();ex.policy_support=true;ex.value_weight=0.;ex.action_values.clear();
        let mut offsets=vec![];
        for action in &proof.actions {
            let mut next=proof.position.clone();next.apply(*action)?;
            let value=match next.outcome() {
                GameOutcome::Win(p)=>if p==proof.position.to_move(){1.}else{-1.},
                GameOutcome::Draw=>0.,GameOutcome::Ongoing=>{
                    let v=candidate.value(&candidate.state_features(&next));
                    if next.to_move()==proof.position.to_move(){v}else{-v}
                }
            };
            offsets.push(o.value_policy_strength*value);
        }
        targets.push(publication::GainTarget {id:proof.id,example:Arc::new(ex),coupled_offsets:offsets});
    }
    let start_choices=proof_choices(&candidate,&proofs,o.value_policy_strength)?;
    let start_fresh=fresh_loss_mode(&candidate,&fresh,Some(&parallel),true)?;
    let loading_seconds=loading.elapsed().as_secs_f64();let mut rows=vec![];
    let mut repeated:Vec<(bool,Vec<u64>)>=vec![];
    for (index,projection) in [false,true,true,false].into_iter().enumerate() {
        let (trajectory,probe)=guard.diagnostic_fresh_gain_probe(&candidate,&targets,projection)?;
        let mut states=vec![];
        for (step,model) in trajectory.iter().enumerate() {
            let choices=proof_choices(model,&proofs,o.value_policy_strength)?;
            let fresh_loss=fresh_loss_mode(model,&fresh,Some(&parallel),true)?;
            let path=out.join(format!("candidate-{index:02}-step-{step:02}.json"));
            MicroArtifact::new(model,artifact.updates,serde_json::json!({"diagnostic_only":true,
                "fresh_gain_probe":true,"projection":projection,"step":step})).save(&path)?;
            states.push(serde_json::json!({"step":step,"model":path,"parameters_sha256":parameters_sha(model),
                "value_bits_vs_start":probe["trajectory_value_bits"][step],"new_proof_choices":choices,"fresh_loss_measured":fresh_loss,"fresh_not_worse_than_start":fresh_loss<=start_fresh+1e-12}));
        }
        let final_model=trajectory.last().ok_or_else(||invalid("fresh gain missing admissible fallback"))?;
        let bits=final_model.parameters().iter().map(|w|w.to_bits()).collect::<Vec<_>>();
        if let Some((_,old))=repeated.iter().find(|(mode,_)|*mode==projection) {
            if old!=&bits {return Err(invalid("fresh gain repeated final model differs"));}
        } else {repeated.push((projection,bits));}
        if trajectory.len()==1 && !bits_equal(final_model,&candidate) {return Err(invalid("fresh gain failed to keep exact fallback"));}
        let final_scores=probe["final_scores"].clone();
        rows.push(serde_json::json!({"projection_enabled":projection,"probe":probe,"trajectory":states,
            "final_scores":final_scores,"best_admissible_is_last":true,"fallback_bits_exact":trajectory.len()!=1 || bits_equal(final_model,&candidate)}));
    }
    for p in [&m.anchor,&m.candidate,&m.fresh_saved_examples,&m.native_config,&m.original_report].into_iter().chain(panel_inputs.iter()) {p.bytes()?;}
    for p in &m.proof_positions {p.input.bytes()?;}
    if fs::read(&candidate_path)?!=candidate_bytes || fs::read(manifest_path)?!=manifest_bytes || fs::read(report_path)?!=report_bytes {
        return Err(invalid("Fresh gain probe inputs changed during measurement"));
    }
    let result=serde_json::json!({"schema":"paisho-gen5-fresh-gain-probe-v1",
        "frontier_manifest":manifest_path,"frontier_sha256":sha256(&manifest_bytes),
        "validation339_report":report_path,"validation339_report_sha256":sha256(&report_bytes),
        "candidate":candidate_path,"candidate_file_sha256":sha256(&candidate_bytes),"anchor":m.anchor,"panels":panel_inputs,
        "anchor_new_choices":anchor_choices,"pre_consolidation_choices":pre_choices,"start_new_choices":start_choices,"start_fresh":start_fresh,"anchor_fresh":anchor_fresh,"loading_seconds_excluded":loading_seconds,
        "runs":rows,"all_input_hashes_unchanged":true,"isolated_policy_adjustments":true,"sgd_or_replay_admission":false,"production_publication":false,
        "scope":"private fresh-proof policy adjustment; original actor011 Guard never reanchored, V frozen, at most4 admitted steps and12 full checks, criteria unchanged; all retained steps and fresh losses measured; no campaign checkpoint or publication"});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}
