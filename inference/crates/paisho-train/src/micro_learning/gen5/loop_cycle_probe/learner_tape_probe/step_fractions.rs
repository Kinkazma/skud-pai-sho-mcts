//! Four fixed single-step displacement curves; no sampling, learning on grid,
//! publication or new game. The short prefix is reconstructed from frozen tape.
use super::*;
use std::io::Read;
const ORDINALS: [usize;4] = [16,205,440,734];
const SHORT_PREFIXES: [usize;4] = [16,0,3,12];
const FRACTIONS: [f64;5] = [0.,0.125,0.25,0.5,1.];
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String, source_plan: Input, tape_manifest: Input,
    primary: Input, validation: Input, anchors: Vec<Input>, probes: Vec<Input>,
    fractions: Vec<f64>, captured_value_association: String, max_seconds: u64,
}
struct Ready {
    before: MicroModel, after: MicroModel, fresh: Vec<Arc<MicroExample>>,
    probe: serde_json::Value, evidence: serde_json::Value,
}
fn small_frame(source: &Input) -> Result<Frame> {
    let bytes=source.bytes()?;
    let mut decoded=Vec::new();
    flate2::read::GzDecoder::new(bytes.as_slice()).take(128*1024*1024+1).read_to_end(&mut decoded)?;
    if decoded.len()>128*1024*1024 {return Err(invalid("selected tape frame exceeds diagnostic bound"));}
    Ok(serde_json::from_slice(&decoded)?)
}
fn blend(before: &MicroModel, after: &MicroModel, alpha: f64) -> Result<MicroModel> {
    if alpha==0. {return Ok(before.clone());}
    if alpha==1. {return Ok(after.clone());}
    if !FRACTIONS.contains(&alpha) {return Err(invalid("unplanned step fraction"));}
    let mut model=MicroModel::from_parameters(before.parameters().iter().zip(after.parameters())
        .map(|(a,b)|a+alpha*(b-a)).collect()).map_err(invalid)?;
    if let Some(bank)=before.sequence_memory(){model=model.with_sequence_memory_owned(bank.clone());}
    Ok(model)
}
// Slots 0..4 are native policy/direct-V/Q/total. Last two reproduce the
// frozen probe's historical powi(2) association, for exact endpoint checks only.
fn means(model: &MicroModel, rows: &[Arc<MicroExample>]) -> Result<[f64;6]> {
    if rows.is_empty(){return Err(invalid("selected full64 has no fresh observations"));}
    let mut sums=[0.;6];
    for e in rows {
        let loss=model.loss_loop_v3(e).map_err(invalid)?;
        let delta=model.value(&e.state)-e.value;
        let direct=0.5*delta*delta*e.value_weight;
        let legacy=0.5*delta.powi(2)*e.value_weight;
        let values=[loss.policy*e.policy_weight,direct,loss.value-direct,loss.total(e.policy_weight),legacy,loss.value-legacy];
        if values.iter().any(|v|!v.is_finite()){return Err(invalid("non-finite fractional fresh metric"));}
        for (sum,value) in sums.iter_mut().zip(values){*sum+=value;}
    }
    for sum in &mut sums{*sum/=rows.len() as f64;}
    Ok(sums)
}
fn verify_endpoint(values: &[f64;6], old: &serde_json::Value) -> Result<()> {
    for (i,key) in [(0,"weighted_policy"),(3,"total"),(4,"direct_value"),(5,"auxiliary_q_by_loss_subtraction")] {
        if old[key].as_f64().map(f64::to_bits)!=Some(values[i].to_bits()) {
            return Err(invalid(format!("fraction endpoint differs from frozen native probe: {key}")));
        }
    }
    Ok(())
}
fn same_gram(a: &serde_json::Value,b: &serde_json::Value) -> bool {
    let decode=|v:&serde_json::Value|serde_json::from_value::<Vec<Vec<f64>>>(v.clone());
    match (decode(a),decode(b)) {
        (Ok(a),Ok(b))=>a.len()==b.len() && a.iter().zip(b).all(|(a,b)|a.len()==b.len() && a.iter().zip(b).all(|(a,b)|a.to_bits()==b.to_bits())),
        _=>false,
    }
}
fn check_clock(start: Instant, limit: u64) -> Result<()> {
    if start.elapsed().as_secs_f64()>limit as f64 {return Err(invalid("step-fraction diagnostic exceeded fixed time bound"));} Ok(())
}
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists(){return Err(invalid("step-fractions output must be new"));}
    let started=Instant::now();let plan_bytes=fs::read(plan_path)?;let plan:Plan=serde_json::from_slice(&plan_bytes)?;
    if plan.schema!="paisho-gen5-tape-step-fractions-v1" || plan.anchors.len()!=4 || plan.probes.len()!=4
        || plan.fractions.iter().map(|x|x.to_bits()).collect::<Vec<_>>() != FRACTIONS.iter().map(|x|x.to_bits()).collect::<Vec<_>>()
        || plan.captured_value_association!="0.5*delta.powi(2)*weight" || !(60..=600).contains(&plan.max_seconds) {
        return Err(invalid("fixed step-fractions protocol changed"));
    }
    let source_bytes=plan.source_plan.bytes()?;let source:TapePlan=serde_json::from_slice(&source_bytes)?;
    if source.schema!="paisho-gen5-learner-tape-v1" || source.blocks.len()!=4
        || source.blocks.iter().map(|b|b.receipts.len()).collect::<Vec<_>>() != [205,232,285,244] {
        return Err(invalid("source four-block tape chronology changed"));
    }
    let manifest=json(&plan.tape_manifest)?;
    if manifest["schema"]!="paisho-gen5-native-learning-tape-v1" || manifest["plan_sha256"]!=sha256(&source_bytes)
        || manifest["sampler_driven_by"]!="recorded-reset" {return Err(invalid("tape manifest not bound to source plan"));}
    let frames:Vec<Vec<Input>>=serde_json::from_value(manifest["frames"].clone())?;
    if frames.iter().map(Vec::len).collect::<Vec<_>>()!=[205,232,285,244] {return Err(invalid("frame grouping changed"));}
    let o:Options=serde_json::from_slice(&source.config.bytes()?)?;
    if !o.learning_loop_v3 || o.publication_guard.as_ref()!=Some(&plan.primary.path)
        || o.publication_validation.as_ref()!=Some(&plan.validation.path) {return Err(invalid("V3 panel/config binding changed"));}
    plan.primary.bytes()?;plan.validation.bytes()?;
    let resume=json(&source.resume)?;
    if plan.anchors[0].path!=source.initial_actor.path || plan.anchors[0].sha256!=source.initial_actor.sha256 {
        return Err(invalid("initial actor binding changed"));
    }
    let (artifact,base)=load_model(&plan.anchors[0],None)?;
    let initial=Arc::new(Snapshot {identity:artifact.identity(),version:resume["publication_guard"]["accepted_version"].as_u64().ok_or_else(||invalid("initial version"))?,
        artifact:Some(Arc::new(artifact)),path:plan.anchors[0].path.clone(),model:Arc::new(base.clone())});
    let mut anchors=vec![base.clone()];
    for input in plan.anchors.iter().skip(1){anchors.push(load_model(input,Some(&base))?.1);}
    let probes=plan.probes.iter().map(json).collect::<Result<Vec<_>>>()?;
    fs::create_dir(out)?;write_json(&out.join("input-plan.json"),&serde_json::from_slice::<serde_json::Value>(&plan_bytes)?)?;
    let (pool,_)=cpu::build_pool(o.threads,None)?;
    // Authenticate proof panels once. Their immutable examples are independent
    // of which accepted anchor is subsequently used to rebuild the normals.
    let mut guard=publication::Guard::open(&plan.primary.path,&out.join("panel-setup"),o.value_policy_strength,initial,&serde_json::Value::Null)?;
    guard.enable_v2(&plan.validation.path)?;guard.enable_v3()?;
    let refs=guard.reference_examples();let validation=guard.diagnostic_validation_examples()?;
    let mut protections=Vec::new();
    for i in 0..4 {
        let probe=&probes[i];
        if probe["schema"]!="paisho-gen5-first-full64-gradient-flow-v1" || probe["block"].as_u64()!=Some(i as u64)
            || probe["receipt_ordinal"].as_u64()!=Some(ORDINALS[i] as u64)
            || probe["receipt_index_in_block"].as_u64()!=Some(SHORT_PREFIXES[i] as u64)
            || probe["batch_index"]!=0 || probe["actual_step_bits_exact"]!=true
            || probe["policy_gradient_constraint"]!=true || probe["anchor_bits"]!=bits(&anchors[i]) {
            return Err(invalid("fixed full64/anchor probe binding changed"));
        }
        let mut p=protection::Protection::new(&anchors[i],refs.clone())?;p.enable_loop_v3();
        p.enable_parallel(&[pool.clone()]);p.enable_validation_value(validation.clone())?;
        let signature=p.diagnostic_gradient_anchor_signature();
        for key in ["anchor_bits","reference_gradient_bits","policy_gradient_constraint"] {
            if signature[key]!=probe[key] {return Err(invalid(format!("cannot reproduce optimizer anchor: {key}")));}
        }
        if signature["loop_v3"]!=true || !same_gram(&signature["gram"],&probe["gram"]) {return Err(invalid("optimizer Gram/V3 changed"));}
        protections.push(p);
    }
    let setup_seconds=started.elapsed().as_secs_f64();let replay_started=Instant::now();
    let mut ready=Vec::new();let mut steps=0usize;
    for (bi,mut p) in protections.into_iter().enumerate() {
        let mut model=anchors[bi].clone();let mut checked=vec![];
        let block_start=ORDINALS[bi]-SHORT_PREFIXES[bi];
        for ri in 0..=SHORT_PREFIXES[bi] {
            check_clock(started,plan.max_seconds)?;
            let frame_source=&frames[bi][ri];let frame=small_frame(frame_source)?;
            validate_frame(&frame,bi,block_start+ri,o.rate)?;
            let expected_source=&source.blocks[bi].receipts[ri].receipt;
            if frame.source_receipt.path!=expected_source.path || frame.source_receipt.sha256!=expected_source.sha256 {
                return Err(invalid("selected frame receipt provenance changed"));
            }
            if ri<SHORT_PREFIXES[bi] && frame.items.len()>=64 {return Err(invalid("an earlier full64 would change selection"));}
            let rows=frame.items.iter().map(|e|restore(&e.example)).collect::<Result<Vec<_>>>()?;
            let batches=if ri==SHORT_PREFIXES[bi] {1}else{frame.rates.len()};
            for (batch_index,batch) in rows.chunks(64).take(batches).enumerate() {
                let selected=ri==SHORT_PREFIXES[bi];let before=model.clone();
                if selected {
                    if batch.len()!=64 || probes[bi]["before_bits"]!=bits(&before)
                        || probes[bi]["rate"].as_f64().map(f64::to_bits)!=Some(frame.rates[batch_index].to_bits())
                        || probes[bi]["l2"].as_f64().map(f64::to_bits)!=Some(1e-5_f64.to_bits()) {
                        return Err(invalid("selected pre-step/rate differs"));
                    }
                    let native=batch.iter().map(|e|encoded(e)).collect::<Vec<_>>();
                    let kinds=frame.items[..64].iter().map(|e|e.kind).collect::<Vec<_>>();
                    if probes[bi]["examples_sha256"]!=sha256(&serde_json::to_vec(&native)?)
                        || probes[bi]["kinds"]!=serde_json::to_value(kinds)? {return Err(invalid("selected gradient inputs differ"));}
                }
                p.train_shared(&mut model,batch,frame.rates[batch_index],1e-5)?;steps+=1;
                if bits(&model)!=frame.expected_after_sgd[batch_index] {return Err(invalid("short tape SGD does not reproduce exact bits"));}
                checked.push(serde_json::json!({"frame":frame_source,"batch":batch_index,"after_bits":bits(&model)}));
                if selected {
                    if probes[bi]["actual_after_bits"]!=bits(&model) {return Err(invalid("full-step probe result changed"));}
                    let fresh=batch.iter().zip(&frame.items[..64]).filter(|(_,e)|e.kind==0).map(|(e,_)|e.clone()).collect::<Vec<_>>();
                    if probes[bi]["fresh_rows"].as_u64()!=Some(fresh.len() as u64) {return Err(invalid("fresh selection changed"));}
                    ready.push(Ready {before,after:model.clone(),fresh,probe:probes[bi].clone(),
                        evidence:serde_json::json!({"block":bi,"anchor":plan.anchors[bi],"probe":plan.probes[bi],"steps":checked.clone()})});
                }
            }
        }
    }
    if steps!=35 || ready.len()!=4 {return Err(invalid("bounded replay step total changed"));}
    let replay_seconds=replay_started.elapsed().as_secs_f64();
    // All four parameter trajectories must match BEFORE any fraction is read.
    let verify_started=Instant::now();let mut endpoints=Vec::new();
    for case in &ready {
        let before=means(&case.before,&case.fresh)?;let after=means(&case.after,&case.fresh)?;
        verify_endpoint(&before,&case.probe["fresh_before"])?;verify_endpoint(&after,&case.probe["fresh_after"])?;
        endpoints.push((before,after));
    }
    let endpoint_seconds=verify_started.elapsed().as_secs_f64();let measure_started=Instant::now();let mut results=vec![];
    for (bi,case) in ready.iter().enumerate() {
        let mut readings=vec![];
        for &alpha in &FRACTIONS {
            check_clock(started,plan.max_seconds)?;
            let model=blend(&case.before,&case.after,alpha)?;
            let values=if alpha==0. {endpoints[bi].0}else if alpha==1. {endpoints[bi].1}else{means(&model,&case.fresh)?};
            readings.push(serde_json::json!({"fraction":alpha,"bits":bits(&model),"fresh_rows":case.fresh.len(),
                "policy":values[0],"direct_value":values[1],"auxiliary_q":values[2],"total":values[3],
                "delta_from_before":[values[0]-endpoints[bi].0[0],values[1]-endpoints[bi].0[1],values[2]-endpoints[bi].0[2],values[3]-endpoints[bi].0[3]]}));
        }
        let result=serde_json::json!({"block":bi,"evidence":case.evidence,"readings":readings,"endpoints_native_bits_exact":true,"frozen_probe_endpoint_metrics_exact":true});
        write_json(&out.join(format!("block-{bi:03}.json")),&result)?;results.push(result);
    }
    for src in plan.anchors.iter().chain(&plan.probes).chain([&plan.source_plan,&plan.tape_manifest,&plan.primary,&plan.validation]) {src.bytes()?;}
    let report=serde_json::json!({"schema":"paisho-gen5-tape-step-fractions-result-v1","plan_sha256":sha256(&plan_bytes),
        "all_four_optimizer_anchors_exact_before_replay":true,"all_35_sgd_bits_exact_before_fraction_reads":true,
        "all_eight_endpoint_metrics_exact_before_intermediate_fractions":true,"results":results,
        "setup_seconds":setup_seconds,"prefix_replay_seconds":replay_seconds,"endpoint_verification_seconds":endpoint_seconds,
        "fraction_read_seconds":measure_started.elapsed().as_secs_f64(),"total_seconds":started.elapsed().as_secs_f64(),
        "replayed_sgd_steps":steps,"new_sampling":0,"grid_sgd_steps":0,"bank_loads":1,"archive_loads":0,
        "fraction_interpolation":"before + alpha*(native_after-before); endpoint0/1 clone exact",
        "scope":"four fixed actual tape steps; fresh kind0 only, no new targets, no learning on grid, no publication/promotion or learning-rate recommendation"});
    write_json(&out.join("report.json"),&report)?;Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractions_keep_endpoints_exact_and_use_fixed_displacement() {
        let a=MicroModel::seeded(19);let b=MicroModel::seeded(20);
        assert!(blend(&a,&b,0.).unwrap().shares_storage_with(&a));
        assert!(blend(&a,&b,1.).unwrap().shares_storage_with(&b));
        let c=blend(&a,&b,0.25).unwrap();
        for ((a,b),c) in a.parameters().iter().zip(b.parameters()).zip(c.parameters()) {assert_eq!(c.to_bits(),(a+0.25*(b-a)).to_bits());}
        assert!(blend(&a,&b,0.3).is_err());
    }
    #[test]
    fn fixed_prefix_count_is_35_and_gram_checks_signed_zero() {
        assert_eq!(SHORT_PREFIXES.iter().sum::<usize>()+4,35);
        assert!(!same_gram(&serde_json::json!([[0.]]),&serde_json::json!([[-0.]])));
        assert!(same_gram(&serde_json::json!([[0.,1.]]),&serde_json::json!([[0.,1.]])));
    }
}
