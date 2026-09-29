//! Frozen ordered V5 kernels: bit hashes include f64 sign bits, never wall times.
//! No model publication, production writes, or strength claim.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, sync::Arc, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn bits(values: &[f64]) -> Vec<u64> { values.iter().map(|x| x.to_bits()).collect() }
fn digest(value: Value) -> String { hash(&serde_json::to_vec(&value).unwrap()) }
struct Case { example: MicroExample, position: Position }
fn load(manifest: &Value, model: &MicroModel) -> Result<Vec<Case>> {
    let mut cases = vec![];
    for row in manifest["rows"].as_array().ok_or("rows")? {
        let bytes = fs::read(row["targets"].as_str().ok_or("targets")?)?;
        if hash(&bytes) != row["targets_sha256"] { return Err("target hash".into()); }
        let mut decoded = vec![];
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut decoded)?;
        let examples: Vec<SavedMicroExample> = serde_json::from_slice(&decoded)?;
        let saved = &examples[row["index"].as_u64().ok_or("index")? as usize];
        let bytes = fs::read(row["psr"].as_str().ok_or("psr")?)?;
        if hash(&bytes) != row["psr_sha256"] { return Err("PSR hash".into()); }
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        record.replay()?;
        let mut position = record.initial_position();
        for &action in record.actions().iter().take(saved.decision - 1) { position.apply(action)?; }
        let mut example = saved.example_for_rules(record.rules())?;
        let state = model.state_features(&position);
        if bits(&state) != bits(&example.state[..state.len()]) { return Err("state replay mismatch".into()); }
        example.state = state;
        let actions = legal_actions(&position);
        if !saved.actions.is_empty() && actions.iter().map(ToString::to_string).collect::<Vec<_>>() != saved.actions {
            return Err(format!("action order mismatch: {} #{} legal {} saved {}", row["targets"], saved.decision, actions.len(),saved.actions.len()).into());
        }
        for (&action, features) in actions.iter().zip(&example.actions) {
            if bits(&micro_action_features(&position, action)) != bits(features) {
                return Err("action features mismatch".into());
            }
        }
        cases.push(Case { example, position });
    }
    Ok(cases)
}
fn search_digest(r: &MicroSearchReport) -> String {
    digest(json!({"selected":r.selected_index,"policy":bits(&r.policy_target),
        "priors":bits(&r.priors),"search_priors":bits(&r.search_priors),
        "visits":r.visits,"new_visits":r.new_visits,"forced":r.new_forced_visits,
        "pruned":r.pruned_visits,"proof":r.proven_value,"proof_actions":r.proven_action_values,
        "value":r.network_value.to_bits(),"q":bits(&r.values),"simulations":r.simulations,
        "tactical":r.tactical_evaluations,"inherited":r.inherited_visits,
        "cache_hits":r.inference_cache_hits,"evals":r.inference_evaluations,
        "state":bits(&r.state),"actions":r.actions.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "action_features":r.action_features.iter().map(|a|bits(a)).collect::<Vec<_>>() }))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len()!=4 { return Err("usage: micro_exact_bench MODEL MANIFEST OUTPUT ROUNDS".into()); }
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let t = Instant::now();
    let artifact = MicroArtifact::load(Path::new(&args[0]))?;
    let model = Arc::new(artifact.model()?);
    let load_seconds = t.elapsed().as_secs_f64();
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let cases = load(&manifest, &model)?;
    let rounds: usize = args[3].parse()?;
    let mut results = vec![];
    let mut inference_seconds = 0.;
    let mut gradient_seconds = 0.;
    let mut search_seconds = 0.;
    for (i,c) in cases.iter().enumerate() {
        let e = &c.example;
        // Warm immutable memory contexts before timing repeated kernels.
        model.memory_context(&e.state, e.sequence_source)?;
        let t = Instant::now();
        let mut prediction = None;
        for _ in 0..rounds {
            let emb = model.embed(std::hint::black_box(&e.state));
            let logits = MicroModel::logits(&emb, &e.actions);
            let priors = if logits.is_empty() { vec![] } else { model.memory_priors(&e.state, &e.actions, &micro_softmax(&logits)?, e.sequence_source)? };
            std::hint::black_box(&priors);
            prediction = Some((emb,logits,priors));
        }
        inference_seconds += t.elapsed().as_secs_f64();
        let (emb,logits,priors) = prediction.ok_or("positive rounds required")?;
        let prediction = json!([bits(&emb.hidden),bits(&emb.policy_context),vec![emb.value.to_bits()],bits(&logits),bits(&priors)]);
        let t = Instant::now();
        let mut gradient = None;
        for _ in 0..rounds { gradient = Some(model.loss_gradient(std::hint::black_box(e))?); }
        gradient_seconds += t.elapsed().as_secs_f64();
        let (loss, g) = gradient.ok_or("positive rounds required")?;
        let forward=model.loss(e)?;
        assert_eq!(loss.value.to_bits(),forward.value.to_bits(),"forward-only value loss");
        assert_eq!(loss.policy.to_bits(),forward.policy.to_bits(),"forward-only policy loss");
        assert_eq!(model.value(&e.state).to_bits(),emb.value.to_bits(),"value-only forward");
        for action in legal_actions(&c.position) {
            let mut p=c.position.clone();p.apply(action)?;
            let x=model.state_features(&p);
            assert_eq!(model.value(&x).to_bits(),model.embed(&x).value.to_bits(),"successor value-only forward");
        }
        let mut policy_independent = None;
        if model.has_spatial() {
            let mut other = e.clone(); other.value = if e.value < 0. { 1. } else { -1. };
            let (_, alternate) = model.loss_gradient(&other)?;
            policy_independent = Some((0..4128).chain(4161..MICRO_VALUE_TRUNK)
                .all(|j|g[j].to_bits()==alternate[j].to_bits()));
        }
        let mut searches = vec![];
        if i < 16 {
            for budget in [256,512] {
                let t = Instant::now();
                let mut session = MicroMctsSession::new(model.clone());
                let options = MicroSearchOptions { proof_search:true,seed:95717+i as u64,
                    dirichlet_fraction:0.25,forced_playout_strength:2.,..Default::default() };
                let report = session.search_with_options(&c.position,budget,None,options)?;
                let retained = session.search_with_options(&c.position,64,None,options)?;
                search_seconds += t.elapsed().as_secs_f64();
                for r in [&report,&retained] {
                    if let Some(raw)=&r.raw_priors {
                        let base=micro_softmax(&MicroModel::logits(&model.embed(&r.state),&r.action_features))?;
                        let original=model.memory_priors(&r.state,&r.action_features,&base,0)?;
                        assert_eq!(bits(raw),bits(&original),"reused raw root prior");
                    }
                }
                searches.push(json!({"budget":budget,"fresh":search_digest(&report),"retained":search_digest(&retained)}));
            }
        }
        results.push(json!({"case":i,"phase":format!("{:?}",c.position.phase()),
            "actions":e.actions.len(),"prediction":digest(prediction),
            "gradient":digest(json!([bits(&g),vec![loss.value.to_bits(),loss.policy.to_bits()]])),
            "value_label_leaves_raw_policy_gradient_identical":policy_independent,"searches":searches}));
    }
    let batch: Vec<_> = cases.iter().map(|c| &c.example).collect();
    let mut learner = model.as_ref().clone();
    let mut updates = vec![];
    let mut batch_seconds = 0.;
    for _ in 0..rounds {
        let t = Instant::now();
        let loss = learner.train_batch_inline(&batch,0.001,1e-6)?;
        batch_seconds += t.elapsed().as_secs_f64();
        updates.push(digest(json!([bits(learner.parameters()),vec![loss.value.to_bits(),loss.policy.to_bits()]])));
    }
    let result = json!({"model":artifact.identity(),"rounds":rounds,"parameters":model.parameters().len(),
        "load_seconds":load_seconds,"inference_seconds":inference_seconds,"gradient_seconds":gradient_seconds,
        "search_seconds":search_seconds,"batch_seconds":batch_seconds,"results":results,"updates":updates});
    fs::write(&args[2],serde_json::to_vec_pretty(&result)?)?;
    println!("{}",json!({"inference":inference_seconds,"gradients":gradient_seconds,"search":search_seconds,"batch":batch_seconds}));
    Ok(())
}
