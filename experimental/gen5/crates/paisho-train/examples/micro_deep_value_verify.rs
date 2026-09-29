//! Frozen ordered V5 kernels: bit hashes include f64 sign bits, never wall times.
//! No model publication, production writes, or strength claim.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path, sync::Arc};
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
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=4 {return Err("usage: micro_deep_value_verify OLD NEW MANIFEST OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let old=MicroArtifact::load(Path::new(&args[0]))?;let new=MicroArtifact::load(Path::new(&args[1]))?;
    let a=Arc::new(old.model()?);let b=Arc::new(new.model()?);
    if !b.has_deep_value() || b.parameters()[..a.parameters().len()]!=*a.parameters() || old.updates!=new.updates {return Err("migration mismatch".into());}
    if !Arc::ptr_eq(a.sequence_memory().ok_or("old bank")?,b.sequence_memory().ok_or("new bank")?) {return Err("bank was copied".into());}
    let manifest:Value=serde_json::from_slice(&fs::read(&args[2])?)?;let cases=load(&manifest,&a)?;
    let mut searches=0;let mut informative_output_gradients=0;let mut old_gradient_parameters=0;
    for (i,c) in cases.iter().enumerate() {
        let x=a.embed(&c.example.state);let y=b.embed(&c.example.state);
        if x.value.to_bits()!=y.value.to_bits() || bits(&MicroModel::logits(&x,&c.example.actions))!=bits(&MicroModel::logits(&y,&c.example.actions)) {return Err("neutral predictions differ".into());}
        let (la,ga)=a.loss_gradient(&c.example)?;let (lb,gb)=b.loss_gradient(&c.example)?;
        if la.value.to_bits()!=lb.value.to_bits() || la.policy.to_bits()!=lb.policy.to_bits() || bits(&ga)!=bits(&gb[..ga.len()]) {return Err("old raw gradients differ".into());}
        old_gradient_parameters+=ga.len();
        informative_output_gradients+=usize::from(gb[MICRO_DEEP_VALUE_PARAMETERS-33..].iter().any(|g|g.abs()>1e-12));
        if i<16 {for budget in [256,512] {
            let mut first=MicroMctsSession::new(a.clone());let mut second=MicroMctsSession::new(b.clone());
            for simulations in [budget,64] {
                let options=MicroSearchOptions {proof_search:true,seed:95717+i as u64,dirichlet_fraction:0.25,forced_playout_strength:2.,..Default::default()};
                let ra=first.search_with_options(&c.position,simulations,None,options)?;
                let rb=second.search_with_options(&c.position,simulations,None,options)?;
                if search_digest(&ra)!=search_digest(&rb) {return Err("neutral search differs".into());}
                searches+=1;
            }
        }}
    }
    fs::write(&args[3],serde_json::to_vec_pretty(&json!({"old":old.identity(),"new":new.identity(),"parameters":b.parameters().len(),
        "old_parameters_exact":a.parameters().len(),"positions":cases.len(),"gradient_prefix_coefficients_exact":old_gradient_parameters,
        "neutral_search_pairs":searches,"informative_output_gradients":informative_output_gradients,"same_bank_arc":true,
        "updates":new.updates,"no_model_updates":true}))?)?;
    Ok(())
}
