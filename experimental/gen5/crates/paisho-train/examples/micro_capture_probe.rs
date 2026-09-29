//! Read-only cross-generation capture diagnostic; no synthetic training targets.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::json;
use std::{fs, path::Path, sync::Arc};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len()!=3 { return Err("usage: micro_capture_probe MODEL PSR OUTPUT".into()); }
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let old: GameRecord = fs::read_to_string(&args[1])?.parse()?;
    old.replay()?;
    let (record,_) = old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?;
    if old.actions()!=record.actions() { return Err("Gen5 rules changed the recorded trajectory".into()); }
    let artifact=MicroArtifact::load(Path::new(&args[0]))?;
    let model=Arc::new(artifact.model()?);
    let mut p=record.initial_position();let mut positions=vec![p.clone()];
    for &action in record.actions() {p.apply(action)?;positions.push(p.clone());}
    let value=|p:&Position| {
        let sign=if p.to_move()==Player::Host {1.} else {-1.};
        sign*model.embed(&model.state_features(p)).value
    };
    let mut rows=vec![];
    for decision in [5,6,7,9,10,11,13,14,15] {
        let p=&positions[decision];let mut searches=vec![];
        let actions=legal_actions(p);
        let state=model.state_features(p);
        let action_features:Vec<_>=actions.iter().map(|a|micro_action_features(p,*a)).collect();
        let logits=MicroModel::logits(&model.embed(&state),&action_features);
        let priors=model.memory_priors(&state,&action_features,&micro_softmax(&logits)?,0)?;
        if [5,6,9,10,13,14].contains(&decision) {
            for budget in [256,512] {
                let r=MicroMctsSession::new(model.clone()).search_with_options(p,budget,None,
                    MicroSearchOptions {proof_search:true,..Default::default()})?;
                let recorded=r.actions.iter().position(|a|*a==record.actions()[decision]).ok_or("recorded legal action")?;
                searches.push(json!({"budget":budget,"selected":r.actions[r.selected_index].to_string(),
                    "recorded_next_action":record.actions()[decision].to_string(),
                    "recorded_selected":recorded==r.selected_index,"recorded_visits":r.visits[recorded],
                    "recorded_q_from_chooser":r.values[recorded],"selected_q_from_chooser":r.values[r.selected_index],
                    "proof_from_chooser":r.proven_value,"actual_simulations":r.simulations,
                    "opened_root_actions":r.new_visits.iter().filter(|v|**v>0).count()}));
            }
        }
        let prior_pick=(0..priors.len()).max_by(|&a,&b|priors[a].total_cmp(&priors[b]).then_with(||b.cmp(&a))).ok_or("legal policy")?;
        rows.push(json!({"after_decision":decision,"chooser":format!("{:?}",p.to_move()),
            "phase":format!("{:?}",p.phase()),"host_value":value(p),"raw_policy_pick":actions[prior_pick].to_string(),
            "searches":searches}));
    }
    let mut deltas=vec![];
    for (before,after) in [(7,11),(11,15)] {
        let a=&positions[before];let b=&positions[after];let x=model.state_features(a);let y=model.state_features(b);
        let changed:Vec<_>=x.iter().zip(&y).enumerate().filter(|(_, (a,b))|a.to_bits()!=b.to_bits())
            .map(|(i,(a,b))|json!({"index":i,"before":a,"after":b})).collect();
        deltas.push(json!({"before":before,"after":after,"same_board":a.board()==b.board(),
            "same_chooser":a.to_move()==b.to_move(),"same_phase":a.phase()==b.phase(),
            "host_value_before":value(a),"host_value_after":value(b),"changed_inputs":changed}));
    }
    fs::write(&args[2],serde_json::to_vec_pretty(&json!({"model":artifact.identity(),
        "parameters":model.parameters().len(),"updates":artifact.updates,"rules":record.rules().to_string(),
        "terminal_outcome":format!("{:?}",positions.last().unwrap().outcome()),"rows":rows,"deltas":deltas,
        "interpretation":"Fixed-model diagnostic, not proof that every capture is bad or a strength evaluation; no training."}))?)?;
    Ok(())
}
