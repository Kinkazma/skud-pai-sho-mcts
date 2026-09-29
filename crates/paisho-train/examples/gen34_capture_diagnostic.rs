//! Fixed-position capture trace; never updates models or produces training games.
use paisho_ai::{MctsConfig, MctsEvaluator, MctsSession};
use paisho_core::{legal_actions, GameRecord, Player};
use paisho_train::gen32::Artifact;
use serde_json::json;
use std::fs;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 && args.len() != 5 {
        return Err("usage: MODEL HUMAN.json OUTPUT.json [capture-coverage]".into());
    }
    let capture_enabled = args.get(4).map(String::as_str) == Some("capture-coverage");
    if args.len() == 5 && !capture_enabled {
        return Err("unknown option".into());
    }
    let artifact = Artifact::load(std::path::Path::new(&args[1]))?;
    let model = artifact.model()?;
    let input: serde_json::Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let record: GameRecord = input["moves"].as_str().ok_or("missing PSR")?.parse()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let mut rows = vec![];
    for mode in ["cold", "retained-human-history"] {
        for seed in [17, 731, 456] {
            let mut session = MctsSession::new(
                seed,
                MctsConfig {
                    simulations: 512,
                    ..Default::default()
                },
                &model,
            )?;
            session.set_solver(true);
            session.set_capture_coverage(capture_enabled);
            let mut p = record.initial_position();
            for (i, action) in record.actions().iter().copied().take(14).enumerate() {
                let decision = i + 1;
                let target = [6, 10, 14].contains(&decision);
                if p.to_move() == Player::Host && (target || mode == "retained-human-history") {
                    if mode == "cold" {
                        session.clear();
                    }
                    let legal = legal_actions(&p);
                    let report = pool.install(|| session.search_until(&p, &legal, None))?;
                    if target {
                        let reply = record.actions()[i + 1];
                        let path = session.trace_path(&[action, reply], Player::Host);
                        let mut after = p.clone();
                        after.apply(action)?;
                        let capture = after.apply(reply)?;
                        let actual = report
                            .actions
                            .iter()
                            .find(|a| a.action == action)
                            .ok_or("actual action absent")?;
                        let selected = report.actions[report.selected_index];
                        let priors = model.policy_bias(&p, &legal)?;
                        let mut selected_position = p.clone();
                        selected_position.apply(selected.action)?;
                        let captures = if selected_position.to_move() != Player::Host {
                            let mut found = Vec::new();
                            for response in legal_actions(&selected_position) {
                                let mut candidate = selected_position.clone();
                                let outcome = candidate.apply(response)?;
                                if let Some(tile) = outcome.captured {
                                    if tile.owner == Player::Host {
                                        found.push(json!({"action":response.to_string(),"captured":format!("{:?}",tile.kind)}));
                                    }
                                }
                            }
                            Some(found)
                        } else {
                            None
                        };

                        let trace:Vec<_>=path.iter().map(|s|json!({"action":s.action.to_string(),"stage":format!("{:?}",s.stage),"chooser":format!("{:?}",s.chooser),"visits":s.visits,"parent_visited_children":s.parent_visited_children,"pending_rank":s.pending_rank,"mean":s.mean_value,"cached_leaf":s.cached_leaf_value,"proof":format!("{:?}",s.proof)})).collect();
                        let coverage = if path[0].visits == 0 {
                            "candidate-not-simulated"
                        } else if path[1].visits == 0 {
                            "capture-not-simulated"
                        } else if path[1].visits == 1 {
                            "capture-leaf-only"
                        } else {
                            "capture-visited"
                        };
                        let ranking:Vec<_>=report.actions.iter().enumerate().map(|(j,a)|json!({"action":a.action.to_string(),"visits":a.visits,"q":if a.visits>0{Some(a.mean_value())}else{None},"bias":priors.as_ref().map(|v|v[j])})).collect();
                        rows.push(json!({"decision":decision,"mode":mode,"seed":seed,"budget":512,"capture_coverage":capture_enabled,"recorded_move":action.to_string(),"selected_move":selected.action.to_string(),"recorded_selected":selected.action==action,"coverage":coverage,"trace":trace,"recorded_q":if actual.visits>0{Some(actual.mean_value())}else{None},"selected_q":selected.mean_value(),"captured":format!("{:?}",capture.captured),"root_value":model.value_at(&p,Player::Host),"after_capture_value":model.value_at(&after,Player::Host),"after_capture_outcome":format!("{:?}",after.outcome()),"actual_simulations":report.simulations,"selected_capture_replies":captures,"maximum_depth":report.maximum_depth,"root_actions":ranking}));
                    }
                }
                p.apply(action)?;
                session.advance(action);
            }
        }
    }
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(
            &json!({"schema":"paisho-capture-diagnostic-v1","model_updates":artifact.updates,"human_metadata":{"agent":input["agent"],"elo":input["elo"],"human_side":input["human_side"]},"limitations":["Original search seed and internal tree not recorded; controlled reproductions only.","Retained mode searches Host decisions then advances recorded human/bot actions, including disagreements.","Capture is verified legal; a material loss is not a proof of whole-game loss.","No match comparison, no training, no production activation."],"rows":rows}),
        )?,
    )?;
    Ok(())
}
