//! Controlled decomposition of noise, forced-playout pruning and proof import.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use std::{fs, sync::Arc};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("MODEL CERTIFICATE OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&args[1])?)?;
    let model = Arc::new(artifact.model()?);
    let data: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let record: GameRecord = data["prefix"].as_str().ok_or("prefix")?.parse()?;
    let p = record.replay()?;
    let cert: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
    assert_eq!(cert.verify(&p)?, GameOutcome::Win(p.to_move()));
    let action: Action = cert.children[0].0.parse()?;
    let mut rows = vec![];
    for installed in [false, true] {
        for noise in [0., 0.25] {
            for forced in [0., 2.] {
                for seed in [37, 913] {
                    let mut session = MicroMctsSession::new(model.clone());
                    if installed {
                        session.install_certificate(&p, &cert)?;
                    }
                    let r = session.search_with_options(
                        &p,
                        512,
                        None,
                        MicroSearchOptions {
                            proof_search: true,
                            seed,
                            dirichlet_fraction: noise,
                            forced_playout_strength: forced,
                            ..Default::default()
                        },
                    )?;
                    let index = r
                        .actions
                        .iter()
                        .position(|a| *a == action)
                        .ok_or("action")?;
                    let ex = MicroExample { policy_support: false, action_values: vec![], 
                        state: r.state.clone(),
                        actions: r.action_features.to_vec(),
                        policy: r.policy_target.clone(),
                        value: 1.,
                        policy_weight: 1.,
                        value_weight: 1.0, sequence_source: 0,
                    };
                    let mut updated = model.as_ref().clone();
                    updated.train_policy_step(&ex, 0.02 / 64.)?;
                    let e = updated.embed(&ex.state);
                    let after = updated.memory_priors(
                        &ex.state,
                        &ex.actions,
                        &micro_softmax(&MicroModel::logits(&e, &ex.actions))?,
                        0,
                    )?;
                    rows.push(json!({"installed":installed,"noise":noise,"forced":forced,"seed":seed,
                "proven":r.proven_value,"action_proven":r.proven_action_values[index],"action":action.to_string(),
                "simulations":r.simulations,"prior":r.priors[index],"search_prior":r.search_priors[index],
                "visits":r.visits[index],"forced_visits":r.new_forced_visits.get(index),"pruned":r.pruned_visits.get(index),
                "q":r.values[index],"target":r.policy_target[index],"after_one_policy_step":after[index],
                "selected":r.actions[r.selected_index].to_string()}));
                }
            }
        }
    }
    fs::write(
        &args[3],
        serde_json::to_vec_pretty(&json!({"rows":rows,"certificate_verified":true,
        "scope":"known winning action; not a proof that every alternative loses; frozen searches and discarded clone updates","production_writes":0}))?,
    )?;
    Ok(())
}
