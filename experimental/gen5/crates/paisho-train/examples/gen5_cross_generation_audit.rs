//! Frozen input/alias audit against the three Gen3.5 witnesses. No learning.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use std::{fs, path::Path, sync::Arc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 { return Err("MODEL ALIAS_DIRECTORY OUTPUT".into()); }
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let artifact = MicroArtifact::load(Path::new(&args[1]))?;
    let model = Arc::new(artifact.model()?);
    let dir = Path::new(&args[2]);
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("aliases.json"))?)?;
    let mut rows = Vec::new();
    let mut reserves_checked = 0;
    for w in manifest["witnesses"].as_array().ok_or("witnesses")? {
        let mut sides = Vec::new();
        let mut states = Vec::new();
        let mut contexts = Vec::new();
        for field in ["prefix_a", "prefix_b"] {
            let record: GameRecord = fs::read_to_string(dir.join(w[field].as_str().unwrap()))?.parse()?;
            let p = record.replay()?;
            assert_eq!(p.rule_profile(), RuleProfileId::SkudPaiShoGen5V1);
            let state = model.state_features(&p);
            assert_eq!(state.len(), 417);
            let mut inventory = Vec::new();
            for (seat, owner) in [p.to_move(), p.to_move().opponent()].into_iter().enumerate() {
                for (i, kind) in STANDARD_TILE_KINDS.iter().enumerate() {
                    let remaining = p.reserve(owner).count(*kind);
                    assert_eq!(state[64 + seat * 12 + i] * 32., remaining as f64);
                    reserves_checked += 1;
                    let on_board = p.board().occupied().filter(|(_, tile)| tile.owner == owner && tile.kind == *kind).count();
                    let initial = record.initial_position();
                    let initial_total = initial.reserve(owner).count(*kind) as usize + initial.board().occupied().filter(|(_, tile)| tile.owner == owner && tile.kind == *kind).count();
                    inventory.push(json!({"seat":seat,"kind_index":i,"reserve":remaining,
                        "board":on_board,"initial_total":initial_total,
                        "absent_from_reserve_and_board":initial_total-remaining as usize-on_board}));
                }
            }
            let actions = legal_actions(&p);
            let features = actions.iter().map(|a| micro_action_features(&p,*a)).collect::<Vec<_>>();
            let embedding = model.embed(&state);
            let prior = model.memory_priors(&state, &features,
                &micro_softmax(&MicroModel::logits(&embedding, &features))?, 0)?;
            let pick = (0..actions.len()).max_by(|a,b| prior[*a].total_cmp(&prior[*b]).then_with(|| b.cmp(a))).unwrap();
            let mut picked = p.clone(); picked.apply(actions[pick])?;
            let mut witness_actions = Vec::new();
            for t in w["same_action_input_different_immediate_result"].as_array().unwrap() {
                let a: Action = t["action"].as_str().unwrap().parse()?;
                let mut next = p.clone(); next.apply(a)?;
                witness_actions.push(json!({"action":a.to_string(),"wins":next.outcome()==GameOutcome::Win(p.to_move()),
                    "prior":prior[actions.iter().position(|candidate| *candidate==a).unwrap()],
                    "logit":MicroModel::logit(&embedding,&micro_action_features(&p,a))}));
            }
            let context = model.memory_context(&state, 0)?.ok_or("memory")?;
            let r = MicroMctsSession::new(model.clone()).search_with_options(&p,512,None,
                MicroSearchOptions {proof_search:true,..Default::default()})?;
            sides.push(json!({"path":w[field],"value":embedding.value,"policy_hidden":embedding.hidden,
                "witness_actions":witness_actions,"inventory":inventory,"legal":features.len(),
                "memory_neighbors":context.neighbors,"memory_patterns":context.patterns,
                "raw_policy_selected":actions[pick].to_string(),"raw_policy_immediate_win":picked.outcome()==GameOutcome::Win(p.to_move()),
                "search_selected":r.actions[r.selected_index].to_string(),"search_proof":r.proven_value,
                "search_simulations":r.simulations,"tactical_evaluations":r.tactical_evaluations}));
            contexts.push(context); states.push(state);
        }
        assert_eq!(&states[0][..128],&states[1][..128]);
        assert_ne!(states[0],states[1]);
        let changed:Vec<_>=(128..417).filter(|i|states[0][*i]!=states[1][*i]).collect();
        rows.push(json!({"old_inputs_equal":true,"new_inputs_equal":false,"changed_board_inputs":changed,
            "memory_same_cache_entry":Arc::ptr_eq(&contexts[0],&contexts[1]),"sides":sides}));
    }
    fs::write(&args[3],serde_json::to_vec_pretty(&json!({"model":artifact.identity(),"updates":artifact.updates,
        "parameters":model.parameters().len(),"reserve_coordinates_checked":reserves_checked,"pairs":rows}))?)?;
    Ok(())
}
