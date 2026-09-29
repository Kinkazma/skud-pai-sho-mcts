//! Neutral migration, active learning and matched whole-search cost, isolated only.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Arc, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}
fn hash(v: &[u8]) -> String {
    format!("{:x}", Sha256::digest(v))
}
fn signature(r: &MicroSearchReport) -> Value {
    json!({"action":r.selected_index,"policy":bits(&r.policy_target),"prior":bits(&r.priors),"search_prior":bits(&r.search_priors),
        "visits":r.visits,"new":r.new_visits,"forced":r.new_forced_visits,"pruned":r.pruned_visits,"q":bits(&r.values),
        "proof":r.proven_value,"children":r.proven_action_values,"simulations":r.simulations,"evaluations":r.inference_evaluations,"hits":r.inference_cache_hits,"inherited":r.inherited_visits})
}
fn main() -> Result<()> {
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 5 {
        return Err("OLD NEW TEACHER_SELECTION OUTPUT_DIRECTORY".into());
    }
    fs::create_dir(&args[4])?;
    let out = Path::new(&args[4]);
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let a = MicroArtifact::load(Path::new(&args[1]))?;
    let b = MicroArtifact::load(Path::new(&args[2]))?;
    let old = Arc::new(a.model()?);
    let new = Arc::new(b.model()?);
    assert_eq!(
        bits(old.parameters()),
        bits(&new.parameters()[..old.parameters().len()])
    );
    assert_eq!(a.updates, b.updates);
    assert!(Arc::ptr_eq(
        old.sequence_memory().unwrap(),
        new.sequence_memory().unwrap()
    ));
    let rows: Vec<Value> = serde_json::from_slice(&fs::read(&args[3])?)?;
    let mut cases = vec![];
    let mut gradient_coordinates = 0;
    let mut informative = 0;
    for row in rows.iter() {
        let path = row["path"].as_str().ok_or("target path")?;
        let bytes = fs::read(path)?;
        assert_eq!(hash(&bytes), row["sha256"]);
        let saved: Vec<SavedMicroExample> =
            serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
        let s = &saved[row["row"].as_u64().ok_or("row")? as usize];
        let bytes = fs::read(path.replace(".targets.json.gz", ".psr"))?;
        assert_eq!(hash(&bytes), row["psr_sha256"]);
        let r: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        r.replay()?;
        let mut position = r.initial_position();
        for a in &r.actions()[..s.decision - 1] {
            position.apply(*a)?;
        }
        let ex = s.example_for_rules(r.rules())?;
        assert_eq!(bits(&old.state_features(&position)), bits(&ex.state));
        assert_eq!(
            legal_actions(&position)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            s.actions
        );
        let ea = old.embed(&ex.state);
        let eb = new.embed(&ex.state);
        assert_eq!(ea.value.to_bits(), eb.value.to_bits());
        let pa = micro_softmax(&MicroModel::logits(&ea, &ex.actions))?;
        assert_eq!(
            bits(&old.memory_priors(&ex.state, &ex.actions, &pa, 0)?),
            bits(&new.memory_priors(&ex.state, &ex.actions, &pa, 0)?)
        );
        let (_, ga) = old.loss_gradient(&ex)?;
        let (_, gb) = new.loss_gradient(&ex)?;
        assert_eq!(bits(&ga), bits(&gb[..ga.len()]));
        gradient_coordinates += ga.len();
        informative += usize::from(
            gb[MICRO_NEURAL_MEMORY_START..]
                .iter()
                .any(|x| x.abs() > 1e-12),
        );
        cases.push((position, ex));
    }
    let mut neutral_searches = 0;
    for (i, (position, _)) in cases.iter().take(8).enumerate() {
        for budget in [256, 512] {
            let mut left = MicroMctsSession::new(old.clone());
            let mut right = MicroMctsSession::new(new.clone());
            left.set_root_value_strength(16.)?;
            right.set_root_value_strength(16.)?;
            for steps in [budget, 64] {
                let opt = MicroSearchOptions {
                    proof_search: true,
                    seed: 9478 + i as u64,
                    dirichlet_fraction: 0.25,
                    forced_playout_strength: 2.,
                    ..Default::default()
                };
                assert_eq!(
                    signature(&left.search_with_options(position, steps, None, opt)?),
                    signature(&right.search_with_options(position, steps, None, opt)?)
                );
                neutral_searches += 1;
            }
        }
    }
    let mut trained = new.as_ref().clone();
    let before = trained
        .loss_gradient(&cases[0].1)?
        .0
        .total(cases[0].1.policy_weight);
    let started = Instant::now();
    for i in 0..24 {
        trained.train_step(&cases[i % 4].1, 0.001, 0.)?;
    }
    let learning_seconds = started.elapsed().as_secs_f64();
    let after = trained
        .loss_gradient(&cases[0].1)?
        .0
        .total(cases[0].1.policy_weight);
    let changed = trained.parameters()[MICRO_NEURAL_MEMORY_START..]
        .iter()
        .zip(&new.parameters()[MICRO_NEURAL_MEMORY_START..])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(changed > 100_000);
    let artifact = MicroArtifact::new(&trained, a.updates + 24, json!({"diagnostic_only":true}));
    artifact.save(&out.join("trained-model.json"))?;
    let mut loaded = MicroArtifact::load(&out.join("trained-model.json"))?.model()?;
    trained.train_step(&cases[0].1, 0.001, 0.)?;
    loaded.train_step(&cases[0].1, 0.001, 0.)?;
    assert_eq!(bits(trained.parameters()), bits(loaded.parameters()));
    // Matched old searches plus one forced read of the neutral head, discarded.
    // This measures added whole-search waiting without changing any tree trajectory.
    let mut timings = vec![];
    for cycle in 0..3 {
        for shadow in [false, true, true, false] {
            for (i, (position, ex)) in cases.iter().take(8).enumerate() {
                for budget in [256, 512] {
                    let start = Instant::now();
                    if shadow {
                        let q = new
                            .neural_action_values(&ex.state, &ex.actions, 0)?
                            .unwrap();
                        assert!(q.iter().all(|v| *v == 0.));
                    }
                    let mut session = MicroMctsSession::new(old.clone());
                    session.set_root_value_strength(16.)?;
                    let r = session.search_with_options(
                        position,
                        budget,
                        None,
                        MicroSearchOptions {
                            proof_search: true,
                            seed: 9478 + i as u64,
                            ..Default::default()
                        },
                    )?;
                    timings.push(json!({"cycle":cycle,"shadow":shadow,"position":i,"actions":ex.actions.len(),"budget":budget,"seconds":start.elapsed().as_secs_f64(),"signature":hash(&serde_json::to_vec(&signature(&r))?)}));
                }
            }
        }
    }
    // Actual trained-head roots exercise the production hook as well.
    let mut active = vec![];
    let trained = Arc::new(trained);
    for (position, _) in cases.iter().take(8) {
        let t = Instant::now();
        let mut session = MicroMctsSession::new(trained.clone());
        session.set_root_value_strength(16.)?;
        let report = session.search_with_options(
            position,
            512,
            None,
            MicroSearchOptions {
                proof_search: true,
                ..Default::default()
            },
        )?;
        assert_eq!(report.actions, legal_actions(position));
        active.push(json!({"seconds":t.elapsed().as_secs_f64(),"legal":report.actions.len(),"simulations":report.simulations}));
    }
    let result = json!({"old_parameters":old.parameters().len(),"new_parameters":new.parameters().len(),"old_parameter_bits_exact":true,"bank_arc_shared":true,"positions":cases.len(),"old_gradient_coordinates_exact":gradient_coordinates,"informative_neural_gradients":informative,"neutral_search_pairs":neutral_searches,"neural_parameters_changed":changed,"loss_before":before,"loss_after":after,"learning_updates":24,"learning_seconds":learning_seconds,"reload_next_update_exact":true,"timings":timings,"active_searches":active,"no_production_writes":true});
    fs::write(
        out.join("verification.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!(
        "{}",
        json!({"neutral_searches":neutral_searches,"positions":cases.len(),"parameters_changed":changed})
    );
    Ok(())
}
