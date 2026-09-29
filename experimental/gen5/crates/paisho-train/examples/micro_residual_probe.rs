//! Bounded frozen residual-head acceptance probe. No campaign mutation.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::json;
use std::{fs, path::Path, sync::Arc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: micro_residual_probe MODEL COUNTEREXAMPLE.psr OUTPUT_DIR".into());
    }
    let output = Path::new(&args[3]);
    fs::create_dir(output)?;
    let artifact = MicroArtifact::load(Path::new(&args[1]))?;
    let old = artifact.model()?;
    let record: GameRecord = fs::read_to_string(&args[2])?.parse()?;
    let mut p = record.initial_position();
    let mut positions = vec![p.clone()];
    for (i, a) in record.actions().iter().take(63).enumerate() {
        p.apply(*a)?;
        if [20, 40, 62].contains(&i) {
            positions.push(p.clone());
        }
    }
    let actions = legal_actions(&p);
    let mut wins = vec![];
    for (i, a) in actions.iter().enumerate() {
        let mut q = p.clone();
        q.apply(*a)?;
        if q.outcome() == GameOutcome::Win(p.to_move()) {
            wins.push(i);
        }
    }
    if wins.is_empty() {
        return Err("expected winning moves".into());
    }
    let mut target = vec![0.; actions.len()];
    for i in &wins {
        target[*i] = 1. / wins.len() as f64;
    }
    let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: 0,
        state: micro_state_features(&p).to_vec(),
        actions: actions
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect(),
        policy: target,
        value: old.embed(&micro_state_features(&p)).value,
        policy_weight: 1.,
    };
    let score = |m: &MicroModel| {
        let logits = MicroModel::logits(&m.embed(&ex.state), &ex.actions);
        let best_win = wins
            .iter()
            .map(|i| logits[*i])
            .fold(f64::NEG_INFINITY, f64::max);
        let best_other = logits
            .iter()
            .enumerate()
            .filter(|(i, _)| !wins.contains(i))
            .map(|(_, x)| *x)
            .fold(f64::NEG_INFINITY, f64::max);
        json!({"winning_margin":best_win-best_other,"winning_mass":micro_softmax(&logits).unwrap().iter().enumerate().filter(|(i,_)|wins.contains(i)).map(|(_,p)|p).sum::<f64>()})
    };
    let upgraded = old.with_residual_policy(29);
    let mut learned = upgraded.clone();
    let initial = score(&old);
    let start = Instant::now();
    let mut curve = vec![];
    for step in 1..=3000 {
        learned.train_batch_inline(&[&ex], 0.05, 1e-5)?;
        if step % 250 == 0 {
            curve.push(json!({"step":step,"seconds":start.elapsed().as_secs_f64(),"score":score(&learned)}));
        }
    }
    let learning_seconds = start.elapsed().as_secs_f64();
    MicroArtifact::new(
        &learned,
        artifact.updates + 3000,
        json!({"diagnostic_only":true,"parent":artifact.identity(),"single_position_fit":true}),
    )
    .save(&output.join("diagnostic-model.json"))?;
    let rows: Vec<_> = positions
        .iter()
        .map(|p| {
            (
                micro_state_features(p),
                legal_actions(p)
                    .iter()
                    .map(|a| micro_action_features(p, *a))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let mut inference = vec![];
    for pass in 0..20 {
        for index in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
            let m = if index == 0 { &old } else { &learned };
            let t = Instant::now();
            for _ in 0..100 {
                for (x, a) in &rows {
                    let emb = m.embed(x);
                    std::hint::black_box(MicroModel::logits(&emb, a));
                }
            }
            inference.push(json!({"pass":pass,"residual":index==1,"seconds":t.elapsed().as_secs_f64(),"positions":400}));
        }
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let mut searches = vec![];
    for budget in [256, 512] {
        for pass in 0..4 {
            for index in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
                let model = Arc::new(if index == 0 {
                    old.clone()
                } else {
                    learned.clone()
                });
                let t = Instant::now();
                let mut outcomes = vec![];
                for p in &positions {
                    let r = pool.install(|| {
                        MicroMctsSession::new(model.clone()).search_with_options(
                            p,
                            budget,
                            None,
                            MicroSearchOptions {
                                proof_search: true,
                                ..Default::default()
                            },
                        )
                    })?;
                    outcomes.push(json!({"selected":r.selected_index,"simulations":r.simulations,"inference":r.inference_evaluations}));
                }
                searches.push(json!({"budget":budget,"pass":pass,"residual":index==1,"seconds":t.elapsed().as_secs_f64(),"positions":positions.len(),"outcomes":outcomes}));
            }
        }
    }
    // Isolate added arithmetic at identical search decisions: a common tiny
    // bias activates the full head but rounds away from every observed logit.
    // Verify tree statistics; never call changed-tree time an isolated overhead.
    let mut neutral_weights = upgraded.parameters().to_vec();
    *neutral_weights.last_mut().unwrap() = f64::MIN_POSITIVE;
    let neutral = Arc::new(MicroModel::from_parameters(neutral_weights)?);
    let baseline = Arc::new(old.clone());
    let mut overhead = vec![];
    for budget in [256, 512] {
        for pass in 0..8 {
            let mut reports = vec![vec![], vec![]];
            for index in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
                let model = if index == 0 {
                    baseline.clone()
                } else {
                    neutral.clone()
                };
                let t = Instant::now();
                for p in &positions {
                    reports[index].push(pool.install(|| {
                        MicroMctsSession::new(model.clone()).search_with_options(
                            p,
                            budget,
                            None,
                            MicroSearchOptions {
                                proof_search: true,
                                ..Default::default()
                            },
                        )
                    })?);
                }
                overhead.push(json!({"budget":budget,"pass":pass,"residual":index==1,"seconds":t.elapsed().as_secs_f64(),"positions":positions.len()}));
            }
            for (a, b) in reports[0].iter().zip(&reports[1]) {
                if a.visits != b.visits
                    || a.priors != b.priors
                    || a.selected_index != b.selected_index
                    || a.inference_evaluations != b.inference_evaluations
                {
                    return Err("neutral residual changed search trace".into());
                }
            }
        }
    }
    let report = json!({"parent":artifact.identity(),"old_parameters":old.parameters().len(),"new_parameters":learned.parameters().len(),"actions":actions.len(),"wins":wins,"initial":initial,"final":score(&learned),"learning_seconds":learning_seconds,"curve":curve,"inference":inference,"searches":searches,"same_trace_overhead":overhead,"concurrent_training":true,"no_strength_claim":true});
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "{}",
        json!({"initial":report["initial"],"final":report["final"],"learning_seconds":learning_seconds})
    );
    Ok(())
}
