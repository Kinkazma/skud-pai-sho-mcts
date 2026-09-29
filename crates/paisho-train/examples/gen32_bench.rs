//! Frozen search and exact-update ABBA measurements, never a strength claim.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::{
    compact_learning::load_model, gen32::Artifact, micro_learning::SavedMicroExample,
};
use std::{env, fs, io::Read, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    let initial = Artifact::load(std::path::Path::new(&args[1]))?.model()?;
    let learned = Artifact::load(std::path::Path::new(&args[2]))?.model()?;
    let old = load_model(std::path::Path::new(&args[3]))?.model()?;
    let record: GameRecord = fs::read_to_string(&args[4])?.parse()?;
    let mut position = record.initial_position();
    let mut positions = vec![];
    for (i, a) in record.actions().iter().enumerate() {
        if [0, 20, 40, 60, 80].contains(&i) {
            positions.push(position.clone())
        }
        position.apply(*a)?;
    }
    let mut rows = vec![];
    for workers in [1, 4, 8, 10] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        for budget in [32, 64, 128, 256, 512] {
            for repeat in 0..2 {
                for (index, p) in positions.iter().enumerate() {
                    let legal = legal_actions(p);
                    if legal.is_empty() {
                        continue;
                    }
                    let models: [(&str, &dyn MctsEvaluator); 3] = if repeat == 0 {
                        [
                            ("gen31", &old),
                            ("neutral32", &initial),
                            ("learned32", &learned),
                        ]
                    } else {
                        [
                            ("learned32", &learned),
                            ("neutral32", &initial),
                            ("gen31", &old),
                        ]
                    };
                    let mut signatures = std::collections::BTreeMap::new();
                    for (name, m) in models {
                        let mut search = MctsSession::new(
                            72 + index as u64,
                            MctsConfig {
                                simulations: budget,
                                ..Default::default()
                            },
                            m,
                        )?;
                        let start = Instant::now();
                        let r = pool.install(|| search.search_until(p, &legal, None))?;
                        let seconds = start.elapsed().as_secs_f64();
                        signatures.insert(
                            name,
                            r.actions
                                .iter()
                                .map(|a| (a.visits, a.value_sum.to_bits()))
                                .collect::<Vec<_>>(),
                        );
                        rows.push(serde_json::json!({"model":name,"workers":workers,"budget":budget,"position":index,"repeat":repeat,"seconds":seconds,"candidates":r.evaluated_actions,"selected":r.selected_index}));
                    }
                    assert_eq!(signatures["gen31"], signatures["neutral32"]);
                }
            }
        }
    }
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(fs::File::open(&args[5])?).read_to_end(&mut bytes)?;
    let saved: Vec<SavedMicroExample> = serde_json::from_slice(&bytes)?;
    let examples: Vec<_> = saved
        .iter()
        .take(16)
        .map(|s| s.example().unwrap())
        .collect();
    let mut warm = learned.clone();
    for ex in &examples {
        warm.train(ex, 0.01)?;
    }
    let mut learning = vec![];
    for mode in ["clone", "direct", "direct", "clone"] {
        let mut model = learned.clone();
        let start = Instant::now();
        for _ in 0..4 {
            for ex in &examples {
                if mode == "direct" {
                    model.train(ex, 0.01)?
                } else {
                    let mut next = model.clone();
                    next.value.train_step(
                        &CompactValueFeatures::from_values(
                            ex.state[..64].try_into().unwrap(),
                            None,
                        )?,
                        ex.value,
                        0.01,
                        0.,
                    )?;
                    let mut policy = ex.clone();
                    policy.value = next.policy.embed(&ex.state).value;
                    next.policy.train_step(&policy, 0.01, 0.)?;
                    model = next;
                }
            }
        }
        learning.push(serde_json::json!({"mode":mode,"updates":examples.len()*4,"seconds":start.elapsed().as_secs_f64(),"parameters":model.policy.parameters(),"value":model.value.weights().as_slice()}));
    }
    assert_eq!(learning[0]["parameters"], learning[1]["parameters"]);
    assert_eq!(learning[0]["value"], learning[1]["value"]);
    for row in &mut learning {
        row.as_object_mut().unwrap().remove("parameters");
        row.as_object_mut().unwrap().remove("value");
    }
    println!(
        "{}",
        serde_json::json!({"search":rows,"learning":learning,"neutral_search_exact":true,"learning_update_exact":true})
    );
    Ok(())
}
