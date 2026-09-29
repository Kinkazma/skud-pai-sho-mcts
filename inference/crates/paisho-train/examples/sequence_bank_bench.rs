use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::*;
use rayon::prelude::*;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Arc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let bytes = fs::read(&a[1])?;
    let spec = SequenceMemorySpec {
        path: a[1].clone(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let bank = sequence_memory_from_bytes(spec, &bytes)?;
    let sources: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(&a[2])?)?;
    let mut positions = vec![];
    for s in sources.iter().filter(|s| s["held_out"] == true).take(20) {
        let r: GameRecord = fs::read_to_string(s["path"].as_str().unwrap())?.parse()?;
        let (r, _) = r.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?;
        let mut p = r.initial_position();
        for (i, act) in r.actions().iter().enumerate() {
            if i % 10 == 0 {
                positions.push(p.clone());
            }
            p.apply(*act)?;
        }
    }
    positions.truncate(128);
    let queries: Vec<_> = positions.iter().map(micro_state_features).collect();
    let keys: Vec<_> = queries.iter().map(sequence_key).collect();
    let exact: Vec<_> = keys
        .iter()
        .zip(&queries)
        .map(|(k, q)| {
            bank.nearest(k, u8::from(q[125] > 0.5), 0, bank.centroids.len(), 8)
                .0
        })
        .collect();
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build()?;
    let mut sweeps = vec![];
    for clusters in [64, 128, 256, 512] {
        let t = Instant::now();
        let b = pool.install(|| {
            SequenceBank::build(bank.entries.clone(), bank.games, bank.human_games, clusters)
        });
        let build = t.elapsed().as_secs_f64();
        for probes in [1, 2, 4, 8, 16] {
            let t = Instant::now();
            let mut hits = 0;
            let mut total = 0;
            let mut candidates = 0;
            for ((k, q), truth) in keys.iter().zip(&queries).zip(&exact) {
                let (found, n) = b.nearest(k, u8::from(q[125] > 0.5), 0, probes, 8);
                candidates += n;
                for (_, i) in found {
                    hits += usize::from(truth.iter().any(|(_, j)| {
                        bank.entries[*j].source == b.entries[i].source
                            && bank.entries[*j].decision == b.entries[i].decision
                    }));
                }
                total += truth.len();
            }
            sweeps.push(json!({"clusters":clusters,"probes":probes,"queries":keys.len(),"seconds":t.elapsed().as_secs_f64(),"recall":hits as f64/total.max(1) as f64,"candidates":candidates,"build_seconds":build}));
        }
    }
    let mut cache = vec![];
    for pass in 0..3 {
        let t = Instant::now();
        for q in &queries {
            std::hint::black_box(bank.context(q, 0));
        }
        cache.push(
            json!({"pass":pass,"seconds":t.elapsed().as_secs_f64(),"telemetry":bank.telemetry()}),
        );
    }
    let mut concurrency = vec![];
    for workers in [1, 4, 8, 10] {
        let p = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        let t = Instant::now();
        p.install(|| {
            (0..1000).into_par_iter().for_each(|i| {
                std::hint::black_box(bank.context(&queries[i % queries.len()], 0));
            })
        });
        concurrency
            .push(json!({"workers":workers,"queries":1000,"seconds":t.elapsed().as_secs_f64()}));
    }
    let old = Arc::new(MicroArtifact::load(Path::new(&a[3]))?.model()?);
    let new = Arc::new(old.with_sequence_memory(bank.clone()));
    let mut search = vec![];
    for budget in [256, 512] {
        for pass in 0..4 {
            let mut outputs = vec![vec![], vec![]];
            for i in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
                let model = if i == 0 { old.clone() } else { new.clone() };
                let t = Instant::now();
                let rs: Vec<_> = pool.install(|| {
                    positions
                        .iter()
                        .take(32)
                        .map(|p| p.clone())
                        .collect::<Vec<_>>()
                        .par_iter()
                        .map(|p| MicroMctsSession::new(model.clone()).search_until(p, budget, None))
                        .collect()
                });
                for r in rs {
                    let r = r?;
                    outputs[i].push((r.selected_index, r.visits));
                }
                search.push(json!({"budget":budget,"pass":pass,"memory":i==1,"seconds":t.elapsed().as_secs_f64(),"positions":outputs[i].len()}));
            }
            if outputs[0] != outputs[1] {
                return Err("zero reader changed search".into());
            }
        }
    }
    let result = json!({"bank_games":bank.games,"bank_entries":bank.entries.len(),"heldout_queries":queries.len(),"sweeps":sweeps,"cache":cache,"concurrency":concurrency,"search":search,"zero_reader_search_exact":true});
    fs::write(&a[4], serde_json::to_vec_pretty(&result)?)?;
    println!("benchmark complete");
    Ok(())
}
