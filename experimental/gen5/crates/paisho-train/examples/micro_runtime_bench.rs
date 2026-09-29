//! Fixed-position, frozen-model CPU pool comparison. No strength claim.
use paisho_ai::MicroMctsSession;
use paisho_core::{GameRecord, };
use paisho_train::micro_learning::MicroArtifact;
use std::{path::PathBuf, sync::Arc, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: micro_runtime_bench MODEL.json".into());
    }
    let artifact = MicroArtifact::load(&PathBuf::from(&args[0]))?;
    let model = Arc::new(artifact.model()?);
    let record: GameRecord =
        include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_reserve_finish.psr").parse()?;
    let mut position = record.initial_position();
    let mut positions = Vec::new();
    for (i, action) in record.actions().iter().enumerate() {
        if i == record.actions().len() / 3 || i == 2 * record.actions().len() / 3 {
            positions.push(position.clone());
        }
        position.apply(*action)?;
    }
    let pools = [
        rayon::ThreadPoolBuilder::new().num_threads(1).build()?,
        rayon::ThreadPoolBuilder::new().num_threads(10).build()?,
    ];
    for (index, p) in positions.iter().enumerate() {
        for budget in [64, 512] {
            let expected = pools[0]
                .install(|| MicroMctsSession::new(model.clone()).search_until(p, budget, None))
                .map_err(std::io::Error::other)?;
            for pass in 0..20 {
                for lane in if pass % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let t = Instant::now();
                    let r = pools[lane]
                        .install(|| {
                            MicroMctsSession::new(model.clone()).search_until(p, budget, None)
                        })
                        .map_err(std::io::Error::other)?;
                    let seconds = t.elapsed().as_secs_f64();
                    assert_eq!(r.visits, expected.visits);
                    assert_eq!(r.values, expected.values);
                    assert_eq!(r.priors, expected.priors);
                    println!(
                        "{}",
                        serde_json::json!({"position":index,"budget":budget,"threads":if lane==0{1}else{10},"pass":pass,"seconds":seconds,"legal_actions":r.actions.len(),"inference_evaluations":r.inference_evaluations,"tree_bytes":r.retained_bytes,"exact_search_parity":true,"model":artifact.identity()})
                    );
                }
            }
        }
    }
    Ok(())
}
