//! Paused-campaign ABBA of round barriers using complete live games and frozen weights.
use paisho_mpsgraph_client::{
    CapacityInferenceBroker, InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel,
    ServiceConfiguration,
};
use paisho_replay::{ReplayDigestV1, ReplayShardV1};
use paisho_train::{collect_live_games, LiveActorsConfiguration};
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: live_round_bench SERVICE CHECKPOINT".into());
    }
    let broker = CapacityInferenceBroker::launch(
        [(64, 8), (128, 4), (1024, 4)]
            .into_iter()
            .map(|(capacity, batch)| ServiceConfiguration {
                executable: PathBuf::from(&args[0]),
                checkpoint: Some(PathBuf::from(&args[1])),
                preset: NetworkPreset::Pure,
                optimization: OptimizationLevel::Level1,
                batch_size: batch,
                legal_action_capacity: capacity,
                inference_slots: 1,
                seed: 1,
            })
            .collect(),
        InferenceBrokerConfiguration::default(),
    )?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(20).build()?;
    let producer = ReplayDigestV1::from_bytes([7; 32]); // Diagnostic identity, never published.
                                                        // Warm all used execution shapes before either measured case.
    let warmup = LiveActorsConfiguration {
        actors: 20,
        target_games: 20,
        ..Default::default()
    };
    let warm = collect_live_games(&warmup, &pool, broker.client()?, producer)?;
    if !warm.target_reached {
        return Err("warmup did not finish".into());
    }
    let mut reference = None;
    for actors in [80, 512, 512, 80] {
        let config = LiveActorsConfiguration {
            actors,
            ..Default::default()
        };
        let timer = Instant::now();
        let report = collect_live_games(&config, &pool, broker.client()?, producer)?;
        let seconds = timer.elapsed().as_secs_f64();
        if !report.target_reached {
            return Err(format!("collection failed: {:?}", report.abort_reason).into());
        }
        println!("actors={actors} games={} attempts={} seconds={seconds:.6} starts_s={:.6} matches_s={:.6}",
            report.retained.len(), report.attempts, report.start_preparation_seconds, report.match_seconds);
        let digest = ReplayShardV1::new(0, report.into_games())?.digest()?;
        if let Some(expected) = reference {
            assert_eq!(digest, expected, "different replay corpus");
        } else {
            reference = Some(digest);
        }
        println!("shard_sha256={digest} identical=true");
    }
    broker.shutdown()?;
    Ok(())
}
