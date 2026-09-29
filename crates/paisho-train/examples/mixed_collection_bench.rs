//! Frozen-checkpoint mixed collection; invoke via benchmark_with_training_paused.py.
use paisho_mpsgraph_client::{
    CapacityInferenceBroker, InferenceBrokerConfiguration, NetworkPreset, OptimizationLevel,
    ServiceConfiguration,
};
use paisho_replay::{ReplayDigestV1, ReplayShardV1};
use paisho_train::live_actors::{collect_live_weighted_games, LiveActorsConfiguration};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    path::PathBuf,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if !(7..=11).contains(&a.len()) {
        return Err(
            "SERVICE CHECKPOINT CLASSES WORKERS WAIT_US GAMES PREPARE_AHEAD [IN_FLIGHT [EXTERNAL_GAMES [HORIZON [LIMIT]]]]".into(),
        );
    }
    let in_flight = a
        .get(7)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let configs = a[2]
        .split(',')
        .map(|s| -> Result<_, Box<dyn Error + Send + Sync>> {
            let (capacity, batch) = s.split_once(':').ok_or("invalid class")?;
            Ok(ServiceConfiguration {
                executable: PathBuf::from(&a[0]),
                checkpoint: Some(PathBuf::from(&a[1])),
                preset: NetworkPreset::Pure,
                optimization: OptimizationLevel::Level1,
                batch_size: batch.parse()?,
                legal_action_capacity: capacity.parse()?,
                inference_slots: in_flight,
                seed: 1,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let broker = CapacityInferenceBroker::launch(
        configs,
        InferenceBrokerConfiguration {
            maximum_batch_wait: Duration::from_micros(a[4].parse()?),
            prepare_ahead: a[6].parse()?,
            maximum_in_flight_batches: in_flight,
        },
    )?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(a[3].parse()?)
        .build()?;
    let games: usize = a[5].parse()?;
    let external_games = a
        .get(8)
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(games / 2);
    let horizon = a.get(9).map(|v| v.parse()).transpose()?.unwrap_or(64);
    let limit = a.get(10).map(|v| v.parse()).transpose()?.unwrap_or(256);
    let config = LiveActorsConfiguration {
        decision_soft_limit: limit,
        neutral_start: if horizon == 0 {
            None
        } else {
            Some(paisho_train::NeutralStartConfigurationV1::new(
                0x4e45_5554_5241_4c31,
                horizon,
                16384,
                16,
            )?)
        },
        actors: 512,
        target_games: games,
        maximum_attempts: games * 4,
        ..Default::default()
    };
    let start = Instant::now();
    let report = collect_live_weighted_games(
        &config,
        external_games,
        &pool,
        broker.client()?,
        ReplayDigestV1::from_bytes([7; 32]),
    )?;
    let seconds = start.elapsed().as_secs_f64();
    let telemetry = broker.shutdown()?;
    let mut lengths: Vec<usize> = report
        .retained
        .iter()
        .map(|r| {
            r.game
                .decisions()
                .iter()
                .filter(|d| d.behavior_value().is_some())
                .count()
        })
        .collect();
    lengths.sort_unstable();
    let retained_decisions: usize = lengths.iter().sum();
    let counts = json!({"games":report.retained.len(),"retained_neural_decisions":retained_decisions,
        "median_retained_neural_decisions":lengths.get(lengths.len()/2),"external_games":external_games,"horizon":horizon,"limit":limit,"interrupted":report.interrupted,"excluded_pairs":report.excluded_pairs,"attempts":report.attempts,"reached":report.target_reached,
        "aborted":report.abort_reason,"starts_seconds":report.start_preparation_seconds,"match_seconds":report.match_seconds});
    let mut trajectories = Sha256::new();
    for result in &report.retained {
        trajectories.update(result.game.game_id().to_le_bytes());
        trajectories.update(result.game.record().to_string().as_bytes());
    }
    let trajectory_digest = ReplayDigestV1::from_bytes(trajectories.finalize().into()).to_string();
    let shard = ReplayShardV1::new(0, report.into_games())?;
    println!(
        "{}",
        json!({"seconds":seconds,"collection":counts,"shard_sha256":shard.digest()?.to_string(),
        "trajectory_sha256":trajectory_digest,"requested":telemetry.requested_positions(),"executed":telemetry.executed_positions(),
        "classes":telemetry.classes.iter().map(|c|json!({"capacity":c.legal_action_capacity,"batch":c.batch_size,
            "requests":c.broker.requested_positions,"batches":c.broker.batches,"padding":c.broker.padded_positions})).collect::<Vec<_>>() })
    );
    if counts["reached"] != true {
        return Err("incomplete collection".into());
    }
    Ok(())
}
