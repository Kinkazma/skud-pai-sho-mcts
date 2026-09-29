//! Fixed-corpus ABBA: disk reopening vs consuming an already resident shard.
//! Run under benchmark_with_training_paused.py. Does not train or alter replays.
use paisho_replay::{
    ReplayDatasetV1, ReplayDigestV1, ReplaySamplerV1, ReplayShardV1, ReplaySnapshotV1,
};
use std::{error::Error, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: resident_dataset_bench SNAPSHOT PRODUCER".into());
    }
    let path = PathBuf::from(&args[0]);
    let directory = path.parent().ok_or("snapshot requires parent")?;
    let snapshot = ReplaySnapshotV1::read(&path)?;
    let producer: ReplayDigestV1 = args[1].parse()?;
    // This benchmark targets the single-shard live format.
    let shard = ReplayShardV1::read(&directory.join("shard.psrbuf"))?;
    let mut reference = None;
    let mut rows = Vec::new();
    for mode in ["disk", "ram", "ram", "disk"] {
        // Live owns this object already; exclude setup cloning from both timings.
        let resident = shard.clone();
        let started = Instant::now();
        let dataset = if mode == "disk" {
            ReplayDatasetV1::from_snapshot_for_behavior(&snapshot, directory, producer)?
        } else {
            ReplayDatasetV1::from_shard_for_behavior(resident, "shard.psrbuf", producer)?
        };
        let seconds = started.elapsed().as_secs_f64();
        assert_eq!(dataset.snapshot_digest(), snapshot.digest());
        let examples = dataset.materialize_all()?;
        if let Some(expected) = &reference {
            assert_eq!(&examples, expected);
        } else {
            reference = Some(examples);
        }
        dataset.preload()?;
        let mut sampler = ReplaySamplerV1::new(&dataset, 73);
        let mut batches = Vec::new();
        for _ in 0..4 {
            let batch = sampler.prepare_batch(64)?;
            batches.push(batch.examples().to_vec());
            sampler.commit_batch(&batch)?;
        }
        rows.push((mode, seconds, dataset.len(), batches));
    }
    for row in &rows {
        assert_eq!(row.3, rows[0].3);
    }
    for (mode, seconds, examples, _) in rows {
        println!("mode={mode} seconds={seconds:.6} examples={examples} identical_examples_and_batches=true");
    }
    Ok(())
}
