//! Export a local PSR highlight from a sealed generation snapshot, without inference.
use paisho_replay::{
    PolicyTargetKindV1, ReplayDigestV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1,
};
use paisho_train::{export_highlighted_game, HighlightedGamesError};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const HELP: &str = "paisho-export-games --snapshot PATH_OR_DIRECTORY --output DIRECTORY \
[--generation N] [--behavior-producer SHA256] [--replay-dir DIRECTORY]
Directory inputs must contain snapshot.psrsnap; otherwise shards live beside the snapshot.
Generation defaults to the nearest generation-N ancestor. Producer defaults to the unique
recorded Behavior producer (specify it explicitly if ambiguous).
Writes generation-{G:020}/best-game.psr and internal meta.json (not a submission payload).
G labels collection, not the subsequently trained model. Prefer win, draw, then loss;
no compatible Behavior game means no .psr. Limits: 4096 decisions / 512 KiB.
No inference, search, WordPress submission or truncation.";

fn main() -> Result<(), HighlightedGamesError> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let mut args = args.into_iter();
    let (mut snapshot_path, mut output, mut replay_dir, mut generation, mut producer) =
        (None, None, None, None, None);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--snapshot" => snapshot_path = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--replay-dir" => replay_dir = Some(PathBuf::from(value)),
            "--generation" => generation = Some(value.parse::<u64>()?),
            "--behavior-producer" => producer = Some(value.parse::<ReplayDigestV1>()?),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    let mut path = snapshot_path.ok_or("missing --snapshot")?;
    let output = output.ok_or("missing --output")?;
    if path.is_dir() {
        path = path.join("snapshot.psrsnap");
    }
    let path = std::fs::canonicalize(path)?;
    let generation = generation
        .or_else(|| infer_generation(&path))
        .ok_or("cannot infer generation; pass --generation N")?;
    let directory = replay_dir.unwrap_or_else(|| path.parent().unwrap().to_owned());
    let snapshot = ReplaySnapshotV1::read(&path)?;
    let mut games = Vec::new();
    let mut ids = HashSet::new();
    for reference in snapshot.shards() {
        let shard = ReplayShardV1::read(&directory.join(reference.file_name()))?;
        if ReplayShardReferenceV1::from_shard(reference.file_name(), &shard)? != *reference {
            return Err(format!("shard does not match snapshot: {}", reference.file_name()).into());
        }
        for game in shard.games() {
            if !ids.insert(game.game_id()) {
                return Err("snapshot contains duplicate game IDs".into());
            }
            games.push(game.clone());
        }
    }
    let producer = match producer {
        Some(producer) => producer,
        None => {
            let producers: HashSet<_> = games
                .iter()
                .flat_map(|game| game.decisions())
                .filter(|decision| decision.policy().kind() == PolicyTargetKindV1::Behavior)
                .map(|decision| decision.policy().producer())
                .collect();
            if producers.len() != 1 {
                return Err(
                    "source must have exactly one Behavior producer, or pass --behavior-producer"
                        .into(),
                );
            }
            *producers.iter().next().unwrap()
        }
    };
    let exported = export_highlighted_game(generation, producer, &games, &output)?;
    println!(
        "generation={generation} neural_wins={} excluded_oversized_wins={}",
        exported.metadata.neural_wins, exported.metadata.excluded_oversized_wins
    );
    println!(
        "psr={}",
        exported
            .psr_path
            .as_deref()
            .map_or("NONE".into(), |path| path.display().to_string())
    );
    println!("internal_metadata={}", exported.metadata_path.display());
    Ok(())
}

fn infer_generation(path: &Path) -> Option<u64> {
    path.ancestors().find_map(|path| {
        path.file_name()?
            .to_str()?
            .strip_prefix("generation-")?
            .parse()
            .ok()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn generation_inference_uses_nearest_exact_component() {
        assert_eq!(
            infer_generation(Path::new(
                "/campaign/generation-00047/actors/snapshot.psrsnap"
            )),
            Some(47)
        );
        assert_eq!(
            infer_generation(Path::new("/generation-12/generation-13/snapshot.psrsnap")),
            Some(13)
        );
        assert_eq!(
            infer_generation(Path::new("/generation-4-backup/snapshot.psrsnap")),
            None
        );
    }
}
