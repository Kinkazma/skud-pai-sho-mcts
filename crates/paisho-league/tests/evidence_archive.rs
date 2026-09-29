use std::fs;
use std::path::{Path, PathBuf};

use paisho_league::verify_archive_manifest;

const SOURCE_REVISION: &str = "762aa418b97fd8b1ecdb6627fb08a952b90a0b42";

#[test]
fn first_rating_ladder_archive_is_complete_and_replayable() {
    let archive = archive_root();
    verify_archive_manifest(&archive).unwrap();

    let run = fs::read_to_string(archive.join("run.tsv")).unwrap();
    assert!(run.contains(&format!("source_revision\t{SOURCE_REVISION}\n")));
    assert!(run.contains("scheduled_games\t180\n"));
    assert!(run.contains("rated_games\t176\n"));
    assert!(run.contains("rated_pairs\t88\n"));
    assert!(run.contains("excluded_pairs\t2\n"));

    let record_count = fs::read_dir(archive.join("records"))
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_type().unwrap().is_file())
        .count();
    assert_eq!(record_count, 180);
}

fn archive_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benchmarks/results/rating-ladder-v1-762aa41-block-17100")
}
