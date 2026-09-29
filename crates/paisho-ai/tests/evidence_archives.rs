use std::fs;
use std::path::{Path, PathBuf};

use paisho_core::{GameOutcome, GameRecord, Player};

const REPLAYABLE_ARCHIVES: [&str; 4] = [
    "mcts-128-vs-32-2156132-block-2000",
    "mcts-128-vs-32-2156132-block-2024",
    "mcts-128-vs-32-f60452d-block-4000",
    "mcts-128-vs-32-f60452d-block-4024",
];

#[test]
fn corrected_wheel_budget_evidence_is_complete_and_replayable() {
    let results = workspace_root().join("benchmarks/results");
    for name in REPLAYABLE_ARCHIVES {
        verify_archive(&results.join(name));
    }
}

fn verify_archive(directory: &Path) {
    let games = fs::read_to_string(directory.join("games.tsv")).unwrap();
    let mut rows = games.lines();
    assert!(rows.next().unwrap().starts_with("pair_id\tstart\tleg\t"));

    let mut count = 0;
    for row in rows {
        let fields: Vec<_> = row.split('\t').collect();
        assert_eq!(fields.len(), 17, "malformed row in {}", directory.display());
        assert!(
            fields[16].is_empty() || fields[16] == "-",
            "archived match contains an error"
        );

        let record_text = fs::read_to_string(directory.join(fields[15])).unwrap();
        let record: GameRecord = record_text.parse().unwrap();
        assert_eq!(record.setup().starting_flower.code(), fields[1]);
        assert_eq!(record.actions().len(), fields[6].parse::<usize>().unwrap());

        let position = record.replay().unwrap();
        let expected = match fields[5] {
            "DECISION_LIMIT" => GameOutcome::Ongoing,
            "HOST_WIN" => GameOutcome::Win(Player::Host),
            "GUEST_WIN" => GameOutcome::Win(Player::Guest),
            "DRAW" => GameOutcome::Draw,
            other => panic!("unknown archived termination `{other}`"),
        };
        assert_eq!(
            position.outcome(),
            expected,
            "outcome mismatch for {}",
            fields[15]
        );
        count += 1;
    }
    assert_eq!(
        count,
        48,
        "unexpected game count in {}",
        directory.display()
    );
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .to_path_buf()
}
