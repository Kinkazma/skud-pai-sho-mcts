use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use paisho_ai::prove_forced_win;
use paisho_core::{GameOutcome, GameRecord};

fn main() {
    let mut arguments = std::env::args().skip(1);
    let directory = arguments.next().map(PathBuf::from).expect(
        "usage: tactical_scan RECORD_PATH [maximum-horizon] [maximum-records] [skip-records]",
    );
    let maximum_horizon = arguments
        .next()
        .map(|text| text.parse().expect("maximum horizon must be an integer"))
        .unwrap_or(3);
    let maximum_records = arguments
        .next()
        .map(|text| text.parse().expect("maximum records must be an integer"))
        .unwrap_or(usize::MAX);
    let skip_records = arguments
        .next()
        .map(|text| text.parse().expect("skip records must be an integer"))
        .unwrap_or(0);
    assert!(arguments.next().is_none(), "too many arguments");

    let mut paths: Vec<_> = if directory.is_file() {
        vec![directory]
    } else {
        fs::read_dir(directory)
            .expect("record directory must be readable")
            .map(|entry| entry.expect("directory entry must be readable").path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "psr"))
            .collect()
    };
    paths.sort();

    let started = Instant::now();
    let mut scanned = 0;
    for path in paths.into_iter().skip(skip_records).take(maximum_records) {
        scan_record(&path, maximum_horizon);
        scanned += 1;
    }
    eprintln!(
        "scanned {scanned} records in {:.3}s",
        started.elapsed().as_secs_f64()
    );
}

fn scan_record(path: &Path, maximum_horizon: usize) {
    let text = fs::read_to_string(path).expect("record must be readable UTF-8");
    let record: GameRecord = text.parse().expect("record must parse");
    let final_position = record.replay().expect("record must replay");
    let GameOutcome::Win(winner) = final_position.outcome() else {
        return;
    };
    if !paisho_core::harmony_ring_owners_for_profile(
        final_position.board(),
        final_position.rule_profile(),
    )
    .contains(&winner)
    {
        return;
    }

    let first_prefix = record.actions().len().saturating_sub(maximum_horizon + 2);
    let mut position = record.initial_position();
    for (index, action) in record.actions().iter().copied().enumerate() {
        if index >= first_prefix && position.outcome() == GameOutcome::Ongoing {
            let mut minimum = None;
            let mut nodes = 0;
            for horizon in 1..=maximum_horizon {
                let proof = prove_forced_win(&position, winner, horizon);
                nodes += proof.visited_positions;
                if proof.is_forced_win() {
                    minimum = Some(horizon);
                    break;
                }
            }
            if let Some(horizon) = minimum.filter(|horizon| *horizon > 1) {
                println!(
                    "{} prefix={} actions={} phase={:?} legal={} to_move={:?} winner={winner:?} minimum_horizon={horizon} nodes={nodes} next={}",
                    path.display(),
                    index,
                    record.actions().len(),
                    position.phase(),
                    paisho_core::legal_actions(&position).len(),
                    position.to_move(),
                    action,
                );
            }
            if minimum.is_none() && position.to_move() != winner {
                let mut after_recorded = position.clone();
                after_recorded
                    .apply(action)
                    .expect("recorded candidate must remain legal");
                for threat_horizon in 1..maximum_horizon {
                    let proof = prove_forced_win(&after_recorded, winner, threat_horizon);
                    if proof.is_forced_win() {
                        println!(
                            "{} DEFENSIBLE prefix={} phase={:?} legal={} defender={:?} attacker={winner:?} blunder={} threat_after={} nodes={}",
                            path.display(),
                            index,
                            position.phase(),
                            paisho_core::legal_actions(&position).len(),
                            position.to_move(),
                            action,
                            threat_horizon,
                            proof.visited_positions,
                        );
                        break;
                    }
                }
            }
        }
        position
            .apply(action)
            .expect("record action must remain legal");
    }
}
