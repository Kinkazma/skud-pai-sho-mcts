use std::error::Error;
use std::path::PathBuf;

use paisho_replay::ReplayDigestV1;
use paisho_train::{
    analyze_behavior_replay_v1, BehaviorReplayStatisticsV1, ReplayClassStatisticsV1,
};

fn main() -> Result<(), Box<dyn Error>> {
    let arguments = Arguments::parse(std::env::args().skip(1))?;
    let statistics = analyze_behavior_replay_v1(
        &arguments.snapshot,
        &arguments.replay_directory,
        arguments.behavior_producer,
    )?;
    print_statistics(&statistics);
    Ok(())
}

struct Arguments {
    snapshot: PathBuf,
    replay_directory: PathBuf,
    behavior_producer: ReplayDigestV1,
}

impl Arguments {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut snapshot = None;
        let mut replay_directory = None;
        let mut behavior_producer = None;
        let mut arguments = arguments.peekable();
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value after {flag}"))?;
            match flag.as_str() {
                "--snapshot" => snapshot = Some(PathBuf::from(value)),
                "--replay-dir" => replay_directory = Some(PathBuf::from(value)),
                "--behavior-producer" => behavior_producer = Some(value.parse::<ReplayDigestV1>()?),
                _ => return Err(format!("unknown argument {flag}").into()),
            }
        }
        Ok(Self {
            snapshot: snapshot.ok_or("missing --snapshot PATH")?,
            replay_directory: replay_directory.ok_or("missing --replay-dir DIR")?,
            behavior_producer: behavior_producer.ok_or("missing --behavior-producer SHA256")?,
        })
    }
}

fn print_statistics(statistics: &BehaviorReplayStatisticsV1) {
    println!("snapshot_sha256={}", statistics.snapshot_sha256);
    println!("behavior_producer={}", statistics.behavior_producer);
    println!("games={}", statistics.games);
    println!("examples={}", statistics.examples);
    println!(
        "behavior_probability_minimum={:.9}",
        statistics.behavior_probability.minimum
    );
    println!(
        "behavior_probability_mean={:.9}",
        statistics.behavior_probability.mean
    );
    println!(
        "behavior_probability_maximum={:.9}",
        statistics.behavior_probability.maximum
    );
    println!(
        "legal_actions_minimum={:.3}",
        statistics.legal_actions.minimum
    );
    println!("legal_actions_mean={:.3}", statistics.legal_actions.mean);
    println!(
        "legal_actions_maximum={:.3}",
        statistics.legal_actions.maximum
    );
    print_class("win", statistics.wins);
    print_class("draw", statistics.draws);
    print_class("loss", statistics.losses);
}

fn print_class(name: &str, statistics: ReplayClassStatisticsV1) {
    println!("{name}_games={}", statistics.games);
    println!("{name}_examples={}", statistics.examples);
    print_optional(
        &format!("{name}_mean_examples_per_game"),
        statistics.mean_examples_per_game,
    );
    print_optional(
        &format!("{name}_mean_behavior_probability"),
        statistics.mean_behavior_probability,
    );
    print_optional(
        &format!("{name}_mean_legal_actions"),
        statistics.mean_legal_actions,
    );
}

fn print_optional(name: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("{name}={value:.9}"),
        None => println!("{name}=none"),
    }
}
