//! Explicit PSR migration. Original profile is replayed first; the target keeps
//! only actions up to its first terminal state. Both outputs are create-new.
use paisho_core::{GameOutcome, GameRecord, RuleProfileId};
use std::{error::Error, fs, io::Write, path::Path};

fn outcome(value: GameOutcome) -> String {
    match value {
        GameOutcome::Win(player) => player.code().to_string(),
        GameOutcome::Draw => "draw".into(),
        GameOutcome::Ongoing => "ongoing".into(),
    }
}

fn write_new(path: &Path, text: &str) -> Result<(), Box<dyn Error>> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(text.as_bytes())?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: revalidate_record INPUT.psr TARGET.psr CANONICAL_SOURCE.psr".into());
    }
    if Path::new(&args[1]).exists() || Path::new(&args[2]).exists() || args[1] == args[2] {
        return Err("outputs must be distinct new files".into());
    }
    let original: GameRecord = fs::read_to_string(&args[0])?.parse()?;
    let original_position = original.replay()?;
    let (target, position) = original.replay_prefix_with_rules(RuleProfileId::CURRENT)?;
    write_new(Path::new(&args[2]), &original.to_string())?;
    write_new(Path::new(&args[1]), &target.to_string())?;
    println!(
        "{{\"source_rules\":\"{}\",\"target_rules\":\"{}\",\"source_decisions\":{},\"target_decisions\":{},\"discarded_decisions\":{},\"source_outcome\":\"{}\",\"target_outcome\":\"{}\"}}",
        original.rules(), target.rules(), original.actions().len(), target.actions().len(),
        original.actions().len()-target.actions().len(), outcome(original_position.outcome()), outcome(position.outcome())
    );
    Ok(())
}
