//! Read and replay one PSR; emit machine-readable derived facts and canonical PSR.
//! No search, inference, or outcome supplied by the JSON exporter is involved.
use paisho_core::{GameOutcome, GameRecord, Player};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .ok_or("usage: verify_record INPUT.psr CANONICAL.psr")?;
    if input == "--batch" {
        if args.next().is_some() {
            return Err("--batch reads one path per stdin line".into());
        }
        use std::io::BufRead;
        for path in std::io::stdin().lock().lines() {
            verify(&path?, None)?;
        }
        return Ok(());
    }
    let output = args
        .next()
        .ok_or("usage: verify_record INPUT.psr CANONICAL.psr")?;
    if args.next().is_some() {
        return Err("usage: verify_record INPUT.psr CANONICAL.psr".into());
    }
    verify(&input, Some(&output))
}

fn verify(input: &str, output: Option<&str>) -> Result<(), Box<dyn std::error::Error>> {
    let raw = std::fs::read_to_string(input)?;
    let record: GameRecord = raw.parse()?;
    let position = record.replay()?;
    let outcome = match position.outcome() {
        GameOutcome::Win(Player::Host) => "host",
        GameOutcome::Win(Player::Guest) => "guest",
        GameOutcome::Draw => "draw",
        GameOutcome::Ongoing => "ongoing",
    };
    use std::io::Write;
    if let Some(output) = output {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?;
        file.write_all(record.to_string().as_bytes())?;
    } else if record.to_string() != raw {
        return Err(format!("noncanonical PSR: {input}").into());
    }
    let rings = paisho_core::harmony_ring_owners_for_profile(position.board(), record.rules());
    let host_ring = rings.contains(&Player::Host);
    let guest_ring = rings.contains(&Player::Guest);
    println!(
        "{{\"rules\":\"{}\",\"decisions\":{},\"completed_turns\":{},\"outcome\":\"{}\",\"host_ring\":{},\"guest_ring\":{}}}",
        record.rules(), record.actions().len(), position.completed_turns(), outcome, host_ring, guest_ring
    );
    Ok(())
}
