//! Strict conversion of original site turn notation into a legally replayed PSR.
//! No result is invented: ongoing prefixes stay ongoing (including resignations).
use paisho_core::{Action, GameOutcome, GameRecord, Position, TurnPhase};
use std::{error::Error, fs, io::Write};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn loadout(entry: &str, player: char) -> Result<String> {
    let body = entry
        .strip_prefix(&format!("0{player}."))
        .ok_or("missing setup")?;
    let mut counts = [0; 4];
    for token in body.split(',') {
        let index = match token {
            "R" => 0,
            "W" => 1,
            "K" => 2,
            "B" => 3,
            _ => return Err("unsupported accent setup".into()),
        };
        counts[index] += 1;
    }
    Ok(counts
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(","))
}
fn turn(entry: &str) -> Result<(u32, char, &str)> {
    let (header, body) = entry.split_once('.').ok_or("missing turn header")?;
    let player = header.chars().last().ok_or("empty turn header")?;
    if !['H', 'G'].contains(&player) {
        return Err("invalid player".into());
    }
    let number = header[..header.len() - 1].parse()?;
    Ok((number, player, body))
}
fn action(body: &str, bonus: bool) -> Result<Action> {
    let text = if body.starts_with('(') {
        let (from, to) = body.split_once(")-(").ok_or("invalid arrangement")?;
        format!(
            "arrange {} {}",
            from.trim_start_matches('('),
            to.strip_suffix(')').ok_or("invalid destination")?
        )
    } else {
        let (tile, coordinates) = body.split_once('(').ok_or("missing tile coordinates")?;
        if let Some((from, to)) = coordinates.split_once(")-(") {
            if tile != "B" {
                return Err("only a Boat can move another tile".into());
            }
            format!(
                "accent B move {from} {}",
                to.strip_suffix(')').ok_or("invalid Boat destination")?
            )
        } else {
            let at = coordinates
                .strip_suffix(')')
                .ok_or("invalid tile coordinate")?;
            let keyword = if ["L", "O"].contains(&tile) {
                "plant-special"
            } else if ["R", "W", "K", "B"].contains(&tile) {
                "accent"
            } else if bonus {
                "bonus-plant"
            } else {
                "plant"
            };
            if keyword == "accent" {
                format!("accent {tile} at {at}")
            } else {
                format!("{keyword} {tile} {at}")
            }
        }
    };
    Ok(text.parse()?)
}
fn convert(text: &str) -> Result<(GameRecord, Position)> {
    let entries: Vec<_> = text.trim().trim_end_matches(';').split(';').collect();
    if entries.len() < 4 {
        return Err("incomplete setup or opening".into());
    }
    let host = entries[..2]
        .iter()
        .find(|s| s.starts_with("0H."))
        .ok_or("missing host setup")?;
    let guest = entries[..2]
        .iter()
        .find(|s| s.starts_with("0G."))
        .ok_or("missing guest setup")?;
    let (n, p, opening) = turn(entries[2])?;
    if n != 1 || p != 'G' {
        return Err("invalid Guest opening".into());
    }
    let flower = opening
        .strip_suffix("(0,-8)")
        .ok_or("nonstandard Guest opening")?;
    if turn(entries[3])? != (1, 'H', format!("{flower}(0,8)").as_str()) {
        return Err("nonstandard Host opening".into());
    }
    let mut record: GameRecord = format!("PAISHO-RECORD 1\nrules skud-pai-sho-2022-03-14\nstart {flower}\nhost-accents {}\nguest-accents {}\nactions\n", loadout(host, 'H')?, loadout(guest, 'G')?).parse()?;
    let mut position = record.replay()?;
    for (index, entry) in entries[4..].iter().enumerate() {
        let (number, player, body) = turn(entry)?;
        if number != 2 + index as u32 / 2 || player != position.to_move().code() {
            return Err(format!("wrong turn or player at source entry {}", index + 5).into());
        }
        let parts: Vec<_> = body.split('+').collect();
        if parts.len() > 2 {
            return Err("multiple bonuses in one turn".into());
        }
        for (part, body) in parts.iter().enumerate() {
            let act = action(body, part == 1)?;
            position
                .apply(act)
                .map_err(|e| format!("source entry {} ({entry}), action {act}: {e}", index + 5))?;
            record.push(act);
        }
        if position.phase() == TurnPhase::HarmonyBonus && position.outcome() == GameOutcome::Ongoing
        {
            position.apply(Action::SkipHarmonyBonus)?;
            record.push(Action::SkipHarmonyBonus);
        }
    }
    Ok((record, position))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: convert_site_record INPUT.txt OUTPUT.psr".into());
    }
    let (record, position) = convert(&fs::read_to_string(&args[0])?)?;
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[1])?;
    output.write_all(record.to_string().as_bytes())?;
    let outcome = match position.outcome() {
        GameOutcome::Win(p) => p.code().to_string(),
        GameOutcome::Draw => "draw".into(),
        GameOutcome::Ongoing => "ongoing".into(),
    };
    println!(
        "{{\"outcome\":\"{outcome}\",\"decisions\":{},\"completed_turns\":{}}}",
        record.actions().len(),
        position.completed_turns()
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    const START: &str = "0H.R,W,K,B;0G.R,W,K,B;1G.R3(0,-8);1H.R3(0,8)";
    #[test]
    fn opening_is_setup_not_duplicate_plants() {
        let (record, position) = convert(START).unwrap();
        assert!(record.actions().is_empty());
        assert_eq!(position.completed_turns(), 0);
    }
    #[test]
    fn validates_player_sequence_and_legality() {
        assert!(convert(&format!("{START};2G.R4(8,0);2H.W5(-8,0)")).is_ok());
        assert!(convert(&format!("{START};2H.R4(8,0)")).is_err());
        assert!(convert(&format!("{START};2G.R4(0,0)")).is_err());
        assert!(convert(&format!("{START};3G.R4(8,0)")).is_err());
    }
    #[test]
    fn compound_formats() {
        assert_eq!(
            action("B(7,1)-(8,2)", true).unwrap().to_string(),
            "accent B move 7,1 8,2"
        );
        assert_eq!(
            action("W5(0,8)", true).unwrap().to_string(),
            "bonus-plant W5 0,8"
        );
        assert!(action("R(7,1)-(8,2)", true).is_err());
    }
}
