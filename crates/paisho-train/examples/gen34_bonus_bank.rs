//! Extend a verified V2 bank with bounded phase-correct bonus anchors.
//! Main entries and the complete original catalog are retained unchanged.
use paisho_ai::*;
use paisho_core::*;
use rayon::prelude::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
#[derive(Deserialize)]
struct Source {
    source: String,
    human: bool,
    held_out: bool,
    external: Option<String>,
}
#[derive(Deserialize)]
struct Row {
    game: u32,
    source: Source,
    psr: String,
}
fn family(a: Action) -> usize {
    match a {
        Action::SkipHarmonyBonus => 0,
        Action::BonusPlantBasic { .. } => 1,
        Action::PlantSpecial {
            flower: SpecialFlower::WhiteLotus,
            ..
        } => 2,
        Action::PlantSpecial {
            flower: SpecialFlower::Orchid,
            ..
        } => 3,
        Action::PlayAccent {
            accent: Accent::Wheel,
            ..
        } => 4,
        Action::PlayAccent {
            accent: Accent::Boat,
            ..
        } => 5,
        Action::PlayAccent {
            accent: Accent::Rock,
            ..
        } => 6,
        Action::PlayAccent {
            accent: Accent::Knotweed,
            ..
        } => 7,
        _ => unreachable!("main action in bonus phase"),
    }
}
fn convert(r: &Row) -> Result<Vec<SequenceEntry>, String> {
    let record: GameRecord = r.psr.parse().map_err(|e| format!("{e}"))?;
    let end = record.replay().map_err(|e| e.to_string())?;
    if record.initial_position().rule_profile() != RuleProfileId::SkudPaiSho2022V2 {
        return Err("wrong rules".into());
    }
    let result = match end.outcome() {
        GameOutcome::Win(p) => p.code().to_string(),
        GameOutcome::Draw => "draw".into(),
        GameOutcome::Ongoing => match &r.source.external {
            Some(x) if r.source.human => x.clone(),
            _ => return Ok(vec![]),
        },
    };
    if r.source.held_out {
        return Ok(vec![]);
    }
    if !["H", "G", "draw"].contains(&result.as_str()) {
        return Err("invalid result".into());
    }
    let source = sequence_source(&r.source.source);
    let mut p = record.initial_position();
    // Deterministic min-hash reservoir, one anchor per played bonus family/game.
    // No preference for a favorable outcome or late/early decision.
    let mut kept: [Option<(u64, SequenceEntry)>; 8] = std::array::from_fn(|_| None);
    for (i, &a) in record.actions().iter().enumerate() {
        if p.phase() == TurnPhase::HarmonyBonus {
            let kind = family(a);
            let mut h = Sha256::new();
            h.update(source.to_le_bytes());
            h.update((i as u64).to_le_bytes());
            let priority = u64::from_le_bytes(h.finalize()[..8].try_into().unwrap());
            if kept[kind].as_ref().map_or(true, |(old, _)| priority < *old) {
                let mut patterns = [[0; 32]; 4];
                patterns[0] =
                    micro_action_features(&p, a).map(|x| (x.clamp(-1., 1.) * 127.).round() as i8);
                kept[kind] = Some((
                    priority,
                    SequenceEntry {
                        key: sequence_key(&micro_state_features(&p)),
                        patterns,
                        source,
                        game: r.game,
                        decision: i as u32,
                        end_decision: i as u32 + 1,
                        outcome: if result == "draw" {
                            0
                        } else if result == p.to_move().code().to_string() {
                            1
                        } else {
                            -1
                        },
                        phase: 1,
                    },
                ));
            }
        }
        p.apply(a).map_err(|e| e.to_string())?;
    }
    Ok(kept.into_iter().flatten().map(|(_, e)| e).collect())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("usage: gen34_bonus_bank OLD_BANK_DIR OUTPUT_DIR".into());
    }
    let old = PathBuf::from(&a[1]);
    let out = PathBuf::from(&a[2]);
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(old.join("summary.json"))?)?;
    if meta["rules"] != "skud-pai-sho-2022-03-14-v2" {
        return Err("requires V2 source bank".into());
    }
    let bytes = fs::read(old.join("memory.bin"))?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if meta["sha256"] != hash {
        return Err("old bank hash mismatch".into());
    }
    let bank = SequenceBank::read_from(
        &mut bytes.as_slice(),
        SequenceMemorySpec {
            path: old.join("memory.bin").display().to_string(),
            sha256: hash.clone(),
        },
    )?;
    let clock = Instant::now();
    let file = BufReader::new(flate2::read::GzDecoder::new(fs::File::open(
        old.join("games.jsonl.gz"),
    )?));
    let rows: Vec<Row> = file
        .lines()
        .map(|l| Ok(serde_json::from_str(&l?)?))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    if rows.len() != bank.games || rows.iter().enumerate().any(|(i, r)| r.game as usize != i) {
        return Err("catalog identity mismatch".into());
    }
    fs::create_dir(&out)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build()?;
    let extras = pool.install(|| rows.par_iter().map(convert).collect::<Result<Vec<_>, _>>())?;
    let added = extras.iter().map(Vec::len).sum::<usize>();
    let main_count = bank.entries.len();
    let mut entries = bank.entries;
    entries.extend(extras.into_iter().flatten());
    let conversion_seconds = clock.elapsed().as_secs_f64();
    let new = pool.install(|| SequenceBank::build(entries, bank.games, bank.human_games, 256));
    let mut writer = BufWriter::new(fs::File::create(out.join("memory.bin"))?);
    new.write_to(&mut writer)?;
    writer.flush()?;
    fs::copy(old.join("games.jsonl.gz"), out.join("games.jsonl.gz"))?;
    let bytes = fs::read(out.join("memory.bin"))?;
    let summary = serde_json::json!({"rules":meta["rules"],"games":bank.games,"human_games":bank.human_games,"heldout_games":meta["heldout_games"],"entries":new.entries.len(),"main_entries_preserved":main_count,"bonus_entries":added,"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes)),"parent_sha256":hash,"catalog_sha256":format!("{:x}",Sha256::digest(fs::read(out.join("games.jsonl.gz"))?)),"segments":"original 20-turn main entries plus one-decision bonus anchors; deterministic reservoir <=1 per played bonus family per game","root_only":false,"training_source_exclusion":true,"conversion_seconds":conversion_seconds,"total_seconds":clock.elapsed().as_secs_f64()});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchors_replay_in_bonus_phase_and_heldout_remains_absent() {
        let psr = include_str!("../../paisho-ai/tests/fixtures/gen34_bonus.psr");
        let mut record: GameRecord = psr.parse().unwrap();
        record.push(Action::SkipHarmonyBonus);
        let mut row = Row {
            game: 0,
            source: Source {
                source: "human/test".into(),
                human: true,
                held_out: false,
                external: Some("H".into()),
            },
            psr: record.to_string(),
        };
        let entries = convert(&row).unwrap();
        assert!(!entries.is_empty());
        assert_eq!(entries, convert(&row).unwrap());
        assert!(entries.len() <= 8);
        for e in entries {
            assert_eq!(e.phase, 1);
            assert_eq!(e.end_decision, e.decision + 1);
            let mut p = record.initial_position();
            for &a in &record.actions()[..e.decision as usize] {
                p.apply(a).unwrap();
            }
            assert_eq!(p.phase(), TurnPhase::HarmonyBonus);
        }
        row.source.held_out = true;
        assert!(convert(&row).unwrap().is_empty());
        row.source.held_out = false;
        row.source.human = false;
        assert!(convert(&row).unwrap().is_empty());
    }
}
