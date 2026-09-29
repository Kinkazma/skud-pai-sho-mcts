//! Offline, bounded immutable bank builder. Originals stay in the packed catalog.
use paisho_ai::*;
use paisho_core::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
#[derive(Deserialize, Serialize)]
struct Source {
    path: String,
    sha256: String,
    source: String,
    human: bool,
    held_out: bool,
    external: Option<String>,
    metadata: serde_json::Value,
}
fn convert(
    s: &Source,
    game: usize,
    rules: RuleProfileId,
) -> Result<(Vec<SequenceEntry>, String), String> {
    let bytes = fs::read(&s.path).map_err(|e| e.to_string())?;
    if format!("{:x}", Sha256::digest(&bytes)) != s.sha256 {
        return Err("source hash mismatch".into());
    }
    let old: GameRecord = std::str::from_utf8(&bytes)
        .map_err(|e| e.to_string())?
        .parse()
        .map_err(|e| format!("{e}"))?;
    let (record, terminal) = old
        .replay_prefix_with_rules(rules)
        .map_err(|e| e.to_string())?;
    let result = match terminal.outcome() {
        GameOutcome::Win(p) => p.code().to_string(),
        GameOutcome::Draw => "draw".into(),
        GameOutcome::Ongoing => match &s.external {
            Some(value) if s.human => value.clone(),
            _ if rules == RuleProfileId::SkudPaiSho2022V2 => {
                return Ok((vec![], record.to_string()))
            }
            _ => return Err("no terminal result".into()),
        },
    };
    let psr = record.to_string();
    if s.held_out {
        return Ok((vec![], psr));
    }
    let mut position = record.initial_position();
    let mut entries = vec![];
    let mut active: Option<(SequenceEntry, Player, u32, [[f64; 32]; 4], [usize; 4])> = None;
    for (i, &action) in record.actions().iter().enumerate() {
        if active.as_ref().is_some_and(|(_, _, turn, _, _)| {
            position.completed_turns() - *turn >= 20 && position.phase() == TurnPhase::Main
        }) {
            let (mut e, _, _, sums, counts) = active.take().unwrap();
            e.end_decision = i as u32;
            finish(&mut e, sums, counts);
            entries.push(e);
        }
        if active.is_none() {
            let player = position.to_move();
            let outcome = if result == "draw" {
                0
            } else if result == player.code().to_string() {
                1
            } else {
                -1
            };
            active = Some((
                SequenceEntry {
                    key: sequence_key(&micro_state_features(&position)),
                    patterns: [[0; 32]; 4],
                    source: sequence_source(&s.source),
                    game: game as u32,
                    decision: i as u32,
                    end_decision: 0,
                    outcome,
                    phase: u8::from(position.phase() == TurnPhase::HarmonyBonus),
                },
                player,
                position.completed_turns(),
                [[0.; 32]; 4],
                [0; 4],
            ));
        }
        let (_, player, turn, sums, counts) = active.as_mut().unwrap();
        if position.to_move() == *player {
            let bin = ((position.completed_turns() - *turn) / 5).min(3) as usize;
            let f = micro_action_features(&position, action);
            counts[bin] += 1;
            for k in 0..32 {
                sums[bin][k] += f[k];
            }
        }
        position.apply(action).map_err(|e| e.to_string())?;
    }
    if let Some((mut e, _, _, sums, counts)) = active {
        e.end_decision = record.actions().len() as u32;
        finish(&mut e, sums, counts);
        entries.push(e);
    }
    Ok((entries, psr))
}
fn finish(e: &mut SequenceEntry, sums: [[f64; 32]; 4], counts: [usize; 4]) {
    for t in 0..4 {
        for k in 0..32 {
            e.patterns[t][k] =
                ((sums[t][k] / counts[t].max(1) as f64).clamp(-1., 1.) * 127.).round() as i8;
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let sources: Vec<Source> = serde_json::from_slice(&fs::read(&a[1])?)?;
    let out = PathBuf::from(&a[2]);
    fs::create_dir(&out)?;
    let workers: usize = a.get(3).map(String::as_str).unwrap_or("8").parse()?;
    let clusters: usize = a.get(4).map(String::as_str).unwrap_or("256").parse()?;
    if sources.is_empty() || sources.len() > 50000 {
        return Err("bank requires 1..50000 games".into());
    }
    let rules: RuleProfileId = a
        .get(5)
        .map(String::as_str)
        .unwrap_or("skud-pai-sho-gen5-v1")
        .parse()?;
    let clock = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let results: Vec<_> = pool.install(|| {
        sources
            .par_iter()
            .enumerate()
            .map(|(i, s)| convert(s, i, rules))
            .collect()
    });
    let mut entries = vec![];
    let file = fs::File::create(out.join("games.jsonl.gz"))?;
    let mut archive =
        flate2::write::GzEncoder::new(BufWriter::new(file), flate2::Compression::fast());
    for (i, (s, result)) in sources.iter().zip(results).enumerate() {
        let (rows, psr) = result.map_err(|e| format!("game {i} {}: {e}", s.path))?;
        entries.extend(rows);
        serde_json::to_writer(
            &mut archive,
            &serde_json::json!({"game":i,"source":s,"psr":psr}),
        )?;
        archive.write_all(b"\n")?;
    }
    archive.finish()?.flush()?;
    let conversion_seconds = clock.elapsed().as_secs_f64();
    let bank = pool.install(|| {
        SequenceBank::build(
            entries,
            sources.len(),
            sources.iter().filter(|s| s.human).count(),
            clusters,
        )
    });
    let mut file = BufWriter::new(fs::File::create(out.join("memory.bin"))?);
    bank.write_to(&mut file)?;
    file.flush()?;
    let bytes = fs::read(out.join("memory.bin"))?;
    let summary = serde_json::json!({"rules":rules.as_str(),"games":bank.games,"human_games":bank.human_games,"heldout_games":sources.iter().filter(|s|s.held_out).count(),"entries":bank.entries.len(),"bytes":bytes.len(),"sha256":format!("{:x}",Sha256::digest(&bytes)),"conversion_seconds":conversion_seconds,"total_seconds":clock.elapsed().as_secs_f64(),"segments":"20 completed player turns, bonus attached, final partial retained","root_only":true,"training_source_exclusion":true,"clusters":clusters});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("{summary}");
    Ok(())
}
