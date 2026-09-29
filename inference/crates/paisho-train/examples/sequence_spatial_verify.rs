//! Read-only bank audit: all anchors replayed, frozen held-out retrieval timings.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::sequence_memory_from_bytes;
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{BufRead, BufReader},
    sync::Arc,
    time::Instant,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn load(path: &str) -> Result<(Arc<SequenceBank>, f64, usize)> {
    let t = Instant::now();
    let bytes = fs::read(path)?;
    let b = sequence_memory_from_bytes(
        SequenceMemorySpec {
            path: path.into(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        },
        &bytes,
    )?;
    Ok((b, t.elapsed().as_secs_f64(), bytes.len()))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("sequence_spatial_verify OLD_BANK NEW_BANK GAMES.jsonl.gz OUTPUT.json".into());
    }
    let (old, old_load, old_bytes) = load(&a[1])?;
    let (new, new_load, new_bytes) = load(&a[2])?;
    assert!(!old.has_spatial() && new.has_spatial());
    assert_eq!(old.entries, new.entries);
    assert_eq!(old.centroids, new.centroids);
    assert_eq!(old.buckets, new.buckets);
    let mut by_game = vec![vec![]; new.games];
    for (i, e) in new.entries.iter().enumerate() {
        by_game[e.game as usize].push(i);
    }
    for indices in &mut by_game {
        indices.sort_by_key(|i| new.entries[*i].decision);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
    let t = Instant::now();
    let lines = BufReader::new(flate2::read::GzDecoder::new(fs::File::open(&a[3])?)).lines();
    let mut audited: Vec<_> = pool.install(|| {
        lines
            .enumerate()
            .par_bridge()
            .map(|(game, line)| -> std::result::Result<_, String> {
                let v: Value = serde_json::from_str(&line.map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
                if v["game"] != game {
                    return Err("catalog ordering".into());
                }
                let r: GameRecord = v["psr"]
                    .as_str()
                    .ok_or("PSR")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
                let human = v["source"]["human"] == true;
                let heldout = v["source"]["held_out"] == true;
                if heldout && !by_game[game].is_empty() {
                    return Err("held-out source entered bank".into());
                }
                let mut p = r.initial_position();
                let mut anchors = by_game[game].iter().peekable();
                let mut queries = vec![];
                for (decision, action) in r.actions().iter().enumerate() {
                    if heldout && queries.len() < 4 && decision % 20 == 0 {
                        queries.push(micro_spatial_state_features(&p));
                    }
                    while anchors
                        .peek()
                        .is_some_and(|i| new.entries[**i].decision as usize == decision)
                    {
                        let i = *anchors.next().unwrap();
                        let e = &new.entries[i];
                        // Independent board extraction, separate from the builder's feature helper.
                        let mut state = vec![0.; 417];
                        for (at, tile) in p.board().occupied() {
                            let own = if tile.owner == p.to_move() { 1. } else { -1. };
                            state[128 + (at.y() + 8) as usize * 17 + (at.x() + 8) as usize] =
                                own * (tile.kind.index() + 1) as f64 / 12.;
                        }
                        let g = SequenceGeometry::from_state(&state)?;
                        if new.geometry(i) != Some(&g)
                            || e.phase != u8::from(p.phase() == TurnPhase::HarmonyBonus)
                            || e.source
                                != sequence_source(v["source"]["source"].as_str().ok_or("source")?)
                        {
                            return Err(format!("anchor alignment {game}:{decision}"));
                        }
                    }
                    p.apply(*action).map_err(|e| e.to_string())?;
                }
                if anchors.next().is_some() {
                    return Err("unreplayed anchor".into());
                }
                Ok((game, human, heldout, by_game[game].len(), queries))
            })
            .collect()
    });
    let mut audited: Vec<_> = audited.drain(..).collect::<std::result::Result<_, _>>()?;
    audited.sort_by_key(|x| x.0);
    let replay_seconds = t.elapsed().as_secs_f64();
    let queries: Vec<_> = audited.iter().flat_map(|x| x.4.iter()).take(128).collect();
    let mut timings = vec![];
    // ABBA direct retrieval avoids cache/order bias; then measure the real cache.
    for (pass, b) in [&old, &new, &new, &old].into_iter().enumerate() {
        let spatial = b.has_spatial();
        let mut times = vec![];
        let mut counts = 0;
        for q in &queries {
            let k = sequence_key(q[..128].try_into().unwrap());
            let g = SequenceGeometry::from_state(q)?;
            let phase = u8::from(q[125] > 0.5);
            let t = Instant::now();
            let (near, n) = if spatial {
                b.nearest_spatial(&k, &g, phase, 0, 4, 8)
            } else {
                b.nearest(&k, phase, 0, 4, 8)
            };
            std::hint::black_box(near);
            times.push(t.elapsed().as_secs_f64() * 1e6);
            counts += n;
        }
        let total: f64 = times.iter().sum();
        times.sort_by(f64::total_cmp);
        timings.push(json!({"pass":pass,"spatial":spatial,"queries":times.len(),"mean_us":total/times.len() as f64,"median_us":times[times.len()/2],"p95_us":times[times.len()*95/100],"candidates":counts}));
    }
    let mut cache = vec![];
    for b in [&old, &new] {
        for pass in 0..2 {
            let t = Instant::now();
            for q in &queries {
                std::hint::black_box(b.context(q, 0));
            }
            cache.push(json!({"spatial":b.has_spatial(),"pass":pass,"queries":queries.len(),"mean_us":t.elapsed().as_secs_f64()*1e6/queries.len() as f64,"telemetry":b.telemetry()}));
        }
    }
    // Approximate bucket retrieval is disclosed; compare to a full spatial scan
    // on the same frozen queries, without fitting or changing retrieval settings.
    let mut recalls = vec![];
    for q in &queries {
        let key = sequence_key(q[..128].try_into().unwrap());
        let g = SequenceGeometry::from_state(q)?;
        let phase = u8::from(q[125] > 0.5);
        let near = new.nearest_spatial(&key, &g, phase, 0, 4, 8).0;
        let exact = new
            .nearest_spatial(&key, &g, phase, 0, new.centroids.len(), 8)
            .0;
        let hits = near
            .iter()
            .filter(|(_, i)| exact.iter().any(|(_, j)| i == j))
            .count();
        recalls.push(json!({"hits":hits,"neighbors":exact.len()}));
    }
    let summary = json!({"games":audited.len(),"human_games":audited.iter().filter(|x|x.1).count(),"heldout_games":audited.iter().filter(|x|x.2).count(),"anchors_verified":audited.iter().map(|x|x.3).sum::<usize>(),"base_entries_exact":true,"old_bytes":old_bytes,"new_bytes":new_bytes,"old_load_seconds":old_load,"new_load_seconds":new_load,"all_records_replay_seconds":replay_seconds,"heldout_queries":queries.len(),"timings":timings,"cache":cache,"spatial_neighbor_recall":recalls});
    fs::write(&a[4], serde_json::to_vec_pretty(&summary)?)?;
    println!(
        "all {} games and {} spatial anchors verified; {} held-out queries",
        audited.len(),
        new.entries.len(),
        queries.len()
    );
    Ok(())
}
