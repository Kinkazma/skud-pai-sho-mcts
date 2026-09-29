use super::*;
use rayon::prelude::*;
use std::{
    sync::atomic::AtomicBool,
    time::{Duration, Instant},
};
/// Frozen alternating-seat full-game comparison, no training or clock cutoff.
pub fn compare(model: &Path, parent: &Path, out: &Path, budget: usize, pairs: usize) -> Result<()> {
    compare_seed(model, parent, out, budget, pairs, 32001)
}

pub fn compare_seed(
    model: &Path,
    parent: &Path,
    out: &Path,
    budget: usize,
    pairs: usize,
    seed: u64,
) -> Result<()> {
    compare_protocol(
        model, parent, out, budget, pairs, seed, false, 0, false, None,
    )
}
/// Explicit candidate-only tactical ablation; reference search stays historical.
pub fn compare_tactics(
    model: &Path,
    parent: &Path,
    out: &Path,
    budget: usize,
    pairs: usize,
    seed: u64,
    guard: usize,
) -> Result<()> {
    compare_protocol(
        model, parent, out, budget, pairs, seed, true, guard, false, None,
    )
}
/// Weight-only comparison: both Gen3.2 snapshots use the same solver.
/// The training reference protocol remains unchanged.
pub fn compare_learning(
    model: &Path,
    parent: &Path,
    out: &Path,
    budget: usize,
    pairs: usize,
    seed: u64,
) -> Result<()> {
    Artifact::load(parent)?;
    compare_protocol(model, parent, out, budget, pairs, seed, true, 0, true, None)
}
/// Frozen human-prefix panel; never feeds these games to training.
pub fn compare_panel(
    model: &Path,
    parent: &Path,
    out: &Path,
    budget: usize,
    seed: u64,
    reference_solver: bool,
    panel: &Path,
) -> Result<()> {
    let meta: serde_json::Value = serde_json::from_slice(&fs::read(panel)?)?;
    let count = meta["cases"]
        .as_array()
        .ok_or_else(|| invalid("missing panel cases"))?
        .len();
    compare_protocol(
        model,
        parent,
        out,
        budget,
        count,
        seed,
        true,
        0,
        reference_solver,
        Some(panel),
    )
}
fn compare_protocol(
    model: &Path,
    parent: &Path,
    out: &Path,
    budget: usize,
    pairs: usize,
    seed: u64,
    solver: bool,
    guard: usize,
    reference_solver: bool,
    panel: Option<&Path>,
) -> Result<()> {
    if guard > 100_000 {
        return Err(invalid("invalid guard budget"));
    }
    if pairs == 0 || ![8, 32, 64, 128, 256, 512, 1024, 2048].contains(&budget) {
        return Err(invalid("invalid comparison"));
    }
    let panel_meta: Option<serde_json::Value> = panel
        .map(|p| -> Result<_> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?;
    let prefixes: Vec<GameRecord> = if let Some(meta) = &panel_meta {
        meta["cases"]
            .as_array()
            .ok_or_else(|| invalid("missing cases"))?
            .iter()
            .map(|case| -> Result<GameRecord> {
                let bytes = fs::read(
                    case["prefix_path"]
                        .as_str()
                        .ok_or_else(|| invalid("missing prefix path"))?,
                )?;
                if Some(sha256(&bytes).as_str()) != case["prefix_sha256"].as_str() {
                    return Err(invalid("prefix hash mismatch"));
                }
                let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
                if record.rules() != RULES || record.replay()?.outcome() != GameOutcome::Ongoing {
                    return Err(invalid("invalid prefix"));
                }
                Ok(record)
            })
            .collect::<Result<_>>()?
    } else {
        vec![]
    };
    fs::create_dir(out)?;
    let candidate_artifact = Artifact::load(model)?;
    let candidate = candidate_artifact.model()?;
    let bytes = fs::read(parent)?;
    let reference_meta: serde_json::Value = serde_json::from_slice(&bytes)?;
    let compact;
    let enriched;
    let reference: &dyn MctsEvaluator =
        if reference_meta["schema"] == "paisho-gen3-policy-memory-v1" {
            enriched = Artifact::load(parent)?.model()?;
            &enriched
        } else {
            compact = load_model(parent)?.model()?;
            &compact
        };
    let options = Options {
        solver,
        tactical_positions: guard,
        budgets: vec![budget],
        caps: vec![86400.],
        decisions: 2048,
        samples: 128,
        learn: false,
        seed,
        ..Default::default()
    };
    // Rayon defaults to all available CPUs; an explicit RAYON_NUM_THREADS
    // bounds each process when coordinating multiple independent cohorts.
    let pool = rayon::ThreadPoolBuilder::new().build()?;
    let mut rows = vec![];
    let (mut wins, mut draws, mut losses, mut unknown) = (0, 0, 0, 0);
    let played_games: Vec<std::result::Result<game::Played, String>> = pool.install(|| {
        (0..pairs * 2)
            .into_par_iter()
            .map(|index| {
                let pair = index / 2;
                let seat = index % 2;
                let game_pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(1)
                    .build()
                    .map_err(|e| e.to_string())?;
                // id/2 determines candidate seat; repeated balanced setup paired below.
                let id = pair * 4 + seat * 2;
                let played = game::play_from_record(
                    id,
                    0,
                    0,
                    candidate_artifact.updates,
                    &candidate,
                    &options,
                    Instant::now() + Duration::from_secs(86400),
                    &AtomicBool::new(false),
                    &game_pool,
                    &out.to_string_lossy(),
                    Some(reference),
                    reference_solver,
                    prefixes.get(pair),
                )?;
                fs::write(
                    out.join(format!("game-{index}.psr")),
                    played.record.to_string(),
                )
                .map_err(|e| e.to_string())?;
                Ok(played)
            })
            .collect()
    });
    for (index, played) in played_games.into_iter().enumerate() {
        let played = played.map_err(invalid)?;
        let seat = index % 2;
        let terminal = played.record.replay()?;
        let p = if seat == 0 {
            Player::Host
        } else {
            Player::Guest
        };
        match terminal.outcome() {
            GameOutcome::Win(w) => {
                if w == p {
                    wins += 1
                } else {
                    losses += 1
                }
            }
            GameOutcome::Draw => draws += 1,
            GameOutcome::Ongoing => unknown += 1,
        };
        let mut row = played.receipt;
        if let Some(meta) = &panel_meta {
            row["panel_case"] = meta["cases"][index / 2].clone();
        }
        row["opponent"] = reference_meta
            .get("generation")
            .and_then(|v| v.as_str())
            .map_or_else(|| "Gen3.1".to_string(), |g| format!("Gen{g}"))
            .into();
        rows.push(row);
    }
    atomic(
        &out.join("report.json"),
        &serde_json::json!({"panel":panel_meta,"solver":solver,"reference_solver":reference_solver,"tactical_positions":guard,"model_sha256":sha256(&fs::read(model)?),"reference_sha256":sha256(&fs::read(parent)?),"rules":RULES.as_str(),"budget":budget,"seed":seed,"pairs":pairs,"wins":wins,"draws":draws,"losses":losses,"unknown":unknown,"games":rows,"conditional_internal_elo":if wins+draws+losses>0 {Some(400.*((wins as f64+draws as f64*0.5+1.)/(losses as f64+draws as f64*0.5+1.)).log10())}else{None},"site_elo":null,"promoted":false}),
    )
}
