//! Offline CURRICULUM_V4 policy relabeling. No actor, learner or service is launched.
//! Only recorded Behavior decisions are eligible; terminal trajectories are unchanged.
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use paisho_ai::{MctsAgent, MctsConfig};
use paisho_core::{legal_actions, Position};
use paisho_model::encode_action_v1;
use paisho_replay::{
    PolicyEntryV1, PolicyTargetKindV1, PolicyTargetV1, ReplayDecisionV1, ReplayDigestV1,
    ReplayGameV1, ReplayShardReferenceV1, ReplayShardV1, ReplaySnapshotV1,
};
use rayon::prelude::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;
const HELP: &str = "paisho-teacher-relabel --source SNAPSHOT_OR_DIRECTORY [--source ...] \
--positions MAX --seed U64 --output NEW_DIRECTORY [--workers N] [--simulations 8|32]
Directories must contain snapshot.psrsnap; shard files live beside the snapshot.
Samples at most MAX recorded Behavior decisions, without replacement, using SHA-256 ranks.
Workers default to available_parallelism; N must be positive.
Independent MCTS targets (8 simulations by default; only 8 or 32 accepted) run in parallel with per-decision seeds and stable publication order.
Uses visits / total visits (temperature 1), terminal WDL values.
Writes teacher-only supervised shards, provenance.json, then snapshot.psrsnap last.
The caller owns bootstrap gating and any later mixture with other training data.
An interrupted output has no completed snapshot; retry with a new output directory.";

#[derive(Debug)]
struct Options {
    sources: Vec<PathBuf>,
    positions: usize,
    seed: u64,
    workers: usize,
    simulations: usize,
    output: PathBuf,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self> {
        let mut arguments = arguments;
        let (mut sources, mut positions, mut seed, mut output) = (Vec::new(), None, None, None);
        let mut workers = std::thread::available_parallelism().map_or(1, |count| count.get());
        let mut simulations = 8;
        while let Some(flag) = arguments.next() {
            let value = arguments
                .next()
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--source" => sources.push(PathBuf::from(value)),
                "--positions" => positions = Some(value.parse::<usize>()?),
                "--seed" => seed = Some(value.parse::<u64>()?),
                "--workers" => workers = value.parse::<usize>()?,
                "--simulations" => simulations = value.parse::<usize>()?,
                "--output" => output = Some(PathBuf::from(value)),
                _ => return Err(format!("unknown option {flag}").into()),
            }
        }
        let positions = positions.ok_or("missing --positions MAX")?;
        if positions == 0 || sources.is_empty() {
            return Err("at least one --source and positive --positions are required".into());
        }
        if workers == 0 {
            return Err("--workers must be positive".into());
        }
        if !matches!(simulations, 8 | 32) {
            return Err("--simulations must be 8 or 32".into());
        }
        Ok(Self {
            sources,
            positions,
            seed: seed.ok_or("missing --seed U64")?,
            workers,
            simulations,
            output: output.ok_or("missing --output NEW_DIRECTORY")?,
        })
    }
}

fn main() -> Result<()> {
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    if arguments.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("{HELP}");
        return Ok(());
    }
    let options = Options::parse(arguments.into_iter())?;
    run_with_workers(&options)
}

fn run_with_workers(options: &Options) -> Result<()> {
    // Independent targets and nested MCTS action ranking share one bounded pool.
    rayon::ThreadPoolBuilder::new()
        .num_threads(options.workers)
        .build()?
        .install(|| run(options))
}

fn digest(bytes: &[u8]) -> ReplayDigestV1 {
    ReplayDigestV1::from_bytes(Sha256::digest(bytes).into())
}

fn teacher_config(simulations: usize) -> MctsConfig {
    MctsConfig {
        simulations,
        ..MctsConfig::default()
    }
}

#[derive(Debug)]
struct SourceShard {
    path: PathBuf,
    reference: ReplayShardReferenceV1,
}

fn read_shard(source: &SourceShard) -> Result<ReplayShardV1> {
    let shard = ReplayShardV1::read(&source.path)?;
    let actual = ReplayShardReferenceV1::from_shard(source.reference.file_name(), &shard)?;
    if actual != source.reference {
        return Err(format!(
            "shard differs from sealed snapshot: {}",
            source.path.display()
        )
        .into());
    }
    Ok(shard)
}

// Lexicographic smallest hashes form a bounded, deterministic sample. Shard identity
// distinguishes identical game IDs across generations; repeated input shards are deduplicated.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct Selection {
    rank: ReplayDigestV1,
    shard: usize,
    game: u64,
    decision: usize,
}

fn position_hash(
    domain: &[u8],
    seed: u64,
    shard: ReplayDigestV1,
    game: u64,
    decision: usize,
) -> ReplayDigestV1 {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(seed.to_le_bytes());
    hash.update(shard.as_bytes());
    hash.update(game.to_le_bytes());
    hash.update((decision as u64).to_le_bytes());
    ReplayDigestV1::from_bytes(hash.finalize().into())
}

fn retain_sample(heap: &mut BinaryHeap<Selection>, candidate: Selection, maximum: usize) {
    heap.push(candidate);
    if heap.len() > maximum {
        heap.pop();
    }
}

fn visit_target(
    position: &Position,
    seed: u64,
    producer: ReplayDigestV1,
    simulations: usize,
) -> Result<PolicyTargetV1> {
    let actions = legal_actions(position);
    if actions.is_empty() {
        return Err("selected position has no legal actions".into());
    }
    let report = MctsAgent::new(seed, teacher_config(simulations))?.search(position, &actions);
    let total: usize = report.actions.iter().map(|action| action.visits).sum();
    if total != simulations {
        return Err(format!("MCTS-{simulations} returned {total} root visits").into());
    }
    let entries = report
        .actions
        .iter()
        .filter(|entry| entry.visits > 0)
        .map(|entry| {
            Ok(PolicyEntryV1::new(
                encode_action_v1(entry.action, position.to_move())?,
                entry.visits as f32 / total as f32,
            )?)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PolicyTargetV1::new(
        PolicyTargetKindV1::MctsVisit,
        producer,
        entries,
    )?)
}

struct TargetJob {
    game_id: u64,
    decision_index: usize,
    position: Position,
    seed: u64,
}

fn run(options: &Options) -> Result<()> {
    if options.output.exists() {
        return Err(format!("output already exists: {}", options.output.display()).into());
    }
    let mut shards = Vec::new();
    let mut sources = Vec::new();
    let mut seen_shards = HashSet::new();
    let mut sample = BinaryHeap::new();
    let mut eligible = 0_u64;
    for source in &options.sources {
        let path = if source.is_dir() {
            source.join("snapshot.psrsnap")
        } else {
            source.clone()
        };
        let path = fs::canonicalize(path)?;
        let snapshot = ReplaySnapshotV1::read(&path)?;
        let mut game_ids = HashSet::new();
        for reference in snapshot.shards() {
            let source_shard = SourceShard {
                path: path.parent().unwrap().join(reference.file_name()),
                reference: reference.clone(),
            };
            let shard = read_shard(&source_shard)?;
            for game in shard.games() {
                if !game_ids.insert(game.game_id()) {
                    return Err(
                        format!("duplicate game {} in {}", game.game_id(), path.display()).into(),
                    );
                }
            }
            if !seen_shards.insert(reference.digest()) {
                continue;
            }
            for game in shard.games() {
                for decision in game
                    .decisions()
                    .iter()
                    .filter(|decision| decision.policy().kind() == PolicyTargetKindV1::Behavior)
                {
                    eligible += 1;
                    retain_sample(
                        &mut sample,
                        Selection {
                            rank: position_hash(
                                b"PAISHO-TEACHER-SAMPLE-V1\0",
                                options.seed,
                                reference.digest(),
                                game.game_id(),
                                decision.decision_index(),
                            ),
                            shard: shards.len(),
                            game: game.game_id(),
                            decision: decision.decision_index(),
                        },
                        options.positions,
                    );
                }
            }
            shards.push(source_shard);
        }
        sources.push(json!({"path": path, "snapshot_sha256": snapshot.digest().to_string()}));
    }
    if sample.is_empty() {
        return Err("sources contain no recorded Behavior decisions".into());
    }
    let selected_count = sample.len();
    let mut groups: BTreeMap<usize, BTreeMap<u64, Vec<usize>>> = BTreeMap::new();
    for selected in sample {
        groups
            .entry(selected.shard)
            .or_default()
            .entry(selected.game)
            .or_default()
            .push(selected.decision);
    }
    let teacher = json!({
        "protocol": format!("paisho-offline-mcts{}-teacher-v1", options.simulations),
        "configuration": format!("{:?}", teacher_config(options.simulations)),
        "source_sha256": env!("PAISHO_BUILD_SOURCE_SHA256"),
        "git_revision": env!("PAISHO_BUILD_GIT_REVISION"),
        "git_dirty": env!("PAISHO_BUILD_GIT_DIRTY"),
        "binary_sha256": digest(&fs::read(std::env::current_exe()?)?).to_string(),
        "policy": "MctsVisit; root visits / total visits; temperature=1",
        "value": "original terminal outcome in current-player perspective",
    });
    let producer = digest(&serde_json::to_vec(&teacher)?);
    // Reserve a fresh directory, never overwrite an earlier corpus. Publish snapshot last.
    fs::create_dir(&options.output)?;
    let mut references = Vec::new();
    let mut positions: Vec<Value> = Vec::new();
    let mut next_game_id = 0_u64;
    for (source_index, selected_games) in groups {
        let source = &shards[source_index];
        let shard = read_shard(source)?;
        let mut selected_records = Vec::new();
        let mut jobs = Vec::new();
        for game in shard.games() {
            let Some(indices) = selected_games.get(&game.game_id()) else {
                continue;
            };
            let mut indices = indices.clone();
            indices.sort_unstable();
            let mut position = game.record().initial_position();
            for (index, &action) in game.record().actions().iter().enumerate() {
                if indices.binary_search(&index).is_ok() {
                    let seed_hash = position_hash(
                        b"PAISHO-TEACHER-SEARCH-V1\0",
                        options.seed,
                        source.reference.digest(),
                        game.game_id(),
                        index,
                    );
                    let search_seed = u64::from_le_bytes(seed_hash.as_bytes()[..8].try_into()?);
                    jobs.push(TargetJob {
                        game_id: next_game_id,
                        decision_index: index,
                        position: position.clone(),
                        seed: search_seed,
                    });
                    let original = game
                        .decisions()
                        .iter()
                        .find(|decision| decision.decision_index() == index)
                        .ok_or("selected decision disappeared")?;
                    positions.push(json!({
                        "source_shard_sha256": source.reference.digest().to_string(),
                        "source_shard_path": source.path,
                        "source_game_id": game.game_id(), "output_game_id": next_game_id,
                        "decision_index": index, "search_seed": search_seed,
                        "behavior_producer_sha256": original.policy().producer().to_string(),
                    }));
                }
                position.apply(action)?;
            }
            selected_records.push((next_game_id, game));
            next_game_id += 1;
        }
        // Indexed collection preserves shard/game/decision order regardless of scheduling.
        // Keep only one source shard's selected positions in memory at a time.
        let targets: Vec<Result<_>> = jobs
            .into_par_iter()
            .map(|job| {
                Ok((
                    job.game_id,
                    ReplayDecisionV1::new(
                        job.decision_index,
                        visit_target(&job.position, job.seed, producer, options.simulations)?,
                    ),
                ))
            })
            .collect();
        let mut decisions_by_game: BTreeMap<u64, Vec<ReplayDecisionV1>> = BTreeMap::new();
        for target in targets {
            let (game_id, decision) = target?;
            decisions_by_game.entry(game_id).or_default().push(decision);
        }
        let mut games = Vec::new();
        for (game_id, game) in selected_records {
            games.push(ReplayGameV1::new(
                game_id,
                game.host_agent(),
                game.guest_agent(),
                game.record().clone(),
                decisions_by_game
                    .remove(&game_id)
                    .ok_or("missing teacher targets")?,
            )?);
        }
        let output_shard = ReplayShardV1::new(references.len() as u64, games)?;
        let name = format!("teacher-{:06}.psrshard", references.len());
        output_shard.write_new(&options.output.join(&name))?;
        references.push(ReplayShardReferenceV1::from_shard(name, &output_shard)?);
    }
    let snapshot = ReplaySnapshotV1::new(references)?;
    let provenance = json!({
        "version": 1, "teacher": teacher, "teacher_producer_sha256": producer.to_string(),
        "cpu_threads": options.workers,
        "sources": sources, "seed": options.seed, "maximum_positions": options.positions,
        "eligible_positions": eligible, "selected_positions": selected_count,
        "sampling": "smallest SHA256(PAISHO-TEACHER-SAMPLE-V1\\0 || seed_le64 || shard_digest || game_id_le64 || decision_index_le64); deduplicate identical shards",
        "search_seed": "first 8 bytes, little endian, of SHA256 with PAISHO-TEACHER-SEARCH-V1\\0 and the same fields",
        "teacher_policy_weight": 1.0, "external_corpus_mixture": "not applied",
        "snapshot_sha256": snapshot.digest().to_string(), "positions": positions,
    });
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(options.output.join("provenance.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&provenance)?)?;
    file.sync_all()?;
    snapshot.write_new(&options.output.join("snapshot.psrsnap"))?;
    println!(
        "positions={selected_count}/{eligible} snapshot={} sha256={}",
        options.output.join("snapshot.psrsnap").display(),
        snapshot.digest()
    );
    Ok(())
}

#[cfg(test)]
#[path = "paisho_teacher_relabel/tests.rs"]
mod tests;
