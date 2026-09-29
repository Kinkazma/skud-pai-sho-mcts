use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use paisho_ai::{
    CompactValueFeatures, StableRng, COMPACT_FEATURE_COUNT, COMPACT_FEATURE_NAMES,
    COMPACT_VALUE_SCHEMA_V1,
};
use paisho_core::{GameOutcome, GameRecord};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{invalid, sha256, Result};

const DATASET_SCHEMA: &str = "paisho-compact-value-dataset-v1";

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrepareOptions {
    pub max_games: usize,
    pub positions_per_game: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginalRecord {
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetExample {
    pub decision_index: usize,
    /// H or G; targets are always from this decision's player-to-move view.
    pub perspective: String,
    pub values: Vec<f64>,
    pub target: f64,
}

impl DatasetExample {
    pub fn features(&self) -> Result<CompactValueFeatures> {
        let values: [f64; COMPACT_FEATURE_COUNT] = self
            .values
            .clone()
            .try_into()
            .map_err(|_| invalid("example requires 64 features"))?;
        Ok(CompactValueFeatures::from_values(values, None)?)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalOutcome {
    pub schema: String,
    pub record_sha256: String,
    pub outcome: String,
    pub kind: String,
    pub source_records: Vec<ExternalSource>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExternalSource {
    pub game_id: u64,
    pub original_sha256: String,
    pub metadata_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetGame {
    pub game_sha256: String,
    /// Preserve the original game-level split across explicit rules migrations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_identity_sha256: Option<String>,
    pub originals: Vec<OriginalRecord>,
    pub held_out: bool,
    /// H, G, or draw: replayed terminal result, or explicitly sourced resignation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_outcome: Option<ExternalOutcome>,
    pub outcome: String,
    pub decisions: usize,
    pub examples: Vec<DatasetExample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompactDataset {
    pub schema: String,
    #[serde(default = "super::legacy_training_rules")]
    pub rules: String,
    pub feature_schema: String,
    pub feature_names: Vec<String>,
    pub source_root: PathBuf,
    pub options: PrepareOptions,
    pub scanned_files: usize,
    pub unique_records: usize,
    pub skipped_nonterminal: usize,
    pub extraction_seconds: f64,
    pub games: Vec<DatasetGame>,
}

impl CompactDataset {
    pub fn validate(&self) -> Result<()> {
        self.rules.parse::<paisho_core::RuleProfileId>()?;
        if self.schema != DATASET_SCHEMA
            || self.feature_schema != COMPACT_VALUE_SCHEMA_V1
            || self.feature_names != COMPACT_FEATURE_NAMES
        {
            return Err(invalid("dataset schema or feature order mismatch"));
        }
        validate_options(self.options)?;
        if self.games.len() > self.options.max_games
            || self.unique_records > self.scanned_files
            || self.games.len() + self.skipped_nonterminal > self.unique_records
            || !self.extraction_seconds.is_finite()
            || self.extraction_seconds < 0.0
        {
            return Err(invalid("invalid dataset counters"));
        }
        let mut ids = BTreeSet::new();
        for game in &self.games {
            if !valid_sha(&game.game_sha256)
                || !ids.insert(&game.game_sha256)
                || game.originals.is_empty()
                || game
                    .split_identity_sha256
                    .as_ref()
                    .is_some_and(|id| !valid_sha(id))
                || game.held_out
                    != held_out(
                        game.split_identity_sha256
                            .as_deref()
                            .unwrap_or(&game.game_sha256),
                    )
                || game
                    .originals
                    .iter()
                    .any(|original| !valid_sha(&original.sha256))
                || !["H", "G", "draw"].contains(&game.outcome.as_str())
                || game.examples.is_empty()
                || game.examples.len() > self.options.positions_per_game
                || game.examples.len() > game.decisions
            {
                return Err(invalid(
                    "invalid game identity, split, provenance, outcome or sample count",
                ));
            }
            if let Some(external) = &game.external_outcome {
                validate_external(external, &game.game_sha256)?;
                if external.outcome != game.outcome {
                    return Err(invalid("external result does not match dataset target"));
                }
            }
            let mut previous = None;
            for example in &game.examples {
                example.features()?;
                if example.decision_index >= game.decisions
                    || previous.is_some_and(|index| index >= example.decision_index)
                    || !["H", "G"].contains(&example.perspective.as_str())
                    || example.target != target(&game.outcome, &example.perspective)
                {
                    return Err(invalid(
                        "invalid example position, perspective or terminal target",
                    ));
                }
                previous = Some(example.decision_index);
            }
        }
        Ok(())
    }
}

pub fn load_dataset(path: &Path) -> Result<CompactDataset> {
    let dataset: CompactDataset = serde_json::from_slice(&fs::read(path)?)?;
    dataset.validate()?;
    Ok(dataset)
}

pub fn prepare_dataset(source: &Path, options: PrepareOptions) -> Result<CompactDataset> {
    validate_options(options)?;
    let started = Instant::now();
    let source_root = source.canonicalize()?;
    let mut paths = Vec::new();
    collect_psrs(&source_root, &mut paths)?;
    paths.sort();
    let scanned_files = paths.len();
    // Canonical PSR identity merges whitespace/encoding-neutral duplicates;
    // every contributing original retains its own byte hash and path.
    let mut records: BTreeMap<String, (GameRecord, Vec<OriginalRecord>)> = BTreeMap::new();
    for path in paths {
        let bytes = fs::read(&path)?;
        let text = std::str::from_utf8(&bytes)?;
        let record: GameRecord = text
            .parse()
            .map_err(|error| invalid(format!("{}: {error}", path.display())))?;
        let canonical = sha256(record.to_string().as_bytes());
        records
            .entry(canonical)
            .or_insert_with(|| (record, Vec::new()))
            .1
            .push(OriginalRecord {
                path,
                sha256: sha256(&bytes),
            });
    }
    let unique_records = records.len();
    let rules = records
        .values()
        .next()
        .map_or(paisho_core::RuleProfileId::CURRENT, |(record, _)| {
            record.rules()
        });
    if records.values().any(|(record, _)| record.rules() != rules) {
        return Err(invalid("cannot mix rule profiles in one training dataset; migrate the source PSRs into a separate corpus first"));
    }
    let mut records = records.into_iter().collect::<Vec<_>>();
    records.sort_by_key(|(identity, _)| seeded_order(options.seed, identity));
    let mut skipped_nonterminal = 0;
    let mut games = Vec::new();
    for (game_sha256, (record, originals)) in records {
        if games.len() == options.max_games {
            break;
        }
        let terminal = record
            .replay()
            .map_err(|error| invalid(format!("{}: {error}", originals[0].path.display())))?;
        let split_identity_sha256 = migration_split(&record, &game_sha256, &originals)?;
        let mut external_outcome: Option<ExternalOutcome> = None;
        for original in &originals {
            let sidecar = original.path.with_extension("outcome.json");
            if sidecar.exists() {
                let external: ExternalOutcome = serde_json::from_slice(&fs::read(sidecar)?)?;
                validate_external(&external, &game_sha256)?;
                if external_outcome
                    .as_ref()
                    .is_some_and(|previous| previous.outcome != external.outcome)
                {
                    return Err(invalid("conflicting external results for canonical game"));
                }
                external_outcome = Some(external);
            }
        }
        if external_outcome.is_some() && terminal.outcome() != GameOutcome::Ongoing {
            return Err(invalid(
                "external resignation result requires an ongoing legal PSR",
            ));
        }
        let outcome = match terminal.outcome() {
            GameOutcome::Win(player) => player.code().to_string(),
            GameOutcome::Draw => "draw".into(),
            GameOutcome::Ongoing => match &external_outcome {
                Some(external) => external.outcome.clone(),
                None => {
                    skipped_nonterminal += 1;
                    continue;
                }
            },
        };
        let decisions = record.actions().len();
        let mut indices: Vec<_> = (0..decisions).collect();
        let digest = seeded_order(options.seed ^ 0x706f736974696f6e, &game_sha256);
        let mut rng = StableRng::new(u64::from_le_bytes(
            digest[..8].try_into().expect("8 hash bytes"),
        ));
        for index in (1..indices.len()).rev() {
            let selected = rng.index(index + 1);
            indices.swap(index, selected);
        }
        indices.truncate(options.positions_per_game);
        indices.sort_unstable();
        let mut wanted = indices.into_iter().peekable();
        let mut examples = Vec::new();
        let mut position = record.initial_position();
        for (decision_index, action) in record.actions().iter().enumerate() {
            if wanted.peek() == Some(&decision_index) {
                wanted.next();
                let perspective = position.to_move().code().to_string();
                let features = CompactValueFeatures::extract(&position, position.to_move());
                if features.terminal_value().is_some() {
                    return Err(invalid("record has actions after a terminal position"));
                }
                examples.push(DatasetExample {
                    decision_index,
                    target: target(&outcome, &perspective),
                    perspective,
                    values: features.values().to_vec(),
                });
            }
            position.apply(*action)?;
        }
        games.push(DatasetGame {
            held_out: held_out(split_identity_sha256.as_deref().unwrap_or(&game_sha256)),
            split_identity_sha256,
            game_sha256,
            originals,
            outcome,
            external_outcome,
            decisions,
            examples,
        });
    }
    let dataset = CompactDataset {
        schema: DATASET_SCHEMA.into(),
        rules: rules.to_string(),
        feature_schema: COMPACT_VALUE_SCHEMA_V1.into(),
        feature_names: COMPACT_FEATURE_NAMES
            .iter()
            .map(|name| (*name).into())
            .collect(),
        source_root,
        options,
        scanned_files,
        unique_records,
        skipped_nonterminal,
        extraction_seconds: started.elapsed().as_secs_f64(),
        games,
    };
    dataset.validate()?;
    Ok(dataset)
}

fn migration_split(
    record: &GameRecord,
    identity: &str,
    originals: &[OriginalRecord],
) -> Result<Option<String>> {
    let mut split = None;
    for original in originals {
        let path = original.path.with_extension("rules-migration.json");
        if !path.exists() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        let split_id = value["split_identity_sha256"]
            .as_str()
            .ok_or_else(|| invalid("rules migration lacks split identity"))?;
        let source_id = value["source_record_sha256"]
            .as_str()
            .ok_or_else(|| invalid("rules migration lacks source identity"))?;
        let source_rules = value["source_rules"]
            .as_str()
            .ok_or_else(|| invalid("rules migration lacks source rules"))?;
        source_rules.parse::<paisho_core::RuleProfileId>()?;
        if value["schema"] != "paisho-rules-migration-v1"
            || value["target_record_sha256"] != identity
            || value["target_rules"] != record.rules().as_str()
            || value["target_decisions"].as_u64() != Some(record.actions().len() as u64)
            || !valid_sha(split_id)
            || !valid_sha(source_id)
            || split.as_ref().is_some_and(|previous| previous != split_id)
        {
            return Err(invalid("invalid or conflicting rules migration identity"));
        }
        split = Some(split_id.to_string());
    }
    Ok(split)
}

fn validate_options(options: PrepareOptions) -> Result<()> {
    if options.max_games == 0 || options.positions_per_game == 0 {
        return Err(invalid("max-games and positions-per-game must be positive"));
    }
    Ok(())
}

fn target(outcome: &str, perspective: &str) -> f64 {
    if outcome == "draw" {
        0.0
    } else if outcome == perspective {
        1.0
    } else {
        -1.0
    }
}

pub(super) fn held_out(game_sha256: &str) -> bool {
    // No seed: future corpus extensions or extraction seeds retain the same
    // game split. Approximately one fifth of independent hashes is held out.
    let digest = Sha256::digest(format!("paisho-compact-heldout-v1:{game_sha256}").as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("8 hash bytes")) % 5 == 0
}

fn seeded_order(seed: u64, identity: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(identity.as_bytes());
    hash.finalize().into()
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn collect_psrs(path: &Path, output: &mut Vec<PathBuf>) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Ok(());
    }
    if metadata.is_dir() {
        for entry in fs::read_dir(path)? {
            collect_psrs(&entry?.path(), output)?;
        }
    } else if metadata.is_file()
        && path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("psr"))
    {
        output.push(path.to_path_buf());
    }
    Ok(())
}

fn validate_external(external: &ExternalOutcome, record_sha256: &str) -> Result<()> {
    if external.schema != "paisho-external-outcome-v1"
        || external.kind != "site_resignation"
        || external.record_sha256 != record_sha256
        || !["H", "G"].contains(&external.outcome.as_str())
        || external.source_records.is_empty()
        || external.source_records.iter().any(|source| {
            source.game_id == 0
                || !valid_sha(&source.original_sha256)
                || !valid_sha(&source.metadata_sha256)
        })
    {
        return Err(invalid("invalid externally sourced resignation result"));
    }
    Ok(())
}
