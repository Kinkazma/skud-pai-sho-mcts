use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use paisho_ai::{CompactValueFeatures, COMPACT_FEATURE_COUNT};
use paisho_core::{BasicFlower, GameRecord, StandardSetup};

use super::*;

const RING: &str = include_str!("../../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
const RESERVE: &str =
    include_str!("../../../paisho-ai/tests/fixtures/site_bot_v1_reserve_finish.psr");

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "paisho-compact-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn fixture_dataset() -> (Temp, CompactDataset) {
    let dir = Temp::new();
    fs::create_dir(dir.0.join("nested")).unwrap();
    let current = |text: &str| {
        text.parse::<GameRecord>()
            .unwrap()
            .replay_prefix_with_rules(paisho_core::RuleProfileId::CURRENT)
            .unwrap()
            .0
            .to_string()
    };
    let ring = current(RING);
    fs::write(dir.0.join("ring.psr"), &ring).unwrap();
    fs::write(dir.0.join("nested/duplicate.psr"), format!("{ring}\n")).unwrap();
    fs::write(dir.0.join("reserve.psr"), current(RESERVE)).unwrap();
    fs::write(
        dir.0.join("nonterminal.psr"),
        GameRecord::new(StandardSetup::balanced(BasicFlower::Red3)).to_string(),
    )
    .unwrap();
    let dataset = prepare_dataset(
        &dir.0,
        PrepareOptions {
            max_games: 10,
            positions_per_game: 32,
            seed: 19,
        },
    )
    .unwrap();
    (dir, dataset)
}

#[test]
fn historical_datasets_remain_readable_but_cannot_supply_current_targets() {
    let dir = Temp::new();
    fs::write(dir.0.join("ring.psr"), RING).unwrap();
    let data = prepare_dataset(
        &dir.0,
        PrepareOptions {
            max_games: 2,
            positions_per_game: 8,
            seed: 1,
        },
    )
    .unwrap();
    assert_eq!(
        data.rules,
        paisho_core::RuleProfileId::SkudPaiSho2022.as_str()
    );
    let mut old = serde_json::to_value(&data).unwrap();
    old.as_object_mut().unwrap().remove("rules");
    let old: CompactDataset = serde_json::from_value(old).unwrap();
    old.validate().unwrap();
    let error = fit_dataset(
        &old,
        &ModelArtifact::legacy(),
        FitOptions {
            epochs: 1,
            patience: 0,
            learning_rate: 0.01,
            l2: 0.0,
            seed: 1,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("current training requires"));
    fs::write(
        dir.0.join("current.psr"),
        GameRecord::new(StandardSetup::balanced(BasicFlower::Red3)).to_string(),
    )
    .unwrap();
    assert!(prepare_dataset(&dir.0, data.options)
        .unwrap_err()
        .to_string()
        .contains("cannot mix rule profiles"));
}

#[test]
fn migrated_game_preserves_original_split_and_rejects_conflicting_receipts() {
    let dir = Temp::new();
    let source: GameRecord = RING.parse().unwrap();
    let current = source
        .replay_prefix_with_rules(paisho_core::RuleProfileId::CURRENT)
        .unwrap()
        .0;
    let source_hash = sha256(source.to_string().as_bytes());
    let current_hash = sha256(current.to_string().as_bytes());
    fs::write(dir.0.join("ring.psr"), current.to_string()).unwrap();
    let receipt = serde_json::json!({"schema":"paisho-rules-migration-v1", "source_record_sha256":source_hash,
        "target_record_sha256":current_hash,"split_identity_sha256":source_hash,"source_rules":source.rules().as_str(),
        "target_rules":current.rules().as_str(),"target_decisions":current.actions().len()});
    fs::write(dir.0.join("ring.rules-migration.json"), receipt.to_string()).unwrap();
    let options = PrepareOptions {
        max_games: 2,
        positions_per_game: 8,
        seed: 1,
    };
    let data = prepare_dataset(&dir.0, options).unwrap();
    assert_eq!(
        data.games[0].split_identity_sha256.as_deref(),
        Some(source_hash.as_str())
    );
    assert_eq!(data.games[0].held_out, dataset::held_out(&source_hash));
    let mut wrong = receipt;
    wrong["target_record_sha256"] = serde_json::json!("0".repeat(64));
    fs::write(dir.0.join("ring.rules-migration.json"), wrong.to_string()).unwrap();
    assert!(prepare_dataset(&dir.0, options).is_err());
}

#[test]
fn preparation_deduplicates_full_games_and_replays_sampled_decisions() {
    let (dir, dataset) = fixture_dataset();
    assert_eq!(dataset.scanned_files, 4);
    assert_eq!(dataset.unique_records, 3);
    assert_eq!(dataset.skipped_nonterminal, 1);
    assert_eq!(dataset.games.len(), 2);
    assert_eq!(
        dataset
            .games
            .iter()
            .map(|game| game.originals.len())
            .sum::<usize>(),
        3
    );
    for game in &dataset.games {
        let record: GameRecord = fs::read_to_string(&game.originals[0].path)
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(game.game_sha256, sha256(record.to_string().as_bytes()));
        for original in &game.originals {
            assert_eq!(original.sha256, sha256(&fs::read(&original.path).unwrap()));
        }
        let mut position = record.initial_position();
        for (index, action) in record.actions().iter().enumerate() {
            if let Some(example) = game
                .examples
                .iter()
                .find(|example| example.decision_index == index)
            {
                assert_eq!(
                    example.values,
                    CompactValueFeatures::extract(&position, position.to_move())
                        .values()
                        .as_slice()
                );
                assert_eq!(example.perspective, position.to_move().code().to_string());
            }
            position.apply(*action).unwrap();
        }
    }
    let repeated = prepare_dataset(&dir.0, dataset.options).unwrap();
    assert_eq!(
        serde_json::to_value(&dataset.games).unwrap(),
        serde_json::to_value(repeated.games).unwrap()
    );
    let changed_seed = prepare_dataset(
        &dir.0,
        PrepareOptions {
            seed: 99,
            ..dataset.options
        },
    )
    .unwrap();
    for game in &dataset.games {
        assert_eq!(
            game.held_out,
            changed_seed
                .games
                .iter()
                .find(|other| other.game_sha256 == game.game_sha256)
                .unwrap()
                .held_out
        );
    }
}

#[test]
fn dataset_reader_rejects_split_target_and_feature_corruption() {
    let (dir, dataset) = fixture_dataset();
    let path = dir.0.join("dataset.json");
    save_json_new(&path, &dataset).unwrap();
    load_dataset(&path).unwrap();
    let mut corrupt = dataset.clone();
    corrupt.games[0].held_out = !corrupt.games[0].held_out;
    assert!(corrupt.validate().is_err());
    corrupt = dataset.clone();
    corrupt.games[0].examples[0].target = 0.1;
    assert!(corrupt.validate().is_err());
    corrupt = dataset.clone();
    corrupt.games[0].examples[0].values[0] = f64::NAN;
    assert!(corrupt.validate().is_err());
    corrupt = dataset.clone();
    corrupt.games.push(corrupt.games[0].clone());
    assert!(corrupt.validate().is_err());
    corrupt = dataset;
    corrupt.feature_names.swap(0, 1);
    fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
    assert!(load_dataset(&path).is_err());
}

#[test]
fn model_roundtrip_preserves_weights_and_rejects_schema_and_overwrite() {
    let dir = Temp::new();
    let path = dir.0.join("model.json");
    let artifact = ModelArtifact::legacy();
    save_model_new(&path, &artifact).unwrap();
    assert_eq!(
        load_model(&path).unwrap().model().unwrap(),
        artifact.model().unwrap()
    );
    assert!(save_model_new(&path, &artifact).is_err());
    let mut corrupt = artifact.clone();
    corrupt.weights.pop();
    assert!(corrupt.model().is_err());
    corrupt = artifact.clone();
    corrupt.schema.push_str("unknown");
    assert!(corrupt.model().is_err());
    corrupt = artifact;
    corrupt.feature_names.swap(0, 1);
    fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
    assert!(load_model(&path).is_err());
}

#[test]
fn fitting_uses_only_training_games_and_keeps_best_held_out_epoch() {
    let (_dir, mut dataset) = fixture_dataset();
    // Synthetic, separable cached examples isolate the fitter from game skill.
    // Distinct game identities force one complete game into each partition.
    for (index, game) in dataset.games.iter_mut().enumerate() {
        game.game_sha256 = (0_u64..)
            .map(|ordinal| sha256(format!("synthetic-{index}-{ordinal}").as_bytes()))
            .find(|identity| dataset::held_out(identity) == (index == 1))
            .unwrap();
        game.held_out = index == 1;
        game.outcome = "H".into();
        game.decisions = 2;
        game.examples = [-1.0, 1.0]
            .into_iter()
            .enumerate()
            .map(|(decision_index, sign)| {
                let mut values = vec![0.0; COMPACT_FEATURE_COUNT];
                values[42] = sign * if index == 0 { 0.8 } else { 0.5 };
                DatasetExample {
                    decision_index,
                    perspective: if sign > 0.0 { "H" } else { "G" }.into(),
                    values,
                    target: sign,
                }
            })
            .collect();
    }
    let options = FitOptions {
        epochs: 100,
        patience: 0,
        learning_rate: 0.5,
        l2: 0.0,
        seed: 5,
    };
    let (model, report) = fit_dataset(&dataset, &ModelArtifact::legacy(), options).unwrap();
    assert_eq!(report.train_games, 1);
    assert_eq!(report.held_out_games, 1);
    assert_eq!(report.train_examples, 2);
    assert_eq!(report.attempted_updates, 200);
    assert_eq!(model.training_steps, report.selected_updates);
    assert!(
        report.after.held_out_half_squared_error < report.before.held_out_half_squared_error * 0.15
    );
    assert!(report
        .epochs
        .iter()
        .all(|epoch| epoch.loss.held_out_half_squared_error
            >= report.after.held_out_half_squared_error - 1e-12));
    // Deliberately contradictory holdout labels: a model that learns the
    // training relation must be rejected in favor of unchanged epoch zero.
    dataset.games[1].outcome = "G".into();
    for example in &mut dataset.games[1].examples {
        example.target = -example.target;
    }
    let (retained, report) = fit_dataset(&dataset, &ModelArtifact::legacy(), options).unwrap();
    assert_eq!(report.selected_epoch, 0);
    assert_eq!(report.selected_updates, 0);
    assert_eq!(retained.model().unwrap(), CompactValueModel::default());
}

#[test]
fn external_resignation_keeps_psr_ongoing_and_checks_hash() {
    let dir = Temp::new();
    let mut record = GameRecord::new(StandardSetup::balanced(BasicFlower::Red3));
    record.push("plant R4 8,0".parse().unwrap());
    let raw = record.to_string();
    let identity = sha256(raw.as_bytes());
    fs::write(dir.0.join("resigned.psr"), &raw).unwrap();
    let mut result = serde_json::json!({"schema":"paisho-external-outcome-v1", "record_sha256":identity,
        "outcome":"H", "kind":"site_resignation", "source_records":[{"game_id":1,
        "original_sha256":identity,"metadata_sha256":identity}]});
    fs::write(dir.0.join("resigned.outcome.json"), result.to_string()).unwrap();
    let options = PrepareOptions {
        max_games: 10,
        positions_per_game: 1024,
        seed: 1,
    };
    let dataset = prepare_dataset(&dir.0, options).unwrap();
    assert_eq!(dataset.games.len(), 1);
    assert_eq!(dataset.games[0].examples[0].target, -1.0); // Guest move; Host wins externally.
    assert!(dataset.games[0].external_outcome.is_some());
    assert_eq!(
        record.replay().unwrap().outcome(),
        paisho_core::GameOutcome::Ongoing
    );
    result["record_sha256"] = serde_json::json!("0".repeat(64));
    fs::write(dir.0.join("resigned.outcome.json"), result.to_string()).unwrap();
    assert!(prepare_dataset(&dir.0, options).is_err());
}

#[test]
fn fitting_patience_counts_only_executed_epochs_and_retains_parent_on_ties() {
    let (_dir, mut dataset) = fixture_dataset();
    for (index, game) in dataset.games.iter_mut().enumerate() {
        game.game_sha256 = (0_u64..)
            .map(|n| sha256(format!("patience-{index}-{n}").as_bytes()))
            .find(|identity| dataset::held_out(identity) == (index == 1))
            .unwrap();
        game.held_out = index == 1;
        game.outcome = "draw".into();
        for example in &mut game.examples {
            example.values.fill(0.0);
            example.target = 0.0;
        }
    }
    let parent = ModelArtifact::legacy();
    let (selected, report) = fit_dataset(
        &dataset,
        &parent,
        FitOptions {
            epochs: 1000,
            patience: 3,
            learning_rate: 0.1,
            l2: 0.0,
            seed: 1,
        },
    )
    .unwrap();
    assert_eq!(report.completed_epochs, 3);
    assert_eq!(report.stop_reason, "validation_patience");
    assert_eq!(report.selected_epoch, 0);
    assert_eq!(report.attempted_updates, 3 * report.train_examples as u64);
    assert_eq!(selected.weights, parent.weights);
    assert_eq!(selected.training_steps, parent.training_steps);
}
