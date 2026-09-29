use std::time::Instant;

use paisho_ai::{CompactValueFeatures, CompactValueModel, StableRng};
use serde::{Deserialize, Serialize};

use super::{invalid, CompactDataset, ModelArtifact, Result};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitOptions {
    pub epochs: usize,
    /// Zero disables early stopping; otherwise stop after this many epochs without a new validation best.
    #[serde(default)]
    pub patience: usize,
    pub learning_rate: f64,
    pub l2: f64,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct LossMetrics {
    pub train_half_squared_error: f64,
    pub held_out_half_squared_error: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct EpochMetrics {
    pub epoch: usize,
    pub loss: LossMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitReport {
    pub schema: String,
    #[serde(default = "super::legacy_training_rules")]
    pub rules: String,
    pub options: FitOptions,
    pub train_games: usize,
    pub held_out_games: usize,
    pub train_examples: usize,
    pub held_out_examples: usize,
    pub before: LossMetrics,
    pub after: LossMetrics,
    pub selected_epoch: usize,
    pub completed_epochs: usize,
    pub stop_reason: String,
    pub attempted_updates: u64,
    pub selected_updates: u64,
    pub update_seconds: f64,
    pub extraction_seconds: f64,
    pub epochs: Vec<EpochMetrics>,
}

pub fn fit_dataset(
    dataset: &CompactDataset,
    parent: &ModelArtifact,
    options: FitOptions,
) -> Result<(ModelArtifact, FitReport)> {
    dataset.validate()?;
    super::require_current_training_rules(&dataset.rules)?;
    if options.epochs == 0
        || !options.learning_rate.is_finite()
        || options.learning_rate <= 0.0
        || !options.l2.is_finite()
        || options.l2 < 0.0
    {
        return Err(invalid("invalid compact fitting options"));
    }
    let mut train = Vec::new();
    let mut held_out = Vec::new();
    for game in &dataset.games {
        let examples = if game.held_out {
            &mut held_out
        } else {
            &mut train
        };
        for example in &game.examples {
            examples.push((example.features()?, example.target));
        }
    }
    if train.is_empty() || held_out.is_empty() {
        return Err(invalid(
            "training requires nonempty, disjoint train and held-out game sets",
        ));
    }
    let mut model = parent.model()?;
    let before = metrics(&model, &train, &held_out);
    let mut best = model.clone();
    let mut best_loss = before;
    let mut selected_epoch = 0;
    let mut epochs = vec![EpochMetrics {
        epoch: 0,
        loss: before,
    }];
    let mut order: Vec<_> = (0..train.len()).collect();
    let mut rng = StableRng::new(options.seed);
    let started = Instant::now();
    for epoch in 1..=options.epochs {
        for index in (1..order.len()).rev() {
            let selected = rng.index(index + 1);
            order.swap(index, selected);
        }
        for index in &order {
            let (features, target) = &train[*index];
            model.train_step(features, *target, options.learning_rate, options.l2)?;
        }
        let loss = metrics(&model, &train, &held_out);
        // Keep the parent on ties. The holdout chooses a candidate only; it is
        // not an independent playing-strength test or a promotion decision.
        if loss.held_out_half_squared_error < best_loss.held_out_half_squared_error - 1e-12 {
            best = model.clone();
            best_loss = loss;
            selected_epoch = epoch;
        }
        epochs.push(EpochMetrics { epoch, loss });
        if options.patience > 0 && epoch - selected_epoch >= options.patience {
            break;
        }
    }
    let selected_updates = (selected_epoch as u64)
        .checked_mul(train.len() as u64)
        .ok_or_else(|| invalid("training step overflow"))?;
    let completed_epochs = epochs.len() - 1;
    let attempted_updates = (completed_epochs as u64)
        .checked_mul(train.len() as u64)
        .ok_or_else(|| invalid("training step overflow"))?;
    let total_steps = parent
        .training_steps
        .checked_add(selected_updates)
        .ok_or_else(|| invalid("training step overflow"))?;
    let report = FitReport {
        schema: "paisho-compact-fit-report-v1".into(),
        rules: dataset.rules.clone(),
        options,
        train_games: dataset.games.iter().filter(|game| !game.held_out).count(),
        held_out_games: dataset.games.iter().filter(|game| game.held_out).count(),
        train_examples: train.len(),
        held_out_examples: held_out.len(),
        before,
        after: best_loss,
        selected_epoch,
        completed_epochs,
        stop_reason: if completed_epochs < options.epochs {
            "validation_patience"
        } else {
            "epoch_budget"
        }
        .into(),
        attempted_updates,
        selected_updates,
        update_seconds: started.elapsed().as_secs_f64(),
        extraction_seconds: dataset.extraction_seconds,
        epochs,
    };
    let artifact = parent.with_model(&best, total_steps, serde_json::json!({
        "method":"terminal-value-sgd-v1", "parent":parent.provenance,"rules":dataset.rules,
        "source_root":dataset.source_root,"dataset_game_ids":dataset.games.iter().map(|game| &game.game_sha256).collect::<Vec<_>>(),
        "prepare_options":dataset.options,"fit_options":options,"selected_epoch":selected_epoch,
        "held_out_use":"epoch-selection-only-not-independent-strength-evidence"
    }));
    Ok((artifact, report))
}

fn metrics(
    model: &CompactValueModel,
    train: &[(CompactValueFeatures, f64)],
    held_out: &[(CompactValueFeatures, f64)],
) -> LossMetrics {
    fn loss(model: &CompactValueModel, examples: &[(CompactValueFeatures, f64)]) -> f64 {
        examples
            .iter()
            .map(|(features, target)| {
                let error = model.predict(features) - target;
                0.5 * error * error
            })
            .sum::<f64>()
            / examples.len() as f64
    }
    LossMetrics {
        train_half_squared_error: loss(model, train),
        held_out_half_squared_error: loss(model, held_out),
    }
}
