use super::*;
use crate::compact_learning::{load_dataset, DatasetGame};
use paisho_core::{legal_actions, GameRecord};
use rayon::prelude::*;
use std::time::Instant;

struct HumanGame {
    id: String,
    held_out: bool,
    examples: Vec<MicroExample>,
}
fn convert(game: &DatasetGame) -> Result<HumanGame> {
    let original = &game.originals[0];
    let bytes = fs::read(&original.path)?;
    if sha256(&bytes) != original.sha256 {
        return Err(invalid("human source bytes changed"));
    }
    let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
    if sha256(record.to_string().as_bytes()) != game.game_sha256
        || record.actions().len() != game.decisions
    {
        return Err(invalid("human canonical record mismatch"));
    }
    let terminal = record.replay()?;
    let outcome = match terminal.outcome() {
        paisho_core::GameOutcome::Win(p) => p.code().to_string(),
        paisho_core::GameOutcome::Draw => "draw".into(),
        paisho_core::GameOutcome::Ongoing => game
            .external_outcome
            .as_ref()
            .ok_or_else(|| invalid("human record has no result"))?
            .outcome
            .clone(),
    };
    if outcome != game.outcome {
        return Err(invalid("human replay outcome mismatch"));
    }
    crate::compact_learning::require_current_training_rules(record.rules().as_str())?;
    let mut p = record.initial_position();
    let mut examples = Vec::new();
    let mut wanted = game.examples.iter().peekable();
    for (decision, &action) in record.actions().iter().enumerate() {
        if wanted.peek().is_some_and(|e| e.decision_index == decision) {
            let old = wanted.next().unwrap();
            if old.perspective != p.to_move().code().to_string() {
                return Err(invalid("human target perspective mismatch"));
            }
            let legal = legal_actions(&p);
            let index = legal
                .iter()
                .position(|a| *a == action)
                .ok_or_else(|| invalid("human action not legal"))?;
            let mut policy = vec![0.0; legal.len()];
            policy[index] = 1.0;
            examples.push(MicroExample { policy_support: false, action_values: vec![], 
            value_weight: 1.0, sequence_source: sequence_source(&format!("human/{}", game.game_sha256)),
                state: micro_state_features(&p).to_vec(),
                actions: legal
                    .into_iter()
                    .map(|a| micro_action_features(&p, a))
                    .collect(),
                policy,
                value: old.target,
                policy_weight: 1.0,
            });
        }
        p.apply(action)?;
    }
    Ok(HumanGame {
        id: game.game_sha256.clone(),
        held_out: game.held_out,
        examples,
    })
}
#[derive(Clone, Copy, Debug, Serialize)]
pub struct MicroFitOptions {
    pub epochs: usize,
    pub patience: usize,
    pub batch: usize,
    pub rate: f64,
    pub l2: f64,
    pub seed: u64,
    pub max_games: usize,
}
fn losses(model: &MicroModel, examples: &[&MicroExample]) -> Result<serde_json::Value> {
    // Inference only; no backward pass when measuring holdout.
    let losses: Vec<_> = examples
        .par_iter()
        .map(|ex| {
            let emb = model.embed(&ex.state);
            let logits = MicroModel::logits(&emb, &ex.actions);
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let logz = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
            (
                0.5 * (emb.value - ex.value).powi(2),
                logits
                    .iter()
                    .zip(&ex.policy)
                    .map(|(l, p)| p * (logz - l))
                    .sum::<f64>(),
            )
        })
        .collect();
    let n = losses.len().max(1) as f64;
    let v = losses.iter().map(|l| l.0).sum::<f64>() / n;
    let p = losses.iter().map(|l| l.1).sum::<f64>() / n;
    if !v.is_finite() || !p.is_finite() {
        return Err(invalid("nonfinite validation loss"));
    }
    Ok(serde_json::json!({"value":v,"policy":p,"total":v+p,"examples":examples.len()}))
}
pub fn fit_micro_human(
    dataset: &Path,
    parent: Option<&Path>,
    output: &Path,
    o: MicroFitOptions,
) -> Result<()> {
    if o.epochs == 0
        || o.batch == 0
        || o.max_games == 0
        || !o.rate.is_finite()
        || o.rate <= 0.0
        || !o.l2.is_finite()
        || o.l2 < 0.0
    {
        return Err(invalid("invalid micro fitting options"));
    }
    fs::create_dir(output)?;
    let started = Instant::now();
    let data = load_dataset(dataset)?;
    crate::compact_learning::require_current_training_rules(&data.rules)?;
    let source_hash = sha256(&fs::read(dataset)?);
    // Conversion errors become Strings across the pool; Box<dyn Error> is local.
    let converted: Vec<_> = data
        .games
        .iter()
        .take(o.max_games)
        .collect::<Vec<_>>()
        .par_iter()
        .map(|g| convert(g).map_err(|e| e.to_string()))
        .collect();
    let games = converted
        .into_iter()
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(invalid)?;
    let mut train: Vec<_> = games
        .iter()
        .filter(|g| !g.held_out)
        .flat_map(|g| &g.examples)
        .collect();
    let validation: Vec<_> = games
        .iter()
        .filter(|g| g.held_out)
        .flat_map(|g| &g.examples)
        .collect();
    if train.is_empty() || validation.is_empty() {
        return Err(invalid("human fit requires train and holdout games"));
    }
    let prepared_seconds = started.elapsed().as_secs_f64();
    let parent = parent.map(MicroArtifact::load).transpose()?;
    let mut model = parent
        .as_ref()
        .map(MicroArtifact::model)
        .transpose()?
        .unwrap_or_else(|| MicroModel::seeded(o.seed));
    let initial = losses(&model, &validation)?;
    let mut best_loss = initial["total"].as_f64().unwrap();
    let mut best = model.clone();
    let mut best_epoch = 0;
    let mut best_updates = parent.as_ref().map_or(0, |p| p.updates);
    let mut updates = best_updates;
    let mut rows = vec![serde_json::json!({"epoch":0,"validation":initial})];
    let mut rng = StableRng::new(o.seed);
    let fit_started = Instant::now();
    for epoch in 1..=o.epochs {
        for i in (1..train.len()).rev() {
            let j = rng.index(i + 1);
            train.swap(i, j);
        }
        for batch in train.chunks(o.batch) {
            model.train_batch(batch, o.rate, o.l2).map_err(invalid)?;
            updates += 1;
        }
        let validation_loss = losses(&model, &validation)?;
        let total = validation_loss["total"].as_f64().unwrap();
        if total < best_loss {
            best_loss = total;
            best = model.clone();
            best_epoch = epoch;
            best_updates = updates;
        }
        let row = serde_json::json!({"epoch":epoch,"validation":validation_loss,"elapsed_seconds":fit_started.elapsed().as_secs_f64()});
        println!("{row}");
        rows.push(row);
        if o.patience > 0 && epoch - best_epoch >= o.patience {
            break;
        }
    }
    let artifact = MicroArtifact::new(
        &best,
        best_updates,
        serde_json::json!({"kind":"human-value-policy","rules":data.rules,"dataset_sha256":source_hash,"parent":parent.as_ref().map(MicroArtifact::identity),"options":o,"best_epoch":best_epoch}),
    );
    artifact.save(&output.join("model.json"))?;
    save_json_new(
        &output.join("report.json"),
        &serde_json::json!({"schema":"paisho-micro-human-fit-v1","rules":data.rules,"options":o,"source_dataset":dataset,"source_sha256":source_hash,"model_identity":artifact.identity(),"games":games.iter().map(|g|serde_json::json!({"id":g.id,"held_out":g.held_out,"examples":g.examples.len()})).collect::<Vec<_>>(),"train_examples":train.len(),"validation_examples":validation.len(),"preparation_seconds":prepared_seconds,"fit_seconds":fit_started.elapsed().as_secs_f64(),"best_epoch":best_epoch,"epochs":rows,"force":"not evaluated"}),
    )?;
    Ok(())
}
