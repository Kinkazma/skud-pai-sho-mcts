//! Learner consumes immutable targets and bounded RAM replay while actors run.
use super::*;

pub(super) fn learn_game(
    game: &PlayedGame,
    options: &Options,
    parent: &ModelArtifact,
    model: &mut CompactValueModel,
    shared: &SharedModel,
    progress: &mut Progress,
    replay: &mut ReplayMemory,
) -> Result<()> {
    let learning_started = Instant::now();
    let persistence_before = progress.coordinator_persistence_seconds;
    let result = (|| {
        let mut candidate = model.clone();
        let mut targets = Vec::new();
        let mut loss = 0.0;
        let mut fresh_count = 0;
        if options.learn && game.error.is_none() {
            let source_run = options
                .output
                .canonicalize()?
                .to_string_lossy()
                .into_owned();
            let fresh: Vec<_> = game
                .samples
                .iter()
                .filter_map(|sample| training_example(game, sample, options.lambda, &source_run))
                .collect();
            fresh_count = fresh.len();
            for entry in fresh {
                loss += candidate.train_step(
                    &entry.sample.features()?,
                    entry.target,
                    options.learning_rate,
                    0.0,
                )?;
                targets.push(serde_json::json!({"source_run":entry.source_run,"source_game":entry.game_id,"source_version":entry.actor_version,
                    "source_weights_sha256":entry.actor_weights_sha256,"decision":entry.sample.decision,
                    "perspective":entry.sample.perspective,"q":entry.sample.q,"target":entry.target,
                    "reason":entry.reason,"replayed":false}));
                replay.push(entry);
            }
            for _ in 0..fresh_count.saturating_mul(options.replay_ratio) {
                let Some(entry) = replay.draw() else {
                    break;
                };
                loss += candidate.train_step(
                    &entry.sample.features()?,
                    entry.target,
                    options.learning_rate,
                    0.0,
                )?;
                targets.push(serde_json::json!({"source_run":entry.source_run,"source_game":entry.game_id,"source_version":entry.actor_version,
                    "source_weights_sha256":entry.actor_weights_sha256,"decision":entry.sample.decision,
                    "perspective":entry.sample.perspective,"q":entry.sample.q,"target":entry.target,
                    "reason":entry.reason,"replayed":true}));
            }
        }
        let previous_version = progress.published_version;
        if !targets.is_empty() {
            let next_version = previous_version
                .checked_add(1)
                .ok_or_else(|| invalid("model version overflow"))?;
            let next_updates = progress
                .updates
                .checked_add(targets.len() as u64)
                .ok_or_else(|| invalid("update count overflow"))?;
            let steps = parent
                .training_steps
                .checked_add(next_updates)
                .ok_or_else(|| invalid("training step overflow"))?;
            let artifact = parent.with_model(&candidate, steps, serde_json::json!({
            "method":"compact-replay-and-repetition-sgd-v2","rules":game.record.rules().as_str(),"version":next_version,
            "initial_model_weights_sha256":weights_hash(&parent.model()?),
            "parent_version":previous_version,"source_game":game.id,"source_snapshot_version":game.version,
            "source_simulations_requested_per_decision":game.simulations,
            "lambda":options.lambda,"learning_rate":options.learning_rate,
            "replay_capacity":options.replay_capacity,"replay_ratio":options.replay_ratio,
            "build_source_sha256":env!("PAISHO_BUILD_SOURCE_SHA256")
        }));
            // This file is durable before actors can observe the new version.
            timed_persistence(progress, || {
                save_model_new(
                    &options
                        .output
                        .join("models")
                        .join(format!("version-{next_version:08}.json")),
                    &artifact,
                )
            })?;
            *shared
                .write()
                .map_err(|_| invalid("shared model lock poisoned"))? =
                (next_version, Arc::new(candidate.clone()));
            *model = candidate;
            progress.published_version = next_version;
            progress.updates = next_updates;
        }
        let receipt = serde_json::json!({"rules":game.record.rules().as_str(),"game_id":game.id,"actor_version":game.version,
            "source_simulations_requested_per_decision":game.simulations,
            "learning_enabled":options.learn,
            "learner_version_before":previous_version,"learner_version_after":progress.published_version,
            "fresh_updates":fresh_count,"replay_updates":targets.len()-fresh_count,"replay_size":replay.len(),
            "replay_draws_total":replay.draws,
            "updates":targets.len(),"mean_preupdate_loss":if targets.is_empty() {None} else {Some(loss/targets.len() as f64)},
            "targets":targets});
        timed_persistence(progress, || {
            save_json_new(
                &options
                    .output
                    .join("games")
                    .join(format!("game-{:08}.learning.json", game.id)),
                &receipt,
            )
        })
    })();
    progress.coordinator_learning_seconds += (learning_started.elapsed().as_secs_f64()
        - (progress.coordinator_persistence_seconds - persistence_before))
        .max(0.0);
    result
}
