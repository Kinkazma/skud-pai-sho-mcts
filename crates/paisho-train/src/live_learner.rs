//! Persistent learner. Durable publication is owned by the block coordinator.
use paisho_model::{
    CheckpointRandomStateV1, CheckpointRequestV1, TerminalPpoParametersV1, TerminalPpoRequestV1,
};
use paisho_mpsgraph_client::{
    read_checkpoint_metadata, MpsGraphProcess, ServiceConfiguration, TrainingCycleProgressV1,
    TrainingCycleRandomStateV1, TrainingCycleRequestV1, TrainingCycleSchedulerV1,
};
use paisho_replay::{ReplayDatasetV1, ReplaySamplerV1};
use serde::{Deserialize, Serialize};
use std::{error::Error, path::Path, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct LiveLearner {
    pub process: MpsGraphProcess,
    shape: (usize, usize),
    rate: f32,
    pub step: u64,
    pub win_duration_reward: bool,
    request_id: u64,
    snapshot: Option<[u8; 32]>,
    replay_index: u64,
    generation: u64,
    sampler_seed: u64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LiveLearningMetrics {
    #[serde(default)]
    pub preload_seconds: f64,
    #[serde(default)]
    pub rollover_seconds: f64,
    #[serde(default)]
    pub batch_preparation_seconds: f64,
    #[serde(default)]
    pub service_roundtrip_seconds: f64,
    pub steps: u64,
    pub examples: u64,
    pub seconds: f64,
    pub mean_policy_loss: f64,
    pub mean_value_loss: f64,
    pub mean_entropy: f64,
    #[serde(default)]
    pub mean_importance_ratio: Option<f64>,
    #[serde(default)]
    pub mean_squared_ratio_deviation: Option<f64>,
    /// Maximum batch mean, not the maximum individual-example deviation.
    #[serde(default)]
    pub maximum_batch_ratio_deviation: Option<f64>,
}

impl LiveLearner {
    pub fn launch(config: ServiceConfiguration, rate: f32) -> Result<Self> {
        let metadata = config
            .checkpoint
            .as_ref()
            .map(|p| read_checkpoint_metadata(p))
            .transpose()?;
        let step = metadata.map_or(0, |m| m.training_step());
        let generation = metadata.map_or(0, |m| m.generation());
        let shape = (config.batch_size, config.legal_action_capacity);
        let process = if config.checkpoint.is_some() {
            MpsGraphProcess::launch_new_generation(config)?
        } else {
            MpsGraphProcess::launch(config)?
        };
        Ok(Self {
            process,
            shape,
            rate,
            step,
            win_duration_reward: false,
            generation,
            request_id: 0,
            snapshot: None,
            replay_index: 0,
            sampler_seed: 0,
        })
    }

    pub fn next_request_id(&mut self) -> Result<u64> {
        let id = self.request_id;
        self.request_id = id.checked_add(1).ok_or("request id overflow")?;
        Ok(id)
    }

    pub fn train_cycle(
        &mut self,
        dataset: &ReplayDatasetV1,
        generation: u64,
        seed: u64,
        steps: u64,
        parameters: TerminalPpoParametersV1,
    ) -> Result<LiveLearningMetrics> {
        validate_legacy_live_dataset(dataset)?;
        if steps == 0 {
            return Err("a live cycle needs at least one training step".into());
        }
        let timer = Instant::now();
        dataset.preload()?;
        let preload_seconds = timer.elapsed().as_secs_f64();
        let timer = Instant::now();
        let digest = *dataset.snapshot_digest().as_bytes();
        let request_id = self.next_request_id()?;
        self.process.begin_new_generation(&TrainingCycleRequestV1 {
            request_id,
            expected_training_step: self.step,
            previous_snapshot_sha256: self.snapshot.map(hex),
            next_progress: TrainingCycleProgressV1 {
                generation,
                replay_index: 0,
                replay_snapshot_sha256: hex(digest),
                scheduler: TrainingCycleSchedulerV1 {
                    learning_rate: self.rate,
                    completed_steps: self.step,
                },
                random_states: vec![TrainingCycleRandomStateV1 {
                    name: "replay-sampler".into(),
                    state: seed,
                }],
            },
        })?;
        self.snapshot = Some(digest);
        self.generation = generation;
        self.sampler_seed = seed;
        self.replay_index = 0;
        let mut sampler = ReplaySamplerV1::new(dataset, seed);
        let started = Instant::now();
        let mut metrics = LiveLearningMetrics {
            preload_seconds,
            rollover_seconds: timer.elapsed().as_secs_f64(),
            ..LiveLearningMetrics::default()
        };
        for _ in 0..steps {
            let timer = Instant::now();
            let batch = sampler.prepare_batch(self.shape.0)?;
            let examples = batch
                .examples()
                .iter()
                .map(|example| {
                    example
                        .to_terminal_ppo_example(
                            example
                                .behavior_value()
                                .expect("checked complete actor values"),
                        )
                        .map(|value| {
                            if self.win_duration_reward {
                                value.with_win_duration(example.remaining_decisions())
                            } else {
                                value
                            }
                        })
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let request_id = self.next_request_id()?;
            let request = TerminalPpoRequestV1::new(
                request_id,
                self.step,
                self.rate,
                parameters,
                self.shape.1,
                digest,
                batch.start_replay_index(),
                examples,
            )?;
            metrics.batch_preparation_seconds += timer.elapsed().as_secs_f64();
            let timer = Instant::now();
            let response = self.process.train_terminal_ppo(&request)?;
            metrics.service_roundtrip_seconds += timer.elapsed().as_secs_f64();
            sampler.commit_batch(&batch)?;
            self.step = response.completed_training_step();
            self.replay_index = response.completed_replay_index();
            metrics.mean_policy_loss += f64::from(response.policy_loss());
            metrics.mean_value_loss += f64::from(response.value_loss());
            metrics.mean_entropy += f64::from(response.entropy());
            *metrics.mean_importance_ratio.get_or_insert(0.0) +=
                f64::from(response.mean_importance_ratio());
            let deviation = f64::from(response.mean_squared_ratio_deviation());
            *metrics.mean_squared_ratio_deviation.get_or_insert(0.0) += deviation;
            metrics.maximum_batch_ratio_deviation = Some(
                metrics
                    .maximum_batch_ratio_deviation
                    .unwrap_or(0.0)
                    .max(deviation),
            );
        }
        metrics.steps = steps;
        metrics.examples = steps
            .checked_mul(self.shape.0 as u64)
            .ok_or("example counter overflow")?;
        metrics.seconds = started.elapsed().as_secs_f64();
        if steps > 0 {
            metrics.mean_policy_loss /= steps as f64;
            metrics.mean_value_loss /= steps as f64;
            metrics.mean_entropy /= steps as f64;
            if let Some(value) = &mut metrics.mean_importance_ratio {
                *value /= steps as f64;
            }
            if let Some(value) = &mut metrics.mean_squared_ratio_deviation {
                *value /= steps as f64;
            }
        }
        Ok(metrics)
    }

    pub fn checkpoint(&mut self, destination: &Path) -> Result<String> {
        let id = self.next_request_id()?;
        let response = self.process.publish_checkpoint(&CheckpointRequestV1::new(
            id,
            self.step,
            self.snapshot.unwrap_or([0; 32]),
            self.replay_index,
            self.generation,
            self.rate,
            vec![CheckpointRandomStateV1::new(
                "replay-sampler",
                self.sampler_seed,
            )?],
            destination.to_str().ok_or("checkpoint path is not UTF-8")?,
        )?)?;
        self.snapshot = Some(self.snapshot.unwrap_or([0; 32]));
        Ok(hex(response.content_sha256()))
    }
}

fn validate_legacy_live_dataset(dataset: &ReplayDatasetV1) -> Result<()> {
    crate::learner::require_legacy_rules(dataset.rule_profile())?;
    if !dataset.has_complete_behavior_values() {
        return Err("live PPO requires recorded actor values".into());
    }
    Ok(())
}

pub fn hex(bytes: [u8; 32]) -> String {
    paisho_replay::ReplayDigestV1::from_bytes(bytes).to_string()
}

#[cfg(test)]
mod timing_tests {
    use super::{validate_legacy_live_dataset, LiveLearningMetrics};
    use paisho_core::{GameRecord, RuleProfileId};
    use paisho_model::encode_action_v1;
    use paisho_replay::{
        PolicyEntryV1, PolicyTargetKindV1, PolicyTargetV1, ReplayDatasetV1, ReplayDecisionV1,
        ReplayDigestV1, ReplayGameV1, ReplayShardV1,
    };

    fn behavior_dataset(rules: RuleProfileId) -> ReplayDatasetV1 {
        let source: GameRecord =
            include_str!("../../paisho-core/tests/fixtures/reported-ring-v1.psr")
                .parse()
                .unwrap();
        let (record, _) = source.replay_prefix_with_rules(rules).unwrap();
        let actor = ReplayDigestV1::from_bytes([43; 32]);
        let action =
            encode_action_v1(record.actions()[0], record.initial_position().to_move()).unwrap();
        let policy = PolicyTargetV1::new(
            PolicyTargetKindV1::Behavior,
            actor,
            vec![PolicyEntryV1::new(action, 1.0).unwrap()],
        )
        .unwrap();
        let decision = ReplayDecisionV1::new(0, policy).with_behavior_value(0.0);
        let game = ReplayGameV1::new(1, actor, actor, record, vec![decision]).unwrap();
        let shard = ReplayShardV1::new(0, vec![game]).unwrap();
        ReplayDatasetV1::from_shard_for_behavior(shard, "profile-test.psrbuf", actor).unwrap()
    }

    #[test]
    fn legacy_live_learner_accepts_v1_and_rejects_v2_before_service_rollover() {
        let legacy = behavior_dataset(RuleProfileId::SkudPaiSho2022);
        validate_legacy_live_dataset(&legacy).unwrap();
        let corrected = behavior_dataset(RuleProfileId::SkudPaiSho2022V2);
        let error = validate_legacy_live_dataset(&corrected).unwrap_err();
        assert!(error
            .to_string()
            .contains("supports only skud-pai-sho-2022-03-14"));
        assert!(error
            .to_string()
            .contains("received skud-pai-sho-2022-03-14-v2"));
    }

    #[test]
    fn older_metrics_remain_readable_without_timing_fields() {
        let metrics: LiveLearningMetrics = serde_json::from_value(serde_json::json!({
            "steps": 256, "examples": 16384, "seconds": 30.0,
            "mean_policy_loss": 0.0, "mean_value_loss": 1.0, "mean_entropy": 4.0
        }))
        .unwrap();
        assert_eq!(metrics.preload_seconds, 0.0);
        assert_eq!(metrics.rollover_seconds, 0.0);
        assert_eq!(metrics.batch_preparation_seconds, 0.0);
        assert_eq!(metrics.service_roundtrip_seconds, 0.0);
        assert_eq!(metrics.examples, 16384);
        assert_eq!(metrics.mean_importance_ratio, None);
        assert_eq!(metrics.mean_squared_ratio_deviation, None);
        assert_eq!(metrics.maximum_batch_ratio_deviation, None);
    }
}
