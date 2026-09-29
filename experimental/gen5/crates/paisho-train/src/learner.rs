use core::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;
use std::time::{Duration, Instant};

use paisho_core::RuleProfileId;
use paisho_model::{
    CheckpointRandomStateV1, CheckpointRequestV1, CheckpointWireError, InferenceRequestV1,
    InferenceWireError, TerminalPpoParametersV1, TerminalPpoRequestV1, TerminalPpoWireError,
    TrainingRequestV1, TrainingWireError,
};
use paisho_mpsgraph_client::{
    read_checkpoint_metadata, CheckpointMetadataError, MpsGraphClientError, MpsGraphProcess,
    NetworkPreset, OptimizationLevel, ServiceConfiguration,
};
use paisho_replay::{
    ReplayDatasetV1, ReplayDatasetV1Error, ReplaySamplerStateV1, ReplaySamplerV1,
    ReplaySamplerV1Error, ReplaySnapshotV1, ReplaySnapshotV1Error, ReplayTerminalPpoExampleError,
    ReplayValidationError,
};

use crate::commit::checkpoint_file_name;
use crate::learner_metrics::write_terminal_ppo_metric_segment_v1;
use crate::{
    discover_latest_commit, LearnerCommit, LearnerCommitError, LearnerIdentityV1,
    LearnerMetricsError, LearnerObjectiveConfiguration, LearnerObjectiveV1, LearnerOriginV1,
    LearnerOriginV1Error, ParentCheckpointV1, TerminalPpoBatchMetricsV1,
    TerminalPpoLearnerObjectiveV1, TerminalPpoMetricsSummaryV1,
};

#[derive(Clone, Debug)]
pub struct LearnerConfiguration {
    pub service_executable: PathBuf,
    pub replay_snapshot: PathBuf,
    pub replay_directory: PathBuf,
    pub run_directory: PathBuf,
    pub initial_checkpoint: Option<PathBuf>,
    pub network_preset: NetworkPreset,
    pub optimization: OptimizationLevel,
    pub batch_size: usize,
    pub legal_action_capacity: usize,
    pub model_seed: u64,
    pub sampler_seed: u64,
    pub generation: u64,
    pub learning_rate: f32,
    pub objective: LearnerObjectiveConfiguration,
    pub target_training_step: u64,
    pub checkpoint_interval: u64,
}

impl LearnerConfiguration {
    fn identity(
        &self,
        replay_snapshot: paisho_replay::ReplayDigestV1,
        objective: LearnerObjectiveV1,
    ) -> Result<LearnerIdentityV1, LearnerCommitError> {
        LearnerIdentityV1 {
            replay_snapshot,
            network_preset: self.network_preset,
            optimization: self.optimization,
            batch_size: self.batch_size,
            legal_action_capacity: self.legal_action_capacity,
            model_seed: self.model_seed,
            sampler_seed: self.sampler_seed,
            generation: self.generation,
            learning_rate: self.learning_rate,
            objective,
        }
        .validate()
    }

    fn validate(&self) -> Result<(), LearnerError> {
        if self.target_training_step == 0 {
            return Err(LearnerError::ZeroTargetTrainingStep);
        }
        if self.checkpoint_interval == 0 {
            return Err(LearnerError::ZeroCheckpointInterval);
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct LearnerReport {
    pub resumed: bool,
    pub has_parent_checkpoint: bool,
    pub started_from_parent_checkpoint: bool,
    pub generation_start_training_step: u64,
    pub initial_training_step: u64,
    pub completed_training_step: u64,
    pub completed_replay_index: u64,
    pub training_steps_this_run: u64,
    pub examples_this_run: u64,
    pub checkpoints_this_run: u64,
    pub checkpoint_elapsed: Duration,
    pub replay_preload_elapsed: Duration,
    pub recorded_behavior_values_complete: Option<bool>,
    pub terminal_ppo_metrics_this_run: Option<TerminalPpoMetricsSummaryV1>,
    pub latest_checkpoint: PathBuf,
    pub elapsed: Duration,
}

pub fn run_learner(configuration: &LearnerConfiguration) -> Result<LearnerReport, LearnerError> {
    configuration.validate()?;
    fs::create_dir_all(&configuration.run_directory).map_err(LearnerError::Io)?;
    let run_directory = fs::canonicalize(&configuration.run_directory).map_err(LearnerError::Io)?;
    if run_directory.to_str().is_none() {
        return Err(LearnerError::NonUtf8Path(run_directory));
    }
    let snapshot =
        ReplaySnapshotV1::read(&configuration.replay_snapshot).map_err(LearnerError::Snapshot)?;
    require_legacy_rules(snapshot.rule_profile())?;
    let supplied_parent = read_supplied_parent(configuration)?;
    let resolved_objective = resolve_objective(configuration)?;
    let dataset = match resolved_objective.identity {
        LearnerObjectiveV1::SupervisedPolicyValue => {
            ReplayDatasetV1::from_snapshot(&snapshot, &configuration.replay_directory)
        }
        LearnerObjectiveV1::TerminalPpo(objective) => ReplayDatasetV1::from_snapshot_for_behavior(
            &snapshot,
            &configuration.replay_directory,
            objective.behavior_producer(),
        ),
    }
    .map_err(LearnerError::Dataset)?;
    let recorded_behavior_values_complete = resolved_objective
        .identity
        .terminal_ppo()
        .map(|_| dataset.has_complete_behavior_values());
    let identity = configuration
        .identity(dataset.snapshot_digest(), resolved_objective.identity)
        .map_err(LearnerError::Commit)?;
    let origin = resolve_origin(
        &run_directory,
        identity,
        supplied_parent.as_ref().map(|parent| parent.provenance),
    )?;
    validate_actor_generation_start(resolved_objective.identity, origin)?;
    let latest = discover_latest_commit(&run_directory, identity).map_err(LearnerError::Commit)?;
    if let Some(commit) = &latest {
        if commit.starting_training_step() != origin.starting_training_step() {
            return Err(LearnerError::CommitOriginMismatch {
                commit: commit.starting_training_step(),
                origin: origin.starting_training_step(),
            });
        }
        origin
            .establish(&run_directory)
            .map_err(LearnerError::Origin)?;
    }
    let resumed = latest.is_some();
    let has_parent_checkpoint = origin.parent().is_some();
    let initial_training_step = latest.as_ref().map_or_else(
        || origin.starting_training_step(),
        LearnerCommit::training_step,
    );
    if initial_training_step > configuration.target_training_step {
        return Err(LearnerError::TargetPrecedesCommittedStep {
            target: configuration.target_training_step,
            committed: initial_training_step,
        });
    }
    if initial_training_step == configuration.target_training_step {
        let latest = latest.ok_or(LearnerError::TargetDoesNotAdvanceGeneration {
            target: configuration.target_training_step,
            generation_start: origin.starting_training_step(),
        })?;
        return Ok(LearnerReport {
            resumed,
            has_parent_checkpoint,
            started_from_parent_checkpoint: false,
            generation_start_training_step: origin.starting_training_step(),
            initial_training_step,
            completed_training_step: initial_training_step,
            completed_replay_index: latest.next_replay_index(),
            training_steps_this_run: 0,
            examples_this_run: 0,
            checkpoints_this_run: 0,
            checkpoint_elapsed: Duration::ZERO,
            replay_preload_elapsed: Duration::ZERO,
            recorded_behavior_values_complete,
            terminal_ppo_metrics_this_run: None,
            latest_checkpoint: latest.checkpoint_path(&run_directory),
            elapsed: Duration::ZERO,
        });
    }

    let (mut training_step, mut next_request_id, sampler_state, restored_checkpoint) =
        match latest.as_ref() {
            Some(commit) => (
                commit.training_step(),
                commit.next_request_id(),
                ReplaySamplerStateV1::new(
                    identity.replay_snapshot,
                    identity.sampler_seed,
                    commit.next_replay_index(),
                ),
                Some(commit.checkpoint_path(&run_directory)),
            ),
            None => {
                let restored_checkpoint = match origin.parent() {
                    Some(parent) => {
                        let supplied = supplied_parent.as_ref().ok_or(
                            LearnerError::InitialCheckpointRequired {
                                expected_sha256: parent.content_sha256(),
                            },
                        )?;
                        Some(supplied.path.clone())
                    }
                    None => None,
                };
                (
                    origin.starting_training_step(),
                    0,
                    ReplaySamplerStateV1::new(identity.replay_snapshot, identity.sampler_seed, 0),
                    restored_checkpoint,
                )
            }
        };
    let mut sampler =
        ReplaySamplerV1::resume(&dataset, sampler_state).map_err(LearnerError::Sampler)?;
    let preload_started = Instant::now();
    dataset.preload().map_err(LearnerError::ReplayPreload)?;
    let replay_preload_elapsed = preload_started.elapsed();
    let service_configuration = ServiceConfiguration {
        executable: configuration.service_executable.clone(),
        preset: configuration.network_preset,
        batch_size: configuration.batch_size,
        legal_action_capacity: configuration.legal_action_capacity,
        inference_slots: 1,
        optimization: configuration.optimization,
        seed: configuration.model_seed,
        checkpoint: restored_checkpoint,
    };
    let started_from_parent_checkpoint = latest.is_none() && has_parent_checkpoint;
    let mut process = if started_from_parent_checkpoint {
        MpsGraphProcess::launch_new_generation(service_configuration)
    } else {
        MpsGraphProcess::launch(service_configuration)
    }
    .map_err(LearnerError::Client)?;
    let mut actor_process = launch_actor_process(
        configuration,
        &resolved_objective,
        recorded_behavior_values_complete == Some(true),
    )?;

    let started = Instant::now();
    let mut checkpoints_this_run = 0_u64;
    let mut checkpoint_elapsed = Duration::ZERO;
    let mut latest_checkpoint = latest
        .as_ref()
        .map(|commit| commit.checkpoint_path(&run_directory));
    let mut terminal_ppo_metrics_this_run = Vec::new();
    let mut pending_terminal_ppo_metrics = Vec::new();
    while training_step < configuration.target_training_step {
        let sampled = sampler
            .prepare_batch(configuration.batch_size)
            .map_err(LearnerError::Sampler)?;
        let following_request_id = next_request_id
            .checked_add(1)
            .ok_or(LearnerError::RequestIdOverflow)?;
        let completed = train_sampled_batch(
            &mut process,
            actor_process.as_mut(),
            identity,
            next_request_id,
            training_step,
            &sampled,
        )?;
        if completed.replay_index != sampled.next_replay_index() {
            return Err(LearnerError::TrainingReplayIndexMismatch {
                expected: sampled.next_replay_index(),
                actual: completed.replay_index,
            });
        }
        if let Some(metrics) = completed.terminal_ppo_metrics {
            terminal_ppo_metrics_this_run.push(metrics);
            pending_terminal_ppo_metrics.push(metrics);
        }
        sampler
            .commit_batch(&sampled)
            .map_err(LearnerError::Sampler)?;
        training_step = completed.training_step;
        next_request_id = following_request_id;

        if training_step % configuration.checkpoint_interval == 0
            || training_step == configuration.target_training_step
        {
            let checkpoint_started = Instant::now();
            let (commit, checkpoint_path) = publish_commit(
                &mut process,
                &run_directory,
                origin,
                training_step,
                sampler.state().next_replay_index(),
                next_request_id,
                &pending_terminal_ppo_metrics,
            )?;
            checkpoint_elapsed += checkpoint_started.elapsed();
            pending_terminal_ppo_metrics.clear();
            next_request_id = commit.next_request_id();
            latest_checkpoint = Some(checkpoint_path);
            checkpoints_this_run = checkpoints_this_run
                .checked_add(1)
                .ok_or(LearnerError::CounterOverflow)?;
        }
    }
    let exit_status = process.shutdown().map_err(LearnerError::Client)?;
    if !exit_status.success() {
        return Err(LearnerError::ServiceExit(exit_status));
    }
    if let Some(actor) = actor_process {
        let exit_status = actor.shutdown().map_err(LearnerError::Client)?;
        if !exit_status.success() {
            return Err(LearnerError::ActorServiceExit(exit_status));
        }
    }
    let training_steps_this_run = training_step - initial_training_step;
    let examples_this_run = training_steps_this_run
        .checked_mul(configuration.batch_size as u64)
        .ok_or(LearnerError::CounterOverflow)?;
    let terminal_ppo_metrics_this_run =
        TerminalPpoMetricsSummaryV1::from_batches(&terminal_ppo_metrics_this_run)
            .map_err(LearnerError::Metrics)?;
    Ok(LearnerReport {
        resumed,
        has_parent_checkpoint,
        started_from_parent_checkpoint,
        generation_start_training_step: origin.starting_training_step(),
        initial_training_step,
        completed_training_step: training_step,
        completed_replay_index: sampler.state().next_replay_index(),
        training_steps_this_run,
        examples_this_run,
        checkpoints_this_run,
        checkpoint_elapsed,
        replay_preload_elapsed,
        recorded_behavior_values_complete,
        terminal_ppo_metrics_this_run,
        latest_checkpoint: latest_checkpoint.expect("target step always publishes a checkpoint"),
        elapsed: started.elapsed(),
    })
}

#[derive(Clone, Debug)]
struct ResolvedLearnerObjective {
    identity: LearnerObjectiveV1,
    actor_checkpoint: Option<PathBuf>,
}

fn resolve_objective(
    configuration: &LearnerConfiguration,
) -> Result<ResolvedLearnerObjective, LearnerError> {
    match &configuration.objective {
        LearnerObjectiveConfiguration::SupervisedPolicyValue => Ok(ResolvedLearnerObjective {
            identity: LearnerObjectiveV1::SupervisedPolicyValue,
            actor_checkpoint: None,
        }),
        LearnerObjectiveConfiguration::TerminalPpo {
            behavior_producer,
            actor_checkpoint,
            parameters,
        } => {
            let (actor_checkpoint, actor_checkpoint_sha256) = match actor_checkpoint {
                Some(path) => {
                    let path = fs::canonicalize(path).map_err(LearnerError::Io)?;
                    let metadata = read_checkpoint_metadata(&path)
                        .map_err(LearnerError::ActorCheckpointMetadata)?;
                    if metadata.network_preset() != configuration.network_preset {
                        return Err(LearnerError::ActorNetworkMismatch {
                            expected: configuration.network_preset,
                            actual: metadata.network_preset(),
                        });
                    }
                    (
                        Some(path),
                        Some(paisho_replay::ReplayDigestV1::from_bytes(
                            metadata.content_sha256(),
                        )),
                    )
                }
                None => (None, None),
            };
            Ok(ResolvedLearnerObjective {
                identity: LearnerObjectiveV1::TerminalPpo(TerminalPpoLearnerObjectiveV1::new(
                    *behavior_producer,
                    actor_checkpoint_sha256,
                    *parameters,
                )),
                actor_checkpoint,
            })
        }
    }
}

fn validate_actor_generation_start(
    objective: LearnerObjectiveV1,
    origin: LearnerOriginV1,
) -> Result<(), LearnerError> {
    let Some(terminal) = objective.terminal_ppo() else {
        return Ok(());
    };
    let actor = terminal.actor_checkpoint_sha256();
    let candidate = origin
        .parent()
        .map(|parent| paisho_replay::ReplayDigestV1::from_bytes(parent.content_sha256()));
    if actor == candidate {
        Ok(())
    } else {
        Err(LearnerError::ActorCandidateStartMismatch { actor, candidate })
    }
}

fn launch_actor_process(
    configuration: &LearnerConfiguration,
    objective: &ResolvedLearnerObjective,
    recorded_behavior_values_complete: bool,
) -> Result<Option<MpsGraphProcess>, LearnerError> {
    if objective.identity.terminal_ppo().is_none() || recorded_behavior_values_complete {
        return Ok(None);
    }
    MpsGraphProcess::launch(ServiceConfiguration {
        executable: configuration.service_executable.clone(),
        preset: configuration.network_preset,
        batch_size: configuration.batch_size,
        legal_action_capacity: configuration.legal_action_capacity,
        inference_slots: 1,
        optimization: configuration.optimization,
        seed: configuration.model_seed,
        checkpoint: objective.actor_checkpoint.clone(),
    })
    .map(Some)
    .map_err(LearnerError::Client)
}

#[derive(Clone, Copy, Debug)]
struct CompletedBatch {
    training_step: u64,
    replay_index: u64,
    terminal_ppo_metrics: Option<TerminalPpoBatchMetricsV1>,
}

fn train_sampled_batch(
    candidate: &mut MpsGraphProcess,
    actor: Option<&mut MpsGraphProcess>,
    identity: LearnerIdentityV1,
    request_id: u64,
    training_step: u64,
    sampled: &paisho_replay::ReplaySampleBatchV1,
) -> Result<CompletedBatch, LearnerError> {
    match identity.objective {
        LearnerObjectiveV1::SupervisedPolicyValue => {
            let examples = sampled
                .examples()
                .iter()
                .map(|example| example.to_training_example())
                .collect::<Result<Vec<_>, _>>()
                .map_err(LearnerError::TrainingWire)?;
            let request = TrainingRequestV1::new(
                request_id,
                training_step,
                identity.learning_rate,
                identity.legal_action_capacity,
                *identity.replay_snapshot.as_bytes(),
                sampled.start_replay_index(),
                examples,
            )
            .map_err(LearnerError::TrainingWire)?;
            let response = candidate.train(&request).map_err(LearnerError::Client)?;
            Ok(CompletedBatch {
                training_step: response.completed_training_step(),
                replay_index: response.completed_replay_index(),
                terminal_ppo_metrics: None,
            })
        }
        LearnerObjectiveV1::TerminalPpo(objective) => train_terminal_ppo_batch(
            candidate,
            actor,
            identity,
            objective.parameters(),
            request_id,
            training_step,
            sampled,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn train_terminal_ppo_batch(
    candidate: &mut MpsGraphProcess,
    actor: Option<&mut MpsGraphProcess>,
    identity: LearnerIdentityV1,
    parameters: TerminalPpoParametersV1,
    request_id: u64,
    training_step: u64,
    sampled: &paisho_replay::ReplaySampleBatchV1,
) -> Result<CompletedBatch, LearnerError> {
    let actor_values = match sampled
        .examples()
        .iter()
        .map(|example| example.behavior_value())
        .collect::<Option<Vec<_>>>()
    {
        Some(values) => values,
        None => {
            let inference = InferenceRequestV1::from_examples(
                request_id,
                sampled
                    .examples()
                    .iter()
                    .map(|example| example.inference().clone())
                    .collect(),
                identity.legal_action_capacity,
            )
            .map_err(LearnerError::InferenceWire)?;
            actor
                .ok_or(LearnerError::MissingActorProcess)?
                .infer(&inference)
                .map_err(LearnerError::Client)?
                .into_outputs()
                .into_iter()
                .map(|output| {
                    let values = output.value_probabilities();
                    values[0] - values[2]
                })
                .collect()
        }
    };
    let examples = sampled
        .examples()
        .iter()
        .zip(actor_values)
        .map(|(example, actor_value)| example.to_terminal_ppo_example(actor_value))
        .collect::<Result<Vec<_>, _>>()
        .map_err(LearnerError::ReplayTerminalPpo)?;
    let request = TerminalPpoRequestV1::new(
        request_id,
        training_step,
        identity.learning_rate,
        parameters,
        identity.legal_action_capacity,
        *identity.replay_snapshot.as_bytes(),
        sampled.start_replay_index(),
        examples,
    )
    .map_err(LearnerError::TerminalPpoWire)?;
    let response = candidate
        .train_terminal_ppo(&request)
        .map_err(LearnerError::Client)?;
    Ok(CompletedBatch {
        training_step: response.completed_training_step(),
        replay_index: response.completed_replay_index(),
        terminal_ppo_metrics: Some(TerminalPpoBatchMetricsV1 {
            training_step: response.completed_training_step(),
            replay_index: response.completed_replay_index(),
            policy_loss: response.policy_loss(),
            value_loss: response.value_loss(),
            entropy: response.entropy(),
            total_loss: response.total_loss(),
            mean_advantage: response.mean_advantage(),
            mean_importance_ratio: response.mean_importance_ratio(),
            mean_squared_ratio_deviation: response.mean_squared_ratio_deviation(),
        }),
    })
}

#[derive(Clone, Debug)]
struct SuppliedParentCheckpoint {
    path: PathBuf,
    provenance: ParentCheckpointV1,
}

fn read_supplied_parent(
    configuration: &LearnerConfiguration,
) -> Result<Option<SuppliedParentCheckpoint>, LearnerError> {
    let Some(path) = &configuration.initial_checkpoint else {
        return Ok(None);
    };
    let path = fs::canonicalize(path).map_err(LearnerError::Io)?;
    let metadata = read_checkpoint_metadata(&path).map_err(LearnerError::CheckpointMetadata)?;
    if metadata.network_preset() != configuration.network_preset {
        return Err(LearnerError::ParentNetworkMismatch {
            expected: configuration.network_preset,
            actual: metadata.network_preset(),
        });
    }
    if metadata.generation() >= configuration.generation {
        return Err(LearnerError::ParentGenerationNotEarlier {
            parent: metadata.generation(),
            child: configuration.generation,
        });
    }
    Ok(Some(SuppliedParentCheckpoint {
        path,
        provenance: ParentCheckpointV1::new(
            metadata.content_sha256(),
            metadata.generation(),
            metadata.training_step(),
        ),
    }))
}

fn resolve_origin(
    run_directory: &Path,
    identity: LearnerIdentityV1,
    supplied_parent: Option<ParentCheckpointV1>,
) -> Result<LearnerOriginV1, LearnerError> {
    let path = LearnerOriginV1::path(run_directory);
    match LearnerOriginV1::read(&path) {
        Ok(stored) => {
            if stored.identity() != identity {
                return Err(LearnerError::OriginIdentityMismatch(path));
            }
            if supplied_parent.is_some() && stored.parent() != supplied_parent {
                return Err(LearnerError::OriginParentMismatch(path));
            }
            Ok(stored)
        }
        Err(LearnerOriginV1Error::Io(error)) if error.kind() == io::ErrorKind::NotFound => {
            LearnerOriginV1::new(identity, supplied_parent).map_err(LearnerError::Origin)
        }
        Err(error) => Err(LearnerError::Origin(error)),
    }
}

fn publish_commit(
    process: &mut MpsGraphProcess,
    run_directory: &Path,
    origin: LearnerOriginV1,
    training_step: u64,
    next_replay_index: u64,
    request_id: u64,
    terminal_ppo_metrics: &[TerminalPpoBatchMetricsV1],
) -> Result<(LearnerCommit, PathBuf), LearnerError> {
    let identity = origin.identity();
    let following_request_id = request_id
        .checked_add(1)
        .ok_or(LearnerError::RequestIdOverflow)?;
    let (checkpoint_file, checkpoint_path) =
        next_checkpoint_file(run_directory, identity.generation, training_step)?;
    let destination = checkpoint_path
        .to_str()
        .ok_or_else(|| LearnerError::NonUtf8Path(checkpoint_path.clone()))?;
    let request = CheckpointRequestV1::new(
        request_id,
        training_step,
        *identity.replay_snapshot.as_bytes(),
        next_replay_index,
        identity.generation,
        identity.learning_rate,
        vec![
            CheckpointRandomStateV1::new("learner-request-id", following_request_id)
                .map_err(LearnerError::CheckpointWire)?,
            CheckpointRandomStateV1::new("replay-sampler-seed", identity.sampler_seed)
                .map_err(LearnerError::CheckpointWire)?,
        ],
        destination,
    )
    .map_err(LearnerError::CheckpointWire)?;
    let response = process
        .publish_checkpoint(&request)
        .map_err(LearnerError::Client)?;
    write_terminal_ppo_metric_segment_v1(
        &checkpoint_path,
        response.content_sha256(),
        identity.generation,
        training_step,
        terminal_ppo_metrics,
    )
    .map_err(LearnerError::Metrics)?;
    origin
        .establish(run_directory)
        .map_err(LearnerError::Origin)?;
    let commit = LearnerCommit::new(
        identity,
        origin.starting_training_step(),
        training_step,
        next_replay_index,
        following_request_id,
        checkpoint_file,
        response.content_sha256(),
    )
    .map_err(LearnerError::Commit)?;
    commit
        .write_idempotently(run_directory)
        .map_err(LearnerError::Commit)?;
    Ok((commit, checkpoint_path))
}

fn next_checkpoint_file(
    run_directory: &Path,
    generation: u64,
    training_step: u64,
) -> Result<(String, PathBuf), LearnerError> {
    let mut attempt = 0_u64;
    loop {
        let file = checkpoint_file_name(generation, training_step, attempt);
        let path = run_directory.join(&file);
        match fs::symlink_metadata(&path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok((file, path)),
            Ok(_) => {
                attempt = attempt
                    .checked_add(1)
                    .ok_or(LearnerError::CheckpointAttemptOverflow)?;
            }
            Err(error) => return Err(LearnerError::Io(error)),
        }
    }
}

/// The retired GPU wire/checkpoint format still declares V1. Refuse V2 before
/// producing durable learner state or sending examples to that service.
pub(crate) fn require_legacy_rules(actual: RuleProfileId) -> Result<(), LearnerError> {
    if actual != RuleProfileId::SkudPaiSho2022 {
        return Err(LearnerError::UnsupportedRuleProfile(actual));
    }
    Ok(())
}

#[derive(Debug)]
pub enum LearnerError {
    UnsupportedRuleProfile(RuleProfileId),
    ZeroTargetTrainingStep,
    ZeroCheckpointInterval,
    TargetPrecedesCommittedStep {
        target: u64,
        committed: u64,
    },
    TargetDoesNotAdvanceGeneration {
        target: u64,
        generation_start: u64,
    },
    InitialCheckpointRequired {
        expected_sha256: [u8; 32],
    },
    ParentNetworkMismatch {
        expected: NetworkPreset,
        actual: NetworkPreset,
    },
    ParentGenerationNotEarlier {
        parent: u64,
        child: u64,
    },
    CommitOriginMismatch {
        commit: u64,
        origin: u64,
    },
    OriginIdentityMismatch(PathBuf),
    OriginParentMismatch(PathBuf),
    RequestIdOverflow,
    CounterOverflow,
    CheckpointAttemptOverflow,
    NonUtf8Path(PathBuf),
    TrainingReplayIndexMismatch {
        expected: u64,
        actual: u64,
    },
    ServiceExit(ExitStatus),
    ActorServiceExit(ExitStatus),
    MissingActorProcess,
    ActorCandidateStartMismatch {
        actor: Option<paisho_replay::ReplayDigestV1>,
        candidate: Option<paisho_replay::ReplayDigestV1>,
    },
    ActorNetworkMismatch {
        expected: NetworkPreset,
        actual: NetworkPreset,
    },
    Snapshot(ReplaySnapshotV1Error),
    Dataset(ReplayDatasetV1Error),
    ReplayPreload(ReplayValidationError),
    Sampler(ReplaySamplerV1Error),
    InferenceWire(InferenceWireError),
    TrainingWire(TrainingWireError),
    TerminalPpoWire(TerminalPpoWireError),
    ReplayTerminalPpo(ReplayTerminalPpoExampleError),
    CheckpointWire(CheckpointWireError),
    Client(MpsGraphClientError),
    CheckpointMetadata(CheckpointMetadataError),
    ActorCheckpointMetadata(CheckpointMetadataError),
    Commit(LearnerCommitError),
    Origin(LearnerOriginV1Error),
    Metrics(LearnerMetricsError),
    Io(io::Error),
}

impl fmt::Display for LearnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedRuleProfile(actual) => write!(
                formatter,
                "the retired MPSGraph/PPO learner supports only {}; received {actual}",
                RuleProfileId::SkudPaiSho2022
            ),
            Self::ZeroTargetTrainingStep => {
                formatter.write_str("target training step must be positive")
            }
            Self::ZeroCheckpointInterval => {
                formatter.write_str("checkpoint interval must be positive")
            }
            Self::TargetPrecedesCommittedStep { target, committed } => write!(
                formatter,
                "target step {target} precedes committed step {committed}"
            ),
            Self::TargetDoesNotAdvanceGeneration {
                target,
                generation_start,
            } => write!(
                formatter,
                "target step {target} does not advance generation start step {generation_start}"
            ),
            Self::InitialCheckpointRequired { expected_sha256 } => write!(
                formatter,
                "this generation has no local commit yet; provide its parent checkpoint {}",
                encode_digest(*expected_sha256)
            ),
            Self::ParentNetworkMismatch { expected, actual } => write!(
                formatter,
                "parent checkpoint network {actual:?} does not match requested network {expected:?}"
            ),
            Self::ParentGenerationNotEarlier { parent, child } => write!(
                formatter,
                "parent checkpoint generation {parent} is not earlier than child generation {child}"
            ),
            Self::CommitOriginMismatch { commit, origin } => write!(
                formatter,
                "learner commit starts at global step {commit}, but generation origin starts at {origin}"
            ),
            Self::OriginIdentityMismatch(path) => write!(
                formatter,
                "learner origin {} belongs to a different run",
                path.display()
            ),
            Self::OriginParentMismatch(path) => write!(
                formatter,
                "learner origin {} names a different parent checkpoint",
                path.display()
            ),
            Self::RequestIdOverflow => formatter.write_str("learner request id overflow"),
            Self::CounterOverflow => formatter.write_str("learner telemetry counter overflow"),
            Self::CheckpointAttemptOverflow => {
                formatter.write_str("learner checkpoint attempt counter overflow")
            }
            Self::NonUtf8Path(path) => {
                write!(formatter, "checkpoint path {} is not UTF-8", path.display())
            }
            Self::TrainingReplayIndexMismatch { expected, actual } => write!(
                formatter,
                "training response advanced replay to {actual}; expected {expected}"
            ),
            Self::ServiceExit(status) => write!(formatter, "MPSGraph service exited with {status}"),
            Self::ActorServiceExit(status) => {
                write!(formatter, "frozen actor service exited with {status}")
            }
            Self::MissingActorProcess => {
                formatter.write_str("terminal PPO training has no frozen actor process")
            }
            Self::ActorCandidateStartMismatch { actor, candidate } => write!(
                formatter,
                "terminal PPO actor checkpoint {actor:?} differs from candidate generation start {candidate:?}"
            ),
            Self::ActorNetworkMismatch { expected, actual } => write!(
                formatter,
                "actor checkpoint network {actual:?} does not match requested network {expected:?}"
            ),
            Self::Snapshot(source) => source.fmt(formatter),
            Self::Dataset(source) => source.fmt(formatter),
            Self::ReplayPreload(source) => source.fmt(formatter),
            Self::Sampler(source) => source.fmt(formatter),
            Self::InferenceWire(source) => source.fmt(formatter),
            Self::TrainingWire(source) => source.fmt(formatter),
            Self::TerminalPpoWire(source) => source.fmt(formatter),
            Self::ReplayTerminalPpo(source) => source.fmt(formatter),
            Self::CheckpointWire(source) => source.fmt(formatter),
            Self::Client(source) => source.fmt(formatter),
            Self::CheckpointMetadata(source) => source.fmt(formatter),
            Self::ActorCheckpointMetadata(source) => source.fmt(formatter),
            Self::Commit(source) => source.fmt(formatter),
            Self::Origin(source) => source.fmt(formatter),
            Self::Metrics(source) => source.fmt(formatter),
            Self::Io(source) => write!(formatter, "learner I/O failed: {source}"),
        }
    }
}

impl std::error::Error for LearnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Snapshot(source) => Some(source),
            Self::Dataset(source) => Some(source),
            Self::ReplayPreload(source) => Some(source),
            Self::Sampler(source) => Some(source),
            Self::InferenceWire(source) => Some(source),
            Self::TrainingWire(source) => Some(source),
            Self::TerminalPpoWire(source) => Some(source),
            Self::ReplayTerminalPpo(source) => Some(source),
            Self::CheckpointWire(source) => Some(source),
            Self::Client(source) => Some(source),
            Self::CheckpointMetadata(source) | Self::ActorCheckpointMetadata(source) => {
                Some(source)
            }
            Self::Commit(source) => Some(source),
            Self::Origin(source) => Some(source),
            Self::Metrics(source) => Some(source),
            Self::Io(source) => Some(source),
            _ => None,
        }
    }
}

fn encode_digest(digest: [u8; 32]) -> String {
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(encoded, "{byte:02x}").unwrap();
    }
    encoded
}

#[cfg(test)]
mod rule_profile_tests {
    use super::*;

    #[test]
    fn legacy_learner_rejects_v2_before_creating_origin_or_opening_service() {
        let directory = std::env::temp_dir().join(format!(
            "paisho-legacy-learner-rules-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        for rules in [
            RuleProfileId::SkudPaiSho2022,
            RuleProfileId::SkudPaiSho2022V2,
        ] {
            let snapshot_path = directory.join(format!("{rules}.psrsnap"));
            ReplaySnapshotV1::with_rules(Vec::new(), rules)
                .unwrap()
                .write_new(&snapshot_path)
                .unwrap();
            let configuration = LearnerConfiguration {
                service_executable: directory.join("missing-service"),
                replay_snapshot: snapshot_path,
                replay_directory: directory.clone(),
                run_directory: directory.join(format!("run-{rules}")),
                initial_checkpoint: None,
                network_preset: NetworkPreset::Micro,
                optimization: OptimizationLevel::Level1,
                batch_size: 8,
                legal_action_capacity: 1024,
                model_seed: 17,
                sampler_seed: 23,
                generation: 0,
                learning_rate: 1.0e-4,
                objective: LearnerObjectiveConfiguration::SupervisedPolicyValue,
                target_training_step: 1,
                checkpoint_interval: 1,
            };
            let error = run_learner(&configuration).unwrap_err();
            if rules == RuleProfileId::SkudPaiSho2022 {
                // V1 reaches the existing empty-dataset validation, not a rules error.
                assert!(matches!(
                    error,
                    LearnerError::Dataset(ReplayDatasetV1Error::Empty)
                ));
            } else {
                assert!(
                    matches!(error, LearnerError::UnsupportedRuleProfile(actual) if actual == rules)
                );
            }
            assert_eq!(
                fs::read_dir(&configuration.run_directory).unwrap().count(),
                0
            );
        }
        fs::remove_dir_all(directory).unwrap();
    }
}
