use std::env;
use std::error::Error;
use std::path::PathBuf;

use paisho_ai::NetworkPolicy;
use paisho_model::TerminalPpoParametersV1;
use paisho_mpsgraph_client::{default_service_path, NetworkPreset, OptimizationLevel};
use paisho_rating::PromotionSprtConfig;
use paisho_train::CurriculumTierV1;

type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub struct ClassShape {
    pub capacity: usize,
    pub batch_size: usize,
}

#[derive(serde::Serialize)]
pub struct Options {
    pub campaign_directory: PathBuf,
    pub target_generation: u64,
    pub service: PathBuf,
    pub initial_checkpoint: Option<PathBuf>,
    pub actor_executable: PathBuf,
    pub learner_executable: PathBuf,
    pub promotion_executable: PathBuf,
    pub curriculum_directory: Option<PathBuf>,
    pub evaluation_executable: PathBuf,
    pub network_preset: NetworkPreset,
    pub optimization: OptimizationLevel,
    pub classes: Vec<ClassShape>,
    pub workers: usize,
    pub actors_per_round: usize,
    pub maximum_batch_wait_microseconds: u64,
    pub model_seed: u64,
    pub opponent: String,
    pub actor_target_games: usize,
    pub actor_maximum_attempts: usize,
    pub actor_decision_soft_limit: usize,
    pub actor_seed: u64,
    pub policy_temperature: f32,
    pub uniform_mix: f32,
    pub start_horizon: Option<usize>,
    pub start_seed: u64,
    pub start_source_decision_limit: usize,
    pub start_source_attempts: usize,
    pub learner_batch_size: usize,
    pub learner_action_capacity: usize,
    pub sampler_seed: u64,
    pub learning_rate: f32,
    pub training_steps_per_generation: u64,
    pub checkpoint_interval: u64,
    pub promotion_every: u64,
    pub evaluation_every: u64,
    pub ppo_clip: f32,
    pub ppo_value_weight: f32,
    pub ppo_entropy_weight: f32,
    pub promotion_pairs_per_batch: usize,
    pub promotion_maximum_attempted_pairs: u64,
    pub promotion_maximum_eligible_pairs: u64,
    pub promotion_decision_soft_limit: usize,
    pub promotion_start_horizon: Option<usize>,
    pub promotion_start_seed: u64,
    pub promotion_start_source_decision_limit: usize,
    pub promotion_start_source_attempts: usize,
    pub promotion_sampling_temperature: Option<f32>,
    pub promotion_sampling_uniform_mix: f32,
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
    pub curriculum_pairs_per_batch: usize,
    pub curriculum_maximum_attempted_pairs: u64,
    pub curriculum_maximum_eligible_pairs: u64,
    pub curriculum_decision_soft_limit: usize,
    pub curriculum_start_horizon: Option<usize>,
    pub curriculum_start_seed: u64,
    pub curriculum_start_source_decision_limit: usize,
    pub curriculum_start_source_attempts: usize,
    pub curriculum_sampling_temperature: Option<f32>,
    pub curriculum_sampling_uniform_mix: f32,
    pub curriculum_lower_elo: f64,
    pub curriculum_center_elo: f64,
    pub curriculum_upper_elo: f64,
    pub curriculum_alpha: f64,
    pub curriculum_beta: f64,
}

impl Options {
    pub fn parse() -> Result<Self, BoxError> {
        Self::parse_from(env::args().skip(1))
    }

    pub(crate) fn parse_from(arguments: impl Iterator<Item = String>) -> Result<Self, BoxError> {
        let available_workers = std::thread::available_parallelism()?.get();
        let repository = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let current_executable = env::current_exe()?;
        let binary_directory = current_executable
            .parent()
            .ok_or("generation executable has no parent directory")?;
        let mut campaign_directory = None;
        let mut target_generation = None;
        let mut service = default_service_path(&repository);
        let mut initial_checkpoint = None;
        let mut actor_executable = binary_directory.join("paisho-actors");
        let mut learner_executable = binary_directory.join("paisho-train");
        let mut promotion_executable = binary_directory.join("paisho-promote");
        let mut curriculum_directory = None;
        let mut evaluation_executable = binary_directory.join("paisho-evaluate");
        let mut network_preset = NetworkPreset::Pure;
        let mut optimization = OptimizationLevel::Level1;
        let mut classes = vec![
            ClassShape {
                capacity: 64,
                batch_size: 8,
            },
            ClassShape {
                capacity: 128,
                batch_size: 4,
            },
            ClassShape {
                capacity: 1_024,
                batch_size: 4,
            },
        ];
        let mut workers = available_workers;
        let mut actors_per_round = None;
        let mut maximum_batch_wait_microseconds = 5_000;
        let mut model_seed = 17;
        let mut opponent = "random".to_owned();
        let mut actor_target_games = 64;
        let mut actor_maximum_attempts = None;
        let mut actor_decision_soft_limit = 2_048;
        let mut actor_seed = 0x5041_4953_484f_2026;
        let mut policy_temperature: f32 = 1.0;
        let mut uniform_mix: f32 = 0.05;
        let mut start_horizon = Some(64);
        let mut start_seed = 0x4e45_5554_5241_4c31;
        let mut start_source_decision_limit = 16_384;
        let mut start_source_attempts = 16;
        let mut learner_batch_size = 64;
        let mut learner_action_capacity = 1_024;
        let mut sampler_seed = 0xC0FF_EE11;
        let mut learning_rate: f32 = 1.0e-4;
        let mut training_steps_per_generation = 100;
        let mut checkpoint_interval = 100;
        let mut promotion_every = 1;
        let mut evaluation_every = 1;
        let mut ppo_clip: f32 = 0.2;
        let mut ppo_value_weight: f32 = 0.5;
        let mut ppo_entropy_weight: f32 = 0.01;
        let mut promotion_pairs_per_batch = None;
        let mut promotion_maximum_attempted_pairs = 800;
        let mut promotion_maximum_eligible_pairs = 400;
        let mut promotion_decision_soft_limit = 2_048;
        let mut promotion_start_horizon = Some(64);
        let mut promotion_start_seed = 0x5052_4f4d_4f54_4553;
        let mut promotion_start_source_decision_limit = 16_384;
        let mut promotion_start_source_attempts = 16;
        let mut promotion_sampling_temperature = Some(1.0);
        let mut promotion_sampling_uniform_mix = 0.05;
        let mut elo0: f64 = 0.0;
        let mut elo1: f64 = 10.0;
        let mut alpha: f64 = 0.05;
        let mut beta: f64 = 0.05;
        let mut curriculum_pairs_per_batch = None;
        let mut curriculum_maximum_attempted_pairs = 800;
        let mut curriculum_maximum_eligible_pairs = 400;
        let mut curriculum_decision_soft_limit = 2_048;
        let mut curriculum_start_horizon = Some(64);
        let mut curriculum_start_seed = 0x4355_5252_4943_554c;
        let mut curriculum_start_source_decision_limit = 16_384;
        let mut curriculum_start_source_attempts = 16;
        let mut curriculum_sampling_temperature = Some(1.0);
        let mut curriculum_sampling_uniform_mix = 0.05;
        let mut curriculum_lower_elo = -100.0;
        let mut curriculum_center_elo = 0.0;
        let mut curriculum_upper_elo = 100.0;
        let mut curriculum_alpha = 0.025;
        let mut curriculum_beta = 0.025;

        let arguments = arguments.collect::<Vec<_>>();
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            if flag == "--help" || flag == "-h" {
                print_usage();
                std::process::exit(0);
            }
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--campaign-dir" => campaign_directory = Some(PathBuf::from(value)),
                "--target-generation" => target_generation = Some(value.parse()?),
                "--service" => service = PathBuf::from(value),
                "--initial-checkpoint" => initial_checkpoint = Some(PathBuf::from(value)),
                "--actors-executable" => actor_executable = PathBuf::from(value),
                "--learner-executable" => learner_executable = PathBuf::from(value),
                "--promote-executable" => promotion_executable = PathBuf::from(value),
                "--curriculum-dir" => curriculum_directory = Some(PathBuf::from(value)),
                "--evaluate-executable" => evaluation_executable = PathBuf::from(value),
                "--preset" => network_preset = parse_preset(value)?,
                "--level" => optimization = parse_level(value)?,
                "--classes" => classes = parse_classes(value)?,
                "--workers" => workers = positive_usize(value, flag)?,
                "--actors" => actors_per_round = Some(positive_usize(value, flag)?),
                "--wait-us" => maximum_batch_wait_microseconds = value.parse()?,
                "--model-seed" => model_seed = value.parse()?,
                "--opponent" => opponent = value.clone(),
                "--actor-games" => actor_target_games = positive_usize(value, flag)?,
                "--actor-max-attempts" => {
                    actor_maximum_attempts = Some(positive_usize(value, flag)?)
                }
                "--actor-decision-limit" => {
                    actor_decision_soft_limit = positive_usize(value, flag)?
                }
                "--actor-seed" => actor_seed = value.parse()?,
                "--temperature" => policy_temperature = value.parse()?,
                "--uniform-mix" => uniform_mix = value.parse()?,
                "--start-horizon" => {
                    let value = value.parse::<usize>()?;
                    start_horizon = (value != 0).then_some(value);
                }
                "--start-seed" => start_seed = value.parse()?,
                "--start-source-limit" => {
                    start_source_decision_limit = positive_usize(value, flag)?
                }
                "--start-source-attempts" => start_source_attempts = positive_usize(value, flag)?,
                "--learner-batch" => learner_batch_size = positive_usize(value, flag)?,
                "--learner-actions" => learner_action_capacity = positive_usize(value, flag)?,
                "--sampler-seed" => sampler_seed = value.parse()?,
                "--learning-rate" => learning_rate = value.parse()?,
                "--steps-per-generation" => {
                    training_steps_per_generation = positive_u64(value, flag)?
                }
                "--checkpoint-every" => checkpoint_interval = positive_u64(value, flag)?,
                "--promotion-every" => promotion_every = positive_u64(value, flag)?,
                "--evaluation-every" => evaluation_every = positive_u64(value, flag)?,
                "--ppo-clip" => ppo_clip = value.parse()?,
                "--ppo-value-weight" => ppo_value_weight = value.parse()?,
                "--ppo-entropy-weight" => ppo_entropy_weight = value.parse()?,
                "--promotion-pairs-per-batch" => {
                    promotion_pairs_per_batch = Some(positive_usize(value, flag)?)
                }
                "--promotion-max-attempted" => {
                    promotion_maximum_attempted_pairs = positive_u64(value, flag)?
                }
                "--promotion-max-eligible" => {
                    promotion_maximum_eligible_pairs = positive_u64(value, flag)?
                }
                "--promotion-decision-limit" => {
                    promotion_decision_soft_limit = positive_usize(value, flag)?
                }
                "--promotion-start-horizon" => {
                    let value = value.parse::<usize>()?;
                    promotion_start_horizon = (value != 0).then_some(value);
                }
                "--promotion-start-seed" => promotion_start_seed = value.parse()?,
                "--promotion-start-source-limit" => {
                    promotion_start_source_decision_limit = positive_usize(value, flag)?
                }
                "--promotion-start-source-attempts" => {
                    promotion_start_source_attempts = positive_usize(value, flag)?
                }
                "--promotion-sampling-temperature" => {
                    let value = value.parse::<f32>()?;
                    promotion_sampling_temperature = (value != 0.0).then_some(value);
                }
                "--promotion-sampling-uniform-mix" => {
                    promotion_sampling_uniform_mix = value.parse()?
                }
                "--elo0" => elo0 = value.parse()?,
                "--elo1" => elo1 = value.parse()?,
                "--alpha" => alpha = value.parse()?,
                "--beta" => beta = value.parse()?,
                "--curriculum-pairs-per-batch" => {
                    curriculum_pairs_per_batch = Some(positive_usize(value, flag)?)
                }
                "--curriculum-max-attempted" => {
                    curriculum_maximum_attempted_pairs = positive_u64(value, flag)?
                }
                "--curriculum-max-eligible" => {
                    curriculum_maximum_eligible_pairs = positive_u64(value, flag)?
                }
                "--curriculum-decision-limit" => {
                    curriculum_decision_soft_limit = positive_usize(value, flag)?
                }
                "--curriculum-start-horizon" => {
                    let value = value.parse::<usize>()?;
                    curriculum_start_horizon = (value != 0).then_some(value);
                }
                "--curriculum-start-seed" => curriculum_start_seed = value.parse()?,
                "--curriculum-start-source-limit" => {
                    curriculum_start_source_decision_limit = positive_usize(value, flag)?
                }
                "--curriculum-start-source-attempts" => {
                    curriculum_start_source_attempts = positive_usize(value, flag)?
                }
                "--curriculum-sampling-temperature" => {
                    let value = value.parse::<f32>()?;
                    curriculum_sampling_temperature = (value != 0.0).then_some(value);
                }
                "--curriculum-sampling-uniform-mix" => {
                    curriculum_sampling_uniform_mix = value.parse()?
                }
                "--curriculum-lower-elo" => curriculum_lower_elo = value.parse()?,
                "--curriculum-center-elo" => curriculum_center_elo = value.parse()?,
                "--curriculum-upper-elo" => curriculum_upper_elo = value.parse()?,
                "--curriculum-alpha" => curriculum_alpha = value.parse()?,
                "--curriculum-beta" => curriculum_beta = value.parse()?,
                _ => return Err(format!("unknown option {flag}").into()),
            }
            index += 2;
        }
        let paired = opponent != "self";
        validate_opponent(&opponent)?;
        let actors_per_round = actors_per_round.unwrap_or_else(|| workers.saturating_mul(8));
        let actor_maximum_attempts = actor_maximum_attempts
            .unwrap_or_else(|| actor_target_games.saturating_mul(4).max(actors_per_round));
        let promotion_pairs_per_batch = promotion_pairs_per_batch.unwrap_or(workers);
        let curriculum_pairs_per_batch = curriculum_pairs_per_batch.unwrap_or(workers);
        if actor_maximum_attempts < actor_target_games {
            return Err("--actor-max-attempts cannot be smaller than --actor-games".into());
        }
        if paired
            && (actor_target_games % 2 != 0 || actors_per_round < 2 || actors_per_round % 2 != 0)
        {
            return Err("distinct opponents require even games and actor rounds".into());
        }
        let actor_maximum_attempts_u64 = u64::try_from(actor_maximum_attempts)
            .map_err(|_| "actor maximum attempts do not fit u64")?;
        if actor_maximum_attempts_u64 >= super::GENERATION_IDENTIFIER_STRIDE
            || promotion_maximum_attempted_pairs >= super::GENERATION_IDENTIFIER_STRIDE
            || curriculum_maximum_attempted_pairs >= super::GENERATION_IDENTIFIER_STRIDE / 2
        {
            return Err("per-generation identifiers exceed their reserved range".into());
        }
        NetworkPolicy::Sample {
            temperature: policy_temperature,
            uniform_mix,
        }
        .validate()?;
        TerminalPpoParametersV1::with_behavior(
            policy_temperature,
            uniform_mix,
            ppo_clip,
            ppo_value_weight,
            ppo_entropy_weight,
        )?;
        if !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("learning rate must be positive and finite".into());
        }
        PromotionSprtConfig::new(elo0, elo1, alpha, beta)?;
        PromotionSprtConfig::new(
            curriculum_center_elo,
            curriculum_upper_elo,
            curriculum_alpha,
            curriculum_beta,
        )?;
        PromotionSprtConfig::new(
            curriculum_lower_elo,
            curriculum_center_elo,
            curriculum_alpha,
            curriculum_beta,
        )?;
        if let Some(temperature) = promotion_sampling_temperature {
            NetworkPolicy::Sample {
                temperature,
                uniform_mix: promotion_sampling_uniform_mix,
            }
            .validate()?;
        }
        if promotion_maximum_attempted_pairs < promotion_maximum_eligible_pairs {
            return Err("promotion attempted-pair budget must cover eligible-pair budget".into());
        }
        if curriculum_maximum_attempted_pairs < curriculum_maximum_eligible_pairs {
            return Err("curriculum attempted-pair budget must cover eligible-pair budget".into());
        }
        if curriculum_sampling_temperature.is_some() {
            NetworkPolicy::Sample {
                temperature: curriculum_sampling_temperature
                    .expect("the branch checks that a temperature exists"),
                uniform_mix: curriculum_sampling_uniform_mix,
            }
            .validate()?;
        }
        if curriculum_directory.is_some() {
            CurriculumTierV1::parse(&opponent)?;
        }
        Ok(Self {
            campaign_directory: campaign_directory.ok_or("missing --campaign-dir DIR")?,
            target_generation: target_generation.ok_or("missing --target-generation N")?,
            service,
            initial_checkpoint,
            actor_executable,
            learner_executable,
            promotion_executable,
            curriculum_directory,
            evaluation_executable,
            network_preset,
            optimization,
            classes,
            workers,
            actors_per_round,
            maximum_batch_wait_microseconds,
            model_seed,
            opponent,
            actor_target_games,
            actor_maximum_attempts,
            actor_decision_soft_limit,
            actor_seed,
            policy_temperature,
            uniform_mix,
            start_horizon,
            start_seed,
            start_source_decision_limit,
            start_source_attempts,
            learner_batch_size,
            learner_action_capacity,
            sampler_seed,
            learning_rate,
            training_steps_per_generation,
            checkpoint_interval,
            promotion_every,
            evaluation_every,
            ppo_clip,
            ppo_value_weight,
            ppo_entropy_weight,
            promotion_pairs_per_batch,
            promotion_maximum_attempted_pairs,
            promotion_maximum_eligible_pairs,
            promotion_decision_soft_limit,
            promotion_start_horizon,
            promotion_start_seed,
            promotion_start_source_decision_limit,
            promotion_start_source_attempts,
            promotion_sampling_temperature,
            promotion_sampling_uniform_mix,
            elo0,
            elo1,
            alpha,
            beta,
            curriculum_pairs_per_batch,
            curriculum_maximum_attempted_pairs,
            curriculum_maximum_eligible_pairs,
            curriculum_decision_soft_limit,
            curriculum_start_horizon,
            curriculum_start_seed,
            curriculum_start_source_decision_limit,
            curriculum_start_source_attempts,
            curriculum_sampling_temperature,
            curriculum_sampling_uniform_mix,
            curriculum_lower_elo,
            curriculum_center_elo,
            curriculum_upper_elo,
            curriculum_alpha,
            curriculum_beta,
        })
    }
}

fn print_usage() {
    println!(
        "usage: paisho-generations --campaign-dir DIR --target-generation N [options]\n\
         --initial-checkpoint PATH --service PATH --preset pure|micro --level 0|1\n\
         --actors-executable PATH --learner-executable PATH --promote-executable PATH\n\
         --opponent random|site|self|mcts:N --actor-games N --actor-max-attempts N\n\
         --workers N --actors N --classes 64:8,128:4,1024:4 --wait-us N\n\
         --actor-decision-limit N --temperature F --uniform-mix F\n\
         --start-horizon N (0 disables) --start-source-limit N --start-source-attempts N\n\
         --learner-batch N --learner-actions N --steps-per-generation N\n\
         --learning-rate F --checkpoint-every N --ppo-clip F\n\
         --ppo-value-weight F --ppo-entropy-weight F\n\
         --promotion-every N --evaluation-every N (generations; default 1)\n\
         --promotion-pairs-per-batch N --promotion-max-attempted N\n\
         --promotion-max-eligible N --promotion-decision-limit N\n\
         --promotion-start-horizon N (0 disables) --promotion-start-seed N\n\
         --promotion-start-source-limit N --promotion-start-source-attempts N\n\
         --promotion-sampling-temperature F (0 uses argmax)\n\
         --promotion-sampling-uniform-mix F\n\
         --elo0 F --elo1 F --alpha F --beta F\n\
         --curriculum-dir DIR --evaluate-executable PATH\n\
         --curriculum-pairs-per-batch N --curriculum-max-attempted N\n\
         --curriculum-max-eligible N --curriculum-decision-limit N\n\
         --curriculum-start-horizon N (0 disables) --curriculum-start-seed N\n\
         --curriculum-start-source-limit N --curriculum-start-source-attempts N\n\
         --curriculum-sampling-temperature F (0 uses argmax)\n\
         --curriculum-sampling-uniform-mix F\n\
         --curriculum-lower-elo F --curriculum-center-elo F\n\
         --curriculum-upper-elo F --curriculum-alpha F --curriculum-beta F"
    );
}

fn parse_preset(value: &str) -> Result<NetworkPreset, BoxError> {
    match value {
        "micro" => Ok(NetworkPreset::Micro),
        "pure" => Ok(NetworkPreset::Pure),
        _ => Err(format!("invalid preset {value}").into()),
    }
}

fn parse_level(value: &str) -> Result<OptimizationLevel, BoxError> {
    match value {
        "0" => Ok(OptimizationLevel::Level0),
        "1" => Ok(OptimizationLevel::Level1),
        _ => Err(format!("invalid optimization level {value}").into()),
    }
}

fn parse_classes(value: &str) -> Result<Vec<ClassShape>, BoxError> {
    let classes = value
        .split(',')
        .map(|part| -> Result<ClassShape, BoxError> {
            let (capacity, batch_size) = part
                .split_once(':')
                .ok_or_else(|| format!("invalid class {part}; expected CAPACITY:BATCH"))?;
            Ok(ClassShape {
                capacity: positive_usize(capacity, "class capacity")?,
                batch_size: positive_usize(batch_size, "class batch")?,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if classes.is_empty() {
        return Err("--classes must contain at least one class".into());
    }
    for pair in classes.windows(2) {
        if pair[0].capacity >= pair[1].capacity {
            return Err("class capacities must be strictly increasing".into());
        }
    }
    Ok(classes)
}

fn validate_opponent(value: &str) -> Result<(), BoxError> {
    match value {
        "self" | "random" | "site" => Ok(()),
        _ => {
            let simulations = value
                .strip_prefix("mcts:")
                .ok_or_else(|| format!("invalid opponent {value}"))?;
            let simulations = positive_usize(simulations, "MCTS simulations")?;
            if simulations > 512 {
                Err("curriculum MCTS opponents are capped at 512 simulations".into())
            } else {
                Ok(())
            }
        }
    }
}

fn positive_usize(value: &str, flag: &str) -> Result<usize, BoxError> {
    let parsed = value
        .parse::<usize>()
        .map_err(|source| format!("invalid {flag}: {source}"))?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
    }
}

fn positive_u64(value: &str, flag: &str) -> Result<u64, BoxError> {
    let parsed = value
        .parse::<u64>()
        .map_err(|source| format!("invalid {flag}: {source}"))?;
    if parsed == 0 {
        Err(format!("{flag} must be positive").into())
    } else {
        Ok(parsed)
    }
}
