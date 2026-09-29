//! Existing independent promotion/Elo protocols, scheduled only at durable boundaries.
use super::{
    journal::{self, Block},
    options::Options,
    seed, BoxError,
};
use paisho_train::{assess_curriculum_evaluation, CurriculumTierV1, EvaluationCampaignArchive};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Assessment {
    pub generation: u64,
    pub candidate_sha256: String,
    pub selected: PathBuf,
    pub champion: PathBuf,
    pub tier: CurriculumTierV1,
    pub evaluation: Option<Value>,
    pub promotion: Option<String>,
}

pub fn due(generation: u64, options: &Options) -> bool {
    generation % options.promotion_every == 0 || generation % options.evaluation_every == 0
}

pub fn settle(
    root: &Path,
    options: &Options,
    block: &Block,
    champion_only: bool,
    mcts_ceiling: usize,
    hold_random: bool,
    site_min_win_rate: bool,
) -> Result<Assessment, BoxError> {
    let path = root
        .join("assessments")
        .join(format!("assessment-{:020}.json", block.generation));
    if path.exists() {
        let stored: Assessment = serde_json::from_slice(&fs::read(path)?)?;
        if stored.candidate_sha256 != block.checkpoint_sha256
            || stored.generation != block.generation
        {
            return Err("assessment does not belong to its durable block".into());
        }
        return Ok(stored);
    }
    let mut assessment = Assessment {
        generation: block.generation,
        candidate_sha256: block.checkpoint_sha256.clone(),
        selected: block.checkpoint.clone(),
        champion: block.champion.clone(),
        tier: block.tier,
        evaluation: None,
        promotion: None,
    };
    if !due(block.generation, options) {
        return Ok(assessment);
    }
    let directory = root
        .join("measurements")
        .join(format!("generation-{:020}", block.generation));
    if block.generation % options.promotion_every == 0 {
        let mut command = Command::new(&options.promotion_executable);
        common(
            &mut command,
            options,
            &directory.join("promotion"),
            &block.checkpoint,
            block.generation,
        );
        command
            .arg("--champion-checkpoint")
            .arg(&block.champion)
            .arg("--pairs-per-batch")
            .arg(options.promotion_pairs_per_batch.to_string())
            .arg("--max-attempted-pairs")
            .arg(options.promotion_maximum_attempted_pairs.to_string())
            .arg("--max-eligible-pairs")
            .arg(options.promotion_maximum_eligible_pairs.to_string())
            .arg("--decision-limit")
            .arg(options.promotion_decision_soft_limit.to_string())
            .arg("--elo0")
            .arg(options.elo0.to_string())
            .arg("--elo1")
            .arg(options.elo1.to_string())
            .arg("--alpha")
            .arg(options.alpha.to_string())
            .arg("--beta")
            .arg(options.beta.to_string());
        sampling_start(
            &mut command,
            options.promotion_sampling_temperature,
            options.promotion_sampling_uniform_mix,
            options.promotion_start_horizon,
            seed(options.promotion_start_seed, block.generation, 11),
            options.promotion_start_source_decision_limit,
            options.promotion_start_source_attempts,
        );
        let output = run(&mut command)?;
        let conclusion = output
            .lines()
            .find_map(|line| line.strip_prefix("conclusion="))
            .ok_or("promotion omitted its conclusion")?;
        apply_promotion(&mut assessment, block, conclusion, champion_only)?;
    }
    if block.generation % options.evaluation_every == 0 && block.tier != CurriculumTierV1::SelfPlay
    {
        let output_dir = directory.join("evaluation");
        let mut command = Command::new(&options.evaluation_executable);
        common(
            &mut command,
            options,
            &output_dir,
            &assessment.selected,
            block.generation,
        );
        command
            .arg("--opponent")
            .arg(block.tier.training_opponent())
            .arg("--pairs-per-batch")
            .arg(options.curriculum_pairs_per_batch.to_string())
            .arg("--max-attempted-pairs")
            .arg(options.curriculum_maximum_attempted_pairs.to_string())
            .arg("--max-eligible-pairs")
            .arg(options.curriculum_maximum_eligible_pairs.to_string())
            .arg("--decision-limit")
            .arg(options.curriculum_decision_soft_limit.to_string())
            .arg("--lower-elo")
            .arg(options.curriculum_lower_elo.to_string())
            .arg("--elo0")
            .arg(options.curriculum_center_elo.to_string())
            .arg("--elo1")
            .arg(options.curriculum_upper_elo.to_string())
            .arg("--alpha")
            .arg(options.curriculum_alpha.to_string())
            .arg("--beta")
            .arg(options.curriculum_beta.to_string());
        sampling_start(
            &mut command,
            options.curriculum_sampling_temperature,
            options.curriculum_sampling_uniform_mix,
            options.curriculum_start_horizon,
            seed(options.curriculum_start_seed, block.generation, 22),
            options.curriculum_start_source_decision_limit,
            options.curriculum_start_source_attempts,
        );
        run(&mut command)?;
        let archive = EvaluationCampaignArchive::open_existing_integrity(&output_dir)?;
        let analysis = archive.published_analysis_integrity()?;
        assessment.tier =
            assess_curriculum_evaluation(block.tier, archive.identity(), &analysis)?.tier_after;
        assessment.tier = bounded_tier(assessment.tier, mcts_ceiling);
        if site_min_win_rate
            && block.tier == CurriculumTierV1::Random
            && assessment.tier != CurriculumTierV1::Random
            && (analysis.attempted_pairs == 0
                || analysis.candidate_wins * 100 < analysis.attempted_pairs * 2 * 70)
        {
            assessment.tier = CurriculumTierV1::Random;
        }
        if hold_random {
            assessment.tier = CurriculumTierV1::Random;
        }
        assessment.evaluation = Some(serde_json::to_value(&analysis)?);
    }
    journal::write_new(&path, &assessment)?;
    Ok(assessment)
}

fn bounded_tier(tier: CurriculumTierV1, ceiling: usize) -> CurriculumTierV1 {
    if ceiling == 32 && matches!(tier, CurriculumTierV1::Mcts128 | CurriculumTierV1::Mcts512) {
        CurriculumTierV1::SelfPlay
    } else {
        tier
    }
}

pub fn learner_checkpoint(
    assessment: &Assessment,
    block: &Block,
    continue_inconclusive: bool,
) -> PathBuf {
    if continue_inconclusive && inconclusive(assessment) {
        block.checkpoint.clone()
    } else {
        assessment.selected.clone()
    }
}

pub fn inconclusive(assessment: &Assessment) -> bool {
    matches!(
        assessment.promotion.as_deref(),
        Some("InconclusiveMaximumEligiblePairs" | "InconclusiveMaximumAttemptedPairs")
    )
}

fn apply_promotion(
    assessment: &mut Assessment,
    block: &Block,
    conclusion: &str,
    champion_only: bool,
) -> Result<(), BoxError> {
    match conclusion {
        "PromoteCandidate" => assessment.champion = block.checkpoint.clone(),
        "RejectCandidate" => assessment.selected = block.champion.clone(),
        "InconclusiveMaximumEligiblePairs" | "InconclusiveMaximumAttemptedPairs" => {
            if champion_only {
                assessment.selected = block.champion.clone();
            }
        }
        _ => return Err(format!("unknown promotion result: {conclusion}").into()),
    }
    assessment.promotion = Some(conclusion.into());
    Ok(())
}

fn common(
    command: &mut Command,
    options: &Options,
    directory: &Path,
    candidate: &Path,
    generation: u64,
) {
    command
        .arg("--output-dir")
        .arg(directory)
        .arg("--service")
        .arg(&options.service)
        .arg("--candidate-checkpoint")
        .arg(candidate)
        .arg("--preset")
        .arg(match options.network_preset {
            paisho_mpsgraph_client::NetworkPreset::Pure => "pure",
            _ => "micro",
        })
        .arg("--level")
        .arg(match options.optimization {
            paisho_mpsgraph_client::OptimizationLevel::Level1 => "1",
            _ => "0",
        })
        .arg("--classes")
        .arg(
            options
                .classes
                .iter()
                .map(|c| format!("{}:{}", c.capacity, c.batch_size))
                .collect::<Vec<_>>()
                .join(","),
        )
        .arg("--workers")
        .arg(options.workers.to_string())
        .arg("--wait-us")
        .arg(options.maximum_batch_wait_microseconds.to_string())
        .arg("--model-seed")
        .arg(seed(options.model_seed, generation, 33).to_string())
        .arg("--first-pair-id")
        .arg(
            generation
                .saturating_mul(1_000_000_000)
                .saturating_add(500_000_000)
                .to_string(),
        );
}

fn sampling_start(
    command: &mut Command,
    temperature: Option<f32>,
    mix: f32,
    horizon: Option<usize>,
    seed: u64,
    limit: usize,
    attempts: usize,
) {
    command
        .arg("--sampling-temperature")
        .arg(temperature.unwrap_or(0.0).to_string())
        .arg("--sampling-uniform-mix")
        .arg(mix.to_string());
    command
        .arg("--start-horizon")
        .arg(horizon.unwrap_or(0).to_string());
    if horizon.is_some() {
        command
            .arg("--start-seed")
            .arg(seed.to_string())
            .arg("--start-source-limit")
            .arg(limit.to_string())
            .arg("--start-source-attempts")
            .arg(attempts.to_string());
    }
}

fn run(command: &mut Command) -> Result<String, BoxError> {
    let output = command.output()?;
    let stdout = String::from_utf8(output.stdout)?;
    print!("{stdout}");
    if !output.status.success() {
        return Err(format!(
            "measurement failed: {}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }
    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceiling_skips_heavy_mcts_but_preserves_legacy() {
        for tier in [CurriculumTierV1::Mcts128, CurriculumTierV1::Mcts512] {
            assert_eq!(bounded_tier(tier, 32), CurriculumTierV1::SelfPlay);
            assert_eq!(bounded_tier(tier, 512), tier);
        }
        assert_eq!(
            bounded_tier(CurriculumTierV1::Mcts32, 32),
            CurriculumTierV1::Mcts32
        );
    }

    #[test]
    fn only_positive_evidence_replaces_champion_in_v5() {
        let block = Block {
            generation: 10,
            checkpoint: "candidate".into(),
            checkpoint_sha256: "hash".into(),
            training_step: 256,
            champion: "incumbent".into(),
            tier: CurriculumTierV1::Random,
            games: 512,
            examples: 16384,
            attempt_directory: "attempt".into(),
        };
        for strict in [false, true] {
            for conclusion in [
                "PromoteCandidate",
                "RejectCandidate",
                "InconclusiveMaximumEligiblePairs",
                "InconclusiveMaximumAttemptedPairs",
            ] {
                let mut assessment = Assessment {
                    generation: 10,
                    candidate_sha256: "hash".into(),
                    selected: block.checkpoint.clone(),
                    champion: block.champion.clone(),
                    tier: block.tier,
                    evaluation: None,
                    promotion: None,
                };
                apply_promotion(&mut assessment, &block, conclusion, strict).unwrap();
                let restored: Assessment =
                    serde_json::from_slice(&serde_json::to_vec(&assessment).unwrap()).unwrap();
                for continuation in [false, true] {
                    let exploratory = continuation && conclusion.starts_with("Inconclusive");
                    assert_eq!(
                        learner_checkpoint(&restored, &block, continuation),
                        if exploratory {
                            block.checkpoint.clone()
                        } else {
                            restored.selected.clone()
                        }
                    );
                }
                let promoted = conclusion == "PromoteCandidate";
                assert_eq!(
                    assessment.champion,
                    if promoted {
                        &block.checkpoint
                    } else {
                        &block.champion
                    }
                    .clone()
                );
                let retained = conclusion == "RejectCandidate" || (strict && !promoted);
                assert_eq!(
                    assessment.selected,
                    if retained {
                        &block.champion
                    } else {
                        &block.checkpoint
                    }
                    .clone()
                );
            }
        }
    }

    #[test]
    fn argmax_is_explicit_instead_of_inheriting_evaluators_sampling_default() {
        for (temperature, expected) in [(None, "0"), (Some(1.0), "1")] {
            let mut command = Command::new("unused-test-program");
            sampling_start(&mut command, temperature, 0.05, None, 17, 16384, 16);
            let arguments = command
                .get_args()
                .map(|a| a.to_str().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(arguments[0..2], ["--sampling-temperature", expected]);
            assert_eq!(arguments[4..6], ["--start-horizon", "0"]);
        }
    }
}
