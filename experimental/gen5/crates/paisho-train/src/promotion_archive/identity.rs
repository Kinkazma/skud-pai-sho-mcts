use paisho_ai::NetworkPolicy;
use paisho_rating::{
    evaluate_promotion_sprt, PentanomialCounts, PromotionDecision, PromotionSprtConfig,
};
use serde::{Deserialize, Serialize};

use crate::{NeutralStartConfigurationV1, PromotionArchiveError};

use super::{invalid, is_lower_hex_digest};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromotionInferenceClassV1 {
    pub legal_action_capacity: usize,
    pub batch_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromotionNeutralStartV1 {
    pub target_remaining_decisions: usize,
    pub seed: u64,
    pub source_decision_limit: usize,
    pub maximum_source_attempts: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PromotionSamplingPolicyV1 {
    pub temperature_bits: u32,
    pub uniform_mix_bits: u32,
}

impl PromotionSamplingPolicyV1 {
    pub fn new(temperature: f32, uniform_mix: f32) -> Result<Self, PromotionArchiveError> {
        let policy = Self {
            temperature_bits: temperature.to_bits(),
            uniform_mix_bits: uniform_mix.to_bits(),
        };
        policy.network_policy()?;
        Ok(policy)
    }

    pub fn network_policy(self) -> Result<NetworkPolicy, PromotionArchiveError> {
        NetworkPolicy::Sample {
            temperature: f32::from_bits(self.temperature_bits),
            uniform_mix: f32::from_bits(self.uniform_mix_bits),
        }
        .validate()
        .map_err(|source| invalid(format!("invalid promotion sampling policy: {source}")))
    }
}

impl PromotionNeutralStartV1 {
    pub fn configuration(self) -> Result<NeutralStartConfigurationV1, PromotionArchiveError> {
        NeutralStartConfigurationV1::new(
            self.seed,
            self.target_remaining_decisions,
            self.source_decision_limit,
            self.maximum_source_attempts,
        )
        .map_err(|source| {
            invalid(format!(
                "invalid promotion neutral-start protocol: {source}"
            ))
        })
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct PromotionRunIdentityV1 {
    pub source_sha256: String,
    pub source_revision: String,
    pub source_dirty: bool,
    pub service_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub champion_checkpoint_sha256: String,
    pub preset: String,
    pub optimization_level: u8,
    pub inference_classes: Vec<PromotionInferenceClassV1>,
    pub workers: usize,
    pub pairs_per_batch: usize,
    pub maximum_attempted_pairs: u64,
    pub maximum_eligible_pairs: u64,
    pub first_pair_id: u64,
    pub decision_soft_limit: usize,
    pub maximum_batch_wait_microseconds: u64,
    pub model_seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neutral_start: Option<PromotionNeutralStartV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling_policy: Option<PromotionSamplingPolicyV1>,
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl PromotionRunIdentityV1 {
    pub fn validate(&self) -> Result<(), PromotionArchiveError> {
        for (name, value) in [
            ("source SHA-256", self.source_sha256.as_str()),
            ("service SHA-256", self.service_sha256.as_str()),
            (
                "candidate checkpoint SHA-256",
                self.candidate_checkpoint_sha256.as_str(),
            ),
            (
                "champion checkpoint SHA-256",
                self.champion_checkpoint_sha256.as_str(),
            ),
        ] {
            if !is_lower_hex_digest(value) {
                return Err(invalid(format!("{name} is not a lowercase SHA-256")));
            }
        }
        if self.source_revision.is_empty() {
            return Err(invalid("source revision cannot be empty"));
        }
        if !matches!(self.preset.as_str(), "pure" | "micro") {
            return Err(invalid("promotion preset must be pure or micro"));
        }
        if self.optimization_level > 1 {
            return Err(invalid("promotion optimization level must be 0 or 1"));
        }
        if self.inference_classes.is_empty() {
            return Err(invalid("promotion needs at least one inference class"));
        }
        let mut previous_capacity = 0;
        for class in &self.inference_classes {
            if class.legal_action_capacity == 0 || class.batch_size == 0 {
                return Err(invalid("inference capacities and batches must be positive"));
            }
            if class.legal_action_capacity <= previous_capacity {
                return Err(invalid(
                    "inference classes must be strictly ordered by capacity",
                ));
            }
            previous_capacity = class.legal_action_capacity;
        }
        if self.workers == 0
            || self.pairs_per_batch == 0
            || self.maximum_attempted_pairs == 0
            || self.maximum_eligible_pairs == 0
            || self.decision_soft_limit == 0
        {
            return Err(invalid("promotion execution limits must be positive"));
        }
        if self.maximum_eligible_pairs > self.maximum_attempted_pairs {
            return Err(invalid(
                "maximum eligible pairs cannot exceed maximum attempted pairs",
            ));
        }
        let final_pair_id = self
            .first_pair_id
            .checked_add(self.maximum_attempted_pairs - 1)
            .ok_or_else(|| invalid("promotion pair identifiers overflow"))?;
        final_pair_id
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("promotion game identifiers overflow"))?;
        self.neutral_start_configuration()?;
        self.network_policy()?;
        self.sprt_configuration()?;
        Ok(())
    }

    pub fn neutral_start_configuration(
        &self,
    ) -> Result<Option<NeutralStartConfigurationV1>, PromotionArchiveError> {
        self.neutral_start
            .map(PromotionNeutralStartV1::configuration)
            .transpose()
    }

    pub fn network_policy(&self) -> Result<NetworkPolicy, PromotionArchiveError> {
        self.sampling_policy
            .map(PromotionSamplingPolicyV1::network_policy)
            .unwrap_or(Ok(NetworkPolicy::Argmax))
    }

    pub fn sprt_configuration(&self) -> Result<PromotionSprtConfig, PromotionArchiveError> {
        PromotionSprtConfig::new(self.elo0, self.elo1, self.alpha, self.beta).map_err(Into::into)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromotionCampaignConclusion {
    PromoteCandidate,
    RejectCandidate,
    InconclusiveMaximumEligiblePairs,
    InconclusiveMaximumAttemptedPairs,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PromotionCampaignProgress {
    pub batches: u64,
    pub attempted_pairs: u64,
    pub eligible_pairs: u64,
    pub excluded_pairs: u64,
    pub pentanomial: PentanomialCounts,
    pub candidate_inference_positions: u64,
    pub champion_inference_positions: u64,
    pub maximum_workers_observed: usize,
}

impl PromotionCampaignProgress {
    pub fn next_pair_id(
        &self,
        identity: &PromotionRunIdentityV1,
    ) -> Result<u64, PromotionArchiveError> {
        identity
            .first_pair_id
            .checked_add(self.attempted_pairs)
            .ok_or_else(|| invalid("next promotion pair identifier overflows"))
    }

    pub fn conclusion(
        &self,
        identity: &PromotionRunIdentityV1,
    ) -> Result<Option<PromotionCampaignConclusion>, PromotionArchiveError> {
        let report = evaluate_promotion_sprt(self.pentanomial, identity.sprt_configuration()?)?;
        let conclusion = match report.decision {
            PromotionDecision::PromoteCandidate => {
                Some(PromotionCampaignConclusion::PromoteCandidate)
            }
            PromotionDecision::RejectCandidate => {
                Some(PromotionCampaignConclusion::RejectCandidate)
            }
            PromotionDecision::Continue
                if self.eligible_pairs >= identity.maximum_eligible_pairs =>
            {
                Some(PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs)
            }
            PromotionDecision::Continue
                if self.attempted_pairs >= identity.maximum_attempted_pairs =>
            {
                Some(PromotionCampaignConclusion::InconclusiveMaximumAttemptedPairs)
            }
            PromotionDecision::Continue => None,
        };
        Ok(conclusion)
    }
}
