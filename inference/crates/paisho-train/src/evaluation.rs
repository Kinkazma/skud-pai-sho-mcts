use core::fmt;

use paisho_ai::{
    Agent, AgentError, AgentTelemetry, HeuristicWeights, MctsAgent, MctsConfig, NetworkPolicy,
    RandomAgent, SiteBotV1, EXHAUSTIVE_ACTION_RANKING, SITE_BOT_V1_SOURCE_COMMIT,
};
use paisho_core::{Action, Position};
use paisho_rating::{
    evaluate_promotion_sprt, AgentId, PentanomialCounts, PromotionDecision, PromotionSprtConfig,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{NeutralStartConfigurationV1, PromotionNeutralStartV1, PromotionSamplingPolicyV1};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvaluationInferenceClassV1 {
    pub legal_action_capacity: usize,
    pub batch_size: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "family", rename_all = "kebab-case")]
pub enum EvaluationOpponentV1 {
    Random,
    SiteBotV1,
    Mcts { simulations: usize },
}

impl EvaluationOpponentV1 {
    pub fn parse(value: &str) -> Result<Self, EvaluationIdentityError> {
        match value {
            "random" => Ok(Self::Random),
            "site" | "site-bot-v1" => Ok(Self::SiteBotV1),
            _ => {
                let simulations = value
                    .strip_prefix("mcts:")
                    .or_else(|| value.strip_prefix("mcts-"))
                    .ok_or_else(|| EvaluationIdentityError::InvalidOpponent(value.to_owned()))?
                    .parse::<usize>()
                    .map_err(|_| EvaluationIdentityError::InvalidOpponent(value.to_owned()))?;
                let opponent = Self::Mcts { simulations };
                opponent.validate()?;
                Ok(opponent)
            }
        }
    }

    pub fn validate(self) -> Result<(), EvaluationIdentityError> {
        if let Self::Mcts { simulations } = self {
            if simulations == 0 || simulations > 512 {
                return Err(EvaluationIdentityError::InvalidMctsBudget(simulations));
            }
        }
        Ok(())
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::SiteBotV1 => "site-bot-v1",
            Self::Mcts { .. } => "mcts",
        }
    }

    pub fn display_label(self) -> String {
        match self {
            Self::Mcts { simulations } => format!("mcts-{simulations}"),
            _ => self.label().to_owned(),
        }
    }

    pub fn descriptor(self, implementation_sha256: &str) -> String {
        match self {
            Self::Random => format!(
                "family=random;implementation_sha256={implementation_sha256};rule_profile=skud-pai-sho-2022-03-14;policy=uniform-legal;seed_policy=paired-seat-v1"
            ),
            Self::SiteBotV1 => format!(
                "family=site-bot-v1;implementation_sha256={implementation_sha256};upstream_revision={SITE_BOT_V1_SOURCE_COMMIT};rule_profile=skud-pai-sho-2022-03-14;policy=source-compatible-one-ply;seed_policy=paired-seat-v1"
            ),
            Self::Mcts { simulations } => {
                let configuration = evaluation_mcts_configuration(simulations);
                let weights = configuration.heuristic_weights;
                format!(
                    "family=heuristic-mcts;implementation_sha256={implementation_sha256};rule_profile=skud-pai-sho-2022-03-14;simulations={};independent_trees={};maximum_tree_depth={};action_rank_batch=all;root_widening_bits={:08x};internal_widening_bits={:08x};rollout_depth={};exploration_bits={:08x};heuristic_bits={:08x},{:08x},{:08x},{:08x},{:08x};seed_policy=paired-seat-v1",
                    configuration.simulations,
                    configuration.independent_trees,
                    configuration.maximum_tree_depth,
                    configuration.root_widening_factor.to_bits(),
                    configuration.progressive_widening_factor.to_bits(),
                    configuration.rollout_depth,
                    configuration.exploration.to_bits(),
                    weights.harmony.to_bits(),
                    weights.midline_harmony.to_bits(),
                    weights.blooming_flower.to_bits(),
                    weights.total_flower.to_bits(),
                    weights.basic_reserve_progress.to_bits(),
                )
            }
        }
    }

    pub fn make(self, seed: u64) -> EvaluationOpponentAgent {
        match self {
            Self::Random => EvaluationOpponentAgent::Random(RandomAgent::new(seed)),
            Self::SiteBotV1 => EvaluationOpponentAgent::Site(SiteBotV1::new(seed)),
            Self::Mcts { simulations } => EvaluationOpponentAgent::Mcts(Box::new(
                MctsAgent::new(seed, evaluation_mcts_configuration(simulations))
                    .expect("a validated evaluation MCTS configuration remains valid"),
            )),
        }
    }
}

pub enum EvaluationOpponentAgent {
    Random(RandomAgent),
    Site(SiteBotV1),
    Mcts(Box<MctsAgent>),
}

impl Agent for EvaluationOpponentAgent {
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        match self {
            Self::Random(agent) => agent.select_action(position, legal_actions),
            Self::Site(agent) => agent.select_action(position, legal_actions),
            Self::Mcts(agent) => agent.select_action(position, legal_actions),
        }
    }

    fn telemetry(&self) -> AgentTelemetry {
        match self {
            Self::Random(agent) => agent.telemetry(),
            Self::Site(agent) => agent.telemetry(),
            Self::Mcts(agent) => agent.telemetry(),
        }
    }

    fn reset_telemetry(&mut self) {
        match self {
            Self::Random(agent) => agent.reset_telemetry(),
            Self::Site(agent) => agent.reset_telemetry(),
            Self::Mcts(agent) => agent.reset_telemetry(),
        }
    }
}

pub fn evaluation_mcts_configuration(simulations: usize) -> MctsConfig {
    MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 96,
        action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationRunIdentityV1 {
    pub source_sha256: String,
    pub source_revision: String,
    pub source_dirty: bool,
    pub service_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub opponent: EvaluationOpponentV1,
    pub opponent_descriptor: String,
    pub opponent_sha256: String,
    pub preset: String,
    pub optimization_level: u8,
    pub inference_classes: Vec<EvaluationInferenceClassV1>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lower_elo: Option<f64>,
    pub alpha: f64,
    pub beta: f64,
}

impl EvaluationRunIdentityV1 {
    pub fn validate(&self) -> Result<(), EvaluationIdentityError> {
        for (name, value) in [
            ("source SHA-256", self.source_sha256.as_str()),
            ("service SHA-256", self.service_sha256.as_str()),
            (
                "candidate checkpoint SHA-256",
                self.candidate_checkpoint_sha256.as_str(),
            ),
            ("opponent SHA-256", self.opponent_sha256.as_str()),
        ] {
            if !is_lower_hex_digest(value) {
                return Err(EvaluationIdentityError::InvalidDigest(name));
            }
        }
        if self.source_revision.is_empty() {
            return Err(EvaluationIdentityError::EmptySourceRevision);
        }
        self.opponent.validate()?;
        let expected_descriptor = self.opponent.descriptor(&self.source_sha256);
        if self.opponent_descriptor != expected_descriptor
            || self.opponent_sha256 != sha256_text(&expected_descriptor)
        {
            return Err(EvaluationIdentityError::OpponentIdentityMismatch);
        }
        if !matches!(self.preset.as_str(), "pure" | "micro") {
            return Err(EvaluationIdentityError::InvalidPreset);
        }
        if self.optimization_level > 1 {
            return Err(EvaluationIdentityError::InvalidOptimizationLevel);
        }
        if self.inference_classes.is_empty() {
            return Err(EvaluationIdentityError::NoInferenceClass);
        }
        let mut previous_capacity = 0;
        for class in &self.inference_classes {
            if class.legal_action_capacity == 0
                || class.batch_size == 0
                || class.legal_action_capacity <= previous_capacity
            {
                return Err(EvaluationIdentityError::InvalidInferenceClasses);
            }
            previous_capacity = class.legal_action_capacity;
        }
        if self.workers == 0
            || self.pairs_per_batch == 0
            || self.maximum_attempted_pairs == 0
            || self.maximum_eligible_pairs == 0
            || self.decision_soft_limit == 0
        {
            return Err(EvaluationIdentityError::InvalidExecutionLimit);
        }
        if self.maximum_eligible_pairs > self.maximum_attempted_pairs {
            return Err(EvaluationIdentityError::InvalidPairBudgets);
        }
        let final_pair_id = self
            .first_pair_id
            .checked_add(self.maximum_attempted_pairs - 1)
            .ok_or(EvaluationIdentityError::IdentifierOverflow)?;
        final_pair_id
            .checked_mul(2)
            .and_then(|value| value.checked_add(1))
            .ok_or(EvaluationIdentityError::IdentifierOverflow)?;
        self.neutral_start_configuration()?;
        self.network_policy()?;
        self.sprt_configuration()?;
        self.lower_sprt_configuration()?;
        Ok(())
    }

    pub fn neutral_start_configuration(
        &self,
    ) -> Result<Option<NeutralStartConfigurationV1>, EvaluationIdentityError> {
        self.neutral_start
            .map(PromotionNeutralStartV1::configuration)
            .transpose()
            .map_err(|source| EvaluationIdentityError::InvalidNeutralStart(source.to_string()))
    }

    pub fn network_policy(&self) -> Result<NetworkPolicy, EvaluationIdentityError> {
        self.sampling_policy
            .map(PromotionSamplingPolicyV1::network_policy)
            .unwrap_or(Ok(NetworkPolicy::Argmax))
            .map_err(|source| EvaluationIdentityError::InvalidSamplingPolicy(source.to_string()))
    }

    pub fn sprt_configuration(&self) -> Result<PromotionSprtConfig, EvaluationIdentityError> {
        PromotionSprtConfig::new(self.elo0, self.elo1, self.alpha, self.beta)
            .map_err(|source| EvaluationIdentityError::InvalidSprt(source.to_string()))
    }

    pub fn lower_sprt_configuration(
        &self,
    ) -> Result<Option<PromotionSprtConfig>, EvaluationIdentityError> {
        self.lower_elo
            .map(|lower| PromotionSprtConfig::new(lower, self.elo0, self.alpha, self.beta))
            .transpose()
            .map_err(|source| EvaluationIdentityError::InvalidSprt(source.to_string()))
    }

    pub fn candidate_agent_id(&self) -> AgentId {
        let policy = match self.sampling_policy {
            None => "argmax".to_owned(),
            Some(policy) => format!(
                "sample;temperature_bits={:08x};uniform_mix_bits={:08x}",
                policy.temperature_bits, policy.uniform_mix_bits
            ),
        };
        let descriptor = format!(
            "family=pure-network;checkpoint_sha256={};preset={};policy={policy};rule_profile=skud-pai-sho-2022-03-14",
            self.candidate_checkpoint_sha256, self.preset,
        );
        AgentId::new(format!("sha256:{}", sha256_text(&descriptor)))
            .expect("a SHA-256 identifier is valid")
    }

    pub fn opponent_agent_id(&self) -> AgentId {
        AgentId::new(format!("sha256:{}", self.opponent_sha256))
            .expect("a validated SHA-256 identifier is valid")
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EvaluationConclusion {
    SupportsUpperEloHypothesis,
    SupportsLowerEloHypothesis,
    SupportsCentralEloWindow,
    InconclusiveMaximumEligiblePairs,
    InconclusiveMaximumAttemptedPairs,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EvaluationProgress {
    pub batches: u64,
    pub attempted_pairs: u64,
    pub eligible_pairs: u64,
    pub excluded_pairs: u64,
    pub pentanomial: PentanomialCounts,
    pub candidate_inference_positions: u64,
    pub maximum_workers_observed: usize,
}

impl EvaluationProgress {
    pub fn next_pair_id(
        &self,
        identity: &EvaluationRunIdentityV1,
    ) -> Result<u64, EvaluationIdentityError> {
        identity
            .first_pair_id
            .checked_add(self.attempted_pairs)
            .ok_or(EvaluationIdentityError::IdentifierOverflow)
    }

    pub fn conclusion(
        &self,
        identity: &EvaluationRunIdentityV1,
    ) -> Result<Option<EvaluationConclusion>, EvaluationIdentityError> {
        let upper = evaluate_promotion_sprt(self.pentanomial, identity.sprt_configuration()?)
            .map_err(|source| EvaluationIdentityError::InvalidSprt(source.to_string()))?;
        let lower = identity
            .lower_sprt_configuration()?
            .map(|configuration| evaluate_promotion_sprt(self.pentanomial, configuration))
            .transpose()
            .map_err(|source| EvaluationIdentityError::InvalidSprt(source.to_string()))?;
        let evidence = window_conclusion(upper.decision, lower.map(|report| report.decision));
        Ok(evidence.or({
            if self.eligible_pairs >= identity.maximum_eligible_pairs {
                Some(EvaluationConclusion::InconclusiveMaximumEligiblePairs)
            } else if self.attempted_pairs >= identity.maximum_attempted_pairs {
                Some(EvaluationConclusion::InconclusiveMaximumAttemptedPairs)
            } else {
                None
            }
        }))
    }
}

fn window_conclusion(
    upper: PromotionDecision,
    lower: Option<PromotionDecision>,
) -> Option<EvaluationConclusion> {
    match (upper, lower) {
        (PromotionDecision::PromoteCandidate, _) => {
            Some(EvaluationConclusion::SupportsUpperEloHypothesis)
        }
        (_, Some(PromotionDecision::RejectCandidate)) => {
            Some(EvaluationConclusion::SupportsLowerEloHypothesis)
        }
        (PromotionDecision::RejectCandidate, Some(PromotionDecision::PromoteCandidate)) => {
            Some(EvaluationConclusion::SupportsCentralEloWindow)
        }
        // The legacy one-sided protocol calls support for elo0 the lower
        // hypothesis. A two-sided window waits for the independent lower
        // comparison before calling the centre supported.
        (PromotionDecision::RejectCandidate, None) => {
            Some(EvaluationConclusion::SupportsLowerEloHypothesis)
        }
        _ => None,
    }
}

pub fn contextual_elo_from_score(score: f64) -> Option<f64> {
    (score.is_finite() && score > 0.0 && score < 1.0)
        .then(|| 400.0 * (score / (1.0 - score)).log10())
}

pub fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex_digest(hasher.finalize().into())
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing a digest to String cannot fail");
    }
    text
}

fn is_lower_hex_digest(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EvaluationIdentityError {
    InvalidDigest(&'static str),
    EmptySourceRevision,
    InvalidOpponent(String),
    InvalidMctsBudget(usize),
    OpponentIdentityMismatch,
    InvalidPreset,
    InvalidOptimizationLevel,
    NoInferenceClass,
    InvalidInferenceClasses,
    InvalidExecutionLimit,
    InvalidPairBudgets,
    IdentifierOverflow,
    InvalidNeutralStart(String),
    InvalidSamplingPolicy(String),
    InvalidSprt(String),
}

impl fmt::Display for EvaluationIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDigest(name) => write!(formatter, "{name} is not a lowercase SHA-256"),
            Self::EmptySourceRevision => formatter.write_str("source revision cannot be empty"),
            Self::InvalidOpponent(value) => write!(formatter, "invalid opponent `{value}`"),
            Self::InvalidMctsBudget(value) => write!(
                formatter,
                "evaluation MCTS budget must be in 1..=512, got {value}"
            ),
            Self::OpponentIdentityMismatch => {
                formatter.write_str("opponent descriptor or digest is inconsistent")
            }
            Self::InvalidPreset => formatter.write_str("preset must be pure or micro"),
            Self::InvalidOptimizationLevel => {
                formatter.write_str("optimization level must be 0 or 1")
            }
            Self::NoInferenceClass => {
                formatter.write_str("evaluation needs at least one inference class")
            }
            Self::InvalidInferenceClasses => formatter
                .write_str("inference classes must be positive and strictly ordered by capacity"),
            Self::InvalidExecutionLimit => {
                formatter.write_str("evaluation execution limits must be positive")
            }
            Self::InvalidPairBudgets => {
                formatter.write_str("maximum eligible pairs cannot exceed maximum attempted pairs")
            }
            Self::IdentifierOverflow => formatter.write_str("evaluation identifiers overflow"),
            Self::InvalidNeutralStart(message) => {
                write!(formatter, "invalid neutral-start protocol: {message}")
            }
            Self::InvalidSamplingPolicy(message) => {
                write!(formatter, "invalid network sampling policy: {message}")
            }
            Self::InvalidSprt(message) => write!(formatter, "invalid Elo test: {message}"),
        }
    }
}

impl std::error::Error for EvaluationIdentityError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opponent_identity_pins_budget_and_implementation() {
        let first = EvaluationOpponentV1::Mcts { simulations: 32 }.descriptor(&"11".repeat(32));
        let larger = EvaluationOpponentV1::Mcts { simulations: 128 }.descriptor(&"11".repeat(32));
        let changed = EvaluationOpponentV1::Mcts { simulations: 32 }.descriptor(&"22".repeat(32));
        assert_ne!(sha256_text(&first), sha256_text(&larger));
        assert_ne!(sha256_text(&first), sha256_text(&changed));
        assert_eq!(
            EvaluationOpponentV1::parse("mcts:512")
                .unwrap()
                .display_label(),
            "mcts-512"
        );
        assert!(EvaluationOpponentV1::parse("mcts:513").is_err());
    }

    #[test]
    fn score_conversion_has_explicit_boundaries() {
        assert_eq!(contextual_elo_from_score(0.5), Some(0.0));
        assert!((contextual_elo_from_score(0.75).unwrap() - 190.848_501_887).abs() < 1e-9);
        assert_eq!(contextual_elo_from_score(0.0), None);
        assert_eq!(contextual_elo_from_score(1.0), None);
    }

    #[test]
    fn two_sided_window_needs_both_central_boundaries() {
        assert_eq!(
            window_conclusion(
                PromotionDecision::PromoteCandidate,
                Some(PromotionDecision::Continue)
            ),
            Some(EvaluationConclusion::SupportsUpperEloHypothesis)
        );
        assert_eq!(
            window_conclusion(
                PromotionDecision::Continue,
                Some(PromotionDecision::RejectCandidate)
            ),
            Some(EvaluationConclusion::SupportsLowerEloHypothesis)
        );
        assert_eq!(
            window_conclusion(
                PromotionDecision::RejectCandidate,
                Some(PromotionDecision::PromoteCandidate)
            ),
            Some(EvaluationConclusion::SupportsCentralEloWindow)
        );
        assert_eq!(
            window_conclusion(
                PromotionDecision::RejectCandidate,
                Some(PromotionDecision::Continue)
            ),
            None
        );
        assert_eq!(
            window_conclusion(PromotionDecision::RejectCandidate, None),
            Some(EvaluationConclusion::SupportsLowerEloHypothesis)
        );
    }
}
