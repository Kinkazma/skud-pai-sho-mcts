use core::fmt;

use serde::{Deserialize, Serialize};

use crate::{
    EvaluationAnalysisV1, EvaluationConclusion, EvaluationOpponentV1, EvaluationRunIdentityV1,
};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CurriculumTierV1 {
    Random,
    SiteBotV1,
    Mcts8,
    Mcts32,
    Mcts128,
    Mcts512,
    SelfPlay,
}

impl CurriculumTierV1 {
    pub fn parse(value: &str) -> Result<Self, CurriculumDecisionError> {
        match value {
            "random" => Ok(Self::Random),
            "site" | "site-bot-v1" => Ok(Self::SiteBotV1),
            "mcts:8" | "mcts-8" => Ok(Self::Mcts8),
            "mcts:32" | "mcts-32" => Ok(Self::Mcts32),
            "mcts:128" | "mcts-128" => Ok(Self::Mcts128),
            "mcts:512" | "mcts-512" => Ok(Self::Mcts512),
            "self" | "self-play" => Ok(Self::SelfPlay),
            _ => Err(CurriculumDecisionError::InvalidTier(value.to_owned())),
        }
    }

    pub const fn next(self) -> Self {
        match self {
            Self::Random => Self::SiteBotV1,
            Self::SiteBotV1 => Self::Mcts8,
            Self::Mcts8 => Self::Mcts32,
            Self::Mcts32 => Self::Mcts128,
            Self::Mcts128 => Self::Mcts512,
            Self::Mcts512 | Self::SelfPlay => Self::SelfPlay,
        }
    }

    pub const fn previous(self) -> Option<Self> {
        match self {
            Self::Random => None,
            Self::SiteBotV1 => Some(Self::Random),
            Self::Mcts8 => Some(Self::SiteBotV1),
            Self::Mcts32 => Some(Self::Mcts8),
            Self::Mcts128 => Some(Self::Mcts32),
            Self::Mcts512 => Some(Self::Mcts128),
            Self::SelfPlay => Some(Self::Mcts512),
        }
    }

    pub const fn fixed_opponent(self) -> Option<EvaluationOpponentV1> {
        match self {
            Self::Random => Some(EvaluationOpponentV1::Random),
            Self::SiteBotV1 => Some(EvaluationOpponentV1::SiteBotV1),
            Self::Mcts8 => Some(EvaluationOpponentV1::Mcts { simulations: 8 }),
            Self::Mcts32 => Some(EvaluationOpponentV1::Mcts { simulations: 32 }),
            Self::Mcts128 => Some(EvaluationOpponentV1::Mcts { simulations: 128 }),
            Self::Mcts512 => Some(EvaluationOpponentV1::Mcts { simulations: 512 }),
            Self::SelfPlay => None,
        }
    }

    pub const fn training_opponent(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::SiteBotV1 => "site",
            Self::Mcts8 => "mcts:8",
            Self::Mcts32 => "mcts:32",
            Self::Mcts128 => "mcts:128",
            Self::Mcts512 => "mcts:512",
            Self::SelfPlay => "self",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::SiteBotV1 => "site-bot-v1",
            Self::Mcts8 => "mcts-8",
            Self::Mcts32 => "mcts-32",
            Self::Mcts128 => "mcts-128",
            Self::Mcts512 => "mcts-512",
            Self::SelfPlay => "self-play",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CurriculumActionV1 {
    Advance,
    Hold,
    Retreat,
    ContinueSelfPlay,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CurriculumReasonV1 {
    UpperEloHypothesisSupported,
    LowerEloHypothesisSupported,
    LowerEloHypothesisAtFloor,
    CentralEloWindowSupported,
    EvaluationInconclusive,
    ScheduledEvaluationDeferred,
    LegacyOneSidedEvidence,
    SelfPlayPromotionHandledSeparately,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct CurriculumEloWindowV1 {
    pub lower_elo: f64,
    pub center_elo: f64,
    pub upper_elo: f64,
    pub alpha: f64,
    pub beta: f64,
}

impl CurriculumEloWindowV1 {
    pub fn from_evaluation(
        identity: &EvaluationRunIdentityV1,
    ) -> Result<Option<Self>, CurriculumDecisionError> {
        let Some(lower_elo) = identity.lower_elo else {
            return Ok(None);
        };
        if !(lower_elo < identity.elo0 && identity.elo0 < identity.elo1) {
            return Err(CurriculumDecisionError::InvalidEloWindow);
        }
        Ok(Some(Self {
            lower_elo,
            center_elo: identity.elo0,
            upper_elo: identity.elo1,
            alpha: identity.alpha,
            beta: identity.beta,
        }))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CurriculumAssessmentV1 {
    pub action: CurriculumActionV1,
    pub reason: CurriculumReasonV1,
    pub tier_after: CurriculumTierV1,
}

pub fn assess_curriculum_evaluation(
    tier: CurriculumTierV1,
    identity: &EvaluationRunIdentityV1,
    analysis: &EvaluationAnalysisV1,
) -> Result<CurriculumAssessmentV1, CurriculumDecisionError> {
    let opponent = tier
        .fixed_opponent()
        .ok_or(CurriculumDecisionError::SelfPlayHasNoFixedOpponent)?;
    if identity.opponent != opponent {
        return Err(CurriculumDecisionError::OpponentMismatch);
    }
    let window = CurriculumEloWindowV1::from_evaluation(identity)?;
    if analysis.elo0.to_bits() != identity.elo0.to_bits()
        || analysis.elo1.to_bits() != identity.elo1.to_bits()
        || analysis.lower_window.is_some() != window.is_some()
    {
        return Err(CurriculumDecisionError::AnalysisWindowMismatch);
    }
    if let (Some(expected), Some(actual)) = (window, analysis.lower_window) {
        if actual.lower_elo.to_bits() != expected.lower_elo.to_bits()
            || actual.center_elo.to_bits() != expected.center_elo.to_bits()
        {
            return Err(CurriculumDecisionError::AnalysisWindowMismatch);
        }
    }
    Ok(assessment_from_conclusion(
        tier,
        analysis.conclusion,
        window.is_some(),
    ))
}

pub const fn self_play_assessment() -> CurriculumAssessmentV1 {
    CurriculumAssessmentV1 {
        action: CurriculumActionV1::ContinueSelfPlay,
        reason: CurriculumReasonV1::SelfPlayPromotionHandledSeparately,
        tier_after: CurriculumTierV1::SelfPlay,
    }
}

pub const fn deferred_evaluation_assessment(tier: CurriculumTierV1) -> CurriculumAssessmentV1 {
    CurriculumAssessmentV1 {
        action: CurriculumActionV1::Hold,
        reason: CurriculumReasonV1::ScheduledEvaluationDeferred,
        tier_after: tier,
    }
}

const fn assessment_from_conclusion(
    tier: CurriculumTierV1,
    conclusion: EvaluationConclusion,
    two_sided: bool,
) -> CurriculumAssessmentV1 {
    match conclusion {
        EvaluationConclusion::SupportsUpperEloHypothesis => CurriculumAssessmentV1 {
            action: CurriculumActionV1::Advance,
            reason: CurriculumReasonV1::UpperEloHypothesisSupported,
            tier_after: tier.next(),
        },
        EvaluationConclusion::SupportsLowerEloHypothesis if two_sided => match tier.previous() {
            Some(previous) => CurriculumAssessmentV1 {
                action: CurriculumActionV1::Retreat,
                reason: CurriculumReasonV1::LowerEloHypothesisSupported,
                tier_after: previous,
            },
            None => CurriculumAssessmentV1 {
                action: CurriculumActionV1::Hold,
                reason: CurriculumReasonV1::LowerEloHypothesisAtFloor,
                tier_after: tier,
            },
        },
        EvaluationConclusion::SupportsLowerEloHypothesis => CurriculumAssessmentV1 {
            action: CurriculumActionV1::Hold,
            reason: CurriculumReasonV1::LegacyOneSidedEvidence,
            tier_after: tier,
        },
        EvaluationConclusion::SupportsCentralEloWindow => CurriculumAssessmentV1 {
            action: CurriculumActionV1::Hold,
            reason: CurriculumReasonV1::CentralEloWindowSupported,
            tier_after: tier,
        },
        EvaluationConclusion::InconclusiveMaximumEligiblePairs
        | EvaluationConclusion::InconclusiveMaximumAttemptedPairs => CurriculumAssessmentV1 {
            action: CurriculumActionV1::Hold,
            reason: CurriculumReasonV1::EvaluationInconclusive,
            tier_after: tier,
        },
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CurriculumDecisionError {
    InvalidTier(String),
    InvalidEloWindow,
    SelfPlayHasNoFixedOpponent,
    OpponentMismatch,
    AnalysisWindowMismatch,
}

impl fmt::Display for CurriculumDecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTier(value) => write!(formatter, "invalid curriculum tier `{value}`"),
            Self::InvalidEloWindow => {
                formatter.write_str("curriculum evaluation requires lower_elo < elo0 < elo1")
            }
            Self::SelfPlayHasNoFixedOpponent => formatter.write_str(
                "self-play progression is decided by checkpoint promotion, not a fixed opponent",
            ),
            Self::OpponentMismatch => {
                formatter.write_str("evaluation opponent differs from the active curriculum tier")
            }
            Self::AnalysisWindowMismatch => {
                formatter.write_str("evaluation analysis differs from its immutable Elo window")
            }
        }
    }
}

impl std::error::Error for CurriculumDecisionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_tiers_advance_and_retreat_without_exceeding_the_mcts_cap() {
        let tiers = [
            CurriculumTierV1::Random,
            CurriculumTierV1::SiteBotV1,
            CurriculumTierV1::Mcts8,
            CurriculumTierV1::Mcts32,
            CurriculumTierV1::Mcts128,
            CurriculumTierV1::Mcts512,
            CurriculumTierV1::SelfPlay,
        ];
        for pair in tiers.windows(2) {
            assert_eq!(pair[0].next(), pair[1]);
            assert_eq!(pair[1].previous(), Some(pair[0]));
        }
        assert_eq!(
            CurriculumTierV1::SelfPlay.next(),
            CurriculumTierV1::SelfPlay
        );
        assert_eq!(CurriculumTierV1::Mcts512.training_opponent(), "mcts:512");
    }

    #[test]
    fn two_sided_evidence_maps_to_advance_hold_and_retreat() {
        let tier = CurriculumTierV1::Mcts32;
        assert_eq!(
            assessment_from_conclusion(
                tier,
                EvaluationConclusion::SupportsUpperEloHypothesis,
                true,
            ),
            CurriculumAssessmentV1 {
                action: CurriculumActionV1::Advance,
                reason: CurriculumReasonV1::UpperEloHypothesisSupported,
                tier_after: CurriculumTierV1::Mcts128,
            }
        );
        assert_eq!(
            assessment_from_conclusion(tier, EvaluationConclusion::SupportsCentralEloWindow, true,)
                .action,
            CurriculumActionV1::Hold
        );
        assert_eq!(
            assessment_from_conclusion(
                tier,
                EvaluationConclusion::SupportsLowerEloHypothesis,
                true,
            )
            .tier_after,
            CurriculumTierV1::Mcts8
        );
    }

    #[test]
    fn random_is_a_floor_and_legacy_lower_evidence_never_means_retreat() {
        assert_eq!(
            assessment_from_conclusion(
                CurriculumTierV1::Random,
                EvaluationConclusion::SupportsLowerEloHypothesis,
                true,
            )
            .reason,
            CurriculumReasonV1::LowerEloHypothesisAtFloor
        );
        assert_eq!(
            assessment_from_conclusion(
                CurriculumTierV1::Mcts32,
                EvaluationConclusion::SupportsLowerEloHypothesis,
                false,
            )
            .action,
            CurriculumActionV1::Hold
        );
    }
}
