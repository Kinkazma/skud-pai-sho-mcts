use core::fmt;

pub const PENTANOMIAL_SPRT_REFERENCE_COMMIT: &str = "b8eecff220b562a0dc2c4e68d1fa02521e06d72c";
pub const PENTANOMIAL_SPRT_REFERENCE_PATH: &str = "server/fishtest/stats/LLRcalc.py";

const OUTCOME_COUNT: usize = 5;
const ZERO_BIN_PRIOR: f64 = 1.0e-3;
const ROOT_MARGIN: f64 = 1.0e-9;
const ROOT_TOLERANCE: f64 = 1.0e-14;
const MAXIMUM_ROOT_ITERATIONS: usize = 256;
const PAIR_SCORES: [f64; OUTCOME_COUNT] = [0.0, 0.25, 0.5, 0.75, 1.0];

/// Counts candidate scores across reversed-seat pairs: 0, 0.5, 1, 1.5, 2.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PentanomialCounts {
    bins: [u64; OUTCOME_COUNT],
}

impl PentanomialCounts {
    pub const fn new(bins: [u64; OUTCOME_COUNT]) -> Self {
        Self { bins }
    }

    pub const fn bins(self) -> [u64; OUTCOME_COUNT] {
        self.bins
    }

    pub fn observe_pair_half_points(
        &mut self,
        candidate_half_points: u8,
    ) -> Result<(), PromotionSprtError> {
        let index = usize::from(candidate_half_points);
        let Some(bin) = self.bins.get_mut(index) else {
            return Err(PromotionSprtError::InvalidPairHalfPoints(
                candidate_half_points,
            ));
        };
        *bin = bin
            .checked_add(1)
            .ok_or(PromotionSprtError::CountOverflow)?;
        Ok(())
    }

    pub fn checked_pairs(self) -> Result<u64, PromotionSprtError> {
        self.bins.into_iter().try_fold(0_u64, |total, count| {
            total
                .checked_add(count)
                .ok_or(PromotionSprtError::CountOverflow)
        })
    }

    fn empirical_score(self, pairs: u64) -> Option<f64> {
        if pairs == 0 {
            return None;
        }
        let weighted = self
            .bins
            .into_iter()
            .zip(PAIR_SCORES)
            .map(|(count, score)| count as f64 * score)
            .sum::<f64>();
        Some(weighted / pairs as f64)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionSprtConfig {
    elo0: f64,
    elo1: f64,
    alpha: f64,
    beta: f64,
}

impl PromotionSprtConfig {
    pub fn new(elo0: f64, elo1: f64, alpha: f64, beta: f64) -> Result<Self, PromotionSprtError> {
        if !elo0.is_finite() || !elo1.is_finite() || elo0 >= elo1 {
            return Err(PromotionSprtError::InvalidEloHypotheses { elo0, elo1 });
        }
        if !alpha.is_finite()
            || !beta.is_finite()
            || alpha <= 0.0
            || beta <= 0.0
            || alpha >= 1.0
            || beta >= 1.0
            || alpha + beta >= 1.0
        {
            return Err(PromotionSprtError::InvalidErrorProbabilities { alpha, beta });
        }
        for (name, elo) in [("elo0", elo0), ("elo1", elo1)] {
            let score = logistic_score(elo);
            if !score.is_finite() || !(0.0..1.0).contains(&score) {
                return Err(PromotionSprtError::EloOutsideNumericalRange { name, elo });
            }
        }
        Ok(Self {
            elo0,
            elo1,
            alpha,
            beta,
        })
    }

    pub const fn elo0(self) -> f64 {
        self.elo0
    }

    pub const fn elo1(self) -> f64 {
        self.elo1
    }

    pub const fn alpha(self) -> f64 {
        self.alpha
    }

    pub const fn beta(self) -> f64 {
        self.beta
    }

    pub fn lower_bound(self) -> f64 {
        self.beta.ln() - (-self.alpha).ln_1p()
    }

    pub fn upper_bound(self) -> f64 {
        (-self.beta).ln_1p() - self.alpha.ln()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromotionDecision {
    Continue,
    PromoteCandidate,
    RejectCandidate,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PromotionSprtReport {
    pub counts: PentanomialCounts,
    pub pairs: u64,
    pub empirical_score: Option<f64>,
    pub log_likelihood_ratio: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub decision: PromotionDecision,
}

pub fn evaluate_promotion_sprt(
    counts: PentanomialCounts,
    config: PromotionSprtConfig,
) -> Result<PromotionSprtReport, PromotionSprtError> {
    let pairs = counts.checked_pairs()?;
    let lower_bound = config.lower_bound();
    let upper_bound = config.upper_bound();
    let log_likelihood_ratio = if pairs == 0 {
        0.0
    } else {
        logistic_log_likelihood_ratio(counts.bins, config.elo0, config.elo1)?
    };
    let decision = if log_likelihood_ratio <= lower_bound {
        PromotionDecision::RejectCandidate
    } else if log_likelihood_ratio >= upper_bound {
        PromotionDecision::PromoteCandidate
    } else {
        PromotionDecision::Continue
    };
    Ok(PromotionSprtReport {
        counts,
        pairs,
        empirical_score: counts.empirical_score(pairs),
        log_likelihood_ratio,
        lower_bound,
        upper_bound,
        decision,
    })
}

fn logistic_log_likelihood_ratio(
    counts: [u64; OUTCOME_COUNT],
    elo0: f64,
    elo1: f64,
) -> Result<f64, PromotionSprtError> {
    let regularized = counts.map(|count| {
        if count == 0 {
            ZERO_BIN_PRIOR
        } else {
            count as f64
        }
    });
    let total = regularized.into_iter().sum::<f64>();
    let empirical = regularized.map(|count| count / total);
    let null = constrained_multinomial_mle(empirical, logistic_score(elo0))?;
    let alternative = constrained_multinomial_mle(empirical, logistic_score(elo1))?;
    let ratio = regularized
        .into_iter()
        .zip(null)
        .zip(alternative)
        .map(|((count, null_probability), alternative_probability)| {
            count * (alternative_probability.ln() - null_probability.ln())
        })
        .sum::<f64>();
    if ratio.is_finite() {
        Ok(ratio)
    } else {
        Err(PromotionSprtError::NumericalFailure(
            "log-likelihood ratio is not finite",
        ))
    }
}

fn constrained_multinomial_mle(
    empirical: [f64; OUTCOME_COUNT],
    expected_score: f64,
) -> Result<[f64; OUTCOME_COUNT], PromotionSprtError> {
    let shifted = PAIR_SCORES.map(|score| score - expected_score);
    let minimum = shifted.into_iter().fold(f64::INFINITY, f64::min);
    let maximum = shifted.into_iter().fold(f64::NEG_INFINITY, f64::max);
    if minimum >= 0.0 || maximum <= 0.0 {
        return Err(PromotionSprtError::NumericalFailure(
            "constrained expectation is outside the outcome support",
        ));
    }
    let mut lower = -1.0 / maximum + ROOT_MARGIN;
    let mut upper = -1.0 / minimum - ROOT_MARGIN;
    let mut root = 0.0;
    let mut converged = false;
    for _ in 0..MAXIMUM_ROOT_ITERATIONS {
        root = 0.5 * (lower + upper);
        let value = secular_value(empirical, shifted, root);
        if !value.is_finite() {
            return Err(PromotionSprtError::NumericalFailure(
                "constrained-likelihood root became non-finite",
            ));
        }
        if value.abs() <= ROOT_TOLERANCE || (upper - lower).abs() <= ROOT_TOLERANCE {
            converged = true;
            break;
        }
        if root == lower || root == upper {
            converged = true;
            break;
        }
        if value > 0.0 {
            lower = root;
        } else {
            upper = root;
        }
    }
    if !converged {
        return Err(PromotionSprtError::NumericalFailure(
            "constrained-likelihood root did not converge",
        ));
    }
    let probabilities =
        core::array::from_fn(|index| empirical[index] / (1.0 + root * shifted[index]));
    if probabilities
        .iter()
        .any(|probability| !probability.is_finite() || *probability <= 0.0)
    {
        return Err(PromotionSprtError::NumericalFailure(
            "constrained probabilities are not positive and finite",
        ));
    }
    Ok(probabilities)
}

fn secular_value(empirical: [f64; OUTCOME_COUNT], shifted: [f64; OUTCOME_COUNT], root: f64) -> f64 {
    empirical
        .into_iter()
        .zip(shifted)
        .map(|(probability, value)| probability * value / (1.0 + root * value))
        .sum()
}

fn logistic_score(elo: f64) -> f64 {
    let log_odds = elo * core::f64::consts::LN_10 / 400.0;
    if log_odds >= 0.0 {
        1.0 / (1.0 + (-log_odds).exp())
    } else {
        let odds = log_odds.exp();
        odds / (1.0 + odds)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PromotionSprtError {
    InvalidPairHalfPoints(u8),
    CountOverflow,
    InvalidEloHypotheses { elo0: f64, elo1: f64 },
    InvalidErrorProbabilities { alpha: f64, beta: f64 },
    EloOutsideNumericalRange { name: &'static str, elo: f64 },
    NumericalFailure(&'static str),
}

impl fmt::Display for PromotionSprtError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPairHalfPoints(value) => write!(
                formatter,
                "a two-game pair awards the candidate 0 through 4 half-points, not {value}"
            ),
            Self::CountOverflow => formatter.write_str("pentanomial pair count overflow"),
            Self::InvalidEloHypotheses { elo0, elo1 } => write!(
                formatter,
                "SPRT requires finite ordered Elo hypotheses, got elo0={elo0}, elo1={elo1}"
            ),
            Self::InvalidErrorProbabilities { alpha, beta } => write!(
                formatter,
                "SPRT requires alpha and beta in (0, 1) with alpha + beta < 1, got {alpha} and {beta}"
            ),
            Self::EloOutsideNumericalRange { name, elo } => write!(
                formatter,
                "SPRT {name}={elo} maps numerically to an exact score boundary"
            ),
            Self::NumericalFailure(reason) => {
                write!(formatter, "pentanomial SPRT numerical failure: {reason}")
            }
        }
    }
}

impl std::error::Error for PromotionSprtError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_evidence_continues_from_a_zero_log_likelihood_ratio() {
        let config = PromotionSprtConfig::new(0.0, 10.0, 0.05, 0.05).unwrap();
        let report = evaluate_promotion_sprt(PentanomialCounts::default(), config).unwrap();
        assert_eq!(report.pairs, 0);
        assert_eq!(report.empirical_score, None);
        assert_eq!(report.log_likelihood_ratio, 0.0);
        assert_eq!(report.decision, PromotionDecision::Continue);
    }

    #[test]
    fn invalid_design_and_pair_scores_are_explicit() {
        assert!(matches!(
            PromotionSprtConfig::new(5.0, 5.0, 0.05, 0.05),
            Err(PromotionSprtError::InvalidEloHypotheses { .. })
        ));
        assert!(matches!(
            PromotionSprtConfig::new(0.0, 5.0, 0.6, 0.4),
            Err(PromotionSprtError::InvalidErrorProbabilities { .. })
        ));
        let mut counts = PentanomialCounts::default();
        assert!(matches!(
            counts.observe_pair_half_points(5),
            Err(PromotionSprtError::InvalidPairHalfPoints(5))
        ));
    }

    #[test]
    fn decisive_evidence_crosses_the_expected_boundary() {
        let config = PromotionSprtConfig::new(0.0, 10.0, 0.05, 0.05).unwrap();
        let promoted =
            evaluate_promotion_sprt(PentanomialCounts::new([0, 0, 10, 40, 120]), config).unwrap();
        let rejected =
            evaluate_promotion_sprt(PentanomialCounts::new([120, 40, 10, 0, 0]), config).unwrap();
        assert_eq!(promoted.decision, PromotionDecision::PromoteCandidate);
        assert_eq!(rejected.decision, PromotionDecision::RejectCandidate);
        assert!(promoted.log_likelihood_ratio > promoted.upper_bound);
        assert!(rejected.log_likelihood_ratio < rejected.lower_bound);
    }

    #[test]
    fn wald_bounds_remain_finite_for_the_smallest_positive_alpha() {
        let config = PromotionSprtConfig::new(0.0, 10.0, f64::from_bits(1), 0.05).unwrap();
        assert!(config.lower_bound().is_finite());
        assert!(config.upper_bound().is_finite());
        assert!(config.lower_bound() < config.upper_bound());
    }

    #[test]
    fn constrained_root_accepts_machine_precision_stagnation() {
        let counts = PentanomialCounts::new([3_657_864, 21_588, 83, 976_477, 2]);
        let config =
            PromotionSprtConfig::new(507.594_817_908_805_45, 910.978_175_980_968_8, 0.05, 0.05)
                .unwrap();
        let report = evaluate_promotion_sprt(counts, config).unwrap();
        assert!(report.log_likelihood_ratio.is_finite());
    }
}
