use core::fmt;

use paisho_model::{ValueClassV1, VALUE_CLASS_COUNT_V1};

use crate::{sampling_policy_probabilities_v1, PolicyDiagnosticsError, SamplingProfileV1};

const CHANGE_TOLERANCE: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyUpdateObservationV1 {
    pub target: ValueClassV1,
    pub legal_actions: usize,
    pub behavior_probability: f64,
    pub candidate_probability: f64,
    pub importance_ratio: f64,
    pub log_importance_ratio: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyUpdateClassSummaryV1 {
    pub examples: u64,
    pub forced_examples: u64,
    /// The remaining statistics exclude forced one-action decisions.
    pub mean_behavior_probability: Option<f64>,
    pub mean_candidate_probability: Option<f64>,
    pub mean_importance_ratio: Option<f64>,
    pub mean_log_importance_ratio: Option<f64>,
    pub increased_fraction: Option<f64>,
    pub unchanged_fraction: Option<f64>,
    pub decreased_fraction: Option<f64>,
    pub below_clip_fraction: Option<f64>,
    pub above_clip_fraction: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyUpdateSummaryV1 {
    pub examples: u64,
    pub forced_examples: u64,
    pub overall: PolicyUpdateClassSummaryV1,
    pub classes: [PolicyUpdateClassSummaryV1; VALUE_CLASS_COUNT_V1],
}

pub fn observe_played_policy_update_v1(
    network_probabilities: &[f32],
    played_action_index: usize,
    behavior_probability: f32,
    target: ValueClassV1,
    profile: SamplingProfileV1,
) -> Result<PolicyUpdateObservationV1, PolicyUpdateDiagnosticsError> {
    if !behavior_probability.is_finite()
        || behavior_probability <= 0.0
        || behavior_probability > 1.0
    {
        return Err(PolicyUpdateDiagnosticsError::InvalidBehaviorProbability(
            behavior_probability,
        ));
    }
    let transformed = sampling_policy_probabilities_v1(network_probabilities, profile)
        .map_err(PolicyUpdateDiagnosticsError::Policy)?;
    let candidate_probability = transformed.get(played_action_index).copied().ok_or(
        PolicyUpdateDiagnosticsError::PlayedActionOutOfRange {
            played: played_action_index,
            legal_actions: transformed.len(),
        },
    )?;
    let behavior_probability = f64::from(behavior_probability);
    let importance_ratio = candidate_probability / behavior_probability;
    Ok(PolicyUpdateObservationV1 {
        target,
        legal_actions: transformed.len(),
        behavior_probability,
        candidate_probability,
        importance_ratio,
        log_importance_ratio: importance_ratio.ln(),
    })
}

pub fn summarize_policy_updates_v1(
    observations: &[PolicyUpdateObservationV1],
    clip_epsilon: f64,
) -> Result<PolicyUpdateSummaryV1, PolicyUpdateDiagnosticsError> {
    if observations.is_empty() {
        return Err(PolicyUpdateDiagnosticsError::EmptyObservations);
    }
    if !clip_epsilon.is_finite() || clip_epsilon <= 0.0 || clip_epsilon >= 1.0 {
        return Err(PolicyUpdateDiagnosticsError::InvalidClipEpsilon(
            clip_epsilon,
        ));
    }
    let examples = u64::try_from(observations.len())
        .map_err(|_| PolicyUpdateDiagnosticsError::CountOverflow)?;
    let forced_examples = count_forced(observations)?;
    let overall = summarize_class(observations, clip_epsilon)?;
    let classes = [ValueClassV1::Win, ValueClassV1::Draw, ValueClassV1::Loss]
        .map(|target| {
            let selected = observations
                .iter()
                .copied()
                .filter(|observation| observation.target == target)
                .collect::<Vec<_>>();
            summarize_class(&selected, clip_epsilon)
        })
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .expect("the WDL class list has the sealed class count");
    Ok(PolicyUpdateSummaryV1 {
        examples,
        forced_examples,
        overall,
        classes,
    })
}

fn count_forced(
    observations: &[PolicyUpdateObservationV1],
) -> Result<u64, PolicyUpdateDiagnosticsError> {
    u64::try_from(
        observations
            .iter()
            .filter(|observation| observation.legal_actions == 1)
            .count(),
    )
    .map_err(|_| PolicyUpdateDiagnosticsError::CountOverflow)
}

fn summarize_class(
    observations: &[PolicyUpdateObservationV1],
    clip_epsilon: f64,
) -> Result<PolicyUpdateClassSummaryV1, PolicyUpdateDiagnosticsError> {
    let examples = u64::try_from(observations.len())
        .map_err(|_| PolicyUpdateDiagnosticsError::CountOverflow)?;
    let forced_examples = count_forced(observations)?;
    let non_forced = observations
        .iter()
        .filter(|observation| observation.legal_actions > 1)
        .collect::<Vec<_>>();
    if non_forced.is_empty() {
        return Ok(PolicyUpdateClassSummaryV1 {
            examples,
            forced_examples,
            mean_behavior_probability: None,
            mean_candidate_probability: None,
            mean_importance_ratio: None,
            mean_log_importance_ratio: None,
            increased_fraction: None,
            unchanged_fraction: None,
            decreased_fraction: None,
            below_clip_fraction: None,
            above_clip_fraction: None,
        });
    }
    let count = non_forced.len() as f64;
    let mean = |value: fn(&PolicyUpdateObservationV1) -> f64| {
        non_forced
            .iter()
            .map(|observation| value(observation))
            .sum::<f64>()
            / count
    };
    let fraction = |predicate: fn(&PolicyUpdateObservationV1) -> bool| {
        non_forced
            .iter()
            .filter(|observation| predicate(observation))
            .count() as f64
            / count
    };
    let lower_clip = 1.0 - clip_epsilon;
    let upper_clip = 1.0 + clip_epsilon;
    Ok(PolicyUpdateClassSummaryV1 {
        examples,
        forced_examples,
        mean_behavior_probability: Some(mean(|item| item.behavior_probability)),
        mean_candidate_probability: Some(mean(|item| item.candidate_probability)),
        mean_importance_ratio: Some(mean(|item| item.importance_ratio)),
        mean_log_importance_ratio: Some(mean(|item| item.log_importance_ratio)),
        increased_fraction: Some(fraction(|item| {
            item.log_importance_ratio > CHANGE_TOLERANCE
        })),
        unchanged_fraction: Some(fraction(|item| {
            item.log_importance_ratio.abs() <= CHANGE_TOLERANCE
        })),
        decreased_fraction: Some(fraction(|item| {
            item.log_importance_ratio < -CHANGE_TOLERANCE
        })),
        below_clip_fraction: Some(
            non_forced
                .iter()
                .filter(|item| item.importance_ratio < lower_clip)
                .count() as f64
                / count,
        ),
        above_clip_fraction: Some(
            non_forced
                .iter()
                .filter(|item| item.importance_ratio > upper_clip)
                .count() as f64
                / count,
        ),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyUpdateDiagnosticsError {
    Policy(PolicyDiagnosticsError),
    InvalidBehaviorProbability(f32),
    PlayedActionOutOfRange { played: usize, legal_actions: usize },
    InvalidClipEpsilon(f64),
    EmptyObservations,
    CountOverflow,
}

impl fmt::Display for PolicyUpdateDiagnosticsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Policy(source) => source.fmt(formatter),
            Self::InvalidBehaviorProbability(value) => {
                write!(
                    formatter,
                    "behavior probability must be in (0, 1], got {value}"
                )
            }
            Self::PlayedActionOutOfRange {
                played,
                legal_actions,
            } => write!(
                formatter,
                "played action {played} is outside {legal_actions} legal actions"
            ),
            Self::InvalidClipEpsilon(value) => {
                write!(formatter, "clip epsilon must be in (0, 1), got {value}")
            }
            Self::EmptyObservations => {
                formatter.write_str("policy-update diagnostics require observations")
            }
            Self::CountOverflow => formatter.write_str("policy-update count exceeds u64"),
        }
    }
}

impl std::error::Error for PolicyUpdateDiagnosticsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Policy(source) => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        candidate: &[f32],
        played: usize,
        behavior: f32,
        target: ValueClassV1,
    ) -> PolicyUpdateObservationV1 {
        observe_played_policy_update_v1(
            candidate,
            played,
            behavior,
            target,
            SamplingProfileV1::new(1.0, 0.0).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn played_action_ratio_uses_the_transformed_candidate_policy() {
        let observed = observe_played_policy_update_v1(
            &[0.64, 0.36],
            0,
            0.5,
            ValueClassV1::Win,
            SamplingProfileV1::new(2.0, 0.1).unwrap(),
        )
        .unwrap();
        assert!((observed.candidate_probability - 0.5642857142857143).abs() < 1.0e-8);
        assert!((observed.importance_ratio - 1.1285714285714286).abs() < 2.0e-8);
    }

    #[test]
    fn summaries_separate_terminal_classes_and_ignore_forced_ratios() {
        let observations = [
            observation(&[0.6, 0.4], 0, 0.5, ValueClassV1::Win),
            observation(&[0.4, 0.6], 0, 0.5, ValueClassV1::Loss),
            observation(&[1.0], 0, 1.0, ValueClassV1::Draw),
        ];
        let summary = summarize_policy_updates_v1(&observations, 0.2).unwrap();
        assert_eq!(summary.examples, 3);
        assert_eq!(summary.forced_examples, 1);
        assert_eq!(summary.classes[0].increased_fraction, Some(1.0));
        assert_eq!(summary.classes[1].forced_examples, 1);
        assert_eq!(summary.classes[1].mean_importance_ratio, None);
        assert_eq!(summary.classes[2].decreased_fraction, Some(1.0));
    }

    #[test]
    fn malformed_inputs_are_rejected() {
        let profile = SamplingProfileV1::new(1.0, 0.0).unwrap();
        assert!(matches!(
            observe_played_policy_update_v1(&[0.5, 0.5], 2, 0.5, ValueClassV1::Win, profile),
            Err(PolicyUpdateDiagnosticsError::PlayedActionOutOfRange { .. })
        ));
        assert!(matches!(
            observe_played_policy_update_v1(&[0.5, 0.5], 0, 0.0, ValueClassV1::Win, profile),
            Err(PolicyUpdateDiagnosticsError::InvalidBehaviorProbability(_))
        ));
        assert_eq!(
            summarize_policy_updates_v1(&[], 0.2),
            Err(PolicyUpdateDiagnosticsError::EmptyObservations)
        );
        let observations = [observation(&[0.5, 0.5], 0, 0.5, ValueClassV1::Draw)];
        assert_eq!(
            summarize_policy_updates_v1(&observations, 0.0),
            Err(PolicyUpdateDiagnosticsError::InvalidClipEpsilon(0.0))
        );
        assert_eq!(
            summarize_policy_updates_v1(&observations, 1.0),
            Err(PolicyUpdateDiagnosticsError::InvalidClipEpsilon(1.0))
        );
    }
}
