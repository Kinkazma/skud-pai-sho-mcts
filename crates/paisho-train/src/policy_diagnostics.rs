use core::fmt;

const DISTRIBUTION_TOLERANCE: f64 = 1.0e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingProfileV1 {
    temperature: f64,
    uniform_mix: f64,
}

impl SamplingProfileV1 {
    pub fn new(temperature: f64, uniform_mix: f64) -> Result<Self, PolicyDiagnosticsError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(PolicyDiagnosticsError::InvalidTemperature(temperature));
        }
        if !uniform_mix.is_finite() || !(0.0..=1.0).contains(&uniform_mix) {
            return Err(PolicyDiagnosticsError::InvalidUniformMix(uniform_mix));
        }
        Ok(Self {
            temperature,
            uniform_mix,
        })
    }

    pub const fn temperature(self) -> f64 {
        self.temperature
    }

    pub const fn uniform_mix(self) -> f64 {
        self.uniform_mix
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyDistributionObservationV1 {
    pub legal_actions: usize,
    pub entropy: f64,
    pub normalized_entropy: f64,
    pub effective_actions: f64,
    pub maximum_probability: f64,
    pub collision_probability: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiagnosticRangeV1 {
    pub minimum: f64,
    pub mean: f64,
    pub maximum: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyDistributionSummaryV1 {
    pub examples: u64,
    pub forced_examples: u64,
    pub legal_actions: DiagnosticRangeV1,
    /// Distribution metrics below exclude forced one-action decisions.
    pub entropy: DiagnosticRangeV1,
    pub normalized_entropy: DiagnosticRangeV1,
    pub effective_actions: DiagnosticRangeV1,
    pub maximum_probability: DiagnosticRangeV1,
    pub collision_probability: DiagnosticRangeV1,
}

pub fn observe_sampling_policy_v1(
    network_probabilities: &[f32],
    profile: SamplingProfileV1,
) -> Result<PolicyDistributionObservationV1, PolicyDiagnosticsError> {
    let transformed = sampling_policy_probabilities_v1(network_probabilities, profile)?;
    observe_distribution(&transformed)
}

/// Apply the exact temperature and uniform-exploration transform used by the
/// recorded pure-network actor and by the terminal PPO objective.
pub fn sampling_policy_probabilities_v1(
    network_probabilities: &[f32],
    profile: SamplingProfileV1,
) -> Result<Vec<f64>, PolicyDiagnosticsError> {
    validate_network_distribution(network_probabilities)?;
    let inverse_temperature = 1.0 / profile.temperature;
    let maximum_logit = network_probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| f64::from(value).ln() * inverse_temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut transformed = network_probabilities
        .iter()
        .map(|&value| {
            if value > 0.0 {
                (f64::from(value).ln() * inverse_temperature - maximum_logit).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let transformed_mass = transformed.iter().sum::<f64>();
    let uniform = 1.0 / transformed.len() as f64;
    let retained = 1.0 - profile.uniform_mix;
    for probability in &mut transformed {
        *probability = retained * (*probability / transformed_mass) + profile.uniform_mix * uniform;
    }
    Ok(transformed)
}

pub fn summarize_policy_observations_v1(
    observations: &[PolicyDistributionObservationV1],
) -> Result<PolicyDistributionSummaryV1, PolicyDiagnosticsError> {
    if observations.is_empty() {
        return Err(PolicyDiagnosticsError::EmptyObservations);
    }
    let examples =
        u64::try_from(observations.len()).map_err(|_| PolicyDiagnosticsError::CountOverflow)?;
    let non_forced = observations
        .iter()
        .copied()
        .filter(|observation| observation.legal_actions > 1)
        .collect::<Vec<_>>();
    if non_forced.is_empty() {
        return Err(PolicyDiagnosticsError::NoNonForcedObservations);
    }
    let non_forced_count =
        u64::try_from(non_forced.len()).map_err(|_| PolicyDiagnosticsError::CountOverflow)?;
    Ok(PolicyDistributionSummaryV1 {
        examples,
        forced_examples: examples - non_forced_count,
        legal_actions: summarize(observations, |item| item.legal_actions as f64),
        entropy: summarize(&non_forced, |item| item.entropy),
        normalized_entropy: summarize(&non_forced, |item| item.normalized_entropy),
        effective_actions: summarize(&non_forced, |item| item.effective_actions),
        maximum_probability: summarize(&non_forced, |item| item.maximum_probability),
        collision_probability: summarize(&non_forced, |item| item.collision_probability),
    })
}

fn validate_network_distribution(probabilities: &[f32]) -> Result<(), PolicyDiagnosticsError> {
    if probabilities.is_empty() {
        return Err(PolicyDiagnosticsError::EmptyDistribution);
    }
    let mut sum = 0.0;
    for (index, &value) in probabilities.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(PolicyDiagnosticsError::InvalidProbability { index, value });
        }
        sum += f64::from(value);
    }
    if (sum - 1.0).abs() > DISTRIBUTION_TOLERANCE {
        return Err(PolicyDiagnosticsError::InvalidDistribution(sum));
    }
    Ok(())
}

fn observe_distribution(
    probabilities: &[f64],
) -> Result<PolicyDistributionObservationV1, PolicyDiagnosticsError> {
    let entropy = -probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| value * value.ln())
        .sum::<f64>();
    let legal_actions = probabilities.len();
    let normalized_entropy = if legal_actions == 1 {
        1.0
    } else {
        entropy / (legal_actions as f64).ln()
    };
    let maximum_probability = probabilities
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or(PolicyDiagnosticsError::EmptyDistribution)?;
    let collision_probability = probabilities
        .iter()
        .map(|probability| probability * probability)
        .sum();
    Ok(PolicyDistributionObservationV1 {
        legal_actions,
        entropy,
        normalized_entropy,
        effective_actions: entropy.exp(),
        maximum_probability,
        collision_probability,
    })
}

fn summarize(
    observations: &[PolicyDistributionObservationV1],
    value: impl Fn(&PolicyDistributionObservationV1) -> f64,
) -> DiagnosticRangeV1 {
    let first = value(&observations[0]);
    let (minimum, sum, maximum) = observations.iter().skip(1).fold(
        (first, first, first),
        |(minimum, sum, maximum), observation| {
            let current = value(observation);
            (minimum.min(current), sum + current, maximum.max(current))
        },
    );
    DiagnosticRangeV1 {
        minimum,
        mean: sum / observations.len() as f64,
        maximum,
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyDiagnosticsError {
    InvalidTemperature(f64),
    InvalidUniformMix(f64),
    EmptyDistribution,
    InvalidProbability { index: usize, value: f32 },
    InvalidDistribution(f64),
    EmptyObservations,
    NoNonForcedObservations,
    CountOverflow,
}

impl fmt::Display for PolicyDiagnosticsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature(value) => {
                write!(
                    formatter,
                    "temperature must be finite and positive, got {value}"
                )
            }
            Self::InvalidUniformMix(value) => {
                write!(formatter, "uniform mix must be in [0, 1], got {value}")
            }
            Self::EmptyDistribution => formatter.write_str("a policy distribution cannot be empty"),
            Self::InvalidProbability { index, value } => {
                write!(formatter, "policy probability {index} is invalid: {value}")
            }
            Self::InvalidDistribution(sum) => {
                write!(formatter, "policy probabilities sum to {sum}, not 1")
            }
            Self::EmptyObservations => {
                formatter.write_str("policy diagnostics require at least one observation")
            }
            Self::NoNonForcedObservations => {
                formatter.write_str("policy diagnostics require at least one non-forced choice")
            }
            Self::CountOverflow => formatter.write_str("policy observation count exceeds u64"),
        }
    }
}

impl std::error::Error for PolicyDiagnosticsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_policy_remains_uniform_under_every_valid_profile() {
        let observation =
            observe_sampling_policy_v1(&[0.25; 4], SamplingProfileV1::new(3.0, 0.2).unwrap())
                .unwrap();
        assert_eq!(observation.legal_actions, 4);
        assert!((observation.entropy - 4.0_f64.ln()).abs() < 1.0e-12);
        assert!((observation.normalized_entropy - 1.0).abs() < 1.0e-12);
        assert!((observation.effective_actions - 4.0).abs() < 1.0e-12);
        assert!((observation.maximum_probability - 0.25).abs() < 1.0e-12);
        assert!((observation.collision_probability - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn uniform_mix_gives_a_floor_to_a_concentrated_policy() {
        let observation = observe_sampling_policy_v1(
            &[1.0, 0.0, 0.0, 0.0],
            SamplingProfileV1::new(1.0, 0.2).unwrap(),
        )
        .unwrap();
        assert!((observation.maximum_probability - 0.85).abs() < 1.0e-12);
        assert!((observation.collision_probability - 0.73).abs() < 1.0e-12);
    }

    #[test]
    fn transformed_probabilities_are_exposed_without_changing_the_actor_math() {
        let transformed = sampling_policy_probabilities_v1(
            &[0.64, 0.36],
            SamplingProfileV1::new(2.0, 0.1).unwrap(),
        )
        .unwrap();
        assert!((transformed[0] - 0.5642857142857143).abs() < 1.0e-8);
        assert!((transformed[1] - 0.4357142857142857).abs() < 1.0e-8);
        assert!((transformed.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn summaries_preserve_ranges_and_reject_empty_input() {
        let profile = SamplingProfileV1::new(1.0, 0.0).unwrap();
        let observations = [
            observe_sampling_policy_v1(&[0.5, 0.5], profile).unwrap(),
            observe_sampling_policy_v1(&[0.8, 0.2], profile).unwrap(),
        ];
        let summary = summarize_policy_observations_v1(&observations).unwrap();
        assert_eq!(summary.examples, 2);
        assert_eq!(summary.forced_examples, 0);
        assert_eq!(summary.maximum_probability.minimum, 0.5);
        assert!((summary.maximum_probability.mean - 0.65).abs() < 1.0e-8);
        assert!((summary.maximum_probability.maximum - 0.8).abs() < 1.0e-8);
        assert_eq!(
            summarize_policy_observations_v1(&[]),
            Err(PolicyDiagnosticsError::EmptyObservations)
        );
        let forced = [observe_sampling_policy_v1(&[1.0], profile).unwrap()];
        assert_eq!(
            summarize_policy_observations_v1(&forced),
            Err(PolicyDiagnosticsError::NoNonForcedObservations)
        );
    }
}
