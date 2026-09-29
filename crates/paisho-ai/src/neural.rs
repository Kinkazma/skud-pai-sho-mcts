use core::fmt;

use paisho_core::{Action, Position};
use paisho_model::{
    encode_action_v1, ActionEncodingError, InferenceExampleEncodingError, InferenceExampleV1,
    InferenceOutputV1, VALUE_CLASS_COUNT_V1,
};
use paisho_mpsgraph_client::CapacityInferenceBrokerClient;

use crate::{Agent, AgentError, AgentTelemetry, StableRng};

const DISTRIBUTION_TOLERANCE: f64 = 1.0e-5;

/// Backend-neutral output consumed by the pure network player.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValueOutput {
    policy_probabilities: Vec<f32>,
    value_probabilities: [f32; VALUE_CLASS_COUNT_V1],
}

impl PolicyValueOutput {
    pub fn new(
        policy_probabilities: Vec<f32>,
        value_probabilities: [f32; VALUE_CLASS_COUNT_V1],
    ) -> Result<Self, PolicyValueOutputError> {
        validate_distribution("policy", &policy_probabilities)?;
        validate_distribution("value", &value_probabilities)?;
        Ok(Self {
            policy_probabilities,
            value_probabilities,
        })
    }

    pub fn policy_probabilities(&self) -> &[f32] {
        &self.policy_probabilities
    }

    pub const fn value_probabilities(&self) -> &[f32; VALUE_CLASS_COUNT_V1] {
        &self.value_probabilities
    }
}

impl TryFrom<InferenceOutputV1> for PolicyValueOutput {
    type Error = PolicyValueOutputError;

    fn try_from(output: InferenceOutputV1) -> Result<Self, Self::Error> {
        Self::new(
            output.policy_probabilities().to_vec(),
            *output.value_probabilities(),
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyValueOutputError {
    EmptyDistribution(&'static str),
    InvalidProbability {
        name: &'static str,
        index: usize,
        value: f32,
    },
    InvalidDistribution {
        name: &'static str,
        sum: f64,
    },
}

impl fmt::Display for PolicyValueOutputError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution(name) => write!(formatter, "{name} distribution is empty"),
            Self::InvalidProbability { name, index, value } => {
                write!(formatter, "{name} probability {index} is invalid: {value}")
            }
            Self::InvalidDistribution { name, sum } => {
                write!(formatter, "{name} probabilities sum to {sum}, not 1")
            }
        }
    }
}

impl std::error::Error for PolicyValueOutputError {}

/// Minimal boundary implemented by MPSGraph today and by future runtimes.
pub trait PolicyValueEvaluator: Clone + Send + Sync {
    fn evaluate(&self, example: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError>;
}

impl PolicyValueEvaluator for CapacityInferenceBrokerClient {
    fn evaluate(&self, example: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError> {
        self.infer_encoded(example)
            .map_err(|source| AgentError::new(format!("MPSGraph inference failed: {source}")))
            .and_then(|output| {
                PolicyValueOutput::try_from(output).map_err(|source| {
                    AgentError::new(format!(
                        "MPSGraph returned an invalid distribution: {source}"
                    ))
                })
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NetworkPolicy {
    Argmax,
    Sample { temperature: f32, uniform_mix: f32 },
}

impl NetworkPolicy {
    pub fn validate(self) -> Result<Self, NetworkPolicyError> {
        match self {
            Self::Argmax => Ok(self),
            Self::Sample {
                temperature,
                uniform_mix,
            } => {
                if !temperature.is_finite() || temperature <= 0.0 {
                    return Err(NetworkPolicyError::InvalidTemperature(temperature));
                }
                if !uniform_mix.is_finite() || !(0.0..=1.0).contains(&uniform_mix) {
                    return Err(NetworkPolicyError::InvalidUniformMix(uniform_mix));
                }
                Ok(self)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NetworkPolicyError {
    InvalidTemperature(f32),
    InvalidUniformMix(f32),
}

impl fmt::Display for NetworkPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTemperature(value) => {
                write!(
                    formatter,
                    "network sampling temperature must be finite and positive, got {value}"
                )
            }
            Self::InvalidUniformMix(value) => write!(
                formatter,
                "network uniform exploration mix must be in [0, 1], got {value}"
            ),
        }
    }
}

impl std::error::Error for NetworkPolicyError {}

#[derive(Clone, Debug, PartialEq)]
pub struct NetworkDecision {
    selected_index: usize,
    network_policy: Vec<f32>,
    behavior_policy: Vec<f32>,
    value_probabilities: [f32; VALUE_CLASS_COUNT_V1],
}

impl NetworkDecision {
    pub const fn selected_index(&self) -> usize {
        self.selected_index
    }

    pub fn network_policy(&self) -> &[f32] {
        &self.network_policy
    }

    pub fn behavior_policy(&self) -> &[f32] {
        &self.behavior_policy
    }

    pub const fn value_probabilities(&self) -> &[f32; VALUE_CLASS_COUNT_V1] {
        &self.value_probabilities
    }
}

#[derive(Clone)]
pub struct PureNetworkAgent<Evaluator = CapacityInferenceBrokerClient> {
    evaluator: Evaluator,
    policy: NetworkPolicy,
    rng: StableRng,
    decisions: usize,
}

impl<Evaluator> PureNetworkAgent<Evaluator>
where
    Evaluator: PolicyValueEvaluator,
{
    pub fn new(
        evaluator: Evaluator,
        seed: u64,
        policy: NetworkPolicy,
    ) -> Result<Self, NetworkPolicyError> {
        Ok(Self {
            evaluator,
            policy: policy.validate()?,
            rng: StableRng::new(seed),
            decisions: 0,
        })
    }

    pub const fn policy(&self) -> NetworkPolicy {
        self.policy
    }

    pub fn decide(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<NetworkDecision, NetworkDecisionError> {
        let example =
            InferenceExampleV1::from_position(position).map_err(NetworkDecisionError::Encoding)?;
        verify_legal_action_contract(position, legal_actions, &example)?;
        let output = self
            .evaluator
            .evaluate(example)
            .map_err(NetworkDecisionError::Inference)?;
        if output.policy_probabilities.len() != legal_actions.len() {
            return Err(NetworkDecisionError::PolicyLengthMismatch {
                legal_actions: legal_actions.len(),
                probabilities: output.policy_probabilities.len(),
            });
        }
        let (selected_index, behavior_policy) =
            select_policy(self.policy, &output.policy_probabilities, &mut self.rng);
        self.decisions += 1;
        Ok(NetworkDecision {
            selected_index,
            network_policy: output.policy_probabilities,
            behavior_policy,
            value_probabilities: output.value_probabilities,
        })
    }
}

impl<Evaluator> Agent for PureNetworkAgent<Evaluator>
where
    Evaluator: PolicyValueEvaluator,
{
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        self.decide(position, legal_actions)
            .map(|decision| decision.selected_index)
            .map_err(|source| AgentError::new(format!("pure-network selection failed: {source}")))
    }

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry {
            decisions: self.decisions,
            ..AgentTelemetry::default()
        }
    }

    fn reset_telemetry(&mut self) {
        self.decisions = 0;
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum NetworkDecisionError {
    Encoding(InferenceExampleEncodingError),
    SuppliedActionEncoding {
        index: usize,
        source: ActionEncodingError,
    },
    LegalActionCountMismatch {
        supplied: usize,
        canonical: usize,
    },
    LegalActionOrderMismatch {
        index: usize,
    },
    Inference(AgentError),
    PolicyLengthMismatch {
        legal_actions: usize,
        probabilities: usize,
    },
}

impl fmt::Display for NetworkDecisionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Encoding(source) => source.fmt(formatter),
            Self::SuppliedActionEncoding { index, source } => {
                write!(formatter, "cannot encode supplied legal action {index}: {source}")
            }
            Self::LegalActionCountMismatch {
                supplied,
                canonical,
            } => write!(
                formatter,
                "agent received {supplied} legal actions but the engine regenerated {canonical}"
            ),
            Self::LegalActionOrderMismatch { index } => write!(
                formatter,
                "agent legal-action order differs from the engine at index {index}"
            ),
            Self::Inference(source) => source.fmt(formatter),
            Self::PolicyLengthMismatch {
                legal_actions,
                probabilities,
            } => write!(
                formatter,
                "network returned {probabilities} policy probabilities for {legal_actions} legal actions"
            ),
        }
    }
}

impl std::error::Error for NetworkDecisionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Encoding(source) => Some(source),
            Self::SuppliedActionEncoding { source, .. } => Some(source),
            Self::Inference(source) => Some(source),
            _ => None,
        }
    }
}

fn verify_legal_action_contract(
    position: &Position,
    legal_actions: &[Action],
    example: &InferenceExampleV1,
) -> Result<(), NetworkDecisionError> {
    if legal_actions.len() != example.legal_actions().len() {
        return Err(NetworkDecisionError::LegalActionCountMismatch {
            supplied: legal_actions.len(),
            canonical: example.legal_actions().len(),
        });
    }
    for (index, (action, canonical)) in legal_actions
        .iter()
        .zip(example.legal_actions())
        .enumerate()
    {
        let encoded = encode_action_v1(*action, position.to_move())
            .map_err(|source| NetworkDecisionError::SuppliedActionEncoding { index, source })?;
        if encoded != *canonical {
            return Err(NetworkDecisionError::LegalActionOrderMismatch { index });
        }
    }
    Ok(())
}

fn select_policy(policy: NetworkPolicy, network: &[f32], rng: &mut StableRng) -> (usize, Vec<f32>) {
    match policy {
        NetworkPolicy::Argmax => {
            let selected = network
                .iter()
                .enumerate()
                .max_by(|(left_index, left), (right_index, right)| {
                    left.total_cmp(right)
                        .then_with(|| right_index.cmp(left_index))
                })
                .map(|(index, _)| index)
                .expect("validated network policies are non-empty");
            let mut behavior = vec![0.0; network.len()];
            behavior[selected] = 1.0;
            (selected, behavior)
        }
        NetworkPolicy::Sample {
            temperature,
            uniform_mix,
        } => {
            let inverse_temperature = 1.0 / f64::from(temperature);
            let maximum_logit = network
                .iter()
                .copied()
                .filter(|value| *value > 0.0)
                .map(|value| f64::from(value).ln() * inverse_temperature)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut probabilities = network
                .iter()
                .map(|&value| {
                    if value > 0.0 {
                        (f64::from(value).ln() * inverse_temperature - maximum_logit).exp()
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>();
            let total = probabilities.iter().sum::<f64>();
            let uniform = 1.0 / probabilities.len() as f64;
            let retained = 1.0 - f64::from(uniform_mix);
            for probability in &mut probabilities {
                *probability = retained * (*probability / total) + f64::from(uniform_mix) * uniform;
            }

            let draw = rng.next_f64();
            let mut cumulative = 0.0;
            let mut selected = probabilities
                .iter()
                .rposition(|probability| *probability > 0.0)
                .expect("validated network policies contain positive probability");
            for (index, probability) in probabilities.iter().copied().enumerate() {
                cumulative += probability;
                if draw < cumulative {
                    selected = index;
                    break;
                }
            }
            (
                selected,
                probabilities
                    .into_iter()
                    .map(|value| value as f32)
                    .collect(),
            )
        }
    }
}

fn validate_distribution(
    name: &'static str,
    probabilities: &[f32],
) -> Result<(), PolicyValueOutputError> {
    if probabilities.is_empty() {
        return Err(PolicyValueOutputError::EmptyDistribution(name));
    }
    let mut sum = 0.0_f64;
    for (index, &value) in probabilities.iter().enumerate() {
        if !value.is_finite() || value < 0.0 {
            return Err(PolicyValueOutputError::InvalidProbability { name, index, value });
        }
        sum += f64::from(value);
    }
    if (sum - 1.0).abs() > DISTRIBUTION_TOLERANCE {
        return Err(PolicyValueOutputError::InvalidDistribution { name, sum });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use paisho_core::{legal_actions, BasicFlower, StandardSetup};

    use super::*;

    #[derive(Clone)]
    struct FixedEvaluator {
        output: Result<PolicyValueOutput, AgentError>,
    }

    impl PolicyValueEvaluator for FixedEvaluator {
        fn evaluate(&self, _example: InferenceExampleV1) -> Result<PolicyValueOutput, AgentError> {
            self.output.clone()
        }
    }

    fn position_and_actions() -> (Position, Vec<Action>) {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let actions = legal_actions(&position);
        (position, actions)
    }

    fn output_with_best(action_count: usize, best: usize) -> PolicyValueOutput {
        let mut policy = vec![0.0; action_count];
        policy[best] = 0.75;
        let remainder = 0.25 / (action_count - 1) as f32;
        for (index, probability) in policy.iter_mut().enumerate() {
            if index != best {
                *probability = remainder;
            }
        }
        PolicyValueOutput::new(policy, [0.2, 0.3, 0.5]).unwrap()
    }

    #[test]
    fn argmax_selects_the_network_maximum_without_search() {
        let (position, actions) = position_and_actions();
        let evaluator = FixedEvaluator {
            output: Ok(output_with_best(actions.len(), 3)),
        };
        let mut agent = PureNetworkAgent::new(evaluator, 7, NetworkPolicy::Argmax).unwrap();
        let decision = agent.decide(&position, &actions).unwrap();

        assert_eq!(decision.selected_index(), 3);
        assert_eq!(decision.behavior_policy()[3], 1.0);
        assert_eq!(decision.behavior_policy().iter().sum::<f32>(), 1.0);
        assert_eq!(decision.value_probabilities(), &[0.2, 0.3, 0.5]);
        assert_eq!(agent.telemetry().decisions, 1);
    }

    #[test]
    fn sampled_policy_is_reproducible_and_preserves_a_distribution() {
        let (position, actions) = position_and_actions();
        let evaluator = FixedEvaluator {
            output: Ok(output_with_best(actions.len(), 2)),
        };
        let policy = NetworkPolicy::Sample {
            temperature: 0.8,
            uniform_mix: 0.1,
        };
        let mut first = PureNetworkAgent::new(evaluator.clone(), 91, policy).unwrap();
        let mut second = PureNetworkAgent::new(evaluator, 91, policy).unwrap();

        for _ in 0..16 {
            let left = first.decide(&position, &actions).unwrap();
            let right = second.decide(&position, &actions).unwrap();
            assert_eq!(left, right);
            assert!((left.behavior_policy().iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
            assert!(left.behavior_policy().iter().all(|value| *value > 0.0));
        }
    }

    #[test]
    fn supplied_action_order_must_match_the_encoded_inference_order() {
        let (position, mut actions) = position_and_actions();
        let evaluator = FixedEvaluator {
            output: Ok(output_with_best(actions.len(), 0)),
        };
        actions.swap(0, 1);
        let mut agent = PureNetworkAgent::new(evaluator, 0, NetworkPolicy::Argmax).unwrap();

        assert!(matches!(
            agent.decide(&position, &actions),
            Err(NetworkDecisionError::LegalActionOrderMismatch { index: 0 })
        ));
        assert_eq!(agent.telemetry().decisions, 0);
    }

    #[test]
    fn policy_configuration_and_outputs_reject_invalid_probabilities() {
        assert_eq!(
            NetworkPolicy::Sample {
                temperature: 0.0,
                uniform_mix: 0.0,
            }
            .validate(),
            Err(NetworkPolicyError::InvalidTemperature(0.0))
        );
        assert!(PolicyValueOutput::new(vec![0.7, 0.4], [0.2, 0.3, 0.5]).is_err());
        assert!(PolicyValueOutput::new(vec![1.0], [0.2, f32::NAN, 0.8]).is_err());
    }
}
