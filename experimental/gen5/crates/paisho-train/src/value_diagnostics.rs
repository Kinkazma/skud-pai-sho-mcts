use core::fmt;

use paisho_model::{ValueClassV1, VALUE_CLASS_COUNT_V1};

const DISTRIBUTION_TOLERANCE: f64 = 1.0e-5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValuePredictionObservationV1 {
    pub target: ValueClassV1,
    pub probabilities: [f32; VALUE_CLASS_COUNT_V1],
}

impl ValuePredictionObservationV1 {
    pub fn new(
        target: ValueClassV1,
        probabilities: [f32; VALUE_CLASS_COUNT_V1],
    ) -> Result<Self, ValueDiagnosticsError> {
        let mut sum = 0.0_f64;
        for (index, value) in probabilities.iter().copied().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(ValueDiagnosticsError::InvalidProbability { index, value });
            }
            sum += f64::from(value);
        }
        if (sum - 1.0).abs() > DISTRIBUTION_TOLERANCE {
            return Err(ValueDiagnosticsError::InvalidDistribution(sum));
        }
        Ok(Self {
            target,
            probabilities,
        })
    }

    pub fn signed_prediction(self) -> f64 {
        f64::from(self.probabilities[ValueClassV1::Win.index()])
            - f64::from(self.probabilities[ValueClassV1::Loss.index()])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueClassDiagnosticsV1 {
    pub examples: u64,
    pub mean_target_probability: Option<f64>,
    pub mean_signed_prediction: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValuePredictionSummaryV1 {
    pub examples: u64,
    pub classes: [ValueClassDiagnosticsV1; VALUE_CLASS_COUNT_V1],
    pub mean_probabilities: [f64; VALUE_CLASS_COUNT_V1],
    pub cross_entropy: f64,
    pub empirical_prior_cross_entropy: f64,
    pub cross_entropy_excess: f64,
    pub brier_score: f64,
    pub empirical_prior_brier_score: f64,
    pub brier_score_excess: f64,
    pub top_one_accuracy: f64,
    pub majority_class_accuracy: f64,
    pub accuracy_over_majority: f64,
    pub mean_signed_prediction: f64,
    pub mean_signed_target: f64,
    pub mean_terminal_residual: f64,
    pub mean_absolute_terminal_residual: f64,
}

pub fn summarize_value_predictions_v1(
    observations: &[ValuePredictionObservationV1],
) -> Result<ValuePredictionSummaryV1, ValueDiagnosticsError> {
    if observations.is_empty() {
        return Err(ValueDiagnosticsError::EmptyObservations);
    }
    let examples =
        u64::try_from(observations.len()).map_err(|_| ValueDiagnosticsError::CountOverflow)?;
    let mut class_counts = [0_u64; VALUE_CLASS_COUNT_V1];
    let mut class_target_probability_sums = [0.0_f64; VALUE_CLASS_COUNT_V1];
    let mut class_signed_prediction_sums = [0.0_f64; VALUE_CLASS_COUNT_V1];
    let mut probability_sums = [0.0_f64; VALUE_CLASS_COUNT_V1];
    let mut cross_entropy = 0.0;
    let mut brier_score = 0.0;
    let mut correct = 0_u64;
    let mut signed_prediction_sum = 0.0;
    let mut signed_target_sum = 0.0;
    let mut terminal_residual_sum = 0.0;
    let mut absolute_terminal_residual_sum = 0.0;

    for observation in observations {
        let target_index = observation.target.index();
        class_counts[target_index] += 1;
        let signed_prediction = observation.signed_prediction();
        let signed_target = f64::from(observation.target.signed_return());
        let terminal_residual = signed_target - signed_prediction;
        signed_prediction_sum += signed_prediction;
        signed_target_sum += signed_target;
        terminal_residual_sum += terminal_residual;
        absolute_terminal_residual_sum += terminal_residual.abs();
        class_signed_prediction_sums[target_index] += signed_prediction;

        let target_probability = f64::from(observation.probabilities[target_index]);
        class_target_probability_sums[target_index] += target_probability;
        cross_entropy -= target_probability.ln();
        let predicted = observation
            .probabilities
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .expect("the WDL distribution has three entries");
        if predicted == target_index {
            correct += 1;
        }
        for (index, probability) in observation.probabilities.iter().copied().enumerate() {
            let probability = f64::from(probability);
            probability_sums[index] += probability;
            let target = f64::from((index == target_index) as u8);
            brier_score += (probability - target).powi(2);
        }
    }

    let denominator = examples as f64;
    let empirical_probabilities = class_counts.map(|count| count as f64 / denominator);
    let empirical_prior_cross_entropy = -empirical_probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > 0.0)
        .map(|probability| probability * probability.ln())
        .sum::<f64>();
    let empirical_prior_brier_score = 1.0
        - empirical_probabilities
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
    let majority_class_accuracy = empirical_probabilities
        .into_iter()
        .reduce(f64::max)
        .expect("the WDL distribution has three entries");
    let cross_entropy = cross_entropy / denominator;
    let brier_score = brier_score / denominator;
    let top_one_accuracy = correct as f64 / denominator;
    let classes = core::array::from_fn(|index| {
        let count = class_counts[index];
        let class_denominator = (count > 0).then_some(count as f64);
        ValueClassDiagnosticsV1 {
            examples: count,
            mean_target_probability: class_denominator
                .map(|value| class_target_probability_sums[index] / value),
            mean_signed_prediction: class_denominator
                .map(|value| class_signed_prediction_sums[index] / value),
        }
    });
    Ok(ValuePredictionSummaryV1 {
        examples,
        classes,
        mean_probabilities: probability_sums.map(|sum| sum / denominator),
        cross_entropy,
        empirical_prior_cross_entropy,
        cross_entropy_excess: cross_entropy - empirical_prior_cross_entropy,
        brier_score,
        empirical_prior_brier_score,
        brier_score_excess: brier_score - empirical_prior_brier_score,
        top_one_accuracy,
        majority_class_accuracy,
        accuracy_over_majority: top_one_accuracy - majority_class_accuracy,
        mean_signed_prediction: signed_prediction_sum / denominator,
        mean_signed_target: signed_target_sum / denominator,
        mean_terminal_residual: terminal_residual_sum / denominator,
        mean_absolute_terminal_residual: absolute_terminal_residual_sum / denominator,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub enum ValueDiagnosticsError {
    InvalidProbability { index: usize, value: f32 },
    InvalidDistribution(f64),
    EmptyObservations,
    CountOverflow,
}

impl fmt::Display for ValueDiagnosticsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProbability { index, value } => {
                write!(formatter, "value probability {index} is invalid: {value}")
            }
            Self::InvalidDistribution(sum) => {
                write!(formatter, "value probabilities sum to {sum}, not 1")
            }
            Self::EmptyObservations => {
                formatter.write_str("value diagnostics require at least one observation")
            }
            Self::CountOverflow => formatter.write_str("value observation count exceeds u64"),
        }
    }
}

impl std::error::Error for ValueDiagnosticsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions_have_zero_error_and_ordered_signed_values() {
        let observations = [
            ValuePredictionObservationV1::new(ValueClassV1::Win, [1.0, 0.0, 0.0]).unwrap(),
            ValuePredictionObservationV1::new(ValueClassV1::Draw, [0.0, 1.0, 0.0]).unwrap(),
            ValuePredictionObservationV1::new(ValueClassV1::Loss, [0.0, 0.0, 1.0]).unwrap(),
        ];
        let summary = summarize_value_predictions_v1(&observations).unwrap();

        assert_eq!(summary.cross_entropy, 0.0);
        assert_eq!(summary.brier_score, 0.0);
        assert_eq!(summary.top_one_accuracy, 1.0);
        assert!((summary.empirical_prior_cross_entropy - 3.0_f64.ln()).abs() < 1.0e-12);
        assert!((summary.cross_entropy_excess + 3.0_f64.ln()).abs() < 1.0e-12);
        assert!((summary.empirical_prior_brier_score - 2.0 / 3.0).abs() < 1.0e-12);
        assert!((summary.brier_score_excess + 2.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(summary.majority_class_accuracy, 1.0 / 3.0);
        assert!((summary.accuracy_over_majority - 2.0 / 3.0).abs() < 1.0e-12);
        assert_eq!(summary.mean_terminal_residual, 0.0);
        assert_eq!(summary.mean_absolute_terminal_residual, 0.0);
        assert_eq!(
            summary.classes[ValueClassV1::Win.index()].mean_signed_prediction,
            Some(1.0)
        );
        assert_eq!(
            summary.classes[ValueClassV1::Draw.index()].mean_signed_prediction,
            Some(0.0)
        );
        assert_eq!(
            summary.classes[ValueClassV1::Loss.index()].mean_signed_prediction,
            Some(-1.0)
        );
    }

    #[test]
    fn uniform_predictions_report_the_balanced_baseline() {
        let observations = [
            ValuePredictionObservationV1::new(ValueClassV1::Win, [1.0 / 3.0; 3]).unwrap(),
            ValuePredictionObservationV1::new(ValueClassV1::Draw, [1.0 / 3.0; 3]).unwrap(),
            ValuePredictionObservationV1::new(ValueClassV1::Loss, [1.0 / 3.0; 3]).unwrap(),
        ];
        let summary = summarize_value_predictions_v1(&observations).unwrap();

        assert!((summary.cross_entropy - 3.0_f64.ln()).abs() < 1.0e-7);
        assert!((summary.brier_score - 2.0 / 3.0).abs() < 1.0e-7);
        assert_eq!(summary.top_one_accuracy, 1.0 / 3.0);
        assert!((summary.cross_entropy_excess).abs() < 1.0e-7);
        assert!((summary.brier_score_excess).abs() < 1.0e-7);
        assert_eq!(summary.majority_class_accuracy, 1.0 / 3.0);
        assert_eq!(summary.accuracy_over_majority, 0.0);
        assert!(summary.mean_signed_prediction.abs() < 1.0e-12);
        assert!(summary.mean_signed_target.abs() < 1.0e-12);
        assert!((summary.mean_absolute_terminal_residual - 2.0 / 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn malformed_distributions_are_rejected() {
        assert!(matches!(
            ValuePredictionObservationV1::new(ValueClassV1::Win, [0.6, 0.3, 0.0]),
            Err(ValueDiagnosticsError::InvalidDistribution(_))
        ));
        assert!(matches!(
            ValuePredictionObservationV1::new(ValueClassV1::Win, [f32::NAN, 0.0, 1.0]),
            Err(ValueDiagnosticsError::InvalidProbability { index: 0, .. })
        ));
    }

    #[test]
    fn impossible_target_has_infinite_cross_entropy() {
        let observation =
            ValuePredictionObservationV1::new(ValueClassV1::Win, [0.0, 0.5, 0.5]).unwrap();
        let summary = summarize_value_predictions_v1(&[observation]).unwrap();

        assert_eq!(summary.cross_entropy, f64::INFINITY);
        assert_eq!(summary.cross_entropy_excess, f64::INFINITY);
    }
}
