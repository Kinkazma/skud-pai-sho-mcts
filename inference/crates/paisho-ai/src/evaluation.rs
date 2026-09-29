use paisho_core::{GameOutcome, Player};

/// A rated game score represented in half-points to avoid floating-point
/// comparisons: loss = 0, draw = 1, win = 2.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameScore {
    Loss,
    Draw,
    Win,
}

impl GameScore {
    pub fn from_outcome(outcome: GameOutcome, candidate: Player) -> Option<Self> {
        match outcome {
            GameOutcome::Win(winner) if winner == candidate => Some(Self::Win),
            GameOutcome::Win(_) => Some(Self::Loss),
            GameOutcome::Draw => Some(Self::Draw),
            GameOutcome::Ongoing => None,
        }
    }

    const fn half_points(self) -> usize {
        match self {
            Self::Loss => 0,
            Self::Draw => 1,
            Self::Win => 2,
        }
    }
}

/// Conservative evidence from paired games with reversed seats. A pair is
/// excluded if either game is unrated, and tied pairs do not enter the sign
/// test. This avoids pretending that the two games sharing a setup are fully
/// independent observations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PairedComparison {
    /// Candidate scores 0, 0.5, 1, 1.5 and 2 across the two reversed-seat games.
    pub zero: usize,
    pub half: usize,
    pub one: usize,
    pub one_and_half: usize,
    pub two: usize,
    pub excluded: usize,
    /// Excluded pairs that remain tied if every missing game is scored as a loss.
    pub excluded_pessimistic_ties: usize,
    /// Excluded pairs that become unfavorable under the same convention.
    pub excluded_pessimistic_losses: usize,
}

impl PairedComparison {
    pub fn observe(&mut self, first: Option<GameScore>, second: Option<GameScore>) {
        let (Some(first), Some(second)) = (first, second) else {
            self.excluded += 1;
            let known_half_points = first.map(GameScore::half_points).unwrap_or(0)
                + second.map(GameScore::half_points).unwrap_or(0);
            if known_half_points >= 2 {
                self.excluded_pessimistic_ties += 1;
            } else {
                self.excluded_pessimistic_losses += 1;
            }
            return;
        };
        match first.half_points() + second.half_points() {
            0 => self.zero += 1,
            1 => self.half += 1,
            2 => self.one += 1,
            3 => self.one_and_half += 1,
            4 => self.two += 1,
            _ => unreachable!("two games cannot award more than four half-points"),
        }
    }

    pub const fn rated_pairs(self) -> usize {
        self.zero + self.half + self.one + self.one_and_half + self.two
    }

    pub const fn favorable(self) -> usize {
        self.one_and_half + self.two
    }

    pub const fn tied(self) -> usize {
        self.one
    }

    pub const fn unfavorable(self) -> usize {
        self.zero + self.half
    }

    pub const fn decisive_pairs(self) -> usize {
        self.favorable() + self.unfavorable()
    }

    /// Exact two-sided binomial sign-test p-value under equal probability of a
    /// favorable or unfavorable decisive pair. Tied/excluded pairs are ignored.
    pub fn exact_two_sided_sign_test_p_value(self) -> f64 {
        exact_two_sided_sign_test(self.favorable(), self.unfavorable())
    }

    /// Treats every missing game as a candidate loss before applying the same
    /// exact sign test. This guards against informative censoring.
    pub fn pessimistic_exact_two_sided_sign_test_p_value(self) -> f64 {
        exact_two_sided_sign_test(
            self.favorable(),
            self.unfavorable() + self.excluded_pessimistic_losses,
        )
    }
}

fn exact_two_sided_sign_test(favorable: usize, unfavorable: usize) -> f64 {
    let trials = favorable + unfavorable;
    if trials == 0 {
        return 1.0;
    }
    let tail_end = favorable.min(unfavorable);
    let log_terms: Vec<_> = (0..=tail_end)
        .map(|successes| log_binomial_probability(trials, successes))
        .collect();
    let maximum = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_tail = maximum
        + log_terms
            .iter()
            .map(|term| (term - maximum).exp())
            .sum::<f64>()
            .ln();
    (2.0 * log_tail.exp()).min(1.0)
}

fn log_binomial_probability(trials: usize, successes: usize) -> f64 {
    let successes = successes.min(trials - successes);
    let log_coefficient = (1..=successes).fold(0.0, |sum, index| {
        sum + ((trials + 1 - index) as f64).ln() - (index as f64).ln()
    });
    log_coefficient - trials as f64 * core::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    use super::{GameScore, PairedComparison};

    #[test]
    fn paired_comparison_counts_half_points_and_exclusions() {
        let mut comparison = PairedComparison::default();
        comparison.observe(Some(GameScore::Win), Some(GameScore::Draw));
        comparison.observe(Some(GameScore::Win), Some(GameScore::Loss));
        comparison.observe(Some(GameScore::Draw), Some(GameScore::Loss));
        comparison.observe(None, Some(GameScore::Win));

        assert_eq!(comparison.zero, 0);
        assert_eq!(comparison.half, 1);
        assert_eq!(comparison.one, 1);
        assert_eq!(comparison.one_and_half, 1);
        assert_eq!(comparison.two, 0);
        assert_eq!(comparison.excluded, 1);
        assert_eq!(comparison.excluded_pessimistic_ties, 1);
        assert_eq!(comparison.excluded_pessimistic_losses, 0);
        assert_eq!(comparison.rated_pairs(), 3);
        assert_eq!(comparison.decisive_pairs(), 2);
    }

    #[test]
    fn exact_sign_test_matches_known_small_cases() {
        let all_one_way = PairedComparison {
            two: 10,
            one: 3,
            excluded: 1,
            ..PairedComparison::default()
        };
        assert!((all_one_way.exact_two_sided_sign_test_p_value() - 0.001_953_125).abs() < 1e-12);

        let eight_to_two = PairedComparison {
            one_and_half: 8,
            half: 2,
            ..PairedComparison::default()
        };
        assert!((eight_to_two.exact_two_sided_sign_test_p_value() - 0.109_375).abs() < 1e-12);

        let tied = PairedComparison {
            two: 5,
            zero: 5,
            ..PairedComparison::default()
        };
        assert_eq!(tied.exact_two_sided_sign_test_p_value(), 1.0);
    }

    #[test]
    fn pessimistic_test_scores_missing_games_as_losses() {
        let mut comparison = PairedComparison::default();
        comparison.observe(Some(GameScore::Win), None);
        comparison.observe(Some(GameScore::Draw), None);
        comparison.observe(None, None);
        assert_eq!(comparison.excluded, 3);
        assert_eq!(comparison.excluded_pessimistic_ties, 1);
        assert_eq!(comparison.excluded_pessimistic_losses, 2);

        let mut evidence = PairedComparison {
            two: 8,
            ..PairedComparison::default()
        };
        evidence.observe(Some(GameScore::Loss), None);
        assert!(
            evidence.pessimistic_exact_two_sided_sign_test_p_value()
                > evidence.exact_two_sided_sign_test_p_value()
        );
    }
}
