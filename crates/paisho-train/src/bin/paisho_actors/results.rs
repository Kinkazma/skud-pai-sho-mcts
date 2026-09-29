use paisho_ai::PairedComparison;
use paisho_core::{GameOutcome, Player};
use paisho_replay::ReplayDigestV1;
use paisho_train::ReplayMatchResult;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct CandidateResults {
    pub(super) pairs: usize,
    pub(super) wins: usize,
    pub(super) draws: usize,
    pub(super) losses: usize,
    pub(super) pentanomial: [usize; 5],
}

impl CandidateResults {
    pub(super) fn from_paired_matches(
        matches: &[ReplayMatchResult],
        candidate: ReplayDigestV1,
    ) -> Result<Self, String> {
        if matches.len() % 2 != 0 {
            return Err("candidate result summary requires complete pairs".to_owned());
        }
        let outcomes = matches
            .iter()
            .map(|result| {
                let game = &result.game;
                let player = candidate_player(game.host_agent(), game.guest_agent(), candidate)?;
                Ok((game.outcome(), player))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Self::from_paired_outcomes(&outcomes)
    }

    fn from_paired_outcomes(outcomes: &[(GameOutcome, Player)]) -> Result<Self, String> {
        if outcomes.len() % 2 != 0 {
            return Err("candidate result summary requires complete pairs".to_owned());
        }
        let mut summary = Self::default();
        for pair in outcomes.chunks_exact(2) {
            let first = summary.absorb(pair[0])?;
            let second = summary.absorb(pair[1])?;
            summary.pentanomial[usize::from(first + second)] += 1;
            summary.pairs += 1;
        }
        Ok(summary)
    }

    pub(super) const fn half_points(self) -> usize {
        self.wins * 2 + self.draws
    }

    pub(super) const fn paired_comparison(self) -> PairedComparison {
        PairedComparison {
            zero: self.pentanomial[0],
            half: self.pentanomial[1],
            one: self.pentanomial[2],
            one_and_half: self.pentanomial[3],
            two: self.pentanomial[4],
            excluded: 0,
            excluded_pessimistic_ties: 0,
            excluded_pessimistic_losses: 0,
        }
    }

    fn absorb(&mut self, (outcome, candidate): (GameOutcome, Player)) -> Result<u8, String> {
        match outcome {
            GameOutcome::Win(winner) if winner == candidate => {
                self.wins += 1;
                Ok(2)
            }
            GameOutcome::Win(_) => {
                self.losses += 1;
                Ok(0)
            }
            GameOutcome::Draw => {
                self.draws += 1;
                Ok(1)
            }
            GameOutcome::Ongoing => {
                Err("candidate result summary received an ongoing game".to_owned())
            }
        }
    }
}

fn candidate_player(
    host: ReplayDigestV1,
    guest: ReplayDigestV1,
    candidate: ReplayDigestV1,
) -> Result<Player, String> {
    match (host == candidate, guest == candidate) {
        (true, false) => Ok(Player::Host),
        (false, true) => Ok(Player::Guest),
        _ => Err("paired result does not contain exactly one candidate seat".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_summary_keeps_game_results_and_all_five_pair_scores() {
        let outcomes = [
            (GameOutcome::Win(Player::Guest), Player::Host),
            (GameOutcome::Win(Player::Host), Player::Guest),
            (GameOutcome::Draw, Player::Host),
            (GameOutcome::Win(Player::Host), Player::Guest),
            (GameOutcome::Win(Player::Host), Player::Host),
            (GameOutcome::Win(Player::Host), Player::Guest),
            (GameOutcome::Win(Player::Host), Player::Host),
            (GameOutcome::Draw, Player::Guest),
            (GameOutcome::Win(Player::Host), Player::Host),
            (GameOutcome::Win(Player::Guest), Player::Guest),
        ];
        let summary = CandidateResults::from_paired_outcomes(&outcomes).unwrap();

        assert_eq!(summary.pairs, 5);
        assert_eq!((summary.wins, summary.draws, summary.losses), (4, 2, 4));
        assert_eq!(summary.half_points(), 10);
        assert_eq!(summary.pentanomial, [1, 1, 1, 1, 1]);
        assert_eq!(summary.paired_comparison().favorable(), 2);
        assert_eq!(summary.paired_comparison().tied(), 1);
        assert_eq!(summary.paired_comparison().unfavorable(), 2);
    }

    #[test]
    fn candidate_seat_must_be_unique() {
        let candidate = ReplayDigestV1::from_bytes([1; 32]);
        let opponent = ReplayDigestV1::from_bytes([2; 32]);
        assert_eq!(
            candidate_player(candidate, opponent, candidate),
            Ok(Player::Host)
        );
        assert!(candidate_player(candidate, candidate, candidate).is_err());
        assert!(candidate_player(opponent, opponent, candidate).is_err());
    }
}
