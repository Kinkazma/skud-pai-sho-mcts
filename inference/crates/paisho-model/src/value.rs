use core::fmt;

use paisho_core::{GameOutcome, Player};

pub const VALUE_CLASS_COUNT_V1: usize = 3;

/// Stable WDL order used by every V1 backend: win, draw, loss for the player
/// whose perspective encoded the position.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ValueClassV1 {
    Win = 0,
    Draw = 1,
    Loss = 2,
}

impl ValueClassV1 {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn one_hot(self) -> [f32; VALUE_CLASS_COUNT_V1] {
        let mut target = [0.0; VALUE_CLASS_COUNT_V1];
        target[self.index()] = 1.0;
        target
    }

    /// Undiscounted terminal return from the encoded player's perspective.
    pub const fn signed_return(self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.0,
            Self::Loss => -1.0,
        }
    }

    pub fn from_terminal_outcome(
        outcome: GameOutcome,
        perspective: Player,
    ) -> Result<Self, ValueTargetError> {
        match outcome {
            GameOutcome::Win(winner) if winner == perspective => Ok(Self::Win),
            GameOutcome::Win(_) => Ok(Self::Loss),
            GameOutcome::Draw => Ok(Self::Draw),
            GameOutcome::Ongoing => Err(ValueTargetError::OngoingGame),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueTargetError {
    OngoingGame,
}

impl fmt::Display for ValueTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OngoingGame => formatter.write_str("an ongoing game has no terminal WDL target"),
        }
    }
}

impl std::error::Error for ValueTargetError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_targets_have_named_win_draw_loss_order() {
        assert_eq!(
            ValueClassV1::from_terminal_outcome(GameOutcome::Win(Player::Host), Player::Host),
            Ok(ValueClassV1::Win)
        );
        assert_eq!(
            ValueClassV1::from_terminal_outcome(GameOutcome::Draw, Player::Guest),
            Ok(ValueClassV1::Draw)
        );
        assert_eq!(
            ValueClassV1::from_terminal_outcome(GameOutcome::Win(Player::Host), Player::Guest),
            Ok(ValueClassV1::Loss)
        );
        assert_eq!(ValueClassV1::Win.one_hot(), [1.0, 0.0, 0.0]);
        assert_eq!(ValueClassV1::Draw.one_hot(), [0.0, 1.0, 0.0]);
        assert_eq!(ValueClassV1::Loss.one_hot(), [0.0, 0.0, 1.0]);
        assert_eq!(ValueClassV1::Win.signed_return(), 1.0);
        assert_eq!(ValueClassV1::Draw.signed_return(), 0.0);
        assert_eq!(ValueClassV1::Loss.signed_return(), -1.0);
    }
}
