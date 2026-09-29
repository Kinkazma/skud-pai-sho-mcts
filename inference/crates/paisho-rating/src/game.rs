use core::fmt;

/// Stable identifier for one immutable agent configuration.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct AgentId(String);

impl AgentId {
    pub fn new(value: impl Into<String>) -> Result<Self, AgentIdError> {
        let value = value.into();
        if value.is_empty() {
            return Err(AgentIdError::Empty);
        }
        if value
            .chars()
            .any(|character| character.is_control() || matches!(character, '\t' | '\n' | '\r'))
        {
            return Err(AgentIdError::ControlCharacter);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentIdError {
    Empty,
    ControlCharacter,
}

impl fmt::Display for AgentIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("an agent id cannot be empty"),
            Self::ControlCharacter => {
                formatter.write_str("an agent id cannot contain control characters")
            }
        }
    }
}

impl std::error::Error for AgentIdError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RatedOutcome {
    HostWin,
    Draw,
    GuestWin,
}

impl RatedOutcome {
    pub const fn host_score(self) -> f64 {
        match self {
            Self::HostWin => 1.0,
            Self::Draw => 0.5,
            Self::GuestWin => 0.0,
        }
    }

    pub const fn reversed(self) -> Self {
        match self {
            Self::HostWin => Self::GuestWin,
            Self::Draw => Self::Draw,
            Self::GuestWin => Self::HostWin,
        }
    }
}

/// One complete, rating-eligible game. `sequence` is the immutable chronology
/// used only by order-dependent calculators; `pair_id` clusters reversed-seat
/// games for uncertainty estimates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RatedGame {
    sequence: u64,
    pair_id: u64,
    host: AgentId,
    guest: AgentId,
    outcome: RatedOutcome,
}

impl RatedGame {
    pub fn new(
        sequence: u64,
        pair_id: u64,
        host: AgentId,
        guest: AgentId,
        outcome: RatedOutcome,
    ) -> Result<Self, RatedGameError> {
        if host == guest {
            return Err(RatedGameError::SameAgent);
        }
        Ok(Self {
            sequence,
            pair_id,
            host,
            guest,
            outcome,
        })
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn pair_id(&self) -> u64 {
        self.pair_id
    }

    pub const fn host(&self) -> &AgentId {
        &self.host
    }

    pub const fn guest(&self) -> &AgentId {
        &self.guest
    }

    pub const fn outcome(&self) -> RatedOutcome {
        self.outcome
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RatedGameError {
    SameAgent,
}

impl fmt::Display for RatedGameError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a rated game needs two distinct agents")
    }
}

impl std::error::Error for RatedGameError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_safe_for_text_archives() {
        assert_eq!(AgentId::new("mcts-128").unwrap().as_str(), "mcts-128");
        assert_eq!(AgentId::new(""), Err(AgentIdError::Empty));
        assert_eq!(AgentId::new("bad\tid"), Err(AgentIdError::ControlCharacter));
    }

    #[test]
    fn outcomes_reverse_without_changing_draws() {
        assert_eq!(RatedOutcome::HostWin.reversed(), RatedOutcome::GuestWin);
        assert_eq!(RatedOutcome::Draw.reversed(), RatedOutcome::Draw);
    }
}
