use core::fmt;
use std::collections::HashSet;

use paisho_model::ActionEncodingV1;

use crate::ReplayDigestV1;

const DISTRIBUTION_TOLERANCE: f64 = 1.0e-5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PolicyTargetKindV1 {
    Behavior = 0,
    MctsVisit = 1,
    Teacher = 2,
    PlayedAction = 3,
}

impl PolicyTargetKindV1 {
    pub(crate) const fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Behavior),
            1 => Some(Self::MctsVisit),
            2 => Some(Self::Teacher),
            3 => Some(Self::PlayedAction),
            _ => None,
        }
    }

    pub(crate) const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolicyEntryV1 {
    action: ActionEncodingV1,
    probability: f32,
}

impl PolicyEntryV1 {
    pub fn new(action: ActionEncodingV1, probability: f32) -> Result<Self, PolicyTargetV1Error> {
        if !probability.is_finite() {
            return Err(PolicyTargetV1Error::NonFiniteProbability);
        }
        if probability <= 0.0 {
            return Err(PolicyTargetV1Error::NonPositiveProbability(probability));
        }
        Ok(Self {
            action,
            probability,
        })
    }

    pub const fn action(self) -> ActionEncodingV1 {
        self.action
    }

    pub const fn probability(self) -> f32 {
        self.probability
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyTargetV1 {
    kind: PolicyTargetKindV1,
    producer: ReplayDigestV1,
    entries: Vec<PolicyEntryV1>,
}

impl PolicyTargetV1 {
    pub fn new(
        kind: PolicyTargetKindV1,
        producer: ReplayDigestV1,
        mut entries: Vec<PolicyEntryV1>,
    ) -> Result<Self, PolicyTargetV1Error> {
        if entries.is_empty() {
            return Err(PolicyTargetV1Error::Empty);
        }
        entries.sort_by_key(|entry| entry.action.slots());
        let mut actions = HashSet::with_capacity(entries.len());
        let mut sum = 0.0_f64;
        for entry in &entries {
            if !actions.insert(entry.action) {
                return Err(PolicyTargetV1Error::DuplicateAction(entry.action));
            }
            sum += f64::from(entry.probability);
        }
        if (sum - 1.0).abs() > DISTRIBUTION_TOLERANCE {
            return Err(PolicyTargetV1Error::InvalidDistribution(sum));
        }
        if kind == PolicyTargetKindV1::PlayedAction
            && (entries.len() != 1 || entries[0].probability != 1.0)
        {
            return Err(PolicyTargetV1Error::PlayedActionMustBeOneHot);
        }
        Ok(Self {
            kind,
            producer,
            entries,
        })
    }

    pub fn one_hot(
        producer: ReplayDigestV1,
        action: ActionEncodingV1,
    ) -> Result<Self, PolicyTargetV1Error> {
        Self::new(
            PolicyTargetKindV1::PlayedAction,
            producer,
            vec![PolicyEntryV1::new(action, 1.0)?],
        )
    }

    pub const fn kind(&self) -> PolicyTargetKindV1 {
        self.kind
    }

    pub const fn producer(&self) -> ReplayDigestV1 {
        self.producer
    }

    pub fn entries(&self) -> &[PolicyEntryV1] {
        &self.entries
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PolicyTargetV1Error {
    Empty,
    NonFiniteProbability,
    NonPositiveProbability(f32),
    DuplicateAction(ActionEncodingV1),
    InvalidDistribution(f64),
    PlayedActionMustBeOneHot,
}

impl fmt::Display for PolicyTargetV1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("a policy target cannot be empty"),
            Self::NonFiniteProbability => {
                formatter.write_str("a policy target contains a non-finite probability")
            }
            Self::NonPositiveProbability(value) => write!(
                formatter,
                "stored policy probabilities must be positive; omit zero entries, got {value}"
            ),
            Self::DuplicateAction(action) => {
                write!(
                    formatter,
                    "policy target repeats action {:?}",
                    action.slots()
                )
            }
            Self::InvalidDistribution(sum) => {
                write!(formatter, "policy target sums to {sum}, not 1")
            }
            Self::PlayedActionMustBeOneHot => {
                formatter.write_str("a played-action target must contain one action at weight 1")
            }
        }
    }
}

impl std::error::Error for PolicyTargetV1Error {}

#[cfg(test)]
mod tests {
    use paisho_model::{ActionFamilyV1, NO_COORDINATE_V1, NO_TILE_V1};

    use super::*;

    fn pass() -> ActionEncodingV1 {
        ActionEncodingV1::from_slots([
            ActionFamilyV1::SkipHarmonyBonus as u16,
            NO_TILE_V1,
            NO_COORDINATE_V1,
            NO_COORDINATE_V1,
        ])
        .unwrap()
    }

    #[test]
    fn sparse_targets_are_canonical_and_normalized() {
        let target = PolicyTargetV1::new(
            PolicyTargetKindV1::Behavior,
            ReplayDigestV1::from_bytes([1; 32]),
            vec![PolicyEntryV1::new(pass(), 1.0).unwrap()],
        )
        .unwrap();
        assert_eq!(target.entries()[0].action(), pass());
        assert!(matches!(
            PolicyTargetV1::new(
                PolicyTargetKindV1::Behavior,
                ReplayDigestV1::from_bytes([1; 32]),
                vec![PolicyEntryV1::new(pass(), 0.5).unwrap()],
            ),
            Err(PolicyTargetV1Error::InvalidDistribution(_))
        ));
    }
}
