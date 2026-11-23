use core::fmt;
use core::str::FromStr;

/// Stable identifier embedded in positions, game records and future checkpoints.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RuleProfileId {
    /// Historical replay semantics, including the old incomplete ring detector.
    SkudPaiSho2022,
    SkudPaiSho2022V2,
    /// Gen5 training variant: the player who leaves the opponent blocked loses.
    SkudPaiShoGen5V1,
}

impl RuleProfileId {
    pub const CURRENT: Self = Self::SkudPaiSho2022V2;

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SkudPaiSho2022 => "skud-pai-sho-2022-03-14",
            Self::SkudPaiSho2022V2 => "skud-pai-sho-2022-03-14-v2",
            Self::SkudPaiShoGen5V1 => "skud-pai-sho-gen5-v1",
        }
    }

    pub const fn profile(self) -> &'static RuleProfile {
        match self {
            Self::SkudPaiSho2022 => &LEGACY_RULE_PROFILE,
            Self::SkudPaiSho2022V2 => &STANDARD_RULE_PROFILE,
            Self::SkudPaiShoGen5V1 => &GEN5_RULE_PROFILE,
        }
    }
}

impl fmt::Display for RuleProfileId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for RuleProfileId {
    type Err = RuleProfileIdError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "skud-pai-sho-2022-03-14" => Ok(Self::SkudPaiSho2022),
            "skud-pai-sho-2022-03-14-v2" => Ok(Self::SkudPaiSho2022V2),
            "skud-pai-sho-gen5-v1" => Ok(Self::SkudPaiShoGen5V1),
            _ => Err(RuleProfileIdError),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleProfile {
    pub id: RuleProfileId,
    pub rulebook_revision: &'static str,
    pub reference_source_commit: &'static str,
    pub formal_same_flower_start: bool,
    pub limited_harmony_bonus_gates: bool,
    pub modern_knotweed: bool,
    pub wheel_may_be_played_near_gates: bool,
    pub rocks_are_unwheelable: bool,
    pub white_lotus_protected_from_basic_flowers: bool,
    pub wild_orchid_captures_any_flower: bool,
    pub complete_harmony_ring_detection: bool,
    pub blocking_player_loses: bool,
}

/// Frozen historical semantics used when reopening a V1 record.
pub const LEGACY_RULE_PROFILE: RuleProfile = RuleProfile {
    id: RuleProfileId::SkudPaiSho2022,
    rulebook_revision: "2022-03-14",
    reference_source_commit: "b849dbdabb1138ff0f6d609adf38b301c2f875ae",
    formal_same_flower_start: true,
    limited_harmony_bonus_gates: true,
    modern_knotweed: true,
    wheel_may_be_played_near_gates: true,
    rocks_are_unwheelable: true,
    white_lotus_protected_from_basic_flowers: true,
    wild_orchid_captures_any_flower: true,
    complete_harmony_ring_detection: false,
    blocking_player_loses: false,
};

/// New games follow the same official rulebook with complete ring detection.
pub const STANDARD_RULE_PROFILE: RuleProfile = RuleProfile {
    id: RuleProfileId::CURRENT,
    complete_harmony_ring_detection: true,
    ..LEGACY_RULE_PROFILE
};

pub const GEN5_RULE_PROFILE: RuleProfile = RuleProfile {
    id: RuleProfileId::SkudPaiShoGen5V1,
    blocking_player_loses: true,
    ..STANDARD_RULE_PROFILE
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleProfileIdError;

impl fmt::Display for RuleProfileIdError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("unknown Skud Pai Sho rule profile")
    }
}

impl std::error::Error for RuleProfileIdError {}
