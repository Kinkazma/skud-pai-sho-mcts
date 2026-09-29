use core::fmt;
use core::str::FromStr;

use crate::{
    Accent, AccentLoadout, Action, ActionNotationError, ApplyError, BasicFlower, Position,
    RuleProfileId, StandardSetup, TileKind,
};

const MAGIC: &str = "PAISHO-RECORD 1";

/// Portable, deterministic input log. Derived position data and outcomes are
/// intentionally recomputed by the pinned rules engine during replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameRecord {
    rules: RuleProfileId,
    setup: StandardSetup,
    actions: Vec<Action>,
}

impl GameRecord {
    pub fn new(setup: StandardSetup) -> Self {
        Self::with_rules(setup, RuleProfileId::CURRENT)
    }

    pub fn with_rules(setup: StandardSetup, rules: RuleProfileId) -> Self {
        Self {
            rules,
            setup,
            actions: Vec::new(),
        }
    }

    pub const fn rules(&self) -> RuleProfileId {
        self.rules
    }

    pub const fn setup(&self) -> StandardSetup {
        self.setup
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    pub fn push(&mut self, action: Action) {
        self.actions.push(action);
    }

    pub fn initial_position(&self) -> Position {
        Position::from_standard_setup_with_rules(self.setup, self.rules)
    }

    pub fn replay(&self) -> Result<Position, ReplayError> {
        let mut position = self.initial_position();
        for (index, action) in self.actions.iter().copied().enumerate() {
            position.apply(action).map_err(|source| ReplayError {
                action_number: index + 1,
                action,
                source,
            })?;
        }
        Ok(position)
    }

    /// Explicitly reinterpret actions under another profile, retaining only the
    /// prefix up to its first terminal state. Never modifies the source record.
    /// Illegal actions before that terminal state still return a precise error.
    /// Old search targets/external outcomes must be revalidated separately.
    pub fn replay_prefix_with_rules(
        &self,
        rules: RuleProfileId,
    ) -> Result<(Self, Position), ReplayError> {
        let mut record = Self::with_rules(self.setup, rules);
        let mut position = record.initial_position();
        for (index, action) in self.actions.iter().copied().enumerate() {
            if position.outcome() != crate::GameOutcome::Ongoing {
                break;
            }
            position.apply(action).map_err(|source| ReplayError {
                action_number: index + 1,
                action,
                source,
            })?;
            record.push(action);
        }
        Ok((record, position))
    }
}

impl fmt::Display for GameRecord {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "{MAGIC}")?;
        writeln!(formatter, "rules {}", self.rules)?;
        writeln!(formatter, "start {}", self.setup.starting_flower.code())?;
        write_loadout(formatter, "host-accents", self.setup.host_accents)?;
        write_loadout(formatter, "guest-accents", self.setup.guest_accents)?;
        writeln!(formatter, "actions")?;
        for action in &self.actions {
            writeln!(formatter, "{action}")?;
        }
        Ok(())
    }
}

impl FromStr for GameRecord {
    type Err = RecordParseError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.trim().is_empty());

        let (line, magic) = next_line(&mut lines, "record signature")?;
        if magic != MAGIC {
            return Err(RecordParseError::UnexpectedLine {
                line,
                expected: MAGIC,
            });
        }

        let (line, rules_line) = next_line(&mut lines, "rules")?;
        let rules = field(rules_line, "rules ")?
            .parse()
            .map_err(|_| RecordParseError::InvalidRuleProfile { line })?;

        let (line, start_line) = next_line(&mut lines, "start")?;
        let starting_flower = parse_basic(field(start_line, "start ")?)
            .ok_or(RecordParseError::InvalidStartingFlower { line })?;

        let (line, host_line) = next_line(&mut lines, "host Accent loadout")?;
        let host_accents = parse_loadout(field(host_line, "host-accents ")?)
            .ok_or(RecordParseError::InvalidAccentLoadout { line })?;

        let (line, guest_line) = next_line(&mut lines, "guest Accent loadout")?;
        let guest_accents = parse_loadout(field(guest_line, "guest-accents ")?)
            .ok_or(RecordParseError::InvalidAccentLoadout { line })?;

        let (line, actions_header) = next_line(&mut lines, "actions")?;
        if actions_header != "actions" {
            return Err(RecordParseError::UnexpectedLine {
                line,
                expected: "actions",
            });
        }

        let mut actions = Vec::new();
        for (zero_based_line, text) in lines {
            let line = zero_based_line + 1;
            actions.push(
                text.trim()
                    .parse()
                    .map_err(|source| RecordParseError::InvalidAction { line, source })?,
            );
        }

        Ok(Self {
            rules,
            setup: StandardSetup {
                host_accents,
                guest_accents,
                starting_flower,
            },
            actions,
        })
    }
}

fn write_loadout(
    formatter: &mut fmt::Formatter<'_>,
    name: &str,
    loadout: AccentLoadout,
) -> fmt::Result {
    writeln!(
        formatter,
        "{name} {},{},{},{}",
        loadout.count(Accent::Rock),
        loadout.count(Accent::Wheel),
        loadout.count(Accent::Knotweed),
        loadout.count(Accent::Boat)
    )
}

fn next_line<'a>(
    lines: &mut impl Iterator<Item = (usize, &'a str)>,
    expected: &'static str,
) -> Result<(usize, &'a str), RecordParseError> {
    lines
        .next()
        .map(|(line, text)| (line + 1, text.trim()))
        .ok_or(RecordParseError::MissingLine { expected })
}

fn field<'a>(text: &'a str, prefix: &'static str) -> Result<&'a str, RecordParseError> {
    text.strip_prefix(prefix)
        .filter(|value| !value.is_empty())
        .ok_or(RecordParseError::InvalidField { expected: prefix })
}

fn parse_basic(text: &str) -> Option<BasicFlower> {
    match text.parse::<TileKind>().ok()? {
        TileKind::Basic(flower) => Some(flower),
        _ => None,
    }
}

fn parse_loadout(text: &str) -> Option<AccentLoadout> {
    let mut counts = text.split(',');
    let rock = counts.next()?.parse().ok()?;
    let wheel = counts.next()?.parse().ok()?;
    let knotweed = counts.next()?.parse().ok()?;
    let boat = counts.next()?.parse().ok()?;
    if counts.next().is_some() {
        return None;
    }
    AccentLoadout::new(rock, wheel, knotweed, boat).ok()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecordParseError {
    MissingLine {
        expected: &'static str,
    },
    UnexpectedLine {
        line: usize,
        expected: &'static str,
    },
    InvalidField {
        expected: &'static str,
    },
    InvalidRuleProfile {
        line: usize,
    },
    InvalidStartingFlower {
        line: usize,
    },
    InvalidAccentLoadout {
        line: usize,
    },
    InvalidAction {
        line: usize,
        source: ActionNotationError,
    },
}

impl fmt::Display for RecordParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingLine { expected } => write!(formatter, "missing {expected} line"),
            Self::UnexpectedLine { line, expected } => {
                write!(formatter, "line {line}: expected {expected}")
            }
            Self::InvalidField { expected } => write!(formatter, "expected field {expected}"),
            Self::InvalidRuleProfile { line } => {
                write!(formatter, "line {line}: unknown rule profile")
            }
            Self::InvalidStartingFlower { line } => {
                write!(formatter, "line {line}: invalid starting Basic Flower")
            }
            Self::InvalidAccentLoadout { line } => {
                write!(formatter, "line {line}: invalid Accent loadout")
            }
            Self::InvalidAction { line, source } => {
                write!(formatter, "line {line}: invalid action: {source}")
            }
        }
    }
}

impl std::error::Error for RecordParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidAction { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplayError {
    pub action_number: usize,
    pub action: Action,
    pub source: ApplyError,
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "action {} (`{}`) cannot be replayed: {}",
            self.action_number, self.action, self.source
        )
    }
}

impl std::error::Error for ReplayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}
