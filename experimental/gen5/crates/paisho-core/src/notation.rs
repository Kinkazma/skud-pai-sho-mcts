use core::fmt;
use core::str::FromStr;

use crate::{
    Accent, AccentPlacement, Action, BasicFlower, Coordinate, CoordinateError, SpecialFlower,
    TileKind,
};

impl fmt::Display for Action {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plant { flower, gate } => write!(formatter, "plant {} {gate}", flower.code()),
            Self::Arrange { from, to } => write!(formatter, "arrange {from} {to}"),
            Self::SkipHarmonyBonus => formatter.write_str("skip-bonus"),
            Self::PlayAccent {
                accent,
                placement: AccentPlacement::At(at),
            } => write!(formatter, "accent {} at {at}", accent.code()),
            Self::PlayAccent {
                accent,
                placement:
                    AccentPlacement::BoatMove {
                        flower,
                        destination,
                    },
            } => write!(
                formatter,
                "accent {} move {flower} {destination}",
                accent.code()
            ),
            Self::PlantSpecial { flower, gate } => {
                write!(formatter, "plant-special {} {gate}", flower.code())
            }
            Self::BonusPlantBasic { flower, gate } => {
                write!(formatter, "bonus-plant {} {gate}", flower.code())
            }
        }
    }
}

impl FromStr for Action {
    type Err = ActionNotationError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let tokens: Vec<_> = text.split_whitespace().collect();
        match tokens.as_slice() {
            [] => Err(ActionNotationError::Empty),
            ["plant", flower, gate] => Ok(Self::Plant {
                flower: parse_basic(flower)?,
                gate: parse_coordinate(gate)?,
            }),
            ["arrange", from, to] => Ok(Self::Arrange {
                from: parse_coordinate(from)?,
                to: parse_coordinate(to)?,
            }),
            ["skip-bonus"] => Ok(Self::SkipHarmonyBonus),
            ["accent", accent, "at", at] => Ok(Self::PlayAccent {
                accent: parse_accent(accent)?,
                placement: AccentPlacement::At(parse_coordinate(at)?),
            }),
            ["accent", accent, "move", flower, destination] => {
                let accent = parse_accent(accent)?;
                if accent != Accent::Boat {
                    return Err(ActionNotationError::InvalidAccentPlacement);
                }
                Ok(Self::PlayAccent {
                    accent,
                    placement: AccentPlacement::BoatMove {
                        flower: parse_coordinate(flower)?,
                        destination: parse_coordinate(destination)?,
                    },
                })
            }
            ["plant-special", flower, gate] => Ok(Self::PlantSpecial {
                flower: parse_special(flower)?,
                gate: parse_coordinate(gate)?,
            }),
            ["bonus-plant", flower, gate] => Ok(Self::BonusPlantBasic {
                flower: parse_basic(flower)?,
                gate: parse_coordinate(gate)?,
            }),
            [known, ..] if is_action_keyword(known) => Err(ActionNotationError::WrongArity),
            _ => Err(ActionNotationError::UnknownAction),
        }
    }
}

fn parse_basic(text: &str) -> Result<BasicFlower, ActionNotationError> {
    match text.parse::<TileKind>() {
        Ok(TileKind::Basic(flower)) => Ok(flower),
        _ => Err(ActionNotationError::InvalidBasicFlower),
    }
}

fn parse_special(text: &str) -> Result<SpecialFlower, ActionNotationError> {
    match text.parse::<TileKind>() {
        Ok(TileKind::WhiteLotus) => Ok(SpecialFlower::WhiteLotus),
        Ok(TileKind::Orchid) => Ok(SpecialFlower::Orchid),
        _ => Err(ActionNotationError::InvalidSpecialFlower),
    }
}

fn parse_accent(text: &str) -> Result<Accent, ActionNotationError> {
    match text {
        "R" => Ok(Accent::Rock),
        "W" => Ok(Accent::Wheel),
        "K" => Ok(Accent::Knotweed),
        "B" => Ok(Accent::Boat),
        _ => Err(ActionNotationError::InvalidAccent),
    }
}

fn parse_coordinate(text: &str) -> Result<Coordinate, ActionNotationError> {
    text.parse().map_err(ActionNotationError::InvalidCoordinate)
}

fn is_action_keyword(text: &str) -> bool {
    matches!(
        text,
        "plant" | "arrange" | "skip-bonus" | "accent" | "plant-special" | "bonus-plant"
    )
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionNotationError {
    Empty,
    UnknownAction,
    WrongArity,
    InvalidBasicFlower,
    InvalidSpecialFlower,
    InvalidAccent,
    InvalidCoordinate(CoordinateError),
    InvalidAccentPlacement,
}

impl fmt::Display for ActionNotationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("action notation is empty"),
            Self::UnknownAction => formatter.write_str("unknown action notation"),
            Self::WrongArity => formatter.write_str("wrong number of action fields"),
            Self::InvalidBasicFlower => formatter.write_str("expected a Basic Flower code"),
            Self::InvalidSpecialFlower => formatter.write_str("expected L or O"),
            Self::InvalidAccent => formatter.write_str("expected R, W, K or B"),
            Self::InvalidCoordinate(error) => write!(formatter, "invalid coordinate: {error}"),
            Self::InvalidAccentPlacement => {
                formatter.write_str("only a Boat may use move placement notation")
            }
        }
    }
}

impl std::error::Error for ActionNotationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCoordinate(error) => Some(error),
            _ => None,
        }
    }
}
