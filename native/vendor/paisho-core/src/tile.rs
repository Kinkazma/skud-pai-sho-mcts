use core::fmt;
use core::str::FromStr;

use crate::Player;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FlowerColor {
    Red,
    White,
}

impl FlowerColor {
    pub const fn opposite(self) -> Self {
        match self {
            Self::Red => Self::White,
            Self::White => Self::Red,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum BasicFlower {
    Red3 = 0,
    Red4 = 1,
    Red5 = 2,
    White3 = 3,
    White4 = 4,
    White5 = 5,
}

pub const BASIC_FLOWERS: [BasicFlower; 6] = [
    BasicFlower::Red3,
    BasicFlower::Red4,
    BasicFlower::Red5,
    BasicFlower::White3,
    BasicFlower::White4,
    BasicFlower::White5,
];

impl BasicFlower {
    pub const fn color(self) -> FlowerColor {
        match self {
            Self::Red3 | Self::Red4 | Self::Red5 => FlowerColor::Red,
            Self::White3 | Self::White4 | Self::White5 => FlowerColor::White,
        }
    }

    pub const fn movement(self) -> u8 {
        match self {
            Self::Red3 | Self::White3 => 3,
            Self::Red4 | Self::White4 => 4,
            Self::Red5 | Self::White5 => 5,
        }
    }

    pub const fn code(self) -> &'static str {
        match self {
            Self::Red3 => "R3",
            Self::Red4 => "R4",
            Self::Red5 => "R5",
            Self::White3 => "W3",
            Self::White4 => "W4",
            Self::White5 => "W5",
        }
    }

    /// Adjacency in the six-tile Circle of Harmony.
    pub const fn harmonizes_with(self, other: Self) -> bool {
        let left = self as i8;
        let right = other as i8;
        let difference = if left >= right {
            left - right
        } else {
            right - left
        };
        difference == 1 || difference == 5
    }

    pub const fn clashes_with(self, other: Self) -> bool {
        self.color() as u8 != other.color() as u8 && self.movement() == other.movement()
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Accent {
    Rock = 0,
    Wheel = 1,
    Knotweed = 2,
    Boat = 3,
}

pub const ACCENTS: [Accent; 4] = [Accent::Rock, Accent::Wheel, Accent::Knotweed, Accent::Boat];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SpecialFlower {
    WhiteLotus,
    Orchid,
}

pub const SPECIAL_FLOWERS: [SpecialFlower; 2] = [SpecialFlower::WhiteLotus, SpecialFlower::Orchid];

impl SpecialFlower {
    pub const fn kind(self) -> TileKind {
        match self {
            Self::WhiteLotus => TileKind::WhiteLotus,
            Self::Orchid => TileKind::Orchid,
        }
    }

    pub const fn code(self) -> &'static str {
        self.kind().code()
    }
}

impl Accent {
    pub const fn code(self) -> &'static str {
        match self {
            Self::Rock => "R",
            Self::Wheel => "W",
            Self::Knotweed => "K",
            Self::Boat => "B",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TileKind {
    Basic(BasicFlower),
    WhiteLotus,
    Orchid,
    Accent(Accent),
}

pub const STANDARD_TILE_KINDS: [TileKind; 12] = [
    TileKind::Basic(BasicFlower::Red3),
    TileKind::Basic(BasicFlower::Red4),
    TileKind::Basic(BasicFlower::Red5),
    TileKind::Basic(BasicFlower::White3),
    TileKind::Basic(BasicFlower::White4),
    TileKind::Basic(BasicFlower::White5),
    TileKind::WhiteLotus,
    TileKind::Orchid,
    TileKind::Accent(Accent::Rock),
    TileKind::Accent(Accent::Wheel),
    TileKind::Accent(Accent::Knotweed),
    TileKind::Accent(Accent::Boat),
];

impl TileKind {
    pub const COUNT: usize = STANDARD_TILE_KINDS.len();

    pub const fn index(self) -> usize {
        match self {
            Self::Basic(flower) => flower as usize,
            Self::WhiteLotus => 6,
            Self::Orchid => 7,
            Self::Accent(accent) => 8 + accent as usize,
        }
    }

    pub const fn code(self) -> &'static str {
        match self {
            Self::Basic(flower) => flower.code(),
            Self::WhiteLotus => "L",
            Self::Orchid => "O",
            Self::Accent(accent) => accent.code(),
        }
    }

    pub const fn movement(self) -> Option<u8> {
        match self {
            Self::Basic(flower) => Some(flower.movement()),
            Self::WhiteLotus => Some(2),
            Self::Orchid => Some(6),
            Self::Accent(_) => None,
        }
    }

    pub const fn is_flower(self) -> bool {
        !matches!(self, Self::Accent(_))
    }

    pub const fn is_basic(self) -> bool {
        matches!(self, Self::Basic(_))
    }

    pub const fn is_special(self) -> bool {
        matches!(self, Self::WhiteLotus | Self::Orchid)
    }

    pub const fn is_accent(self) -> bool {
        matches!(self, Self::Accent(_))
    }
}

impl fmt::Display for TileKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl FromStr for TileKind {
    type Err = TileCodeError;

    fn from_str(code: &str) -> Result<Self, Self::Err> {
        match code {
            "R3" => Ok(Self::Basic(BasicFlower::Red3)),
            "R4" => Ok(Self::Basic(BasicFlower::Red4)),
            "R5" => Ok(Self::Basic(BasicFlower::Red5)),
            "W3" => Ok(Self::Basic(BasicFlower::White3)),
            "W4" => Ok(Self::Basic(BasicFlower::White4)),
            "W5" => Ok(Self::Basic(BasicFlower::White5)),
            "L" => Ok(Self::WhiteLotus),
            "O" => Ok(Self::Orchid),
            "R" => Ok(Self::Accent(Accent::Rock)),
            "W" => Ok(Self::Accent(Accent::Wheel)),
            "K" => Ok(Self::Accent(Accent::Knotweed)),
            "B" => Ok(Self::Accent(Accent::Boat)),
            _ => Err(TileCodeError),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TileCodeError;

impl fmt::Display for TileCodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("unknown standard Skud Pai Sho tile code")
    }
}

impl std::error::Error for TileCodeError {}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Tile {
    pub owner: Player,
    pub kind: TileKind,
}

impl Tile {
    pub const fn new(owner: Player, kind: TileKind) -> Self {
        Self { owner, kind }
    }

    /// Returns the owner of a valid pairwise Harmony before board-line effects.
    pub const fn harmony_owner_with(self, other: Self) -> Option<Player> {
        match (self.kind, other.kind) {
            (TileKind::Basic(left), TileKind::Basic(right)) => {
                if self.owner as u8 == other.owner as u8 && left.harmonizes_with(right) {
                    Some(self.owner)
                } else {
                    None
                }
            }
            (TileKind::WhiteLotus, TileKind::Basic(_)) => Some(other.owner),
            (TileKind::Basic(_), TileKind::WhiteLotus) => Some(self.owner),
            _ => None,
        }
    }

    pub const fn clashes_with(self, other: Self) -> bool {
        match (self.kind, other.kind) {
            (TileKind::Basic(left), TileKind::Basic(right)) => left.clashes_with(right),
            _ => false,
        }
    }
}
