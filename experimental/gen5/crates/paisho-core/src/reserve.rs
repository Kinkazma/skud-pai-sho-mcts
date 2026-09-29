use core::fmt;

use crate::{Accent, TileKind, ACCENTS, BASIC_FLOWERS};

/// The four Accent Tiles selected from the two available copies of each kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccentLoadout {
    counts: [u8; ACCENTS.len()],
}

impl AccentLoadout {
    pub const SELECTED_COUNT: u8 = 4;

    pub fn new(rock: u8, wheel: u8, knotweed: u8, boat: u8) -> Result<Self, AccentLoadoutError> {
        let counts = [rock, wheel, knotweed, boat];
        if counts.iter().any(|count| *count > 2) {
            return Err(AccentLoadoutError::MoreThanTwoOfAKind);
        }
        let total = counts.iter().copied().sum::<u8>();
        if total != Self::SELECTED_COUNT {
            return Err(AccentLoadoutError::WrongTotal { total });
        }
        Ok(Self { counts })
    }

    pub const fn balanced() -> Self {
        Self { counts: [1; 4] }
    }

    pub const fn count(self, accent: Accent) -> u8 {
        self.counts[accent as usize]
    }
}

impl Default for AccentLoadout {
    fn default() -> Self {
        Self::balanced()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AccentLoadoutError {
    MoreThanTwoOfAKind,
    WrongTotal { total: u8 },
}

impl fmt::Display for AccentLoadoutError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MoreThanTwoOfAKind => {
                formatter.write_str("a standard set has at most two of each Accent Tile")
            }
            Self::WrongTotal { total } => {
                write!(formatter, "choose exactly four Accent Tiles, not {total}")
            }
        }
    }
}

impl std::error::Error for AccentLoadoutError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Reserve {
    counts: [u8; TileKind::COUNT],
}

impl Reserve {
    pub fn standard(loadout: AccentLoadout) -> Self {
        let mut reserve = Self {
            counts: [0; TileKind::COUNT],
        };
        for flower in BASIC_FLOWERS {
            reserve.counts[TileKind::Basic(flower).index()] = 3;
        }
        reserve.counts[TileKind::WhiteLotus.index()] = 1;
        reserve.counts[TileKind::Orchid.index()] = 1;
        for accent in ACCENTS {
            reserve.counts[TileKind::Accent(accent).index()] = loadout.count(accent);
        }
        reserve
    }

    pub const fn count(&self, kind: TileKind) -> u8 {
        self.counts[kind.index()]
    }

    pub fn take(&mut self, kind: TileKind) -> bool {
        let count = &mut self.counts[kind.index()];
        if *count == 0 {
            return false;
        }
        *count -= 1;
        true
    }

    pub fn put_back(&mut self, kind: TileKind) {
        self.counts[kind.index()] += 1;
    }

    pub fn basic_count(&self) -> u8 {
        BASIC_FLOWERS
            .iter()
            .map(|flower| self.count(TileKind::Basic(*flower)))
            .sum()
    }

    pub fn total_count(&self) -> u8 {
        self.counts.iter().copied().sum()
    }
}
