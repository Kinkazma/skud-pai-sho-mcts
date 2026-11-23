#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Player {
    Host = 0,
    Guest = 1,
}

impl Player {
    pub const fn opponent(self) -> Self {
        match self {
            Self::Host => Self::Guest,
            Self::Guest => Self::Host,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn code(self) -> char {
        match self {
            Self::Host => 'H',
            Self::Guest => 'G',
        }
    }
}
