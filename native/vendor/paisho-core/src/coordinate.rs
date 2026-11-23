use core::fmt;
use core::str::FromStr;

/// Distance from the centre to an edge of the 17 × 17 coordinate envelope.
pub const BOARD_RADIUS: i8 = 8;

/// Width and height of the dense grid used by the rules engine and neural input.
pub const BOARD_SIZE: usize = 17;

/// A coordinate in site notation: `x` grows rightward and `y` upward.
///
/// A coordinate can denote one of the 40 non-playable corners inside the dense
/// 17 × 17 envelope. Use `point_type_at` to distinguish those positions.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Coordinate {
    x: i8,
    y: i8,
}

impl Coordinate {
    /// Creates a coordinate inside the dense 17 × 17 envelope.
    pub const fn new(x: i8, y: i8) -> Result<Self, CoordinateError> {
        if x < -BOARD_RADIUS || x > BOARD_RADIUS || y < -BOARD_RADIUS || y > BOARD_RADIUS {
            Err(CoordinateError::OutsideEnvelope {
                x: x as i16,
                y: y as i16,
            })
        } else {
            Ok(Self { x, y })
        }
    }

    pub(crate) const fn validated(x: i8, y: i8) -> Self {
        Self { x, y }
    }

    /// Converts dense zero-based `(row, column)` indices to site coordinates.
    pub const fn from_grid(row: usize, column: usize) -> Result<Self, CoordinateError> {
        if row >= BOARD_SIZE || column >= BOARD_SIZE {
            return Err(CoordinateError::GridIndexOutsideEnvelope { row, column });
        }

        Ok(Self {
            x: column as i8 - BOARD_RADIUS,
            y: BOARD_RADIUS - row as i8,
        })
    }

    pub const fn x(self) -> i8 {
        self.x
    }

    pub const fn y(self) -> i8 {
        self.y
    }

    /// Zero-based row, with row zero at the top like the reference site.
    pub const fn row(self) -> usize {
        (BOARD_RADIUS - self.y) as usize
    }

    /// Zero-based column, with column zero at the left like the reference site.
    pub const fn column(self) -> usize {
        (self.x + BOARD_RADIUS) as usize
    }

    pub const fn dense_index(self) -> usize {
        self.row() * BOARD_SIZE + self.column()
    }

    pub const fn translated(self, delta_x: i8, delta_y: i8) -> Option<Self> {
        let x = self.x + delta_x;
        let y = self.y + delta_y;
        if x < -BOARD_RADIUS || x > BOARD_RADIUS || y < -BOARD_RADIUS || y > BOARD_RADIUS {
            None
        } else {
            Some(Self::validated(x, y))
        }
    }

    pub const fn rotate_clockwise(self) -> Self {
        Self::validated(self.y, -self.x)
    }

    pub const fn rotate_180(self) -> Self {
        Self::validated(-self.x, -self.y)
    }

    pub const fn mirror_across_vertical_axis(self) -> Self {
        Self::validated(-self.x, self.y)
    }
}

impl fmt::Display for Coordinate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{},{}", self.x, self.y)
    }
}

impl FromStr for Coordinate {
    type Err = CoordinateError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let (x_text, y_text) = text
            .split_once(',')
            .ok_or(CoordinateError::InvalidNotation)?;
        if y_text.contains(',') {
            return Err(CoordinateError::InvalidNotation);
        }

        let x = x_text
            .trim()
            .parse::<i16>()
            .map_err(|_| CoordinateError::InvalidNumber)?;
        let y = y_text
            .trim()
            .parse::<i16>()
            .map_err(|_| CoordinateError::InvalidNumber)?;

        if x < -(BOARD_RADIUS as i16)
            || x > BOARD_RADIUS as i16
            || y < -(BOARD_RADIUS as i16)
            || y > BOARD_RADIUS as i16
        {
            return Err(CoordinateError::OutsideEnvelope { x, y });
        }

        Ok(Self::validated(x as i8, y as i8))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CoordinateError {
    InvalidNotation,
    InvalidNumber,
    OutsideEnvelope { x: i16, y: i16 },
    GridIndexOutsideEnvelope { row: usize, column: usize },
}

impl fmt::Display for CoordinateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidNotation => formatter.write_str("expected coordinate notation x,y"),
            Self::InvalidNumber => formatter.write_str("coordinate components must be integers"),
            Self::OutsideEnvelope { x, y } => {
                write!(formatter, "coordinate {x},{y} is outside -8..=8")
            }
            Self::GridIndexOutsideEnvelope { row, column } => {
                write!(formatter, "grid index ({row},{column}) is outside 0..17")
            }
        }
    }
}

impl std::error::Error for CoordinateError {}
