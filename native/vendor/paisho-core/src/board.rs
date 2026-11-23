use core::fmt;

use crate::{point_type_at, Coordinate, Tile, BOARD_SIZE};

pub const CELL_COUNT: usize = BOARD_SIZE * BOARD_SIZE;

#[derive(Clone, Debug)]
pub struct Board {
    cells: [Option<Tile>; CELL_COUNT],
    rows: [u32; BOARD_SIZE],
    columns: [u32; BOARD_SIZE],
    clash: std::sync::OnceLock<bool>,
    harmonies: std::sync::OnceLock<std::sync::Arc<[crate::Harmony]>>,
}

impl Board {
    pub const fn empty() -> Self {
        Self {
            cells: [None; CELL_COUNT],
            rows: gate_masks(),
            columns: gate_masks(),
            clash: std::sync::OnceLock::new(),
            harmonies: std::sync::OnceLock::new(),
        }
    }

    pub const fn get(&self, coordinate: Coordinate) -> Option<Tile> {
        self.cells[index(coordinate)]
    }

    pub const fn is_empty(&self, coordinate: Coordinate) -> bool {
        self.get(coordinate).is_none()
    }

    pub fn place(&mut self, coordinate: Coordinate, tile: Tile) -> Result<(), BoardError> {
        if !point_type_at(coordinate).is_playable() {
            return Err(BoardError::NonPlayable(coordinate));
        }
        if self.get(coordinate).is_some() {
            return Err(BoardError::Occupied(coordinate));
        }
        self.replace(coordinate, Some(tile));
        Ok(())
    }

    pub fn remove(&mut self, coordinate: Coordinate) -> Option<Tile> {
        self.replace(coordinate, None)
    }

    #[inline]
    pub fn occupied(&self) -> impl Iterator<Item = (Coordinate, Tile)> + '_ {
        Occupied {
            board: self,
            row: 0,
            remaining: self.rows[0],
        }
    }

    pub fn occupied_count(&self) -> usize {
        self.occupied().count()
    }

    pub(crate) fn replace(&mut self, coordinate: Coordinate, tile: Option<Tile>) -> Option<Tile> {
        self.clash.take();
        self.harmonies.take();
        if !point_type_at(coordinate).is_gate() {
            let row = coordinate.row();
            let column = coordinate.column();
            if tile.is_some() {
                self.rows[row] |= 1 << column;
                self.columns[column] |= 1 << row;
            } else {
                self.rows[row] &= !(1 << column);
                self.columns[column] &= !(1 << row);
            }
        }
        core::mem::replace(&mut self.cells[index(coordinate)], tile)
    }

    pub(crate) fn cached_clash(&self, calculate: impl FnOnce() -> bool) -> bool {
        *self.clash.get_or_init(calculate)
    }
    /// An exact-board cache shared by clones until either board is mutated.
    /// All piece changes pass through replace(), including accent effects.
    pub(crate) fn cached_harmonies(
        &self,
        calculate: impl FnOnce() -> Vec<crate::Harmony>,
    ) -> &[crate::Harmony] {
        self.harmonies.get_or_init(|| calculate().into()).as_ref()
    }
    /// Cached occupied intersections and permanent gate sentinels per line.
    /// The nearest sentinel stops visibility even when its gate is empty.
    pub(crate) fn line_mask(&self, origin: Coordinate, horizontal: bool) -> u32 {
        if horizontal {
            self.rows[origin.row()]
        } else {
            self.columns[origin.column()]
        }
    }
    pub(crate) fn nearest_on_line(
        origin: Coordinate,
        dx: i8,
        dy: i8,
        mask: u32,
    ) -> Option<Coordinate> {
        let horizontal = dx != 0;
        let at = if horizontal {
            origin.column()
        } else {
            origin.row()
        };
        let forward = dx > 0 || dy < 0;
        let mask = if forward {
            mask & (u32::MAX << (at + 1))
        } else {
            mask & ((1 << at) - 1)
        };
        if mask == 0 {
            return None;
        }
        let next = if forward {
            mask.trailing_zeros() as usize
        } else {
            31 - mask.leading_zeros() as usize
        };
        let coordinate = Coordinate::from_grid(
            if horizontal { origin.row() } else { next },
            if horizontal { next } else { origin.column() },
        )
        .ok()?;
        (!point_type_at(coordinate).is_gate()).then_some(coordinate)
    }
    pub(crate) fn relocate(&mut self, from: Coordinate, to: Coordinate) -> Option<Tile> {
        let moving = self
            .remove(from)
            .expect("relocate requires an occupied source");
        self.replace(to, Some(moving))
    }
}

/// Iterate the existing row masks without nested iterator state machines.
/// Empty permanent gate sentinels are filtered, preserving dense row-major order.
struct Occupied<'a> {
    board: &'a Board,
    row: usize,
    remaining: u32,
}
impl Iterator for Occupied<'_> {
    type Item = (Coordinate, Tile);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining != 0 {
                let column = self.remaining.trailing_zeros() as usize;
                self.remaining &= self.remaining - 1;
                if let Some(tile) = self.board.cells[self.row * BOARD_SIZE + column] {
                    let at = Coordinate::from_grid(self.row, column)
                        .expect("stored row mask is inside the board envelope");
                    return Some((at, tile));
                }
            } else {
                if self.row >= BOARD_SIZE - 1 {
                    return None;
                }
                self.row += 1;
                self.remaining = self.board.rows[self.row];
            }
        }
    }
}

impl PartialEq for Board {
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells
    }
}
impl Eq for Board {}

impl Default for Board {
    fn default() -> Self {
        Self::empty()
    }
}

const fn gate_masks() -> [u32; BOARD_SIZE] {
    let mut masks = [0; BOARD_SIZE];
    masks[0] = 1 << (BOARD_SIZE / 2);
    masks[BOARD_SIZE - 1] = 1 << (BOARD_SIZE / 2);
    masks[BOARD_SIZE / 2] = 1 | (1 << (BOARD_SIZE - 1));
    masks
}

const fn index(coordinate: Coordinate) -> usize {
    coordinate.row() * BOARD_SIZE + coordinate.column()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoardError {
    NonPlayable(Coordinate),
    Occupied(Coordinate),
}

impl fmt::Display for BoardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPlayable(coordinate) => {
                write!(formatter, "{coordinate} is not a playable intersection")
            }
            Self::Occupied(coordinate) => write!(formatter, "{coordinate} is occupied"),
        }
    }
}

impl std::error::Error for BoardError {}

#[cfg(test)]
mod occupied_tests {
    use super::*;
    #[test]
    fn harmony_cache_is_shared_only_until_board_mutation() {
        let mut original = Board::empty();
        let first = Coordinate::from_grid(8, 8).unwrap();
        let second = Coordinate::from_grid(8, 10).unwrap();
        original
            .place(
                first,
                Tile::new(
                    crate::Player::Host,
                    crate::TileKind::Basic(crate::BasicFlower::Red3),
                ),
            )
            .unwrap();
        original
            .place(
                second,
                Tile::new(
                    crate::Player::Host,
                    crate::TileKind::Basic(crate::BasicFlower::White4),
                ),
            )
            .unwrap();
        let before = crate::harmonies(&original);
        let mut child = original.clone();
        assert!(std::sync::Arc::ptr_eq(
            original.harmonies.get().unwrap(),
            child.harmonies.get().unwrap()
        ));
        child.remove(second);
        assert!(child.harmonies.get().is_none());
        assert!(crate::harmonies(&child).is_empty());
        assert_eq!(crate::harmonies(&original), before);
        assert!(original.harmonies.get().is_some());
    }
    #[test]
    fn sparse_iterator_preserves_dense_order_through_all_mutations_and_gates() {
        let mut board = Board::empty();
        let points: Vec<_> = crate::all_coordinates()
            .filter(|p| point_type_at(*p).is_playable())
            .collect();
        let check = |b: &Board| {
            let expected: Vec<_> = (0..CELL_COUNT)
                .filter_map(|i| {
                    b.cells[i].map(|t| {
                        (
                            Coordinate::from_grid(i / BOARD_SIZE, i % BOARD_SIZE).unwrap(),
                            t,
                        )
                    })
                })
                .collect();
            assert_eq!(b.occupied().collect::<Vec<_>>(), expected);
            assert_eq!(b.occupied_count(), expected.len());
        };
        check(&board);
        for (i, p) in points.iter().enumerate() {
            let tile = Tile::new(
                crate::Player::Host,
                crate::TileKind::Basic(crate::BasicFlower::Red3),
            );
            board.place(*p, tile).unwrap();
            if i % 13 == 0 {
                check(&board);
                check(&board.clone());
            }
        }
        check(&board);
        for p in points.iter().step_by(2) {
            board.remove(*p);
            check(&board);
        }
        for pair in points.windows(2) {
            if board.get(pair[0]).is_some() {
                board.relocate(pair[0], pair[1]);
                check(&board);
            }
        }
        for p in points {
            board.remove(p);
        }
        check(&board);
    }
}
