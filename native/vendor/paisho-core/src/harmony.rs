use crate::{point_type_at, Accent, Board, Coordinate, Player, Tile, TileKind};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LineOrientation {
    Horizontal,
    Vertical,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Harmony {
    pub first: Coordinate,
    pub second: Coordinate,
    pub owner: Player,
    pub orientation: LineOrientation,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Clash {
    pub first: Coordinate,
    pub second: Coordinate,
    pub orientation: LineOrientation,
}

/// Visit harmonies in canonical order, reusing the immutable exact-board cache.
/// The first query builds accent masks and the harmony list once per snapshot.
pub fn visit_harmonies(board: &Board, mut visitor: impl FnMut(Harmony)) {
    for harmony in board.cached_harmonies(|| {
        let mut result = Vec::new();
        visit_uncached_harmonies(board, |h| result.push(h));
        result
    }) {
        visitor(*harmony);
    }
}

fn visit_uncached_harmonies(board: &Board, mut visitor: impl FnMut(Harmony)) {
    let (mut rock_rows, mut rock_columns) = (0u32, 0u32);
    let mut suppressed = [0u32; crate::BOARD_SIZE];
    for (at, tile) in board.occupied() {
        if tile.kind == TileKind::Accent(Accent::Rock) {
            rock_rows |= 1 << at.row();
            rock_columns |= 1 << at.column();
        }
        if tile.kind == TileKind::Accent(Accent::Knotweed) {
            for near in crate::surrounding_neighbors(at) {
                suppressed[near.row()] |= 1 << near.column();
            }
        }
    }
    visit_visible_pairs(board, |pair| {
        let Some(owner) = pair.first_tile.harmony_owner_with(pair.second_tile) else {
            return false;
        };
        let rock = match pair.orientation {
            LineOrientation::Horizontal => rock_rows & (1 << pair.first.row()) != 0,
            LineOrientation::Vertical => rock_columns & (1 << pair.first.column()) != 0,
        };
        if !rock
            && suppressed[pair.first.row()] & (1 << pair.first.column()) == 0
            && suppressed[pair.second.row()] & (1 << pair.second.column()) == 0
        {
            visitor(Harmony {
                first: pair.first,
                second: pair.second,
                owner,
                orientation: pair.orientation,
            });
        }
        false
    });
}
pub fn harmonies(board: &Board) -> Vec<Harmony> {
    let mut result = Vec::new();
    visit_harmonies(board, |h| result.push(h));
    result
}

pub fn clashes(board: &Board) -> Vec<Clash> {
    visible_pairs(board)
        .into_iter()
        .filter(|pair| pair.first_tile.clashes_with(pair.second_tile))
        .map(|pair| Clash {
            first: pair.first,
            second: pair.second,
            orientation: pair.orientation,
        })
        .collect()
}

pub fn has_clash(board: &Board) -> bool {
    board.cached_clash(|| {
        visit_visible_pairs(board, |pair| pair.first_tile.clashes_with(pair.second_tile))
    })
}

/// Checks only the visible pairs whose identity can change after a relocation.
/// The caller must establish that `board` itself contains no Clash.
pub(crate) fn relocation_creates_clash(
    board: &Board,
    from: Coordinate,
    to: Coordinate,
    moving: Tile,
) -> bool {
    for (delta_x, delta_y) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        if nearest_tile_after_relocation(board, to, delta_x, delta_y, from, to, moving)
            .is_some_and(|other| moving.clashes_with(other))
        {
            return true;
        }
    }

    if point_type_at(from).is_gate() {
        return false;
    }
    for ((first_x, first_y), (second_x, second_y)) in [((1, 0), (-1, 0)), ((0, 1), (0, -1))] {
        let first = nearest_tile_after_relocation(board, from, first_x, first_y, from, to, moving);
        let second =
            nearest_tile_after_relocation(board, from, second_x, second_y, from, to, moving);
        if first
            .zip(second)
            .is_some_and(|(left, right)| left.clashes_with(right))
        {
            return true;
        }
    }
    false
}

#[derive(Clone, Copy)]
struct VisiblePair {
    first: Coordinate,
    second: Coordinate,
    first_tile: Tile,
    second_tile: Tile,
    orientation: LineOrientation,
}

fn visible_pairs(board: &Board) -> Vec<VisiblePair> {
    let mut pairs = Vec::with_capacity(board.occupied_count().saturating_mul(2));
    let stopped = visit_visible_pairs(board, |pair| {
        pairs.push(pair);
        false
    });
    debug_assert!(!stopped);
    pairs
}

fn visit_visible_pairs(board: &Board, mut visitor: impl FnMut(VisiblePair) -> bool) -> bool {
    for (first, first_tile) in board.occupied() {
        if point_type_at(first).is_gate() {
            continue;
        }

        for (delta_x, delta_y, orientation) in [
            (1, 0, LineOrientation::Horizontal),
            (0, -1, LineOrientation::Vertical),
        ] {
            if let Some(second) = Board::nearest_on_line(
                first,
                delta_x,
                delta_y,
                board.line_mask(first, delta_x != 0),
            ) {
                if visitor(VisiblePair {
                    first,
                    second,
                    first_tile,
                    second_tile: board.get(second).unwrap(),
                    orientation,
                }) {
                    return true;
                }
            }
        }
    }
    false
}

fn nearest_tile_after_relocation(
    board: &Board,
    origin: Coordinate,
    delta_x: i8,
    delta_y: i8,
    from: Coordinate,
    to: Coordinate,
    moving: Tile,
) -> Option<Tile> {
    let horizontal = delta_x != 0;
    let mut mask = board.line_mask(origin, horizontal);
    let same_line = |p: Coordinate| {
        if horizontal {
            p.row() == origin.row()
        } else {
            p.column() == origin.column()
        }
    };
    let bit = |p: Coordinate| 1u32 << if horizontal { p.column() } else { p.row() };
    if same_line(from) && !point_type_at(from).is_gate() {
        mask &= !bit(from);
    }
    if same_line(to) && !point_type_at(to).is_gate() {
        mask |= bit(to);
    }
    let at = Board::nearest_on_line(origin, delta_x, delta_y, mask)?;
    if at == to {
        Some(moving)
    } else {
        board.get(at)
    }
}

#[cfg(test)]
fn line_has_rock(board: &Board, coordinate: Coordinate, orientation: LineOrientation) -> bool {
    board.occupied().any(|(other, tile)| {
        tile.kind == TileKind::Accent(Accent::Rock)
            && match orientation {
                LineOrientation::Horizontal => other.y() == coordinate.y(),
                LineOrientation::Vertical => other.x() == coordinate.x(),
            }
    })
}

#[cfg(test)]
fn is_suppressed_by_knotweed(board: &Board, coordinate: Coordinate) -> bool {
    crate::surrounding_neighbors(coordinate).any(|neighbor| {
        board
            .get(neighbor)
            .is_some_and(|tile| tile.kind == TileKind::Accent(Accent::Knotweed))
    })
}

#[cfg(test)]
mod visibility_cache_tests {
    use super::*;
    fn reference(board: &Board, origin: Coordinate, dx: i8, dy: i8) -> Option<(Coordinate, Tile)> {
        let mut at = origin;
        while let Some(next) = at.translated(dx, dy) {
            at = next;
            if !point_type_at(at).is_playable() || point_type_at(at).is_gate() {
                return None;
            }
            if let Some(tile) = board.get(at) {
                return Some((at, tile));
            }
        }
        None
    }
    fn check(board: &Board) {
        let expected: Vec<_> = visible_pairs(board)
            .into_iter()
            .filter_map(|pair| {
                let owner = pair.first_tile.harmony_owner_with(pair.second_tile)?;
                if line_has_rock(board, pair.first, pair.orientation)
                    || is_suppressed_by_knotweed(board, pair.first)
                    || is_suppressed_by_knotweed(board, pair.second)
                {
                    return None;
                }
                Some(Harmony {
                    first: pair.first,
                    second: pair.second,
                    owner,
                    orientation: pair.orientation,
                })
            })
            .collect();
        assert_eq!(
            harmonies(board),
            expected,
            "streamed harmonies must preserve identities and order"
        );
        let mut clash = false;
        for (at, tile) in board.occupied() {
            for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let old = reference(board, at, dx, dy);
                let new = Board::nearest_on_line(at, dx, dy, board.line_mask(at, dx != 0))
                    .map(|p| (p, board.get(p).unwrap()));
                assert_eq!(new, old, "visibility at {at} direction {dx},{dy}");
                if !point_type_at(at).is_gate() {
                    clash |= old.is_some_and(|(_, other)| tile.clashes_with(other));
                }
            }
        }
        assert_eq!(has_clash(board), clash);
        assert_eq!(has_clash(board), clash); // exercise cached result
    }
    #[test]
    fn masks_and_clash_cache_match_reference_after_mutation_and_clone() {
        let points: Vec<_> = crate::all_coordinates()
            .filter(|p| point_type_at(*p).is_playable())
            .collect();
        let flowers = [
            crate::BasicFlower::Red3,
            crate::BasicFlower::Red4,
            crate::BasicFlower::White3,
            crate::BasicFlower::White4,
        ];
        let mut rng = 91u64;
        for _ in 0..128 {
            let mut board = Board::empty();
            for i in 0..48 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let at = points[(rng >> 32) as usize % points.len()];
                board.remove(at);
                let tile = Tile::new(
                    if i % 2 == 0 {
                        Player::Host
                    } else {
                        Player::Guest
                    },
                    match i % 9 {
                        0 => TileKind::Accent(Accent::Rock),
                        1 => TileKind::Accent(Accent::Knotweed),
                        2 => TileKind::WhiteLotus,
                        _ => TileKind::Basic(flowers[i % 4]),
                    },
                );
                board.place(at, tile).unwrap();
                if i % 8 == 0 {
                    check(&board);
                }
            }
            check(&board);
            let mut cloned = board.clone();
            for (from, tile) in board.occupied().take(8) {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let to = points[(rng >> 32) as usize % points.len()];
                if from == to {
                    continue;
                }
                for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    cloned = board.clone();
                    cloned.relocate(from, to);
                    assert_eq!(
                        nearest_tile_after_relocation(&board, to, dx, dy, from, to, tile),
                        reference(&cloned, to, dx, dy).map(|(_, t)| t)
                    );
                }
                check(&cloned);
            }
            assert_eq!(board, board.clone());
        }
    }
}
