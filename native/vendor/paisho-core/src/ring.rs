use crate::{
    harmonies, Board, Coordinate, Harmony, LineOrientation, Player, RuleProfileId, CELL_COUNT,
};

/// Returns every player whose own Harmony graph contains a cycle enclosing the
/// centre without a tile or Harmony segment touching the centre.
pub fn harmony_ring_owners(board: &Board) -> Vec<Player> {
    harmony_ring_owners_for_profile(board, RuleProfileId::CURRENT)
}

/// Profile-aware ring detection for immutable historical game replay.
pub fn harmony_ring_owners_for_profile(board: &Board, rules: RuleProfileId) -> Vec<Player> {
    let all_harmonies = harmonies(board);
    [Player::Host, Player::Guest]
        .into_iter()
        .filter(|player| player_has_enclosing_cycle(&all_harmonies, *player, rules))
        .collect()
}

pub fn midline_crossing_harmony_count(board: &Board, player: Player) -> usize {
    harmonies(board)
        .into_iter()
        .filter(|harmony| harmony.owner == player && harmony_crosses_midline(*harmony))
        .count()
}

pub fn harmony_crosses_midline(harmony: Harmony) -> bool {
    match harmony.orientation {
        LineOrientation::Horizontal => {
            harmony.first.x() < 0 && harmony.second.x() > 0 && harmony.first.y() != 0
        }
        LineOrientation::Vertical => {
            harmony.first.y() > 0 && harmony.second.y() < 0 && harmony.first.x() != 0
        }
    }
}

fn player_has_enclosing_cycle(harmonies: &[Harmony], player: Player, rules: RuleProfileId) -> bool {
    let mut tree: [Vec<Coordinate>; CELL_COUNT] = std::array::from_fn(|_| Vec::new());
    let mut disjoint_set = DisjointSet::new();

    for harmony in harmonies.iter().filter(|harmony| harmony.owner == player) {
        let first = harmony.first;
        let second = harmony.second;
        // Such an edge cannot belong to a winning ring. Including it in the
        // spanning forest can hide an enclosing outer cycle by splitting it
        // into fundamental cycles that all touch the centre. Keep the edge in
        // harmonies(), but exclude it before constructing the V2 cycle basis.
        if rules.profile().complete_harmony_ring_detection && segment_touches_centre(first, second)
        {
            continue;
        }
        if disjoint_set.find(first.dense_index()) != disjoint_set.find(second.dense_index()) {
            disjoint_set.union(first.dense_index(), second.dense_index());
            tree[first.dense_index()].push(second);
            tree[second.dense_index()].push(first);
        } else if let Some(path) = tree_path(&tree, first, second) {
            if cycle_encloses_centre(&path) {
                return true;
            }
        }
    }
    false
}

/// Finds the unique path in the current spanning forest, including both ends.
fn tree_path(
    tree: &[Vec<Coordinate>; CELL_COUNT],
    start: Coordinate,
    target: Coordinate,
) -> Option<Vec<Coordinate>> {
    let mut parent: [Option<Coordinate>; CELL_COUNT] = [None; CELL_COUNT];
    let mut visited = [false; CELL_COUNT];
    let mut queue = [start; CELL_COUNT];
    let mut head = 0;
    let mut tail = 1;
    visited[start.dense_index()] = true;

    while head < tail {
        let current = queue[head];
        head += 1;
        if current == target {
            break;
        }
        for neighbor in &tree[current.dense_index()] {
            if !visited[neighbor.dense_index()] {
                visited[neighbor.dense_index()] = true;
                parent[neighbor.dense_index()] = Some(current);
                queue[tail] = *neighbor;
                tail += 1;
            }
        }
    }
    if !visited[target.dense_index()] {
        return None;
    }

    let mut reversed = vec![target];
    let mut cursor = target;
    while cursor != start {
        cursor = parent[cursor.dense_index()]?;
        reversed.push(cursor);
    }
    reversed.reverse();
    Some(reversed)
}

fn cycle_encloses_centre(vertices: &[Coordinate]) -> bool {
    if vertices.len() < 4 {
        return false;
    }

    let mut winding_number = 0_i32;
    for index in 0..vertices.len() {
        let start = vertices[index];
        let end = vertices[(index + 1) % vertices.len()];
        if segment_touches_centre(start, end) {
            return false;
        }

        if start.y() <= 0 {
            if end.y() > 0 && is_left(start, end) > 0 {
                winding_number += 1;
            }
        } else if end.y() <= 0 && is_left(start, end) < 0 {
            winding_number -= 1;
        }
    }
    winding_number != 0
}

fn segment_touches_centre(start: Coordinate, end: Coordinate) -> bool {
    if (start.x() == 0 && start.y() == 0) || (end.x() == 0 && end.y() == 0) {
        return true;
    }
    (start.x() == 0 && end.x() == 0 && signs_span_zero(start.y(), end.y()))
        || (start.y() == 0 && end.y() == 0 && signs_span_zero(start.x(), end.x()))
}

const fn signs_span_zero(first: i8, second: i8) -> bool {
    (first < 0 && second > 0) || (first > 0 && second < 0)
}

const fn is_left(start: Coordinate, end: Coordinate) -> i32 {
    (end.x() as i32 - start.x() as i32) * -(start.y() as i32)
        - (-(start.x() as i32)) * (end.y() as i32 - start.y() as i32)
}

struct DisjointSet {
    parent: [usize; CELL_COUNT],
    rank: [u8; CELL_COUNT],
}

impl DisjointSet {
    fn new() -> Self {
        Self {
            parent: std::array::from_fn(|index| index),
            rank: [0; CELL_COUNT],
        }
    }

    fn find(&mut self, index: usize) -> usize {
        if self.parent[index] != index {
            self.parent[index] = self.find(self.parent[index]);
        }
        self.parent[index]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        match self.rank[left_root].cmp(&self.rank[right_root]) {
            core::cmp::Ordering::Less => self.parent[left_root] = right_root,
            core::cmp::Ordering::Greater => self.parent[right_root] = left_root,
            core::cmp::Ordering::Equal => {
                self.parent[right_root] = left_root;
                self.rank[left_root] += 1;
            }
        }
    }
}
