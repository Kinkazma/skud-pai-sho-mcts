use crate::{all_coordinates, orthogonal_neighbors, Board, Coordinate, CELL_COUNT};

const UNREACHABLE: u8 = u8::MAX;

/// Length of the shortest orthogonal path through playable, unoccupied
/// intermediate points. The destination itself may be occupied.
pub fn clear_path_distance(
    board: &Board,
    from: Coordinate,
    to: Coordinate,
    maximum: u8,
) -> Option<u8> {
    if from == to || board.get(from).is_none() {
        return None;
    }
    let distance = clear_path_distances(board, from, maximum)[to.dense_index()];
    (distance != UNREACHABLE).then_some(distance)
}

/// Every physically reachable destination, before capture, Garden, Gate,
/// Orchid, Knotweed and Clash legality filters are applied.
pub fn reachable_points(board: &Board, from: Coordinate, maximum: u8) -> Vec<Coordinate> {
    if board.get(from).is_none() {
        return Vec::new();
    }
    let distances = clear_path_distances(board, from, maximum);
    all_coordinates()
        .filter(|coordinate| {
            let distance = distances[coordinate.dense_index()];
            distance != UNREACHABLE && distance > 0
        })
        .collect()
}

fn clear_path_distances(board: &Board, from: Coordinate, maximum: u8) -> [u8; CELL_COUNT] {
    let mut distances = [UNREACHABLE; CELL_COUNT];
    let mut queue = [from; CELL_COUNT];
    let mut head = 0;
    let mut tail = 1;
    distances[from.dense_index()] = 0;

    while head < tail {
        let current = queue[head];
        head += 1;
        let next_distance = distances[current.dense_index()] + 1;
        if next_distance > maximum {
            continue;
        }

        for neighbor in orthogonal_neighbors(current) {
            let index = neighbor.dense_index();
            if distances[index] != UNREACHABLE {
                continue;
            }
            distances[index] = next_distance;

            // Occupied points can be capture destinations, never intermediate.
            if board.get(neighbor).is_none() {
                queue[tail] = neighbor;
                tail += 1;
            }
        }
    }

    distances
}
