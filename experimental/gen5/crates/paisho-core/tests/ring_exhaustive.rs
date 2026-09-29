use paisho_core::{
    harmonies, harmony_ring_owners, Board, Coordinate, GameRecord, Player, Tile, TileKind,
};

// Independent reference: enumerate every simple undirected graph cycle,
// then use the total angle subtended by its edges at the origin.
// No spanning forest, disjoint set, or production winding-number code.
#[derive(Default)]
struct Exhaustive {
    cycles: usize,
    valid: Vec<Vec<Coordinate>>,
}
fn encloses(cycle: &[usize], coords: &[Coordinate]) -> bool {
    let mut angle = 0.0;
    for i in 0..cycle.len() {
        let a = coords[cycle[i]];
        let b = coords[cycle[(i + 1) % cycle.len()]];
        let (ax, ay, bx, by) = (a.x() as i32, a.y() as i32, b.x() as i32, b.y() as i32);
        let cross = ax * by - ay * bx;
        let dot = ax * bx + ay * by;
        // Origin lies on this segment iff collinear and endpoint rays oppose.
        if cross == 0 && dot <= 0 {
            return false;
        }
        angle += (cross as f64).atan2(dot as f64);
    }
    angle.abs() > std::f64::consts::PI
}
fn dfs(
    start: usize,
    coords: &[Coordinate],
    adj: &[Vec<usize>],
    path: &mut Vec<usize>,
    seen: &mut [bool],
    result: &mut Exhaustive,
) {
    let current = *path.last().unwrap();
    for &next in &adj[current] {
        if next == start && path.len() >= 3 && path[1] < current {
            result.cycles += 1;
            if encloses(path, coords) {
                result.valid.push(path.iter().map(|&v| coords[v]).collect());
            }
        } else if next > start && !seen[next] {
            seen[next] = true;
            path.push(next);
            dfs(start, coords, adj, path, seen, result);
            path.pop();
            seen[next] = false;
        }
    }
}
fn exhaustive(board: &Board, player: Player) -> Exhaustive {
    let edges: Vec<_> = harmonies(board)
        .into_iter()
        .filter(|h| h.owner == player)
        .collect();
    let mut coords: Vec<_> = edges.iter().flat_map(|h| [h.first, h.second]).collect();
    coords.sort();
    coords.dedup();
    let mut adj = vec![vec![]; coords.len()];
    for h in edges {
        let a = coords.binary_search(&h.first).unwrap();
        let b = coords.binary_search(&h.second).unwrap();
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut result = Exhaustive::default();
    for start in 0..coords.len() {
        let mut seen = vec![false; coords.len()];
        seen[start] = true;
        dfs(
            start,
            &coords,
            &adj,
            &mut vec![start],
            &mut seen,
            &mut result,
        );
    }
    result
}

fn expected(board: &Board) -> Vec<Player> {
    [Player::Host, Player::Guest]
        .into_iter()
        .filter(|p| !exhaustive(board, *p).valid.is_empty())
        .collect()
}

#[test]
fn corrected_rings_match_exhaustive_cycles_on_the_reported_trajectory() {
    let record: GameRecord = include_str!("fixtures/reported-ring-v1.psr")
        .parse()
        .unwrap();
    let mut position = record.initial_position();
    for (i, action) in record.actions().iter().enumerate() {
        position.apply(*action).unwrap();
        assert_eq!(
            harmony_ring_owners(position.board()),
            expected(position.board()),
            "decision {}",
            i + 1
        );
    }
}

#[test]
fn corrected_rings_are_complete_for_all_subsets_seats_and_symmetries() {
    let tiles = [
        (-3, -5, "R5"),
        (0, -5, "W3"),
        (2, -5, "R5"),
        (2, 6, "R4"),
        (0, 6, "L"),
        (0, 4, "R5"),
        (-3, 4, "R4"),
    ];
    for owner in [Player::Host, Player::Guest] {
        for reflected in [false, true] {
            for rotation in 0..4 {
                for mask in 0u8..128 {
                    let mut board = Board::empty();
                    for (i, (x, y, kind)) in tiles.iter().enumerate() {
                        if mask & (1 << i) == 0 {
                            continue;
                        }
                        let mut at = Coordinate::new(*x, *y).unwrap();
                        if reflected {
                            at = at.mirror_across_vertical_axis();
                        }
                        for _ in 0..rotation {
                            at = at.rotate_clockwise();
                        }
                        board
                            .place(at, Tile::new(owner, kind.parse::<TileKind>().unwrap()))
                            .unwrap();
                    }
                    let reference = expected(&board);
                    if mask == 127 {
                        assert_eq!(reference, vec![owner]);
                    }
                    assert_eq!(
                        harmony_ring_owners(&board),
                        reference,
                        "{owner:?} reflected={reflected} rotation={rotation} mask={mask}"
                    );
                }
            }
        }
    }
}
