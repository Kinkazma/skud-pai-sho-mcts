//! Exercise the actual V5 input function on legal, cross-owner lotus positions.
//! The production compact graph counts only edge endpoints, unlike the R1
//! component target, which also includes isolated pieces. Both partition edges
//! by harmony owner and preserve the actual ownership of each piece.
use paisho_ai::{micro_spatial_state_features, COMPACT_FEATURE_SCALES, MICRO_INPUTS};
use paisho_core::*;
use std::collections::{BTreeMap, BTreeSet};

fn graph_counts(p: &Position, owner: Player) -> [usize; 6] {
    let mut adjacent: BTreeMap<Coordinate, Vec<Coordinate>> = BTreeMap::new();
    let edges = harmonies(p.board());
    let mut edge_count = 0;
    for h in edges.iter().filter(|h| h.owner == owner) {
        adjacent.entry(h.first).or_default().push(h.second);
        adjacent.entry(h.second).or_default().push(h.first);
        edge_count += 1;
    }
    let mut seen = BTreeSet::new();
    let mut sizes = vec![];
    for &first in adjacent.keys() {
        if !seen.insert(first) {
            continue;
        }
        let mut pending = vec![first];
        let mut size = 0;
        while let Some(at) = pending.pop() {
            size += 1;
            for &next in &adjacent[&at] {
                if seen.insert(next) {
                    pending.push(next);
                }
            }
        }
        sizes.push(size);
    }
    [
        adjacent.len(),
        adjacent.values().filter(|v| v.len() >= 2).count(),
        adjacent.values().filter(|v| v.len() >= 3).count(),
        sizes.iter().copied().max().unwrap_or(0),
        sizes.len(),
        edge_count + sizes.len() - adjacent.len(),
    ]
}

#[test]
fn v5_inputs_keep_harmony_owner_separate_from_lotus_owner() {
    for text in [
        include_str!("fixtures/cross_owner_lotus.psr"),
        include_str!("fixtures/owner_separated_components.psr"),
    ] {
        let r: GameRecord = text.parse().unwrap();
        let p = r.replay().unwrap();
        assert_eq!(p.outcome(), GameOutcome::Ongoing);
        let x = micro_spatial_state_features(&p);
        assert_eq!(x.len(), 417);
        let own = graph_counts(&p, p.to_move());
        let other = graph_counts(&p, p.to_move().opponent());
        for i in 0..6 {
            assert_eq!(
                x[39 + i],
                (own[i] as f64 - other[i] as f64) / COMPACT_FEATURE_SCALES[39 + i]
            );
        }
        let mut borrowed = 0;
        for h in harmonies(p.board()) {
            for at in [h.first, h.second] {
                let tile = p.board().get(at).unwrap();
                if tile.owner != h.owner {
                    borrowed += 1;
                    assert_eq!(tile.kind, TileKind::WhiteLotus);
                    let cell = (at.y() + 8) as usize * 17 + (at.x() + 8) as usize;
                    let sign = if tile.owner == p.to_move() { 1.0 } else { -1.0 };
                    assert_eq!(
                        x[MICRO_INPUTS + cell],
                        sign * (tile.kind.index() + 1) as f64 / 12.0
                    );
                }
            }
        }
        assert!(borrowed > 0);
    }
}
