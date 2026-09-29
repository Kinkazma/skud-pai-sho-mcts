use paisho_ai::{HarmonyCycleGeometry, MicroRelations};
use paisho_core::*;
use serde_json::{json, Value};
use std::collections::BTreeSet;

pub struct Facts {
    pub rel: MicroRelations,
    pub rings: Vec<Player>,
    pub score: [usize; 2],
}
// Game-semantic identity, excluding mutable caches and the descriptive turn
// counter. Equivalent positions reached at different plies must not leak splits.
pub fn position_hash(p: &Position) -> String {
    let pieces: Vec<_> = p
        .board()
        .occupied()
        .map(|(at, t)| (at.dense_index(), t.owner.index(), t.kind.index()))
        .collect();
    let reserves =
        [Player::Host, Player::Guest].map(|s| STANDARD_TILE_KINDS.map(|k| p.reserve(s).count(k)));
    super::hash(
        format!(
            "{:?}|{:?}|{:?}|{:?}|{pieces:?}|{reserves:?}",
            p.rule_profile(),
            p.to_move(),
            p.phase(),
            p.outcome()
        )
        .as_bytes(),
    )
}
// Conservative leakage audit only, never a generated/assumed-legal replay.
// These four maps preserve garden colours; 90-degree rotations do not.
pub fn symmetry_hash(p: &Position) -> String {
    let mut variants = vec![];
    for transform in 0..4 {
        for flip in 0..2usize {
            let seat = |s: Player| s.index() ^ flip;
            let mut pieces: Vec<_> = p
                .board()
                .occupied()
                .map(|(at, t)| {
                    let (x, y) = transform_xy(at.x(), at.y(), transform);
                    (x, y, seat(t.owner), t.kind.index())
                })
                .collect();
            pieces.sort();
            let mut reserves = [[0u8; TileKind::COUNT]; 2];
            for s in [Player::Host, Player::Guest] {
                reserves[seat(s)] = STANDARD_TILE_KINDS.map(|k| p.reserve(s).count(k));
            }
            let outcome = match p.outcome() {
                GameOutcome::Ongoing => 0,
                GameOutcome::Draw => 1,
                GameOutcome::Win(s) => 2 + seat(s),
            };
            variants.push(format!(
                "{:?}|{}|{:?}|{outcome}|{pieces:?}|{reserves:?}",
                p.rule_profile(),
                seat(p.to_move()),
                p.phase()
            ));
        }
    }
    super::hash(variants.iter().min().unwrap().as_bytes())
}
fn transform_xy(x: i8, y: i8, t: usize) -> (i8, i8) {
    match t {
        0 => (x, y),
        1 => (-x, -y),
        2 => (y, x),
        _ => (-y, -x),
    }
}
impl Facts {
    pub fn new(p: &Position) -> Self {
        Self {
            rel: MicroRelations::extract(p, Player::Host),
            rings: harmony_ring_owners_for_profile(p.board(), p.rule_profile()),
            score: [Player::Host, Player::Guest]
                .map(|s| midline_crossing_harmony_count(p.board(), s)),
        }
    }
    pub fn geometry_tags(&self) -> Vec<String> {
        self.rel
            .cycles
            .iter()
            .map(|c| {
                match c.geometry {
                    HarmonyCycleGeometry::EnclosingCentre => "cycle_enclosing",
                    HarmonyCycleGeometry::OffCentre => "cycle_off_centre",
                    HarmonyCycleGeometry::TouchingCentre => "cycle_touching",
                }
                .to_owned()
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
    pub fn json(&self, p: &Position) -> Value {
        let pieces: Vec<_> = p
            .board()
            .occupied()
            .map(|(at, t)| {
                json!({"at":[at.x(),at.y()],
            "owner":t.owner.code().to_string(),"kind":t.kind.code()})
            })
            .collect();
        json!({"to_move":p.to_move().code().to_string(),"phase":format!("{:?}",p.phase()),
            "outcome":format!("{:?}",p.outcome()),"midline_score_host_guest":self.score,
            "basic_reserves_host_guest":[p.reserve(Player::Host).basic_count(),p.reserve(Player::Guest).basic_count()],
            "pieces":pieces,"components_host_guest":self.components(p),
            "ring_owners":self.rings.iter().map(|s|s.code().to_string()).collect::<Vec<_>>(),
            "edges":self.rel.edges.iter().map(|h|json!({"owner":h.owner.code().to_string(),
                "first":[h.first.x(),h.first.y()],"second":[h.second.x(),h.second.y()],
                "midline":harmony_crosses_midline(*h)})).collect::<Vec<_>>(),
            "cycle_witnesses":self.rel.cycles.iter().map(|c|json!({"owner":c.owner.code().to_string(),
                "geometry":format!("{:?}",c.geometry),
                "vertices":c.vertices.iter().map(|v|[v.x(),v.y()]).collect::<Vec<_>>()})).collect::<Vec<_>>()})
    }
    // Per-owner edge graph: all of that owner's pieces (even isolated) plus
    // endpoints of its harmonies, which may include the opponent's white lotus.
    // Never merge paths through edges owned by the other player.
    pub fn components(&self, p: &Position) -> [usize; 2] {
        self.owner_graphs(p).map(|g| g[2])
    }
    // [vertices, edges, components, cycle rank], Host then Guest.
    pub fn owner_graphs(&self, p: &Position) -> [[usize; 4]; 2] {
        fn root(p: &[usize], mut i: usize) -> usize {
            while p[i] != i {
                i = p[i];
            }
            i
        }
        [Player::Host, Player::Guest].map(|s| {
            let mut parent: Vec<_> = (0..CELL_COUNT).collect();
            let mut vertices: BTreeSet<_> = p
                .board()
                .occupied()
                .filter(|(_, t)| t.owner == s)
                .map(|(at, _)| at.dense_index())
                .collect();
            let mut edges = 0;
            for h in self.rel.edges.iter().filter(|h| h.owner == s) {
                let a = h.first.dense_index();
                let b = h.second.dense_index();
                vertices.extend([a, b]);
                let ra = root(&parent, a);
                let rb = root(&parent, b);
                parent[ra] = rb;
                edges += 1;
            }
            let components = vertices
                .iter()
                .map(|&i| root(&parent, i))
                .collect::<BTreeSet<_>>()
                .len();
            [
                vertices.len(),
                edges,
                components,
                edges + components - vertices.len(),
            ]
        })
    }
}
pub fn result(o: GameOutcome, player: Player) -> &'static str {
    match o {
        GameOutcome::Ongoing => "ongoing",
        GameOutcome::Draw => "draw",
        GameOutcome::Win(p) if p == player => "win",
        GameOutcome::Win(_) => "loss",
    }
}
pub fn ending(before: &Position, action: Action, after: &Position) -> &'static str {
    if after.outcome() == GameOutcome::Ongoing {
        return "ongoing";
    }
    // Match the engine's transition priority, not an assumed loser by reserves.
    if matches!(
        action,
        Action::Plant { .. } | Action::BonusPlantBasic { .. }
    ) && before.reserve(before.to_move()).basic_count() == 1
        && after.reserve(before.to_move()).basic_count() == 0
    {
        return "exhaustion";
    }
    if !harmony_ring_owners_for_profile(after.board(), after.rule_profile()).is_empty() {
        "ring"
    } else {
        "other_terminal"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cross_owner_lotus_belongs_to_harmony_graph_vertices() {
        let r: GameRecord = include_str!("fixtures/cross_owner_lotus.psr")
            .parse()
            .unwrap();
        let p = r.replay().unwrap();
        let f = Facts::new(&p);
        let g = f.owner_graphs(&p)[0];
        assert_eq!(g, [11, 5, 6, 0]);
        assert_eq!(
            p.board()
                .occupied()
                .filter(|(_, t)| t.owner == Player::Host)
                .count(),
            10
        );
        assert!(f.rel.edges.iter().any(|h| h.owner == Player::Host
            && [h.first, h.second]
                .iter()
                .any(|&at| p.board().get(at).unwrap().owner == Player::Guest)));
    }
    #[test]
    fn leakage_maps_preserve_every_topological_point_type() {
        for at in all_coordinates() {
            for t in 0..4 {
                let (x, y) = transform_xy(at.x(), at.y(), t);
                assert_eq!(
                    point_type_at(at),
                    point_type_at(Coordinate::new(x, y).unwrap())
                );
            }
        }
    }
    #[test]
    fn opponent_edges_do_not_merge_our_components() {
        let r: GameRecord = include_str!("fixtures/owner_separated_components.psr")
            .parse()
            .unwrap();
        let p = r.replay().unwrap();
        let f = Facts::new(&p);
        assert_eq!(f.components(&p), [11, 7]);
        for (s, g) in [Player::Host, Player::Guest]
            .into_iter()
            .zip(f.owner_graphs(&p))
        {
            assert_eq!(g[3], f.rel.cycles.iter().filter(|c| c.owner == s).count());
        }
    }
}
