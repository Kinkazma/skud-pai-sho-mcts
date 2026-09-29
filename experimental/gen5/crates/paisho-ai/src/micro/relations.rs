//! Explicit rule-derived relations for isolated Gen5 representation experiments.
//! No production schema or forward path is changed. Input facts describe only
//! the current position; successor facts are separate supervised targets.
use super::MICRO_ACTION_INPUTS;
use paisho_core::{
    harmonies, harmony_crosses_midline, harmony_ring_owners_for_profile, Action,
    Board, Coordinate, GameOutcome, Harmony, Player, Position, TurnPhase, CELL_COUNT,
};
use std::collections::BTreeSet;

pub const MICRO_RELATION_SCHEMA: &str = "paisho-gen5-relations-v1";
pub const MICRO_RELATION_GLOBAL: usize = 20;
pub const MICRO_RELATION_EDGE: usize = 43;
pub const MICRO_RELATION_TARGETS: usize = 12;
pub const MICRO_RELATION_TARGET_NAMES: [&str; 12] = [
    "own_harmonies_created", "own_harmonies_removed", "opponent_harmonies_created",
    "opponent_harmonies_removed", "own_midline_delta", "opponent_midline_delta",
    "own_cycle_rank_delta", "opponent_cycle_rank_delta", "own_ring_after",
    "opponent_ring_after", "basic_exhaustion_ending", "other_terminal_ending",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HarmonyCycleGeometry { EnclosingCentre, OffCentre, TouchingCentre }

/// A fundamental-cycle witness, not an enumeration/count of every simple cycle.
#[derive(Clone, Debug)]
pub struct HarmonyCycleWitness {
    pub owner: Player,
    pub vertices: Vec<Coordinate>,
    pub geometry: HarmonyCycleGeometry,
}

#[derive(Clone, Debug)]
pub struct MicroRelations {
    pub perspective: Player,
    pub edges: Vec<Harmony>,
    pub cycles: Vec<HarmonyCycleWitness>,
    /// Per own/opponent: edges/32, midline/16, cycle rank/16, enclosing flag,
    /// off-centre basis witnesses/16, touching witnesses/16, basic reserve/54,
    /// horizontal/32, vertical/32. Then main/bonus flags.
    pub global: [f64; MICRO_RELATION_GLOBAL],
    board: Board,
}

fn touches(a: Coordinate, b: Coordinate) -> bool {
    let spans = |a: i8, b: i8| a.min(b) <= 0 && a.max(b) >= 0;
    (a.x() == 0 && b.x() == 0 && spans(a.y(), b.y()))
        || (a.y() == 0 && b.y() == 0 && spans(a.x(), b.x()))
}

fn geometry(path: &[Coordinate]) -> HarmonyCycleGeometry {
    let mut winding = 0i32;
    for (&a, &b) in path.iter().zip(path.iter().cycle().skip(1)).take(path.len()) {
        if touches(a, b) { return HarmonyCycleGeometry::TouchingCentre; }
        let left = (b.x() as i32 - a.x() as i32) * -(a.y() as i32)
            + a.x() as i32 * (b.y() as i32 - a.y() as i32);
        if a.y() <= 0 && b.y() > 0 && left > 0 { winding += 1; }
        if a.y() > 0 && b.y() <= 0 && left < 0 { winding -= 1; }
    }
    if path.len() >= 4 && winding != 0 { HarmonyCycleGeometry::EnclosingCentre }
    else { HarmonyCycleGeometry::OffCentre }
}

fn path(tree: &[Vec<Coordinate>], from: Coordinate, to: Coordinate) -> Option<Vec<Coordinate>> {
    let mut parent = [None; CELL_COUNT];
    let mut seen = [false; CELL_COUNT];
    let mut queue = vec![from];
    seen[from.dense_index()] = true;
    let mut i = 0;
    while i < queue.len() {
        let p = queue[i]; i += 1;
        if p == to {
            let mut result = vec![to]; let mut p = to;
            while p != from { p = parent[p.dense_index()]?; result.push(p); }
            result.reverse(); return Some(result);
        }
        for &next in &tree[p.dense_index()] {
            if !seen[next.dense_index()] {
                seen[next.dense_index()] = true;
                parent[next.dense_index()] = Some(p); queue.push(next);
            }
        }
    }
    None
}

fn edge_key(a: Coordinate, b: Coordinate) -> (Coordinate, Coordinate) {
    if a < b { (a,b) } else { (b,a) }
}
fn cycle_edges(v: &[Coordinate]) -> BTreeSet<(Coordinate, Coordinate)> {
    v.iter().zip(v.iter().cycle().skip(1)).take(v.len())
        .map(|(&a,&b)|edge_key(a,b)).collect()
}

fn witnesses(edges: &[Harmony], owner: Player) -> (usize, Vec<HarmonyCycleWitness>) {
    let mut seen = BTreeSet::new(); let mut result = vec![]; let mut rank = 0;
    // A centre-touching chord can hide an enclosing outer cycle in the full
    // graph's basis. Also inspect the basis after excluding such edges, exactly
    // the filtering used by the current rule detector. Keep original edge IDs.
    for exclude_centre in [false, true] {
        let mut tree: Vec<Vec<Coordinate>> = vec![vec![]; CELL_COUNT];
        for h in edges.iter().filter(|h|h.owner == owner) {
            if exclude_centre && touches(h.first,h.second) { continue; }
            if let Some(vertices) = path(&tree,h.first,h.second) {
                if !exclude_centre { rank += 1; }
                if seen.insert(cycle_edges(&vertices)) {
                    result.push(HarmonyCycleWitness { owner, geometry: geometry(&vertices), vertices });
                }
            } else {
                tree[h.first.dense_index()].push(h.second);
                tree[h.second.dense_index()].push(h.first);
            }
        }
    }
    (rank,result)
}

impl MicroRelations {
    pub fn extract(p: &Position, perspective: Player) -> Self {
        Self::from_board(p.board(), perspective, p.rule_profile(),
            [p.reserve(perspective).basic_count(),p.reserve(perspective.opponent()).basic_count()],p.phase())
    }
    fn from_board(board: &Board, perspective: Player, rules: paisho_core::RuleProfileId,
        reserves: [u8;2], phase: TurnPhase) -> Self {
        let edges = harmonies(board);
        let rings = harmony_ring_owners_for_profile(board,rules);
        let mut global = [0.;MICRO_RELATION_GLOBAL]; let mut cycles = vec![];
        for (seat,owner) in [perspective,perspective.opponent()].into_iter().enumerate() {
            let (rank, found) = witnesses(&edges,owner); let offset = seat*9;
            global[offset] = edges.iter().filter(|h|h.owner==owner).count() as f64/32.;
            global[offset+1] = edges.iter().filter(|h|h.owner==owner && harmony_crosses_midline(**h)).count() as f64/16.;
            global[offset+2] = rank as f64/16.;
            global[offset+3] = f64::from(rings.contains(&owner));
            global[offset+4] = found.iter().filter(|c|c.geometry==HarmonyCycleGeometry::OffCentre).count() as f64/16.;
            global[offset+5] = found.iter().filter(|c|c.geometry==HarmonyCycleGeometry::TouchingCentre).count() as f64/16.;
            global[offset+6] = f64::from(reserves[seat])/54.;
            global[offset+7] = edges.iter().filter(|h|h.owner==owner && h.first.y()==h.second.y()).count() as f64/32.;
            global[offset+8] = edges.iter().filter(|h|h.owner==owner && h.first.x()==h.second.x()).count() as f64/32.;
            cycles.extend(found);
        }
        global[18] = f64::from(phase==TurnPhase::Main);
        global[19] = f64::from(phase==TurnPhase::HarmonyBonus);
        Self { perspective, edges, cycles, global, board: board.clone() }
    }

    /// One token per exact harmony pair. Endpoint identities are categorical;
    /// endpoint coordinates and offsets relative to the proposed action remain
    /// explicit. No successor, teacher estimate or result is an input.
    pub fn action_tokens(&self, action: &[f64;MICRO_ACTION_INPUTS]) -> Vec<[f64;MICRO_RELATION_EDGE]> {
        let sets: Vec<_> = self.cycles.iter().map(|c|(c.geometry,cycle_edges(&c.vertices))).collect();
        self.edges.iter().map(|h| {
            let mut x = [0.;MICRO_RELATION_EDGE];
            x[0] = if h.owner==self.perspective {1.} else {-1.};
            x[1]=f64::from(h.first.x())/8.; x[2]=f64::from(h.first.y())/8.;
            x[3]=f64::from(h.second.x())/8.; x[4]=f64::from(h.second.y())/8.;
            x[5+self.board.get(h.first).expect("harmony endpoint").kind.index()]=1.;
            x[17+self.board.get(h.second).expect("harmony endpoint").kind.index()]=1.;
            x[29]=f64::from(h.first.y()==h.second.y());
            x[30]=f64::from(harmony_crosses_midline(*h));
            x[31]=f64::from(touches(h.first,h.second));
            let key=edge_key(h.first,h.second);
            for (kind,set) in &sets {
                if set.contains(&key) {
                    x[match kind {HarmonyCycleGeometry::EnclosingCentre=>32,HarmonyCycleGeometry::OffCentre=>33,HarmonyCycleGeometry::TouchingCentre=>34}]=1.;
                }
            }
            for i in 0..4 { x[35+i]=x[1+i]-action[18+i%2]; x[39+i]=x[1+i]-action[20+i%2]; }
            x
        }).collect()
    }
}

/// Facts after a specified legal action, reserved for supervised outputs.
pub fn micro_relation_targets(p: &Position, action: Action) -> Result<[f64;MICRO_RELATION_TARGETS],String> {
    let before=MicroRelations::extract(p,p.to_move());
    let mut next=p.clone(); next.apply(action).map_err(|e|e.to_string())?;
    let after=MicroRelations::extract(&next,p.to_move());
    let mut y=[0.;MICRO_RELATION_TARGETS];
    for (seat,owner) in [p.to_move(),p.to_move().opponent()].into_iter().enumerate() {
        y[2*seat]=after.edges.iter().filter(|h|h.owner==owner && !before.edges.contains(h)).count() as f64/8.;
        y[2*seat+1]=before.edges.iter().filter(|h|h.owner==owner && !after.edges.contains(h)).count() as f64/8.;
        y[4+seat]=2.*(after.global[9*seat+1]-before.global[9*seat+1]);
        y[6+seat]=2.*(after.global[9*seat+2]-before.global[9*seat+2]);
        y[8+seat]=after.global[9*seat+3];
    }
    let exhaustion=matches!(action,Action::Plant{..}|Action::BonusPlantBasic{..})
        && next.reserve(p.to_move()).basic_count()==0;
    y[10]=f64::from(exhaustion);
    y[11]=f64::from(next.outcome()!=GameOutcome::Ongoing && !exhaustion && y[8]==0. && y[9]==0.);
    Ok(y)
}

#[cfg(test)]
#[path="relations_tests.rs"]
mod tests;
