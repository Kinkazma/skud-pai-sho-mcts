//! Current-state graph for the optional neutral V7 branch and R2 experiments.
//! No successors, proof labels, cycle classification or model weights are read.
use paisho_core::{
    harmony_crosses_midline, visit_harmonies, Board, Coordinate, LineOrientation, Player, Position,
    CELL_COUNT,
};

pub const MICRO_PIECE_GRAPH_SCHEMA: &str = "paisho-micro-piece-graph-16-6-v1";
pub const MICRO_PIECE_INPUTS: usize = 16;
pub const MICRO_HARMONY_INPUTS: usize = 6;

#[derive(Clone, Debug)]
pub struct MicroPieceNode {
    pub at: Coordinate,
    pub owner: Player,
    pub features: [f64; MICRO_PIECE_INPUTS],
}

#[derive(Clone, Debug)]
pub struct MicroHarmonyMessage {
    pub source: usize,
    pub destination: usize,
    pub features: [f64; MICRO_HARMONY_INPUTS],
}

#[derive(Clone, Debug)]
pub struct MicroPieceGraph {
    pub perspective: Player,
    pub nodes: Vec<MicroPieceNode>,
    pub messages: Vec<MicroHarmonyMessage>,
}

impl MicroPieceGraph {
    pub fn extract(position: &Position, perspective: Player) -> Self {
        Self::from_board(position.board(), perspective)
    }
    /// Exact categorical reconstruction from the existing 417-input schema.
    /// Uses only the board prefix; no future target or history is available here.
    pub fn from_spatial_features(state: &[f64]) -> Result<Self, String> {
        if state.len()!=super::MICRO_SPATIAL_INPUTS {return Err("piece graph requires spatial inputs".into());}
        let mut board=Board::empty();
        for (i,&value) in state[super::MICRO_INPUTS..].iter().enumerate() {
            if value==0. {continue;}
            let code=(value.abs()*12.).round() as usize;
            if !value.is_finite() || !(1..=12).contains(&code) || (value.abs()-code as f64/12.).abs()>1e-12 {
                return Err("invalid categorical board input".into());
            }
            let at=Coordinate::new((i%17) as i8-8,(i/17) as i8-8).map_err(|e|e.to_string())?;
            board.place(at,paisho_core::Tile {owner:if value>0. {Player::Host}else{Player::Guest},kind:paisho_core::STANDARD_TILE_KINDS[code-1]}).map_err(|e|e.to_string())?;
        }
        Ok(Self::from_board(&board,Player::Host))
    }
    fn from_board(board: &Board, perspective: Player) -> Self {
        let mut indices = [usize::MAX; CELL_COUNT];
        let nodes: Vec<_> = board
            .occupied()
            .enumerate()
            .map(|(i, (at, tile))| {
                indices[at.dense_index()] = i;
                let mut features = [0.; MICRO_PIECE_INPUTS];
                features[tile.kind.index()] = 1.;
                features[12 + usize::from(tile.owner != perspective)] = 1.;
                features[14] = f64::from(at.x()) / 8.;
                features[15] = f64::from(at.y()) / 8.;
                MicroPieceNode {
                    at,
                    owner: tile.owner,
                    features,
                }
            })
            .collect();
        let mut messages = vec![];
        visit_harmonies(board, |h| {
            for (from, to) in [(h.first, h.second), (h.second, h.first)] {
                let mut features = [0.; MICRO_HARMONY_INPUTS];
                features[usize::from(h.owner != perspective)] = 1.;
                features[2] = f64::from(to.x() - from.x()) / 16.;
                features[3] = f64::from(to.y() - from.y()) / 16.;
                features[4] = f64::from(h.orientation == LineOrientation::Horizontal);
                features[5] = f64::from(harmony_crosses_midline(h));
                messages.push(MicroHarmonyMessage {
                    source: indices[from.dense_index()],
                    destination: indices[to.dense_index()],
                    features,
                });
            }
        });
        Self {
            perspective,
            nodes,
            messages,
        }
    }
}
