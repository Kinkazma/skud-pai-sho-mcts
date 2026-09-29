use paisho_ai::{CpuMctsEvaluator, HeuristicWeights, MctsAgent, MctsConfig, MctsEvaluator};
use paisho_core::{legal_actions, BasicFlower, Player, Position, StandardSetup};

struct BadEvaluator;
impl MctsEvaluator for BadEvaluator {
    fn evaluate(&self, _: &[Position], _: Player, _: HeuristicWeights) -> Result<Vec<f32>, String> {
        Ok(vec![f32::NAN])
    }
}
struct BrokenEvaluator;
impl MctsEvaluator for BrokenEvaluator {
    fn evaluate(&self, _: &[Position], _: Player, _: HeuristicWeights) -> Result<Vec<f32>, String> {
        Err("service stopped".into())
    }
}
#[test]
fn evaluator_failures_are_errors_not_cpu_fallbacks() {
    let p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let actions = legal_actions(&p);
    for evaluator in [&BadEvaluator as &dyn MctsEvaluator, &BrokenEvaluator] {
        let mut agent = MctsAgent::new(
            71,
            MctsConfig {
                simulations: 32,
                ..MctsConfig::default()
            },
        )
        .unwrap();
        assert!(agent
            .search_with_evaluator(&p, &actions, evaluator)
            .is_err());
    }
    let mut agent = MctsAgent::new(71, MctsConfig::default()).unwrap();
    assert!(agent
        .search_with_evaluator(&p, &[], &CpuMctsEvaluator)
        .is_err());
}

#[test]
#[ignore = "Metal hardware; run under training pause wrapper with PAISHO_MCTS_METAL_SERVICE and PAISHO_MCTS_METAL_KERNEL"]
fn metal_features_match_board_oracle_with_all_tiles_and_line_effects() {
    use paisho_ai::{MetalMctsEvaluator, StableRng};
    use paisho_core::{
        harmonies, harmony_crosses_midline, playable_coordinates, point_type_at, Board, Tile,
        STANDARD_TILE_KINDS,
    };
    use std::{path::Path, time::Duration};
    let service = std::env::var("PAISHO_MCTS_METAL_SERVICE").unwrap();
    let kernel = std::env::var("PAISHO_MCTS_METAL_KERNEL").unwrap();
    // Chunking across service capacity and an incomplete final GPU batch.
    let gpu =
        MetalMctsEvaluator::launch(Path::new(&service), Path::new(&kernel), 127, Duration::ZERO)
            .unwrap();
    let coords: Vec<_> = playable_coordinates().collect();
    let mut rng = StableRng::new(5523);
    let mut input = Vec::new();
    let mut expected = Vec::new();
    for id in 0..1003 {
        let mut board = Board::empty();
        let mut wire = [0u8; 289];
        for &c in &coords {
            if rng.index(100) >= id % 100 {
                continue;
            }
            let kind = STANDARD_TILE_KINDS[rng.index(STANDARD_TILE_KINDS.len())];
            let owner = if rng.index(2) == 0 {
                Player::Host
            } else {
                Player::Guest
            };
            board.place(c, Tile::new(owner, kind)).unwrap();
            wire[c.dense_index()] = (1 + kind.index() + 12 * owner.index()) as u8;
        }
        let mut f = [0i32; 4];
        for h in harmonies(&board) {
            let sign = if h.owner == Player::Host { 1 } else { -1 };
            f[0] += sign;
            if harmony_crosses_midline(h) {
                f[1] += sign;
            }
        }
        for (c, t) in board.occupied() {
            if t.kind.is_flower() {
                let sign = if t.owner == Player::Host { 1 } else { -1 };
                f[3] += sign;
                if !point_type_at(c).is_gate() {
                    f[2] += sign;
                }
            }
        }
        expected.push(f);
        input.extend(wire);
    }
    assert_eq!(gpu.board_features(input).unwrap(), expected);
    assert!(gpu.board_features(vec![0; 288]).is_err());
    assert!(gpu.board_features(Vec::new()).is_err());
}
