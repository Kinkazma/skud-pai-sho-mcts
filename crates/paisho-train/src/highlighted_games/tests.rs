use super::*;
use paisho_core::GameRecord;
use paisho_model::encode_action_v1;
use paisho_replay::{PolicyEntryV1, PolicyTargetV1, ReplayDecisionV1};

const RING: &str = include_str!("../../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
const PRODUCER: ReplayDigestV1 = ReplayDigestV1::from_bytes([1; 32]);
const OTHER: ReplayDigestV1 = ReplayDigestV1::from_bytes([2; 32]);

fn game(id: u64, record: GameRecord, candidate: Player) -> ReplayGameV1 {
    let mut position = record.initial_position();
    let mut last = None;
    for (index, &action) in record.actions().iter().enumerate() {
        if position.to_move() == candidate {
            last = Some(ReplayDecisionV1::new(
                index,
                PolicyTargetV1::new(
                    PolicyTargetKindV1::Behavior,
                    PRODUCER,
                    vec![
                        PolicyEntryV1::new(encode_action_v1(action, candidate).unwrap(), 1.0)
                            .unwrap(),
                    ],
                )
                .unwrap(),
            ));
        }
        position.apply(action).unwrap();
    }
    ReplayGameV1::new(
        id,
        if candidate == Player::Host {
            PRODUCER
        } else {
            OTHER
        },
        if candidate == Player::Guest {
            PRODUCER
        } else {
            OTHER
        },
        record,
        vec![last.unwrap()],
    )
    .unwrap()
}

fn winner(record: &GameRecord) -> Player {
    let GameOutcome::Win(winner) = record.replay().unwrap().outcome() else {
        panic!("fixture must win")
    };
    winner
}

#[test]
fn preserves_full_prefix_and_exact_terminal_board_from_either_viewpoint() {
    let record: GameRecord = RING.parse().unwrap();
    let final_position = record.replay().unwrap();
    for candidate in [Player::Host, Player::Guest] {
        let replay = game(1, record.clone(), candidate);
        let selected = select_highlighted_game(47, PRODUCER, [&replay]).unwrap();
        assert_eq!(
            selected.record.as_deref(),
            Some(record.to_string().as_str())
        );
        let metadata = selected.metadata.selected.unwrap();
        assert_eq!(metadata.neural_side, side(candidate));
        assert_eq!(
            metadata.neural_result,
            if candidate == winner(&record) {
                "win"
            } else {
                "loss"
            }
        );
        assert_eq!(
            metadata.terminal_outcome,
            format!("{}-wins", side(winner(&record)))
        );
        assert_eq!(metadata.total_decisions, record.actions().len());
        assert_eq!(metadata.completed_turns, final_position.completed_turns());
        assert!(metadata.unrecorded_prefix_decisions > 0);
        assert_eq!(
            metadata.final_board.len(),
            final_position.board().occupied_count()
        );
        for (tile, (coordinate, expected)) in metadata
            .final_board
            .iter()
            .zip(final_position.board().occupied())
        {
            assert_eq!((tile.x, tile.y), (coordinate.x(), coordinate.y()));
            assert_eq!(tile.owner, side(expected.owner));
            assert_eq!(tile.code, expected.kind.code());
        }
        assert_eq!(
            metadata.terminal_midline_margin,
            midline_crossing_harmony_count(final_position.board(), candidate) as i64
                - midline_crossing_harmony_count(final_position.board(), candidate.opponent())
                    as i64
        );
    }
}

#[test]
fn wins_before_losses_and_ties_ignore_input_order() {
    let record: GameRecord = RING.parse().unwrap();
    let won = winner(&record);
    let low = game(3, record.clone(), won);
    let high = game(9, record.clone(), won);
    let loss = game(0, record, won.opponent());
    for games in [[&loss, &low, &high], [&high, &low, &loss]] {
        let result = select_highlighted_game(7, PRODUCER, games).unwrap();
        assert_eq!(result.metadata.selected.unwrap().game_id, 3);
        assert_eq!(result.metadata.neural_wins, 2);
    }
    let selection = select_highlighted_game(7, PRODUCER, [&loss]).unwrap();
    let mut loss_metadata = selection.metadata.selected.unwrap();
    let mut draw = loss_metadata.clone();
    draw.neural_result = "draw".into();
    draw.terminal_midline_margin = -100;
    assert!(score(&draw) > score(&loss_metadata));
    loss_metadata.neural_result = "win".into();
    assert!(score(&loss_metadata) > score(&draw));
}

#[test]
fn absent_behavior_producer_does_not_invent_neural_evidence() {
    let record: GameRecord = RING.parse().unwrap();
    let replay = game(1, record.clone(), winner(&record));
    assert!(select_highlighted_game(1, OTHER, [&replay])
        .unwrap()
        .record
        .is_none());
    assert!(select_highlighted_game(1, PRODUCER, [])
        .unwrap()
        .record
        .is_none());
}

#[test]
fn exact_inclusive_browser_limits() {
    assert!(compatible(4096, 512 * 1024));
    assert!(!compatible(4097, 512 * 1024));
    assert!(!compatible(4096, 512 * 1024 + 1));
}

#[test]
fn oversized_win_is_excluded_not_truncated_and_compatible_loss_is_retained() {
    let record: GameRecord = RING.parse().unwrap();
    let won = winner(&record);
    let mut long = GameRecord::with_rules(record.setup(), record.rules());
    for &action in &record.actions()[..2] {
        long.push(action);
    }
    for _ in 0..1024 {
        for action in [
            "arrange -2,-6 -2,-5",
            "arrange 1,8 1,7",
            "arrange -2,-5 -2,-6",
            "arrange 1,7 1,8",
        ] {
            long.push(action.parse().unwrap());
        }
    }
    for &action in &record.actions()[2..] {
        long.push(action);
    }
    let too_long = game(0, long, won);
    let compatible_loss = game(1, record.clone(), won.opponent());
    let selection = select_highlighted_game(7, PRODUCER, [&too_long, &compatible_loss]).unwrap();
    assert_eq!(selection.metadata.excluded_oversized_games, 1);
    assert_eq!(selection.metadata.excluded_oversized_wins, 1);
    assert_eq!(selection.metadata.selected.unwrap().neural_result, "loss");
    assert_eq!(selection.record.unwrap(), record.to_string());
}

#[test]
fn publishes_psr_only_user_artifact_and_idempotent_internal_metadata() {
    let record: GameRecord = RING.parse().unwrap();
    let replay = game(1, record.clone(), winner(&record));
    let directory = std::env::temp_dir().join(format!(
        "paisho-highlight-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let exported = export_highlighted_game(47, PRODUCER, [&replay], &directory).unwrap();
    let path = exported.psr_path.unwrap();
    assert_eq!(
        path,
        directory.join("generation-00000000000000000047/best-game.psr")
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap(), record.to_string());
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(exported.metadata_path).unwrap()).unwrap();
    assert_eq!(metadata["generation"], 47);
    assert!(metadata.get("elo").is_none());
    assert!(metadata.get("moves").is_none());
    export_highlighted_game(47, PRODUCER, [&replay], &directory).unwrap();
    let none = export_highlighted_game(48, OTHER, [&replay], &directory).unwrap();
    assert!(none.psr_path.is_none());
    assert!(!directory
        .join("generation-00000000000000000048/best-game.psr")
        .exists());
    std::fs::remove_dir_all(&directory).unwrap();
}
