//! Local PSR highlights from existing terminal replays. No search, inference or submission.
use std::cmp::Reverse;
use std::error::Error;
use std::path::{Path, PathBuf};

use paisho_core::{midline_crossing_harmony_count, GameOutcome, Player};
use paisho_replay::{PolicyTargetKindV1, ReplayDigestV1, ReplayGameV1};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const HIGHLIGHT_MAXIMUM_DECISIONS: usize = 4096;
pub const HIGHLIGHT_MAXIMUM_BYTES: usize = 512 * 1024;
pub type HighlightedGamesError = Box<dyn Error + Send + Sync>;
const SELECTION: &str = "recorded-Behavior-viewpoint; win-draw-loss; terminal-midline-margin-desc; terminal-ring-desc; game-id-asc; record-sha256-asc; host-before-guest";

#[derive(Clone, Debug, Serialize)]
pub struct HighlightedBoardTile {
    pub x: i8,
    pub y: i8,
    pub owner: String,
    pub code: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct HighlightedGameMetadata {
    pub game_id: u64,
    pub neural_side: String,
    pub neural_result: String,
    pub terminal_outcome: String,
    pub total_decisions: usize,
    pub completed_turns: u32,
    pub final_board: Vec<HighlightedBoardTile>,
    pub host_producer: String,
    pub guest_producer: String,
    pub terminal_neural_midline_harmonies: usize,
    pub terminal_opponent_midline_harmonies: usize,
    pub terminal_midline_margin: i64,
    pub terminal_neural_ring: bool,
    pub record_sha256: String,
    pub record_bytes: usize,
    pub record_decisions: usize,
    /// Zero-based record indices, including any retained decisions for both seats in self-play.
    pub recorded_neural_decision_indices: Vec<usize>,
    /// Decisions before the first stored target of any kind, not attributed to the network.
    /// Shards alone do not distinguish a neutral prefix from an initial unrecorded opponent move.
    pub unrecorded_prefix_decisions: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HighlightedGamesMetadata {
    pub format: String,
    pub generation: u64,
    pub neural_producer: String,
    pub selection: String,
    pub prefix_note: String,
    pub scope: String,
    pub games_considered: usize,
    pub neural_wins: usize,
    pub excluded_oversized_wins: usize,
    pub excluded_oversized_games: usize,
    pub maximum_decisions: usize,
    pub maximum_bytes: usize,
    pub selected: Option<HighlightedGameMetadata>,
}

#[derive(Clone, Debug)]
pub struct HighlightedGameSelection {
    pub record: Option<String>,
    pub metadata: HighlightedGamesMetadata,
}

#[derive(Clone, Debug)]
pub struct HighlightedGameExport {
    /// Only this artifact is intended for the user-facing gallery/local replay importer.
    pub psr_path: Option<PathBuf>,
    /// Internal provenance; not a contribution payload or user replay artifact.
    pub metadata_path: PathBuf,
    pub metadata: HighlightedGamesMetadata,
}

fn compatible(decisions: usize, bytes: usize) -> bool {
    decisions <= HIGHLIGHT_MAXIMUM_DECISIONS && bytes <= HIGHLIGHT_MAXIMUM_BYTES
}

fn side(player: Player) -> &'static str {
    match player {
        Player::Host => "host",
        Player::Guest => "guest",
    }
}

fn score(metadata: &HighlightedGameMetadata) -> (u8, i64, bool, Reverse<u64>, Reverse<&str>) {
    (
        match metadata.neural_result.as_str() {
            "win" => 2,
            "draw" => 1,
            _ => 0,
        },
        metadata.terminal_midline_margin,
        metadata.terminal_neural_ring,
        Reverse(metadata.game_id),
        Reverse(&metadata.record_sha256),
    )
}

/// Select a compatible game from a recorded Behavior viewpoint, preferring win,
/// draw, then loss. Generation labels collection, not the subsequently trained model.
/// All source actions remain intact, including unrecorded neutral prefixes.
pub fn select_highlighted_game<'a>(
    generation: u64,
    producer: ReplayDigestV1,
    games: impl IntoIterator<Item = &'a ReplayGameV1>,
) -> Result<HighlightedGameSelection, HighlightedGamesError> {
    let mut result = HighlightedGameSelection {
        record: None,
        metadata: HighlightedGamesMetadata {
            format: "paisho-local-neural-highlight-v1".into(), generation,
            neural_producer: producer.to_string(), selection: SELECTION.into(),
            prefix_note: "Full original record retained. Decisions without recorded Behavior evidence are not attributed to this neural producer; the unrecorded prefix can include neutral random play or an opponent move. Recorded indices are zero-based.".into(),
            scope: "Local neural replay only; no human participant or human Elo asserted; never submit to the contribution endpoint.".into(),
            games_considered: 0, neural_wins: 0, excluded_oversized_wins: 0, excluded_oversized_games: 0,
            maximum_decisions: HIGHLIGHT_MAXIMUM_DECISIONS, maximum_bytes: HIGHLIGHT_MAXIMUM_BYTES,
            selected: None,
        },
    };
    for game in games {
        result.metadata.games_considered += 1;
        let neural_indices: Vec<_> = game
            .decisions()
            .iter()
            .filter(|decision| {
                decision.policy().kind() == PolicyTargetKindV1::Behavior
                    && decision.policy().producer() == producer
            })
            .map(|decision| decision.decision_index())
            .collect();
        if neural_indices.is_empty() {
            continue;
        }
        // Reconstruct the final board and the actual actor of each recorded decision.
        // Actor identities alone are insufficient: in self-play only the losing seat
        // might have recorded targets in a sparse source shard.
        let mut final_position = game.record().initial_position();
        let mut recorded_sides = [false; 2];
        for (index, &action) in game.record().actions().iter().enumerate() {
            if neural_indices.binary_search(&index).is_ok() {
                recorded_sides[final_position.to_move().index()] = true;
            }
            final_position.apply(action)?;
        }
        let is_neural_win =
            matches!(game.outcome(), GameOutcome::Win(winner) if recorded_sides[winner.index()]);
        if is_neural_win {
            result.metadata.neural_wins += 1;
        }
        let record = game.record().to_string();
        if !compatible(game.record().actions().len(), record.len()) {
            result.metadata.excluded_oversized_games += 1;
            if is_neural_win {
                result.metadata.excluded_oversized_wins += 1;
            }
            continue;
        }
        for winner in [Player::Host, Player::Guest] {
            if !recorded_sides[winner.index()] {
                continue;
            }
            let neural_result = match game.outcome() {
                GameOutcome::Win(side) if side == winner => "win",
                GameOutcome::Win(_) => "loss",
                GameOutcome::Draw => "draw",
                GameOutcome::Ongoing => continue,
            };
            let neural_midline = midline_crossing_harmony_count(final_position.board(), winner);
            let opponent_midline =
                midline_crossing_harmony_count(final_position.board(), winner.opponent());
            let record_digest =
                ReplayDigestV1::from_bytes(Sha256::digest(record.as_bytes()).into());
            let metadata = HighlightedGameMetadata {
                game_id: game.game_id(),
                neural_side: side(winner).into(),
                neural_result: neural_result.into(),
                terminal_outcome: match game.outcome() {
                    GameOutcome::Win(Player::Host) => "host-wins",
                    GameOutcome::Win(Player::Guest) => "guest-wins",
                    GameOutcome::Draw => "draw",
                    GameOutcome::Ongoing => unreachable!(),
                }
                .into(),
                total_decisions: game.record().actions().len(),
                completed_turns: final_position.completed_turns(),
                final_board: final_position
                    .board()
                    .occupied()
                    .map(|(coordinate, tile)| HighlightedBoardTile {
                        x: coordinate.x(),
                        y: coordinate.y(),
                        owner: side(tile.owner).into(),
                        code: tile.kind.code().into(),
                    })
                    .collect(),
                host_producer: game.host_agent().to_string(),
                guest_producer: game.guest_agent().to_string(),
                terminal_neural_midline_harmonies: neural_midline,
                terminal_opponent_midline_harmonies: opponent_midline,
                terminal_midline_margin: neural_midline as i64 - opponent_midline as i64,
                terminal_neural_ring: paisho_core::harmony_ring_owners_for_profile(
                    final_position.board(),
                    final_position.rule_profile(),
                )
                .contains(&winner),
                record_sha256: record_digest.to_string(),
                record_bytes: record.len(),
                record_decisions: game.record().actions().len(),
                recorded_neural_decision_indices: neural_indices.clone(),
                unrecorded_prefix_decisions: game.decisions()[0].decision_index(),
            };
            if result
                .metadata
                .selected
                .as_ref()
                .map_or(true, |old| score(&metadata) > score(old))
            {
                result.record = Some(record.clone());
                result.metadata.selected = Some(metadata);
            }
        }
    }
    Ok(result)
}

/// Idempotent per-generation local publication. Different existing contents are never
/// overwritten. Call once over the complete generation, not once per input shard.
pub fn write_highlighted_game(
    output_directory: &Path,
    selected: HighlightedGameSelection,
) -> Result<HighlightedGameExport, HighlightedGamesError> {
    let directory =
        output_directory.join(format!("generation-{:020}", selected.metadata.generation));
    let psr_path = selected
        .record
        .as_ref()
        .map(|_| directory.join("best-game.psr"));
    let metadata_path = directory.join("meta.json");
    // Publish provenance first; the PSR is the gallery-visible completion artifact.
    crate::atomic_file::write_idempotently(
        &metadata_path,
        &serde_json::to_vec_pretty(&selected.metadata)?,
    )?;
    if let (Some(path), Some(record)) = (&psr_path, selected.record) {
        crate::atomic_file::write_idempotently(path, record.as_bytes())?;
    }
    Ok(HighlightedGameExport {
        psr_path,
        metadata_path,
        metadata: selected.metadata,
    })
}

pub fn export_highlighted_game<'a>(
    generation: u64,
    producer: ReplayDigestV1,
    games: impl IntoIterator<Item = &'a ReplayGameV1>,
    output_directory: &Path,
) -> Result<HighlightedGameExport, HighlightedGamesError> {
    write_highlighted_game(
        output_directory,
        select_highlighted_game(generation, producer, games)?,
    )
}

#[cfg(test)]
#[path = "highlighted_games/tests.rs"]
mod tests;
