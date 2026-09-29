use std::collections::BTreeMap;

use paisho_ai::{GameScore, PairedComparison};
use paisho_core::{GameOutcome, Player};

use crate::{AgentDefinition, LeagueExecution, LADDER_EDGE_ALIASES};

#[derive(Clone, Debug)]
pub struct PairwiseSummary {
    pub edge_index: usize,
    pub first_alias: &'static str,
    pub second_alias: &'static str,
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
    pub unfinished: usize,
    pub paired: PairedComparison,
}

pub fn summarize_pairwise(
    execution: &LeagueExecution,
    agents: &[AgentDefinition],
) -> Vec<PairwiseSummary> {
    LADDER_EDGE_ALIASES
        .iter()
        .enumerate()
        .map(|(edge_index, &(first_alias, second_alias))| {
            let first_agent = agents
                .iter()
                .position(|agent| agent.alias() == first_alias)
                .expect("every ladder alias has an agent definition");
            let edge_games: Vec<_> = execution
                .games
                .iter()
                .filter(|game| game.scheduled.edge_index == edge_index)
                .collect();
            let mut wins = 0;
            let mut draws = 0;
            let mut losses = 0;
            let mut unfinished = 0;
            let mut pairs: BTreeMap<u64, Vec<_>> = BTreeMap::new();
            for game in edge_games {
                let score = score_for_agent(game, first_agent);
                match score {
                    Some(GameScore::Win) => wins += 1,
                    Some(GameScore::Draw) => draws += 1,
                    Some(GameScore::Loss) => losses += 1,
                    None => unfinished += 1,
                }
                pairs.entry(game.scheduled.pair_id).or_default().push(score);
            }
            let mut paired = PairedComparison::default();
            for scores in pairs.values() {
                if scores.len() == 2 {
                    paired.observe(scores[0], scores[1]);
                } else {
                    paired.observe(scores.first().copied().flatten(), None);
                }
            }
            PairwiseSummary {
                edge_index,
                first_alias,
                second_alias,
                wins,
                draws,
                losses,
                unfinished,
                paired,
            }
        })
        .collect()
}

fn score_for_agent(game: &crate::PlayedGame, agent: usize) -> Option<GameScore> {
    let result = game.result.as_ref().ok()?;
    let outcome = result.scored_outcome()?;
    let player = if game.scheduled.host_agent == agent {
        Player::Host
    } else if game.scheduled.guest_agent == agent {
        Player::Guest
    } else {
        return None;
    };
    match outcome {
        GameOutcome::Win(winner) if winner == player => Some(GameScore::Win),
        GameOutcome::Win(_) => Some(GameScore::Loss),
        GameOutcome::Draw => Some(GameScore::Draw),
        GameOutcome::Ongoing => None,
    }
}

#[cfg(test)]
mod tests {
    use paisho_ai::{AgentTelemetry, MatchResult, MatchTermination};
    use paisho_core::{GameOutcome, GameRecord};

    use super::*;
    use crate::{build_ladder_schedule, default_agent_definitions, LeagueExecution, PlayedGame};

    #[test]
    fn pairwise_summary_uses_one_agents_perspective_across_reversed_seats() {
        let agents = default_agent_definitions("revision");
        let schedule = build_ladder_schedule(&agents, 1, 12_000).unwrap();
        let record: GameRecord =
            include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let final_position = record.replay().unwrap();
        let outcome = final_position.outcome();
        assert_ne!(outcome, GameOutcome::Ongoing);
        let games = schedule[..2]
            .iter()
            .map(|scheduled| PlayedGame {
                scheduled: *scheduled,
                result: Ok(MatchResult {
                    task_id: scheduled.game_id,
                    final_position: final_position.clone(),
                    record: record.clone(),
                    termination: MatchTermination::Rules(outcome),
                    host_telemetry: AgentTelemetry::default(),
                    guest_telemetry: AgentTelemetry::default(),
                }),
            })
            .collect();
        let execution = LeagueExecution {
            games,
            elapsed_seconds: 0.0,
            observed_match_workers: 1,
            worker_capacity: 1,
            available_parallelism: 1,
        };
        let summaries = summarize_pairwise(&execution, &agents);
        assert_eq!(summaries[0].wins, 1);
        assert_eq!(summaries[0].losses, 1);
        assert_eq!(summaries[0].draws, 0);
        assert_eq!(summaries[0].paired.one, 1);
        assert_eq!(summaries[0].paired.rated_pairs(), 1);
    }
}
