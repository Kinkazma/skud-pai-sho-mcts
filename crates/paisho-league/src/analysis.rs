use std::collections::BTreeMap;

use paisho_ai::{MatchTermination, StableRng};
use paisho_core::{GameOutcome, Player};
use paisho_rating::{
    fit_davidson_for_agents, simulate_site_elo, simulate_site_elo_in_order, AgentId, DavidsonError,
    DavidsonFit, DavidsonOptions, RatedGame, RatedOutcome, SiteEloConfig, SiteEloError, SiteEloRun,
};

use crate::{AgentDefinition, LeagueExecution, PlayedGame};

const CONTROLLED_ORDER_SEEDS: [u64; 4] = [
    0x4f52_4445_525f_3031,
    0x4f52_4445_525f_3032,
    0x4f52_4445_525f_3033,
    0x4f52_4445_525f_3034,
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExcludedPair {
    pub pair_id: u64,
    pub reason: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SiteOrderRun {
    pub label: String,
    pub run: SiteEloRun,
}

#[derive(Debug, PartialEq)]
pub struct LeagueAnalysis {
    pub rated_games: Vec<RatedGame>,
    pub rated_game_counts: BTreeMap<AgentId, usize>,
    pub excluded_pairs: Vec<ExcludedPair>,
    pub internal_fit: Result<DavidsonFit, DavidsonError>,
    pub site_order_runs: Vec<SiteOrderRun>,
    /// Extrema across the six recorded orders, not mathematical bounds over
    /// all possible permutations.
    pub site_sampled_order_ranges: BTreeMap<AgentId, (i32, i32)>,
}

pub fn analyze(
    execution: &LeagueExecution,
    agents: &[AgentDefinition],
    site_initial_rating: i32,
) -> Result<LeagueAnalysis, SiteEloError> {
    let (rated_games, excluded_pairs) = collect_rated_games(execution, agents);
    let agent_ids: Vec<_> = agents.iter().map(|agent| agent.id().clone()).collect();
    let mut rated_game_counts: BTreeMap<_, _> =
        agent_ids.iter().cloned().map(|agent| (agent, 0)).collect();
    for game in &rated_games {
        *rated_game_counts
            .get_mut(game.host())
            .expect("rated host belongs to the league") += 1;
        *rated_game_counts
            .get_mut(game.guest())
            .expect("rated guest belongs to the league") += 1;
    }
    let internal_fit =
        fit_davidson_for_agents(&agent_ids, &rated_games, DavidsonOptions::default());
    let site_config = SiteEloConfig::new(site_initial_rating);
    let canonical = simulate_site_elo(&agent_ids, &rated_games, site_config)?;
    let mut reverse_games = rated_games.clone();
    reverse_games.sort_by_key(|game| core::cmp::Reverse(game.sequence()));
    let reverse = simulate_site_elo_in_order(&agent_ids, &reverse_games, site_config)?;
    let mut site_order_runs = vec![
        SiteOrderRun {
            label: "canonical-schedule".to_owned(),
            run: canonical,
        },
        SiteOrderRun {
            label: "reverse".to_owned(),
            run: reverse,
        },
    ];
    for seed in CONTROLLED_ORDER_SEEDS {
        let mut shuffled = rated_games.clone();
        stable_shuffle(&mut shuffled, seed);
        site_order_runs.push(SiteOrderRun {
            label: format!("shuffle-{seed:016x}"),
            run: simulate_site_elo_in_order(&agent_ids, &shuffled, site_config)?,
        });
    }
    let mut site_sampled_order_ranges = BTreeMap::new();
    for agent in &agent_ids {
        let mut values = site_order_runs.iter().map(|order| order.run.ratings[agent]);
        let first = values
            .next()
            .expect("there is always a canonical-order run");
        let (minimum, maximum) = values.fold((first, first), |(minimum, maximum), value| {
            (minimum.min(value), maximum.max(value))
        });
        site_sampled_order_ranges.insert(agent.clone(), (minimum, maximum));
    }
    Ok(LeagueAnalysis {
        rated_games,
        rated_game_counts,
        excluded_pairs,
        internal_fit,
        site_order_runs,
        site_sampled_order_ranges,
    })
}

fn collect_rated_games(
    execution: &LeagueExecution,
    agents: &[AgentDefinition],
) -> (Vec<RatedGame>, Vec<ExcludedPair>) {
    let mut pairs: BTreeMap<u64, Vec<&PlayedGame>> = BTreeMap::new();
    for game in &execution.games {
        pairs.entry(game.scheduled.pair_id).or_default().push(game);
    }
    let mut rated_games = Vec::new();
    let mut excluded_pairs = Vec::new();
    for (pair_id, mut games) in pairs {
        games.sort_by_key(|game| game.scheduled.sequence);
        let reason = pair_exclusion_reason(&games, agents.len());
        if let Some(reason) = reason {
            excluded_pairs.push(ExcludedPair { pair_id, reason });
            continue;
        }
        for game in games {
            let outcome = game
                .result
                .as_ref()
                .expect("validated pair has successful games")
                .scored_outcome()
                .expect("validated pair has rated outcomes");
            rated_games.push(
                RatedGame::new(
                    game.scheduled.sequence,
                    pair_id,
                    agents[game.scheduled.host_agent].id().clone(),
                    agents[game.scheduled.guest_agent].id().clone(),
                    rated_outcome(outcome),
                )
                .expect("a scheduled league game has distinct agents"),
            );
        }
    }
    rated_games.sort_by_key(RatedGame::sequence);
    (rated_games, excluded_pairs)
}

fn pair_exclusion_reason(games: &[&PlayedGame], agent_count: usize) -> Option<&'static str> {
    if games.len() != 2
        || games[0].scheduled.leg != 0
        || games[1].scheduled.leg != 1
        || games[0].scheduled.edge_index != games[1].scheduled.edge_index
        || games[0].scheduled.game_id == games[1].scheduled.game_id
        || games[0].scheduled.sequence == games[1].scheduled.sequence
        || games[0].scheduled.sequence != games[0].scheduled.game_id
        || games[1].scheduled.sequence != games[1].scheduled.game_id
        || games[0].scheduled.host_agent >= agent_count
        || games[0].scheduled.guest_agent >= agent_count
        || games[1].scheduled.host_agent >= agent_count
        || games[1].scheduled.guest_agent >= agent_count
        || games[0].scheduled.host_agent == games[0].scheduled.guest_agent
        || games[0].scheduled.host_agent != games[1].scheduled.guest_agent
        || games[0].scheduled.guest_agent != games[1].scheduled.host_agent
        || games[0].scheduled.setup != games[1].scheduled.setup
        || games[0].scheduled.host_seed != games[1].scheduled.host_seed
        || games[0].scheduled.guest_seed != games[1].scheduled.guest_seed
    {
        return Some("malformed-pair");
    }
    if games.iter().any(|game| game.result.is_err()) {
        return Some("match-error");
    }
    if games.iter().any(|game| {
        let result = game.result.as_ref().expect("errors handled above");
        match result.termination {
            MatchTermination::Rules(outcome) => {
                outcome == GameOutcome::Ongoing || result.final_position.outcome() != outcome
            }
            MatchTermination::DecisionLimit => {
                result.final_position.outcome() != GameOutcome::Ongoing
            }
        }
    }) {
        return Some("invalid-match-result");
    }
    if games.iter().any(|game| {
        game.result
            .as_ref()
            .expect("errors handled above")
            .scored_outcome()
            .is_none()
    }) {
        return Some("incomplete-pair");
    }
    None
}

const fn rated_outcome(outcome: GameOutcome) -> RatedOutcome {
    match outcome {
        GameOutcome::Win(Player::Host) => RatedOutcome::HostWin,
        GameOutcome::Win(Player::Guest) => RatedOutcome::GuestWin,
        GameOutcome::Draw => RatedOutcome::Draw,
        GameOutcome::Ongoing => unreachable!(),
    }
}

fn stable_shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = StableRng::new(seed);
    for upper in (1..values.len()).rev() {
        values.swap(upper, rng.index(upper + 1));
    }
}

#[cfg(test)]
mod tests {
    use paisho_ai::{AgentTelemetry, MatchResult, MatchTermination};
    use paisho_core::{GameRecord, Position, StandardSetup, BASIC_FLOWERS};

    use super::*;

    fn scheduled_pair(setup: StandardSetup) -> [crate::ScheduledGame; 2] {
        [
            crate::ScheduledGame {
                sequence: 20,
                game_id: 20,
                pair_id: 10,
                edge_index: 0,
                leg: 0,
                host_agent: 0,
                guest_agent: 1,
                setup,
                host_seed: 7,
                guest_seed: 9,
            },
            crate::ScheduledGame {
                sequence: 21,
                game_id: 21,
                pair_id: 10,
                edge_index: 0,
                leg: 1,
                host_agent: 1,
                guest_agent: 0,
                setup,
                host_seed: 7,
                guest_seed: 9,
            },
        ]
    }

    fn result(
        game_id: u64,
        record: GameRecord,
        final_position: Position,
        termination: MatchTermination,
    ) -> MatchResult {
        MatchResult {
            task_id: game_id,
            final_position,
            record,
            termination,
            host_telemetry: AgentTelemetry::default(),
            guest_telemetry: AgentTelemetry::default(),
        }
    }

    #[test]
    fn controlled_shuffle_is_seeded_and_preserves_members() {
        let mut first = vec![0, 1, 2, 3, 4, 5, 6];
        let mut repeated = first.clone();
        stable_shuffle(&mut first, 7);
        stable_shuffle(&mut repeated, 7);
        assert_eq!(first, repeated);
        assert_ne!(first, vec![0, 1, 2, 3, 4, 5, 6]);
        first.sort_unstable();
        assert_eq!(first, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn a_single_unrated_leg_excludes_the_whole_pair() {
        let completed_record: GameRecord =
            include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let setup = completed_record.setup();
        let completed_position = completed_record.replay().unwrap();
        assert_ne!(completed_position.outcome(), GameOutcome::Ongoing);
        let scheduled = scheduled_pair(setup);
        let unfinished_record = GameRecord::new(setup);
        let unfinished_position = Position::from_standard_setup(setup);
        let games = [
            PlayedGame {
                scheduled: scheduled[0],
                result: Ok(result(
                    scheduled[0].game_id,
                    completed_record,
                    completed_position.clone(),
                    MatchTermination::Rules(completed_position.outcome()),
                )),
            },
            PlayedGame {
                scheduled: scheduled[1],
                result: Ok(result(
                    scheduled[1].game_id,
                    unfinished_record,
                    unfinished_position,
                    MatchTermination::DecisionLimit,
                )),
            },
        ];
        let references: Vec<_> = games.iter().collect();
        assert_eq!(
            pair_exclusion_reason(&references, 2),
            Some("incomplete-pair")
        );
    }

    #[test]
    fn an_inconsistent_rules_termination_is_rejected() {
        let setup = StandardSetup::balanced(BASIC_FLOWERS[0]);
        let scheduled = scheduled_pair(setup);
        let games: Vec<_> = scheduled
            .iter()
            .map(|game| PlayedGame {
                scheduled: *game,
                result: Ok(result(
                    game.game_id,
                    GameRecord::new(setup),
                    Position::from_standard_setup(setup),
                    MatchTermination::Rules(GameOutcome::Ongoing),
                )),
            })
            .collect();
        let references: Vec<_> = games.iter().collect();
        assert_eq!(
            pair_exclusion_reason(&references, 2),
            Some("invalid-match-result")
        );
    }
}
