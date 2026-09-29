use core::fmt;
use std::collections::{BTreeMap, BTreeSet};

use crate::{AgentId, RatedGame};

/// Hard-coded by The Garden Gate's public `js/util/elo.js` implementation.
pub const SITE_ELO_K_FACTOR: i32 = 32;
/// Official source revision used to pin the calculator contract.
pub const SITE_ELO_SOURCE_COMMIT: &str = "b849dbdabb1138ff0f6d609adf38b301c2f875ae";
/// Path within the official SkudPaiSho repository at the pinned revision.
pub const SITE_ELO_SOURCE_PATH: &str = "js/util/elo.js";

/// The public JavaScript does not define account initialization, so callers
/// must provide it explicitly and preserve that choice with their evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SiteEloConfig {
    pub initial_rating: i32,
}

impl SiteEloConfig {
    pub const fn new(initial_rating: i32) -> Self {
        Self { initial_rating }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SiteEloUpdate {
    pub sequence: u64,
    pub pair_id: u64,
    pub host: AgentId,
    pub guest: AgentId,
    pub host_rating_before: i32,
    pub guest_rating_before: i32,
    pub expected_host_score: f64,
    pub actual_host_score: f64,
    pub delta: i32,
    pub host_rating_after: i32,
    pub guest_rating_after: i32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SiteEloRun {
    pub initial_rating: i32,
    pub ratings: BTreeMap<AgentId, i32>,
    pub updates: Vec<SiteEloUpdate>,
}

/// Replays the games by increasing immutable sequence number. The sequence is
/// a caller-defined canonical order; it need not be wall-clock chronology.
pub fn simulate_site_elo(
    agents: &[AgentId],
    games: &[RatedGame],
    config: SiteEloConfig,
) -> Result<SiteEloRun, SiteEloError> {
    validate_unique_sequences(games)?;
    let mut ordered: Vec<_> = games.iter().collect();
    ordered.sort_by_key(|game| game.sequence());
    simulate_references_in_order(agents, &ordered, config)
}

/// Replays exactly the supplied order. This is the entry point for controlled
/// order-sensitivity experiments; sequence numbers remain unique identifiers.
pub fn simulate_site_elo_in_order(
    agents: &[AgentId],
    games: &[RatedGame],
    config: SiteEloConfig,
) -> Result<SiteEloRun, SiteEloError> {
    validate_unique_sequences(games)?;
    let ordered: Vec<_> = games.iter().collect();
    simulate_references_in_order(agents, &ordered, config)
}

fn simulate_references_in_order(
    agents: &[AgentId],
    games: &[&RatedGame],
    config: SiteEloConfig,
) -> Result<SiteEloRun, SiteEloError> {
    let mut ratings = initial_ratings(agents, config.initial_rating)?;
    let mut updates = Vec::with_capacity(games.len());
    for game in games {
        let host_rating = ratings
            .get(game.host())
            .copied()
            .ok_or_else(|| SiteEloError::UnknownAgent(game.host().clone()))?;
        let guest_rating = ratings
            .get(game.guest())
            .copied()
            .ok_or_else(|| SiteEloError::UnknownAgent(game.guest().clone()))?;
        let expected_host_score = expected_score(host_rating, guest_rating);
        let actual_host_score = game.outcome().host_score();
        let delta = javascript_round(
            f64::from(SITE_ELO_K_FACTOR) * (actual_host_score - expected_host_score),
        );
        let host_after = host_rating
            .checked_add(delta)
            .ok_or(SiteEloError::RatingOverflow)?;
        let guest_after = guest_rating
            .checked_sub(delta)
            .ok_or(SiteEloError::RatingOverflow)?;
        ratings.insert(game.host().clone(), host_after);
        ratings.insert(game.guest().clone(), guest_after);
        updates.push(SiteEloUpdate {
            sequence: game.sequence(),
            pair_id: game.pair_id(),
            host: game.host().clone(),
            guest: game.guest().clone(),
            host_rating_before: host_rating,
            guest_rating_before: guest_rating,
            expected_host_score,
            actual_host_score,
            delta,
            host_rating_after: host_after,
            guest_rating_after: guest_after,
        });
    }
    Ok(SiteEloRun {
        initial_rating: config.initial_rating,
        ratings,
        updates,
    })
}

fn initial_ratings(
    agents: &[AgentId],
    initial_rating: i32,
) -> Result<BTreeMap<AgentId, i32>, SiteEloError> {
    if agents.is_empty() {
        return Err(SiteEloError::NoAgents);
    }
    let mut ratings = BTreeMap::new();
    for agent in agents {
        if ratings.insert(agent.clone(), initial_rating).is_some() {
            return Err(SiteEloError::DuplicateAgent(agent.clone()));
        }
    }
    Ok(ratings)
}

fn validate_unique_sequences(games: &[RatedGame]) -> Result<(), SiteEloError> {
    let mut sequences = BTreeSet::new();
    for game in games {
        if !sequences.insert(game.sequence()) {
            return Err(SiteEloError::DuplicateSequence(game.sequence()));
        }
    }
    Ok(())
}

fn expected_score(rating: i32, opponent_rating: i32) -> f64 {
    1.0 / (1.0 + 10.0_f64.powf((f64::from(opponent_rating) - f64::from(rating)) / 400.0))
}

/// JavaScript `Math.round`: nearest integer with exact ties toward positive
/// infinity. Its distinguishable negative zero becomes integer zero here.
fn javascript_round(value: f64) -> i32 {
    (value + 0.5).floor() as i32
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SiteEloError {
    NoAgents,
    DuplicateAgent(AgentId),
    UnknownAgent(AgentId),
    DuplicateSequence(u64),
    RatingOverflow,
}

impl fmt::Display for SiteEloError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAgents => formatter.write_str("site Elo simulation needs at least one agent"),
            Self::DuplicateAgent(agent) => write!(formatter, "duplicate agent `{agent}`"),
            Self::UnknownAgent(agent) => {
                write!(formatter, "game references unknown agent `{agent}`")
            }
            Self::DuplicateSequence(sequence) => {
                write!(formatter, "duplicate game sequence {sequence}")
            }
            Self::RatingOverflow => formatter.write_str("site Elo rating overflow"),
        }
    }
}

impl std::error::Error for SiteEloError {}

#[cfg(test)]
mod tests {
    use crate::{RatedGame, RatedOutcome};

    use super::*;

    fn id(value: &str) -> AgentId {
        AgentId::new(value).unwrap()
    }

    fn game(sequence: u64, host: &str, guest: &str, outcome: RatedOutcome) -> RatedGame {
        RatedGame::new(sequence, sequence / 2, id(host), id(guest), outcome).unwrap()
    }

    #[test]
    fn equal_ratings_match_the_public_k32_formula() {
        let agents = [id("a"), id("b")];
        let run = simulate_site_elo(
            &agents,
            &[game(0, "a", "b", RatedOutcome::HostWin)],
            SiteEloConfig::new(1_000),
        )
        .unwrap();
        assert_eq!(run.ratings[&id("a")], 1_016);
        assert_eq!(run.ratings[&id("b")], 984);
        assert_eq!(run.updates[0].delta, 16);
        assert_eq!(run.updates[0].expected_host_score, 0.5);
    }

    #[test]
    fn javascript_ties_round_toward_positive_infinity() {
        assert_eq!(javascript_round(1.5), 2);
        assert_eq!(javascript_round(0.5), 1);
        assert_eq!(javascript_round(-0.5), 0);
        assert_eq!(javascript_round(-1.5), -1);
    }

    #[test]
    fn chronological_entry_point_sorts_but_explicit_order_does_not() {
        let agents = [id("a"), id("b"), id("c")];
        let games = [
            game(2, "a", "c", RatedOutcome::GuestWin),
            game(0, "a", "b", RatedOutcome::HostWin),
            game(1, "b", "c", RatedOutcome::HostWin),
        ];
        let chronological = simulate_site_elo(&agents, &games, SiteEloConfig::new(1_000)).unwrap();
        assert_eq!(chronological.updates[0].sequence, 0);
        let supplied =
            simulate_site_elo_in_order(&agents, &games, SiteEloConfig::new(1_000)).unwrap();
        assert_eq!(supplied.updates[0].sequence, 2);
        assert_ne!(chronological.ratings, supplied.ratings);
    }

    #[test]
    fn malformed_histories_are_rejected() {
        let agents = [id("a"), id("b")];
        let duplicate = [
            game(7, "a", "b", RatedOutcome::Draw),
            game(7, "b", "a", RatedOutcome::Draw),
        ];
        assert_eq!(
            simulate_site_elo(&agents, &duplicate, SiteEloConfig::new(1_000)),
            Err(SiteEloError::DuplicateSequence(7))
        );
        assert_eq!(
            simulate_site_elo(
                &agents,
                &[game(1, "a", "missing", RatedOutcome::Draw)],
                SiteEloConfig::new(1_000)
            ),
            Err(SiteEloError::UnknownAgent(id("missing")))
        );
    }
}
