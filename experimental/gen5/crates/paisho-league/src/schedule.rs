use core::fmt;
use std::collections::BTreeMap;

use paisho_core::{StandardSetup, BASIC_FLOWERS};

use crate::AgentDefinition;

pub const LADDER_EDGE_ALIASES: [(&str, &str); 5] = [
    ("site-bot-v1", "mcts-8"),
    ("mcts-8", "mcts-32"),
    ("mcts-32", "mcts-128"),
    ("mcts-128", "mcts-512"),
    ("site-bot-v1", "mcts-128"),
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScheduledGame {
    pub sequence: u64,
    pub game_id: u64,
    pub pair_id: u64,
    pub edge_index: usize,
    pub leg: u8,
    pub host_agent: usize,
    pub guest_agent: usize,
    pub setup: StandardSetup,
    pub host_seed: u64,
    pub guest_seed: u64,
}

pub fn build_ladder_schedule(
    agents: &[AgentDefinition],
    pairs_per_edge: usize,
    first_pair_id: u64,
) -> Result<Vec<ScheduledGame>, ScheduleError> {
    if pairs_per_edge == 0 {
        return Err(ScheduleError::ZeroPairs);
    }
    let aliases: BTreeMap<_, _> = agents
        .iter()
        .enumerate()
        .map(|(index, agent)| (agent.alias(), index))
        .collect();
    if aliases.len() != agents.len() {
        return Err(ScheduleError::DuplicateAlias);
    }
    let pair_count = pairs_per_edge
        .checked_mul(LADDER_EDGE_ALIASES.len())
        .ok_or(ScheduleError::IdOverflow)?;
    let game_count = pair_count.checked_mul(2).ok_or(ScheduleError::IdOverflow)?;
    let mut schedule = Vec::new();
    schedule
        .try_reserve_exact(game_count)
        .map_err(|_| ScheduleError::CapacityUnavailable)?;
    for (edge_index, (first_alias, second_alias)) in LADDER_EDGE_ALIASES.iter().enumerate() {
        let first = aliases
            .get(first_alias)
            .copied()
            .ok_or_else(|| ScheduleError::MissingAlias((*first_alias).to_owned()))?;
        let second = aliases
            .get(second_alias)
            .copied()
            .ok_or_else(|| ScheduleError::MissingAlias((*second_alias).to_owned()))?;
        for offset in 0..pairs_per_edge {
            let pair_offset = edge_index
                .checked_mul(pairs_per_edge)
                .and_then(|value| value.checked_add(offset))
                .ok_or(ScheduleError::IdOverflow)?;
            let pair_id = first_pair_id
                .checked_add(pair_offset as u64)
                .ok_or(ScheduleError::IdOverflow)?;
            let setup = StandardSetup::balanced(BASIC_FLOWERS[offset % BASIC_FLOWERS.len()]);
            for leg in 0..2_u8 {
                let game_id = pair_id
                    .checked_mul(2)
                    .and_then(|value| value.checked_add(u64::from(leg)))
                    .ok_or(ScheduleError::IdOverflow)?;
                let (host_agent, guest_agent) = if leg == 0 {
                    (first, second)
                } else {
                    (second, first)
                };
                schedule.push(ScheduledGame {
                    sequence: game_id,
                    game_id,
                    pair_id,
                    edge_index,
                    leg,
                    host_agent,
                    guest_agent,
                    setup,
                    host_seed: host_seed(pair_id),
                    guest_seed: guest_seed(pair_id),
                });
            }
        }
    }
    Ok(schedule)
}

const fn host_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x484f_5354_5f4c_4731
}

const fn guest_seed(pair_id: u64) -> u64 {
    pair_id ^ 0x4755_4553_545f_4c31
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScheduleError {
    ZeroPairs,
    DuplicateAlias,
    MissingAlias(String),
    IdOverflow,
    CapacityUnavailable,
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPairs => formatter.write_str("a league edge needs at least one pair"),
            Self::DuplicateAlias => formatter.write_str("league agent aliases must be unique"),
            Self::MissingAlias(alias) => write!(formatter, "missing league agent alias `{alias}`"),
            Self::IdOverflow => formatter.write_str("league identifiers overflow u64 or usize"),
            Self::CapacityUnavailable => {
                formatter.write_str("league schedule is too large to allocate")
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::default_agent_definitions;

    #[test]
    fn ladder_is_reversed_paired_and_cycles_openings() {
        let agents = default_agent_definitions("revision");
        let schedule = build_ladder_schedule(&agents, 6, 12_000).unwrap();
        assert_eq!(schedule.len(), LADDER_EDGE_ALIASES.len() * 6 * 2);
        for legs in schedule.chunks_exact(2) {
            assert_eq!(legs[0].pair_id, legs[1].pair_id);
            assert_eq!(legs[0].host_agent, legs[1].guest_agent);
            assert_eq!(legs[0].guest_agent, legs[1].host_agent);
            assert_eq!(legs[0].setup, legs[1].setup);
            assert_eq!(legs[0].host_seed, legs[1].host_seed);
            assert_eq!(legs[0].guest_seed, legs[1].guest_seed);
            assert_ne!(legs[0].game_id, legs[1].game_id);
            assert_eq!(legs[0].sequence, legs[0].game_id);
            assert_eq!(legs[1].sequence, legs[1].game_id);
        }
        for edge in 0..LADDER_EDGE_ALIASES.len() {
            let openings: Vec<_> = schedule
                .iter()
                .filter(|game| game.edge_index == edge && game.leg == 0)
                .map(|game| game.setup.starting_flower)
                .collect();
            assert_eq!(openings, BASIC_FLOWERS);
        }
    }

    #[test]
    fn disjoint_pair_blocks_have_disjoint_global_sequences() {
        let agents = default_agent_definitions("revision");
        let first = build_ladder_schedule(&agents, 6, 12_000).unwrap();
        let second = build_ladder_schedule(&agents, 6, 13_000).unwrap();
        let first_sequences: std::collections::BTreeSet<_> =
            first.iter().map(|game| game.sequence).collect();
        assert!(second
            .iter()
            .all(|game| !first_sequences.contains(&game.sequence)));
    }

    #[test]
    fn an_impossibly_large_schedule_returns_an_error_instead_of_panicking() {
        let agents = default_agent_definitions("revision");
        let pairs = usize::MAX / (LADDER_EDGE_ALIASES.len() * 2);
        assert!(build_ladder_schedule(&agents, pairs, 0).is_err());
    }
}
