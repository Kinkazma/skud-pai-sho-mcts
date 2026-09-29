use std::collections::{BTreeMap, BTreeSet};

use paisho_rating::{simulate_site_elo, AgentId, RatedGame, RatedOutcome, SiteEloConfig};

#[test]
fn rust_updates_match_the_pinned_official_javascript() {
    let games = parse_games(include_str!("fixtures/independent_reference_games.tsv"));
    let (expected_updates, expected_ratings) =
        parse_expected(include_str!("fixtures/independent_site_elo_expected.tsv"));
    let agents: Vec<_> = games
        .iter()
        .flat_map(|game| [game.host().clone(), game.guest().clone()])
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let run = simulate_site_elo(&agents, &games, SiteEloConfig::new(1_000)).unwrap();

    assert_eq!(run.updates.len(), expected_updates.len());
    for (actual, expected) in run.updates.iter().zip(expected_updates) {
        assert_eq!(actual.sequence, expected.sequence);
        assert_eq!(actual.pair_id, expected.pair_id);
        assert_eq!(actual.host.as_str(), expected.host);
        assert_eq!(actual.guest.as_str(), expected.guest);
        assert_eq!(actual.host_rating_before, expected.host_before);
        assert_eq!(actual.guest_rating_before, expected.guest_before);
        assert_eq!(actual.delta, expected.delta);
        assert_eq!(actual.host_rating_after, expected.host_after);
        assert_eq!(actual.guest_rating_after, expected.guest_after);
    }
    for (agent, expected) in expected_ratings {
        assert_eq!(run.ratings[&AgentId::new(agent).unwrap()], expected);
    }
}

#[derive(Debug)]
struct ExpectedUpdate<'a> {
    sequence: u64,
    pair_id: u64,
    host: &'a str,
    guest: &'a str,
    host_before: i32,
    guest_before: i32,
    delta: i32,
    host_after: i32,
    guest_after: i32,
}

fn parse_expected(text: &str) -> (Vec<ExpectedUpdate<'_>>, BTreeMap<&str, i32>) {
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("PAISHO-SITE-ELO-INDEPENDENT-REFERENCE\t1")
    );
    let mut updates = Vec::new();
    let mut ratings = BTreeMap::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        match fields[0] {
            "update" => updates.push(ExpectedUpdate {
                sequence: fields[1].parse().unwrap(),
                pair_id: fields[2].parse().unwrap(),
                host: fields[3],
                guest: fields[4],
                host_before: fields[5].parse().unwrap(),
                guest_before: fields[6].parse().unwrap(),
                delta: fields[7].parse().unwrap(),
                host_after: fields[8].parse().unwrap(),
                guest_after: fields[9].parse().unwrap(),
            }),
            "rating" => {
                ratings.insert(fields[1], fields[2].parse().unwrap());
            }
            other => panic!("unknown expected row {other}"),
        }
    }
    (updates, ratings)
}

fn parse_games(text: &str) -> Vec<RatedGame> {
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("sequence\tpair_id\thost\tguest\toutcome")
    );
    lines
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            let outcome = match fields[4] {
                "H" => RatedOutcome::HostWin,
                "D" => RatedOutcome::Draw,
                "G" => RatedOutcome::GuestWin,
                other => panic!("unknown fixture outcome {other}"),
            };
            RatedGame::new(
                fields[0].parse().unwrap(),
                fields[1].parse().unwrap(),
                AgentId::new(fields[2]).unwrap(),
                AgentId::new(fields[3]).unwrap(),
                outcome,
            )
            .unwrap()
        })
        .collect()
}
