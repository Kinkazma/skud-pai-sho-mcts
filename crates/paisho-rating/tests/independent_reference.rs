use std::collections::BTreeMap;

use paisho_rating::{fit_davidson, AgentId, DavidsonOptions, RatedGame, RatedOutcome};

#[test]
fn rust_fit_matches_independent_scipy_bfgs_fixture() {
    let games = parse_games(include_str!("fixtures/independent_reference_games.tsv"));
    let expected = parse_expected(include_str!("fixtures/independent_reference_expected.tsv"));
    let fit = fit_davidson(&games, DavidsonOptions::default()).unwrap();

    for rating in &fit.ratings {
        let name = format!("rating:{}", rating.agent);
        assert_close(rating.elo.estimate, expected[&name], 2e-6);
        assert_uncertainty(&expected, &name, rating.elo, 3e-6);
    }
    assert_close(
        fit.host_advantage_elo.unwrap().estimate,
        expected["host_advantage_elo"],
        2e-6,
    );
    assert_uncertainty(
        &expected,
        "host_advantage_elo",
        fit.host_advantage_elo.unwrap(),
        3e-6,
    );
    assert_close(
        fit.draw_log_weight.unwrap().estimate,
        expected["draw_log_weight"],
        2e-8,
    );
    assert_uncertainty(
        &expected,
        "draw_log_weight",
        fit.draw_log_weight.unwrap(),
        3e-8,
    );
    assert_close(fit.log_likelihood, expected["log_likelihood"], 2e-9);
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

fn parse_expected(text: &str) -> BTreeMap<String, f64> {
    let mut lines = text.lines();
    assert_eq!(
        lines.next(),
        Some("PAISHO-DAVIDSON-INDEPENDENT-REFERENCE\t2")
    );
    let mut values = BTreeMap::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        match fields[0] {
            "rating" => {
                values.insert(format!("rating:{}", fields[1]), fields[2].parse().unwrap());
            }
            "uncertainty" => {
                for (field, value) in [
                    ("estimate", fields[2]),
                    ("model_se", fields[3]),
                    ("model_low", fields[4]),
                    ("model_high", fields[5]),
                    ("cluster_se", fields[6]),
                    ("cluster_low", fields[7]),
                    ("cluster_high", fields[8]),
                ] {
                    values.insert(
                        format!("uncertainty:{}:{field}", fields[1]),
                        value.parse().unwrap(),
                    );
                }
            }
            name => {
                values.insert(name.to_owned(), fields[1].parse().unwrap());
            }
        }
    }
    values
}

fn assert_uncertainty(
    expected: &BTreeMap<String, f64>,
    name: &str,
    estimate: paisho_rating::ParameterEstimate,
    tolerance: f64,
) {
    let cluster = estimate.paired_cluster.unwrap();
    for (field, actual) in [
        ("estimate", estimate.estimate),
        ("model_se", estimate.model.standard_error),
        ("model_low", estimate.model.interval_95.lower),
        ("model_high", estimate.model.interval_95.upper),
        ("cluster_se", cluster.standard_error),
        ("cluster_low", cluster.interval_95.lower),
        ("cluster_high", cluster.interval_95.upper),
    ] {
        assert_close(
            actual,
            expected[&format!("uncertainty:{name}:{field}")],
            tolerance,
        );
    }
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.12}, got {actual:.12}"
    );
}
