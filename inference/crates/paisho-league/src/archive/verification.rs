use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

use paisho_ai::{AgentTelemetry, MatchResult, MatchTermination};
use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId};

use crate::{LeagueExecution, PlayedGame};

use super::{
    agents_text, games_and_telemetry_text, internal_ratings_text, invalid_data, pairwise_text,
    run_text, schedule_text, site_ratings_text, site_updates_text, validate_before_write,
    ArchiveMetadata,
};

const TELEMETRY_HEADER: &str = "sequence\trole\tagent_id\tdecisions\tsimulations\tevaluated_actions\texpanded_nodes\tgenerated_nodes\tgenerated_actions\tmaximum_search_depth\tmaximum_search_trees\tmaximum_search_workers\tmaximum_search_worker_capacity\tmaximum_action_ranking_workers\tmaximum_action_ranking_worker_capacity\trollout_steps";
const GAMES_HEADER: &str = "sequence\tgame_id\tpair_id\trating_eligible\thost_result\ttermination\tdecisions\trecord\terror";

pub(super) fn verify_archive_semantics(directory: &Path) -> io::Result<()> {
    super::verify_archive_structure(directory)?;

    let actual_run = read(directory, "run.tsv")?;
    let run = parse_run(&actual_run)?;
    let rule_profile = required(&run, "rule_profile")?
        .parse::<RuleProfileId>()
        .map_err(|error| invalid_data(format!("unsupported archive rule profile: {error}")))?;
    let metadata = ArchiveMetadata {
        source_revision: required(&run, "source_revision")?.to_owned(),
        pairs_per_edge: parse(required(&run, "pairs_per_edge")?, "pairs per edge")?,
        first_pair_id: parse(required(&run, "pair_id_first")?, "first pair id")?,
        decision_soft_limit: parse(
            required(&run, "decision_soft_limit")?,
            "decision soft limit",
        )?,
        site_initial_rating: parse(
            required(&run, "site_initial_rating")?,
            "site initial rating",
        )?,
        site_initialization_status: required(&run, "site_initialization_status")?.to_owned(),
    };
    let agents =
        crate::agents::agent_definitions_for_rules(&metadata.source_revision, rule_profile);
    compare(
        "agents.tsv",
        &read(directory, "agents.tsv")?,
        &agents_text(&agents),
    )?;

    let schedule =
        crate::build_ladder_schedule(&agents, metadata.pairs_per_edge, metadata.first_pair_id)
            .map_err(|error| invalid_data(format!("cannot reconstruct schedule: {error}")))?;
    compare(
        "schedule.tsv",
        &read(directory, "schedule.tsv")?,
        &schedule_text(&agents, &schedule),
    )?;

    let telemetry = parse_telemetry(&read(directory, "telemetry.tsv")?)?;
    let games = parse_games(
        directory,
        &read(directory, "games.tsv")?,
        &schedule,
        &agents,
        telemetry,
    )?;
    let execution = LeagueExecution {
        games,
        elapsed_seconds: parse(required(&run, "elapsed_seconds")?, "elapsed seconds")?,
        observed_match_workers: parse(
            required(&run, "observed_match_workers")?,
            "observed workers",
        )?,
        worker_capacity: parse(required(&run, "worker_capacity")?, "worker capacity")?,
        available_parallelism: parse(
            required(&run, "available_parallelism")?,
            "available parallelism",
        )?,
    };
    let analysis = crate::analyze(&execution, &agents, metadata.site_initial_rating)
        .map_err(|error| invalid_data(format!("cannot reproduce analysis: {error}")))?;
    validate_before_write(
        &metadata,
        &agents,
        &schedule,
        &execution,
        &analysis,
        rule_profile,
    )?;

    compare(
        "run.tsv",
        &actual_run,
        &run_text(&metadata, &schedule, &execution, &analysis, rule_profile),
    )?;
    let (expected_games, expected_telemetry) =
        games_and_telemetry_text(&agents, &execution, &analysis);
    compare("games.tsv", &read(directory, "games.tsv")?, &expected_games)?;
    compare(
        "telemetry.tsv",
        &read(directory, "telemetry.tsv")?,
        &expected_telemetry,
    )?;
    compare(
        "pairwise.tsv",
        &read(directory, "pairwise.tsv")?,
        &pairwise_text(&execution, &agents),
    )?;
    compare(
        "ratings-internal.tsv",
        &read(directory, "ratings-internal.tsv")?,
        &internal_ratings_text(&agents, &analysis),
    )?;
    compare(
        "ratings-garden-gate.tsv",
        &read(directory, "ratings-garden-gate.tsv")?,
        &site_ratings_text(&metadata, &agents, &analysis),
    )?;
    compare_site_updates(
        "ratings-garden-gate-updates.tsv",
        &read(directory, "ratings-garden-gate-updates.tsv")?,
        &site_updates_text(&analysis.site_order_runs),
    )
}

fn parse_run(text: &str) -> io::Result<BTreeMap<String, String>> {
    let mut lines = text.lines();
    if lines.next() != Some("PAISHO-RATING-LEAGUE\t1") {
        return Err(invalid_data("unsupported or missing league run signature"));
    }
    let mut values = BTreeMap::new();
    for line in lines {
        let (key, value) = line
            .split_once('\t')
            .ok_or_else(|| invalid_data("malformed run.tsv row"))?;
        if key.is_empty() || values.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(invalid_data("empty or duplicate run.tsv key"));
        }
    }
    Ok(values)
}

fn parse_telemetry(text: &str) -> io::Result<BTreeMap<(u64, &'static str), ArchivedTelemetry>> {
    let mut lines = text.lines();
    if lines.next() != Some(TELEMETRY_HEADER) {
        return Err(invalid_data("unexpected telemetry.tsv header"));
    }
    let mut values = BTreeMap::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 16 {
            return Err(invalid_data("malformed telemetry.tsv row"));
        }
        let sequence = parse(fields[0], "telemetry sequence")?;
        let role = match fields[1] {
            "Host" => "Host",
            "Guest" => "Guest",
            _ => return Err(invalid_data("unknown telemetry role")),
        };
        let telemetry = AgentTelemetry {
            decisions: parse(fields[3], "telemetry decisions")?,
            simulations: parse(fields[4], "telemetry simulations")?,
            evaluated_actions: parse(fields[5], "telemetry evaluated actions")?,
            expanded_nodes: parse(fields[6], "telemetry expanded nodes")?,
            generated_nodes: parse(fields[7], "telemetry generated nodes")?,
            generated_actions: parse(fields[8], "telemetry generated actions")?,
            maximum_search_depth: parse(fields[9], "telemetry search depth")?,
            maximum_search_trees: parse(fields[10], "telemetry search trees")?,
            maximum_search_workers: parse(fields[11], "telemetry search workers")?,
            maximum_search_worker_capacity: parse(fields[12], "telemetry worker capacity")?,
            maximum_action_ranking_workers: parse(fields[13], "ranking workers")?,
            maximum_action_ranking_worker_capacity: parse(fields[14], "ranking capacity")?,
            rollout_steps: parse(fields[15], "telemetry rollout steps")?,
        };
        let archived = ArchivedTelemetry {
            agent_id: fields[2].to_owned(),
            telemetry,
        };
        if values.insert((sequence, role), archived).is_some() {
            return Err(invalid_data("duplicate telemetry row"));
        }
    }
    Ok(values)
}

fn parse_games(
    directory: &Path,
    text: &str,
    schedule: &[crate::ScheduledGame],
    agents: &[crate::AgentDefinition],
    mut telemetry: BTreeMap<(u64, &'static str), ArchivedTelemetry>,
) -> io::Result<Vec<PlayedGame>> {
    let mut lines = text.lines();
    if lines.next() != Some(GAMES_HEADER) {
        return Err(invalid_data("unexpected games.tsv header"));
    }
    let rows: Vec<_> = lines.collect();
    if rows.len() != schedule.len() {
        return Err(invalid_data(
            "games.tsv length differs from reconstructed schedule",
        ));
    }
    let mut games = Vec::with_capacity(schedule.len());
    for (line, scheduled) in rows.into_iter().zip(schedule) {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 9 {
            return Err(invalid_data("malformed games.tsv row"));
        }
        if parse::<u64>(fields[0], "game sequence")? != scheduled.sequence
            || parse::<u64>(fields[1], "game id")? != scheduled.game_id
            || parse::<u64>(fields[2], "game pair id")? != scheduled.pair_id
        {
            return Err(invalid_data("games.tsv identifiers differ from schedule"));
        }
        if !matches!(fields[3], "0" | "1") {
            return Err(invalid_data("invalid rating eligibility flag"));
        }
        if fields[7] == "-" || fields[8] != "-" {
            return Err(invalid_data(
                "sealed rating archives require a replayable successful game",
            ));
        }
        let expected_record = format!("records/game-{:020}.psr", scheduled.game_id);
        if fields[7] != expected_record {
            return Err(invalid_data("unexpected game-record path"));
        }
        let record_text = fs::read_to_string(directory.join(&expected_record))?;
        let record: GameRecord = record_text
            .parse()
            .map_err(|error| invalid_data(format!("record cannot parse: {error}")))?;
        if record.to_string() != record_text {
            return Err(invalid_data("game record is not in canonical text form"));
        }
        if record.setup() != scheduled.setup
            || record.actions().len() != parse::<usize>(fields[6], "decision count")?
        {
            return Err(invalid_data(
                "game record differs from schedule or decision count",
            ));
        }
        let final_position = record
            .replay()
            .map_err(|error| invalid_data(format!("record cannot replay: {error}")))?;
        let termination = parse_termination(fields[4], fields[5], final_position.outcome())?;
        let host = telemetry
            .remove(&(scheduled.sequence, "Host"))
            .ok_or_else(|| invalid_data("missing host telemetry"))?;
        let guest = telemetry
            .remove(&(scheduled.sequence, "Guest"))
            .ok_or_else(|| invalid_data("missing guest telemetry"))?;
        if host.agent_id != agents[scheduled.host_agent].id().as_str()
            || guest.agent_id != agents[scheduled.guest_agent].id().as_str()
        {
            return Err(invalid_data("telemetry agent differs from scheduled agent"));
        }
        games.push(PlayedGame {
            scheduled: *scheduled,
            result: Ok(MatchResult {
                task_id: scheduled.game_id,
                final_position,
                record,
                termination,
                host_telemetry: host.telemetry,
                guest_telemetry: guest.telemetry,
            }),
        });
    }
    if !telemetry.is_empty() {
        return Err(invalid_data("telemetry.tsv has unreferenced rows"));
    }
    Ok(games)
}

fn parse_termination(
    result: &str,
    termination: &str,
    replayed: GameOutcome,
) -> io::Result<MatchTermination> {
    match (result, termination, replayed) {
        ("H", "HOST_WIN", GameOutcome::Win(Player::Host)) => Ok(MatchTermination::Rules(replayed)),
        ("G", "GUEST_WIN", GameOutcome::Win(Player::Guest)) => {
            Ok(MatchTermination::Rules(replayed))
        }
        ("D", "DRAW", GameOutcome::Draw) => Ok(MatchTermination::Rules(replayed)),
        ("U", "DECISION_LIMIT", GameOutcome::Ongoing) => Ok(MatchTermination::DecisionLimit),
        _ => Err(invalid_data(
            "game result, termination and replayed outcome disagree",
        )),
    }
}

fn read(directory: &Path, name: &str) -> io::Result<String> {
    fs::read_to_string(directory.join(name))
}

fn compare(name: &str, actual: &str, expected: &str) -> io::Result<()> {
    if actual == expected {
        return Ok(());
    }
    let first_difference = actual
        .lines()
        .zip(expected.lines())
        .position(|(actual, expected)| actual != expected)
        .map(|index| index + 1)
        .unwrap_or_else(|| actual.lines().count().min(expected.lines().count()) + 1);
    Err(invalid_data(format!(
        "{name} differs from deterministic semantic recomputation at line {first_difference}"
    )))
}

fn compare_site_updates(name: &str, actual: &str, expected: &str) -> io::Result<()> {
    let actual_lines: Vec<_> = actual.lines().collect();
    let expected_lines: Vec<_> = expected.lines().collect();
    if actual_lines.len() != expected_lines.len() || actual_lines.first() != expected_lines.first()
    {
        return compare(name, actual, expected);
    }
    for (line_index, (actual_line, expected_line)) in
        actual_lines.iter().zip(&expected_lines).enumerate().skip(1)
    {
        let actual_fields: Vec<_> = actual_line.split('\t').collect();
        let expected_fields: Vec<_> = expected_line.split('\t').collect();
        if actual_fields.len() != 13 || expected_fields.len() != 13 {
            return Err(invalid_data(format!(
                "malformed {name} row at line {}",
                line_index + 1
            )));
        }
        for field_index in 0..13 {
            if field_index == 8 {
                // `powf` may differ by one ULP between build profiles. This
                // probability is diagnostic; the rounded delta stays exact.
                let actual_score: f64 = parse(actual_fields[field_index], "expected host score")?;
                let expected_score: f64 =
                    parse(expected_fields[field_index], "expected host score")?;
                if !actual_score.is_finite() || (actual_score - expected_score).abs() > 1e-15 {
                    return Err(invalid_data(format!(
                        "{name} differs from semantic recomputation at line {}, field expected_host",
                        line_index + 1
                    )));
                }
            } else if actual_fields[field_index] != expected_fields[field_index] {
                return Err(invalid_data(format!(
                    "{name} differs from semantic recomputation at line {}, field {}",
                    line_index + 1,
                    field_index + 1
                )));
            }
        }
    }
    Ok(())
}

fn required<'a>(values: &'a BTreeMap<String, String>, key: &str) -> io::Result<&'a str> {
    values
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| invalid_data(format!("run.tsv is missing {key}")))
}

fn parse<T>(text: &str, name: &str) -> io::Result<T>
where
    T: core::str::FromStr,
{
    text.parse()
        .map_err(|_| invalid_data(format!("invalid {name} `{text}`")))
}

struct ArchivedTelemetry {
    agent_id: String,
    telemetry: AgentTelemetry,
}
