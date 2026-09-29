use core::fmt::Write as _;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use paisho_ai::{AgentTelemetry, MatchResult, MatchTermination};
use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId, TurnPhase};
use paisho_rating::{
    ParameterEstimate, RatedGame, SITE_ELO_K_FACTOR, SITE_ELO_SOURCE_COMMIT, SITE_ELO_SOURCE_PATH,
};
use sha2::{Digest, Sha256};

use crate::{
    summarize_pairwise, AgentDefinition, LeagueAnalysis, LeagueExecution, ScheduledGame,
    SiteOrderRun,
};

mod verification;

#[derive(Clone, Debug)]
pub struct ArchiveMetadata {
    pub source_revision: String,
    pub pairs_per_edge: usize,
    pub first_pair_id: u64,
    pub decision_soft_limit: usize,
    pub site_initial_rating: i32,
    pub site_initialization_status: String,
}

/// Checks the target and performs a create/sync/remove probe before a long run.
/// The full validation is repeated when the archive is finally written.
pub fn preflight_archive_path(directory: &Path) -> io::Result<()> {
    let (parent, partial) = archive_target_paths(directory)?;
    fs::create_dir_all(parent)?;
    fs::create_dir(&partial)?;
    let sync_result = sync_directory(&partial);
    let remove_result = fs::remove_dir(&partial);
    sync_result.and(remove_result)
}

pub fn write_archive(
    directory: &Path,
    metadata: &ArchiveMetadata,
    agents: &[AgentDefinition],
    schedule: &[ScheduledGame],
    execution: &LeagueExecution,
    analysis: &LeagueAnalysis,
) -> io::Result<()> {
    if metadata.decision_soft_limit == 0 {
        return Err(invalid_data("decision soft limit must be positive"));
    }
    validate_before_write(
        metadata,
        agents,
        schedule,
        execution,
        analysis,
        RuleProfileId::CURRENT,
    )?;
    let (parent, partial) = archive_target_paths(directory)?;
    fs::create_dir_all(&parent)?;
    fs::create_dir(&partial)?;
    fs::create_dir(partial.join("records"))?;

    fs::write(
        partial.join("run.tsv"),
        run_text(
            metadata,
            schedule,
            execution,
            analysis,
            RuleProfileId::CURRENT,
        ),
    )?;
    fs::write(partial.join("agents.tsv"), agents_text(agents))?;
    fs::write(
        partial.join("schedule.tsv"),
        schedule_text(agents, schedule),
    )?;
    write_records(&partial, execution)?;
    let (games, telemetry) = games_and_telemetry_text(agents, execution, analysis);
    fs::write(partial.join("games.tsv"), games)?;
    fs::write(partial.join("telemetry.tsv"), telemetry)?;
    fs::write(
        partial.join("pairwise.tsv"),
        pairwise_text(execution, agents),
    )?;
    fs::write(
        partial.join("ratings-internal.tsv"),
        internal_ratings_text(agents, analysis),
    )?;
    fs::write(
        partial.join("ratings-garden-gate.tsv"),
        site_ratings_text(metadata, agents, analysis),
    )?;
    fs::write(
        partial.join("ratings-garden-gate-updates.tsv"),
        site_updates_text(&analysis.site_order_runs),
    )?;
    write_sha256_manifest(&partial)?;
    verify_archive_manifest(&partial)?;
    sync_archive_tree(&partial)?;
    fs::rename(partial, directory)?;
    sync_directory(&parent)
}

fn validate_before_write(
    metadata: &ArchiveMetadata,
    agents: &[AgentDefinition],
    schedule: &[ScheduledGame],
    execution: &LeagueExecution,
    analysis: &LeagueAnalysis,
    rule_profile: RuleProfileId,
) -> io::Result<()> {
    if agents != crate::agents::agent_definitions_for_rules(&metadata.source_revision, rule_profile)
    {
        return Err(invalid_data(
            "agent definitions differ from the sealed ladder profile",
        ));
    }
    let expected_schedule =
        crate::build_ladder_schedule(agents, metadata.pairs_per_edge, metadata.first_pair_id)
            .map_err(|error| invalid_data(format!("invalid archive schedule metadata: {error}")))?;
    if schedule != expected_schedule {
        return Err(invalid_data(
            "schedule differs from the sealed ladder profile and metadata",
        ));
    }
    if schedule.len() != execution.games.len() {
        return Err(invalid_data("schedule and execution lengths differ"));
    }
    for (scheduled, played) in schedule.iter().zip(&execution.games) {
        if *scheduled != played.scheduled {
            return Err(invalid_data("execution order differs from the schedule"));
        }
        let result = played.result.as_ref().map_err(|error| {
            invalid_data(format!(
                "game {} failed and cannot enter a sealed rating archive: {error}",
                scheduled.game_id
            ))
        })?;
        if result.task_id != scheduled.game_id {
            return Err(invalid_data("match task id differs from scheduled game id"));
        }
        if result.record.rules() != rule_profile {
            return Err(invalid_data(
                "record rules differ from the league rule profile",
            ));
        }
        if result.record.setup() != scheduled.setup {
            return Err(invalid_data("record setup differs from the schedule"));
        }
        let replayed = result.record.replay().map_err(|error| {
            invalid_data(format!(
                "game {} does not replay: {error}",
                scheduled.game_id
            ))
        })?;
        if replayed != result.final_position {
            return Err(invalid_data(format!(
                "game {} replays to a different position",
                scheduled.game_id
            )));
        }
        let decisions = result.record.actions().len();
        let (host_decisions, guest_decisions) = decision_counts(&result.record)?;
        if result.host_telemetry.decisions != host_decisions
            || result.guest_telemetry.decisions != guest_decisions
        {
            return Err(invalid_data(
                "per-player decision telemetry differs from the replay",
            ));
        }
        let within_limit = decisions <= metadata.decision_soft_limit
            || valid_bonus_grace(&result.record, metadata.decision_soft_limit);
        match result.termination {
            MatchTermination::Rules(outcome)
                if outcome != GameOutcome::Ongoing
                    && result.final_position.outcome() == outcome
                    && within_limit => {}
            MatchTermination::DecisionLimit
                if result.final_position.outcome() == GameOutcome::Ongoing
                    && result.final_position.phase() == TurnPhase::Main
                    && (decisions == metadata.decision_soft_limit
                        || valid_bonus_grace(&result.record, metadata.decision_soft_limit)) => {}
            _ => {
                return Err(invalid_data(
                    "match termination or decision count differs from the configured policy",
                ));
            }
        }
    }
    validate_parallel_execution(schedule.len(), execution)?;
    let recomputed = crate::analyze(execution, agents, metadata.site_initial_rating)
        .map_err(|error| invalid_data(format!("rating analysis cannot be reproduced: {error}")))?;
    if &recomputed != analysis {
        return Err(invalid_data(
            "supplied rating analysis differs from deterministic recomputation",
        ));
    }
    Ok(())
}

fn valid_bonus_grace(record: &GameRecord, decision_soft_limit: usize) -> bool {
    if record.actions().len() != decision_soft_limit.saturating_add(1) {
        return false;
    }
    let mut position = record.initial_position();
    for action in record.actions().iter().take(decision_soft_limit) {
        if position.apply(*action).is_err() {
            return false;
        }
    }
    position.outcome() == GameOutcome::Ongoing && position.phase() == TurnPhase::HarmonyBonus
}

fn decision_counts(record: &GameRecord) -> io::Result<(usize, usize)> {
    let mut position = record.initial_position();
    let mut counts = [0_usize; 2];
    for action in record.actions() {
        let player = position.to_move();
        position
            .apply(*action)
            .map_err(|error| invalid_data(format!("record decision cannot apply: {error}")))?;
        counts[player.index()] += 1;
    }
    Ok((counts[Player::Host.index()], counts[Player::Guest.index()]))
}

fn archive_target_paths(directory: &Path) -> io::Result<(PathBuf, PathBuf)> {
    if directory.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "the evidence directory already exists",
        ));
    }
    let file_name = directory.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "the evidence directory needs a final path component",
        )
    })?;
    let parent = directory
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_owned();
    let partial = parent.join(format!(".{}.partial", file_name.to_string_lossy()));
    if partial.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("stale partial evidence exists at {}", partial.display()),
        ));
    }
    Ok((parent, partial))
}

fn validate_parallel_execution(
    scheduled_games: usize,
    execution: &LeagueExecution,
) -> io::Result<()> {
    if execution.available_parallelism < 2
        || execution.worker_capacity != execution.available_parallelism
        || scheduled_games < execution.available_parallelism
        || execution.observed_match_workers != execution.available_parallelism
    {
        return Err(invalid_data(format!(
            "evidence run did not use every available CPU worker: observed={}, rayon_capacity={}, available={}, games={scheduled_games}",
            execution.observed_match_workers,
            execution.worker_capacity,
            execution.available_parallelism,
        )));
    }
    Ok(())
}

fn run_text(
    metadata: &ArchiveMetadata,
    schedule: &[ScheduledGame],
    execution: &LeagueExecution,
    analysis: &LeagueAnalysis,
    rule_profile: RuleProfileId,
) -> String {
    let successful = execution
        .games
        .iter()
        .filter(|game| game.result.is_ok())
        .count();
    let errors = execution.games.len() - successful;
    let interruptions = execution
        .games
        .iter()
        .filter(|game| {
            matches!(
                game.result,
                Ok(MatchResult {
                    termination: MatchTermination::DecisionLimit,
                    ..
                })
            )
        })
        .count();
    let mut text = String::new();
    writeln!(text, "PAISHO-RATING-LEAGUE\t1").unwrap();
    writeln!(text, "source_revision\t{}", metadata.source_revision).unwrap();
    writeln!(text, "rule_profile\t{rule_profile}").unwrap();
    writeln!(text, "schedule_profile\tladder-v1").unwrap();
    writeln!(text, "pair_id_first\t{}", metadata.first_pair_id).unwrap();
    writeln!(text, "pairs_per_edge\t{}", metadata.pairs_per_edge).unwrap();
    writeln!(text, "scheduled_games\t{}", schedule.len()).unwrap();
    writeln!(text, "successful_games\t{successful}").unwrap();
    writeln!(text, "match_errors\t{errors}").unwrap();
    writeln!(text, "decision_limit_interruptions\t{interruptions}").unwrap();
    writeln!(text, "rated_games\t{}", analysis.rated_games.len()).unwrap();
    writeln!(text, "rated_pairs\t{}", analysis.rated_games.len() / 2).unwrap();
    writeln!(text, "excluded_pairs\t{}", analysis.excluded_pairs.len()).unwrap();
    writeln!(
        text,
        "decision_soft_limit\t{}",
        metadata.decision_soft_limit
    )
    .unwrap();
    writeln!(
        text,
        "decision_limit_policy\tcomplete-in-flight-harmony-bonus"
    )
    .unwrap();
    writeln!(text, "opening_schedule\tsix-basic-flowers-per-edge-offset").unwrap();
    writeln!(
        text,
        "pairing_policy\treversed-seats-same-setup-and-seat-seeds"
    )
    .unwrap();
    writeln!(
        text,
        "observed_match_workers\t{}",
        execution.observed_match_workers
    )
    .unwrap();
    writeln!(text, "worker_capacity\t{}", execution.worker_capacity).unwrap();
    writeln!(
        text,
        "available_parallelism\t{}",
        execution.available_parallelism
    )
    .unwrap();
    writeln!(text, "elapsed_seconds\t{:.6}", execution.elapsed_seconds).unwrap();
    writeln!(text, "internal_rating_model\tBradley-Terry-Davidson-MLE").unwrap();
    writeln!(text, "internal_elo_scale\t400").unwrap();
    writeln!(text, "internal_elo_center\t1500").unwrap();
    writeln!(
        text,
        "internal_rating_label\tElo interne non calibre au site"
    )
    .unwrap();
    writeln!(
        text,
        "internal_fit_status\t{}",
        if analysis.internal_fit.is_ok() {
            "ok-provisional"
        } else {
            "error"
        }
    )
    .unwrap();
    if let Err(error) = &analysis.internal_fit {
        writeln!(text, "internal_fit_error\t{}", sanitize(error)).unwrap();
    }
    writeln!(
        text,
        "site_rating_label\tElo simule - formule The Garden Gate"
    )
    .unwrap();
    writeln!(
        text,
        "site_initial_rating\t{}",
        metadata.site_initial_rating
    )
    .unwrap();
    writeln!(
        text,
        "site_initialization_status\t{}",
        sanitize(&metadata.site_initialization_status)
    )
    .unwrap();
    writeln!(text, "site_k_factor\t{SITE_ELO_K_FACTOR}").unwrap();
    writeln!(text, "site_formula_source_commit\t{SITE_ELO_SOURCE_COMMIT}").unwrap();
    writeln!(text, "site_formula_source_path\t{SITE_ELO_SOURCE_PATH}").unwrap();
    text
}

fn agents_text(agents: &[AgentDefinition]) -> String {
    let mut text = String::from("alias\tagent_id\tfingerprint_sha256\tfamily\tdescriptor\n");
    for agent in agents {
        writeln!(
            text,
            "{}\t{}\t{}\t{}\t{}",
            agent.alias(),
            agent.id(),
            agent.fingerprint(),
            agent.family(),
            agent.descriptor()
        )
        .unwrap();
    }
    text
}

fn schedule_text(agents: &[AgentDefinition], schedule: &[ScheduledGame]) -> String {
    let mut text = String::from(
        "sequence\tgame_id\tpair_id\tedge\tleg\thost_agent_id\tguest_agent_id\tstarting_flower\thost_seed\tguest_seed\n",
    );
    for game in schedule {
        writeln!(
            text,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            game.sequence,
            game.game_id,
            game.pair_id,
            game.edge_index,
            game.leg,
            agents[game.host_agent].id(),
            agents[game.guest_agent].id(),
            game.setup.starting_flower.code(),
            game.host_seed,
            game.guest_seed,
        )
        .unwrap();
    }
    text
}

fn write_records(directory: &Path, execution: &LeagueExecution) -> io::Result<()> {
    for played in &execution.games {
        if let Ok(result) = &played.result {
            let record_name = format!("game-{:020}.psr", played.scheduled.game_id);
            fs::write(
                directory.join("records").join(record_name),
                result.record.to_string(),
            )?;
        }
    }
    Ok(())
}

fn games_and_telemetry_text(
    agents: &[AgentDefinition],
    execution: &LeagueExecution,
    analysis: &LeagueAnalysis,
) -> (String, String) {
    let rated_sequences: BTreeSet<_> = analysis
        .rated_games
        .iter()
        .map(RatedGame::sequence)
        .collect();
    let mut games = String::from(
        "sequence\tgame_id\tpair_id\trating_eligible\thost_result\ttermination\tdecisions\trecord\terror\n",
    );
    let mut telemetry = String::from(
        "sequence\trole\tagent_id\tdecisions\tsimulations\tevaluated_actions\texpanded_nodes\tgenerated_nodes\tgenerated_actions\tmaximum_search_depth\tmaximum_search_trees\tmaximum_search_workers\tmaximum_search_worker_capacity\tmaximum_action_ranking_workers\tmaximum_action_ranking_worker_capacity\trollout_steps\n",
    );
    for played in &execution.games {
        match &played.result {
            Ok(result) => {
                let record_name = format!("game-{:020}.psr", played.scheduled.game_id);
                writeln!(
                    games,
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\trecords/{}\t-",
                    played.scheduled.sequence,
                    played.scheduled.game_id,
                    played.scheduled.pair_id,
                    usize::from(rated_sequences.contains(&played.scheduled.sequence)),
                    host_result_code(result),
                    termination_code(result.termination),
                    result.record.actions().len(),
                    record_name,
                )
                .unwrap();
                append_telemetry(
                    &mut telemetry,
                    played.scheduled.sequence,
                    "Host",
                    &agents[played.scheduled.host_agent].id().to_string(),
                    result.host_telemetry,
                );
                append_telemetry(
                    &mut telemetry,
                    played.scheduled.sequence,
                    "Guest",
                    &agents[played.scheduled.guest_agent].id().to_string(),
                    result.guest_telemetry,
                );
            }
            Err(error) => {
                writeln!(
                    games,
                    "{}\t{}\t{}\t0\tE\tERROR\t0\t-\t{}",
                    played.scheduled.sequence,
                    played.scheduled.game_id,
                    played.scheduled.pair_id,
                    sanitize(error),
                )
                .unwrap();
            }
        }
    }
    (games, telemetry)
}

fn append_telemetry(
    text: &mut String,
    sequence: u64,
    role: &str,
    agent_id: &str,
    value: AgentTelemetry,
) {
    writeln!(
        text,
        "{sequence}\t{role}\t{agent_id}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        value.decisions,
        value.simulations,
        value.evaluated_actions,
        value.expanded_nodes,
        value.generated_nodes,
        value.generated_actions,
        value.maximum_search_depth,
        value.maximum_search_trees,
        value.maximum_search_workers,
        value.maximum_search_worker_capacity,
        value.maximum_action_ranking_workers,
        value.maximum_action_ranking_worker_capacity,
        value.rollout_steps,
    )
    .unwrap();
}

fn internal_ratings_text(agents: &[AgentDefinition], analysis: &LeagueAnalysis) -> String {
    let mut text = String::new();
    writeln!(text, "PAISHO-INTERNAL-RATINGS\t1").unwrap();
    writeln!(text, "label\tElo interne non calibre au site").unwrap();
    let Ok(fit) = &analysis.internal_fit else {
        writeln!(text, "status\terror").unwrap();
        writeln!(
            text,
            "error\t{}",
            sanitize(analysis.internal_fit.as_ref().unwrap_err())
        )
        .unwrap();
        return text;
    };
    writeln!(text, "status\tok-provisional").unwrap();
    writeln!(text, "tie_model\t{:?}", fit.tie_model).unwrap();
    writeln!(text, "games\t{}", fit.games).unwrap();
    writeln!(text, "pairs\t{}", fit.pairs).unwrap();
    writeln!(text, "iterations\t{}", fit.iterations).unwrap();
    writeln!(text, "log_likelihood\t{:.12}", fit.log_likelihood).unwrap();
    writeln!(
        text,
        "information_condition_number\t{:.12}",
        fit.information_condition_number
    )
    .unwrap();
    if let Some(host) = fit.host_advantage_elo {
        append_parameter(&mut text, "host_advantage_elo", host);
    }
    if let Some(draw) = fit.draw_log_weight {
        append_parameter(&mut text, "draw_log_weight", draw);
        writeln!(text, "draw_weight\t{:.12}", fit.draw_weight()).unwrap();
    } else {
        writeln!(text, "draw_log_weight\tBOUNDARY_NEGATIVE_INFINITY").unwrap();
        writeln!(text, "draw_weight\t0").unwrap();
    }
    writeln!(
        text,
        "alias\tagent_id\telo\tmodel_se\tmodel_ci95_low\tmodel_ci95_high\tpaired_cluster_se\tpaired_cluster_ci95_low\tpaired_cluster_ci95_high\trated_games\trating_status"
    )
    .unwrap();
    let aliases: BTreeMap<_, _> = agents
        .iter()
        .map(|agent| (agent.id(), agent.alias()))
        .collect();
    for rating in &fit.ratings {
        let (cluster_se, cluster_low, cluster_high) = uncertainty_fields(rating.elo);
        writeln!(
            text,
            "{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\tprovisional",
            aliases[&rating.agent],
            rating.agent,
            rating.elo.estimate,
            rating.elo.model.standard_error,
            rating.elo.model.interval_95.lower,
            rating.elo.model.interval_95.upper,
            cluster_se,
            cluster_low,
            cluster_high,
            rating.rated_games,
        )
        .unwrap();
    }
    text
}

fn pairwise_text(execution: &LeagueExecution, agents: &[AgentDefinition]) -> String {
    let mut text = String::from(
        "edge\tfirst_alias\tsecond_alias\twins\tdraws\tlosses\tunfinished\tpentanomial_0\tpentanomial_0_5\tpentanomial_1\tpentanomial_1_5\tpentanomial_2\texcluded_pairs\tpaired_sign_test_two_sided_p\tpaired_sign_test_pessimistic_two_sided_p\n",
    );
    for summary in summarize_pairwise(execution, agents) {
        writeln!(
            text,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}",
            summary.edge_index,
            summary.first_alias,
            summary.second_alias,
            summary.wins,
            summary.draws,
            summary.losses,
            summary.unfinished,
            summary.paired.zero,
            summary.paired.half,
            summary.paired.one,
            summary.paired.one_and_half,
            summary.paired.two,
            summary.paired.excluded,
            summary.paired.exact_two_sided_sign_test_p_value(),
            summary
                .paired
                .pessimistic_exact_two_sided_sign_test_p_value(),
        )
        .unwrap();
    }
    text
}

fn append_parameter(text: &mut String, name: &str, parameter: ParameterEstimate) {
    let (cluster_se, cluster_low, cluster_high) = uncertainty_fields(parameter);
    writeln!(
        text,
        "{name}\t{:.9}\tmodel_se={:.9}\tmodel_ci95=[{:.9},{:.9}]\tpaired_cluster_se={}\tpaired_cluster_ci95=[{},{}]",
        parameter.estimate,
        parameter.model.standard_error,
        parameter.model.interval_95.lower,
        parameter.model.interval_95.upper,
        cluster_se,
        cluster_low,
        cluster_high,
    )
    .unwrap();
}

fn uncertainty_fields(parameter: ParameterEstimate) -> (String, String, String) {
    parameter.paired_cluster.map_or_else(
        || ("NA".to_owned(), "NA".to_owned(), "NA".to_owned()),
        |uncertainty| {
            (
                format!("{:.6}", uncertainty.standard_error),
                format!("{:.6}", uncertainty.interval_95.lower),
                format!("{:.6}", uncertainty.interval_95.upper),
            )
        },
    )
}

fn site_ratings_text(
    metadata: &ArchiveMetadata,
    agents: &[AgentDefinition],
    analysis: &LeagueAnalysis,
) -> String {
    let mut text = String::new();
    writeln!(text, "PAISHO-GARDEN-GATE-ELO-SIMULATION\t1").unwrap();
    writeln!(text, "label\tElo simule - formule The Garden Gate").unwrap();
    writeln!(text, "initial_rating\t{}", metadata.site_initial_rating).unwrap();
    writeln!(
        text,
        "initialization_status\t{}",
        sanitize(&metadata.site_initialization_status)
    )
    .unwrap();
    writeln!(text, "k_factor\t{SITE_ELO_K_FACTOR}").unwrap();
    writeln!(text, "formula_source_commit\t{SITE_ELO_SOURCE_COMMIT}").unwrap();
    writeln!(text, "formula_source_path\t{SITE_ELO_SOURCE_PATH}").unwrap();
    write!(text, "alias\tagent_id").unwrap();
    for order in &analysis.site_order_runs {
        write!(text, "\t{}", order.label).unwrap();
    }
    writeln!(
        text,
        "\tsampled_minimum\tsampled_maximum\tsampled_order_spread\trated_games\trating_status"
    )
    .unwrap();
    for agent in agents {
        write!(text, "{}\t{}", agent.alias(), agent.id()).unwrap();
        for order in &analysis.site_order_runs {
            write!(text, "\t{}", order.run.ratings[agent.id()]).unwrap();
        }
        let (minimum, maximum) = analysis.site_sampled_order_ranges[agent.id()];
        let rated_games = analysis.rated_game_counts[agent.id()];
        let status = if rated_games == 0 {
            "unrated-initial-value-only"
        } else {
            "simulated-provisional"
        };
        writeln!(
            text,
            "\t{minimum}\t{maximum}\t{}\t{rated_games}\t{status}",
            maximum - minimum
        )
        .unwrap();
    }
    text
}

fn site_updates_text(orders: &[SiteOrderRun]) -> String {
    let mut text = String::from(
        "order\tordinal\tsequence\tpair_id\thost_agent_id\tguest_agent_id\thost_before\tguest_before\texpected_host\tactual_host\tdelta\thost_after\tguest_after\n",
    );
    for order in orders {
        for (ordinal, update) in order.run.updates.iter().enumerate() {
            writeln!(
                text,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.17}\t{:.1}\t{}\t{}\t{}",
                order.label,
                ordinal,
                update.sequence,
                update.pair_id,
                update.host,
                update.guest,
                update.host_rating_before,
                update.guest_rating_before,
                update.expected_host_score,
                update.actual_host_score,
                update.delta,
                update.host_rating_after,
                update.guest_rating_after,
            )
            .unwrap();
        }
    }
    text
}

fn host_result_code(result: &MatchResult) -> &'static str {
    match result.scored_outcome() {
        Some(GameOutcome::Win(Player::Host)) => "H",
        Some(GameOutcome::Draw) => "D",
        Some(GameOutcome::Win(Player::Guest)) => "G",
        Some(GameOutcome::Ongoing) | None => "U",
    }
}

fn termination_code(termination: MatchTermination) -> &'static str {
    match termination {
        MatchTermination::DecisionLimit => "DECISION_LIMIT",
        MatchTermination::Rules(GameOutcome::Win(Player::Host)) => "HOST_WIN",
        MatchTermination::Rules(GameOutcome::Win(Player::Guest)) => "GUEST_WIN",
        MatchTermination::Rules(GameOutcome::Draw) => "DRAW",
        MatchTermination::Rules(GameOutcome::Ongoing) => "INVALID_ONGOING",
    }
}

fn sanitize(value: &impl ToString) -> String {
    value.to_string().replace(['\t', '\n', '\r'], " ")
}

fn sync_archive_tree(directory: &Path) -> io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            sync_archive_tree(&path)?;
        } else {
            fs::File::open(path)?.sync_all()?;
        }
    }
    sync_directory(directory)
}

fn sync_directory(directory: &Path) -> io::Result<()> {
    fs::File::open(directory)?.sync_all()
}

fn write_sha256_manifest(directory: &Path) -> io::Result<()> {
    let files = archive_files(directory)?;
    let mut manifest = String::new();
    for relative in files {
        let digest = sha256_hex(&directory.join(&relative))?;
        writeln!(manifest, "{digest}  {}", relative.display()).unwrap();
    }
    fs::write(directory.join("MANIFEST.sha256"), manifest)
}

pub fn verify_archive_manifest(directory: &Path) -> io::Result<()> {
    verification::verify_archive_semantics(directory)
}

fn verify_archive_structure(directory: &Path) -> io::Result<()> {
    let manifest_path = directory.join("MANIFEST.sha256");
    if !fs::symlink_metadata(&manifest_path)?.file_type().is_file() {
        return Err(invalid_data("MANIFEST.sha256 must be a regular file"));
    }
    let manifest = fs::read_to_string(manifest_path)?;
    let mut expected_paths = BTreeSet::new();
    for line in manifest.lines() {
        let (expected, relative_text) = line
            .split_once("  ")
            .ok_or_else(|| invalid_data("malformed SHA-256 manifest"))?;
        if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(invalid_data("malformed SHA-256 digest"));
        }
        let relative = PathBuf::from(relative_text);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(invalid_data("unsafe path in SHA-256 manifest"));
        }
        if !expected_paths.insert(relative.clone()) {
            return Err(invalid_data("duplicate path in SHA-256 manifest"));
        }
        let actual = sha256_hex(&directory.join(&relative))?;
        if actual != expected {
            return Err(invalid_data(format!(
                "SHA-256 mismatch for {relative_text}"
            )));
        }
    }
    let actual_paths: BTreeSet<_> = archive_files(directory)?.into_iter().collect();
    if expected_paths != actual_paths {
        return Err(invalid_data(
            "manifest file set differs from archive contents",
        ));
    }
    verify_archive_contents(directory, &actual_paths)
}

fn verify_archive_contents(directory: &Path, paths: &BTreeSet<PathBuf>) -> io::Result<()> {
    const REQUIRED: [&str; 9] = [
        "run.tsv",
        "agents.tsv",
        "schedule.tsv",
        "games.tsv",
        "telemetry.tsv",
        "pairwise.tsv",
        "ratings-internal.tsv",
        "ratings-garden-gate.tsv",
        "ratings-garden-gate-updates.tsv",
    ];
    for required in REQUIRED {
        if !paths.contains(Path::new(required)) {
            return Err(invalid_data(format!(
                "archive is missing required file {required}"
            )));
        }
    }
    let run = fs::read_to_string(directory.join("run.tsv"))?;
    if run.lines().next() != Some("PAISHO-RATING-LEAGUE\t1") {
        return Err(invalid_data("unsupported or missing league run signature"));
    }

    let schedule = parse_schedule_index(&fs::read_to_string(directory.join("schedule.tsv"))?)?;
    let games = fs::read_to_string(directory.join("games.tsv"))?;
    let mut lines = games.lines();
    if lines.next()
        != Some(
            "sequence\tgame_id\tpair_id\trating_eligible\thost_result\ttermination\tdecisions\trecord\terror",
        )
    {
        return Err(invalid_data("unexpected games.tsv header"));
    }
    let mut game_index = Vec::new();
    let mut referenced_records = BTreeSet::new();
    let mut sequences = BTreeSet::new();
    let mut game_ids = BTreeSet::new();
    for line in lines {
        let fields: Vec<_> = line.split('\t').collect();
        if fields.len() != 9 {
            return Err(invalid_data("malformed games.tsv row"));
        }
        let sequence = parse_field::<u64>(fields[0], "game sequence")?;
        let game_id = parse_field::<u64>(fields[1], "game id")?;
        let pair_id = parse_field::<u64>(fields[2], "pair id")?;
        let eligible = match fields[3] {
            "0" => false,
            "1" => true,
            _ => return Err(invalid_data("invalid rating eligibility flag")),
        };
        if !sequences.insert(sequence) || !game_ids.insert(game_id) {
            return Err(invalid_data("duplicate game sequence or id"));
        }
        game_index.push((sequence, game_id, pair_id));
        if fields[7] == "-" {
            if fields[4] != "E"
                || fields[5] != "ERROR"
                || fields[6] != "0"
                || fields[8] == "-"
                || eligible
            {
                return Err(invalid_data("malformed failed-game archive row"));
            }
            continue;
        }
        if fields[8] != "-" {
            return Err(invalid_data("successful game unexpectedly has an error"));
        }
        let decisions = parse_field::<usize>(fields[6], "decision count")?;
        let relative = safe_record_path(fields[7])?;
        if !referenced_records.insert(relative.clone()) {
            return Err(invalid_data("a game record is referenced more than once"));
        }
        let record: GameRecord = fs::read_to_string(directory.join(&relative))?
            .parse()
            .map_err(|error| {
                invalid_data(format!(
                    "record {} cannot parse: {error}",
                    relative.display()
                ))
            })?;
        if record.actions().len() != decisions {
            return Err(invalid_data(format!(
                "record {} decision count differs from games.tsv",
                relative.display()
            )));
        }
        let final_position = record.replay().map_err(|error| {
            invalid_data(format!(
                "record {} cannot replay: {error}",
                relative.display()
            ))
        })?;
        verify_archived_outcome(fields[4], fields[5], eligible, final_position.outcome())?;
    }
    if game_index.is_empty() || game_index != schedule {
        return Err(invalid_data(
            "games.tsv index is empty or differs from schedule.tsv",
        ));
    }
    let actual_records: BTreeSet<_> = paths
        .iter()
        .filter(|path| path.starts_with("records"))
        .cloned()
        .collect();
    if referenced_records != actual_records {
        return Err(invalid_data(
            "record files differ from successful games.tsv references",
        ));
    }
    Ok(())
}

fn parse_schedule_index(text: &str) -> io::Result<Vec<(u64, u64, u64)>> {
    let mut lines = text.lines();
    if lines.next()
        != Some(
            "sequence\tgame_id\tpair_id\tedge\tleg\thost_agent_id\tguest_agent_id\tstarting_flower\thost_seed\tguest_seed",
        )
    {
        return Err(invalid_data("unexpected schedule.tsv header"));
    }
    lines
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            if fields.len() != 10 {
                return Err(invalid_data("malformed schedule.tsv row"));
            }
            Ok((
                parse_field(fields[0], "schedule sequence")?,
                parse_field(fields[1], "schedule game id")?,
                parse_field(fields[2], "schedule pair id")?,
            ))
        })
        .collect()
}

fn safe_record_path(text: &str) -> io::Result<PathBuf> {
    let path = PathBuf::from(text);
    let mut components = path.components();
    if components.next() != Some(Component::Normal("records".as_ref()))
        || components.any(|component| !matches!(component, Component::Normal(_)))
        || path.extension().and_then(|value| value.to_str()) != Some("psr")
    {
        return Err(invalid_data("unsafe or invalid game-record path"));
    }
    Ok(path)
}

fn verify_archived_outcome(
    result: &str,
    termination: &str,
    eligible: bool,
    outcome: GameOutcome,
) -> io::Result<()> {
    // A terminal leg can be ineligible because its reversed partner hit the
    // decision limit; whole-pair eligibility is checked by semantic analysis.
    let valid = match (result, termination, outcome) {
        ("H", "HOST_WIN", GameOutcome::Win(Player::Host))
        | ("G", "GUEST_WIN", GameOutcome::Win(Player::Guest))
        | ("D", "DRAW", GameOutcome::Draw) => true,
        ("U", "DECISION_LIMIT", GameOutcome::Ongoing) => !eligible,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(invalid_data(
            "archived result, termination, eligibility and replay disagree",
        ))
    }
}

fn parse_field<T>(text: &str, name: &str) -> io::Result<T>
where
    T: core::str::FromStr,
{
    text.parse()
        .map_err(|_| invalid_data(format!("invalid {name} `{text}`")))
}

fn archive_files(directory: &Path) -> io::Result<Vec<PathBuf>> {
    fn visit(root: &Path, relative: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
        for entry in fs::read_dir(root.join(relative))? {
            let entry = entry?;
            let child = relative.join(entry.file_name());
            let file_type = entry.file_type()?;
            if file_type.is_symlink() {
                return Err(invalid_data(format!(
                    "archive contains symbolic link {}",
                    child.display()
                )));
            }
            if file_type.is_dir() {
                visit(root, &child, files)?;
            } else if file_type.is_file() && child != Path::new("MANIFEST.sha256") {
                files.push(child);
            } else if !file_type.is_file() {
                return Err(invalid_data(format!(
                    "archive contains a non-regular entry {}",
                    child.display()
                )));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    visit(directory, Path::new(""), &mut files)?;
    files.sort();
    Ok(files)
}

fn sha256_hex(path: &Path) -> io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(digest, "{byte:02x}").unwrap();
    }
    Ok(digest)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use paisho_core::{legal_actions, Action, Position, TurnPhase};
    use paisho_rating::RatedOutcome;

    use super::*;

    fn temporary_path(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "paisho-league-{label}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn archive_preflight_leaves_no_target_or_partial_directory() {
        let target = temporary_path("preflight").join("evidence");
        preflight_archive_path(&target).unwrap();
        assert!(!target.exists());
        assert!(!target.with_file_name(".evidence.partial").exists());
        fs::create_dir(&target).unwrap();
        assert_eq!(
            preflight_archive_path(&target).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        fs::remove_dir_all(target.parent().unwrap()).unwrap();
    }

    #[test]
    fn evidence_requires_every_available_worker_and_more_than_one() {
        let valid = LeagueExecution {
            games: Vec::new(),
            elapsed_seconds: 0.0,
            observed_match_workers: 10,
            worker_capacity: 10,
            available_parallelism: 10,
        };
        validate_parallel_execution(20, &valid).unwrap();
        let single_core = LeagueExecution {
            games: Vec::new(),
            elapsed_seconds: 0.0,
            observed_match_workers: 1,
            worker_capacity: 1,
            available_parallelism: 1,
        };
        assert!(validate_parallel_execution(20, &single_core).is_err());
        let underused = LeagueExecution {
            observed_match_workers: 9,
            ..valid
        };
        assert!(validate_parallel_execution(20, &underused).is_err());
    }

    #[test]
    fn a_second_main_action_is_not_mistaken_for_bonus_grace() {
        let setup = paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3);
        let mut position = Position::from_standard_setup(setup);
        let mut record = GameRecord::new(setup);
        for _ in 0..2 {
            let action = legal_actions(&position)
                .into_iter()
                .find(|action| matches!(action, Action::Plant { .. }))
                .unwrap();
            position.apply(action).unwrap();
            record.push(action);
        }
        assert_eq!(position.phase(), TurnPhase::Main);
        assert!(!valid_bonus_grace(&record, 1));
    }

    #[test]
    fn complete_writer_output_verifies_and_replays() {
        let target = temporary_path("writer").join("evidence");
        let agents = crate::default_agent_definitions("revision");
        let schedule = crate::build_ladder_schedule(&agents, 1, 12_000).unwrap();
        let games = schedule
            .iter()
            .map(|scheduled| {
                let mut final_position = Position::from_standard_setup(scheduled.setup);
                let player = final_position.to_move();
                let action = legal_actions(&final_position)
                    .into_iter()
                    .find(|action| matches!(action, Action::Plant { .. }))
                    .unwrap();
                final_position.apply(action).unwrap();
                assert_eq!(final_position.phase(), TurnPhase::Main);
                let mut record = GameRecord::new(scheduled.setup);
                record.push(action);
                let mut host_telemetry = AgentTelemetry::default();
                let mut guest_telemetry = AgentTelemetry::default();
                match player {
                    Player::Host => host_telemetry.decisions = 1,
                    Player::Guest => guest_telemetry.decisions = 1,
                }
                crate::PlayedGame {
                    scheduled: *scheduled,
                    result: Ok(MatchResult {
                        task_id: scheduled.game_id,
                        final_position,
                        record,
                        termination: MatchTermination::DecisionLimit,
                        host_telemetry,
                        guest_telemetry,
                    }),
                }
            })
            .collect();
        let mut execution = LeagueExecution {
            games,
            elapsed_seconds: 0.001,
            observed_match_workers: 2,
            worker_capacity: 2,
            available_parallelism: 2,
        };
        let analysis = crate::analyze(&execution, &agents, 1_000).unwrap();
        let metadata = ArchiveMetadata {
            source_revision: "revision".to_owned(),
            pairs_per_edge: 1,
            first_pair_id: 12_000,
            decision_soft_limit: 1,
            site_initial_rating: 1_000,
            site_initialization_status: "test\tfixture\nstatus".to_owned(),
        };
        write_archive(
            &target, &metadata, &agents, &schedule, &execution, &analysis,
        )
        .unwrap();
        verify_archive_manifest(&target).unwrap();
        assert!(fs::read_to_string(target.join("run.tsv"))
            .unwrap()
            .contains(&format!("rule_profile\t{}\n", RuleProfileId::CURRENT)));

        let first = execution.games[0].result.as_mut().unwrap();
        let old = GameRecord::with_rules(first.record.setup(), RuleProfileId::SkudPaiSho2022);
        first.record = old;
        first.final_position = first.record.replay().unwrap();
        let error = validate_before_write(
            &metadata,
            &agents,
            &schedule,
            &execution,
            &analysis,
            RuleProfileId::CURRENT,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "record rules differ from the league rule profile"
        );
        assert_eq!(archive_files(&target).unwrap().len(), 19);
        assert!(fs::read_to_string(target.join("ratings-garden-gate.tsv"))
            .unwrap()
            .contains("unrated-initial-value-only"));

        let telemetry_path = target.join("telemetry.tsv");
        let original_telemetry = fs::read_to_string(&telemetry_path).unwrap();
        let mut lines: Vec<Vec<String>> = original_telemetry
            .lines()
            .map(|line| line.split('\t').map(str::to_owned).collect())
            .collect();
        let first_decisions = lines[1][3].clone();
        lines[1][3] = lines[2][3].clone();
        lines[2][3] = first_decisions;
        let swapped = lines
            .into_iter()
            .map(|fields| fields.join("\t"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        fs::write(&telemetry_path, swapped).unwrap();
        write_sha256_manifest(&target).unwrap();
        assert!(verify_archive_manifest(&target).is_err());
        fs::write(&telemetry_path, original_telemetry).unwrap();

        fs::write(target.join("ratings-internal.tsv"), "forged\n").unwrap();
        write_sha256_manifest(&target).unwrap();
        assert!(verify_archive_manifest(&target).is_err());
        fs::remove_dir_all(target.parent().unwrap()).unwrap();
    }

    #[test]
    fn manifest_is_relative_complete_and_detects_extra_files() {
        let root = temporary_path("manifest");
        fs::create_dir_all(root.join("records")).unwrap();
        let record_text =
            include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
        let record: GameRecord = record_text.parse().unwrap();
        let final_position = record.replay().unwrap();
        let (host_result, termination) = match final_position.outcome() {
            GameOutcome::Win(Player::Host) => ("H", "HOST_WIN"),
            GameOutcome::Win(Player::Guest) => ("G", "GUEST_WIN"),
            GameOutcome::Draw => ("D", "DRAW"),
            GameOutcome::Ongoing => panic!("the fixture must be terminal"),
        };
        fs::write(root.join("run.tsv"), "PAISHO-RATING-LEAGUE\t1\n").unwrap();
        fs::write(root.join("records/game.psr"), record_text).unwrap();
        fs::write(
            root.join("schedule.tsv"),
            "sequence\tgame_id\tpair_id\tedge\tleg\thost_agent_id\tguest_agent_id\tstarting_flower\thost_seed\tguest_seed\n20\t20\t10\t0\t0\ta\tb\tR4\t7\t9\n",
        )
        .unwrap();
        fs::write(
            root.join("games.tsv"),
            format!(
                "sequence\tgame_id\tpair_id\trating_eligible\thost_result\ttermination\tdecisions\trecord\terror\n20\t20\t10\t1\t{host_result}\t{termination}\t{}\trecords/game.psr\t-\n",
                record.actions().len()
            ),
        )
        .unwrap();
        for name in [
            "agents.tsv",
            "telemetry.tsv",
            "pairwise.tsv",
            "ratings-internal.tsv",
            "ratings-garden-gate.tsv",
            "ratings-garden-gate-updates.tsv",
        ] {
            fs::write(root.join(name), "fixture\n").unwrap();
        }
        write_sha256_manifest(&root).unwrap();
        verify_archive_structure(&root).unwrap();
        let manifest = fs::read_to_string(root.join("MANIFEST.sha256")).unwrap();
        assert!(manifest.contains("records/game.psr"));
        assert!(!manifest.contains(&root.display().to_string()));

        fs::write(root.join("unlisted.txt"), "extra\n").unwrap();
        assert!(verify_archive_structure(&root).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rated_outcome_codes_remain_site_compatible() {
        assert_eq!(RatedOutcome::HostWin.host_score(), 1.0);
        assert_eq!(RatedOutcome::Draw.host_score(), 0.5);
        assert_eq!(RatedOutcome::GuestWin.host_score(), 0.0);
        assert!(
            verify_archived_outcome("H", "HOST_WIN", false, GameOutcome::Win(Player::Host),)
                .is_ok()
        );
    }
}
