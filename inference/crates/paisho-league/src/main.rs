use std::io;
use std::path::PathBuf;
use std::process::Command;

use paisho_ai::MatchConfig;
use paisho_league::{
    analyze, build_ladder_schedule, default_agent_definitions, preflight_archive_path,
    run_schedule, summarize_pairwise, verify_archive_manifest, write_archive, AgentDefinition,
    ArchiveMetadata, LeagueAnalysis, LeagueExecution, LADDER_EDGE_ALIASES,
};

const BUILD_SOURCE_REVISION: &str = env!("PAISHO_BUILD_GIT_REVISION");
const BUILD_SOURCE_DIRTY_TEXT: &str = env!("PAISHO_BUILD_GIT_DIRTY");

fn main() {
    if let Err(error) = run() {
        eprintln!("rating league failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1).peekable();
    if arguments
        .peek()
        .is_some_and(|argument| argument == "--verify")
    {
        arguments.next();
        let directory = arguments
            .next()
            .ok_or("usage: paisho-league --verify EVIDENCE-DIRECTORY")?;
        if arguments.next().is_some() {
            return Err("usage: paisho-league --verify EVIDENCE-DIRECTORY".into());
        }
        verify_archive_manifest(&PathBuf::from(&directory))?;
        println!("verified league archive: {directory}");
        return Ok(());
    }
    let pairs_per_edge = positive_argument(&mut arguments, "pairs per edge", 6);
    let decision_soft_limit = positive_argument(&mut arguments, "decision soft limit", 512);
    let first_pair_id = u64_argument(&mut arguments, "first pair id", 10_000);
    let site_initial_rating = i32_argument(&mut arguments, "site initial rating", 1_000);
    let output_directory = arguments.next().map(PathBuf::from);
    if arguments.next().is_some() {
        return Err("usage: paisho-league [pairs-per-edge] [decision-soft-limit] [first-pair-id] [site-initial-rating] [evidence-directory]".into());
    }

    if let Some(directory) = &output_directory {
        preflight_archive_path(directory)?;
    }
    let source_revision = source_identity(output_directory.is_some())?;
    let agents = default_agent_definitions(&source_revision);
    let schedule = build_ladder_schedule(&agents, pairs_per_edge, first_pair_id)?;
    println!("Pai Sho rating ladder v1");
    println!("source identity: {source_revision}");
    println!(
        "agents: {}, edges: {}",
        agents.len(),
        LADDER_EDGE_ALIASES.len()
    );
    println!(
        "pairs per edge: {pairs_per_edge}, games scheduled: {}",
        schedule.len()
    );
    println!("decision soft limit: {decision_soft_limit}");

    let execution = run_schedule(
        &schedule,
        &agents,
        MatchConfig {
            decision_soft_limit,
        },
    );
    let analysis = analyze(&execution, &agents, site_initial_rating)?;
    print_execution(&execution, &analysis, &agents);

    let evidence_was_sealed = if let Some(directory) = output_directory {
        verify_tracked_source(&source_revision)?;
        let metadata = ArchiveMetadata {
            source_revision,
            pairs_per_edge,
            first_pair_id,
            decision_soft_limit,
            site_initial_rating,
            site_initialization_status:
                "caller-specified; public client does not expose account initialization".to_owned(),
        };
        write_archive(
            &directory, &metadata, &agents, &schedule, &execution, &analysis,
        )?;
        println!("sealed evidence: {}", directory.display());
        true
    } else {
        false
    };
    if analysis.internal_fit.is_err() {
        let preservation = if evidence_was_sealed {
            "raw games were preserved in the sealed diagnostic archive"
        } else {
            "no archive was requested, so rerun with an evidence directory to preserve raw games"
        };
        return Err(format!("the internal MLE is not publishable; {preservation}").into());
    }
    Ok(())
}

fn print_execution(
    execution: &LeagueExecution,
    analysis: &LeagueAnalysis,
    agents: &[AgentDefinition],
) {
    let errors = execution
        .games
        .iter()
        .filter(|game| game.result.is_err())
        .count();
    println!(
        "observed match workers: {}/{} (available CPUs: {})",
        execution.observed_match_workers,
        execution.worker_capacity,
        execution.available_parallelism,
    );
    println!("elapsed: {:.3}s", execution.elapsed_seconds);
    println!(
        "throughput: {:.3} games/s",
        execution.games.len() as f64 / execution.elapsed_seconds
    );
    println!(
        "rated games/pairs: {}/{}; excluded pairs: {}; match errors: {}",
        analysis.rated_games.len(),
        analysis.rated_games.len() / 2,
        analysis.excluded_pairs.len(),
        errors
    );
    print_pairwise(execution, agents);
    match &analysis.internal_fit {
        Ok(fit) => {
            println!("Elo interne non calibre au site (provisional)");
            for rating in &fit.ratings {
                let alias = agents
                    .iter()
                    .find(|agent| agent.id() == &rating.agent)
                    .map(AgentDefinition::alias)
                    .unwrap_or("unknown");
                println!(
                    "  {alias}: {:.1}; model CI [{:.1}, {:.1}] ({} games)",
                    rating.elo.estimate,
                    rating.elo.model.interval_95.lower,
                    rating.elo.model.interval_95.upper,
                    rating.rated_games,
                );
                if let Some(cluster) = rating.elo.paired_cluster {
                    println!(
                        "    paired-cluster CI [{:.1}, {:.1}]",
                        cluster.interval_95.lower, cluster.interval_95.upper
                    );
                } else {
                    println!("    paired-cluster CI unavailable (degenerate cluster covariance)");
                }
            }
            if let Some(host) = fit.host_advantage_elo {
                println!(
                    "  Host advantage: {:.1}; model CI [{:.1}, {:.1}] Elo",
                    host.estimate, host.model.interval_95.lower, host.model.interval_95.upper
                );
                if let Some(cluster) = host.paired_cluster {
                    println!(
                        "    paired-cluster CI [{:.1}, {:.1}]",
                        cluster.interval_95.lower, cluster.interval_95.upper
                    );
                }
            }
            println!(
                "  tie model: {:?}; draw weight: {:.4}; log likelihood: {:.6}",
                fit.tie_model,
                fit.draw_weight(),
                fit.log_likelihood
            );
        }
        Err(error) => println!("internal rating unavailable: {error}"),
    }
    println!("Elo simule - formule The Garden Gate");
    let canonical = &analysis.site_order_runs[0];
    for agent in agents {
        let rating = canonical.run.ratings[agent.id()];
        let (minimum, maximum) = analysis.site_sampled_order_ranges[agent.id()];
        let rated_games = analysis.rated_game_counts[agent.id()];
        let status = if rated_games == 0 {
            "unrated; displayed value is initialization only"
        } else {
            "simulated-provisional"
        };
        println!(
            "  {}: {} canonical-schedule; sampled-order range [{}, {}]; {} games; {status}",
            agent.alias(),
            rating,
            minimum,
            maximum,
            rated_games,
        );
    }
}

fn print_pairwise(execution: &LeagueExecution, agents: &[AgentDefinition]) {
    println!("pairwise results (first alias perspective W/D/L/U)");
    for summary in summarize_pairwise(execution, agents) {
        println!(
            "  {} vs {}: {}/{}/{}/{}; paired p={:.6}",
            summary.first_alias,
            summary.second_alias,
            summary.wins,
            summary.draws,
            summary.losses,
            summary.unfinished,
            summary.paired.exact_two_sided_sign_test_p_value(),
        );
    }
}

fn source_identity(require_clean: bool) -> io::Result<String> {
    let revision = git_output(&["rev-parse", "HEAD"])?;
    let status = git_output(&["status", "--porcelain"])?;
    if require_clean {
        let expected = std::env::var("PAISHO_SOURCE_REVISION").map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "PAISHO_SOURCE_REVISION is required when writing evidence",
            )
        })?;
        validate_evidence_identity(
            &expected,
            &revision,
            BUILD_SOURCE_REVISION,
            BUILD_SOURCE_DIRTY_TEXT == "true",
            &status,
        )?;
        Ok(revision)
    } else if status.is_empty()
        && BUILD_SOURCE_REVISION == revision
        && BUILD_SOURCE_DIRTY_TEXT != "true"
    {
        Ok(BUILD_SOURCE_REVISION.to_owned())
    } else {
        Ok(format!(
            "{}+runtime-head-{revision}+dirty-exploratory",
            BUILD_SOURCE_REVISION
        ))
    }
}

fn validate_evidence_identity(
    requested: &str,
    runtime_revision: &str,
    build_revision: &str,
    build_was_dirty: bool,
    status: &str,
) -> io::Result<()> {
    if requested != runtime_revision {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("requested revision {requested}, but HEAD is {runtime_revision}"),
        ));
    }
    if build_revision != runtime_revision {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "this binary was built from {build_revision}, but the worktree is at {runtime_revision}; rebuild before writing evidence"
            ),
        ));
    }
    if build_was_dirty {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "this binary was built from a dirty worktree; rebuild from the clean requested commit",
        ));
    }
    if !status.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the Git worktree must be clean before an evidence run",
        ));
    }
    Ok(())
}

fn verify_tracked_source(expected: &str) -> io::Result<()> {
    if git_output(&["rev-parse", "HEAD"])? != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "HEAD changed while the evidence run was executing",
        ));
    }
    for arguments in [
        ["diff", "--quiet"].as_slice(),
        ["diff", "--cached", "--quiet"].as_slice(),
    ] {
        let status = Command::new("git")
            .arg("-C")
            .arg(env!("CARGO_MANIFEST_DIR"))
            .args(arguments)
            .status()?;
        if !status.success() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "tracked source changed while the evidence run was executing",
            ));
        }
    }
    if !git_output(&["status", "--porcelain"])?.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the Git worktree stopped being clean while the evidence run was executing",
        ));
    }
    Ok(())
}

fn git_output(arguments: &[&str]) -> io::Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(arguments)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&output.stderr).trim().to_owned(),
        ));
    }
    String::from_utf8(output.stdout)
        .map(|text| text.trim().to_owned())
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn positive_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: usize,
) -> usize {
    let Some(text) = arguments.next() else {
        return default;
    };
    text.parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .unwrap_or_else(|| {
            eprintln!("{name} must be a positive integer, got `{text}`");
            std::process::exit(2);
        })
}

fn u64_argument(arguments: &mut impl Iterator<Item = String>, name: &str, default: u64) -> u64 {
    arguments.next().map_or(default, |text| {
        text.parse::<u64>().unwrap_or_else(|_| {
            eprintln!("{name} must be a non-negative integer, got `{text}`");
            std::process::exit(2);
        })
    })
}

fn i32_argument(arguments: &mut impl Iterator<Item = String>, name: &str, default: i32) -> i32 {
    arguments.next().map_or(default, |text| {
        text.parse::<i32>().unwrap_or_else(|_| {
            eprintln!("{name} must be an integer, got `{text}`");
            std::process::exit(2);
        })
    })
}

#[cfg(test)]
mod tests {
    use super::validate_evidence_identity;

    #[test]
    fn evidence_identity_rejects_a_stale_binary_and_dirty_source() {
        validate_evidence_identity("abc", "abc", "abc", false, "").unwrap();
        assert!(validate_evidence_identity("abc", "abc", "old", false, "").is_err());
        assert!(validate_evidence_identity("abc", "abc", "abc", true, "").is_err());
        assert!(validate_evidence_identity("abc", "abc", "abc", false, "?? source.rs").is_err());
        assert!(validate_evidence_identity("wanted", "actual", "actual", false, "").is_err());
    }
}
