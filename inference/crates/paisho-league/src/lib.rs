//! Reproducible heterogeneous agent leagues and immutable evidence archives.

mod agents;
mod analysis;
mod archive;
mod pairwise;
mod runner;
mod schedule;

pub use agents::{default_agent_definitions, AgentDefinition, LeagueAgent};
pub use analysis::{analyze, LeagueAnalysis, SiteOrderRun};
pub use archive::{
    preflight_archive_path, verify_archive_manifest, write_archive, ArchiveMetadata,
};
pub use pairwise::{summarize_pairwise, PairwiseSummary};
pub use runner::{run_schedule, LeagueExecution, PlayedGame};
pub use schedule::{build_ladder_schedule, ScheduleError, ScheduledGame, LADDER_EDGE_ALIASES};
