//! CPU-side opponents, search and parallel match execution.

mod agent;
mod compact_value;
mod evaluation;
mod heuristic;
mod matchplay;
mod mcts;
mod mcts_metal;
mod micro;
pub use micro::*;
mod neural;
mod rng;
mod site_bot;
mod tactical;

pub use agent::{Agent, AgentError, AgentTelemetry, RandomAgent};
pub use compact_value::{
    CompactValueError, CompactValueFeatures, CompactValueModel, COMPACT_FEATURE_COUNT,
    COMPACT_FEATURE_NAMES, COMPACT_FEATURE_SCALES, COMPACT_VALUE_SCHEMA_V1,
};
pub use evaluation::{GameScore, PairedComparison};
pub use heuristic::{evaluate_position, GreedyAgent, HeuristicWeights};
pub use matchplay::{
    play_match, play_match_from_record, play_parallel, MatchConfig, MatchError, MatchResult,
    MatchTask, MatchTermination, ParallelMatchResults,
};
pub use mcts::{
    ActionStatistics, CpuMctsEvaluator, MctsAgent, MctsConfig, MctsConfigError, MctsEvaluator,
    MctsReuseStatistics, MctsSession, SearchReport, EXHAUSTIVE_ACTION_RANKING,
};
pub use neural::{
    NetworkDecision, NetworkDecisionError, NetworkPolicy, NetworkPolicyError, PolicyValueEvaluator,
    PolicyValueOutput, PolicyValueOutputError, PureNetworkAgent,
};
pub use rng::StableRng;
pub use site_bot::{
    site_v1_action_score, site_v1_cycle_length, site_v1_cycle_progress_score, site_v1_main_actions,
    site_v1_transition_score, SiteBotV1, SiteV1Features, SITE_BOT_V1_SOURCE_COMMIT,
    SITE_BOT_V1_WIN_SCORE,
};
pub use tactical::{prove_forced_win, TacticalProof, TacticalVerdict};

pub use mcts_metal::{MetalMctsEvaluator, MetalMctsTelemetry};

#[cfg(unix)]
pub use mcts_metal::RemoteMetalMctsEvaluator;

mod sequence_memory;
pub use sequence_memory::*;

mod gen32;
pub use gen32::*;
