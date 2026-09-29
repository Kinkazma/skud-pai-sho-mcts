//! Order-independent internal ratings and The Garden Gate Elo replay.

mod davidson;
mod game;
mod promotion;
mod site_elo;

pub use davidson::{
    fit_davidson, fit_davidson_for_agents, AgentRating, ConfidenceInterval, DavidsonError,
    DavidsonFit, DavidsonOptions, PairwiseEloDifference, ParameterEstimate, TieModel, Uncertainty,
};
pub use game::{AgentId, AgentIdError, RatedGame, RatedGameError, RatedOutcome};
pub use promotion::{
    evaluate_promotion_sprt, PentanomialCounts, PromotionDecision, PromotionSprtConfig,
    PromotionSprtError, PromotionSprtReport, PENTANOMIAL_SPRT_REFERENCE_COMMIT,
    PENTANOMIAL_SPRT_REFERENCE_PATH,
};
pub use site_elo::{
    simulate_site_elo, simulate_site_elo_in_order, SiteEloConfig, SiteEloError, SiteEloRun,
    SiteEloUpdate, SITE_ELO_K_FACTOR, SITE_ELO_SOURCE_COMMIT, SITE_ELO_SOURCE_PATH,
};
