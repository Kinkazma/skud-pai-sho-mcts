use core::fmt::Write as _;

use paisho_ai::{
    Agent, AgentError, AgentTelemetry, HeuristicWeights, MctsAgent, MctsConfig, SiteBotV1,
    EXHAUSTIVE_ACTION_RANKING, SITE_BOT_V1_SOURCE_COMMIT,
};
use paisho_core::{Action, Position, RuleProfileId};
use paisho_rating::AgentId;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AgentKind {
    SiteBotV1,
    Mcts { simulations: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AgentDefinition {
    alias: String,
    id: AgentId,
    fingerprint: String,
    descriptor: String,
    kind: AgentKind,
}

impl AgentDefinition {
    pub fn alias(&self) -> &str {
        &self.alias
    }

    pub fn id(&self) -> &AgentId {
        &self.id
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn descriptor(&self) -> &str {
        &self.descriptor
    }

    pub fn family(&self) -> &'static str {
        match self.kind {
            AgentKind::SiteBotV1 => "site-bot-v1",
            AgentKind::Mcts { .. } => "heuristic-mcts",
        }
    }

    pub(crate) fn make(&self, seed: u64) -> LeagueAgent {
        match self.kind {
            AgentKind::SiteBotV1 => LeagueAgent::SiteBot(SiteBotV1::new(seed)),
            AgentKind::Mcts { simulations } => LeagueAgent::Mcts(Box::new(
                MctsAgent::new(seed, league_mcts_config(simulations))
                    .expect("the sealed league MCTS configuration is valid"),
            )),
        }
    }
}

pub enum LeagueAgent {
    SiteBot(SiteBotV1),
    Mcts(Box<MctsAgent>),
}

impl Agent for LeagueAgent {
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        match self {
            Self::SiteBot(agent) => agent.select_action(position, legal_actions),
            Self::Mcts(agent) => agent.select_action(position, legal_actions),
        }
    }

    fn telemetry(&self) -> AgentTelemetry {
        match self {
            Self::SiteBot(agent) => agent.telemetry(),
            Self::Mcts(agent) => agent.telemetry(),
        }
    }

    fn reset_telemetry(&mut self) {
        match self {
            Self::SiteBot(agent) => agent.reset_telemetry(),
            Self::Mcts(agent) => agent.reset_telemetry(),
        }
    }
}

pub fn default_agent_definitions(source_revision: &str) -> Vec<AgentDefinition> {
    agent_definitions_for_rules(source_revision, RuleProfileId::CURRENT)
}

/// Reconstruct archived identities without applying today's profile to them.
pub(crate) fn agent_definitions_for_rules(
    source_revision: &str,
    rules: RuleProfileId,
) -> Vec<AgentDefinition> {
    let mut definitions = vec![site_definition(source_revision, rules)];
    definitions.extend(
        [8, 32, 128, 512]
            .into_iter()
            .map(|simulations| mcts_definition(source_revision, simulations, rules)),
    );
    definitions
}

pub fn league_mcts_config(simulations: usize) -> MctsConfig {
    MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 96,
        action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
        root_widening_factor: 1.0,
        progressive_widening_factor: 1.0,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    }
}

fn site_definition(source_revision: &str, rules: RuleProfileId) -> AgentDefinition {
    let descriptor = format!(
        "family=site-bot-v1;implementation_revision={source_revision};upstream_revision={SITE_BOT_V1_SOURCE_COMMIT};rule_profile={rules};policy=source-compatible-one-ply;seed_policy=seat-bound-splitmix64"
    );
    definition("site-bot-v1", AgentKind::SiteBotV1, descriptor)
}

fn mcts_definition(
    source_revision: &str,
    simulations: usize,
    rules: RuleProfileId,
) -> AgentDefinition {
    let config = league_mcts_config(simulations);
    let weights = config.heuristic_weights;
    let descriptor = format!(
        "family=heuristic-mcts;implementation_revision={source_revision};rule_profile={rules};simulations={};independent_trees={};maximum_tree_depth={};action_rank_batch=all;rollout_depth={};exploration_bits={:08x};heuristic_bits={:08x},{:08x},{:08x},{:08x},{:08x};seed_policy=seat-bound-splitmix64",
        config.simulations,
        config.independent_trees,
        config.maximum_tree_depth,
        config.rollout_depth,
        config.exploration.to_bits(),
        weights.harmony.to_bits(),
        weights.midline_harmony.to_bits(),
        weights.blooming_flower.to_bits(),
        weights.total_flower.to_bits(),
        weights.basic_reserve_progress.to_bits(),
    );
    definition(
        &format!("mcts-{simulations}"),
        AgentKind::Mcts { simulations },
        descriptor,
    )
}

fn definition(alias: &str, kind: AgentKind, descriptor: String) -> AgentDefinition {
    let fingerprint = sha256_text(&descriptor);
    let id = AgentId::new(format!("sha256:{fingerprint}"))
        .expect("a lowercase SHA-256 digest is a valid agent id");
    AgentDefinition {
        alias: alias.to_owned(),
        id,
        fingerprint,
        descriptor,
        kind,
    }
}

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    let mut digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(digest, "{byte:02x}").expect("writing to a String cannot fail");
    }
    digest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_identity_changes_with_revision_and_budget() {
        let first = default_agent_definitions("revision-a");
        let repeated = default_agent_definitions("revision-a");
        let changed = default_agent_definitions("revision-b");
        assert_eq!(first[2].fingerprint, repeated[2].fingerprint);
        assert_ne!(first[1].fingerprint, first[2].fingerprint);
        assert_ne!(first[2].fingerprint, changed[2].fingerprint);
        assert_eq!(first[0].alias, "site-bot-v1");
        assert_eq!(first[4].alias, "mcts-512");
    }

    #[test]
    fn current_identities_are_distinct_from_reconstructed_legacy_identities() {
        let current = default_agent_definitions("same-source");
        let legacy = agent_definitions_for_rules("same-source", RuleProfileId::SkudPaiSho2022);
        for (current, legacy) in current.iter().zip(&legacy) {
            assert!(current
                .descriptor()
                .contains(&format!("rule_profile={};", RuleProfileId::CURRENT)));
            assert!(legacy
                .descriptor()
                .contains("rule_profile=skud-pai-sho-2022-03-14;"));
            assert_ne!(current.id(), legacy.id());
            assert_eq!(current.alias(), legacy.alias());
        }
    }
}
