//! Evidence stays distinct from the scalar target chosen for the optimizer.
use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TargetEvidence {
    /// Explicit verified support-set objective, distinct from search estimates.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub policy_support: bool,
    /// Missing identifies legacy targets; never reinterpret their coupled prior as raw.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub policy_coordinates: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub search_prior: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coupling_strength: Option<f64>,
    pub observed_value: Option<f64>,
    /// Hash of the completed PSR producing observed_value, including for reanalysis.
    pub observed_psr: Option<String>,
    pub estimated_value: Option<f64>,
    pub value_weight: f64,
    pub policy_source: String,
    /// Present for estimated improved targets; aligns with SavedMicroExample.actions.
    pub completed_action_values: Vec<f64>,
    /// Total search visits per action, including retained visits. Empty is legacy
    /// provenance absence, never evidence that all actions were unvisited.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub action_value_visits: Vec<usize>,
    pub target_prior: Vec<f64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub excluded_actions: Vec<bool>,
    pub player: String,
    pub actor: String,
}
impl TargetEvidence {
    pub(super) fn validate(&self) -> Result<()> {
        if self.policy_support
            && !["verified-search", "verified-regulatory-win"].contains(&self.policy_source.as_str())
        {
            return Err(invalid("policy support requires proof provenance"));
        }
        if !["", "raw-policy-v1"].contains(&self.policy_coordinates.as_str())
            || self
                .coupling_strength
                .is_some_and(|v| !v.is_finite() || !(0.0..=16.0).contains(&v))
            || !self.search_prior.is_empty()
                && (self.search_prior.len() != self.target_prior.len()
                    || self.search_prior.iter().any(|v| !v.is_finite() || *v < 0.)
                    || (self.search_prior.iter().sum::<f64>() - 1.).abs() > 1e-8)
        {
            return Err(invalid("invalid policy coordinate provenance"));
        }
        if self
            .observed_value
            .into_iter()
            .chain(self.estimated_value)
            .any(|v| !v.is_finite() || v.abs() > 1.)
            || !self.value_weight.is_finite()
            || !(0.0..=1.0).contains(&self.value_weight)
            || self.observed_value.is_some() != self.observed_psr.is_some()
            || self
                .observed_psr
                .as_ref()
                .is_some_and(|h| h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()))
            || !["H", "G"].contains(&self.player.as_str())
            || self.actor.is_empty()
            || ![
                "verified-regulatory-win",
                "verified-search",
                "full-search-estimate",
                "observed-action-value-only",
            ]
            .contains(&self.policy_source.as_str())
        {
            return Err(invalid("invalid learning evidence provenance"));
        }
        if !self.excluded_actions.is_empty()
            && self.excluded_actions.len() != self.target_prior.len()
        {
            return Err(invalid("invalid excluded teacher actions"));
        }
        if !self.action_value_visits.is_empty()
            && self.action_value_visits.len() != self.completed_action_values.len()
        {
            return Err(invalid("invalid action-value visit provenance"));
        }
        if self.completed_action_values.len() != self.target_prior.len()
            || self
                .completed_action_values
                .iter()
                .any(|q| !q.is_finite() || q.abs() > 1. + 1e-9)
            || self.target_prior.iter().any(|p| !p.is_finite() || *p < 0.)
            || !self.target_prior.is_empty()
                && (self.target_prior.iter().sum::<f64>() - 1.).abs() > 1e-8
        {
            return Err(invalid("invalid per-action teacher evidence"));
        }
        Ok(())
    }
    pub(super) fn legacy_coupled_policy(&self) -> bool {
        self.policy_source == "full-search-estimate" && self.policy_coordinates.is_empty()
    }
}
