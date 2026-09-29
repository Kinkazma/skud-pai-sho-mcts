//! Internal optimizer/recall recovery payload; not a replacement for archived provenance.
use super::*;
#[derive(Clone, Serialize, Deserialize)]
pub(super) struct ResumeExample {
    #[serde(default, skip_serializing_if="std::ops::Not::not")]
    policy_support: bool,
    #[serde(default, skip_serializing_if="std::ops::Not::not")]
    trusted_action_values: bool,
    #[serde(default, skip_serializing_if="Vec::is_empty")]
    action_values: Vec<Option<f64>>,
    sequence_source: u64,
    value_weight: f64,
    state: Vec<f64>,
    actions: Vec<[f64; MICRO_ACTION_INPUTS]>,
    policy: Vec<f64>,
    value: f64,
    policy_weight: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn v3_resume_never_reintroduces_legacy_auxiliary_completion() {
        let ex=MicroExample {policy_support:false,action_values:vec![Some(0.6),None],sequence_source:19,
            value_weight:0.25,state:vec![0.;417],actions:vec![[0.;32];2],policy:vec![0.4,0.6],
            value:-0.3,policy_weight:1.};
        let old=serde_json::to_vec(&ResumeExample::from(&ex)).unwrap();
        let legacy:ResumeExample=serde_json::from_slice(&old).unwrap();
        assert_eq!(legacy.clone().example().unwrap().action_values,ex.action_values);
        let migrated=legacy.example_with_trusted_q(true).unwrap();
        assert!(migrated.action_values.is_empty());
        assert_eq!(migrated.policy,ex.policy);assert_eq!(migrated.state,ex.state);
        assert_eq!(migrated.actions,ex.actions);assert_eq!(migrated.value,ex.value);
        assert_eq!(migrated.value_weight,ex.value_weight);assert_eq!(migrated.sequence_source,ex.sequence_source);
        let new=serde_json::to_vec(&ResumeExample::from_with_trusted_q(&ex,true)).unwrap();
        let restored:ResumeExample=serde_json::from_slice(&new).unwrap();
        assert_eq!(restored.example_with_trusted_q(true).unwrap().action_values,ex.action_values);
    }
    #[test]
    fn explicit_support_survives_v3_resume_without_reclassifying_legacy_rows() {
        let ex=MicroExample {policy_support:true,action_values:vec![],sequence_source:19,value_weight:0.,state:vec![0.;417],actions:vec![[0.;32];2],policy:vec![0.5,0.5],value:1.,policy_weight:1.};
        let bytes=serde_json::to_vec(&ResumeExample::from_with_trusted_q(&ex,true)).unwrap();
        let restored:ResumeExample=serde_json::from_slice(&bytes).unwrap();
        assert!(restored.clone().example_with_trusted_q(true).unwrap().policy_support);
        assert!(!restored.example().unwrap().policy_support);
        let mut old:serde_json::Value=serde_json::from_slice(&bytes).unwrap();old.as_object_mut().unwrap().remove("policy_support");
        let restored:ResumeExample=serde_json::from_value(old).unwrap();
        assert!(!restored.example_with_trusted_q(true).unwrap().policy_support);
    }
}
impl From<&MicroExample> for ResumeExample {
    fn from(e: &MicroExample) -> Self {
        Self {
            policy_support: e.policy_support,
            trusted_action_values: false,
            action_values: e.action_values.clone(),
            sequence_source: e.sequence_source,
            value_weight: e.value_weight,
            state: e.state.clone(),
            actions: e.actions.clone(),
            policy: e.policy.clone(),
            value: e.value,
            policy_weight: e.policy_weight,
        }
    }
}
impl ResumeExample {
    pub fn from_with_trusted_q(e: &MicroExample, trusted_q: bool) -> Self {
        let mut out=Self::from(e);out.trusted_action_values=trusted_q;out
    }
    pub fn example(self) -> Result<Arc<MicroExample>> {
        self.example_with_trusted_q(false)
    }
    pub fn example_with_trusted_q(self, trusted_q: bool) -> Result<Arc<MicroExample>> {
        let values=if trusted_q && !self.trusted_action_values {vec![]} else {self.action_values};
        let e = MicroExample { policy_support: trusted_q && self.policy_support, action_values: values,
            sequence_source: self.sequence_source,
            value_weight: self.value_weight,
            state: self.state,
            actions: self.actions,
            policy: self.policy,
            value: self.value,
            policy_weight: self.policy_weight,
        };
        e.validate().map_err(invalid)?;
        Ok(Arc::new(e))
    }
}
