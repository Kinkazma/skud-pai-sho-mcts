//! Offline bit-level inventory. Never called by the training hot path.
use super::*;
use sha2::{Digest,Sha256};
pub(crate) fn example_digest(h:&mut Sha256,e:&MicroExample)->Result<()> {
    for values in [&e.state[..], &e.policy[..], &e.action_values.iter().map(|x|x.unwrap_or(f64::NAN)).collect::<Vec<_>>()[..]] {
        h.update((values.len() as u64).to_le_bytes());
        for x in values {h.update(x.to_bits().to_le_bytes());}
    }
    h.update((e.actions.len() as u64).to_le_bytes());
    for row in &e.actions {for x in row {h.update(x.to_bits().to_le_bytes());}}
    for x in [e.value,e.policy_weight,e.value_weight] {h.update(x.to_bits().to_le_bytes());}
    h.update(e.sequence_source.to_le_bytes());h.update([u8::from(e.policy_support)]);
    h.update(serde_json::to_vec(&e.structured)?);Ok(())
}
impl Memory {
    pub(crate) fn portable_digest(&self)->Result<serde_json::Value> {
        let mut h=Sha256::new();
        for entry in &self.entries {
            h.update(serde_json::to_vec(&entry.lane)?);
            h.update([u8::from(entry.trainable),u8::from(entry._correction_lifetime.is_some())]);
            h.update((entry.index as u64).to_le_bytes());example_digest(&mut h,&entry.example)?;
        }
        Ok(serde_json::json!({"positions":self.len(),"sha256":format!("{:x}",h.finalize()),
            "correction_positions":self.correction_len(),"bytes":self.bytes}))
    }
}
