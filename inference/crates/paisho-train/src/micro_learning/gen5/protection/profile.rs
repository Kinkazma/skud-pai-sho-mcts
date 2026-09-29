//! Current-process training work only; observations never affect optimization.
use super::*;
#[derive(Default)]
pub(super) struct Profile {
    pub gradient_buffer_allocations:Arc<std::sync::atomic::AtomicUsize>,
    pub batches:usize,
    pub examples:usize,
    pub actions:usize,
    pub duplicate_examples:usize,
    pub inactive_policy_examples_with_actions:usize,
    pub content_sample_examples:usize,
    pub content_sample_duplicates:usize,
    pub input_sample_duplicates:usize,
    pub gradient_wall_seconds:f64,
    pub gradient_worker_wall_seconds:f64,
    pub reduction_seconds:f64,
    pub update_seconds:f64,
}
impl Profile {
    pub fn observe(&mut self,batch:&[&MicroExample]) {
        self.batches+=1;self.examples+=batch.len();
        self.actions+=batch.iter().map(|e|e.actions.len()).sum::<usize>();
        self.inactive_policy_examples_with_actions+=batch.iter().filter(|e|
            !e.actions.is_empty() && e.policy_weight==0. &&
            (e.value_weight==0. || e.action_values.iter().all(Option::is_none))).count();
        let unique=batch.iter().map(|e|*e as *const MicroExample as usize).collect::<std::collections::HashSet<_>>();
        self.duplicate_examples+=batch.len()-unique.len();
        if self.batches<=128 {
            use std::hash::{Hash,Hasher};
            let inputs=batch.iter().map(|e| {
                let mut h=std::collections::hash_map::DefaultHasher::new();
                e.sequence_source.hash(&mut h); e.state.len().hash(&mut h);e.actions.len().hash(&mut h);
                for v in e.state.iter().chain(e.actions.iter().flatten()) {v.to_bits().hash(&mut h);}
                h.finish()
            }).collect::<std::collections::HashSet<_>>();
            self.input_sample_duplicates+=batch.len()-inputs.len();
            let keys=batch.iter().map(|e| {
                let mut h=std::collections::hash_map::DefaultHasher::new();
                e.sequence_source.hash(&mut h);
                e.policy_support.hash(&mut h);
                for v in [e.value,e.value_weight,e.policy_weight] {v.to_bits().hash(&mut h);}
                for n in [e.state.len(),e.actions.len(),e.policy.len(),e.action_values.len()] {n.hash(&mut h);}
                for v in e.state.iter().chain(e.actions.iter().flatten()).chain(&e.policy) {v.to_bits().hash(&mut h);}
                for q in &e.action_values {q.map(f64::to_bits).hash(&mut h);}
                h.finish()
            }).collect::<std::collections::HashSet<_>>();
            self.content_sample_examples+=batch.len();
            self.content_sample_duplicates+=batch.len()-keys.len();
        }
    }
    pub fn progress(&self)->serde_json::Value {
        serde_json::json!({"scope":"current process; gradient wall is caller dispatch/wait excluding its ordered reduction; worker task times overlap both, not CPU seconds; input/content hash sampling first 128 batches only","ordered_reduction_during_dispatch":true,"gradient_buffer_allocations":self.gradient_buffer_allocations.load(std::sync::atomic::Ordering::Relaxed),"batches":self.batches,"examples":self.examples,"action_rows":self.actions,"duplicate_examples":self.duplicate_examples,"content_sample_examples":self.content_sample_examples,"content_sample_duplicates":self.content_sample_duplicates,"input_sample_duplicates":self.input_sample_duplicates,"inactive_policy_examples_with_actions":self.inactive_policy_examples_with_actions,"gradient_wall_seconds":self.gradient_wall_seconds,"gradient_worker_wall_seconds":self.gradient_worker_wall_seconds,"reduction_seconds":self.reduction_seconds,"update_seconds":self.update_seconds})
    }
}
