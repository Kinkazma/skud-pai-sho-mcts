//! One append-only observation per due V3 transaction, plus its saved final tail.
//! No additional model reads or changes to selection, gates, RNG or learning.
//! Its bounded I/O overhead is included in native active runtime measurements.
use super::*;
use std::io::Write;
#[derive(Clone, Copy, Serialize)]
pub(super) struct Counts {
    receipts: usize,
    terminals: usize,
    version: u64,
    updates: u64,
    fresh_eligible: usize,
    fresh_learned: usize,
    presentations: usize,
    durable_recall: usize,
}
impl Counts {
    pub fn capture(
        receipts: usize,
        version: u64,
        updates: u64,
        lanes: &BTreeMap<String, Counters>,
        quotas: &recall::Quotas,
    ) -> Self {
        Self {
            receipts,
            terminals: lanes.iter().filter(|(lane, _)| lane.as_str() != "Reanalysis")
                .map(|(_, l)| l.terminal).sum(),
            version,
            updates,
            fresh_eligible: lanes.values().map(|l| l.eligible).sum(),
            fresh_learned: lanes.values().map(|l| l.fresh_used).sum(),
            presentations: quotas.consumed_examples,
            durable_recall: quotas.consumed_recall,
        }
    }
    fn difference(self, old: Self) -> Result<Self> {
        macro_rules! sub {
            ($field:ident) => {
                self.$field.checked_sub(old.$field).ok_or_else(|| {
                    invalid(concat!(
                        "publication log counter decreased: ",
                        stringify!($field)
                    ))
                })?
            };
        }
        Ok(Self {
            receipts: sub!(receipts),
            terminals: sub!(terminals),
            version: sub!(version),
            updates: sub!(updates),
            fresh_eligible: sub!(fresh_eligible),
            fresh_learned: sub!(fresh_learned),
            presentations: sub!(presentations),
            durable_recall: sub!(durable_recall),
        })
    }
}
pub(super) struct Log {
    file: fs::File,
    last: Counts,
    last_active: f64,
    events: usize,
    resumed: bool,
    migrated: bool,
}
fn actor(s: &Snapshot) -> serde_json::Value {
    serde_json::json!({"identity":s.identity,"version":s.version,
    "artifact_updates":s.artifact.as_ref().map(|a|a.updates)})
}
fn decision(guard: &serde_json::Value, checked: bool) -> serde_json::Value {
    if !checked {
        return serde_json::json!({"decision":"not-checked","checks":guard["checks"],"rejected":guard["rejected"]});
    }
    let admitted = guard["last_decision"] == "accepted-transaction";
    let rejected = guard["last_decision"] == "rejected-transaction";
    serde_json::json!({"checks":guard["checks"],"rejected":guard["rejected"],"decision":guard["last_decision"],
        "accepted_branch":if admitted {guard["accepted_branch"].clone()}else{serde_json::Value::Null},
        "accepted_fraction":if admitted {guard["accepted_fraction"].clone()}else{serde_json::Value::Null},
        "repair":if admitted {guard["repair"].clone()}else{serde_json::Value::Null},
        "reasons":if rejected {guard["reasons"].clone()}else{serde_json::Value::Null},
        "branch_attempts":if admitted||rejected {guard["branch_attempts"].clone()}else{serde_json::Value::Null}})
}
fn append(file: &mut impl Write, row: &serde_json::Value) -> Result<()> {
    let mut bytes = serde_json::to_vec(row)?;
    bytes.push(b'\n');
    file.write_all(&bytes)?;
    Ok(())
}
impl Log {
    pub fn open(path: &Path, counts: Counts, resumed: bool, migrated: bool) -> Result<Self> {
        let file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self {
            file,
            last: counts,
            last_active: 0.,
            events: 0,
            resumed,
            migrated,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn boundary(
        &mut self,
        receipt_id: usize,
        start: f64,
        end: f64,
        counts: Counts,
        before: &Snapshot,
        candidate: &Snapshot,
        after: &Snapshot,
        working: &Snapshot,
        guard: &serde_json::Value,
        protection: &serde_json::Value,
        rebased: bool,
    ) -> Result<()> {
        let delta = counts.difference(self.last)?;
        let unchanged = before.identity == after.identity;
        let row = serde_json::json!({"schema":"paisho-gen5-publication-transaction-v1","kind":"boundary","receipt_id":receipt_id,
            "event_in_process":self.events,"process_id":std::process::id(),"active_seconds":end,
            "previous_boundary_end_active_seconds":self.last_active,"control_start_active_seconds":start,
            "control_seconds_excluding_journal_write":end-start,"window_seconds_including_control":end-self.last_active,
            "counts":counts,"since_previous_observed_boundary":delta,"first_window_since_process_start":self.events==0,
            "resumed_inputs":self.resumed,"legacy_weight_migration":self.migrated,
            "first_window_excludes_any_prior_process_inflight_updates":self.events==0&&self.resumed&&!self.migrated,
            "actor_before":actor(before),"candidate_after_consolidation":actor(candidate),"actor_after":actor(after),"working_after":actor(working),
            "actor_identity_changed":!unchanged,"learner_rebased_to_accepted":rebased,
            "block_ended_on_same_actor":rebased&&unchanged,
            "updates_consumed_in_block_ending_on_same_actor":if rebased&&unchanged {Some(delta.updates)}else{None},
            "publication":decision(guard,rebased),"consolidation":protection["last"],
            "consolidation_checks":protection["checks"],"consolidation_accepted":protection["accepted_corrections"],
            "consolidation_seconds_cumulative":protection["seconds"],"fresh_guard_positions":protection["fresh_positions"],
            "retained_gradient_norm_sum":protection["retained_norm_sum"],"zero_steps":protection["zero_steps"],
            "played_link":"join later existing receipt.collector to actor_after.identity; in-flight old collectors can finish later",
            "updates_count_consumed_optimizer_steps_not_individually_preserved_steps":true});
        append(&mut self.file, &row)?;
        // Same deferred durability primitive as the native pending receipt files.
        // Errors propagate; no success is silently recorded after an I/O failure.
        paisho_platform::sync_before_batch_commit(&self.file)?;
        self.last = counts;
        self.last_active = end;
        self.events += 1;
        Ok(())
    }
    pub fn finish(
        &mut self,
        active: f64,
        counts: Counts,
        working: &Snapshot,
        accepted: &Snapshot,
    ) -> Result<()> {
        let delta = counts.difference(self.last)?;
        append(
            &mut self.file,
            &serde_json::json!({"schema":"paisho-gen5-publication-transaction-v1","kind":"saved-final-tail",
            "process_id":std::process::id(),"active_seconds":active,"counts":counts,"since_previous_observed_boundary":delta,
            "learner":actor(working),"actor":actor(accepted),"unpublished_tail_updates_in_this_process":delta.updates,
            "tail_is_saved_for_resume_not_counted_as_rejected":true,"observed_boundaries":self.events,
            "first_window_excludes_any_prior_process_inflight_updates":self.events==0&&self.resumed&&!self.migrated}),
        )?;
        paisho_platform::sync_before_batch_commit(&self.file)?;
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn counts() -> Counts {
        Counts {
            receipts: 10,
            terminals: 4,
            version: 9,
            updates: 12,
            fresh_eligible: 20,
            fresh_learned: 18,
            presentations: 90,
            durable_recall: 45,
        }
    }
    #[test]
    fn unchanged_never_reuses_an_old_successful_fraction_or_repair() {
        let stale = serde_json::json!({"checks":9,"rejected":3,"last_decision":"unchanged","accepted_branch":"full","accepted_fraction":0.125,
            "repair":{"steps":1},"reasons":["stale reason"],"branch_attempts":[{"kind":"full"}]});
        let normalized = decision(&stale, true);
        for field in [
            "repair",
            "accepted_fraction",
            "accepted_branch",
            "reasons",
            "branch_attempts",
        ] {
            assert!(normalized[field].is_null(), "{field}");
        }
        assert_eq!(decision(&stale, false)["decision"], "not-checked");
        assert!(decision(&stale, false)["repair"].is_null());
        let mut live = stale;
        live["last_decision"] = "accepted-transaction".into();
        assert_eq!(decision(&live, true)["accepted_fraction"], 0.125);
        live["last_decision"] = "rejected-transaction".into();
        assert!(decision(&live, true)["repair"].is_null());
        assert_eq!(decision(&live, true)["reasons"][0], "stale reason");
    }
    #[test]
    fn deltas_never_reset_consumed_counters_and_reject_rewind() {
        let a = counts();
        let mut b = a;
        b.updates += 5;
        b.presentations += 192;
        let d = b.difference(a).unwrap();
        assert_eq!(d.updates, 5);
        assert_eq!(d.receipts, 0);
        assert_eq!(d.presentations, 192);
        assert!(a.difference(b).is_err());
        assert_eq!(b.updates, 17);
    }
    #[test]
    fn jsonl_is_append_only_and_io_failure_is_reported() {
        let mut bytes = vec![];
        append(
            &mut bytes,
            &serde_json::json!({"kind":"boundary","x":"a\nb"}),
        )
        .unwrap();
        append(&mut bytes, &serde_json::json!({"kind":"saved-final-tail"})).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 2);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(lines[0]).unwrap()["x"],
            "a\nb"
        );
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("sentinel"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        assert!(append(&mut Broken, &serde_json::json!({"kind":"boundary"})).is_err());
    }
}
