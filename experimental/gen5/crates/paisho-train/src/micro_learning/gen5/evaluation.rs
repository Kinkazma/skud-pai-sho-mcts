//! Fixed paired starts and immutable models. Each reference owns its own lot.
use super::*;
use std::sync::Mutex;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Attempt {
    pub reference: String,
    pub budget: usize,
    pub model: String,
    pub version: u64,
    pub lot: usize,
    pub slot: usize,
    pub prefix: String,
}
#[derive(Clone, Default, Serialize, Deserialize)]
struct Saved {
    reference: String,
    budget: usize,
    lot: usize,
    model: String,
    version: u64,
    path: PathBuf,
    starts: String,
    results: Vec<Option<i8>>,
    last_completed: String,
    #[serde(default)]
    minimum_search_depth: usize,
}
struct Slot {
    saved: Saved,
    snapshot: Option<Arc<Snapshot>>,
    reserved: [bool; 100],
}
pub(super) struct Evaluations {
    slots: Vec<Mutex<Slot>>,
    starts: Vec<cases::Case>,
    fingerprint: String,
    minimum_search_depth: usize,
    out: PathBuf,
    retired: Mutex<Vec<PathBuf>>,
}
pub(super) struct Job {
    pub snapshot: Arc<Snapshot>,
    pub case: cases::Case,
    pub attempt: Attempt,
    pub seat: Player,
}
impl Evaluations {
    pub fn open(
        out: &Path,
        cases: &[cases::Case],
        options: &Options,
        restored: &serde_json::Value,
        base: &MicroModel,
    ) -> Result<Self> {
        let specs = &options.opponents;
        let mut seen = std::collections::BTreeSet::new();
        let starts: Vec<_> = cases
            .iter()
            .filter(|c| !c.record.actions().is_empty() && seen.insert(c.source.clone()))
            .take(50)
            .enumerate()
            .map(|(i, c)| {
                let record = cases::prefix(&c.record, (1 + i % 8).min(c.record.actions().len()));
                cases::Case {
                    identity: sha256(record.to_string().as_bytes()),
                    source: c.source.clone(),
                    zone: 0,
                    record,
                }
            })
            .collect();
        if starts.len() != 50 {
            return Err(invalid(
                "frozen evaluation requires fifty distinct human sources",
            ));
        }
        let fingerprint_at = |minimum: usize| -> Result<String> {
            let mut protocol=serde_json::json!({"starts":starts.iter().map(|c|(&c.source,&c.identity)).collect::<Vec<_>>(),"mode":options.mode,"value_policy_strength":options.value_policy_strength,"proof_search":options.proof_search,"decision_limit":options.decision_limit,"candidate_budget":512,"fixed_seed":true,"external_proofs":false});
            // Preserve the historical zero-floor fingerprint for exact resumption.
            if minimum > 0 {protocol["minimum_search_depth"]=minimum.into();}
            Ok(sha256(&serde_json::to_vec(&protocol)?))
        };
        let fingerprint = fingerprint_at(options.minimum_search_depth)?;
        let old: Vec<Saved> = if restored.is_null() {
            vec![]
        } else {
            serde_json::from_value(restored.clone())?
        };
        if !old.is_empty() && old.len() != specs.len() {
            return Err(invalid("frozen reference count changed"));
        }
        fs::create_dir_all(out.join("frozen-evaluations"))?;
        let mut slots = vec![];
        for (i, spec) in specs.iter().enumerate() {
            let mut saved = old.get(i).cloned().unwrap_or_default();
            if !saved.reference.is_empty()
                && (saved.reference != spec.sha256
                    || saved.starts != fingerprint_at(saved.minimum_search_depth)?
                    || saved.results.len() != 100
                    || saved
                        .results
                        .iter()
                        .flatten()
                        .any(|z| !(-1..=2).contains(z)))
            {
                return Err(invalid("frozen evaluation resume identity mismatch"));
            }
            if saved.minimum_search_depth != options.minimum_search_depth {
                // A different search protocol starts a new complete lot with the
                // current actor. Never mix old partial results into its gate.
                // The source checkpoint retains the previous lot's full record.
                saved = Saved { lot: saved.lot, ..Default::default() };
            }
            let snapshot = if saved.model.is_empty() {
                None
            } else {
                Some(publication::load_snapshot(
                    &saved.path,
                    &saved.model,
                    saved.version,
                    base,
                )?)
            };
            slots.push(Mutex::new(Slot {
                saved,
                snapshot,
                reserved: [false; 100],
            }));
        }
        Ok(Self {
            slots,
            starts,
            fingerprint,
            minimum_search_depth: options.minimum_search_depth,
            out: out.to_path_buf(),
            retired: Mutex::new(vec![]),
        })
    }
    pub fn reserve(
        &self,
        index: usize,
        reference: &str,
        budget: usize,
        current: Arc<Snapshot>,
    ) -> Result<Option<Job>> {
        let mut slot = self.slots[index].lock().unwrap();
        if slot.snapshot.is_none() {
            if slot.saved.last_completed == current.identity {
                return Ok(None);
            }
            let lot = slot.saved.lot + 1;
            let path = self
                .out
                .join("frozen-evaluations")
                .join(format!("reference-{index}-lot-{lot}.json"));
            publication::save_snapshot(&current, &path)?;
            slot.saved = Saved {
                reference: reference.into(),
                budget,
                lot,
                model: current.identity.clone(),
                version: current.version,
                path,
                starts: self.fingerprint.clone(),
                results: vec![None; 100],
                last_completed: slot.saved.last_completed.clone(),
                minimum_search_depth: self.minimum_search_depth,
            };
            slot.snapshot = Some(current);
        }
        if slot.saved.reference != reference || slot.saved.budget != budget {
            return Err(invalid("in-flight frozen lot changed reference/budget"));
        }
        let Some(n) = (0..100).find(|&n| slot.saved.results[n].is_none() && !slot.reserved[n])
        else {
            return Ok(None);
        };
        slot.reserved[n] = true;
        let case = self.starts[n / 2].clone();
        Ok(Some(Job {
            snapshot: slot.snapshot.as_ref().unwrap().clone(),
            seat: if n % 2 == 0 {
                Player::Host
            } else {
                Player::Guest
            },
            attempt: Attempt {
                reference: reference.into(),
                budget,
                model: slot.saved.model.clone(),
                version: slot.saved.version,
                lot: slot.saved.lot,
                slot: n,
                prefix: case.identity.clone(),
            },
            case,
        }))
    }
    /// Called before the learner consumes this game's positions.
    pub fn observe(
        &self,
        index: usize,
        game: &collector::Played,
        ladder: &ladder::Shared,
    ) -> Result<()> {
        let Some(a) = &game.evaluation else {
            return Ok(());
        };
        let mut slot = self.slots[index].lock().unwrap();
        if a.slot >= 100
            || a.lot != slot.saved.lot
            || a.model != slot.saved.model
            || a.reference != slot.saved.reference
            || a.budget != slot.saved.budget
            || a.version != slot.saved.version
            || a.prefix != self.starts[a.slot / 2].identity
            || game.snapshot.identity != a.model
            || game.reference_budget != Some(a.budget)
            || !game.measurement
            || game.reanalysis
            || game.candidate_seat
                != (if a.slot % 2 == 0 {
                    Player::Host
                } else {
                    Player::Guest
                })
            || sha256(
                cases::prefix(&game.record, game.prefix_decisions)
                    .to_string()
                    .as_bytes(),
            ) != a.prefix
        {
            return Err(invalid("frozen evaluation receipt identity mismatch"));
        }
        if slot.saved.results[a.slot].is_some() {
            return Err(invalid("duplicate frozen evaluation result"));
        }
        slot.reserved[a.slot] = false;
        if game.error.is_some() || game.campaign_censored || game.termination == "user-stop" {
            return Ok(());
        }
        slot.saved.results[a.slot] = Some(match game.outcome {
            GameOutcome::Win(p) if p == game.candidate_seat => 1,
            GameOutcome::Win(_) => -1,
            GameOutcome::Draw => 0,
            _ => 2,
        });
        if slot.saved.results.iter().all(Option::is_some) {
            let mut state = ladder.write().unwrap();
            state.record_frozen(
                &slot.saved.results,
                a.budget,
                a.version,
                &a.model,
                &a.reference,
                a.lot,
            )?;
            slot.saved.last_completed = slot.saved.model.clone();
            slot.saved.model.clear();
            self.retired.lock().unwrap().push(slot.saved.path.clone());
            slot.snapshot = None;
        }
        Ok(())
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::to_value(
            self.slots
                .iter()
                .map(|s| s.lock().unwrap().saved.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    pub fn committed(&self) -> Result<()> {
        for p in self.retired.lock().unwrap().drain(..) {
            if p.starts_with(&self.out) {
                fs::remove_file(p)?;
            }
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests;
