//! Final actor transaction: learned choices and the accepted fresh contract.
use super::*;

impl Guard {
    pub fn enable_transfer(&mut self) -> Result<()> {
        if !self.state.learning_loop_v3 || self.parallel.is_none() {
            return Err(invalid(
                "publication transfer requires V3 and native worker pool",
            ));
        }
        let mut registry = if self.state.learned_choices.is_null() {
            if self.state.publication_transfer {
                return Err(invalid("missing durable learned choices"));
            }
            learned_choices::Registry::new(self.accepted.identity.clone(), &self.accepted.model)?
        } else {
            learned_choices::Registry::restore(
                &self.state.learned_choices,
                &self.accepted.identity,
                &self.accepted.model,
            )?
        };
        // Three frozen complete-publication ABBA comparisons retain every
        // result bit and reduce preparation + publication cost by 15.96%.
        // Restore still verifies the durable obligations before caching them.
        registry.configure_read_optimization(true, self.parallel.as_ref());
        self.state.publication_transfer = true;
        self.state.learned_choices = registry.checkpoint()?;
        self.learned_choices = Some(registry);
        Ok(())
    }
    pub fn set_publication_fresh(
        &mut self,
        contract: Option<(Vec<Arc<MicroExample>>, f64)>,
    ) -> Result<()> {
        if !self.state.publication_transfer {
            return Err(invalid("fresh transmission is not enabled"));
        }
        self.publication_fresh = contract
            .map(|(rows, ceiling)| {
                if rows.len() > 64 || !ceiling.is_finite() {
                    return Err(invalid("invalid bounded publication fresh contract"));
                }
                Ok(policy_transfer::FreshLimit { rows, ceiling })
            })
            .transpose()?;
        Ok(())
    }
    pub fn observe_consumed(
        &mut self,
        game: &collector::Played,
        saved: &[SavedMicroExample],
        fully_learned: bool,
        source_run: &str,
        updates: u64,
    ) -> Result<()> {
        if !fully_learned || !self.state.publication_transfer {
            return Ok(());
        }
        let book = self
            .learned_choices
            .as_mut()
            .ok_or_else(|| invalid("missing learned registry"))?;
        let mut changed = false;
        for (decision, certificate) in &game.certificates {
            let Some(row) = saved.iter().find(|s| s.decision == *decision) else {
                continue;
            };
            if let Some(proof) = learned_choices::ConsumedProof::from_played_target(
                game,
                row,
                *decision,
                certificate,
                source_run,
                updates,
            )? {
                changed |= book
                    .admit_consumed(proof, true, &self.accepted.model)?
                    .inserted;
            }
        }
        if changed {
            self.state.learned_choices = book.checkpoint()?;
        }
        Ok(())
    }
    pub(super) fn fresh_accepts(&self, model: &MicroModel) -> Result<bool> {
        self.publication_fresh
            .as_ref()
            .map_or(Ok(true), |f| Ok(f.loss(model, self)? <= f.ceiling + 1e-12))
    }
    /// Runs before Guard::publish: neither the anchor nor the registry changes.
    pub(super) fn transmit(
        &self,
        seed: Arc<Snapshot>,
        teacher: &Arc<Snapshot>,
    ) -> Result<(Option<(Arc<Snapshot>, Vec<Score>)>, serde_json::Value)> {
        let started = Instant::now();
        let book = self
            .learned_choices
            .as_ref()
            .ok_or_else(|| invalid("missing transfer registry"))?;
        let active = book.active_rows();
        let pending = book.pending_rows();
        let mut indexed = active.clone();
        indexed.extend(pending.clone());
        let mut rows = indexed.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>();
        let mut before = policy_transfer::readings(&seed.model, &rows)?;
        let taught = policy_transfer::readings(&teacher.model, &rows[active.len()..])?;
        let new_target =
            (active.len()..rows.len()).find(|&i| !before[i].raw && taught[i - active.len()].raw);
        let seed_fresh_pass = self.fresh_accepts(&seed.model)?;
        let mut target = new_target;
        if target.is_none() && !seed_fresh_pass {
            // Fresh-only repair still retains a real old proved choice. It is
            // never reported as a new acquisition of that already correct row.
            target = before.iter().position(|r| r.raw && r.bad.is_some());
            if target.is_none() {
                if let Some((i, row)) = self
                    .rows
                    .iter()
                    .enumerate()
                    .find(|(i, r)| self.score.raw[*i] && r.valid.iter().any(|v| !*v))
                {
                    rows.push(row.clone());
                    indexed.push((format!("fixed-primary-{i}"), row.clone()));
                    before = policy_transfer::readings(&seed.model, &rows)?;
                    target = Some(rows.len() - 1);
                }
            }
        }
        let (model, detail) = if let Some(target) = target {
            policy_transfer::relay(
                self,
                &seed.model,
                &rows,
                &before,
                target,
                true,
                true,
                6,
                self.publication_fresh.as_ref(),
            )?
        } else {
            (
                seed.model.as_ref().clone(),
                serde_json::json!({"accepted":false,"reason":"no eligible lost learned choice; fresh contract retained"}),
            )
        };
        let fresh_pass = self.fresh_accepts(&model)?;
        let (scores, reasons) = self.checked_v3(&model)?;
        let admissible = fresh_pass && reasons.is_empty();
        let changed = model
            .parameters()
            .iter()
            .zip(seed.model.parameters())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        let target_key = new_target.map(|i| indexed[i].0.clone());
        let summary = serde_json::json!({"enabled":true,"attempted":target.is_some(),"relayed":detail["accepted"],
            "admissible":admissible,"new_target":target_key,"fresh_only":new_target.is_none()&&target.is_some(),
            "seed":seed.identity,"teacher":teacher.identity,"seed_fresh_pass":seed_fresh_pass,"fresh_pass":fresh_pass,
            "fresh_ceiling":self.publication_fresh.as_ref().map(|f|f.ceiling),"reasons":reasons,
            "corrections":detail["corrections"],"candidate_full_checks":detail["candidate_full_checks"],
            "kl_gradient_rows":detail["kl_gradient_rows"],"fresh_gradient_rows":detail["fresh_gradient_rows"],
            "relay_seconds":detail["total_seconds"],"total_seconds":started.elapsed().as_secs_f64(),
            "old_choices":active.len(),"pending_choices":pending.len(),"changed_seed":changed,
            "value_parameters_exact_seed":true,"private_stop":detail["stop"]});
        if !admissible {
            return Ok((None, summary));
        }
        let final_actor = if changed {
            let updates = teacher
                .artifact
                .as_ref()
                .ok_or_else(|| invalid("transfer teacher has no native artifact"))?
                .updates;
            let artifact = Arc::new(MicroArtifact::new(
                &model,
                updates,
                serde_json::json!({
                "kind":"gen5-acquired-policy-transfer-v1","seed":seed.identity,"teacher":teacher.identity,"transfer":summary}),
            ));
            let identity = artifact.identity();
            Arc::new(Snapshot {
                model: Arc::new(model),
                version: teacher.version,
                path: self
                    .out
                    .join("accepted")
                    .join(format!("model-{identity}.json")),
                identity,
                artifact: Some(artifact),
            })
        } else {
            seed
        };
        Ok((Some((final_actor, scores)), summary))
    }
    pub(super) fn merge_transfer_feedback(
        &self,
        old: Vec<Arc<MicroExample>>,
    ) -> Vec<Arc<MicroExample>> {
        let Some(book) = &self.learned_choices else {
            return old;
        };
        let mut dynamic = book.pending_rows();
        dynamic.extend(book.active_rows());
        merge_feedback_examples(
            old,
            dynamic.iter().map(|(_, row)| row.example.clone()),
            self.state.feedback_cursor,
        )
    }
}

fn merge_feedback_examples(
    old: Vec<Arc<MicroExample>>,
    dynamic: impl Iterator<Item = Arc<MicroExample>> + Clone + ExactSizeIterator,
    cursor: usize,
) -> Vec<Arc<MicroExample>> {
    let count = dynamic.len();
    if count == 0 {
        return old;
    }
    // The shared cursor advances by 32. Interleaving can leave only 16 dynamic
    // slots, so advance this window by 16 to cover a stable population fully.
    let mut extra = dynamic.cycle().skip((cursor / 2) % count).take(count);
    let mut old = old.into_iter();
    let mut out = vec![];
    loop {
        let pair = [old.next(), extra.next()];
        if pair.iter().all(Option::is_none) {
            break;
        }
        for row in pair.into_iter().flatten() {
            if !out
                .iter()
                .any(|e: &Arc<MicroExample>| e.state == row.state && e.actions == row.actions)
            {
                out.push(row);
            }
            if out.len() == 32 {
                return out;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn publication_transfer_feedback_covers_stable_64_and_128_with_fixed_rows() {
        // Only the merge/dedup coordinates matter; no inference or rule replay.
        let row = |id: f64| Arc::new(MicroExample { structured: Vec::new(),
            state: vec![id], actions: vec![], policy: vec![], value: 0.,
            policy_weight: 1., value_weight: 0., action_values: vec![],
            sequence_source: 0, policy_support: true,
        });
        for count in [64, 128] {
            let dynamic = (0..count).map(|i| row(i as f64)).collect::<Vec<_>>();
            for fixed_count in [0, 16, 32] {
                let old = (0..fixed_count).map(|i| row(-1. - i as f64)).collect::<Vec<_>>();
                let mut seen = BTreeSet::new();
                let mut legacy_seen = BTreeSet::new();
                for publication in 1..=count / 16 {
                    let cursor = publication * 32;
                    let merged = merge_feedback_examples(old.clone(), dynamic.iter().cloned(), cursor);
                    assert_eq!(merged.len(), 32);
                    let ids = merged.iter().filter(|r| r.state[0] >= 0.)
                        .map(|r| r.state[0] as usize).collect::<Vec<_>>();
                    assert_eq!(ids.len(), if fixed_count == 0 {32} else {16});
                    seen.extend(ids);
                    legacy_seen.extend((0..16).map(|offset| (cursor + offset) % count));
                }
                assert_eq!(seen, (0..count).collect());
                // Explicit counterexample: the old stride skips half the set
                // when at least 16 fixed rows consume the other output slots.
                if fixed_count >= 16 {
                    assert_eq!(legacy_seen.len(), count / 2);
                }
            }
        }
    }
}
