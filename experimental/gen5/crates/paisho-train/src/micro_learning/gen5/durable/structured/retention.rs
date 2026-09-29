//! Signed rare-event coverage and a bounded, persistent seed reservation.
use super::*;
const ANCHOR_CAPACITY: usize = 128;
const ANCHOR_BYTES: usize = BYTE_CAP / 2;
const MIN_CLASS_POSITIONS: usize = 4;

fn merged(old: &MicroExample, new: &MicroExample, pinned: bool) -> Result<Arc<MicroExample>> {
    if old.state != new.state || old.actions != new.actions {
        return Err(invalid("same structured key has different state/actions"));
    }
    // Seeds retain their original proved P/V teaching. Ordinary rows retain the
    // incoming P/V/Q provenance; only exact structured evidence is combined.
    let mut out = if pinned { old.clone() } else { new.clone() };
    out.structured = (0..old.actions.len())
        .map(|i| {
            match (
                old.structured.get(i).and_then(Option::as_ref),
                new.structured.get(i).and_then(Option::as_ref),
            ) {
                (Some(a), Some(b)) => a.merge_evidence(b).map(Some).map_err(invalid),
                (Some(t), None) | (None, Some(t)) => Ok(Some(t.clone())),
                (None, None) => Ok(None),
            }
        })
        .collect::<Result<_>>()?;
    Ok(Arc::new(out))
}

impl Structured {
    /// Plan all removals before mutating the bank. Group limits never fall back
    /// to erasing the final counterexample; decline the arrival instead.
    pub(super) fn admit_checked(
        &mut self,
        key: String,
        mut group: String,
        mut example: Arc<MicroExample>,
        seed: bool,
    ) -> Result<bool> {
        if example.structured.iter().all(Option::is_none) {
            return Ok(false);
        }
        example.validate().map_err(invalid)?;
        let replacing = self.rows.iter().position(|r| r.key == key);
        let pinned = seed || self.anchors.contains(&key);
        if let Some(i) = replacing {
            group = self.rows[i].group.clone();
            example = merged(&self.rows[i].example, &example, self.anchors.contains(&key))?;
        }
        let mut mask = [false; 20];
        for t in example.structured.iter().flatten() {
            for (j, event) in t.events.iter().enumerate() {
                if let Some(v) = event {
                    mask[2 * j + usize::from(*v)] = true;
                }
            }
        }
        if !mask.iter().any(|v| *v) {
            return Ok(false);
        }
        let bytes = example.state.len() * 8
            + example.actions.len() * 32 * 8
            + example.policy.len() * 8
            + example.action_values.len() * 16
            + example.structured.len() * 256
            + key.len()
            + group.len();
        if pinned {
            let anchors = self
                .rows
                .iter()
                .filter(|r| self.anchors.contains(&r.key) && r.key != key);
            let (n, b) = anchors.fold((1, bytes), |(n, b), r| (n + 1, b + r.bytes));
            if n > ANCHOR_CAPACITY || b > ANCHOR_BYTES {
                return Err(invalid("structured seed reservation exceeded"));
            }
        }
        if bytes > BYTE_CAP {
            self.rejected_capacity += 1;
            return Ok(false);
        }
        let before: [usize; 20] =
            std::array::from_fn(|j| self.rows.iter().filter(|r| r.mask[j]).count());
        let floors = before.map(|n| n.min(MIN_CLASS_POSITIONS));
        let mut counts = before;
        let mut removed = vec![false; self.rows.len()];
        let mut total_bytes = self.bytes + bytes;
        let mut total = self.rows.len() + 1;
        let mut same_group = self.rows.iter().filter(|r| r.group == group).count() + 1;
        if let Some(i) = replacing {
            removed[i] = true;
            total -= 1;
            total_bytes -= self.rows[i].bytes;
            same_group -= 1;
            for j in 0..20 {
                counts[j] -= usize::from(self.rows[i].mask[j]);
            }
        }
        for j in 0..20 {
            counts[j] += usize::from(mask[j]);
        }
        while same_group > GROUP_CAP || total > CAPACITY || total_bytes > BYTE_CAP {
            let group_full = same_group > GROUP_CAP;
            let victim = self
                .rows
                .iter()
                .enumerate()
                .find(|(i, r)| {
                    !removed[*i]
                        && !self.anchors.contains(&r.key)
                        && (!group_full || r.group == group)
                        && (0..20).all(|j| !r.mask[j] || counts[j] > floors[j])
                })
                .map(|(i, _)| i);
            let Some(i) = victim else {
                self.rejected_capacity += 1;
                return Ok(false);
            };
            let row = &self.rows[i];
            removed[i] = true;
            total -= 1;
            total_bytes -= row.bytes;
            same_group -= usize::from(row.group == group);
            for j in 0..20 {
                counts[j] -= usize::from(row.mask[j]);
            }
        }
        for i in (0..removed.len()).rev().filter(|&i| removed[i]) {
            self.remove(i);
        }
        if pinned {
            self.anchors.insert(key.clone());
        }
        self.rows.push_back(Row {
            key,
            group,
            example,
            mask,
            bytes,
        });
        self.bytes += bytes;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ex(positive: bool, threat: bool) -> Arc<MicroExample> {
        let mut e = super::super::tests::row(positive).as_ref().clone();
        if threat {
            let t = e.structured[0].as_mut().unwrap();
            t.events[5] = Some(false);
            t.threat = MicroThreatEvidence::Absent { examined: 3 };
        }
        Arc::new(e)
    }
    #[test]
    fn negative_and_positive_coverage_survive_many_capacities_and_group_pressure() {
        let mut bank = Structured::default();
        for i in 0..8 {
            bank.admit(format!("rare{i}"), "crowded".into(), ex(i % 2 == 0, true));
        }
        for i in 0..(CAPACITY * 4) {
            bank.admit(
                format!("common{i}"),
                if i % 2 == 0 {
                    "crowded".into()
                } else {
                    i.to_string()
                },
                ex(false, false),
            );
        }
        assert!(bank.rows.iter().filter(|r| r.mask[1]).count() >= 4);
        assert!(bank.rows.iter().filter(|r| r.mask[10]).count() >= 4);
        assert!(bank.rows.iter().filter(|r| r.group == "crowded").count() <= GROUP_CAP);
        assert!(bank.rows.len() <= CAPACITY && bank.bytes <= BYTE_CAP);
    }
    #[test]
    fn partial_arrival_merges_known_threat_and_conflict_is_atomic() {
        let mut bank = Structured::default();
        bank.admit("same".into(), "group".into(), ex(true, true));
        bank.admit("same".into(), "group".into(), ex(true, false));
        assert_eq!(
            bank.rows[0].example.structured[0].as_ref().unwrap().events[5],
            Some(false)
        );
        let old = bank.rows[0].example.clone();
        let bytes = bank.bytes;
        bank.admit("same".into(), "group".into(), ex(false, false));
        assert!(Arc::ptr_eq(&old, &bank.rows[0].example));
        assert_eq!(bytes, bank.bytes);
        assert_eq!(bank.rejected_conflicts, 1);
    }
    #[test]
    fn anchors_survive_overflow_and_exact_checkpoint_continuation() {
        let mut a = Structured::default();
        for i in 0..128 {
            assert!(a
                .admit_checked(
                    format!("seed{i}"),
                    i.to_string(),
                    ex(i % 2 == 0, true),
                    true
                )
                .unwrap());
        }
        for i in 0..1024 {
            a.admit(format!("new{i}"), i.to_string(), ex(false, false));
        }
        assert_eq!(a.anchors.len(), 128);
        assert!(a.anchors.iter().all(|k| a.rows.iter().any(|r| &r.key == k)));
        let path = std::env::temp_dir().join(format!("gen5-recall-pins-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        a.draw(71);
        a.checkpoint(&path, true, &mut vec![]).unwrap();
        let mut b = Structured::default();
        b.restore(&a.progress().unwrap(), true).unwrap();
        assert_eq!(a.anchors, b.anchors);
        assert_eq!(a.bytes, b.bytes);
        for _ in 0..80 {
            assert_eq!(
                a.draw(7).iter().map(|e| &e.structured).collect::<Vec<_>>(),
                b.draw(7).iter().map(|e| &e.structured).collect::<Vec<_>>()
            );
        }
        fs::remove_dir_all(path).unwrap();
    }
}
