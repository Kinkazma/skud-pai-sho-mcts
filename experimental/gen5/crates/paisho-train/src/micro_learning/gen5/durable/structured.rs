//! Bounded stratified recall, inside (not in addition to) the caller's recall quota.
use super::*;
use std::collections::VecDeque;
mod retention;
mod repair_probe;
pub use repair_probe::run as probe_recall_repair;
const CAPACITY: usize = 512;
const GROUP_CAP: usize = 16;
const BYTE_CAP: usize = 32 * 1024 * 1024;
pub(super) fn decode(
    p: &paisho_core::Position,
    legal: &[paisho_core::Action],
    labels: &[(String, MicroStructuredTarget)],
) -> Result<Vec<Option<MicroStructuredTarget>>> {
    if labels.is_empty() {
        return Ok(vec![]);
    }
    let before = MicroRelations::extract(p, p.to_move());
    let mut out = vec![None; legal.len()];
    for (name, t) in labels {
        t.validate().map_err(invalid)?;
        let a: paisho_core::Action = name.parse()?;
        let i = legal
            .iter()
            .position(|b| *b == a)
            .ok_or_else(|| invalid("structured action illegal"))?;
        if out[i].is_some() {
            return Err(invalid("duplicate structured action"));
        }
        let mut q = p.clone();
        q.apply(a)?;
        let threat = match &t.threat {
            MicroThreatEvidence::Present { reply } => {
                micro_verify_threat_witness(&q, p.to_move(), reply.parse()?).map_err(invalid)?
            }
            MicroThreatEvidence::Absent { examined } => {
                let actual =
                    micro_immediate_threat(&q, p.to_move(), usize::MAX).map_err(invalid)?;
                if actual
                    != (MicroImmediateThreat::Absent {
                        examined: *examined,
                    })
                {
                    return Err(invalid("invalid complete threat census"));
                }
                actual
            }
            _ => micro_immediate_threat(&q, p.to_move(), 0).map_err(invalid)?,
        };
        let verified = MicroStructuredTarget::from_successor(p, a, &q, &before, &threat);
        if verified != *t {
            return Err(invalid("structured consequences differ from legal replay"));
        }
        out[i] = Some(t.clone());
    }
    Ok(out)
}
#[derive(Clone)]
struct Row {
    key: String,
    group: String,
    example: Arc<MicroExample>,
    mask: [bool; 20],
    bytes: usize,
}
#[derive(Clone, Default)]
pub(super) struct Structured {
    rows: VecDeque<Row>,
    bytes: usize,
    bucket: usize,
    cursor: [usize; 20],
    credit: usize,
    checkpoint: serde_json::Value,
    anchors: std::collections::BTreeSet<String>,
    rejected_conflicts: usize,
    rejected_capacity: usize,
}
#[derive(Serialize, Deserialize)]
struct Saved {
    rows: Vec<(String, String, super::super::resume_example::ResumeExample)>,
    bucket: usize,
    cursor: [usize; 20],
    credit: usize,
    #[serde(default)]
    anchors: std::collections::BTreeSet<String>,
    #[serde(default)]
    rejected_conflicts: usize,
    #[serde(default)]
    rejected_capacity: usize,
}
/// Offline seed admission replays every action and complete negative census.
/// It fills an existing recall quota; it neither appends FIFO rows nor runs SGD.
pub(in crate::micro_learning::gen5) fn prepare_seed_recall(
    examples: &[SavedMicroExample],
    sources: &[serde_json::Value],
    model: &MicroModel,
    previous: &serde_json::Value,
    out: &Path,
) -> Result<serde_json::Value> {
    if examples.len() != sources.len() || examples.is_empty() {
        return Err(invalid("unaligned structured seed sources"));
    }
    let mut bank = Structured::default();
    bank.restore(previous, true)?;
    for (s, source) in examples.iter().zip(sources) {
        let bytes = fs::read(
            source["psr"]
                .as_str()
                .ok_or_else(|| invalid("seed PSR missing"))?,
        )?;
        if sha256(&bytes) != source["sha256"] {
            return Err(invalid("seed PSR hash changed"));
        }
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        if record.rules() != RULES {
            return Err(invalid("seed rules mismatch"));
        }
        let p = record.replay()?;
        let legal = paisho_core::legal_actions(&p);
        let e = s.example_for_rules_with_trusted_q(RULES, true)?;
        if p.outcome() != GameOutcome::Ongoing
            || s.decision != record.actions().len() + 1
            || e.state != model.state_features(&p)
            || s.actions != legal.iter().map(ToString::to_string).collect::<Vec<_>>()
            || e.actions
                != legal
                    .iter()
                    .map(|&a| micro_action_features(&p, a))
                    .collect::<Vec<_>>()
        {
            return Err(invalid("structured seed differs from its legal source"));
        }
        let pairs = s
            .actions
            .iter()
            .cloned()
            .zip(s.structured.iter().cloned())
            .filter_map(|(a, t)| t.map(|t| (a, t)))
            .collect::<Vec<_>>();
        if decode(&p, &legal, &pairs)? != e.structured {
            return Err(invalid("seed consequences changed"));
        }
        if e.policy_weight > 0. {
            if !e.policy_support
                || e.policy.iter().zip(&legal).any(|(mass, &a)| {
                    let mut q = p.clone();
                    q.apply(a).unwrap();
                    *mass > 0. && q.outcome() != GameOutcome::Win(p.to_move())
                })
            {
                return Err(invalid("seed policy lacks exact winning support"));
            }
        }
        if e.value_weight > 0. && (e.value != 1. || e.policy_weight <= 0.) {
            return Err(invalid("seed value is not proved"));
        }
        if !bank.admit_checked(
            sha256(record.to_string().as_bytes()),
            source["group"]
                .as_str()
                .ok_or_else(|| invalid("seed group missing"))?
                .into(),
            Arc::new(e),
            true,
        )? {return Err(invalid("structured seed cannot fit its bounded reservation"));}
    }
    fs::create_dir_all(out)?;
    bank.checkpoint(out, true, &mut vec![])?;
    bank.progress()
        .ok_or_else(|| invalid("empty structured recall after seeding"))
}
impl Structured {
    /// Explicit recovery from a hashed, previously verified seed checkpoint.
    pub(super) fn import_anchors(&mut self, source: &StructuredRecallAnchors) -> Result<()> {
        let mut seeds=Self::default();
        seeds.restore(&serde_json::json!({"checkpoint":{"path":source.path,"sha256":source.sha256}}),true)?;
        if seeds.rows.is_empty() {return Err(invalid("empty structured seed checkpoint"));}
        let mut repaired=self.clone();
        for row in seeds.rows {
            if repaired.anchors.contains(&row.key) {continue;}
            if !repaired.admit_checked(row.key,row.group,row.example,true)? {
                return Err(invalid("seed recovery cannot fit the bounded recall bank"));
            }
        }
        *self=repaired;
        Ok(())
    }
    pub fn admit(&mut self, key: String, group: String, example: Arc<MicroExample>) {
        // Invalid/conflicting arrivals cannot erase already verified evidence.
        // Record rejection without failing otherwise valid campaign learning.
        if self.admit_checked(key, group, example, false).is_err() {
            self.rejected_conflicts += 1;
        }
    }
    fn remove(&mut self, i: usize) {
        if let Some(row) = self.rows.remove(i) {
            self.bytes -= row.bytes;
        }
    }
    pub fn draw(&mut self, n: usize) -> Vec<Arc<MicroExample>> {
        if self.rows.is_empty() {
            return vec![];
        }
        self.credit += n;
        let requested = self.credit / 4;
        self.credit %= 4;
        let mut out = Vec::with_capacity(requested);
        for _ in 0..requested {
            for _ in 0..20 {
                let bucket = self.bucket % 20;
                self.bucket = self.bucket.wrapping_add(1);
                let mut groups: Vec<&str> = self
                    .rows
                    .iter()
                    .filter(|r| r.mask[bucket])
                    .map(|r| r.group.as_str())
                    .collect();
                groups.sort_unstable();
                groups.dedup();
                if groups.is_empty() {
                    continue;
                }
                let cursor = self.cursor[bucket];
                self.cursor[bucket] = cursor.wrapping_add(1);
                let group = groups[cursor % groups.len()];
                let rows: Vec<_> = self
                    .rows
                    .iter()
                    .filter(|r| r.mask[bucket] && r.group == group)
                    .collect();
                out.push(rows[(cursor / groups.len()) % rows.len()].example.clone());
                break;
            }
        }
        out
    }
    pub fn progress(&self) -> Option<serde_json::Value> {
        if self.rows.is_empty() {
            None
        } else {
            Some(serde_json::json!({"positions":self.rows.len(),
            "bytes":self.bytes,"credit":self.credit,"checkpoint":self.checkpoint,
            "retention":"signed-coverage-and-bounded-anchors-v1","anchors":self.anchors.len(),
            "rejected_conflicts":self.rejected_conflicts,"rejected_capacity":self.rejected_capacity,
            "bucket_populations":(0..20).map(|j|self.rows.iter().filter(|r|r.mask[j]).count()).collect::<Vec<_>>()}))
        }
    }
    pub fn checkpoint(
        &mut self,
        out: &Path,
        trusted: bool,
        retired: &mut Vec<PathBuf>,
    ) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let data = Saved {
            rows: self
                .rows
                .iter()
                .map(|r| {
                    (
                        r.key.clone(),
                        r.group.clone(),
                        super::super::resume_example::ResumeExample::from_with_trusted_q(
                            &r.example, trusted,
                        ),
                    )
                })
                .collect(),
            bucket: self.bucket,
            cursor: self.cursor,
            credit: self.credit,
            anchors: self.anchors.clone(),
            rejected_conflicts: self.rejected_conflicts,
            rejected_capacity: self.rejected_capacity,
        };
        let hash = sha256(&serde_json::to_vec(&data)?);
        let path = out.join(format!("structured-recall-{hash}.json"));
        if !path.exists() {
            write(&path, &data)?;
        }
        if let Some(old) = self.checkpoint["path"].as_str() {
            let old = PathBuf::from(old);
            if old != path && old.starts_with(out) {
                retired.push(old);
            }
        }
        self.checkpoint = serde_json::json!({"path":path,"sha256":hash});
        Ok(())
    }
    pub fn restore(&mut self, progress: &serde_json::Value, trusted: bool) -> Result<()> {
        let saved = &progress["checkpoint"];
        let Some(path) = saved["path"].as_str() else {
            return Ok(());
        };
        let bytes = fs::read(path)?;
        if sha256(&bytes) != saved["sha256"].as_str().unwrap_or("") {
            return Err(invalid("structured recall changed"));
        }
        let data: Saved = serde_json::from_slice(&bytes)?;
        if data.credit >= 4 || data.rows.len() > CAPACITY {
            return Err(invalid("invalid structured recall clock/capacity"));
        }
        let mut restored = Self::default();
        for (key, group, example) in data.rows {
            let count = restored.rows.len();
            let pinned = data.anchors.contains(&key);
            if restored.rows.iter().any(|r|r.key==key)
                || !restored.admit_checked(key, group, example.example_with_trusted_q(trusted)?, pinned)? {
                return Err(invalid("structured recovery cannot preserve every row"));
            }
            if restored.rows.len() != count + 1 {return Err(invalid("structured recovery would evict saved rows"));}
        }
        if restored.anchors != data.anchors {return Err(invalid("missing structured anchor"));}
        restored.bucket = data.bucket;
        restored.cursor = data.cursor;
        restored.credit = data.credit;
        restored.checkpoint = saved.clone();
        restored.rejected_conflicts = data.rejected_conflicts;
        restored.rejected_capacity = data.rejected_capacity;
        *self = restored;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub(super) fn row(value: bool) -> Arc<MicroExample> {
        let mut e = super::super::super::super::tactics::fixture()
            .example_for_rules(RULES)
            .unwrap();
        let mut events = [None; 10];
        events[0] = Some(value);
        e.structured = vec![None; e.actions.len()];
        e.structured[0] = Some(MicroStructuredTarget {
            counts: [0.; 10],
            events,
            threat: MicroThreatEvidence::Unknown,
        });
        Arc::new(e)
    }
    #[test]
    fn quota_is_internal_and_checkpoint_continuation_exact() {
        let path =
            std::env::temp_dir().join(format!("paisho-structured-recall-{}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        let mut a = Structured::default();
        a.admit("positive".into(), "a".into(), row(true));
        a.admit("negative".into(), "b".into(), row(false));
        assert!(a.draw(3).is_empty());
        assert_eq!(a.draw(1).len(), 1);
        a.draw(3);
        a.checkpoint(&path, false, &mut vec![]).unwrap();
        let mut b = Structured::default();
        b.restore(&a.progress().unwrap(), false).unwrap();
        for _ in 0..40 {
            let x = a.draw(3);
            let y = b.draw(3);
            assert_eq!(x.len(), y.len());
            for (x, y) in x.iter().zip(y) {
                assert_eq!(x.structured, y.structured);
            }
        }
        assert_eq!(a.progress().unwrap()["bucket_populations"][10], 0);
        fs::remove_dir_all(path).unwrap();
    }
    #[test]
    fn one_source_cannot_fill_recall_bank() {
        let mut a = Structured::default();
        for i in 0..100 {
            a.admit(i.to_string(), "same".into(), row(i % 2 == 0));
        }
        assert_eq!(a.rows.len(), GROUP_CAP);
        assert!(a.bytes <= BYTE_CAP);
    }
}
