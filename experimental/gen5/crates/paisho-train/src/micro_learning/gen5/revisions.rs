//! An estimate adds a teaching proposal; it cannot erase empirical outcomes.
use super::*;
use durable::Lesson;
use std::collections::BTreeMap;
#[derive(Serialize, Deserialize)]
struct Revision {
    rules: String,
    prefix_sha256: String,
    version: u64,
    bundle: PathBuf,
    lesson: Lesson,
    lesson_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proof: Option<Lesson>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    estimate: Option<Lesson>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    observed: Option<Lesson>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    outcomes: BTreeMap<String, f64>,
}
fn proof(l: &Lesson) -> bool {
    l.reason == "search-proven-value"
}
fn observed(l: &Lesson) -> bool {
    l.evidence
        .as_ref()
        .is_some_and(|e| e.observed_value.is_some())
        || [
            "rules-terminal-z",
            "observed-outcome-with-search-policy",
            "empirical-outcome-mean",
            "reanalysis-with-observed-outcome",
        ]
        .contains(&l.reason.as_str())
}
fn verify(r: &Revision, key: &str) -> Result<()> {
    if r.rules != RULES.as_str()
        || r.prefix_sha256 != key
        || sha256(&serde_json::to_vec(&r.lesson)?) != r.lesson_sha256
        || r.outcomes.values().any(|v| !v.is_finite() || v.abs() > 1.)
    {
        return Err(invalid("durable revision identity mismatch"));
    }
    Ok(())
}
impl Revision {
    fn merge(&mut self, incoming: Lesson, id: String, version: u64) -> Result<()> {
        let structured=merge_structured(&self.lesson.structured,&incoming.structured)?;
        if self.proof.is_none() && proof(&self.lesson) {
            self.proof = Some(self.lesson.clone());
        }
        if self.outcomes.is_empty() && observed(&self.lesson) {
            if let Some(e) = &self.lesson.evidence {
                if let Some(z) = e.observed_value {
                    self.outcomes.insert(e.observed_psr.clone().unwrap(), z);
                }
            } else {
                self.outcomes.insert(
                    format!("legacy-bundle:{}", self.bundle.display()),
                    self.lesson.value,
                );
            }
            self.observed = Some(self.lesson.clone());
        }
        if proof(&incoming) {
            self.proof = Some(incoming.clone());
        }
        if let Some(e) = &incoming.evidence {
            if let Some(z) = e.observed_value {
                self.outcomes
                    .insert(e.observed_psr.clone().unwrap_or(id), z);
                self.observed = Some(incoming.clone());
            }
        }
        if !proof(&incoming) && incoming.policy_weight > 0. && version >= self.version {
            self.estimate = Some(incoming.clone());
        }
        self.lesson = self
            .proof
            .clone()
            .or_else(|| self.estimate.clone())
            .or_else(|| self.observed.clone())
            .unwrap_or(incoming);
        if self.proof.is_none() && !self.outcomes.is_empty() {
            self.lesson.value = self.outcomes.values().sum::<f64>() / self.outcomes.len() as f64;
            self.lesson.reason = "observed-outcome-with-search-policy".into();
            if let Some(e) = &mut self.lesson.evidence {
                e.value_weight = 1.;
            }
        }
        self.lesson.structured=structured;
        self.version = self.version.max(version);
        Ok(())
    }
}
pub(super) fn store(
    root: &Path,
    game: &collector::Played,
    saved: &[SavedMicroExample],
    bundle: &Path,
) -> Result<()> {
    let directory = root.join("revisions");
    fs::create_dir_all(&directory)?;
    for s in saved.iter().filter(|s| {
        game.reanalysis || s.decision == game.prefix_decisions + 1 || s.correction_priority || !s.structured.is_empty()
    }) {
        let key = sha256(
            cases::prefix(&game.record, s.decision - 1)
                .to_string()
                .as_bytes(),
        );
        let path = directory.join(format!("{key}.json"));
        let incoming = Lesson::from_saved(s);
        let mut r = if let Some(bytes) = read_revision(&path)? {
            let r: Revision = serde_json::from_slice(&bytes)?;
            verify(&r, &key)?;
            r
        } else {
            Revision {
                rules: RULES.to_string(),
                prefix_sha256: key.clone(),
                version: game.snapshot.version,
                bundle: bundle.to_path_buf(),
                lesson: incoming.clone(),
                lesson_sha256: String::new(),
                proof: None,
                estimate: None,
                observed: None,
                outcomes: BTreeMap::new(),
            }
        };
        if game.loop_repair {
            r.merge(
                incoming,
                sha256(game.record.to_string().as_bytes()),
                game.snapshot.version,
            )?;
        } else {
            if r.version > game.snapshot.version {
                continue;
            }
            r.lesson = incoming;
            r.version = game.snapshot.version;
        }
        r.bundle = bundle.to_path_buf();
        r.lesson_sha256 = sha256(&serde_json::to_vec(&r.lesson)?);
        durable::write_pending(&path, &r)?;
    }
    Ok(())
}
// Historical release archives compress dense annotations without changing any
// JSON numeric token. New native revisions still supersede them atomically as JSON.
fn read_revision(path: &Path) -> Result<Option<Vec<u8>>> {
    if path.exists() { return Ok(Some(fs::read(path)?)); }
    let compressed = path.with_extension("json.gz");
    if !compressed.exists() { return Ok(None); }
    use std::io::Read;
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(fs::File::open(compressed)?).read_to_end(&mut bytes)?;
    Ok(Some(bytes))
}
pub(super) fn lookup(root: &Path, key: &str) -> Result<Option<Lesson>> {
    let path = root.join("revisions").join(format!("{key}.json"));
    let Some(bytes) = read_revision(&path)? else {return Ok(None);};
    let r: Revision = serde_json::from_slice(&bytes)?;
    verify(&r, key)?;
    Ok(Some(r.lesson))
}
pub(super) fn resolve(original: &Lesson, revision: Option<Lesson>, repaired: bool) -> Result<Lesson> {
    let mut l = revision.unwrap_or_else(|| original.clone());
    if repaired && observed(original) && !proof(&l) && !observed(&l) {
        l.value = original.value;
        l.reason = "observed-outcome-with-search-policy".into();
        if let Some(e) = &mut l.evidence {
            e.value_weight = 1.;
        }
    }
    let structured=merge_structured(&original.structured,&l.structured)?;
    if repaired && proof(original) && !proof(&l) { l=original.clone(); }
    l.structured=structured;
    Ok(l)
}
fn merge_structured(old:&[(String,MicroStructuredTarget)],new:&[(String,MicroStructuredTarget)])
    -> Result<Vec<(String,MicroStructuredTarget)>> {
    let mut out:BTreeMap<String,MicroStructuredTarget>=old.iter().cloned().collect();
    for (action,target) in new {
        let target=if let Some(previous)=out.get(action){previous.merge_evidence(target).map_err(invalid)?}else{target.validate().map_err(invalid)?;target.clone()};
        out.insert(action.clone(),target);
    }
    Ok(out.into_iter().collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compressed_revision_preserves_lesson_and_plain_updates_take_priority() {
        use std::io::Write;
        let dir=std::env::temp_dir().join(format!("portable-revision-{}",std::process::id()));
        fs::create_dir_all(dir.join("revisions")).unwrap();
        let key="portable-fixture";
        let l=lesson(-1.,Some(-1.),"a","rules-terminal-z");
        let mut r=Revision {rules:RULES.to_string(),prefix_sha256:key.into(),version:7,
            bundle:"fixture".into(),lesson:l.clone(),lesson_sha256:sha256(&serde_json::to_vec(&l).unwrap()),
            proof:None,estimate:None,observed:Some(l),outcomes:BTreeMap::new()};
        let bytes=serde_json::to_vec(&r).unwrap();
        let path=dir.join("revisions").join(format!("{key}.json"));
        let mut encoder=flate2::write::GzEncoder::new(Vec::new(),flate2::Compression::fast());
        encoder.write_all(&bytes).unwrap();
        fs::write(path.with_extension("json.gz"),encoder.finish().unwrap()).unwrap();
        assert_eq!(read_revision(&path).unwrap().unwrap(),bytes);
        assert_eq!(serde_json::to_vec(&lookup(&dir,key).unwrap().unwrap()).unwrap(),serde_json::to_vec(&r.lesson).unwrap());
        r.lesson=lesson(1.,Some(1.),"b","rules-terminal-z");
        r.lesson_sha256=sha256(&serde_json::to_vec(&r.lesson).unwrap());
        durable::write_pending(&path,&r).unwrap();
        assert_eq!(lookup(&dir,key).unwrap().unwrap().value,1.);
        fs::remove_file(&path).unwrap();
        fs::write(path.with_extension("json.gz"),b"corrupt").unwrap();
        assert!(lookup(&dir,key).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    fn lesson(z: f64, obs: Option<f64>, id: &str, reason: &str) -> Lesson {
        Lesson::from_saved(&SavedMicroExample { structured: Vec::new(),
            evidence: Some(TargetEvidence { policy_support:false,
                policy_coordinates: String::new(),
                search_prior: vec![],
                coupling_strength: None,
                observed_value: obs,
                observed_psr: obs.map(|_| id.repeat(64)),
                estimated_value: Some(z),
                value_weight: if obs.is_some() { 1. } else { 0.25 },
                policy_source: "full-search-estimate".into(),
                completed_action_values: vec![],
                action_value_visits: vec![],
                target_prior: vec![],
                excluded_actions: vec![],
                player: "H".into(),
                actor: "test".into(),
            }),
            rules: RULES.to_string(),
            source_run: "test".into(),
            game_id: "1".into(),
            decision: 1,
            collector: "a".repeat(64),
            budget: 512,
            inherited_visits: 0,
            new_visits: vec![],
            policy_raw_visits: vec![],
            policy_pruned_visits: vec![],
            tactical: None,
            correction_priority: false,
            actions: vec![],
            state: vec![],
            action_features: vec![],
            policy: vec![],
            value: z,
            policy_weight: 1.,
            reason: reason.into(),
        })
    }
    #[test]
    fn experience_survives_q_and_reanalysis_does_not_duplicate_outcomes() {
        let loss = lesson(-1., Some(-1.), "a", "rules-terminal-z");
        let mut r = Revision {
            rules: RULES.to_string(),
            prefix_sha256: "key".into(),
            version: 1,
            bundle: "bundle".into(),
            lesson: loss.clone(),
            lesson_sha256: String::new(),
            proof: None,
            estimate: None,
            observed: None,
            outcomes: BTreeMap::new(),
        };
        for i in 2..108 {
            r.merge(lesson(0.8, None, "b", "fresh-reanalysis-q"), "q".into(), i).unwrap();
            assert_eq!(r.lesson.value, -1.);
        }
        r.merge(
            lesson(0.9, Some(-1.), "a", "reanalysis-with-observed-outcome"),
            "q".into(),
            108,
        ).unwrap();
        assert_eq!(r.outcomes.len(), 1);
        assert_eq!(r.lesson.value, -1.);
        r.merge(
            lesson(1., Some(1.), "b", "rules-terminal-z"),
            "new".into(),
            109,
        ).unwrap();
        assert_eq!(r.outcomes.len(), 2);
        assert_eq!(r.lesson.value, 0.);
        r.merge(
            lesson(1., None, "c", "search-proven-value"),
            "proof".into(),
            110,
        ).unwrap();
        r.merge(
            lesson(-0.9, None, "c", "fresh-reanalysis-q"),
            "q".into(),
            111,
        ).unwrap();
        assert_eq!(r.lesson.value, 1.);
        assert_eq!(r.outcomes.len(), 2);
        let copy: Revision = serde_json::from_slice(&serde_json::to_vec(&r).unwrap()).unwrap();
        assert_eq!(copy.lesson.value, 1.);
        assert_eq!(copy.outcomes.len(), 2);
        let resolved = resolve(
            &loss,
            Some(lesson(0.9, None, "b", "fresh-reanalysis-q")),
            true,
        ).unwrap();
        assert_eq!(resolved.value, -1.);
        assert_eq!(resolved.evidence.unwrap().value_weight, 1.);
    }
}
