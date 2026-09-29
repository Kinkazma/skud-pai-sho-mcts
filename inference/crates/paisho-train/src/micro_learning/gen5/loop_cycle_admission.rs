//! Diagnostic receipt adapter mirroring archive::Ready.lessons/runtime admission.
//! No recall draw or optimizer update is performed by this adapter.
use super::*;

pub(super) struct Lesson {
    key: String,
    group: String,
    example: Arc<MicroExample>,
    proof: bool,
    decision: usize,
}

fn eligible(saved: &SavedMicroExample, spatial: bool, reanalysis: bool, loop_repair: bool) -> bool {
    spatial
        && (saved.correction_priority
            || reanalysis
            || loop_repair
                && saved
                    .tactical
                    .as_ref()
                    .is_some_and(|t| t.root_value.is_some()))
}

fn prefix_key(record: &GameRecord, decision: usize) -> Result<String> {
    let before = decision
        .checked_sub(1)
        .ok_or_else(|| invalid("fresh lesson decision must be one-based"))?;
    // Reanalysis evaluates the next decision without appending an action. Its
    // Saved index is record.len()+1 and its root is the complete current PSR.
    if before > record.actions().len() {
        return Err(invalid("fresh lesson prefix is outside its recorded game"));
    }
    Ok(sha256(cases::prefix(record, before).to_string().as_bytes()))
}

pub(super) fn prepare(
    game: &collector::Played,
    saved: &[SavedMicroExample],
    owned: &[Arc<MicroExample>],
) -> Result<Vec<Lesson>> {
    if saved.len() != owned.len() {
        return Err(invalid("cannot admit unsampled fresh lessons"));
    }
    let group = game.case.as_ref().map_or_else(
        || sha256(game.record.to_string().as_bytes()),
        |case| case.human_source.clone(),
    );
    saved
        .iter()
        .zip(owned)
        .filter(|(s, _)| {
            eligible(
                s,
                game.snapshot.model.has_spatial(),
                game.reanalysis,
                game.loop_repair,
            )
        })
        .map(|(s, example)| {
            Ok(Lesson {
                key: prefix_key(&game.record, s.decision)?,
                group: group.clone(),
                example: example.clone(),
                decision: s.decision,
                // Root proof provenance is essential: V/Q=1 alone is not a proof.
                proof: s.tactical.as_ref().is_some_and(|t| t.root_value.is_some()),
            })
        })
        .collect()
}

/// Called only after every learning batch of this receipt completed successfully.
/// Native admit routes proof values into Balanced, winning policies into Winning,
/// and proof/existing lessons into the pool. It leaves all recall quotas untouched.
pub(super) fn complete(archive: &mut durable::Archive, lessons: Vec<Lesson>) -> serde_json::Value {
    let before = archive.progress();
    let submitted = lessons.iter().map(|l| serde_json::json!({
        "key":l.key,"group":l.group,"decision":l.decision,"root_proof":l.proof,
        "value":l.example.value,"value_weight":l.example.value_weight,"policy_weight":l.example.policy_weight,
    })).collect::<Vec<_>>();
    for l in lessons {
        archive.admit(l.key, l.group, l.example, l.proof);
    }
    serde_json::json!({"fully_learned":true,"submitted":submitted,"before":before,"after":archive.progress()})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reanalysis_next_decision_uses_complete_prefix_without_appending_a_move() {
        let record: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
        ))
        .parse()
        .unwrap();
        let before = record.to_string();
        let mut saved = crate::micro_learning::tactics::fixture();
        saved.decision = record.actions().len() + 1;
        assert!(eligible(&saved, true, true, true));
        assert_eq!(
            prefix_key(&record, saved.decision).unwrap(),
            sha256(before.as_bytes())
        );
        assert_eq!(record.to_string(), before);
        let empty = cases::prefix(&record, 0);
        assert_eq!(
            prefix_key(&empty, 1).unwrap(),
            sha256(empty.to_string().as_bytes())
        );
        assert!(prefix_key(&record, 0).is_err());
        assert!(prefix_key(&record, saved.decision + 1).is_err());
    }

    #[test]
    fn root_proofs_bypass_priority_limit_but_empirical_or_action_only_values_do_not() {
        let mut s = crate::micro_learning::tactics::fixture();
        assert!(!s.correction_priority);
        assert!(eligible(&s, true, false, true));
        assert!(!eligible(&s, true, false, false));
        assert!(!eligible(&s, false, true, true));
        // Both a proved loss and a proved draw qualify just as a win does.
        for z in [-1, 0, 1] {
            s.tactical.as_mut().unwrap().root_value = Some(z);
            assert!(eligible(&s, true, false, true));
        }
        s.tactical.as_mut().unwrap().root_value = None;
        // Fixture still has V=1 and a Q=1 action: neither authenticates the root.
        assert!(!eligible(&s, true, false, true));
        assert!(eligible(&s, true, true, true));
        s.correction_priority = true;
        assert!(eligible(&s, true, false, false));
    }

    #[test]
    fn completed_receipt_populates_native_winning_cache_without_consuming_recall() {
        let root = std::env::temp_dir().join(format!(
            "gen5-probe-admission-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut archive = durable::Archive::open(&root).unwrap();
        archive.policy_consolidation(true);
        archive.proof_recall = true;
        let saved = crate::micro_learning::tactics::fixture();
        let example = Arc::new(saved.example_for_rules(RULES).unwrap());
        let make = |key: &str, proof| Lesson {
            key: key.into(),
            group: "test-source".into(),
            example: example.clone(),
            proof,
            decision: 1,
        };
        // Pending lessons do not mutate either cache before receipt completion.
        let pending = vec![make("proved", true), make("unproved-singleton", false)];
        assert_eq!(archive.progress()["winning"]["cache_positions"], 0);
        let report = complete(&mut archive, pending);
        assert_eq!(report["after"]["winning"]["cache_positions"], 1);
        assert_eq!(report["after"]["cache_positions"], 1);
        assert_eq!(report["after"]["winning"]["draws"], 0);
        assert_eq!(report["after"]["draws"], 0);
        assert_eq!(report["after"]["proof_credit"], 0.);
        // Native copy-on-write policy-only admission must not erase fresh V.
        assert_eq!(example.value_weight, 1.);
        let repeated = complete(&mut archive, vec![make("proved", true)]);
        assert_eq!(repeated["after"]["winning"]["cache_positions"], 1);
        assert_eq!(repeated["after"]["cache_positions"], 1);
        drop(archive);
        fs::remove_dir_all(root).unwrap();
    }
}
