use super::*;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_gate_requires_complete_consistent_lot_and_sixty_wins() {
        let mut a = ladder::State {
            reference: "ref".into(),
            max_stage: 7,
            ..Default::default()
        };
        let mut b = a.clone();
        let mut results = vec![Some(-1); 100];
        for z in &mut results[..59] {
            *z = Some(1);
        }
        assert!(a
            .record_frozen(&results[..99], 8, 7, "model", "ref", 1)
            .is_err());
        assert!(a
            .record_frozen(&results, 8, 7, "model", "other", 1)
            .is_err());
        a.record_frozen(&results, 8, 7, "model", "ref", 1).unwrap();
        assert_eq!(a.budget(), 8);
        results[59] = Some(1);
        b.record_frozen(&results, 8, 7, "model", "ref", 1).unwrap();
        assert_eq!(b.budget(), 32);
        assert_eq!(a.budget(), 8);
        let r: ladder::State = serde_json::from_slice(&serde_json::to_vec(&b).unwrap()).unwrap();
        assert_eq!(r.batches[0].frozen.as_ref().unwrap()["model"], "model");
    }
}
#[cfg(test)]
mod scheduling_tests {
    use super::*;
    #[test]
    fn reservations_are_independent_freeze_weights_and_resume_unfinished_slots() {
        let root = std::env::temp_dir().join(format!("gen5-frozen-lots-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let r: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
        ))
        .parse()
        .unwrap();
        let cases: Vec<_> = (0..50)
            .map(|i| cases::Case {
                identity: format!("case-{i}"),
                source: format!("source-{i}"),
                zone: 0,
                record: r.clone(),
            })
            .collect();
        let m = Arc::new(MicroModel::seeded(4).with_spatial_policy());
        let artifact = Arc::new(MicroArtifact::new(&m, 1, serde_json::json!({"test":true})));
        let s = Arc::new(Snapshot {
            identity: artifact.identity(),
            artifact: Some(artifact),
            model: m.clone(),
            path: "unused".into(),
            version: 1,
        });
        let specs = (0..5)
            .map(|i| OpponentSpec {
                generation: format!("3.{}", i + 1),
                model: "unused".into(),
                sha256: format!("ref-{i}"),
                solver: false,
            })
            .collect();
        let o = Options {
            opponents: specs,
            ..Default::default()
        };
        let e = Evaluations::open(&root, &cases, &o, &serde_json::Value::Null, &m).unwrap();
        let first = e.reserve(0, "ref-0", 8, s.clone()).unwrap().unwrap();
        for _ in 1..100 {
            e.reserve(0, "ref-0", 8, s.clone()).unwrap().unwrap();
        }
        assert!(e.reserve(0, "ref-0", 8, s.clone()).unwrap().is_none());
        assert!(e.reserve(1, "ref-1", 8, s.clone()).unwrap().is_some());
        let state = e.progress();
        let copy = Evaluations::open(&root, &cases, &o, &state, &m).unwrap();
        let resumed = copy.reserve(0, "ref-0", 8, s).unwrap().unwrap();
        assert_eq!(first.attempt.slot, resumed.attempt.slot);
        assert_eq!(first.attempt.model, resumed.attempt.model);
        assert_eq!(
            first.snapshot.model.parameters(),
            resumed.snapshot.model.parameters()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
