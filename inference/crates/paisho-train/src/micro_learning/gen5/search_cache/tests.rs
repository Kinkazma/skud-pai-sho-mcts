use super::*;
#[test]
fn semantic_context_changes_invalidate_and_seed_does_not_for_plain_puct() {
    let o = MicroSearchOptions::default();
    let base = key("m", "p", 512, 16., o, &None).unwrap();
    let other = MicroProofCertificate {
        outcome: 1,
        children: vec![],
    };
    for k in [
        key("n", "p", 512, 16., o, &None),
        key("m", "q", 512, 16., o, &None),
        key("m", "p", 256, 16., o, &None),
        key("m", "p", 512, 8., o, &None),
        key(
            "m",
            "p",
            512,
            16.,
            MicroSearchOptions {
                proof_search: !o.proof_search,
                ..o
            },
            &None,
        ),
        key("m", "p", 512, 16., o, &Some(other)),
    ] {
        assert_ne!(base, k.unwrap());
    }
    assert_eq!(
        base,
        key(
            "m",
            "p",
            512,
            16.,
            MicroSearchOptions { seed: 99, ..o },
            &None
        )
        .unwrap()
    );
}
#[test]
fn native_cached_reanalysis_preserves_targets_and_distinct_observed_outcomes() {
    let record: GameRecord = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
    ))
    .parse()
    .unwrap();
    let prefix = cases::prefix(&record, 0);
    let model = Arc::new(MicroModel::seeded(7).with_spatial_policy());
    let a = Arc::new(MicroArtifact::new(
        &model,
        0,
        serde_json::json!({"test":true}),
    ));
    let snapshot = Arc::new(Snapshot {
        identity: a.identity(),
        artifact: Some(a),
        model,
        version: 0,
        path: PathBuf::new(),
    });
    let cache = Arc::new(ReanalysisCache::new(2 * 1024 * 1024));
    let mut o = Options {
        learning_loop_repair: true,
        learning_loop_v2: true,
        reanalysis_cache: Some(cache.clone()),
        value_policy_strength: 16.,
        budgets: vec![(8, 1.)],
        ..Default::default()
    };
    let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
    let run = |id, o: &Options| {
        collector::play_from(
            id,
            snapshot.clone(),
            snapshot.clone(),
            None,
            o,
            paisho_platform::training_time::now() + Duration::from_secs(30),
            &pool,
            "cache-test",
            Some(&prefix),
            true,
            None,
        )
    };
    o.observed_origin = Some((-1., "a".repeat(64)));
    let mut first = run(0, &o);
    assert!(first.error.is_none(), "{:?}", first.error);
    assert!(!first.reused_search);
    o.observed_origin = Some((1., "b".repeat(64)));
    let mut hit = run(1, &o);
    assert!(hit.error.is_none(), "{:?}", hit.error);
    assert!(hit.reused_search);
    assert_eq!(hit.simulations, 0);
    assert_eq!(hit.evals, 0);
    o.reanalysis_cache = None;
    let mut fresh = run(1, &o);
    assert_eq!(
        fresh.search_evidence_simulations,
        hit.search_evidence_simulations
    );
    let ta = collector::targets(&mut first);
    let tb = collector::targets(&mut hit);
    let tc = collector::targets(&mut fresh);
    assert_eq!(
        serde_json::to_value(&tb).unwrap(),
        serde_json::to_value(&tc).unwrap()
    );
    assert_eq!(ta[0].policy, tb[0].policy);
    assert_ne!(
        ta[0].evidence.as_ref().unwrap().observed_value,
        tb[0].evidence.as_ref().unwrap().observed_value
    );
    assert_ne!(ta[0].game_id, tb[0].game_id);
    assert_eq!(cache.progress()["hits"], 1);
    assert_eq!(
        tb[0].evidence.as_ref().unwrap().policy_coordinates,
        "raw-policy-v1"
    );
}
