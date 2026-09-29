use super::*;
use std::collections::BTreeMap;

fn entry(i: usize, time: u64, wins: Option<usize>) -> Entry {
    let mut results = BTreeMap::new();
    if let Some(w) = wins {
        results.insert(
            "reference-a:MCTS8".into(),
            Results {
                attempts: [8, 8],
                wins: [w, w],
                cases: Default::default(),
            },
        );
    }
    Entry {
        path: format!("model-{i}").into(),
        version: i as u64,
        ordinal: i,
        elapsed_millis: time,
        identity: format!("identity-{i}"),
        results,
    }
}
fn snapshot(version: u64) -> Arc<Snapshot> {
    let model = Arc::new(MicroModel::seeded(17));
    let artifact = MicroArtifact::new(&model, version, serde_json::json!({"test":true}));
    Arc::new(Snapshot {
        version,
        identity: artifact.identity(),
        model,
        artifact: Some(Arc::new(artifact)),
        path: PathBuf::from("never-read"),
    })
}
fn prune(entries: &mut Vec<Entry>, keep: usize) {
    let refs: Vec<_> = entries.iter().collect();
    let chosen = selection::select(&refs, keep, refs.len() - 1);
    let mut i = 0;
    entries.retain(|_| {
        let yes = chosen.contains(&i);
        i += 1;
        yes
    });
}

#[test]
fn repeated_online_pruning_covers_the_night_without_scores() {
    let mut entries = vec![];
    for i in 0..1061 {
        entries.push(entry(i, i as u64 * 30_000, None));
        prune(&mut entries, 64);
        assert_eq!(entries.len(), (i + 1).min(64));
        assert_eq!(entries.first().unwrap().ordinal, 0);
        assert_eq!(entries.last().unwrap().ordinal, i);
    }
    let ordinals: Vec<_> = entries.iter().map(|e| e.ordinal).collect();
    let gap = ordinals.windows(2).map(|w| w[1] - w[0]).max().unwrap();
    assert!(gap < 100, "{ordinals:?}; maximum gap {gap}");
    for quarter in 0..4 {
        assert!(
            ordinals
                .iter()
                .filter(|&&i| i >= quarter * 265 && i < (quarter + 1) * 265)
                .count()
                >= 8
        );
    }
    eprintln!(
        "1061 online saves: max_gap_seconds={}, ordinals={ordinals:?}",
        gap * 30
    );
}

#[test]
fn strong_neighbours_are_thinned_while_distant_strength_survives() {
    let entries: Vec<_> = (0..1061)
        .map(|i| {
            entry(
                i,
                i as u64 * 30_000,
                Some(if [200, 201, 700].contains(&i) { 8 } else { 0 }),
            )
        })
        .collect();
    let refs: Vec<_> = entries.iter().collect();
    let chosen = selection::select(&refs, 64, 1060);
    assert!(chosen.contains(&200));
    assert!(
        !chosen.contains(&201),
        "near-identical adjacent champion wasted a slot"
    );
    assert!(chosen.contains(&700));
    assert!(chosen.contains(&0) && chosen.contains(&1060));
}

#[test]
fn progressive_pruning_keeps_spaced_champions_and_middle_anchors() {
    let mut entries = vec![];
    for i in 0..4000 {
        let strong = (600..605).contains(&i) || (2800..2805).contains(&i);
        entries.push(entry(i, i as u64 * 30_000, strong.then_some(8)));
        prune(&mut entries, 64);
    }
    assert!(entries.iter().any(|e| (600..605).contains(&e.ordinal)));
    assert!(entries.iter().any(|e| (2800..2805).contains(&e.ordinal)));
    assert!(entries.iter().any(|e| (1700..2200).contains(&e.ordinal)));
    assert!(
        entries
            .windows(2)
            .map(|w| w[1].elapsed_millis - w[0].elapsed_millis)
            .max()
            .unwrap()
            < 12_000_000
    );
}

#[test]
fn spacing_uses_elapsed_time_even_with_bursts_and_out_of_order_collectors() {
    let mut entries = vec![];
    for i in 0..1000 {
        let time = if i < 500 {
            i as u64
        } else {
            500 + (i - 500) as u64 * 60_000
        };
        let mut e = entry(i, time, None);
        e.version = (1000 - i) as u64; // completion order is not model-version order
        entries.push(e);
        prune(&mut entries, 64);
    }
    assert!(
        entries
            .iter()
            .filter(|e| e.elapsed_millis > 7_000_000 && e.elapsed_millis < 23_000_000)
            .count()
            >= 20
    );
    let refs: Vec<_> = entries.iter().collect();
    let selected = selection::select(&refs, 16, 5);
    assert!(
        selected.contains(&5),
        "current recovery must survive regardless of timestamp/order"
    );
}

#[test]
fn sparse_measurements_admit_exact_collectors_without_misattribution() {
    let mut r = Retention::default();
    let current = snapshot(10);
    let teacher = snapshot(7);
    r.add("current".into(), &current).unwrap();
    r.observe_result(&teacher, "reference-a:MCTS8".into(), "case-a", 0, true);
    assert!(r.entries[0].results.is_empty());
    assert_eq!(r.pending.len(), 1);
    assert!(Arc::ptr_eq(&r.pending[0].snapshot, &teacher));
    assert_eq!(r.pending[0].entry.identity, teacher.identity);
    let score = r.pending[0].entry.results["reference-a:MCTS8"]
        .score()
        .unwrap();
    assert!((score - 7. / 12.).abs() < 1e-12);
    assert!(Results::default().score().is_none());
    r.observe_result(&teacher, "reference-a:MCTS8".into(), "case-a", 0, true);
    r.observe_result(&teacher, "reference-a:MCTS8".into(), "case-a", 1, false);
    r.observe_result(&teacher, "reference-a:MCTS16".into(), "case-a", 0, false);
    assert_eq!(
        r.pending[0].entry.results["reference-a:MCTS8"].attempts,
        [1, 1]
    );
    assert_eq!(
        r.pending[0].entry.results["reference-a:MCTS16"].attempts,
        [1, 0]
    );
}

#[test]
fn pending_snapshots_are_bounded_and_never_clone_the_model_bank() {
    let mut r = Retention::default();
    let mut weak = vec![];
    for version in 0..100 {
        let s = snapshot(version);
        weak.push(Arc::downgrade(&s));
        r.observe_result(&s, "reference-a:MCTS8".into(), "case", 0, true);
        assert!(r.pending.len() <= Retention::PENDING_LIMIT);
    }
    assert_eq!(
        weak.iter().filter(|s| s.upgrade().is_some()).count(),
        Retention::PENDING_LIMIT
    );
    drop(r.take_candidates());
    assert!(weak.iter().all(|s| s.upgrade().is_none()));
}

#[test]
fn quality_rankings_do_not_pool_different_reference_budgets() {
    let mut entries: Vec<_> = (0..256)
        .map(|i| entry(i, i as u64 * 30_000, None))
        .collect();
    entries[40].results.insert(
        "a:MCTS8".into(),
        Results {
            attempts: [8, 8],
            wins: [8, 8],
            ..Default::default()
        },
    );
    entries[180].results.insert(
        "b:MCTS512".into(),
        Results {
            attempts: [8, 8],
            wins: [5, 5],
            ..Default::default()
        },
    );
    let refs: Vec<_> = entries.iter().collect();
    let chosen = selection::select(&refs, 32, 255);
    assert!(chosen.contains(&40) && chosen.contains(&180));
}

#[test]
fn observation_filters_skip_retries_reanalysis_unknowns_and_errors() {
    let mut r = Retention::default();
    let spec = OpponentSpec {
        generation: "3.1".into(),
        model: "unused".into(),
        sha256: "hash".into(),
        solver: false,
    };
    let mut game = collector::Played {
        reused_search: false,
        search_cache_key: None,
        search_evidence_simulations: 0,
        loop_repair: false,
        measurement: false,
        evaluation: None,
        observed_origin: None,
        case: Some(cases::Attempt {
            opponent_generation: Some("3.1".into()),
            actor: 0,
            case: "case".into(),
            human_source: "source".into(),
            zone: 0,
            prefix_decisions: 0,
            before: Default::default(),
            after: Default::default(),
            kind: "test".into(),
        }),
        ack: None,
        certificates: vec![],
        reanalysis: false,
        prefix_decisions: 0,
        case_mode: true,
        id: 0,
        lane: Lane::Historical,
        snapshot: snapshot(1),
        opponent: "Gen3.1".into(),
        reference_budget: Some(8),
        candidate_seat: Player::Host,
        record: GameRecord::with_rules(
            paisho_core::StandardSetup::balanced(paisho_core::BasicFlower::Red3),
            RULES,
        ),
        outcome: GameOutcome::Ongoing,
        termination: "decision-limit".into(),
        error: None,
        samples: vec![],
        cycle: None,
        seconds: 0.,
        cap_seconds: None,
        campaign_censored: false,
        search_seconds: [0.; 2],
        pool_wait_seconds: 0.,
        maintenance_seconds: 0.,
        sample_seconds: 0.,
        simulations: 0,
        tactical_evaluations: 0,
        inherited: 0,
        hits: 0,
        evals: 0,
        policy_searches: 0,
        policy_coverage_sum: 0.,
        forced_playouts: 0,
        pruned_policy_visits: 0,
    };
    r.observe(&game, std::slice::from_ref(&spec));
    game.outcome = GameOutcome::Win(Player::Host);
    game.case.as_mut().unwrap().before.attempts = 1;
    r.observe(&game, std::slice::from_ref(&spec));
    game.case.as_mut().unwrap().before.attempts = 0;
    game.reanalysis = true;
    r.observe(&game, std::slice::from_ref(&spec));
    game.reanalysis = false;
    game.error = Some("test".into());
    r.observe(&game, std::slice::from_ref(&spec));
    game.error = None;
    r.observe(&game, &[]);
    assert!(r.pending.is_empty());
    r.observe(&game, std::slice::from_ref(&spec));
    assert_eq!(r.pending.len(), 1);
    assert_eq!(r.pending[0].entry.results["Gen3.1:hash:MCTS8"].wins, [1, 0]);
}
