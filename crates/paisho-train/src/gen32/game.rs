use super::*;
use crate::compact_selfplay::reuse::Repetitions;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};
pub(super) struct Played {
    pub record: GameRecord,
    pub targets: Vec<SavedMicroExample>,
    pub receipt: serde_json::Value,
}
#[cfg(test)]
pub(super) fn play(
    id: usize,
    actor: usize,
    ordinal: usize,
    version: u64,
    model: &Gen32Model,
    o: &Options,
    end: Instant,
    stop: &AtomicBool,
    pool: &rayon::ThreadPool,
    source: &str,
    other: Option<&dyn MctsEvaluator>,
) -> std::result::Result<Played, String> {
    play_with_reference_solver(
        id, actor, ordinal, version, model, o, end, stop, pool, source, other, false,
    )
}
pub(super) fn play_with_reference_solver(
    id: usize,
    actor: usize,
    ordinal: usize,
    version: u64,
    model: &Gen32Model,
    o: &Options,
    end: Instant,
    stop: &AtomicBool,
    pool: &rayon::ThreadPool,
    source: &str,
    other: Option<&dyn MctsEvaluator>,
    reference_solver: bool,
) -> std::result::Result<Played, String> {
    play_from_record(
        id,
        actor,
        ordinal,
        version,
        model,
        o,
        end,
        stop,
        pool,
        source,
        other,
        reference_solver,
        None,
    )
}
pub(super) fn play_from_record(
    id: usize,
    actor: usize,
    ordinal: usize,
    version: u64,
    model: &Gen32Model,
    o: &Options,
    end: Instant,
    stop: &AtomicBool,
    pool: &rayon::ThreadPool,
    source: &str,
    other: Option<&dyn MctsEvaluator>,
    reference_solver: bool,
    prefix: Option<&GameRecord>,
) -> std::result::Result<Played, String> {
    let began = Instant::now();
    let slot = if o.fixed_actor_budgets { actor } else { id } % o.budgets.len();
    let budget = o.budgets[slot];
    let training_reference = o.historical_reference.is_some() || !o.historical_pool.is_empty();
    let historical = other.is_some() || (!training_reference && ordinal % 2 == 1);
    // Training reference games have no individual wall timeout. Campaign end,
    // legal repetition detection and the decision limit remain effective.
    let deadline = if training_reference && historical {
        end
    } else {
        end.min(began + Duration::from_secs_f64(o.caps[slot]))
    };
    let heuristic = CpuMctsEvaluator;
    let seat_index = if training_reference && historical {
        (ordinal + actor) / o.historical_every
    } else if other.is_some() {
        id / 2
    } else {
        ordinal / 2
    };
    let candidate = if seat_index % 2 == 0 {
        Player::Host
    } else {
        Player::Guest
    };
    let opponent = other.unwrap_or(&heuristic);
    let setup_index = if training_reference && historical {
        seat_index / 2
    } else if other.is_some() {
        id / 4
    } else {
        ordinal / 2
    };
    let setup = StandardSetup::balanced(BASIC_FLOWERS[setup_index % BASIC_FLOWERS.len()]);
    let mut record = prefix
        .cloned()
        .unwrap_or_else(|| GameRecord::with_rules(setup, RULES));
    if record.rules() != RULES {
        return Err("evaluation prefix rules mismatch".into());
    }
    let prefix_decisions = record.actions().len();
    let mut position = record.replay().map_err(|e| e.to_string())?;
    if position.outcome() != GameOutcome::Ongoing {
        return Err("evaluation prefix already terminal".into());
    }
    let mut repetitions = Repetitions::new(&position, 4);
    let config = MctsConfig {
        simulations: budget,
        ..Default::default()
    };
    let evaluators: [&dyn MctsEvaluator; 2] = [Player::Host, Player::Guest].map(|seat| {
        if !historical || seat == candidate {
            model as &dyn MctsEvaluator
        } else {
            opponent
        }
    });
    let mut sessions = [
        MctsSession::new(o.seed + id as u64, config, evaluators[0])?,
        MctsSession::new(o.seed + id as u64 + 1, config, evaluators[1])?,
    ];
    for seat in [Player::Host, Player::Guest] {
        sessions[seat.index()].set_solver(if historical && seat != candidate {
            reference_solver
        } else {
            o.solver
        });
    }
    let identity = if let Some(residual) = &model.value_residual {
        sha256(
            &serde_json::to_vec(&(
                GEN3_VALUE_RESIDUAL_SCHEMA,
                model.value.weights().as_slice(),
                model.value_extra.map(|w| w.to_vec()),
                residual.parameters(),
                model.policy.parameters(),
                model.memory_scope.name(),
                model.policy.sequence_memory().map(|b| &b.spec),
            ))
            .map_err(|e| e.to_string())?,
        )
    } else if model.value_extra.is_some() || model.memory_scope != Gen3MemoryScope::Root {
        sha256(
            &serde_json::to_vec(&(
                model.value.weights().as_slice(),
                model.value_extra.map(|w| w.to_vec()),
                model.policy.parameters(),
                model.memory_scope.name(),
                model.policy.sequence_memory().map(|b| &b.spec),
            ))
            .map_err(|e| e.to_string())?,
        )
    } else {
        sha256(
            &serde_json::to_vec(&(model.value.weights().as_slice(), model.policy.parameters()))
                .map_err(|e| e.to_string())?,
        )
    };
    let mut samples = Vec::new();
    let mut corrections = Vec::new();
    let mut tactical_positions = 0;
    let mut tactical_seconds = 0.;
    let mut tactical_changes = 0;
    let mut rng = StableRng::new(o.seed ^ id as u64);
    let mut seen = 0;
    let mut termination = "decision-limit";
    let mut cycle = None;
    let mut search_seconds = 0.;
    let mut candidate_seconds = 0.0;
    let mut opponent_seconds = 0.0;
    let mut candidate_decisions = 0;
    let mut opponent_decisions = 0;
    let mut generated = 0;
    let mut reused = 0;
    while record.actions().len() - prefix_decisions < o.decisions
        && position.outcome() == GameOutcome::Ongoing
    {
        if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
            termination = "time-limit";
            break;
        }
        let mut unsampled = None;
        let legal = legal_actions(&position);
        if legal.is_empty() {
            termination = "blocked-unknown";
            break;
        }
        let player = position.to_move();
        let is_candidate = !historical || player == candidate;
        let started = Instant::now();
        let report = pool
            .install(|| sessions[player.index()].search_until(&position, &legal, Some(deadline)))?;
        let mut selected = report.selected_index;
        let mut guard = Gen32GuardReport::default();
        if is_candidate {
            let guard_started = Instant::now();
            let guard_limit = if sessions[player.index()].action_proofs()[selected]
                == Some(GameOutcome::Win(player))
            {
                0
            } else {
                o.tactical_positions
            };
            guard = gen32_tactical_guard(&position, &legal, &report, guard_limit, Some(deadline));
            tactical_seconds += guard_started.elapsed().as_secs_f64();
            tactical_positions += guard.visited;
            // A proved winning search action must never be overridden.
            if sessions[player.index()].action_proofs()[selected] != Some(GameOutcome::Win(player))
            {
                selected = guard.selected;
            }
            tactical_changes += usize::from(selected != report.selected_index);
        }
        let used = started.elapsed().as_secs_f64();
        search_seconds += used;
        if is_candidate {
            candidate_seconds += used;
            candidate_decisions += 1;
        } else {
            opponent_seconds += used;
            opponent_decisions += 1;
        }
        generated += report.evaluated_actions;
        if is_candidate {
            seen += 1;
            let total: usize = report.actions.iter().map(|a| a.visits).sum();
            if total > 0 {
                let mut sample = SavedMicroExample {
                    rules: RULES.to_string(),
                    source_run: source.into(),
                    game_id: id.to_string(),
                    decision: record.actions().len() + 1,
                    collector: identity.clone(),
                    budget,
                    inherited_visits: sessions[player.index()]
                        .reuse_statistics()
                        .inherited_root_visits,
                    new_visits: report.actions.iter().map(|a| a.visits).collect(),
                    policy_raw_visits: vec![],
                    policy_pruned_visits: vec![],
                    tactical: None,
                    correction_priority: false,
                    actions: legal.iter().map(ToString::to_string).collect(),
                    state: micro_state_features(&position).to_vec(),
                    action_features: legal
                        .iter()
                        .map(|a| micro_action_features(&position, *a).to_vec())
                        .collect(),
                    policy: sessions[player.index()].learning_policy().to_vec(),
                    value: sessions[player.index()].action_value_estimates()[report.selected_index],
                    policy_weight: 1.0,
                    reason: "pending".into(),
                };
                super::tactics::correct(
                    &mut sample,
                    model,
                    &position,
                    sessions[player.index()].action_proofs(),
                    &guard,
                    selected,
                )?;
                if sample.correction_priority && corrections.len() < 4 {
                    corrections.push((sample.clone(), player));
                }
                let entry = (sample, player);
                if samples.len() < o.samples {
                    samples.push(entry)
                } else {
                    let i = rng.index(seen);
                    if i < o.samples {
                        samples[i] = entry
                    } else {
                        unsampled = Some(entry)
                    }
                }
            }
        }
        let action = legal[selected];
        position.apply(action).map_err(|e| e.to_string())?;
        record.push(action);
        for session in &mut sessions {
            reused += usize::from(session.advance(action));
        }
        if let Some(c) = repetitions.observe(&position, player, record.actions().len()) {
            if let Some(entry) = unsampled.take() {
                samples.pop();
                samples.push(entry);
            }
            cycle = Some(c);
            termination = "repetition";
            break;
        }
    }
    if position.outcome() != GameOutcome::Ongoing {
        termination = "rules-terminal"
    }
    samples.retain(|(s, _)| !corrections.iter().any(|(c, _)| c.decision == s.decision));
    samples.extend(corrections);
    let targets = samples
        .into_iter()
        .filter_map(|(mut ex, player)| {
            if let Some(value) = ex.tactical.as_ref().and_then(|t| t.root_value) {
                ex.value = value as f64;
                ex.reason = "search-proven-value".into();
                return Some(ex);
            }
            if let Some(c) = &cycle {
                if c.loser.player() != player || ex.decision < c.first_decision {
                    return None;
                }
                ex.tactical = None;
                ex.correction_priority = false;
                ex.value = -1.;
                ex.policy_weight = 0.;
                ex.actions.clear();
                ex.action_features.clear();
                ex.policy.clear();
                ex.new_visits.clear();
                ex.reason = "repetition-training-loss".into();
            } else {
                let z = match position.outcome() {
                    GameOutcome::Win(p) => {
                        if p == player {
                            1.
                        } else {
                            -1.
                        }
                    }
                    GameOutcome::Draw => 0.,
                    GameOutcome::Ongoing => return None,
                };
                ex.value = (1. - o.lambda) * ex.value + o.lambda * z;
                ex.reason = "rules-terminal-q-mix".into();
                let bounded = super::tactics::respect_draw_floor(ex.value, ex.tactical.as_ref());
                if bounded != ex.value {
                    ex.value = bounded;
                    ex.reason = "rules-terminal-q-mix-draw-floor".into();
                }
            }
            Some(ex)
        })
        .collect::<Vec<_>>();
    let receipt = serde_json::json!({"id":id,"actor":actor,"ordinal":ordinal,"collector_version":version,"collector_sha256":identity,"reference_sha256":if historical {o.historical_reference_sha256.as_deref()}else{None},"game_cap_seconds":if training_reference && historical {None}else{Some(o.caps[slot])},"budget":budget,"rules":RULES.as_str(),"historical":historical,"opponent":if other.is_some(){"Gen3.1"}else if historical{"historical-heuristic"}else{"selfplay"},"candidate_seat":format!("{candidate:?}"),"outcome":format!("{:?}",position.outcome()),"termination":termination,"decisions":record.actions().len(),"seconds":began.elapsed().as_secs_f64(),"search_seconds":search_seconds,"candidate_search_seconds":candidate_seconds,"opponent_search_seconds":opponent_seconds,"candidate_decisions":candidate_decisions,"opponent_decisions":opponent_decisions,"generated_candidates":generated,"solver":o.solver,"reference_solver":reference_solver,"prefix_decisions":prefix_decisions,"tactical_positions":tactical_positions,"tactical_seconds":tactical_seconds,"tactical_changes":tactical_changes,"retained_advances":reused,"cycle":cycle,"eligible":targets.len()});
    Ok(Played {
        record,
        targets,
        receipt,
    })
}

#[cfg(test)]
mod comparison_tests {
    use super::*;
    #[test]
    fn residual_collects_real_terminal_targets_and_changes_collector_identity() {
        let full: GameRecord =
            include_str!("../../../paisho-ai/tests/fixtures/gen34-r5-capture.psr")
                .parse()
                .unwrap();
        let mut prefix = GameRecord::with_rules(full.setup(), full.rules());
        for a in &full.actions()[..full.actions().len() - 1] {
            prefix.push(*a);
        }
        let old = Gen32Model::from_gen31(load_model_for_test(), 71)
            .with_value128([0.; 64])
            .unwrap();
        let mut new = old.clone().with_value_residual(71);
        let o = Options {
            budgets: vec![8],
            caps: vec![10.],
            decisions: 1,
            solver: true,
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let play = |m| {
            play_from_record(
                0,
                0,
                0,
                0,
                m,
                &o,
                Instant::now() + Duration::from_secs(10),
                &AtomicBool::new(false),
                &pool,
                "residual-test",
                None,
                false,
                Some(&prefix),
            )
            .unwrap()
        };
        let before = play(&old);
        let after = play(&new);
        assert_eq!(before.record, after.record);
        assert_eq!(after.receipt["termination"], "rules-terminal");
        assert_eq!(after.targets.len(), 1);
        assert_ne!(before.targets[0].collector, after.targets[0].collector);
        let target = after.targets[0].example().unwrap();
        assert_eq!(target.value, 1.);
        let frozen = new.value_residual.clone();
        new.train(&target, 0.01).unwrap();
        assert_ne!(new.value_residual, frozen);
        assert!(!frozen.unwrap().active());
    }
    #[test]
    fn historical_wrapper_preserves_search_and_symmetric_mode_enables_both_seats() {
        let model = Gen32Model {
            value_residual: None,
            value_extra: None,
            memory_scope: Gen3MemoryScope::Root,
            value: load_model_for_test(),
            policy: MicroModel::seeded(42),
        };
        let o = Options {
            budgets: vec![8],
            caps: vec![10.],
            decisions: 2,
            solver: true,
            learn: false,
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for id in [0, 2] {
            let play = |symmetric| {
                play_with_reference_solver(
                    id,
                    0,
                    0,
                    0,
                    &model,
                    &o,
                    Instant::now() + Duration::from_secs(10),
                    &AtomicBool::new(false),
                    &pool,
                    "test",
                    Some(&model),
                    symmetric,
                )
                .unwrap()
            };
            let default = super::play(
                id,
                0,
                0,
                0,
                &model,
                &o,
                Instant::now() + Duration::from_secs(10),
                &AtomicBool::new(false),
                &pool,
                "test",
                Some(&model),
            )
            .unwrap();
            let explicit = play(false);
            assert_eq!(default.record.to_string(), explicit.record.to_string());
            assert_eq!(explicit.receipt["reference_solver"], false);
            let symmetric = play(true);
            assert_eq!(symmetric.receipt["reference_solver"], true);
            assert_eq!(symmetric.receipt["solver"], true);
            symmetric.record.replay().unwrap();
        }
    }
    #[test]
    fn evaluation_prefix_is_retained_and_limit_counts_new_decisions() {
        let model = Gen32Model {
            value_residual: None,
            value_extra: None,
            memory_scope: Gen3MemoryScope::Root,
            value: load_model_for_test(),
            policy: MicroModel::seeded(42),
        };
        let o = Options {
            budgets: vec![8],
            caps: vec![10.],
            decisions: 2,
            solver: true,
            learn: false,
            ..Default::default()
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let mut prefix = GameRecord::with_rules(StandardSetup::balanced(BASIC_FLOWERS[0]), RULES);
        for _ in 0..2 {
            let p = prefix.replay().unwrap();
            prefix.push(legal_actions(&p)[0]);
        }
        for id in [0, 2] {
            let played = play_from_record(
                id,
                0,
                0,
                0,
                &model,
                &o,
                Instant::now() + Duration::from_secs(10),
                &AtomicBool::new(false),
                &pool,
                "prefix-test",
                Some(&model),
                true,
                Some(&prefix),
            )
            .unwrap();
            assert_eq!(&played.record.actions()[..2], prefix.actions());
            assert_eq!(played.record.actions().len(), 4);
            assert_eq!(played.receipt["prefix_decisions"], 2);
            assert_eq!(
                played.receipt["candidate_seat"],
                if id == 0 { "Host" } else { "Guest" }
            );
            played.record.replay().unwrap();
        }
        assert_eq!(prefix.actions().len(), 2);
    }
    fn load_model_for_test() -> CompactValueModel {
        ModelArtifact::legacy().model().unwrap()
    }
}
