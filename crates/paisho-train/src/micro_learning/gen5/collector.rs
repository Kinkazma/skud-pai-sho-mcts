use super::*;
use crate::compact_selfplay::reuse::RepetitionLoss;
use paisho_core::legal_actions;

pub(super) struct Sample {
    saved: SavedMicroExample,
    player: Player,
    q: f64,
}
pub(super) struct Played {
    pub case: Option<cases::Attempt>,
    pub ack: Option<std::sync::mpsc::SyncSender<case_actor::Feedback>>,
    pub certificates: Vec<(usize, MicroProofCertificate)>,
    pub reanalysis: bool,
    pub prefix_decisions: usize,
    pub case_mode: bool,
    pub id: usize,
    pub lane: Lane,
    pub snapshot: Arc<Snapshot>,
    pub opponent: String,
    pub reference_budget: Option<usize>,
    pub candidate_seat: Player,
    pub record: GameRecord,
    pub outcome: GameOutcome,
    pub termination: String,
    pub error: Option<String>,
    pub samples: Vec<Sample>,
    pub cycle: Option<RepetitionLoss>,
    pub seconds: f64,
    pub cap_seconds: Option<f64>,
    pub campaign_censored: bool,
    pub search_seconds: [f64; 2],
    pub pool_wait_seconds: f64,
    pub maintenance_seconds: f64,
    pub sample_seconds: f64,
    pub simulations: usize,
    pub inherited: usize,
    pub hits: usize,
    pub evals: usize,
    pub policy_searches: usize,
    pub policy_coverage_sum: f64,
    pub forced_playouts: usize,
    pub pruned_policy_visits: usize,
}
/// Each game holds immutable models; every actual action advances every session.
/// Gen3.1 runs only in the secondary CPU pool supplied by the scheduler.
pub(super) fn play(
    id: usize,
    snapshot: Arc<Snapshot>,
    opponent: Arc<Snapshot>,
    legacy: Option<(&CompactValueModel, usize)>,
    o: &Options,
    end: Instant,
    pool: &cpu::Executor,
    source: &str,
) -> Played {
    play_from(
        id, snapshot, opponent, legacy, o, end, pool, source, None, false, None,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn play_from(
    id: usize,
    snapshot: Arc<Snapshot>,
    opponent: Arc<Snapshot>,
    legacy: Option<(&CompactValueModel, usize)>,
    o: &Options,
    end: Instant,
    pool: &cpu::Executor,
    source: &str,
    start: Option<&GameRecord>,
    reanalysis: bool,
    proofs: Option<&durable::Proofs>,
) -> Played {
    let started = Instant::now();
    let cap = legacy.map_or(
        if start.is_some() {
            None
        } else {
            Some(o.game_seconds)
        },
        |(_, b)| {
            if o.legacy_replay || o.historical_unlimited_budgets.contains(&b) {
                None
            } else {
                Some(
                    o.historical_seconds
                        .iter()
                        .find(|(v, _)| *v == b)
                        .unwrap()
                        .1,
                )
            }
        },
    );
    let game_deadline = cap.map(|seconds| started + Duration::from_secs_f64(seconds));
    let deadline = game_deadline.map_or(end, |limit| end.min(limit));
    let mut rng = StableRng::new(o.seed.wrapping_add(id as u64));
    let setup = super::super::compare::comparison_setup(rng.index(114), o.seed, true);
    let record = start
        .cloned()
        .unwrap_or_else(|| GameRecord::with_rules(setup, RULES));
    let prefix_decisions = record.actions().len();
    let (mut position, mut repeat) = cases::context(&record).expect("validated exact case prefix");
    let random_seat = if rng.index(2) == 0 {
        Player::Host
    } else {
        Player::Guest
    };
    let candidate_seat = o.candidate_seat.unwrap_or(random_seat);
    let same = legacy.is_none() && snapshot.identity == opponent.identity;
    let lane = if reanalysis {
        Lane::Reanalysis
    } else if legacy.is_some() {
        Lane::Historical
    } else if same {
        Lane::Selfplay
    } else {
        Lane::Checkpoint
    };
    let mut sessions = if same || legacy.is_some() {
        vec![MicroMctsSession::new(snapshot.model.clone())]
    } else {
        vec![
            MicroMctsSession::new(if candidate_seat == Player::Host {
                snapshot.model.clone()
            } else {
                opponent.model.clone()
            }),
            MicroMctsSession::new(if candidate_seat == Player::Guest {
                snapshot.model.clone()
            } else {
                opponent.model.clone()
            }),
        ]
    };
    let mut old = legacy.map(|(model, budget)| {
        MctsSession::new(
            o.seed.wrapping_add(id as u64),
            MctsConfig {
                simulations: budget,
                ..Default::default()
            },
            model,
        )
        .expect("validated fixed budget")
    });
    let mut game = Played {
        case: None,
        ack: None,
        certificates: vec![],
        reanalysis,
        prefix_decisions,
        case_mode: o.case_curriculum.is_some() || start.is_some(),
        id,
        lane,
        snapshot: snapshot.clone(),
        opponent: if legacy.is_some() {
            "Gen3.1".into()
        } else {
            opponent.identity.clone()
        },
        reference_budget: legacy.map(|(_, b)| b),
        candidate_seat,
        record,
        outcome: GameOutcome::Ongoing,
        termination: "decision-limit".into(),
        error: None,
        samples: Vec::new(),
        cycle: None,
        seconds: 0.0,
        cap_seconds: cap,
        campaign_censored: game_deadline.map_or(true, |limit| end < limit),
        search_seconds: [0.0; 2],
        pool_wait_seconds: 0.0,
        maintenance_seconds: 0.0,
        sample_seconds: 0.0,
        simulations: 0,
        inherited: 0,
        hits: 0,
        evals: 0,
        policy_searches: 0,
        policy_coverage_sum: 0.0,
        forced_playouts: 0,
        pruned_policy_visits: 0,
    };
    let result = (|| -> std::result::Result<(), String> {
        for decision in 0..o.decision_limit {
            if o.stop_signal.as_ref().is_some_and(|s| s.load(std::sync::atomic::Ordering::Relaxed)) {
                game.termination = "user-stop".into(); break;
            }
            if position.outcome() != GameOutcome::Ongoing {
                game.termination = "rules-terminal".into();
                break;
            }
            if Instant::now() >= deadline {
                game.termination = "wall-limit".into();
                break;
            }
            let player = position.to_move();
            let historical_turn = old.is_some() && player != candidate_seat;
            let queued = Instant::now();
            let action = if historical_turn {
                let (report, actions, search) = pool.install(|| {
                    let t = Instant::now();
                    let actions = legal_actions(&position);
                    let report =
                        old.as_mut()
                            .unwrap()
                            .search_until(&position, &actions, Some(deadline));
                    (report, actions, t.elapsed().as_secs_f64())
                });
                game.search_seconds[1] += search;
                game.pool_wait_seconds += (queued.elapsed().as_secs_f64() - search).max(0.0);
                let report = report?;
                game.simulations += report.simulations;
                if report.simulations == 0 && Instant::now() >= deadline {
                    game.termination = "wall-limit".into();
                    break;
                }
                actions[report.selected_index]
            } else {
                let draw = rng.next_f64();
                let mut cumulative = 0.0;
                let budget = o
                    .budgets
                    .iter()
                    .find(|(_, w)| {
                        cumulative += w;
                        draw < cumulative
                    })
                    .unwrap_or(o.budgets.last().unwrap())
                    .0;
                let search_options = o.search(rng.next_u64(), !reanalysis)?;
                let (report, search) = pool.install(|| {
                    let t = Instant::now();
                    let session_index = if sessions.len() == 1 {
                        0
                    } else {
                        player.index()
                    };
                    if let Some(proofs) = proofs {
                        if !proofs.read().unwrap().is_empty() {
                            let key =
                                crate::compact_learning::sha256(game.record.to_string().as_bytes());
                            let certificate = match proof_cache::lookup(proofs, &key) {
                                Ok(c) => c,
                                Err(e) => return (Err(e.to_string()), t.elapsed().as_secs_f64()),
                            };
                            if let Some(c) = certificate {
                                if let Err(e) =
                                    sessions[session_index].install_certificate(&position, &c)
                                {
                                    return (Err(e), t.elapsed().as_secs_f64());
                                }
                            }
                        }
                    }
                    let result = sessions[session_index].search_with_options(
                        &position,
                        budget,
                        Some(deadline),
                        search_options,
                    );
                    (result, t.elapsed().as_secs_f64())
                });
                game.search_seconds[0] += search;
                game.pool_wait_seconds += (queued.elapsed().as_secs_f64() - search).max(0.0);
                let report = report?;
                if report.proven_value.is_some() && start.is_some() {
                    let index = if sessions.len() == 1 {
                        0
                    } else {
                        player.index()
                    };
                    if let Some(c) = sessions[index].certificate(10_000) {
                        game.certificates.push((prefix_decisions + decision + 1, c));
                    }
                }
                if report.simulations == 0 && report.proven_value.is_none() {
                    game.termination = "wall-limit".into();
                    break;
                }
                game.simulations += report.simulations;
                game.inherited += report.inherited_visits;
                game.hits += report.inference_cache_hits;
                game.evals += report.inference_evaluations;
                game.forced_playouts += report.new_forced_visits.iter().sum::<usize>();
                let sampling = Instant::now();
                // Gumbel already samples without replacement at root. Do not replace
                // its recommended action with a fresh visit draw.
                let has_proof = report.proven_action_values.iter().any(Option::is_some);
                let selected =
                    if !reanalysis && decision < 40 && search_options.mode == MicroSearchMode::Puct
                    {
                        let mut pick = report.selected_index;
                        if has_proof {
                            // Preserve exploration among the unrefuted alternatives.
                            let mut draw = rng.next_f64();
                            for (i, probability) in report.policy_target.iter().enumerate() {
                                if draw < *probability {
                                    pick = i;
                                    break;
                                }
                                draw -= probability;
                            }
                        } else {
                            let mut draw = rng.index(report.visits.iter().sum());
                            for (i, n) in report.visits.iter().enumerate() {
                                if draw < *n {
                                    pick = i;
                                    break;
                                }
                                draw -= n;
                            }
                        }
                        pick
                    } else {
                        report.selected_index
                    };
                let high = o.budgets.iter().map(|(b, _)| *b).max().unwrap();
                let policy_weight = if report.proven_value == Some(-1) {
                    0.0
                } else if report.proven_value.is_some()
                    || (budget >= o.policy_min_budget.min(high) && report.simulations == budget)
                {
                    1.0
                } else {
                    0.0
                };
                if policy_weight > 0.0 {
                    game.pruned_policy_visits += report.pruned_visits.iter().sum::<usize>();
                    game.policy_searches += 1;
                    game.policy_coverage_sum += report.new_visits.iter().filter(|n| **n > 0).count()
                        as f64
                        / report.actions.len() as f64;
                }
                let collector = if same || player == candidate_seat {
                    &snapshot.identity
                } else {
                    &opponent.identity
                };
                game.samples.push(Sample {
                    player,
                    q: report.values[selected],
                    saved: SavedMicroExample {
                        tactical: TacticalEvidence::from_report(&report, policy_weight > 0.0),
                        correction_priority: false,
                        policy_raw_visits: if policy_weight > 0.0
                            && report.pruned_visits.iter().any(|n| *n > 0)
                            && report.visits.iter().sum::<usize>()
                                > report.pruned_visits.iter().sum::<usize>()
                        {
                            report.visits.clone()
                        } else {
                            vec![]
                        },
                        policy_pruned_visits: if policy_weight > 0.0
                            && report.pruned_visits.iter().any(|n| *n > 0)
                            && report.visits.iter().sum::<usize>()
                                > report.pruned_visits.iter().sum::<usize>()
                        {
                            report.pruned_visits.clone()
                        } else {
                            vec![]
                        },
                        rules: RULES.to_string(),
                        source_run: source.into(),
                        game_id: id.to_string(),
                        decision: prefix_decisions + decision + 1,
                        collector: collector.clone(),
                        budget,
                        inherited_visits: report.inherited_visits,
                        new_visits: if policy_weight > 0.0 {
                            report.new_visits.clone()
                        } else {
                            vec![]
                        },
                        actions: if policy_weight > 0.0 {
                            report.actions.iter().map(ToString::to_string).collect()
                        } else {
                            vec![]
                        },
                        state: report.state.to_vec(),
                        action_features: if policy_weight > 0.0 {
                            report.action_features.iter().map(|f| f.to_vec()).collect()
                        } else {
                            vec![]
                        },
                        policy: if policy_weight > 0.0 {
                            report.policy_target.clone()
                        } else {
                            vec![]
                        },
                        value: 0.0,
                        policy_weight,
                        reason: String::new(),
                    },
                });
                game.sample_seconds += sampling.elapsed().as_secs_f64();
                report.actions[selected]
            };
            if reanalysis {
                game.termination = "reanalysis-search".into();
                break;
            }
            let t = Instant::now();
            position.apply(action).map_err(|e| e.to_string())?;
            game.record.push(action);
            pool.install(|| -> std::result::Result<(), String> {
                for session in &mut sessions {
                    session.advance(action)?;
                }
                if let Some(old) = &mut old {
                    old.advance(action);
                }
                Ok(())
            })?;
            game.cycle = repeat.observe(&position, player, game.record.actions().len());
            game.maintenance_seconds += t.elapsed().as_secs_f64();
            if game.cycle.is_some() {
                game.termination = "repetition-training-loss".into();
                break;
            }
        }
        Ok(())
    })();
    game.error = result.err();
    game.outcome = position.outcome();
    if game.error.is_some() {
        game.termination = "engine-error".into();
    } else if game.outcome != GameOutcome::Ongoing {
        game.termination = "rules-terminal".into();
    }
    // Only a deadline-truncated game is censored by the campaign end.
    game.campaign_censored &= game.termination == "wall-limit";
    game.seconds = started.elapsed().as_secs_f64();
    game
}
pub(super) fn targets(game: &mut Played) -> Vec<SavedMicroExample> {
    if game.error.is_some() {
        return vec![];
    }
    let mut saved: Vec<_> = game
        .samples
        .drain(..)
        .filter_map(|mut sample| {
            if let Some(proven) = sample.saved.tactical.as_ref().and_then(|t| t.root_value) {
                // A solved position has a valid minimax label even if the actual
                // game later hits a cap. This never changes its PSR/Elo outcome.
                sample.saved.value = proven as f64;
                sample.saved.reason = "search-proven-value".into();
            } else if let Some(cycle) = &game.cycle {
                if sample.player != cycle.loser.player()
                    || sample.saved.decision < cycle.first_decision
                {
                    return None;
                }
                sample.saved.value = -1.0;
                sample.saved.policy_weight = 0.0;
                sample.saved.policy.clear();
                sample.saved.actions.clear();
                sample.saved.action_features.clear();
                sample.saved.new_visits.clear();
                sample.saved.policy_raw_visits.clear();
                sample.saved.policy_pruned_visits.clear();
                sample.saved.tactical = None;
                sample.saved.correction_priority = false;
                sample.saved.reason = "repetition-training-loss".into();
            } else if game.reanalysis {
                if sample.saved.policy_weight == 0.0 {
                    return None;
                }
                sample.saved.value = sample.q;
                sample.saved.reason = "fresh-reanalysis-q".into();
            } else {
                let z = match game.outcome {
                    GameOutcome::Win(p) => {
                        if p == sample.player {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    GameOutcome::Draw => 0.0,
                    GameOutcome::Ongoing => return None,
                };
                sample.saved.value = if game.case_mode {
                    z
                } else {
                    0.5 * sample.q + 0.5 * z
                };
                sample.saved.reason = if game.case_mode {
                    "rules-terminal-z"
                } else {
                    "rules-terminal-q-mix"
                }
                .into();
            }
            Some(sample.saved)
        })
        .collect();
    super::super::tactics::prioritize(&mut saved);
    saved
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn truncated_game_keeps_only_exact_position_proofs_without_fabricating_a_win() {
        let o = Options {
            decision_limit: 0,
            ..Default::default()
        };
        let model = Arc::new(MicroModel::seeded(5));
        let snapshot = Arc::new(Snapshot {
            artifact: None,
            version: 0,
            identity: "a".repeat(64),
            model,
            path: PathBuf::new(),
        });
        let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
        let mut game = play(
            0,
            snapshot.clone(),
            snapshot,
            None,
            &o,
            Instant::now() + Duration::from_secs(1),
            &pool,
            "test",
        );
        let proven = super::super::super::tactics::fixture();
        let mut unknown = proven.clone();
        unknown.tactical = None;
        game.samples = vec![
            Sample {
                saved: proven,
                player: Player::Host,
                q: -0.5,
            },
            Sample {
                saved: unknown,
                player: Player::Host,
                q: 0.99,
            },
        ];
        game.cycle = Some(RepetitionLoss {
            loser: crate::compact_selfplay::Seat::Host,
            first_decision: 999,
            last_decision: 1000,
            period: 2,
            cycles: 4,
        });
        let result = targets(&mut game);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, 1.0);
        assert_eq!(result[0].reason, "search-proven-value");
        assert!(result[0].correction_priority);
        assert!(result[0].example_for_rules(RULES).is_ok());
        assert_eq!(game.outcome, GameOutcome::Ongoing);
        assert_eq!(game.termination, "decision-limit");
        assert_eq!(
            game.record.replay().unwrap().outcome(),
            GameOutcome::Ongoing
        );
    }
}

/// Reanalyse disagreements on the losing side; always retain the case root as a
/// candidate, and preserve the complete exact prefix for every selected state.
pub(super) fn correction_prefixes(game: &Played, maximum: usize) -> Vec<GameRecord> {
    let mut samples: Vec<_> = game
        .samples
        .iter()
        .filter(|s| match game.outcome {
            GameOutcome::Win(winner) => s.player != winner,
            _ => true,
        })
        .collect();
    samples.sort_by(|a, b| {
        let error = |s: &Sample| match game.outcome {
            GameOutcome::Win(w) => (s.q - if w == s.player { 1.0 } else { -1.0 }).abs(),
            _ => s.q.abs(),
        };
        error(b).total_cmp(&error(a))
    });
    let mut indices = vec![game.prefix_decisions];
    for s in samples {
        if indices.len() >= maximum {
            break;
        }
        let index = s.saved.decision - 1;
        if !indices.contains(&index) {
            indices.push(index);
        }
    }
    indices
        .into_iter()
        .map(|i| cases::prefix(&game.record, i))
        .collect()
}
