use super::*;
use crate::compact_selfplay::reuse::RepetitionLoss;
use paisho_core::legal_actions;
mod evidence;

pub(super) struct Sample {
    saved: SavedMicroExample,
    player: Player,
    q: f64,
}
pub(super) struct Played {
    pub reused_search: bool,
    pub search_cache_key: Option<String>,
    pub search_evidence_simulations: usize,
    pub case: Option<cases::Attempt>,
    pub ack: Option<std::sync::mpsc::SyncSender<case_actor::Feedback>>,
    pub certificates: Vec<(usize, MicroProofCertificate)>,
    pub reanalysis: bool,
    pub loop_repair: bool,
    pub measurement: bool,
    pub evaluation: Option<evaluation::Attempt>,
    pub observed_origin: Option<(f64, String)>,
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
    pub tactical_evaluations: usize,
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
    let reference: Option<(&dyn MctsEvaluator, usize, String, bool)> = o
        .match_reference
        .as_ref()
        .map(|(m, b)| {
            (
                m.as_ref() as &dyn MctsEvaluator,
                *b,
                m.name(),
                m.spec.solver,
            )
        })
        .or_else(|| legacy.map(|(m, b)| (m as &dyn MctsEvaluator, b, "Gen3.1".into(), false)));
    let started = paisho_platform::training_time::now();
    let cap = reference.as_ref().map_or(
        if start.is_some() {
            None
        } else {
            Some(o.game_seconds)
        },
        |(_, b, _, _)| {
            if o.legacy_replay || o.historical_unlimited_budgets.contains(b) {
                None
            } else {
                Some(o.historical_seconds.iter().find(|(v, _)| v == b).unwrap().1)
            }
        },
    );
    let game_deadline = cap.map(|seconds| started + Duration::from_secs_f64(seconds));
    let deadline = game_deadline.map_or(end, |limit| end.min(limit));
    let mut rng = StableRng::new(if o.measurement {
        0x66726f7a656e
    } else {
        o.seed.wrapping_add(id as u64)
    });
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
    let same = reference.is_none() && snapshot.identity == opponent.identity;
    let lane = if reanalysis {
        Lane::Reanalysis
    } else if reference.is_some() {
        Lane::Historical
    } else if same {
        Lane::Selfplay
    } else {
        Lane::Checkpoint
    };
    let mut sessions = if same || reference.is_some() {
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
    for session in &mut sessions {
        session
            .set_root_value_strength(o.value_policy_strength)
            .expect("validated coupling");
    }
    let mut old = reference.as_ref().map(|(model, budget, _, solver)| {
        let mut session = MctsSession::new(
            o.seed.wrapping_add(id as u64),
            MctsConfig {
                simulations: *budget,
                ..Default::default()
            },
            *model,
        )
        .expect("validated fixed budget");
        session.set_solver(*solver);
        session
    });
    let mut game = Played {
        reused_search: false,
        search_cache_key: None,
        search_evidence_simulations: 0,
        case: None,
        ack: None,
        certificates: vec![],
        reanalysis,
        loop_repair: o.learning_loop_repair,
        measurement: o.measurement,
        evaluation: None,
        observed_origin: o.observed_origin.clone(),
        prefix_decisions,
        case_mode: o.case_curriculum.is_some() || start.is_some(),
        id,
        lane,
        snapshot: snapshot.clone(),
        opponent: reference
            .as_ref()
            .map_or_else(|| opponent.identity.clone(), |r| r.2.clone()),
        reference_budget: reference.as_ref().map(|r| r.1),
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
        tactical_evaluations: 0,
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
            if o.stop_signal
                .as_ref()
                .is_some_and(|s| s.load(std::sync::atomic::Ordering::Relaxed))
            {
                game.termination = "user-stop".into();
                break;
            }
            if position.outcome() != GameOutcome::Ongoing {
                game.termination = "rules-terminal".into();
                break;
            }
            if paisho_platform::training_time::now() >= deadline {
                game.termination = "wall-limit".into();
                break;
            }
            let player = position.to_move();
            let historical_turn = old.is_some() && player != candidate_seat;
            let queued = paisho_platform::training_time::now();
            let action = if historical_turn {
                let (report, actions, search) = pool.install(|| {
                    let t = paisho_platform::training_time::now();
                    let actions = legal_actions(&position);
                    let report =
                        old.as_mut()
                            .unwrap()
                            .search_until(&position, &actions, Some(deadline));
                    (report, actions, paisho_platform::training_time::elapsed(t).as_secs_f64())
                });
                game.search_seconds[1] += search;
                game.pool_wait_seconds += (paisho_platform::training_time::elapsed(queued).as_secs_f64() - search).max(0.0);
                let report = report?;
                game.simulations += report.simulations;
                if report.simulations == 0 && paisho_platform::training_time::now() >= deadline {
                    game.termination = "wall-limit".into();
                    break;
                }
                let action = actions[report.selected_index];
                if o.learning_loop_repair {
                    let (sample, certificate) = evidence::opponent(
                        &position,
                        action,
                        &actions,
                        &snapshot,
                        &format!(
                            "{}:{}",
                            game.opponent,
                            o.match_reference
                                .as_ref()
                                .map_or("legacy", |(r, _)| r.spec.sha256.as_str())
                        ),
                        game.reference_budget.unwrap(),
                        id,
                        source,
                        prefix_decisions + decision + 1,
                    )?;
                    game.samples.push(sample);
                    if let Some(c) = certificate {
                        game.certificates.push((prefix_decisions + decision + 1, c));
                    }
                }
                action
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
                let search_options = o.search(rng.next_u64(), !reanalysis && !o.measurement)?;
                let (report, search) = if o.learning_loop_v2 {
                    pool.install(|| {
                        let t = paisho_platform::training_time::now();
                        let i = if sessions.len() == 1 {
                            0
                        } else {
                            player.index()
                        };
                        let identity = if same || player == candidate_seat {
                            &snapshot.identity
                        } else {
                            &opponent.identity
                        };
                        let r = search_cache::search(
                            &mut sessions[i],
                            &position,
                            &game.record,
                            identity,
                            budget,
                            o.value_policy_strength,
                            search_options,
                            deadline,
                            proofs,
                            if reanalysis && decision == 0 {
                                o.reanalysis_cache.as_deref()
                            } else {
                                None
                            },
                        );
                        (r, paisho_platform::training_time::elapsed(t).as_secs_f64())
                    })
                } else {
                    let (r, seconds) = pool.install(|| {
                        let t = paisho_platform::training_time::now();
                        let session_index = if sessions.len() == 1 {
                            0
                        } else {
                            player.index()
                        };
                        if let Some(proofs) = proofs {
                            if !proofs.read().unwrap().is_empty() {
                                let key = crate::compact_learning::sha256(
                                    game.record.to_string().as_bytes(),
                                );
                                let certificate = match proof_cache::lookup(proofs, &key) {
                                    Ok(c) => c,
                                    Err(e) => {
                                        return (Err(e.to_string()), paisho_platform::training_time::elapsed(t).as_secs_f64())
                                    }
                                };
                                if let Some(c) = certificate {
                                    if let Err(e) =
                                        sessions[session_index].install_certificate(&position, &c)
                                    {
                                        return (Err(e), paisho_platform::training_time::elapsed(t).as_secs_f64());
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
                        (result, paisho_platform::training_time::elapsed(t).as_secs_f64())
                    });
                    (r.map(|r| (r, None, false, None)), seconds)
                };
                game.search_seconds[0] += search;
                game.pool_wait_seconds += (paisho_platform::training_time::elapsed(queued).as_secs_f64() - search).max(0.0);
                let (mut report, cached_certificate, reused_search, cache_key) = report?;
                game.reused_search = reused_search;
                game.search_cache_key = cache_key;
                game.search_evidence_simulations += report.simulations;
                let completed_q = if o.learning_loop_repair {
                    let (target, q) = teaching::target(&report).map_err(|e| e.to_string())?;
                    report.policy_target = target;
                    Some(q)
                } else {
                    None
                };
                // Playing keeps the coupled search distribution. Only the raw
                // network target changes coordinates; no successor values are recomputed.
                let raw_prior = if o.learning_loop_v2 {
                    if let Some(prior) = &report.raw_priors {
                        prior.as_ref().clone()
                    } else {
                    let model = if same || player == candidate_seat {
                        &snapshot.model
                    } else {
                        &opponent.model
                    };
                    let base = micro_softmax(&MicroModel::logits(
                        &model.embed(&report.state),
                        &report.action_features,
                    ))?;
                    model.memory_priors(&report.state, &report.action_features, &base, 0)?
                    }
                } else {
                    report.priors.clone()
                };
                let learned_policy = if o.learning_loop_v2 {
                    teaching::target_with_prior(&report, &raw_prior)
                        .map_err(|e| e.to_string())?
                        .0
                } else {
                    report.policy_target.clone()
                };
                if report.proven_value.is_some() && (start.is_some() || o.learning_loop_repair) {
                    let index = if sessions.len() == 1 {
                        0
                    } else {
                        player.index()
                    };
                    if let Some(c) =
                        cached_certificate.or_else(|| sessions[index].certificate(10_000))
                    {
                        game.certificates.push((prefix_decisions + decision + 1, c));
                    }
                }
                if report.simulations == 0 && report.proven_value.is_none() {
                    game.termination = "wall-limit".into();
                    break;
                }
                if !reused_search {
                    game.simulations += report.simulations;
                    game.tactical_evaluations += report.tactical_evaluations;
                    game.inherited += report.inherited_visits;
                    game.hits += report.inference_cache_hits;
                    game.evals += report.inference_evaluations;
                    game.forced_playouts += report.new_forced_visits.iter().sum::<usize>();
                }
                let sampling = paisho_platform::training_time::now();
                // Gumbel already samples without replacement at root. Do not replace
                // its recommended action with a fresh visit draw.
                let has_proof = report.proven_action_values.iter().any(Option::is_some);
                let selected = if !reanalysis
                    && !o.measurement
                    && (if o.learning_loop_repair {
                        prefix_decisions + decision < 40
                    } else {
                        decision < 40
                    })
                    && search_options.mode == MicroSearchMode::Puct
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
                        evidence: if o.learning_loop_repair {
                            Some(TargetEvidence { policy_support:false,
                                policy_coordinates: if o.learning_loop_v2 {
                                    "raw-policy-v1".into()
                                } else {
                                    String::new()
                                },
                                search_prior: if o.learning_loop_v2 && policy_weight > 0. {
                                    report.priors.clone()
                                } else {
                                    vec![]
                                },
                                coupling_strength: o
                                    .learning_loop_v2
                                    .then_some(o.value_policy_strength),
                                observed_value: None,
                                observed_psr: None,
                                estimated_value: Some(report.values[selected]),
                                value_weight: 1.,
                                policy_source: if report.proven_value.is_some() {
                                    "verified-search"
                                } else {
                                    "full-search-estimate"
                                }
                                .into(),
                                action_value_visits: if o.learning_loop_v3 && policy_weight > 0. {
                                    report.visits.clone()
                                } else { vec![] },
                                completed_action_values: if policy_weight > 0. {
                                    completed_q.unwrap_or_default()
                                } else {
                                    vec![]
                                },
                                target_prior: if policy_weight > 0. {
                                    raw_prior.clone()
                                } else {
                                    vec![]
                                },
                                excluded_actions: if policy_weight > 0. {
                                    (0..report.actions.len())
                                        .map(|i| {
                                            report.proven_action_values.get(i).copied().flatten()
                                                == Some(-1)
                                        })
                                        .collect()
                                } else {
                                    vec![]
                                },
                                player: player.code().to_string(),
                                actor: collector.clone(),
                            })
                        } else {
                            None
                        },
                        tactical: TacticalEvidence::from_report(&report, policy_weight > 0.0),
                        correction_priority: false,
                        policy_raw_visits: if !o.learning_loop_repair
                            && policy_weight > 0.0
                            && report.pruned_visits.iter().any(|n| *n > 0)
                            && report.visits.iter().sum::<usize>()
                                > report.pruned_visits.iter().sum::<usize>()
                        {
                            report.visits.clone()
                        } else {
                            vec![]
                        },
                        policy_pruned_visits: if !o.learning_loop_repair
                            && policy_weight > 0.0
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
                            learned_policy
                        } else {
                            vec![]
                        },
                        value: 0.0,
                        policy_weight,
                        reason: String::new(),
                    },
                });
                game.sample_seconds += paisho_platform::training_time::elapsed(sampling).as_secs_f64();
                report.actions[selected]
            };
            if o.learning_loop_v3 {
                let decision=prefix_decisions+decision+1;
                if let (Some(sample),Some((proof_decision,certificate)))=(game.samples.last_mut(),game.certificates.last()) {
                    if sample.saved.decision==decision && *proof_decision==decision
                        && sample.saved.policy_weight>0.
                        && sample.saved.tactical.as_ref().is_some_and(|t|t.root_value==Some(1)) {
                        let started=paisho_platform::training_time::now();
                        action_values::attach_winning_support(&mut sample.saved,&position,certificate).map_err(|e|e.to_string())?;
                        game.sample_seconds+=paisho_platform::training_time::elapsed(started).as_secs_f64();
                    }
                }
            }
            if reanalysis {
                game.termination = "reanalysis-search".into();
                break;
            }
            let t = paisho_platform::training_time::now();
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
            game.maintenance_seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
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
    game.seconds = paisho_platform::training_time::elapsed(started).as_secs_f64();
    game
}
pub(super) fn targets(game: &mut Played) -> Vec<SavedMicroExample> {
    if game.error.is_some() {
        return vec![];
    }
    let psr_hash = sha256(game.record.to_string().as_bytes());
    let mut saved: Vec<_> = game
        .samples
        .drain(..)
        .filter_map(|mut sample| {
            if let Some(e) = &mut sample.saved.evidence {
                let observed = if game.reanalysis {
                    game.observed_origin
                        .clone()
                        .map(|(z, h)| (if sample.player == Player::Host { z } else { -z }, h))
                } else {
                    match game.outcome {
                        GameOutcome::Win(w) => {
                            Some((if w == sample.player { 1. } else { -1. }, psr_hash.clone()))
                        }
                        GameOutcome::Draw => Some((0., psr_hash.clone())),
                        _ => None,
                    }
                };
                if let Some((z, hash)) = observed {
                    e.observed_value = Some(z);
                    e.observed_psr = Some(hash);
                }
            }

            if game.loop_repair
                && !game.reanalysis
                && game.outcome == GameOutcome::Ongoing
                && !sample
                    .saved
                    .tactical
                    .as_ref()
                    .is_some_and(|t| t.root_value.is_some())
            {
                if sample.saved.policy_weight == 0. {
                    return None;
                }
                sample.saved.value = sample.q;
                sample.saved.evidence.as_mut().unwrap().value_weight = 0.;
                sample.saved.reason = "censored-policy-only".into();
                return Some(sample.saved);
            }
            if let Some(proven) = sample.saved.tactical.as_ref().and_then(|t| t.root_value) {
                // A solved position has a valid minimax label even if the actual
                // game later hits a cap. This never changes its PSR/Elo outcome.
                sample.saved.value = proven as f64;
                sample.saved.reason = "search-proven-value".into();
            } else if let Some(cycle) = &game.cycle {
                // A repeated unfinished game is censored evidence, not a regulatory loss.
                if game.snapshot.model.has_spatial() {
                    return None;
                }
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
                if let Some(e) = &mut sample.saved.evidence {
                    sample.saved.value = e.observed_value.unwrap_or(sample.q);
                    e.value_weight = if e.observed_value.is_some() { 1. } else { 0.25 };
                    sample.saved.reason = if e.observed_value.is_some() {
                        "reanalysis-with-observed-outcome"
                    } else {
                        "fresh-reanalysis-q"
                    }
                    .into();
                } else {
                    sample.saved.value = sample.q;
                    sample.saved.reason = "fresh-reanalysis-q".into();
                }
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
            paisho_platform::training_time::now() + Duration::from_secs(1),
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
#[cfg(test)]
mod loop_tests {
    use super::*;
    #[test]
    fn opponent_decision_proof_and_empirical_sign_survive_target_recording() {
        let record: GameRecord = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../paisho-ai/tests/fixtures/micro-alias-0-a.psr"
        ))
        .parse()
        .unwrap();
        let p = record.replay().unwrap();
        let legal = legal_actions(&p);
        let action = *legal
            .iter()
            .find(|a| {
                let mut n = p.clone();
                n.apply(**a).unwrap();
                n.outcome() == GameOutcome::Win(p.to_move())
            })
            .unwrap();
        let snapshot = Arc::new(Snapshot {
            artifact: None,
            version: 2,
            identity: "a".repeat(64),
            model: Arc::new(MicroModel::seeded(4).with_spatial_policy()),
            path: "unused".into(),
        });
        let (sample, cert) = evidence::opponent(
            &p,
            action,
            &legal,
            &snapshot,
            "Gen3.5:hash",
            8,
            1,
            "run",
            record.actions().len() + 1,
        )
        .unwrap();
        cert.as_ref().unwrap().verify(&p).unwrap();
        let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
        let mut game = play_from(
            1,
            snapshot.clone(),
            snapshot.clone(),
            None,
            &Options {
                learning_loop_repair: true,
                ..Default::default()
            },
            paisho_platform::training_time::now(),
            &pool,
            "run",
            Some(&record),
            false,
            None,
        );
        game.record.push(action);
        game.outcome = GameOutcome::Win(p.to_move());
        game.error = None;
        game.samples = vec![sample];
        game.certificates = vec![(game.record.actions().len(), cert.unwrap())];
        let rows = targets(&mut game);
        assert_eq!(rows.len(), 1);
        let s = &rows[0];
        s.example_for_rules(RULES).unwrap();
        assert_eq!(s.value, 1.);
        assert_eq!(s.evidence.as_ref().unwrap().observed_value, Some(1.));
        assert_eq!(
            s.evidence.as_ref().unwrap().policy_source,
            "verified-regulatory-win"
        );
        let p = record.initial_position();
        let legal = legal_actions(&p);
        let (sample, proof) = evidence::opponent(
            &p,
            legal[0],
            &legal,
            &snapshot,
            "Gen3.5:hash",
            8,
            1,
            "run",
            1,
        )
        .unwrap();
        assert!(proof.is_none());
        assert_eq!(sample.saved.policy_weight, 0.);
        assert!(sample.saved.actions.is_empty());
    }
}
