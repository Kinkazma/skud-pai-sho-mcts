//! Bounded native collector -> archive -> recall -> protected learner -> guard -> actor.
//! Deterministic serial receipt order is deliberate. This does not measure the
//! asynchronous production campaign, its complete FIFO, or its opponent mixture.
use super::*;
use std::collections::{BTreeSet, VecDeque};
#[path = "loop_cycle_admission.rs"]
mod admission;
#[path = "final_evaluation_probe.rs"]
mod final_evaluation_probe;
#[path = "loop_cycle_probe/replay.rs"]
mod replay;
pub use final_evaluation_probe::run as run_final_evaluation;
#[path = "loop_cycle_probe/learner_tape_probe.rs"]
mod learner_tape_probe;
pub use learner_tape_probe::measure_search_transfer;
pub use learner_tape_probe::{run as run_learner_tape,preflight as preflight_learner_tape,run_step_fractions};
pub use learner_tape_probe::measure_base_policy;

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn bytes(&self) -> Result<Vec<u8>> {
        let b = fs::read(&self.path)?;
        if sha256(&b) != self.sha256 {
            return Err(invalid(format!(
                "changed diagnostic input {}",
                self.path.display()
            )));
        }
        Ok(b)
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Prefix {
    input: Input,
    decisions: usize,
    source_group: String,
    #[serde(default)]
    reanalysis: bool,
}
impl Prefix {
    fn load(&self) -> Result<GameRecord> {
        let bytes = self.input.bytes()?;
        let original: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        original.replay()?;
        let all = if original.rules() == RULES {
            original
        } else {
            original.replay_prefix_with_rules(RULES)?.0
        };
        if self.decisions > all.actions().len() || self.source_group.is_empty() {
            return Err(invalid("invalid diagnostic prefix provenance"));
        }
        all.replay()?;
        let r = cases::prefix(&all, self.decisions);
        if r.replay()?.outcome() != GameOutcome::Ongoing {
            return Err(invalid("terminal diagnostic prefix"));
        }
        Ok(r)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    initial_model: Input,
    #[serde(default)]
    initial_learner: Option<Input>,
    initial_examples: Input,
    recall_bundles: Vec<Input>,
    recall_proofs: Vec<Input>,
    cycles: Vec<Vec<Prefix>>,
    evaluation: Vec<Prefix>,
    /// True enables the newly repaired publication protocol, not a weight migration.
    repaired: bool,
    /// Transfer only certified choices whose native targets were fully consumed.
    #[serde(default)]
    publication_transfer: bool,
    /// Authenticated historical consumption receipts, capped at the initial actor.
    #[serde(default)]
    initial_transfer_proofs: Option<Input>,
    #[serde(default)]
    policy_confidence_constraint: Option<bool>,
    /// Diagnostic comparator for the new V3 runtime's aligned value constraint.
    #[serde(default)]
    validation_value_constraint: bool,
    /// Opt-in native receipt protocol: all fresh, Memory replay, immediate lesson
    /// admission, observed reanalysis targets, and no cursor-consuming warmup.
    #[serde(default)]
    immediate_lesson_admission: bool,
    /// Reuse the already verified tape consolidation in the bounded relay follow-up.
    #[serde(default)]
    diagnostic_two_choice_consolidation: bool,
    seed: u64,
    /// Bounded diagnostic FIFO; must never be labelled production throughput.
    fifo_capacity: usize,
    /// Fixed, position-independent stride selection of each new game's targets.
    max_fresh_per_game: usize,
    /// New decisions, not a whole-record cutoff including the given prefix.
    decision_limit: usize,
    max_seconds: u64,
}

fn snapshot(
    m: &MicroModel,
    updates: u64,
    version: u64,
    out: &Path,
    label: &str,
) -> Result<Arc<Snapshot>> {
    let a = Arc::new(MicroArtifact::new(
        m,
        updates,
        serde_json::json!({"diagnostic_only":true,"label":label,"version":version}),
    ));
    let path = out.join(format!("{label}.json"));
    a.save(&path)?;
    Ok(Arc::new(Snapshot {
        identity: a.identity(),
        artifact: Some(a),
        version,
        model: Arc::new(m.clone()),
        path,
    }))
}
fn frozen_evaluation(
    actor: &Arc<Snapshot>,
    anchor: &Arc<Snapshot>,
    prefixes: &[(Prefix, GameRecord)],
    o: &Options,
    pool: &cpu::Executor,
    end: Instant,
    out: &Path,
) -> Result<serde_json::Value> {
    fs::create_dir(out)?;
    let t = Instant::now();
    let mut rows = vec![];
    let mut wdl = [0usize; 4];
    for (i, (spec, record)) in prefixes.iter().enumerate() {
        for seat in [Player::Host, Player::Guest] {
            if Instant::now() >= end {
                return Err(invalid("diagnostic deadline before frozen game completion"));
            }
            let id = i * 2 + usize::from(seat == Player::Guest);
            let mut options = o.clone();
            options.measurement = true;
            options.candidate_seat = Some(seat);
            options.observed_origin = None;
            let game = collector::play_from(
                id,
                actor.clone(),
                anchor.clone(),
                None,
                &options,
                end,
                pool,
                "gen5-loop-probe-frozen",
                Some(record),
                false,
                None,
            );
            if let Some(e) = game.error {
                return Err(invalid(e));
            }
            if game.campaign_censored {
                return Err(invalid(
                    "censored frozen evaluation; increase diagnostic bound",
                ));
            }
            let psr = game.record.to_string();
            let replayed = game.record.replay()?;
            if replayed.outcome() != game.outcome {
                return Err(invalid("frozen PSR outcome mismatch"));
            }
            let result = match game.outcome {
                GameOutcome::Win(p) if p == seat => 0,
                GameOutcome::Draw => 1,
                GameOutcome::Win(_) => 2,
                GameOutcome::Ongoing => 3,
            };
            wdl[result] += 1;
            let path = out.join(format!("game-{id:04}.psr"));
            fs::write(&path, &psr)?;
            rows.push(serde_json::json!({"id":id,"source_group":spec.source_group,"candidate_seat":format!("{seat:?}"),
                "result_index_wdlu":result,"outcome":format!("{:?}",game.outcome),"termination":game.termination,
                "decisions":game.record.actions().len()-game.prefix_decisions,"psr":path,"sha256":sha256(psr.as_bytes()),
                "seconds":game.seconds,"simulations":game.simulations,"collector":actor.identity,"opponent":anchor.identity}));
        }
    }
    let report = serde_json::json!({"wdlu":wdl,"rows":rows,"seconds":t.elapsed().as_secs_f64(),"learned":0,
        "paired_source_groups":prefixes.len(),"seeds_budgets_prefixes_fixed":true,"not_independent_games_within_source":true});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}

/// All mutable paths are children of a fresh output directory. Input hashes and
/// frozen source separation are checked before and after the diagnostic.
pub fn run(config: &Path, plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("diagnostic output must not exist"));
    }
    let loading = Instant::now();
    let config_bytes = fs::read(config)?;
    let plan_bytes = fs::read(plan_path)?;
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    let mut o: Options = serde_json::from_slice(&config_bytes)?;
    if plan.schema != "paisho-gen5-loop-cycle-probe-v1"
        || !(2..=24).contains(&plan.cycles.len())
        || plan.cycles.iter().any(|c| c.is_empty() || c.len() > 64)
        || plan.evaluation.is_empty()
        || plan.evaluation.len() > 32
        || plan.recall_bundles.is_empty()
        || plan.recall_bundles.len() > 512
        || plan.recall_proofs.is_empty()
        || plan.recall_proofs.len() > 512
        || !(64..=327680).contains(&plan.fifo_capacity)
        || !(1..=512).contains(&plan.max_fresh_per_game)
        || !(1..=2048).contains(&plan.decision_limit)
        || !(30..=3600).contains(&plan.max_seconds)
        || o.replay_ratio != 4
        || o.recall_fraction != 0.5
        || !o.learning_loop_v2
        || (plan.diagnostic_two_choice_consolidation && !plan.repaired)
        || (plan.publication_transfer
            && (!plan.repaired || !plan.immediate_lesson_admission))
        || (plan.initial_transfer_proofs.is_some() && !plan.publication_transfer)
        || (plan.immediate_lesson_admission
            && (!o.structural_repair || !o.learning_loop_repair || !o.learn))
    {
        return Err(invalid("invalid bounded native loop plan"));
    }
    let all_inputs = std::iter::once(&plan.initial_model)
        .chain(plan.initial_learner.iter())
        .chain(plan.initial_transfer_proofs.iter())
        .chain(std::iter::once(&plan.initial_examples))
        .chain(plan.recall_bundles.iter())
        .chain(plan.recall_proofs.iter())
        .chain(plan.cycles.iter().flatten().map(|s| &s.input))
        .chain(plan.evaluation.iter().map(|s| &s.input))
        .cloned()
        .collect::<Vec<_>>();
    for input in &all_inputs {
        input.bytes()?;
    }
    let train_sources = plan
        .cycles
        .iter()
        .flatten()
        .map(|r| r.source_group.clone())
        .collect::<BTreeSet<_>>();
    if plan
        .evaluation
        .iter()
        .any(|r| r.reanalysis || train_sources.contains(&r.source_group))
    {
        return Err(invalid(
            "frozen evaluation source overlaps the training prefixes",
        ));
    }
    let training = plan
        .cycles
        .iter()
        .map(|cycle| {
            cycle
                .iter()
                .map(|r| Ok((r.clone(), r.load()?)))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let evaluation = plan
        .evaluation
        .iter()
        .map(|r| Ok((r.clone(), r.load()?)))
        .collect::<Result<Vec<_>>>()?;
    let train_states = training
        .iter()
        .flatten()
        .map(|(_, r)| sha256(r.to_string().as_bytes()))
        .collect::<BTreeSet<_>>();
    if evaluation
        .iter()
        .any(|(_, r)| train_states.contains(&sha256(r.to_string().as_bytes())))
    {
        return Err(invalid("frozen evaluation prefix is a training prefix"));
    }
    let a: MicroArtifact = serde_json::from_slice(&plan.initial_model.bytes()?)?;
    let actor_model = a.model()?;
    let learner_artifact = plan
        .initial_learner
        .as_ref()
        .map(|i| {
            Ok::<MicroArtifact, Box<dyn std::error::Error>>(serde_json::from_slice(&i.bytes()?)?)
        })
        .transpose()?;
    let mut model = match &learner_artifact {
        Some(a) => a.model()?,
        None => actor_model.clone(),
    };
    if model.schema() != actor_model.schema()
        || model.feature_schema() != actor_model.feature_schema()
    {
        return Err(invalid("learner and actor schemas differ"));
    }
    let mut updates = learner_artifact.as_ref().unwrap_or(&a).updates;
    let initial_parameters = model.parameters().to_vec();
    fs::create_dir(out)?;
    fs::write(out.join("plan.json"), &plan_bytes)?;
    fs::write(out.join("input-config.json"), &config_bytes)?;
    let copied = out.join("recall-input");
    fs::create_dir(&copied)?;
    fs::create_dir(copied.join("proofs"))?;
    for (proof, inputs) in [(false, &plan.recall_bundles), (true, &plan.recall_proofs)] {
        for input in inputs {
            let name = input
                .path
                .file_name()
                .ok_or_else(|| invalid("input filename"))?;
            let dest = if proof {
                copied.join("proofs").join(name)
            } else {
                copied.join(name)
            };
            if dest.exists() {
                return Err(invalid("duplicate recall filename in input manifest"));
            }
            fs::write(dest, input.bytes()?)?;
        }
    }
    let initial = snapshot(&actor_model, a.updates, 0, out, "actor-000")?;
    let guard_path = o
        .publication_guard
        .as_ref()
        .ok_or_else(|| invalid("missing guard"))?;
    let validation_path = o
        .publication_validation
        .as_ref()
        .ok_or_else(|| invalid("missing validation"))?;
    let guard_hash = sha256(&fs::read(guard_path)?);
    let validation_hash = sha256(&fs::read(validation_path)?);
    let mut guard = publication::Guard::open(
        guard_path,
        &out.join("publication"),
        o.value_policy_strength,
        initial.clone(),
        &serde_json::Value::Null,
    )?;
    guard.enable_v2(validation_path)?;
    if plan.repaired {
        guard.enable_v3()?;
    }
    let mut p = protection::Protection::new(&actor_model, guard.reference_examples())?;
    if plan.repaired {
        p.enable_loop_v3();
        if let Some(active) = plan.policy_confidence_constraint {
            p.diagnostic_set_policy_gradient_constraint(active);
        }
    }
    let (pool, _) = cpu::build_pool(o.threads, None)?;
    guard.enable_parallel(&[pool.clone()]);
    p.enable_parallel(&[pool.clone()]);
    if plan.publication_transfer {
        guard.enable_transfer()?;
        if let Some(input) = &plan.initial_transfer_proofs {
            // The initial actor, not a possibly newer private learner, fixes the
            // historical cutoff. The Guard verifies the exact consumed proofs.
            guard.diagnostic_bootstrap_transfer(&input.path, &input.sha256, a.updates)?;
        }
    }
    let initial_publication = guard.progress();
    if plan.validation_value_constraint {
        p.enable_validation_value(guard.diagnostic_validation_examples()?)?;
    }
    let executor = cpu::Executor::direct(pool.clone());
    let archive_path = out.join("archive");
    let mut archive = durable::Archive::open(&archive_path)?;
    archive.add_read_only(&copied)?;
    archive.policy_consolidation(true);
    archive.proof_recall = true;
    archive.trusted_action_values = plan.repaired;
    archive.enable_coverage(out, &serde_json::Value::Null, &model, &archive_path)?;
    archive.seed_values(guard.reference_examples());
    let examples: Vec<SavedMicroExample> = serde_json::from_slice(&plan.initial_examples.bytes()?)?;
    let mut fifo: VecDeque<_> = if plan.immediate_lesson_admission {
        VecDeque::new()
    } else {
        examples
            .iter()
            .map(|s| {
                s.example_for_rules_with_trusted_q(RULES, plan.repaired)
                    .map(Arc::new)
            })
            .collect::<Result<_>>()?
    };
    let evaluation_states = evaluation
        .iter()
        .map(|(_, record)| Ok(actor_model.state_features(&record.replay()?)))
        .collect::<Result<Vec<_>>>()?;
    if examples
        .iter()
        .any(|e| evaluation_states.iter().any(|s| *s == e.state))
    {
        return Err(invalid(
            "frozen evaluation root is present in the initial diagnostic FIFO",
        ));
    }
    for manifest in [guard_path, validation_path] {
        let panel: serde_json::Value = serde_json::from_slice(&fs::read(manifest)?)?;
        for row in panel["rows"]
            .as_array()
            .ok_or_else(|| invalid("corrective rows"))?
        {
            let record: GameRecord = row["prefix"]
                .as_str()
                .ok_or_else(|| invalid("corrective prefix"))?
                .parse()?;
            let state = actor_model.state_features(&record.replay()?);
            if evaluation_states.contains(&state) {
                return Err(invalid(
                    "frozen evaluation root is a corrective proof position",
                ));
            }
        }
    }
    if examples.is_empty() {
        return Err(invalid("initial diagnostic FIFO empty"));
    }
    while fifo.len() > plan.fifo_capacity {
        fifo.pop_front();
    }
    let mut native_replay = if plan.immediate_lesson_admission {
        Some(replay::Replay::open(
            &o,
            &plan.initial_examples,
            examples.len(),
            plan.fifo_capacity,
            plan.repaired,
            model.has_spatial(),
            out,
        )?)
    } else {
        None
    };
    let mut rng = StableRng::new(plan.seed);
    // Preserve old plans exactly. Production opens/preloads the queues without
    // consuming RNG, coverage cursors or a deferred 64-example presentation plan.
    if !plan.immediate_lesson_admission {
        let warm = archive.rehearse_cached(64, &mut rng, &model)?;
        let winning = archive.last_winning_draws;
        if warm.len() != 64 {
            return Err(invalid("bounded recall fixture cannot supply both queues"));
        }
        for (i, ex) in warm.into_iter().enumerate() {
            archive.defer(ex, i < winning);
        }
    }
    assert_eq!(initial_parameters, model.parameters());
    let loading_seconds = loading.elapsed().as_secs_f64();
    let active = Instant::now();
    let end = active + Duration::from_secs(plan.max_seconds);
    o.seed = plan.seed;
    o.learning_loop_v3 = plan.repaired;
    o.publication_transfer = plan.publication_transfer;
    o.measurement = false;
    o.decision_limit = plan.decision_limit;
    o.stop_signal = None;
    o.end_unix_seconds = None;
    o.observed_origin = None;
    // Explicit laboratory protocol: actual self-play and optional one-position
    // reanalysis. Five-reference curriculum and the scheduler are not reproduced.
    o.match_reference = None;
    o.opponents.clear();
    o.candidate_seat = None;
    let initial_scores = guard.diagnostic_measure(&actor_model)?;
    let initial_learner_scores = guard.diagnostic_measure(&model)?;
    let initial_games = frozen_evaluation(
        &initial,
        &initial,
        &evaluation,
        &o,
        &executor,
        end,
        &out.join("evaluation-000"),
    )?;
    let mut previous_evaluation = initial_games.clone();
    let mut quotas = recall::Quotas::default();
    let mut cycles = vec![];
    let mut snapshots = vec![
        serde_json::json!({"label":"initial_actor","path":initial.path,"sha256":sha256(&fs::read(&initial.path)?),"coupled":true}),
    ];
    for (ci, rows) in training.iter().enumerate() {
        let cycle_start = Instant::now();
        let cycle_dir = out.join(format!("cycle-{:03}", ci + 1));
        fs::create_dir(&cycle_dir)?;
        let old_actor = guard.accepted();
        let before_scores = guard.diagnostic_measure(&model)?;
        let mut receipts = vec![];
        let mut search_seconds = 0.;
        let mut learning_seconds = 0.;
        let mut recall_seconds = 0.;
        let mut archive_seconds = 0.;
        let mut fresh_count = 0;
        let mut recall_count = 0;
        let mut learned_count = 0;
        let mut correction_replay_count = 0usize;
        for (ri, (spec, prefix)) in rows.iter().enumerate() {
            if Instant::now() >= end {
                return Err(invalid("diagnostic deadline before cycle completion"));
            }
            let id = ci * 1000 + ri;
            let actor = guard.accepted();
            let (receipt_options, reanalysis_protocol) = if plan.immediate_lesson_admission {
                replay::reanalysis_options(&o, spec)?
            } else {
                (o.clone(), serde_json::Value::Null)
            };
            let t = Instant::now();
            let mut game = collector::play_from(
                id,
                actor.clone(),
                actor.clone(),
                None,
                &receipt_options,
                end,
                &executor,
                "gen5-loop-probe-training",
                Some(prefix),
                spec.reanalysis,
                Some(&archive.proofs),
            );
            search_seconds += t.elapsed().as_secs_f64();
            if let Some(e) = &game.error {
                return Err(invalid(e.clone()));
            }
            if game.campaign_censored {
                return Err(invalid("censored training game invalidates comparison"));
            }
            game.case = Some(cases::Attempt {
                opponent_generation: None,
                actor: 0,
                case: format!("probe-{ci}-{ri}"),
                human_source: spec.source_group.clone(),
                zone: 0,
                prefix_decisions: prefix.actions().len(),
                before: Default::default(),
                after: Default::default(),
                kind: if spec.reanalysis {
                    "reanalysis"
                } else {
                    "loop-probe-selfplay"
                }
                .into(),
            });
            let t = Instant::now();
            let saved = collector::targets(&mut game);
            if game.record.replay()?.outcome() != game.outcome {
                return Err(invalid("training PSR replay mismatch"));
            }
            let psr = game.record.to_string();
            let path = cycle_dir.join(format!("game-{ri:03}.psr"));
            fs::write(&path, &psr)?;
            let targets = cycle_dir.join(format!("game-{ri:03}.targets.json.gz"));
            save_examples_new(&targets, &saved)?;
            if let Some(path) =
                durable::persist_mode(&archive_path, &game, &saved, &archive.proofs, false)?
            {
                archive.add(path);
            }
            for (decision, _) in &game.certificates {
                let key = sha256(
                    cases::prefix(&game.record, decision - 1)
                        .to_string()
                        .as_bytes(),
                );
                archive.register_persisted_proof(
                    archive_path.join("proofs").join(format!("{key}.json")),
                );
            }
            let stride = saved.len().div_ceil(plan.max_fresh_per_game).max(1);
            if plan.immediate_lesson_admission && stride != 1 {
                return Err(invalid("immediate lesson admission requires every fresh row; increase max_fresh_per_game"));
            }
            let owned = saved
                .iter()
                .step_by(stride)
                .map(|s| {
                    s.example_for_rules_with_trusted_q(RULES, plan.repaired)
                        .map(Arc::new)
                })
                .collect::<Result<Vec<_>>>()?;
            let lessons = if plan.immediate_lesson_admission {
                admission::prepare(&game, &saved, &owned)?
            } else {
                vec![]
            };
            if owned.iter().any(|e| evaluation_states.contains(&e.state)) {
                return Err(invalid(
                    "new collection collided with a frozen evaluation root",
                ));
            }
            archive_seconds += t.elapsed().as_secs_f64();
            let mut teachers = std::collections::BTreeMap::<String, usize>::new();
            for s in &saved {
                *teachers
                    .entry(
                        s.evidence
                            .as_ref()
                            .map_or("legacy", |e| e.policy_source.as_str())
                            .into(),
                    )
                    .or_default() += 1;
            }
            let targets_hash = sha256(&fs::read(&targets)?);
            receipts.push(serde_json::json!({"id":id,"actor":actor.identity,"actor_version":actor.version,"source_group":spec.source_group,
                "reanalysis":spec.reanalysis,"terminal":game.outcome!=GameOutcome::Ongoing,"outcome":format!("{:?}",game.outcome),"termination":game.termination,
                "new_decisions":game.record.actions().len()-game.prefix_decisions,"all_fresh":saved.len(),"selected_fresh":owned.len(),"stride":stride,
                "teachers":teachers,"observed_targets":saved.iter().filter(|s|s.evidence.as_ref().is_some_and(|e|e.observed_value.is_some())).count(),
                "reanalysis_protocol":reanalysis_protocol,"certificates":game.certificates.len(),"psr":path,"psr_sha256":sha256(psr.as_bytes()),"targets":targets,"targets_sha256":targets_hash}));
            p.observe(&owned);
            fresh_count += owned.len();
            let (due, _) = quotas.allocate(owned.len(), o.replay_ratio, 0.5, 0.);
            let t = Instant::now();
            let recalled = archive.rehearse_cached(due, &mut rng, &model)?;
            if recalled.len() != due {
                return Err(invalid(
                    "actual native recall could not supply 50 percent quota",
                ));
            }
            recall_seconds += t.elapsed().as_secs_f64();
            recall_count += recalled.len();
            let mut mixed = owned
                .iter()
                .cloned()
                .map(|e| (e, false))
                .collect::<Vec<_>>();
            mixed.extend(recalled.into_iter().map(|e| (e, true)));
            for _ in 0..owned.len() * o.replay_ratio - due {
                let ex = if let Some(memory) = &mut native_replay {
                    let (ex, kind) = memory.draw(&mut rng, &owned, game.lane)?;
                    correction_replay_count += usize::from(kind == 3);
                    ex
                } else {
                    fifo[rng.index(fifo.len())].clone()
                };
                mixed.push((ex, false));
            }
            shuffle(&mut mixed, &mut rng);
            let t = Instant::now();
            for batch in mixed.chunks(64) {
                if Instant::now() >= end {
                    return Err(invalid("diagnostic deadline during transaction"));
                }
                let shared = batch.iter().map(|(e, _)| e.clone()).collect::<Vec<_>>();
                p.train_shared(&mut model, &shared, o.rate * batch.len() as f64 / 64., 1e-5)?;
                updates += 1;
                learned_count += batch.len();
                quotas.consumed_examples += batch.len();
                quotas.consumed_recall += batch.iter().filter(|(_, recall)| *recall).count();
            }
            learning_seconds += t.elapsed().as_secs_f64();
            if let Some(memory) = &mut native_replay {
                let t = Instant::now();
                memory.admit(&owned, &targets, &targets_hash, game.lane, &saved)?;
                archive_seconds += t.elapsed().as_secs_f64();
            }
            // All batches completed; any interrupted/erroring batch returned
            // above. As in runtime, these lessons cannot supply their own first
            // receipt's recall. Publication remains the laboratory cycle boundary.
            if plan.immediate_lesson_admission {
                let t = Instant::now();
                let admitted = admission::complete(&mut archive, lessons);
                receipts.last_mut().unwrap()["immediate_lesson_admission"] = admitted;
                if plan.publication_transfer {
                    // Every fresh target completed SGD above. Neither persisted
                    // proofs nor an unfinished receipt alone can enter retention.
                    guard.observe_consumed(
                        &game,
                        &saved,
                        true,
                        "gen5-loop-probe-training",
                        updates,
                    )?;
                }
                archive_seconds += t.elapsed().as_secs_f64();
            }
            if !plan.immediate_lesson_admission {
                fifo.extend(owned);
                while fifo.len() > plan.fifo_capacity {
                    fifo.pop_front();
                }
            }
        }
        let before_consolidation = snapshot(
            &model,
            updates,
            ci as u64 + 1,
            out,
            &format!("learner-pre-{:03}", ci + 1),
        )?;
        let unprotected_scores = guard.diagnostic_measure(&model)?;
        if plan.repaired {
            archive.focus_policy(guard.observe_candidate(&model)?);
        }
        let t = Instant::now();
        if plan.publication_transfer {
            p.consolidate_for_publication(&mut model)?;
            guard.set_publication_fresh(p.publication_fresh_contract()?)?;
        } else if plan.diagnostic_two_choice_consolidation {
            p.diagnostic_consolidate_two_choices(&mut model, true, true, |_| {})?;
        } else {
            p.consolidate(&mut model)?;
        }
        let consolidation_seconds = t.elapsed().as_secs_f64();
        let learner = snapshot(
            &model,
            updates,
            ci as u64 + 1,
            out,
            &format!("learner-{:03}", ci + 1),
        )?;
        let learner_scores = guard.diagnostic_measure(&model)?;
        let t = Instant::now();
        let focus = guard.consider(learner.clone(), true)?.unwrap_or_default();
        let focus_count = focus.len();
        archive.focus_policy(focus);
        let publication_seconds = t.elapsed().as_secs_f64();
        let actor = guard.accepted();
        let t = Instant::now();
        if plan.repaired {
            model = actor.model.as_ref().clone();
            p.adopt_accepted(&model)?;
        }
        let anchor_adoption_seconds = t.elapsed().as_secs_f64();
        let committed = snapshot(
            &model,
            updates,
            ci as u64 + 1,
            out,
            &format!("committed-{:03}", ci + 1),
        )?;
        let actor_path = out.join(format!("actor-{:03}.json", ci + 1));
        publication::save_snapshot(&actor, &actor_path)?;
        let actor_changed = actor
            .model
            .parameters()
            .iter()
            .zip(old_actor.model.parameters())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        let games = if actor_changed {
            frozen_evaluation(
                &actor,
                &initial,
                &evaluation,
                &o,
                &executor,
                end,
                &out.join(format!("evaluation-{:03}", ci + 1)),
            )?
        } else {
            previous_evaluation.clone()
        };
        previous_evaluation = games.clone();
        snapshots.push(serde_json::json!({"label":format!("actor_{:03}",ci+1),"path":actor_path,"sha256":sha256(&fs::read(&actor_path)?),"coupled":true}));
        let row = serde_json::json!({"cycle":ci+1,"old_actor":old_actor.identity,"learner":learner.identity,"actor":actor.identity,"actor_weights_changed":actor_changed,
            "scores_before":before_scores,"scores_unconsolidated":unprotected_scores,"scores_learner":learner_scores,"scores_actor":guard.diagnostic_measure(&actor.model)?,
            "learner_before_consolidation":before_consolidation.path,"committed_working_model":committed.path,"fresh":fresh_count,"durable_recall":recall_count,"learned":learned_count,"focus_rows":focus_count,
            "receipts":receipts,"protection":p.progress(),"publication":guard.progress(),"recall":archive.progress(),"quotas":quotas,
            "replay":native_replay.as_ref().map(|m|m.progress()),"correction_replay_learned":correction_replay_count,
            "timing":{"collection":search_seconds,"archive":archive_seconds,"recall":recall_seconds,"learning":learning_seconds,"consolidation":consolidation_seconds,"publication":publication_seconds,"anchor_adoption":anchor_adoption_seconds,"whole_cycle":cycle_start.elapsed().as_secs_f64()},
            "frozen_games":games,"frozen_games_reused_unchanged_actor":!actor_changed});
        fs::write(
            cycle_dir.join("report.json"),
            serde_json::to_vec_pretty(&row)?,
        )?;
        cycles.push(row);
        println!(
            "{}",
            serde_json::json!({"cycle":ci+1,"actor_changed":actor_changed,"updates":updates,"fresh":fresh_count,"recall":recall_count,"elapsed":active.elapsed().as_secs_f64()})
        );
    }
    for input in &all_inputs {
        input.bytes()?;
    }
    if sha256(&fs::read(config)?) != sha256(&config_bytes)
        || sha256(&fs::read(plan_path)?) != sha256(&plan_bytes)
        || sha256(&fs::read(guard_path)?) != guard_hash
        || sha256(&fs::read(validation_path)?) != validation_hash
    {
        return Err(invalid(
            "diagnostic configuration or corrective panel changed during probe",
        ));
    }
    let report = serde_json::json!({"schema":"paisho-gen5-loop-cycle-result-v1","diagnostic_only":true,"repaired_publication":plan.repaired,"policy_confidence_constraint":plan.policy_confidence_constraint.unwrap_or(true),
        "publication_transfer":plan.publication_transfer,"initial_transfer_proofs":plan.initial_transfer_proofs,"initial_publication":initial_publication,"publication":guard.progress(),
        "immediate_lesson_admission":plan.immediate_lesson_admission,"diagnostic_two_choice_consolidation":plan.diagnostic_two_choice_consolidation,"recall_warmup_draws":if plan.immediate_lesson_admission {0} else {64},
        "receipt_protocol":if plan.immediate_lesson_admission {"all-fresh-native-memory-admission-no-warmup-v1"} else {"legacy-sampled-uniform-warmup-v1"},
        "replay_sampling":{"implementation":if plan.immediate_lesson_admission {"native Memory"} else {"uniform bounded VecDeque"},"production_correction_fraction_configured":o.correction_replay_fraction,"production_priority_correction_sampling_reproduced":plan.immediate_lesson_admission},
        "initial_model":plan.initial_model,"initial_learner":plan.initial_learner,"initial_scores":initial_scores,"initial_learner_scores":initial_learner_scores,"initial_frozen_games":initial_games,"snapshots":snapshots,"cycles":cycles,
        "inputs":all_inputs,"loading_seconds_excluded":loading_seconds,"active_seconds":active.elapsed().as_secs_f64(),
        "fifo_capacity":plan.fifo_capacity,"initial_fifo_examples":examples.len(),"evaluation_root_collisions_in_fifo_corrective_or_new_fresh":0,"recall_source_bundles":plan.recall_bundles.len(),"recall_source_proofs":plan.recall_proofs.len(),
        "native_paths":["collector::play_from","collector::targets","SavedMicroExample::example_for_rules","durable::persist_mode","Archive::rehearse_cached","Quotas::allocate","Protection::train_shared",if plan.publication_transfer {"Protection::consolidate_for_publication"} else {"Protection::consolidate"},"Guard::consider"],
        "transfer_native_paths":if plan.publication_transfer {vec!["Guard::enable_transfer","Guard::observe_consumed","Protection::publication_fresh_contract","Guard::set_publication_fresh"]} else {vec![]},
        "additional_native_paths":if plan.immediate_lesson_admission {vec!["Archive::admit (Balanced, Winning, pool)","Memory::add","Memory::draw","Memory::draw_correction"]} else {vec![]},
        "limitations":["serial laboratory receipt order","bounded initial FIFO and recall archive, not the complete production RAM state","legacy mode alone uses uniform FIFO without production correction-priority sampling","publication once per cycle, not the production timed cadence; immediate admission precedes that cycle publication","selfplay plus requested reanalyses, not production five-opponent mixture","explicit reanalysis prefixes, not the production Request scheduling frequency","no human quota in diagnostic minibatches","known corrective panels are not holdout validation","evaluation source separation is only with the new training prefixes; historical contamination requires separate audit","not a proof of monotonic global strength","active time includes diagnostics and frozen evaluation, not a production throughput estimate"]});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
