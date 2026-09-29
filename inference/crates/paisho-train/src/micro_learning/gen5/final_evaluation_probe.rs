//! One frozen, paired evaluation of the eight pre-reserved sources.
//! This is a child of loop_cycle_probe so it reuses its exact game reader/player.
//! No archive, replay, teaching, optimizer, consolidation or publication is run.
use super::*;

const RESERVATION_SHA256: &str =
    "83842c1f0a73b90ad55e1698f38b40b1ce623f5a04aff27ffaa9d069751ff9d4";
const INITIAL_ACTOR_SHA256: &str =
    "08568cc5cb986b24f4b0615556b12cdd5c865bd7d846850ace6654d0325ac424";
const DECISION_LIMIT: usize = 256;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FinalPlan {
    schema: String,
    reservation: Input,
    config: Input,
    candidate: Input,
    initial_actor: Input,
    /// Explicit gate. A template with false cannot open any reserved PSR.
    candidate_and_protocol_frozen: bool,
    freeze_note: String,
    decision_limit: usize,
    max_seconds: u64,
}

fn input_snapshot(input: &Input, base: Option<&MicroModel>) -> Result<Arc<Snapshot>> {
    let artifact: MicroArtifact = serde_json::from_slice(&input.bytes()?)?;
    let identity = artifact.identity();
    let version = artifact.provenance["version"].as_u64().unwrap_or(0);
    if let Some(base) = base {
        return publication::load_snapshot(&input.path, &identity, version, base);
    }
    let model = Arc::new(artifact.model()?);
    Ok(Arc::new(Snapshot {
        artifact: Some(Arc::new(artifact)),
        model,
        identity,
        version,
        path: input.path.clone(),
    }))
}

/// The caller must freeze and authorize this plan before invoking it. Parsing a
/// plan or compiling this helper performs no reserved-position model evaluation.
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("final evaluation output must not exist"));
    }
    let plan_bytes = fs::read(plan_path)?;
    let plan: FinalPlan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "paisho-gen5-final-reserve-evaluation-v1"
        || !plan.candidate_and_protocol_frozen
        || plan.freeze_note.trim().is_empty()
        || plan.decision_limit != DECISION_LIMIT
        || !(120..=3600).contains(&plan.max_seconds)
        || plan.reservation.sha256 != RESERVATION_SHA256
        || plan.initial_actor.sha256 != INITIAL_ACTOR_SHA256
    {
        return Err(invalid("final evaluation requires the frozen plan and exact reserved inputs"));
    }
    let reservation_bytes = plan.reservation.bytes()?;
    let reservation: serde_json::Value = serde_json::from_slice(&reservation_bytes)?;
    if reservation["schema"] != "paisho-gen5-loop-final-reserve-v1"
        || reservation["status"] != "standby-unqueried"
        || reservation["allow_learning"] != false
        || reservation["allow_corrective_feedback"] != false
        || reservation["source_count"] != 8
        || reservation["paired_games"] != 16
        || reservation["native_evaluations_performed"] != 0
        || reservation["models_read_on_reserved_positions"] != 0
    {
        return Err(invalid("reserved evaluation metadata changed"));
    }
    let specifications: Vec<Prefix> =
        serde_json::from_value(reservation["evaluation"].clone())?;
    let groups = specifications.iter().map(|p| &p.source_group).collect::<BTreeSet<_>>();
    let hashes = specifications.iter().map(|p| &p.input.sha256).collect::<BTreeSet<_>>();
    if specifications.len() != 8 || groups.len() != 8 || hashes.len() != 8
        || specifications.iter().any(|p| p.reanalysis || p.source_group.is_empty())
    {
        return Err(invalid("all eight distinct reserved prefixes are required"));
    }
    let dataset: Input = serde_json::from_value(reservation["dataset"].clone())?;
    let config_bytes = plan.config.bytes()?;
    let mut options: Options = serde_json::from_slice(&config_bytes)?;
    // Validate the frozen training recipe before changing only evaluation fields.
    options.validate()?;
    if options.mode != "puct" || options.budgets != vec![(256, 0.5), (512, 0.5)]
        || options.value_policy_strength != 16. || !options.proof_search
        || !options.learning_loop_v2 || !options.neural_memory
    {
        return Err(invalid("final evaluation must preserve the frozen search/budget/coupling recipe"));
    }
    dataset.bytes()?;
    // These are immutable models. Load the large sequence bank once and share it.
    // No inference has been made and no reserved PSR has been opened so far.
    let initial = input_snapshot(&plan.initial_actor, None)?;
    let candidate = input_snapshot(&plan.candidate, Some(&initial.model))?;
    if candidate.model.parameters().len() != initial.model.parameters().len()
        || candidate.model.schema() != initial.model.schema()
        || candidate.model.feature_schema() != initial.model.feature_schema()
        || !candidate.model.has_neural_memory()
    {
        return Err(invalid("final models must have the same neural-memory architecture"));
    }
    let original_parameters = [initial.model.parameters().to_vec(), candidate.model.parameters().to_vec()];
    let inputs = std::iter::once(plan.reservation.clone())
        .chain(std::iter::once(plan.config.clone()))
        .chain(std::iter::once(plan.initial_actor.clone()))
        .chain(std::iter::once(plan.candidate.clone()))
        .chain(std::iter::once(dataset))
        .chain(specifications.iter().map(|p| p.input.clone()))
        .collect::<Vec<_>>();
    fs::create_dir(out)?;
    fs::write(out.join("frozen-plan.json"), &plan_bytes)?;
    fs::write(out.join("frozen-reservation.json"), &reservation_bytes)?;
    let prefixes = specifications.into_iter()
        .map(|spec| spec.load().map(|record| (spec, record)))
        .collect::<Result<Vec<_>>>()?;
    let (pool, _) = cpu::build_pool(options.threads, None)?;
    let executor = cpu::Executor::direct(pool);
    options.learn = false;
    options.measurement = true;
    options.decision_limit = DECISION_LIMIT;
    options.stop_signal = None;
    options.end_unix_seconds = None;
    options.observed_origin = None;
    options.match_reference = None;
    options.reanalysis_cache = None;
    options.opponents.clear();
    options.candidate_seat = None;
    // frozen_evaluation uses exactly the same seed/budget sequence for each seat,
    // search(training=false), selected_index, and no visit-temperature sampling.
    // play_from may construct transient Sample objects; they are dropped, never
    // converted to targets or written to an archive. No learning APIs are called.
    let started = Instant::now();
    let end = started + Duration::from_secs(plan.max_seconds);
    let games = frozen_evaluation(
        &candidate, &initial, &prefixes, &options, &executor, end, &out.join("games"),
    )?;
    let rows = games["rows"].as_array().ok_or_else(|| invalid("missing final game rows"))?;
    if rows.len() != 16 { return Err(invalid("incomplete final evaluation")); }
    let mut pairs = vec![];
    let mut unresolved = 0;
    for (index, (spec, _)) in prefixes.iter().enumerate() {
        let mut outcomes = [0usize; 4];
        let mut values = vec![];
        for (offset, seat) in ["Host", "Guest"].iter().enumerate() {
            let row = &rows[index * 2 + offset];
            let result = row["result_index_wdlu"].as_u64()
                .ok_or_else(|| invalid("missing final game outcome"))? as usize;
            if row["id"] != index * 2 + offset || row["candidate_seat"] != *seat
                || row["source_group"] != spec.source_group || result > 3
                || row["decisions"].as_u64().map_or(true, |n| n > DECISION_LIMIT as u64)
            {
                return Err(invalid("invalid final source/seat binding"));
            }
            outcomes[result] += 1;
            values.push(row.clone());
        }
        unresolved += outcomes[3];
        pairs.push(serde_json::json!({
            "index":index,"source_group":spec.source_group,"source":spec.input,
            "prefix_decisions":spec.decisions,"wdlu":outcomes,"games":values,
            "paired_score_if_both_terminal":if outcomes[3] == 0 {
                Some((outcomes[0] as f64 + 0.5 * outcomes[1] as f64) / 2.)
            } else {None},
        }));
    }
    for (snapshot, before) in [&initial, &candidate].into_iter().zip(&original_parameters) {
        if !snapshot.model.parameters().iter().zip(before).all(|(a,b)| a.to_bits() == b.to_bits()) {
            return Err(invalid("frozen evaluation altered model parameters"));
        }
    }
    for input in &inputs { input.bytes()?; }
    if fs::read(plan_path)? != plan_bytes { return Err(invalid("frozen plan changed during evaluation")); }
    let report = serde_json::json!({
        "schema":"paisho-gen5-final-reserve-evaluation-result-v1",
        "plan":{"path":plan_path,"sha256":sha256(&plan_bytes)},
        "freeze_note":plan.freeze_note,"input_hashes_verified_before_and_after":true,
        "inputs":inputs,"candidate_identity":candidate.identity,"initial_actor_identity":initial.identity,
        "parameters_unchanged_by_bits":true,"games":games,"paired_source_blocks":pairs,
        "source_blocks":8,"paired_games":16,"unresolved_games":unresolved,
        "decision_limit_new_decisions":DECISION_LIMIT,"max_active_seconds":plan.max_seconds,
        "search":{"mode":options.mode,"budgets":options.budgets,"value_policy_strength":options.value_policy_strength,
            "measurement":true,"training_exploration":false,"visit_temperature_sampling":false,
            "proof_search":options.proof_search,"same_recipe_for_both_models":true},
        "learned_examples":0,"archive_writes":0,"corrective_feedback":0,
        "no_production_mutation":true,"unresolved_is_not_draw":true,
        "limits":["Eight source blocks, not sixteen independent observations.",
            "All eight reserved sources are included; no outcome-based filtering or retries.",
            "Repetition and 256-decision limits remain unresolved unless the rules report a terminal outcome.",
            "A wall-censored run fails instead of being reported as a complete comparison.",
            "Sources were excluded from current corrective sets; absence from the historical bank/model lineage is not established.",
            "No Elo, long-term learning-loop, no-forgetting or production-throughput guarantee."]
    });
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
