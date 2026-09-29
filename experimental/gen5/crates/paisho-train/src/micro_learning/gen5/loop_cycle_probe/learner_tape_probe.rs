//! Diagnostic only. Native reset records a fixed real-data tape once; a private
//! persistent learner and an exact reset replay consume it without resampling.
//! No collector/search is run. This is not reconstruction of C1's original RNG.
use super::*;
#[path = "learner_tape_probe/tape_search_measure.rs"]
mod tape_search_measure;
pub use tape_search_measure::run as measure_search_transfer;
use std::{collections::BTreeMap, io::Write};
#[path = "learner_tape_probe/step_fractions.rs"]
mod step_fractions;
pub use step_fractions::run as run_step_fractions;
#[path = "learner_tape_probe/base_policy_measure.rs"]
mod base_policy_measure;
pub use base_policy_measure::run as measure_base_policy;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TapePlan {
    schema: String,
    config: Input,
    resume: Input,
    initial_actor: Input,
    collectors: Vec<Input>,
    human_dataset: Input,
    initial_examples: Input,
    recall_bundles: Vec<Input>,
    recall_proofs: Vec<Input>,
    blocks: Vec<Block>,
    seed: u64,
    fifo_capacity: usize,
    max_seconds: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Block {
    capture: Input,
    retention: Input,
    expected_updates: usize,
    receipts: Vec<Receipt>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    receipt: Input,
    targets: Input,
    psr: Input,
    bundle: Input,
}
#[derive(Clone, Serialize, Deserialize)]
struct Item {
    example: resume_example::ResumeExample,
    kind: u8,
    lane: Lane,
}
#[derive(Serialize, Deserialize)]
struct Frame {
    schema: String,
    block: usize,
    ordinal: usize,
    source_receipt: Input,
    fresh: Vec<resume_example::ResumeExample>,
    items: Vec<Item>,
    rates: Vec<f64>,
    expected_after_sgd: Vec<String>,
    rng_after_sampling: u64,
}
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Reset,
    Persistent,
}
// These exact native inputs survive all three arms. Serialization is evidence,
// not a reason to reload data that remains available in this process.
struct ResidentFrame {
    source: Input,
    block: usize,
    ordinal: usize,
    fresh: Vec<Arc<MicroExample>>,
    items: Vec<(Arc<MicroExample>, u8, Lane)>,
    rates: Vec<f64>,
    expected_after_sgd: Vec<String>,
}
struct ResidentExamples {
    by_content: BTreeMap<String, Arc<MicroExample>>,
    bytes: usize,
}
impl ResidentExamples {
    fn intern(&mut self, ex: Arc<MicroExample>) -> Result<Arc<MicroExample>> {
        let encoded = encoded(&ex);
        let bytes = serde_json::to_vec(&encoded)?;
        let key = sha256(&bytes);
        if let Some(existing) = self.by_content.get(&key) {
            return Ok(existing.clone());
        }
        // Native recovery conversion must preserve every serialized optimizer bit.
        let recovered: resume_example::ResumeExample = serde_json::from_slice(&bytes)?;
        let _: Arc<MicroExample> = restore(&recovered)?;
        if serde_json::to_vec(&recovered)? != bytes {
            return Err(invalid("tape JSON changed an optimizer input"));
        }
        self.bytes += std::mem::size_of::<MicroExample>()
            + 8 * (ex.state.len() + ex.policy.len())
            + std::mem::size_of::<[f64; 32]>() * ex.actions.len()
            + std::mem::size_of::<Option<f64>>() * ex.action_values.len();
        if self.bytes > 4 * 1024 * 1024 * 1024usize {
            return Err(invalid("native tape exceeds fixed 4 GiB input bound"));
        }
        self.by_content.insert(key, ex.clone());
        Ok(ex)
    }
}
fn select_gradient_batch(selected: &mut bool, len: usize) -> bool {
    if !*selected && len == 64 { *selected = true; true } else { false }
}
fn next_learner(mode: Mode, before_control: &MicroModel, accepted: &MicroModel) -> MicroModel {
    match mode {
        Mode::Reset => accepted.clone(),
        Mode::Persistent => before_control.clone(),
    }
}
fn bits(model: &MicroModel) -> String {
    let bytes = model
        .parameters()
        .iter()
        .flat_map(|x| x.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    sha256(&bytes)
}
fn input(value: &serde_json::Value) -> Result<Input> {
    Ok(Input {
        path: PathBuf::from(
            value["path"]
                .as_str()
                .ok_or_else(|| invalid("missing input path"))?,
        ),
        sha256: value["sha256"]
            .as_str()
            .ok_or_else(|| invalid("missing input hash"))?
            .into(),
    })
}
fn json(source: &Input) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice(&source.bytes()?)?)
}
fn write_json(path: &Path, value: &impl Serialize) -> Result<Input> {
    let bytes = serde_json::to_vec(value)?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(Input {
        path: path.into(),
        sha256: sha256(&bytes),
    })
}
fn write_frame(path: &Path, frame: &Frame) -> Result<Input> {
    let file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    let mut zipped = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
    serde_json::to_writer(&mut zipped, frame)?;
    zipped.finish()?.sync_all()?;
    Ok(Input {
        path: path.into(),
        sha256: sha256(&fs::read(path)?),
    })
}
#[cfg(test)]
fn read_frame(source: &Input) -> Result<Frame> {
    let bytes = source.bytes()?;
    Ok(serde_json::from_reader(flate2::read::GzDecoder::new(
        bytes.as_slice(),
    ))?)
}
fn encoded(ex: &MicroExample) -> resume_example::ResumeExample {
    resume_example::ResumeExample::from_with_trusted_q(ex, true)
}
fn restore(ex: &resume_example::ResumeExample) -> Result<Arc<MicroExample>> {
    let result = ex.clone().example_with_trusted_q(true)?;
    if serde_json::to_vec(ex)? != serde_json::to_vec(&encoded(&result))? {
        return Err(invalid(
            "tape trusted-Q/support roundtrip changed optimizer input",
        ));
    }
    Ok(result)
}
fn snapshot_owned(
    model: &MicroModel,
    updates: u64,
    version: u64,
    out: &Path,
    role: &str,
) -> Result<Arc<Snapshot>> {
    let artifact = Arc::new(MicroArtifact::new(
        model,
        updates,
        serde_json::json!({
        "diagnostic_only":true,"kind":"gen5-learner-tape","role":role,"version":version}),
    ));
    let path = out.join(format!("{role}-{version:03}.json"));
    // Serialize this already loaded native model; never reload the external bank.
    write_json(&path, artifact.as_ref())?;
    Ok(Arc::new(Snapshot {
        identity: artifact.identity(),
        artifact: Some(artifact),
        version,
        path,
        model: Arc::new(model.clone()),
    }))
}
fn load_model(source: &Input, bank: Option<&MicroModel>) -> Result<(MicroArtifact, MicroModel)> {
    let artifact: MicroArtifact = serde_json::from_slice(&source.bytes()?)?;
    let model = if let Some(base) = bank {
        if serde_json::to_value(&artifact.sequence_memory)?
            != serde_json::to_value(base.sequence_memory().map(|b| &b.spec))?
        {
            return Err(invalid("tape models use different immutable banks"));
        }
        let mut model =
            MicroModel::from_parameters(artifact.parameters.clone()).map_err(invalid)?;
        if let Some(bank) = base.sequence_memory() {
            model = model.with_sequence_memory_owned(bank.clone());
        }
        model
    } else {
        artifact.model()?
    };
    if model.schema() != artifact.schema || model.feature_schema() != artifact.feature_schema {
        return Err(invalid("tape model schema mismatch"));
    }
    Ok((artifact, model))
}

// collector.rs intentionally omits all action tensors for pure value rows.
// Never infer an omitted policy from the legal actions of that position.
fn empty_value_target(row: &SavedMicroExample) -> bool {
    row.policy_weight == 0.
        && row.actions.is_empty()
        && row.action_features.is_empty()
        && row.policy.is_empty()
        && row.new_visits.is_empty()
}

// Existing observations only. Empty `samples` is deliberate: targets are loaded
// exactly from Saved and this adapter is NEVER sent to collector::targets/play.
fn recorded_game(
    source: &Receipt,
    models: &BTreeMap<String, Arc<Snapshot>>,
) -> Result<(collector::Played, Vec<SavedMicroExample>)> {
    let r = json(&source.receipt)?;
    if r["fully_learned"] != true || !r["error"].is_null() || r["campaign_censored"] == true {
        return Err(invalid("tape requires complete uncensored real receipts"));
    }
    let bytes = source.psr.bytes()?;
    let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
    if record.rules() != RULES
        || record.to_string().as_bytes() != bytes
        || r["psr_sha256"] != source.psr.sha256
    {
        return Err(invalid("recorded PSR identity/rules mismatch"));
    }
    let outcome = record.replay()?.outcome();
    if r["outcome"] != format!("{outcome:?}") {
        return Err(invalid("recorded outcome mismatch"));
    }
    let saved = decode_examples(&source.targets.bytes()?)?;
    if saved.is_empty()
        || r["eligible_examples"].as_u64() != Some(saved.len() as u64)
        || r["fresh_used"].as_u64() != Some(saved.len() as u64)
        || r["targets_sha256"] != source.targets.sha256
    {
        return Err(invalid("recorded dense fresh targets mismatch"));
    }
    let collector = r["collector"]
        .as_str()
        .ok_or_else(|| invalid("recorded collector missing"))?;
    let original = models
        .get(collector)
        .ok_or_else(|| invalid("unbound recorded collector model"))?;
    let snapshot = Arc::new(Snapshot {
        version: r["collector_version"]
            .as_u64()
            .ok_or_else(|| invalid("collector version"))?,
        identity: collector.into(),
        artifact: original.artifact.clone(),
        model: original.model.clone(),
        path: original.path.clone(),
    });
    let bytes = source.bundle.bytes()?;
    let bundle: serde_json::Value =
        serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
    if bundle["schema"] != "paisho-gen5-durable-lessons-v1"
        || bundle["rules"] != RULES.as_str()
        || bundle["game_id"] != r["id"]
        || bundle["psr_sha256"] != source.psr.sha256
        || bundle["psr"] != record.to_string()
        || bundle["source"] != collector
        || bundle["lessons"]
            != serde_json::to_value(
                saved
                    .iter()
                    .map(durable::Lesson::from_saved)
                    .collect::<Vec<_>>(),
            )?
    {
        return Err(invalid(
            "compact receipt does not encode these exact real Saved targets",
        ));
    }
    // Reconstruct rules/features, never predictions or targets. Reanalysis may
    // supervise the next root at len(record)+1 without an appended action.
    let mut states = vec![record.initial_position()];
    for action in record.actions() {
        let mut next = states.last().unwrap().clone();
        next.apply(*action)?;
        states.push(next);
    }
    for row in &saved {
        let position = states
            .get(
                row.decision
                    .checked_sub(1)
                    .ok_or_else(|| invalid("zero Saved decision"))?,
            )
            .ok_or_else(|| invalid("Saved decision outside bound PSR"))?;
        let legal = paisho_core::legal_actions(position);
        let expected_state = original.model.state_features(position);
        let equal = |a: &[f64], b: &[f64]| {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
        };
        let native_actions = legal.iter().map(ToString::to_string).collect::<Vec<_>>();
        let empty_value_only = empty_value_target(row);
        let id_ok = row.game_id
            == r["id"]
                .as_u64()
                .ok_or_else(|| invalid("receipt id"))?
                .to_string();
        let collector_ok = row.collector == collector;
        let state_ok = equal(&row.state, &expected_state);
        let order_ok = empty_value_only || row.actions == native_actions;
        let feature_count_ok = empty_value_only || row.action_features.len() == legal.len();
        let features_ok = empty_value_only
            || row
                .action_features
                .iter()
                .zip(&legal)
                .all(|(f, a)| equal(f, &micro_action_features(position, *a)));
        if !id_ok || !collector_ok || !state_ok || !order_ok || !feature_count_ok || !features_ok {
            let state_difference = (0..row.state.len().max(expected_state.len())).find(|&i| {
                row.state.get(i).map(|v| v.to_bits()) != expected_state.get(i).map(|v| v.to_bits())
            });
            let order_difference = (0..row.actions.len().max(native_actions.len()))
                .find(|&i| row.actions.get(i) != native_actions.get(i));
            let mut feature_difference = None;
            for (i, (recorded, action)) in row.action_features.iter().zip(&legal).enumerate() {
                let expected = micro_action_features(position, *action);
                if let Some(j) = (0..recorded.len().max(expected.len())).find(|&j| {
                    recorded.get(j).map(|v| v.to_bits()) != expected.get(j).map(|v| v.to_bits())
                }) {
                    feature_difference = Some(serde_json::json!({"action_index":i,"coordinate":j,
                        "recorded":recorded.get(j),"native":expected.get(j),
                        "recorded_bits":recorded.get(j).map(|v|format!("{:016x}",v.to_bits())),
                        "native_bits":expected.get(j).map(|v|format!("{:016x}",v.to_bits()))}));
                    break;
                }
            }
            return Err(invalid(format!(
                "real Saved source/state/legal-action features mismatch: {}",
                serde_json::json!({
                "receipt":source.receipt.path,"receipt_id":r["id"],"saved_game_id":row.game_id,
                "decision":row.decision,"policy_weight":row.policy_weight,"empty_value_only":empty_value_only,
                "id_ok":id_ok,"collector_ok":collector_ok,"state_ok":state_ok,"order_ok":order_ok,
                "feature_count_ok":feature_count_ok,"features_ok":features_ok,
                "recorded_action_count":row.actions.len(),"native_action_count":legal.len(),
                "state_difference":state_difference.map(|i|serde_json::json!({"coordinate":i,
                    "recorded":row.state.get(i),"native":expected_state.get(i),
                    "recorded_bits":row.state.get(i).map(|v|format!("{:016x}",v.to_bits())),
                    "native_bits":expected_state.get(i).map(|v|format!("{:016x}",v.to_bits()))})),
                "order_difference":order_difference.map(|i|serde_json::json!({"index":i,
                    "recorded":row.actions.get(i),"native":native_actions.get(i)})),
                "feature_difference":feature_difference})
            )));
        }
    }
    let certificates: Vec<(usize, MicroProofCertificate)> =
        serde_json::from_value(bundle["proofs"].clone())?;
    for (decision, certificate) in &certificates {
        if *decision == 0 || *decision > record.actions().len() + 1 {
            return Err(invalid("certificate decision bounds"));
        }
        certificate
            .verify(&cases::prefix(&record, decision - 1).replay()?)
            .map_err(invalid)?;
    }
    let num = |k: &str| r[k].as_u64().unwrap_or(0) as usize;
    let scalar = |k: &str| r[k].as_f64().unwrap_or(0.);
    let game = collector::Played {
        reused_search: r["reused_search"] == true,
        search_cache_key: r["search_cache_key"].as_str().map(str::to_owned),
        search_evidence_simulations: num("search_evidence_simulations"),
        case: serde_json::from_value(r["case"].clone())?,
        ack: None,
        certificates,
        reanalysis: r["reanalysis"] == true,
        loop_repair: true,
        measurement: false,
        evaluation: None,
        observed_origin: None,
        prefix_decisions: num("prefix_decisions"),
        case_mode: !r["case"].is_null(),
        id: num("id"),
        lane: serde_json::from_value(r["lane"].clone())?,
        snapshot,
        opponent: r["opponent"].as_str().unwrap_or("").into(),
        reference_budget: r["reference_budget"].as_u64().map(|v| v as usize),
        candidate_seat: match r["candidate_seat"].as_str() {
            Some("Host") => Player::Host,
            Some("Guest") => Player::Guest,
            _ => return Err(invalid("recorded seat")),
        },
        record,
        outcome,
        termination: r["termination"].as_str().unwrap_or("").into(),
        error: None,
        samples: vec![],
        cycle: None,
        seconds: scalar("seconds"),
        cap_seconds: r["cap_seconds"].as_f64(),
        campaign_censored: false,
        search_seconds: serde_json::from_value(r["search_seconds"].clone())?,
        pool_wait_seconds: scalar("pool_wait_seconds"),
        maintenance_seconds: scalar("maintenance_seconds"),
        sample_seconds: scalar("sample_seconds"),
        simulations: num("simulations"),
        tactical_evaluations: num("tactical_evaluations"),
        inherited: num("inherited_visits"),
        hits: num("inference_cache_hits"),
        evals: num("inference_evaluations"),
        policy_searches: num("policy_searches"),
        policy_coverage_sum: scalar("policy_coverage_sum"),
        forced_playouts: num("forced_playouts"),
        pruned_policy_visits: num("pruned_policy_visits"),
    };
    Ok((game, saved))
}

struct Arm {
    model: MicroModel,
    p: protection::Protection,
    guard: publication::Guard,
    updates: u64,
    version: u64,
    path: PathBuf,
}
impl Arm {
    fn new(
        initial: &Arc<Snapshot>,
        o: &Options,
        resume: &serde_json::Value,
        pool: &Arc<rayon::ThreadPool>,
        path: PathBuf,
    ) -> Result<Self> {
        fs::create_dir(&path)?;
        let mut guard = publication::Guard::open(
            o.publication_guard.as_ref().unwrap(),
            &path.join("publication"),
            o.value_policy_strength,
            initial.clone(),
            &resume["publication_guard"],
        )?;
        guard.enable_v2(o.publication_validation.as_ref().unwrap())?;
        guard.enable_v3()?;
        guard.enable_parallel(&[pool.clone()]);
        if bits(&guard.accepted().model) != bits(&initial.model) {
            return Err(invalid("tape initial accepted actor mismatch"));
        }
        let mut p = protection::Protection::new(&initial.model, guard.reference_examples())?;
        p.enable_loop_v3();
        p.enable_parallel(&[pool.clone()]);
        p.enable_validation_value(guard.diagnostic_validation_examples()?)?;
        p.restore(&resume["protection"])?;
        p.adopt_accepted(&initial.model)?;
        Ok(Self {
            model: initial.model.as_ref().clone(),
            p,
            guard,
            updates: resume["updates"]
                .as_u64()
                .ok_or_else(|| invalid("resume updates"))?,
            version: initial.version,
            path,
        })
    }
    fn boundary(
        &mut self,
        block: usize,
        mode: Mode,
        mut archive: Option<&mut durable::Archive>,
    ) -> Result<(serde_json::Value, Vec<Arc<MicroExample>>, [MicroModel; 4])> {
        let t = Instant::now();
        let before_control = self.model.clone(); // MUST precede the first repair/rollback.
        let incoming = snapshot_owned(
            &before_control,
            self.updates,
            self.version,
            &self.path,
            "incoming",
        )?;
        let early_focus = self.guard.observe_candidate(&before_control)?;
        if let Some(d) = archive.as_deref_mut() {
            d.focus_policy(early_focus);
        }
        let mut candidate = before_control.clone();
        self.p
            .diagnostic_consolidate_two_choices(&mut candidate, true, true, |_| {})?;
        let consolidated = snapshot_owned(
            &candidate,
            self.updates,
            self.version,
            &self.path,
            "consolidated",
        )?;
        let focus = self
            .guard
            .consider(consolidated.clone(), true)?
            .ok_or_else(|| invalid("forced tape boundary not checked"))?;
        if let Some(d) = archive.as_deref_mut() {
            d.focus_policy(focus.clone());
        }
        let accepted = self.guard.accepted();
        self.p.adopt_accepted(&accepted.model)?; // Never anchor constraints on private weights.
        self.model = next_learner(mode, &before_control, &accepted.model);
        let working = snapshot_owned(
            &self.model,
            self.updates,
            self.version,
            &self.path,
            "working",
        )?;
        let actor = snapshot_owned(
            &accepted.model,
            self.updates,
            self.version,
            &self.path,
            "actor",
        )?;
        Ok((
            serde_json::json!({"block":block,"updates":self.updates,"seconds":t.elapsed().as_secs_f64(),
            "incoming":incoming.path,"consolidated":consolidated.path,"actor":actor.path,"working":working.path,
            "incoming_bits":bits(&before_control),"consolidated_bits":bits(&candidate),"actor_bits":bits(&accepted.model),
            "working_bits":bits(&self.model),"private_preserved_exact":mode==Mode::Persistent && self.model.shares_storage_with(&before_control),
            "protection":self.p.progress(),"publication":self.guard.progress()}),
            focus,
            [
                before_control,
                candidate,
                accepted.model.as_ref().clone(),
                self.model.clone(),
            ],
        ))
    }
}

fn validate_frame(frame: &Frame, block: usize, ordinal: usize, rate: f64) -> Result<()> {
    if frame.schema != "paisho-gen5-native-learning-tape-receipt-v1"
        || frame.block != block
        || frame.ordinal != ordinal
        || frame.fresh.is_empty()
        || frame.items.len() != frame.fresh.len() * 5
        || frame.items.iter().filter(|i| i.kind == 0).count() != frame.fresh.len()
        || frame.items.iter().any(|i| i.kind > 5)
        || frame.rates.len() != frame.items.len().div_ceil(64)
        || frame.expected_after_sgd.len() != frame.rates.len()
    {
        return Err(invalid("invalid native learning tape frame"));
    }
    for (batch, r) in frame.items.chunks(64).zip(&frame.rates) {
        if r.to_bits() != (rate * batch.len() as f64 / 64.).to_bits() {
            return Err(invalid("tape learning rate changed"));
        }
    }
    Ok(())
}
fn deterministic_state(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .filter(|(k, _)| {
                    !k.contains("seconds")
                        && !k.ends_with("_path")
                        && *k != "checkpoint"
                        && *k != "training_profile"
                })
                .map(|(k, v)| (k.clone(), deterministic_state(v)))
                .collect(),
        ),
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.iter().map(deterministic_state).collect())
        }
        _ => value.clone(),
    }
}
fn scalar_metrics(model: &MicroModel, example: &MicroExample) -> Result<serde_json::Value> {
    let loss = model.loss_loop_v3(example).map_err(invalid)?;
    let embedding = model.embed(&example.state);
    let delta = embedding.value - example.value;
    let direct = 0.5 * delta * delta * example.value_weight;
    let q = loss.value - direct;
    if !q.is_finite() || q < -1e-12 {
        return Err(invalid("native auxiliary loss decomposition invalid"));
    }
    let policy = loss.policy * example.policy_weight;
    let (mass, correct, played_mass, played_correct) =
        if example.policy_support && example.policy_weight > 0. && !example.actions.is_empty() {
            let p = micro_softmax(&MicroModel::logits(&embedding, &example.actions))
                .map_err(invalid)?;
            let p = model
                .memory_priors(
                    &example.state,
                    &example.actions,
                    &p,
                    example.sequence_source,
                )
                .map_err(invalid)?;
            let index = (0..p.len())
                .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            let played = model
                .memory_priors(
                    &example.state,
                    &example.actions,
                    &micro_softmax(&MicroModel::logits(&embedding, &example.actions))
                        .map_err(invalid)?,
                    0,
                )
                .map_err(invalid)?;
            let played_index = (0..played.len())
                .max_by(|&a, &b| played[a].total_cmp(&played[b]).then_with(|| b.cmp(&a)))
                .unwrap();
            (
                Some(
                    p.iter()
                        .zip(&example.policy)
                        .filter(|(_, t)| **t > 0.)
                        .map(|(p, _)| *p)
                        .sum::<f64>(),
                ),
                Some(example.policy[index] > 0.),
                Some(
                    played
                        .iter()
                        .zip(&example.policy)
                        .filter(|(_, t)| **t > 0.)
                        .map(|(p, _)| *p)
                        .sum::<f64>(),
                ),
                Some(example.policy[played_index] > 0.),
            )
        } else {
            (None, None, None, None)
        };
    Ok(
        serde_json::json!({"total":loss.total(example.policy_weight),"policy":policy,"direct_value":direct,
        "auxiliary_q":q,"support_mass":mass,"raw_in_support":correct,
        "played_raw_support_mass":played_mass,"played_raw_in_support":played_correct,
        "loss_context":"native Saved sequence_source exclusion","played_raw_context":"no exclusion; raw only, no coupled ranking or MCTS"}),
    )
}

pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    run_mode(plan_path, out, false)
}
/// Read-only native source/rules/feature validation, no optimizer/model forward.
pub fn preflight(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    run_mode(plan_path, out, true)
}
fn run_mode(plan_path: &Path, out: &Path, preflight: bool) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("tape diagnostic output must be new"));
    }
    let plan_bytes = fs::read(plan_path)?;
    let plan: TapePlan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "paisho-gen5-learner-tape-v1"
        || plan.blocks.len() != 4
        || plan.blocks.iter().map(|b| b.receipts.len()).sum::<usize>() != 966
        || plan
            .blocks
            .iter()
            .map(|b| b.expected_updates)
            .collect::<Vec<_>>()
            != [373, 438, 546, 474]
        || plan.fifo_capacity != 4096
        || !(120..=3600).contains(&plan.max_seconds)
    {
        return Err(invalid("fixed four-block C1 tape plan changed"));
    }
    let o: Options = serde_json::from_slice(&plan.config.bytes()?)?;
    if !o.learning_loop_v3
        || !o.structural_repair
        || o.replay_ratio != 4
        || o.recall_fraction != 0.5
        || o.historical_replay_fraction != 0.
        || o.correction_replay_fraction != 0.05
        || o.human_fraction != 0.02
        || o.human_dataset.as_ref() != Some(&plan.human_dataset.path)
    {
        return Err(invalid("tape sampler settings changed"));
    }
    plan.human_dataset.bytes()?;
    let resume = json(&plan.resume)?;
    let (a, base) = load_model(&plan.initial_actor, None)?;
    let initial = Arc::new(Snapshot {
        identity: a.identity(),
        version: resume["publication_guard"]["accepted_version"]
            .as_u64()
            .unwrap(),
        artifact: Some(Arc::new(a)),
        path: plan.initial_actor.path.clone(),
        model: Arc::new(base.clone()),
    });
    let mut collectors = BTreeMap::new();
    for source in &plan.collectors {
        let (a, m) = load_model(source, Some(&base))?;
        collectors.insert(
            a.identity(),
            Arc::new(Snapshot {
                identity: a.identity(),
                version: 0,
                artifact: Some(Arc::new(a)),
                path: source.path.clone(),
                model: Arc::new(m),
            }),
        );
    }
    if preflight {
        let t = Instant::now();
        let mut receipts = 0usize;
        let mut fresh = 0usize;
        let mut proofs = 0usize;
        let mut updates = 0usize;
        for (bi, block) in plan.blocks.iter().enumerate() {
            let capture = json(&block.capture)?;
            let manifest = json(&block.retention)?;
            let entries = manifest["target_inputs"]
                .as_array()
                .ok_or_else(|| invalid("preflight sources"))?;
            if entries.len() != block.receipts.len()
                || capture["context"]["check"].as_u64() != Some(181 + bi as u64)
            {
                return Err(invalid("preflight source boundary mismatch"));
            }
            let mut block_updates = 0;
            for (src, entry) in block.receipts.iter().zip(entries) {
                for (key, expected) in [
                    ("receipt", &src.receipt),
                    ("targets", &src.targets),
                    ("psr", &src.psr),
                ] {
                    let found = input(&entry[key])?;
                    if found.path != expected.path || found.sha256 != expected.sha256 {
                        return Err(invalid("preflight source order changed"));
                    }
                }
                let (game, saved) = recorded_game(src, &collectors)?;
                for row in &saved {
                    let _ = row.example_for_rules_with_trusted_q(RULES, true)?;
                }
                receipts += 1;
                fresh += saved.len();
                proofs += game.certificates.len();
                block_updates += (saved.len() * 5).div_ceil(64);
            }
            if block_updates != block.expected_updates {
                return Err(invalid("preflight update count mismatch"));
            }
            updates += block_updates;
        }
        if (receipts, fresh, updates) != (966, 13038, 1831) {
            return Err(invalid("preflight totals changed"));
        }
        fs::create_dir(out)?;
        let result = serde_json::json!({"schema":"paisho-gen5-learner-tape-preflight-v1", "plan_sha256":sha256(&plan_bytes),
            "receipts":receipts,"fresh":fresh,"updates":updates,"certificates":proofs,"seconds":t.elapsed().as_secs_f64(),
            "model_forward_calls":0,"optimizer_steps":0,"native_saved_conversion":true,"rules_features_certificates_verified":true});
        write_json(&out.join("report.json"), &result)?;
        return Ok(result);
    }
    fs::create_dir(out)?;
    write_json(
        &out.join("input-plan.json"),
        &serde_json::from_slice::<serde_json::Value>(&plan_bytes)?,
    )?;
    let (pool, _) = cpu::build_pool(o.threads, None)?;
    let human = memory::human(&plan.human_dataset.path, out, &pool, base.has_spatial())?;
    if human.is_empty() {
        return Err(invalid("native human replay empty"));
    }
    let mut recorder = Arm::new(&initial, &o, &resume, &pool, out.join("recorded-reset"))?;
    let archive_path = out.join("archive");
    let copied = out.join("initial-recall");
    fs::create_dir(&copied)?;
    fs::create_dir(copied.join("proofs"))?;
    for (proof, inputs) in [(false, &plan.recall_bundles), (true, &plan.recall_proofs)] {
        for src in inputs {
            let dest = if proof {
                copied.join("proofs")
            } else {
                copied.clone()
            }
            .join(src.path.file_name().unwrap());
            let mut f = fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(dest)?;
            f.write_all(&src.bytes()?)?;
        }
    }
    let mut archive = durable::Archive::open(&archive_path)?;
    archive.add_read_only(&copied)?;
    archive.policy_consolidation(true);
    archive.proof_recall = true;
    archive.trusted_action_values = true;
    archive.enable_parallel(&[pool.clone()]);
    archive.enable_coverage(out, &serde_json::Value::Null, &base, &archive_path)?;
    archive.seed_values(recorder.guard.reference_examples());
    archive.focus_policy(recorder.guard.correction_examples()?);
    let seed_rows: Vec<SavedMicroExample> =
        serde_json::from_slice(&plan.initial_examples.bytes()?)?;
    let mut fifo = replay::Replay::open(
        &o,
        &plan.initial_examples,
        seed_rows.len(),
        4096,
        true,
        base.has_spatial(),
        out,
    )?;
    let mut quotas: recall::Quotas = serde_json::from_value(resume["recall_quotas"].clone())?;
    quotas.validate()?;
    let mut rng = StableRng::new(plan.seed);
    let mut tape: Vec<Vec<Input>> = vec![];
    let mut resident_tape: Vec<Vec<ResidentFrame>> = vec![];
    let mut resident_examples = ResidentExamples {
        by_content: BTreeMap::new(),
        bytes: 0,
    };
    let mut resident_models: BTreeMap<String, Vec<[MicroModel; 4]>> = BTreeMap::new();
    let mut original = vec![];
    let mut original_models = vec![];
    let mut fresh64: VecDeque<Arc<MicroExample>> = VecDeque::new();
    let mut boundary_fresh = vec![];
    let mut proof_rows = vec![];
    let mut proof_keys = BTreeSet::new();
    let quota_start = quotas.clone();
    let started = Instant::now();
    let end = started + Duration::from_secs(plan.max_seconds);
    let mut tape_io_seconds = 0.;
    let mut evidence_hash_seconds = 0.;
    let mut sampling_seconds = 0.;
    let mut total_fresh = 0usize;
    fs::create_dir(out.join("tape"))?;
    let mut recording_sgd_seconds = 0.;
    let mut ordinal = 0;
    let mut admissions = vec![];
    let mut gradient_flow = vec![];
    let mut gradient_diagnostic_seconds = 0.;
    fs::create_dir(out.join("gradient-flow"))?;
    for (bi, block) in plan.blocks.iter().enumerate() {
        let mut gradient_selected = false;
        let capture = json(&block.capture)?;
        if capture["context"]["check"].as_u64() != Some(181 + bi as u64)
            || capture["context"]["version"].as_u64()
                != Some(initial.version + ordinal as u64 + block.receipts.len() as u64)
            || capture["context"]["updates_consumed"].as_u64()
                != Some(recorder.updates + block.expected_updates as u64)
        {
            return Err(invalid("C1 capture chronology/update boundaries changed"));
        }
        let manifest = json(&block.retention)?;
        if manifest["boundary_check"].as_u64() != Some(181 + bi as u64)
            || manifest["target_inputs"].as_array().map(Vec::len) != Some(block.receipts.len())
        {
            return Err(invalid("C1 retention boundary/source count changed"));
        }
        for (entry, src) in manifest["target_inputs"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&block.receipts)
        {
            for (key, expected) in [
                ("receipt", &src.receipt),
                ("targets", &src.targets),
                ("psr", &src.psr),
            ] {
                let found = input(&entry[key])?;
                if found.path != expected.path || found.sha256 != expected.sha256 {
                    return Err(invalid("C1 exact chronological source order changed"));
                }
            }
        }
        let before = recorder.updates;
        let mut frames = vec![];
        let mut resident_frames = vec![];
        let mut pending_admission = None;
        for (ri, source) in block.receipts.iter().enumerate() {
            if Instant::now() >= end {
                return Err(invalid("tape recording exceeded fixed bound"));
            }
            let sampling_started = Instant::now();
            let (game, saved) = recorded_game(source, &collectors)?;
            let owned = saved
                .iter()
                .map(|s| {
                    s.example_for_rules_with_trusted_q(RULES, true)
                        .map(Arc::new)
                })
                .collect::<Result<Vec<_>>>()?;
            let lessons = admission::prepare(&game, &saved, &owned)?;
            if let Some(path) =
                durable::persist_mode(&archive_path, &game, &saved, &archive.proofs, false)?
            {
                archive.add(path);
            }
            for (decision, certificate) in &game.certificates {
                let prefix = cases::prefix(&game.record, decision - 1);
                let key = sha256(prefix.to_string().as_bytes());
                archive.register_persisted_proof(
                    archive_path.join("proofs").join(format!("{key}.json")),
                );
                if proof_rows.len() < 64 && !proof_keys.contains(&key) {
                    if let Some((index, s)) = saved.iter().enumerate().find(|(_, s)| {
                        s.decision == *decision
                            && s.tactical.as_ref().is_some_and(|t| t.root_value == Some(1))
                    }) {
                        let position = prefix.replay()?;
                        let legal = paisho_core::legal_actions(&position);
                        let support =
                            action_values::verified_winning_policy(&position, &legal, certificate)?;
                        let example = &owned[index];
                        if !example.policy_support
                            || example.policy.len() != support.len()
                            || example
                                .policy
                                .iter()
                                .zip(&support)
                                .any(|(a, b)| (*a > 0.) != (*b > 0.))
                        {
                            return Err(invalid(
                                "new proof target lacks complete regulatory support",
                            ));
                        }
                        proof_keys.insert(key.clone());
                        proof_rows.push(serde_json::json!({"key":key,"first_block":bi,"receipt":source.receipt,"saved_index":index,
                            "decision":s.decision,"example":encoded(example)}));
                    }
                }
            }
            recorder.p.observe(&owned);
            let (due, mut human_due) = quotas.allocate(owned.len(), 4, 0.5, 0.02);
            let recalled = archive.rehearse_cached(due, &mut rng, &recorder.model)?;
            if recalled.len() != due {
                return Err(invalid("tape recall quota unavailable"));
            }
            let winning = archive.last_winning_draws;
            let mut mixed = owned
                .iter()
                .cloned()
                .map(|e| (e, 0, game.lane))
                .collect::<Vec<_>>();
            mixed.extend(
                recalled
                    .into_iter()
                    .enumerate()
                    .map(|(i, e)| (e, if i < winning { 5 } else { 4 }, Lane::Selfplay)),
            );
            for _ in 0..owned.len() * 4 - due {
                if human_due > 0 {
                    human_due -= 1;
                    mixed.push((human[rng.index(human.len())].clone(), 2, Lane::Selfplay));
                } else {
                    let (e, kind, lane) = fifo.draw_with_lane(&mut rng, &owned, game.lane)?;
                    mixed.push((e, kind, lane));
                }
            }
            shuffle(&mut mixed, &mut rng);
            sampling_seconds += sampling_started.elapsed().as_secs_f64();
            let io_started = Instant::now();
            let owned = owned
                .into_iter()
                .map(|e| resident_examples.intern(e))
                .collect::<Result<Vec<_>>>()?;
            let mixed = mixed
                .into_iter()
                .map(|(e, k, l)| Ok((resident_examples.intern(e)?, k, l)))
                .collect::<Result<Vec<_>>>()?;
            tape_io_seconds += io_started.elapsed().as_secs_f64();
            total_fresh += owned.len();
            for ex in &owned {
                fresh64.push_back(ex.clone());
                if fresh64.len() > 64 {
                    fresh64.pop_front();
                }
            }
            let mut frame = Frame {
                schema: "paisho-gen5-native-learning-tape-receipt-v1".into(),
                block: bi,
                ordinal,
                source_receipt: source.receipt.clone(),
                fresh: owned.iter().map(|e| encoded(e)).collect(),
                items: mixed
                    .iter()
                    .map(|(e, kind, lane)| Item {
                        example: encoded(e),
                        kind: *kind,
                        lane: *lane,
                    })
                    .collect(),
                rates: vec![],
                expected_after_sgd: vec![],
                rng_after_sampling: rng.state(),
            };
            for (batch_index, batch) in mixed.chunks(64).enumerate() {
                let rate = o.rate * batch.len() as f64 / 64.;
                let examples = batch.iter().map(|(e, _, _)| e.clone()).collect::<Vec<_>>();
                // Fixed before any SGD result: first COMPLETE minibatch per block.
                // Extra reads/backwards are outside the unchanged SGD timer.
                let probe = if select_gradient_batch(&mut gradient_selected, batch.len()) {
                    let t = Instant::now();
                    let kinds = batch.iter().map(|(_, k, _)| *k).collect::<Vec<_>>();
                    let pending = recorder.p.diagnostic_gradient_flow(&recorder.model, &examples, &kinds, rate, 1e-5)?;
                    gradient_diagnostic_seconds += t.elapsed().as_secs_f64();
                    Some(pending)
                } else { None };
                let t = Instant::now();
                recorder
                    .p
                    .train_shared(&mut recorder.model, &examples, rate, 1e-5)?;
                recording_sgd_seconds += t.elapsed().as_secs_f64();
                recorder.updates += 1;
                frame.rates.push(rate);
                let h = Instant::now();
                frame.expected_after_sgd.push(bits(&recorder.model));
                evidence_hash_seconds += h.elapsed().as_secs_f64();
                if let Some(pending) = probe {
                    let t = Instant::now();
                    let mut result = pending.finish(&recorder.model)?;
                    if result["actual_after_bits"].as_str() != frame.expected_after_sgd.last().map(String::as_str) {
                        return Err(invalid("gradient-flow result does not bind the native tape SGD"));
                    }
                    result["block"] = bi.into();
                    result["receipt_ordinal"] = ordinal.into();
                    result["receipt_index_in_block"] = ri.into();
                    result["batch_index"] = batch_index.into();
                    result["source_receipt"] = serde_json::to_value(&source.receipt)?;
                    result["lanes"] = serde_json::to_value(batch.iter().map(|(_,_,l)|l).collect::<Vec<_>>())?;
                    let descriptor = write_json(&out.join("gradient-flow").join(format!("block-{bi:03}.json")), &result)?;
                    gradient_flow.push(serde_json::json!({"block":bi,"available":true,"result":descriptor,
                        "additional_backward_calls":128,"actual_step_bits_exact":true}));
                    gradient_diagnostic_seconds += t.elapsed().as_secs_f64();
                }
                quotas.consumed_examples += batch.len();
                quotas.consumed_recall += batch.iter().filter(|(_, k, _)| *k >= 4).count();
            }
            validate_frame(&frame, bi, ordinal, o.rate)?;
            let t = Instant::now();
            let frame_source = write_frame(
                &out.join("tape")
                    .join(format!("receipt-{ordinal:04}.json.gz")),
                &frame,
            )?;
            tape_io_seconds += t.elapsed().as_secs_f64();
            frames.push(frame_source.clone());
            resident_frames.push(ResidentFrame {
                source: frame_source,
                block: bi,
                ordinal,
                fresh: owned.clone(),
                items: mixed,
                rates: frame.rates,
                expected_after_sgd: frame.expected_after_sgd,
            });
            fifo.admit(
                &owned,
                &source.targets.path,
                &source.targets.sha256,
                game.lane,
                &saved,
            )?;
            if ri + 1 == block.receipts.len() {
                pending_admission = Some((ordinal, source.receipt.sha256.clone(), lessons));
            } else {
                admissions.push(
                    serde_json::json!({"ordinal":ordinal,"receipt_sha256":source.receipt.sha256,
                    "after_publication":false,"result":admission::complete(&mut archive,lessons)}),
                );
            }
            quotas.settle(0, 0, 0.5);
            ordinal += 1;
        }
        if !gradient_selected {
            gradient_flow.push(serde_json::json!({"block":bi,"available":false,
                "reason":"no complete64 minibatch in this block; no substitution", "additional_backward_calls":0}));
        }
        if recorder.updates - before != block.expected_updates as u64 {
            return Err(invalid("real receipt batch count changed"));
        }
        recorder.version += block.receipts.len() as u64;
        let (result, focus, models) = recorder.boundary(bi, Mode::Reset, Some(&mut archive))?;
        let _ = focus;
        let (receipt_ordinal, receipt_sha256, lessons) = pending_admission
            .take()
            .ok_or_else(|| invalid("missing boundary admission"))?;
        admissions.push(
            serde_json::json!({"ordinal":receipt_ordinal,"receipt_sha256":receipt_sha256,
            "after_publication":true,"result":admission::complete(&mut archive,lessons)}),
        );
        original.push(result);
        original_models.push(models);
        tape.push(frames);
        resident_tape.push(resident_frames);
        boundary_fresh.push(fresh64.iter().cloned().collect::<Vec<_>>());
    }
    if total_fresh != 13038
        || ordinal != 966
        || quotas.consumed_examples - quota_start.consumed_examples != 65190
        || quotas.consumed_recall - quota_start.consumed_recall != 32595
    {
        return Err(invalid("fixed native tape totals or 50% recall changed"));
    }
    resident_models.insert("recorded-reset".into(), original_models);
    let admission_manifest = write_json(&out.join("admissions.json"), &admissions)?;
    let archive_progress = archive.progress();
    let tape_manifest = write_json(
        &out.join("tape-manifest.json"),
        &serde_json::json!({"schema":"paisho-gen5-native-learning-tape-v1",
        "plan_sha256":sha256(&plan_bytes),"frames":tape,"proof_cohort":proof_rows,"proof_cohort_scope":"first 64 chronological distinct certified winning roots of the tape; historical novelty not established","quota":quotas,"quota_start":quota_start,
        "fifo":fifo.progress(),"admissions":admission_manifest,"archive":archive_progress,"new_sampling_not_original_c1":true,"sampler_driven_by":"recorded-reset"}),
    )?;
    drop(archive);
    drop(fifo);
    drop(human);
    drop(recorder);
    let mut arms = serde_json::Map::new();
    arms.insert(
        "recorded-reset".into(),
        serde_json::json!({"boundaries":original,"sgd_seconds":recording_sgd_seconds}),
    );
    for (name, mode) in [
        ("persistent", Mode::Persistent),
        ("reset-replay", Mode::Reset),
    ] {
        let mut arm = Arm::new(&initial, &o, &resume, &pool, out.join(name))?;
        let mut boundaries = vec![];
        let mut boundary_models = vec![];
        let mut sgd_seconds = 0.;
        let mut n = 0;
        let mut hash_seconds = 0.;
        for (bi, frames) in resident_tape.iter().enumerate() {
            let before = arm.updates;
            for frame in frames {
                if Instant::now() >= end {
                    return Err(invalid("tape replay exceeded fixed bound"));
                }
                if frame.block != bi
                    || frame.ordinal != n
                    || frame.source.sha256
                        != tape[bi][n - resident_tape.iter().take(bi).map(Vec::len).sum::<usize>()]
                            .sha256
                {
                    return Err(invalid("resident tape order changed"));
                }
                arm.p.observe(&frame.fresh);
                for (i, batch) in frame.items.chunks(64).enumerate() {
                    let examples = batch.iter().map(|(e, _, _)| e.clone()).collect::<Vec<_>>();
                    let t = Instant::now();
                    arm.p
                        .train_shared(&mut arm.model, &examples, frame.rates[i], 1e-5)?;
                    sgd_seconds += t.elapsed().as_secs_f64();
                    arm.updates += 1;
                    let t = Instant::now();
                    if (mode == Mode::Reset || bi == 0)
                        && bits(&arm.model) != frame.expected_after_sgd[i]
                    {
                        return Err(invalid(
                            "reset replay/initial shared block is not bit exact",
                        ));
                    }
                    hash_seconds += t.elapsed().as_secs_f64();
                }
                n += 1;
            }
            if arm.updates - before != plan.blocks[bi].expected_updates as u64 {
                return Err(invalid("replayed batch count mismatch"));
            }
            arm.version += frames.len() as u64;
            let (result, _, models) = arm.boundary(bi, mode, None)?;
            if mode == Mode::Reset {
                for key in [
                    "incoming_bits",
                    "consolidated_bits",
                    "actor_bits",
                    "working_bits",
                ] {
                    if result[key] != arms["recorded-reset"]["boundaries"][bi][key] {
                        return Err(invalid("reset boundary bits changed"));
                    }
                }
            }
            if mode == Mode::Reset {
                for key in ["protection", "publication"] {
                    if deterministic_state(&result[key])
                        != deterministic_state(&arms["recorded-reset"]["boundaries"][bi][key])
                    {
                        return Err(invalid(format!(
                            "reset boundary {key} state/counters changed"
                        )));
                    }
                }
            }
            boundaries.push(result);
            boundary_models.push(models);
        }
        arms.insert(name.into(),serde_json::json!({"boundaries":boundaries,"sgd_seconds":sgd_seconds,"evidence_hash_seconds":hash_seconds}));
        resident_models.insert(name.into(), boundary_models);
    }
    // Measurement is read-only AFTER the tape and arm trajectories are complete.
    // These known learned rows never become additional admission criteria.
    let mut cohorts = vec![];
    for block in &plan.blocks {
        let manifest = json(&block.retention)?;
        let saved: Vec<SavedMicroExample> =
            serde_json::from_slice(&input(&manifest["union_saved"])?.bytes()?)?;
        let examples = saved
            .iter()
            .map(|s| s.example_for_rules_with_trusted_q(RULES, true))
            .collect::<Result<Vec<_>>>()?;
        cohorts.push((manifest, examples));
    }
    let measurement_started = Instant::now();
    let mut measurements = vec![];
    let mut measure = |label: &str, m: &MicroModel| -> Result<()> {
        if Instant::now() >= end {
            return Err(invalid("tape measurement exceeded fixed bound"));
        }
        let mut panels = vec![];
        for (ci, (manifest, rows)) in cohorts.iter().enumerate() {
            let scored = rows
                .iter()
                .map(|e| scalar_metrics(m, e))
                .collect::<Result<Vec<_>>>()?;
            let mut views = vec![];
            for view in manifest["views"]
                .as_array()
                .ok_or_else(|| invalid("retention views"))?
            {
                let mut totals = [0.; 4];
                let mut weight = 0.;
                for item in view["entries"]
                    .as_array()
                    .ok_or_else(|| invalid("retention weights"))?
                {
                    let i = item["union_index"]
                        .as_u64()
                        .ok_or_else(|| invalid("retention index"))?
                        as usize;
                    let n = item["weight_numerator"]
                        .as_u64()
                        .ok_or_else(|| invalid("retention numerator"))?;
                    let d = item["weight_denominator"]
                        .as_u64()
                        .filter(|d| *d > 0)
                        .ok_or_else(|| invalid("retention denominator"))?;
                    let w = n as f64 / d as f64;
                    if i >= scored.len()
                        || item["weight"].as_f64().map(f64::to_bits) != Some(w.to_bits())
                    {
                        return Err(invalid("retention weight mismatch"));
                    }
                    weight += w;
                    for (k, key) in ["total", "policy", "direct_value", "auxiliary_q"]
                        .iter()
                        .enumerate()
                    {
                        totals[k] += w * scored[i][key].as_f64().unwrap();
                    }
                }
                if (weight - 1.).abs() > 1e-12 {
                    return Err(invalid("retention weights do not sum to one"));
                }
                views.push(serde_json::json!({"name":view["name"],"total":totals[0],"policy":totals[1],"direct_value":totals[2],"auxiliary_q":totals[3]}));
            }
            panels.push(serde_json::json!({"block":ci,"rows":scored,"views":views}));
        }
        let proofs=proof_rows.iter().map(|v|->Result<serde_json::Value>{
            let e:resume_example::ResumeExample=serde_json::from_value(v["example"].clone())?;
            let e=restore(&e)?;
            Ok(serde_json::json!({"key":v["key"],"first_block":v["first_block"],"metric":scalar_metrics(m,&e)?}))
        }).collect::<Result<Vec<_>>>()?;
        let fresh64=boundary_fresh.iter().enumerate().map(|(block,rows)|->Result<serde_json::Value> {
            let values=rows.iter().map(|e|scalar_metrics(m,e)).collect::<Result<Vec<_>>>()?;
            let mean=|key:&str|values.iter().map(|v|v[key].as_f64().unwrap()).sum::<f64>()/values.len() as f64;
            Ok(serde_json::json!({"block":block,"rows":rows.len(),"total":mean("total"),"policy":mean("policy"),
                "direct_value":mean("direct_value"),"auxiliary_q":mean("auxiliary_q")}))
        }).collect::<Result<Vec<_>>>()?;
        measurements.push(serde_json::json!({"model":label,"bits":bits(m),"retention":panels,"proofs":proofs,"fresh64":fresh64}));
        Ok(())
    };
    measure("initial-actor", &base)?;
    for name in ["recorded-reset", "persistent"] {
        for (bi, models) in resident_models[name].iter().enumerate() {
            for (role, model) in ["incoming", "consolidated", "actor", "working"]
                .iter()
                .zip(models)
            {
                if bits(model)
                    != arms[name]["boundaries"][bi][format!("{role}_bits")]
                        .as_str()
                        .unwrap()
                {
                    return Err(invalid("resident model differs from boundary evidence"));
                }
                measure(&format!("{name}-{bi}-{role}"), model)?;
            }
        }
    }
    let measurement_seconds = measurement_started.elapsed().as_secs_f64();

    let report = serde_json::json!({"schema":"paisho-gen5-learner-tape-result-v1","diagnostic_only":true,
        "plan_sha256":sha256(&plan_bytes),"tape":tape_manifest,"arms":arms,"measurements":measurements,
        "seconds":started.elapsed().as_secs_f64(),"measurement_seconds":measurement_seconds,
        "gradient_flow":gradient_flow,"gradient_diagnostic_seconds":gradient_diagnostic_seconds,
        "gradient_selection":"first full64 minibatch of each of four recorded-reset blocks; unavailable if none; no substitutions",
        "gradient_timing_scope":"all extra backward/forward/verification/io outside SGD timer; may warm read caches, not a throughput ABBA",
        "recording_sampling_seconds":sampling_seconds,"tape_serialization_interning_seconds":tape_io_seconds,
        "recording_evidence_hash_seconds":evidence_hash_seconds,"resident_native_example_bytes":resident_examples.bytes,
        "resident_unique_examples":resident_examples.by_content.len(),"tape_reloaded_between_arms":false,
        "new_games":0,"new_searches":0,"fifo_capacity":4096,
        "same_tape_all_arms":true,"reset_replay_every_sgd_and_boundary_exact":true,"reset_replay_control_counters_exact_excluding_timing_and_paths":true,"first_block_common_sgd_exact":true,
        "same_fresh64_top2_interior_and_guard":true,"pool_threads":o.threads,"pool_topology":"one diagnostic native pool; not production five shards of two","measurement_rows_used_for_admission":false,
        "scope":"four fixed C1 receipt groups, new bounded native recall sampled once under reset; not original C1 minibatches, no production strength conclusion"});
    write_json(&out.join("report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gradient_selection_skips_short_batches_and_never_substitutes() {
        let mut selected=false;
        assert!(!select_gradient_batch(&mut selected,5));
        assert!(!select_gradient_batch(&mut selected,63));
        assert!(select_gradient_batch(&mut selected,64));
        assert!(!select_gradient_batch(&mut selected,64));
        assert!(!select_gradient_batch(&mut selected,17));
        assert!(selected);
    }
    #[test]
    fn private_learner_is_saved_before_candidate_repair_and_rollback() {
        let original = MicroModel::seeded(41);
        let saved = original.clone();
        let candidate = MicroModel::seeded(42);
        let accepted = MicroModel::seeded(43);
        assert_ne!(bits(&candidate), bits(&saved));
        assert!(next_learner(Mode::Persistent, &saved, &accepted).shares_storage_with(&original));
        assert!(next_learner(Mode::Reset, &saved, &accepted).shares_storage_with(&accepted));
    }
    #[test]
    fn tape_preserves_trusted_sparse_q_support_context_and_signed_zero() {
        let ex = MicroExample { structured: Vec::new(),
            state: vec![-0.; 417],
            actions: vec![[0.; 32]; 2],
            policy: vec![1., 0.],
            action_values: vec![Some(1.), None],
            policy_support: true,
            value_weight: 0.25,
            policy_weight: 1.,
            value: -1.,
            sequence_source: 91,
        };
        let source = encoded(&ex);
        let bytes = serde_json::to_vec(&source).unwrap();
        let restored: resume_example::ResumeExample = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            serde_json::to_vec(&encoded(&restore(&restored).unwrap())).unwrap(),
            bytes
        );
    }
    #[test]
    fn tape_frame_rejects_changed_budget_rate_and_fresh_count() {
        let ex = MicroExample { structured: Vec::new(),
            state: vec![0.; 417],
            actions: vec![[0.; 32]],
            policy: vec![1.],
            action_values: vec![],
            policy_support: false,
            value_weight: 1.,
            policy_weight: 1.,
            value: 0.,
            sequence_source: 0,
        };
        let mut frame = Frame {
            schema: "paisho-gen5-native-learning-tape-receipt-v1".into(),
            block: 0,
            ordinal: 0,
            source_receipt: Input {
                path: PathBuf::from("unused"),
                sha256: "fixture".into(),
            },
            fresh: vec![encoded(&ex)],
            items: (0..5)
                .map(|i| Item {
                    example: encoded(&ex),
                    kind: if i == 0 { 0 } else { 1 },
                    lane: Lane::Selfplay,
                })
                .collect(),
            rates: vec![0.02 * 5. / 64.],
            expected_after_sgd: vec!["fixture".into()],
            rng_after_sampling: 1,
        };
        assert!(validate_frame(&frame, 0, 0, 0.02).is_ok());
        frame.rates[0] = 0.02;
        assert!(validate_frame(&frame, 0, 0, 0.02).is_err());
        frame.rates[0] = 0.02 * 5. / 64.;
        frame.items[1].kind = 0;
        assert!(validate_frame(&frame, 0, 0, 0.02).is_err());
    }
    #[test]
    fn tape_interning_preserves_context_and_signed_zero() {
        let ex = MicroExample { structured: Vec::new(),
            state: vec![0.; 417],
            actions: vec![[0.; 32]],
            policy: vec![1.],
            action_values: vec![],
            policy_support: false,
            value_weight: 1.,
            policy_weight: 1.,
            value: 0.,
            sequence_source: 0,
        };
        let mut resident = ResidentExamples {
            by_content: BTreeMap::new(),
            bytes: 0,
        };
        let a = resident.intern(Arc::new(ex.clone())).unwrap();
        let b = resident.intern(Arc::new(ex.clone())).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let mut other = ex.clone();
        other.sequence_source = 1;
        assert!(!Arc::ptr_eq(&a, &resident.intern(Arc::new(other)).unwrap()));
        let mut other = ex;
        other.state[0] = -0.;
        assert!(!Arc::ptr_eq(&a, &resident.intern(Arc::new(other)).unwrap()));
    }
    #[test]
    fn preflight_accepts_only_the_native_empty_value_shape() {
        let mut row = crate::micro_learning::tactics::fixture();
        row.policy_weight = 0.;
        row.structured.clear();row.actions.clear();
        row.action_features.clear();
        row.policy.clear();
        row.new_visits.clear();
        assert!(empty_value_target(&row));
        row.policy_weight = 1.;
        assert!(!empty_value_target(&row));
        row.policy_weight = 0.;
        row.action_features.push(vec![0.; 32]);
        assert!(!empty_value_target(&row));
        row.action_features.clear();
        row.policy.push(1.);
        assert!(!empty_value_target(&row));
        row.policy.clear();
        row.new_visits.push(1);
        assert!(!empty_value_target(&row));
    }
}
