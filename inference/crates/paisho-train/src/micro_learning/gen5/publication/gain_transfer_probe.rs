//! Isolated, source-authenticated transfer of learned proof choices after Guard.
//! The frozen selection supplies chronological proofs; model eligibility never
//! replaces one selected source by another. No campaign or Guard publication.
use super::*;
mod policy_relay;
pub use policy_relay::run as run_policy_relay;
use std::collections::BTreeSet;
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
                "gain transfer input changed: {}",
                self.path.display()
            )));
        }
        Ok(b)
    }
}
fn input(v: &serde_json::Value) -> Result<Input> {
    Ok(serde_json::from_value(v.clone())?)
}
fn read(v: &serde_json::Value) -> Result<Vec<u8>> {
    input(v)?.bytes()
}
fn snapshot(
    v: &serde_json::Value,
    version: u64,
    base: Option<&MicroModel>,
) -> Result<Arc<Snapshot>> {
    let i = input(v)?;
    let a: MicroArtifact = serde_json::from_slice(&i.bytes()?)?;
    if let Some(base) = base {
        return load_snapshot(&i.path, &a.identity(), version, base);
    }
    let model = a.model()?;
    Ok(Arc::new(Snapshot {
        identity: a.identity(),
        artifact: Some(Arc::new(a)),
        model: Arc::new(model),
        version,
        path: i.path,
    }))
}
fn parameters(model: &MicroModel) -> String {
    sha256(
        &model
            .parameters()
            .iter()
            .flat_map(|p| p.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
pub(super) fn value_equal(a: &MicroModel, b: &MicroModel) -> bool {
    a.parameters().len() == b.parameters().len()
        && a.parameters()
            .iter()
            .zip(b.parameters())
            .enumerate()
            .all(|(i, (a, b))| !branches::value_parameter(i) || a.to_bits() == b.to_bits())
}
fn phase(
    rows: &[Arc<Witness>],
    model: &MicroModel,
    beta: f64,
    parallel: Option<&cpu::Ordered>,
) -> Result<Score> {
    read_cache::Reads::default().evaluate(rows, model, beta, parallel)
}
fn pairs(before: &Score, after: &Score) -> serde_json::Value {
    let counts = |a: &[bool], b: &[bool]| {
        serde_json::json!({"before":a.iter().filter(|v|**v).count(),"after":b.iter().filter(|v|**v).count(),
        "gains":a.iter().zip(b).filter(|(a,b)|!**a&&**b).count(),"losses":a.iter().zip(b).filter(|(a,b)|**a&&!**b).count()})
    };
    serde_json::json!({"raw":counts(&before.raw,&after.raw),"coupled":counts(&before.coupled,&after.coupled)})
}
pub(super) fn fresh_loss(
    model: &MicroModel,
    rows: &[Arc<MicroExample>],
    parallel: &cpu::Ordered,
) -> Result<f64> {
    let m = model.clone();
    let n = rows.len();
    let f = move |e: &Arc<MicroExample>| {
        m.loss_loop_v3(e)
            .map(|l| l.total(e.policy_weight) / n as f64)
    };
    let mut loss = 0.;
    for v in parallel.map_owned(rows.to_vec(), |e| e.actions.len(), f) {
        loss += v.map_err(invalid)?;
    }
    if !loss.is_finite() {
        return Err(invalid("non-finite fixed fresh loss"));
    }
    Ok(loss)
}
fn fresh(
    capsule: &serde_json::Value,
    current: &serde_json::Value,
    previous: &serde_json::Value,
) -> Result<Vec<Arc<MicroExample>>> {
    let saved: Vec<SavedMicroExample> = serde_json::from_slice(&read(&capsule["fresh"])?)?;
    let provenance = capsule["fresh_provenance"]
        .as_array()
        .ok_or_else(|| invalid("fresh provenance missing"))?;
    if saved.len() != 64 || provenance.len() != 64 {
        return Err(invalid(
            "gain transfer fresh must contain exactly64 bounded rows",
        ));
    }
    let mut all = vec![];
    // Two adjacent recorded cycles are sufficient for these preselected capsules.
    // In particular, cycle24 has only55 fresh rows and carries9 rows from cycle23.
    for report in [previous, current] {
        let cycle = report["cycle"]
            .as_u64()
            .ok_or_else(|| invalid("fresh cycle missing"))?;
        for (ri, receipt) in report["receipts"]
            .as_array()
            .ok_or_else(|| invalid("fresh receipts missing"))?
            .iter()
            .enumerate()
        {
            if receipt["stride"] != 1 {
                return Err(invalid("gain transfer fresh requires actual stride1"));
            }
            let i = Input {
                path: PathBuf::from(
                    receipt["targets"]
                        .as_str()
                        .ok_or_else(|| invalid("fresh targets missing"))?,
                ),
                sha256: receipt["targets_sha256"]
                    .as_str()
                    .ok_or_else(|| invalid("fresh target hash missing"))?
                    .into(),
            };
            let rows = decode_examples(&i.bytes()?)?;
            if receipt["selected_fresh"].as_u64() != Some(rows.len() as u64) {
                return Err(invalid("fresh selected row count differs"));
            }
            for (index, s) in rows.iter().enumerate() {
                all.push((
                    cycle,
                    ri,
                    index,
                    receipt["id"].clone(),
                    i.clone(),
                    serde_json::to_value(s)?,
                ));
            }
        }
    }
    if all.len() < 64 {
        return Err(invalid(
            "two recorded cycles cannot reconstruct fixed fresh64",
        ));
    }
    let tail = &all[all.len() - 64..];
    for (((cycle, receipt_order, index, receipt_id, src, row), p), saved) in
        tail.iter().zip(provenance).zip(&saved)
    {
        let selected = input(&p["targets"])?;
        if p["cycle"].as_u64() != Some(*cycle)
            || p["receipt_order"].as_u64() != Some(*receipt_order as u64)
            || p["index"].as_u64() != Some(*index as u64)
            || &p["receipt_id"] != receipt_id
            || selected.sha256 != src.sha256
            || selected.path.canonicalize()? != src.path.canonicalize()?
            || &serde_json::to_value(saved)? != row
            || p["decision"].as_u64() != Some(saved.decision as u64)
        {
            return Err(invalid(
                "fixed fresh64 is not the actual chronological Protection tail",
            ));
        }
    }
    saved
        .iter()
        .map(|s| {
            s.example_for_rules_with_trusted_q(RULES, true)
                .map(Arc::new)
        })
        .collect()
}

pub fn run(selection_path: &Path, cycle: usize, out: &Path) -> Result<serde_json::Value> {
    run_with_schedule(selection_path,cycle,out,false)
}
pub fn run_relinearized(selection_path: &Path, cycle: usize, out: &Path) -> Result<serde_json::Value> {
    run_with_schedule(selection_path,cycle,out,true)
}
fn run_with_schedule(selection_path:&Path,cycle:usize,out:&Path,relinearize:bool)->Result<serde_json::Value> {
    if out.exists() || ![23, 24].contains(&cycle) {
        return Err(invalid(
            "gain transfer requires cycle23/24 and new output directory",
        ));
    }
    let loading = Instant::now();
    let selection_bytes = fs::read(selection_path)?;
    let selection: serde_json::Value = serde_json::from_slice(&selection_bytes)?;
    if selection["schema"] != "gen5-two-cycle-gain-transfer-frozen-selection-v1"
        || selection["maximum_selected_per_transaction"] != 8
        || selection["maximum_admitted_steps"] != 4
        || selection["maximum_complete_guard_checks"] != 12
    {
        return Err(invalid("gain transfer frozen protocol mismatch"));
    }
    // Bind BOTH capsule selections before any prediction or tuning, then load
    // only the requested cycle. Running23 does not evaluate the reserved24.
    let first = input(&selection["first_transaction"])?;
    let reserved = input(&selection["reserved_second_transaction"])?;
    first.bytes()?;
    reserved.bytes()?;
    let capsule_input = if cycle == 23 {
        first.clone()
    } else {
        reserved.clone()
    };
    let capsule_bytes = capsule_input.bytes()?;
    let capsule: serde_json::Value = serde_json::from_slice(&capsule_bytes)?;
    if capsule["cycle"] != cycle {
        return Err(invalid("gain transfer capsule cycle differs"));
    }
    let report: serde_json::Value = serde_json::from_slice(&read(&capsule["cycle_report"])?)?;
    let previous: serde_json::Value = serde_json::from_slice(&read(&capsule["previous_report"])?)?;
    if report["cycle"] != cycle || previous["cycle"] != cycle - 1 {
        return Err(invalid("gain transfer reports are not adjacent"));
    }
    let options: Options = serde_json::from_slice(&read(&capsule["config"])?)?;
    let gp = input(&capsule["guard"])?;
    let vp = input(&capsule["validation"])?;
    gp.bytes()?;
    vp.bytes()?;
    if options.value_policy_strength != 16.
        || options.publication_guard.as_ref() != Some(&gp.path)
        || options.publication_validation.as_ref() != Some(&vp.path)
    {
        return Err(invalid("gain transfer panel/config mismatch"));
    }
    let frozen_plan: serde_json::Value = serde_json::from_slice(&read(&capsule["loop_plan"])?)?;
    if frozen_plan["immediate_lesson_admission"] != true || frozen_plan["max_fresh_per_game"] != 512
    {
        return Err(invalid(
            "gain transfer requires faithful all-fresh source run",
        ));
    }
    let anchor = snapshot(
        &capsule["guard_anchor"],
        previous["publication"]["accepted_version"]
            .as_u64()
            .unwrap_or((cycle - 1) as u64),
        None,
    )?;
    let pre = snapshot(&capsule["learner_pre"], cycle as u64, Some(&anchor.model))?;
    let post = snapshot(&capsule["learner_post"], cycle as u64, Some(&anchor.model))?;
    let start = snapshot(
        &capsule["admissible_start"],
        cycle as u64,
        Some(&anchor.model),
    )?;
    if report["old_actor"] != anchor.identity
        || report["learner"] != post.identity
        || report["actor"] != start.identity
        || previous["actor"] != anchor.identity
    {
        return Err(invalid(
            "gain transfer model identities differ from native reports",
        ));
    }
    let specs = capsule["proofs"]
        .as_array()
        .ok_or_else(|| invalid("gain transfer proof selection missing"))?;
    if !(1..=8).contains(&specs.len()) {
        return Err(invalid("gain transfer requires1..8 preselected proofs"));
    }
    let mut witnesses = vec![];
    let mut metadata = vec![];
    let mut seen = BTreeSet::new();
    let mut last = None;
    for spec in specs {
        let order = (
            spec["receipt_order"]
                .as_u64()
                .ok_or_else(|| invalid("proof receipt missing"))?,
            spec["decision"]
                .as_u64()
                .ok_or_else(|| invalid("proof decision missing"))?,
            spec["key"]
                .as_str()
                .ok_or_else(|| invalid("proof key missing"))?
                .to_string(),
        );
        if last.as_ref().is_some_and(|p| p >= &order) || !seen.insert(order.2.clone()) {
            return Err(invalid("proof selection is not chronological and unique"));
        }
        last = Some(order.clone());
        let receipt = report["receipts"]
            .get(order.0 as usize)
            .ok_or_else(|| invalid("proof receipt outside cycle"))?;
        let (w, m) = dynamic_proof_probe::transfer_witness(spec, &anchor.model, receipt)?;
        witnesses.push(w);
        metadata.push(m);
    }
    let fresh = fresh(&capsule, &report, &previous)?;
    fs::create_dir(out)?;
    fs::write(out.join("frozen-selection.json"), &selection_bytes)?;
    fs::write(out.join("frozen-capsule.json"), &capsule_bytes)?;
    let mut guard = Guard::open(
        &gp.path,
        &out.join("guard"),
        options.value_policy_strength,
        anchor.clone(),
        &previous["publication"],
    )?;
    guard.enable_v2(&vp.path)?;
    guard.enable_v3()?;
    let (pool, _) = cpu::build_pool(options.threads, None)?;
    guard.enable_parallel(&[pool.clone()]);
    let parallel = cpu::Ordered::new(&[pool]);
    let (start_guard, why) = guard.checked_v3(&start.model)?;
    if !why.is_empty() {
        return Err(invalid(format!(
            "gain transfer start not admissible under prior actor: {why:?}"
        )));
    }
    let mut phases = vec![];
    for snapshot in [&anchor, &pre, &post, &start] {
        phases.push(phase(
            &witnesses,
            &snapshot.model,
            options.value_policy_strength,
            guard.parallel.as_ref(),
        )?);
    }
    let mut targets = vec![];
    let mut goals = vec![];
    let mut eligibility = vec![];
    for (i, w) in witnesses.iter().enumerate() {
        let goal = gain_probe::GainGoals {
            raw: phases[2].raw[i] && !phases[3].raw[i],
            coupled: phases[2].coupled[i] && !phases[3].coupled[i],
        };
        eligibility.push(serde_json::json!({"id":i,"key":specs[i]["key"],"raw_goal":goal.raw,"coupled_goal":goal.coupled,
            "eligible":goal.raw||goal.coupled,"new_raw_acquisition_during_block":!phases[0].raw[i]&&phases[2].raw[i],
            "new_coupled_acquisition_during_block":!phases[0].coupled[i]&&phases[2].coupled[i],"selected_even_if_ineligible":true}));
        let inputs = successor_inputs(w, &start.model)?;
        let offsets = inputs
            .iter()
            .map(|(sign, state)| {
                options.value_policy_strength
                    * if state.is_empty() {
                        *sign
                    } else {
                        let v = start.model.value(state);
                        if *sign == 1. {
                            v
                        } else {
                            -v
                        }
                    }
            })
            .collect();
        let mut ex = w.example.as_ref().clone();
        ex.policy_support = true;
        ex.sequence_source = 0;
        ex.value_weight = 0.;
        ex.action_values.clear();
        targets.push(GainTarget {
            id: i,
            example: Arc::new(ex),
            coupled_offsets: offsets,
        });
        goals.push(goal);
    }
    let phase_fresh = [&anchor, &pre, &post, &start]
        .iter()
        .map(|s| fresh_loss(&s.model, &fresh, &parallel))
        .collect::<Result<Vec<_>>>()?;
    let loading_seconds = loading.elapsed().as_secs_f64();
    let mut trials = vec![];
    // An efficacy comparison of different algorithms, not an exact speed ABBA.
    // No hyperparameter is adjusted between the two predeclared modes/cycles.
    for projection in [false, true] {
        let (trajectory, probe) = if projection && relinearize {
            guard.diagnostic_fresh_gain_transfer_relinearized(&start.model,&targets,&goals,&fresh)?
        } else {
            guard.diagnostic_fresh_gain_transfer(&start.model,&targets,&goals,&fresh,projection)?
        };
        let mut steps = vec![];
        for (i, m) in trajectory.iter().enumerate() {
            let scores = phase(
                &witnesses,
                m,
                options.value_policy_strength,
                guard.parallel.as_ref(),
            )?;
            let f = fresh_loss(m, &fresh, &parallel)?;
            if !value_equal(&start.model, m) || f > phase_fresh[3] + 1e-12 {
                return Err(invalid(
                    "retained gain trajectory violated frozenV or fixed fresh ceiling",
                ));
            }
            let path = out.join(format!("projection-{projection}-step-{i:02}.json"));
            MicroArtifact::new(m,start.artifact.as_ref().unwrap().updates,serde_json::json!({"diagnostic_only":true,"kind":"bounded-proof-gain-transfer","cycle":cycle,"projection":projection,"step":i})).save(&path)?;
            steps.push(serde_json::json!({"step":i,"model":path,"parameters":parameters(m),"proof_scores":scores,"fresh_loss":f,
                "pairs_vs_start":pairs(&phases[3],&scores),"pairs_vs_post_learner":pairs(&phases[2],&scores),"value_bits_exact":true}));
        }
        trials.push(serde_json::json!({"projection":projection,"probe":probe,"trajectory":steps}));
    }
    first.bytes()?;
    reserved.bytes()?;
    if fs::read(selection_path)? != selection_bytes || capsule_input.bytes()? != capsule_bytes {
        return Err(invalid("gain transfer frozen selection changed"));
    }
    for key in [
        "guard_anchor",
        "learner_pre",
        "learner_post",
        "admissible_start",
        "guard",
        "validation",
        "config",
        "loop_plan",
        "cycle_report",
        "previous_report",
        "fresh",
    ] {
        read(&capsule[key])?;
    }
    for spec in specs {
        read(&spec["proof"])?;
        read(&spec["targets"])?;
    }
    let result = serde_json::json!({"schema":"paisho-gen5-proof-gain-transfer-result-v1","cycle":cycle,"relinearize_after_accepted":relinearize,
        "selection_sha256":sha256(&selection_bytes),"capsule":capsule_input,"metadata":metadata,"eligibility":eligibility,
        "phase_order":["prior_actor","learner_pre","learner_post","admissible_start"],"phases":phases,"phase_fresh_losses":phase_fresh,
        "learner_post_and_start_value_bits_equal":value_equal(&post.model,&start.model),"start_guard":start_guard,
        "loading_and_initial_measurements_seconds":loading_seconds,"runs":trials,"diagnostic_only":true,
        "fixed_target_selection_no_replacement":true,"fixed_fresh64_provenance_verified":true,"native_proofs_verified":true,
        "scope":"bounded re-acquisition and retention inside an isolated policy adjustment; not future SGD retention, campaign throughput, search-win improvement or runtime activation"});
    fs::write(out.join("report.json"), serde_json::to_vec_pretty(&result)?)?;
    Ok(result)
}
