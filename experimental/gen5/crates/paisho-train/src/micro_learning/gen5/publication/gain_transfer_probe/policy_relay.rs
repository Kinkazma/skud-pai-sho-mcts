//! Isolated safe-actor proof relay. No runtime caller, sampler or search.
use super::super::{dynamic_proof_probe, gain_probe};
use super::*;
use super::super::policy_transfer::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema: String,
    source_plan: Input,
    tape_report: Input,
    blocks: Vec<usize>,
    selected_per_block: usize,
    max_corrections: usize,
    max_backtracks: usize,
    max_candidate_guard_checks: usize,
    max_seconds: u64,
    #[serde(default)]
    adaptive_counterexamples: bool,
    #[serde(default)]
    adaptive_kl: bool,
    #[serde(default)]
    preserve_fresh_gain: bool,
    #[serde(default)]
    arbitrary_cases: Vec<Case>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    label: String,
    anchor: Input,
    seed: Input,
    teacher: Input,
    target: usize,
    #[serde(default)]
    fresh: Option<Input>,
}
fn load_json(v: &serde_json::Value) -> Result<serde_json::Value> {
    Ok(serde_json::from_slice(&read(v)?)?)
}
fn write(path: &Path, value: &impl Serialize) -> Result<Input> {
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, &bytes)?;
    Ok(Input {
        path: path.into(),
        sha256: sha256(&bytes),
    })
}
fn report_model(
    boundary: &serde_json::Value,
    role: &str,
    base: &MicroModel,
) -> Result<Arc<Snapshot>> {
    let path = PathBuf::from(
        boundary[role]
            .as_str()
            .ok_or_else(|| invalid("relay missing bound model"))?,
    );
    let bytes = fs::read(&path)?;
    let artifact: MicroArtifact = serde_json::from_slice(&bytes)?;
    let version = boundary["publication"]["accepted_version"]
        .as_u64()
        .ok_or_else(|| invalid("relay version"))?;
    let model = load_snapshot(&path, &artifact.identity(), version, base)?;
    if parameters(&model.model) != boundary[format!("{role}_bits")].as_str().unwrap_or("") {
        return Err(invalid(
            "relay model differs from frozen tape parameter hash",
        ));
    }
    Ok(model)
}
fn loaded_cohort(
    source: &serde_json::Value,
    manifest: &serde_json::Value,
    base: &MicroModel,
    out: &Path,
) -> Result<(Vec<Arc<Witness>>, Vec<serde_json::Value>)> {
    let entries = manifest["proof_cohort"]
        .as_array()
        .ok_or_else(|| invalid("relay cohort missing"))?;
    if entries.len() != 64 {
        return Err(invalid("relay requires original64 proof cohort"));
    }
    fs::create_dir(out)?;
    let mut sources = std::collections::BTreeMap::new();
    let mut ordinal = 0;
    for (bi, b) in source["blocks"]
        .as_array()
        .ok_or_else(|| invalid("relay source blocks missing"))?
        .iter()
        .enumerate()
    {
        for s in b["receipts"]
            .as_array()
            .ok_or_else(|| invalid("relay receipts missing"))?
        {
            let key = input(&s["receipt"])?;
            sources.insert((key.path, key.sha256), (bi, ordinal, s.clone()));
            ordinal += 1;
        }
    }
    if ordinal != 966 {
        return Err(invalid("relay real receipt count changed"));
    }
    let mut cache = std::collections::BTreeMap::<
        usize,
        (
            serde_json::Value,
            Vec<SavedMicroExample>,
            serde_json::Value,
            GameRecord,
        ),
    >::new();
    let mut rows = vec![];
    let mut metadata = vec![];
    let mut keys = BTreeSet::new();
    let mut last_ordinal = 0;
    for (i, entry) in entries.iter().enumerate() {
        let ri = input(&entry["receipt"])?;
        let (bi, ordinal, desc) = sources
            .get(&(ri.path.clone(), ri.sha256.clone()))
            .ok_or_else(|| invalid("proof receipt outside original tape"))?;
        if entry["first_block"].as_u64() != Some(*bi as u64) || *ordinal < last_ordinal {
            return Err(invalid("proof source chronology changed"));
        }
        last_ordinal = *ordinal;
        if !cache.contains_key(ordinal) {
            let r = load_json(&desc["receipt"])?;
            let psr = read(&desc["psr"])?;
            let record: GameRecord = std::str::from_utf8(&psr)?.parse()?;
            let saved = decode_examples(&read(&desc["targets"])?)?;
            let bytes = read(&desc["bundle"])?;
            let bundle: serde_json::Value =
                serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
            if record.rules() != RULES
                || record.to_string().as_bytes() != psr
                || r["fully_learned"] != true
                || !r["error"].is_null()
                || r["fresh_used"].as_u64() != Some(saved.len() as u64)
                || r["targets_sha256"] != desc["targets"]["sha256"]
                || r["psr_sha256"] != desc["psr"]["sha256"]
                || bundle["schema"] != "paisho-gen5-durable-lessons-v1"
                || bundle["rules"] != RULES.as_str()
                || bundle["psr"] != record.to_string()
                || bundle["psr_sha256"] != r["psr_sha256"]
                || bundle["game_id"] != r["id"]
                || bundle["source"] != r["collector"]
                || bundle["lessons"]
                    != serde_json::to_value(
                        saved
                            .iter()
                            .map(durable::Lesson::from_saved)
                            .collect::<Vec<_>>(),
                    )?
            {
                return Err(invalid(
                    "relay cohort Saved/receipt/bundle native identity mismatch",
                ));
            }
            cache.insert(*ordinal, (r, saved, bundle, record));
        }
        let (r, saved, bundle, record) = &cache[ordinal];
        let index = entry["saved_index"]
            .as_u64()
            .ok_or_else(|| invalid("proof Saved index"))? as usize;
        let s = saved
            .get(index)
            .ok_or_else(|| invalid("proof Saved absent"))?;
        let decision = entry["decision"]
            .as_u64()
            .ok_or_else(|| invalid("proof decision"))? as usize;
        if decision == 0 || decision > record.actions().len() + 1 || decision != s.decision {
            return Err(invalid("relay proof decision bounds"));
        }
        let prefix = cases::prefix(record, decision - 1).to_string();
        let key = sha256(prefix.as_bytes());
        if entry["key"] != key || !keys.insert(key.clone()) {
            return Err(invalid("relay proof key mismatch/duplicate"));
        }
        let native = s.example_for_rules_with_trusted_q(RULES, true)?;
        if serde_json::to_value(resume_example::ResumeExample::from_with_trusted_q(
            &native, true,
        ))? != entry["example"]
        {
            return Err(invalid("relay cohort native Resume target mismatch"));
        }
        let certificates: Vec<(usize, MicroProofCertificate)> =
            serde_json::from_value(bundle["proofs"].clone())?;
        let certificate = certificates
            .iter()
            .find(|(d, _)| *d == decision)
            .ok_or_else(|| invalid("relay native certificate absent"))?;
        // This export is the actual recorded certificate and exact PSR prefix;
        // transfer_witness below independently replays and verifies the rules.
        let proof = write(
            &out.join(format!("{key}.json")),
            &serde_json::json!({"prefix":prefix,"rules":RULES.as_str(),"certificate":certificate.1}),
        )?;
        let receipt = serde_json::json!({"targets":desc["targets"]["path"],"targets_sha256":desc["targets"]["sha256"],
            "stride":1,"actor":r["collector"],"source_group":r["psr_sha256"]});
        let spec = serde_json::json!({"key":key,"receipt_order":ordinal,"target_index":index,"decision":decision,
            "proof":proof,"targets":desc["targets"],"collector_identity":r["collector"],"source_group":r["psr_sha256"],
            "native_source":{"receipt":desc["receipt"],"psr":desc["psr"]}});
        let (witness, verified) = dynamic_proof_probe::transfer_witness(&spec, base, &receipt)?;
        metadata.push(serde_json::json!({"index":i,"first_block":bi,"key":key,"receipt":entry["receipt"],"sources":desc,"native_proof":proof,"verified":verified}));
        rows.push(witness);
    }
    Ok((rows, metadata))
}
fn bound_measurement<'a>(
    report: &'a serde_json::Value,
    name: &str,
) -> Result<&'a serde_json::Value> {
    report["measurements"]
        .as_array()
        .and_then(|a| a.iter().find(|v| v["model"] == name))
        .ok_or_else(|| invalid("relay frozen measurement missing"))
}
fn exact_raw(
    rows: &[Reading],
    old: &serde_json::Value,
    cohort: &[serde_json::Value],
) -> Result<()> {
    let proof = old["proofs"]
        .as_array()
        .ok_or_else(|| invalid("relay proof measurements missing"))?;
    if proof.len() != rows.len() {
        return Err(invalid("relay proof count changed"));
    }
    for ((r, p), c) in rows.iter().zip(proof).zip(cohort) {
        if p["key"] != c["key"]
            || p["metric"]["played_raw_in_support"] != r.raw
            || p["metric"]["played_raw_support_mass"]
                .as_f64()
                .map(f64::to_bits)
                != Some(r.mass.to_bits())
        {
            return Err(invalid("relay raw initial read differs from frozen tape"));
        }
    }
    Ok(())
}
fn retention_measure(
    model: &MicroModel,
    panels: &[(serde_json::Value, Vec<MicroExample>)],
) -> Result<serde_json::Value> {
    let mut out = vec![];
    for (bi, (manifest, rows)) in panels.iter().enumerate() {
        let scored = rows
            .iter()
            .map(|e| -> Result<[f64; 4]> {
                let l = model.loss_loop_v3(e).map_err(invalid)?;
                let delta = model.embed(&e.state).value - e.value;
                let direct = 0.5 * delta * delta * e.value_weight;
                let v = [
                    l.total(e.policy_weight),
                    l.policy * e.policy_weight,
                    direct,
                    l.value - direct,
                ];
                if v.iter().any(|v| !v.is_finite()) {
                    return Err(invalid("non-finite post-relay retention"));
                }
                Ok(v)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut views = vec![];
        for view in manifest["views"]
            .as_array()
            .ok_or_else(|| invalid("relay retention views"))?
        {
            let mut sum = [0.; 4];
            let mut mass = 0.;
            for item in view["entries"]
                .as_array()
                .ok_or_else(|| invalid("relay retention entries"))?
            {
                let index = item["union_index"]
                    .as_u64()
                    .ok_or_else(|| invalid("relay retention index"))?
                    as usize;
                let n = item["weight_numerator"]
                    .as_u64()
                    .ok_or_else(|| invalid("retention numerator"))?;
                let d = item["weight_denominator"]
                    .as_u64()
                    .filter(|d| *d > 0)
                    .ok_or_else(|| invalid("retention denominator"))?;
                let w = n as f64 / d as f64;
                let v = scored
                    .get(index)
                    .ok_or_else(|| invalid("retention index outside rows"))?;
                if item["weight"].as_f64().map(f64::to_bits) != Some(w.to_bits()) {
                    return Err(invalid("retention weights changed"));
                }
                mass += w;
                for (a, b) in sum.iter_mut().zip(v) {
                    *a += w * b;
                }
            }
            if (mass - 1.).abs() > 1e-12 {
                return Err(invalid("retention weights sum"));
            }
            views.push(serde_json::json!({"name":view["name"],"total":sum[0],"policy":sum[1],"direct_value":sum[2],"auxiliary_q":sum[3]}));
        }
        out.push(serde_json::json!({"block":bi,"rows":scored,"views":views}));
    }
    Ok(serde_json::json!(out))
}
fn run_cases(
    plan: &Plan,
    plan_bytes: &[u8],
    options: &Options,
    initial: &Snapshot,
    cohort: &[Arc<Witness>],
    metadata: &[serde_json::Value],
    pool: &Arc<rayon::ThreadPool>,
    out: &Path,
    begin: Instant,
) -> Result<serde_json::Value> {
    if plan.arbitrary_cases.len() != 4 {
        return Err(invalid("relay continuation requires four fixed cases"));
    }
    let mut selected = vec![];
    let mut loaded = vec![];
    for case in &plan.arbitrary_cases {
        if case.target >= cohort.len() || case.label.is_empty() {
            return Err(invalid("invalid fixed continuation target"));
        }
        let anchor = snapshot(&serde_json::to_value(&case.anchor)?, 0, Some(&initial.model))?;
        let seed = snapshot(&serde_json::to_value(&case.seed)?, 0, Some(&initial.model))?;
        let teacher = snapshot(&serde_json::to_value(&case.teacher)?, 0, Some(&initial.model))?;
        let before = readings(&seed.model, cohort)?;
        let taught = readings(&teacher.model, cohort)?;
        if before[case.target].raw || !taught[case.target].raw {
            return Err(invalid("fixed continuation target is not an acquired then lost choice"));
        }
        selected.push(serde_json::json!({"label":case.label,"target":case.target,"key":metadata[case.target]["key"],
            "anchor":case.anchor,"anchor_bits":parameters(&anchor.model),"seed":case.seed,"seed_bits":parameters(&seed.model),
            "teacher":case.teacher,"teacher_bits":parameters(&teacher.model),"seed_all":before,"teacher_all":taught}));
        let fresh = case.fresh.as_ref().map(|i| -> Result<Vec<Arc<MicroExample>>> {
            let saved: Vec<SavedMicroExample> = serde_json::from_slice(&i.bytes()?)?;
            if saved.len()!=64 {return Err(invalid("continuation fresh window must be64"));}
            saved.iter().map(|s|s.example_for_rules_with_trusted_q(RULES,true).map(Arc::new)).collect()
        }).transpose()?;
        loaded.push((anchor, seed, before, teacher, fresh));
    }
    let selection = write(&out.join("selection-before-fit.json"), &selected)?;
    let loading_seconds = begin.elapsed().as_secs_f64();
    let mut results = vec![];
    for (i, (case, (anchor, seed, before, teacher, fresh))) in plan.arbitrary_cases.iter().zip(loaded).enumerate() {
        if begin.elapsed().as_secs_f64() > plan.max_seconds as f64 {
            return Err(invalid("relay continuation wall bound exceeded"));
        }
        let case_out = out.join(format!("case-{i:03}"));
        fs::create_dir(&case_out)?;
        let setup = Instant::now();
        let mut guard = Guard::open(options.publication_guard.as_ref().unwrap(), &case_out.join("guard"),
            options.value_policy_strength, anchor.clone(), &serde_json::Value::Null)?;
        guard.enable_v2(options.publication_validation.as_ref().unwrap())?;
        guard.enable_v3()?;
        guard.enable_parallel(&[pool.clone()]);
        let guard_setup_seconds = setup.elapsed().as_secs_f64();
        let preparation=Instant::now();
        let fresh_limit = fresh.as_ref().map(|rows| -> Result<FreshLimit> {
            let pool=guard.parallel.as_ref().unwrap();
            let old=fresh_loss(&anchor.model,rows,pool)?;
            let learned=fresh_loss(&teacher.model,rows,pool)?;
            Ok(FreshLimit {rows:rows.clone(),ceiling:if learned<old {old-0.05*(old-learned)} else {learned}})
        }).transpose()?;
        if plan.preserve_fresh_gain && fresh_limit.is_none() {return Err(invalid("missing frozen fresh64 inputs"));}
        let fresh_preparation_seconds=preparation.elapsed().as_secs_f64();
        let (after, detail) = relay(&guard, &seed.model, cohort, &before, case.target, plan.adaptive_counterexamples, plan.adaptive_kl, plan.max_corrections,
            if plan.preserve_fresh_gain {fresh_limit.as_ref()} else {None})?;
        let measure = Instant::now();
        let fresh_measurement = fresh.as_ref().map(|rows| -> Result<serde_json::Value> {
            let pool=guard.parallel.as_ref().unwrap();
            let old=fresh_loss(&anchor.model,rows,pool)?;
            let learned=fresh_loss(&teacher.model,rows,pool)?;
            let seed_loss=fresh_loss(&seed.model,rows,pool)?;
            let after_loss=fresh_loss(&after,rows,pool)?;
            let ceiling=if learned<old {old-0.05*(old-learned)} else {learned};
            Ok(serde_json::json!({"rows":rows.len(),"anchor":old,"consolidated":learned,"seed":seed_loss,"after":after_loss,
                "consolidation_ceiling":ceiling,"seed_passes":seed_loss<=ceiling+1e-12,"after_passes":after_loss<=ceiling+1e-12,
                "after_no_worse_than_seed":after_loss<=seed_loss+1e-12,"measurement_only":!plan.preserve_fresh_gain}))
        }).transpose()?;
        let artifact = MicroArtifact::new(&after, seed.artifact.as_ref().unwrap().updates, serde_json::json!({"diagnostic_only":true,
            "kind":"proof-policy-relay-continuation","case":case.label,"before":seed.identity}));
        let after_file = write(&case_out.join("after.json"), &artifact)?;
        let row = serde_json::json!({"case":case.label,"target":case.target,"before":case.seed,"after":after_file,
            "before_bits":parameters(&seed.model),"after_bits":parameters(&after),"guard_setup_seconds":guard_setup_seconds,
            "seed_preserved_on_failure":detail["accepted"]==true||parameters(&after)==parameters(&seed.model),"detail":detail,
            "fresh":fresh_measurement,"fresh_preparation_seconds":fresh_preparation_seconds,"fresh_measurement_seconds":measure.elapsed().as_secs_f64()});
        write(&case_out.join("report.json"), &row)?;
        results.push(row);
    }
    let result = serde_json::json!({"schema":"paisho-gen5-policy-relay-continuation-result-v1","plan_sha256":sha256(plan_bytes),
        "source_plan":plan.source_plan,"tape_report":plan.tape_report,"selection_before_fit":selection,"results":results,
        "seconds":begin.elapsed().as_secs_f64(),"loading_seconds":loading_seconds,"bank_loads":1,"diagnostic_only":true,
        "new_games":0,"searches_during_optimization":0,"fresh_loss_is_not_admission":!plan.preserve_fresh_gain});
    write(&out.join("report.json"), &result)?;
    Ok(result)
}
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("relay output must be new"));
    }
    let begin = Instant::now();
    let plan_bytes = fs::read(plan_path)?;
    let plan: Plan = serde_json::from_slice(&plan_bytes)?;
    if plan.schema != "paisho-gen5-tape-policy-relay-v1"
        || plan.blocks != [0, 1, 2, 3]
        || plan.selected_per_block != 1
        || !(STEPS..=6).contains(&plan.max_corrections)
        || plan.max_backtracks != HALVES
        || plan.max_candidate_guard_checks != CHECKS
        || plan.max_seconds != 600
    {
        return Err(invalid("relay fixed four-case budget changed"));
    }
    let source_bytes = plan.source_plan.bytes()?;
    let source: serde_json::Value = serde_json::from_slice(&source_bytes)?;
    let report: serde_json::Value = serde_json::from_slice(&plan.tape_report.bytes()?)?;
    if source["schema"] != "paisho-gen5-learner-tape-v1"
        || report["plan_sha256"] != sha256(&source_bytes)
        || report["reset_replay_every_sgd_and_boundary_exact"] != true
        || report["same_tape_all_arms"] != true
    {
        return Err(invalid("relay requires exact authenticated native tape"));
    }
    let manifest = load_json(&report["tape"])?;
    if manifest["plan_sha256"] != sha256(&source_bytes)
        || manifest["schema"] != "paisho-gen5-native-learning-tape-v1"
    {
        return Err(invalid("relay tape manifest binding"));
    }
    let options: Options = serde_json::from_slice(&read(&source["config"])?)?;
    let resume = load_json(&source["resume"])?;
    let initial = snapshot(
        &source["initial_actor"],
        resume["publication_guard"]["accepted_version"]
            .as_u64()
            .ok_or_else(|| invalid("relay initial version"))?,
        None,
    )?;
    let primary = options
        .publication_guard
        .as_ref()
        .ok_or_else(|| invalid("relay primary panel missing"))?;
    let validation = options
        .publication_validation
        .as_ref()
        .ok_or_else(|| invalid("relay validation panel missing"))?;
    if !options.learning_loop_v3 || options.value_policy_strength != 16. {
        return Err(invalid("relay requires original V3 control"));
    }
    fs::create_dir(out)?;
    write(
        &out.join("input-plan.json"),
        &serde_json::from_slice::<serde_json::Value>(&plan_bytes)?,
    )?;
    let (pool, _) = cpu::build_pool(options.threads, None)?;
    let (cohort, metadata) =
        loaded_cohort(&source, &manifest, &initial.model, &out.join("proofs"))?;
    let cohort_native = write(&out.join("cohort.json"), &metadata)?;
    if !plan.arbitrary_cases.is_empty() {
        if resume["publication_guard"]["manifest"] != sha256(&fs::read(primary)?)
            || resume["publication_guard"]["validation_manifest"] != sha256(&fs::read(validation)?) {
            return Err(invalid("continuation Guard panels changed"));
        }
        return run_cases(&plan, &plan_bytes, &options, &initial, &cohort, &metadata, &pool, out, begin);
    }
    let boundaries = report["arms"]["persistent"]["boundaries"]
        .as_array()
        .ok_or_else(|| invalid("persistent tape boundaries missing"))?;
    if boundaries.len() != 4 {
        return Err(invalid("relay requires four boundaries"));
    }
    let primary_sha = sha256(&fs::read(primary)?);
    let validation_sha = sha256(&fs::read(validation)?);
    if resume["publication_guard"]["manifest"] != primary_sha
        || resume["publication_guard"]["validation_manifest"] != validation_sha
        || boundaries.iter().any(|b| {
            b["publication"]["manifest"] != primary_sha
                || b["publication"]["validation_manifest"] != validation_sha
        })
    {
        return Err(invalid(
            "relay Guard panel content is not the frozen tape/resume content",
        ));
    }
    let mut models = vec![];
    for b in boundaries {
        models.push((
            report_model(b, "actor", &initial.model)?,
            report_model(b, "incoming", &initial.model)?,
        ));
    }
    // Fix all four selections from authenticated prior outputs before new fits.
    let mut selected = vec![];
    let mut start_reads = vec![];
    let mut selection = vec![];
    for bi in 0..4 {
        let seed = readings(&models[bi].0.model, &cohort)?;
        let teacher = readings(&models[bi].1.model, &cohort)?;
        exact_raw(
            &seed,
            bound_measurement(&report, &format!("persistent-{bi}-actor"))?,
            &metadata,
        )?;
        exact_raw(
            &teacher,
            bound_measurement(&report, &format!("persistent-{bi}-incoming"))?,
            &metadata,
        )?;
        let target = (0..64).find(|&i| {
            metadata[i]["first_block"].as_u64().unwrap() <= bi as u64
                && teacher[i].raw
                && !seed[i].raw
        });
        selection.push(serde_json::json!({"block":bi,"selected":target,"key":target.map(|i|metadata[i]["key"].clone()),
            "seed_actor":models[bi].0.path,"seed_bits":parameters(&models[bi].0.model),"teacher":models[bi].1.path,
            "teacher_bits":parameters(&models[bi].1.model),"teacher_all":teacher,"seed_all":seed}));
        selected.push(target);
        start_reads.push(seed);
    }
    let selection_input = write(&out.join("selection-before-fit.json"), &selection)?;
    let loading_seconds = begin.elapsed().as_secs_f64();
    let mut results = vec![];
    let mut applied = vec![];
    for bi in 0..4 {
        if begin.elapsed().as_secs_f64() > plan.max_seconds as f64 {
            return Err(invalid("relay fixed wall bound exceeded"));
        }
        let case_out = out.join(format!("block-{bi:03}"));
        fs::create_dir(&case_out)?;
        let anchor = if bi == 0 {
            initial.clone()
        } else {
            models[bi - 1].0.clone()
        };
        let seed = &models[bi].0;
        let setup = Instant::now();
        let mut guard = Guard::open(
            primary,
            &case_out.join("guard"),
            options.value_policy_strength,
            anchor.clone(),
            &serde_json::Value::Null,
        )?;
        guard.enable_v2(validation)?;
        guard.enable_v3()?;
        guard.enable_parallel(&[pool.clone()]);
        let guard_setup_seconds = setup.elapsed().as_secs_f64();
        let (after, detail) = if let Some(target) = selected[bi] {
            relay(&guard, &seed.model, &cohort, &start_reads[bi], target, plan.adaptive_counterexamples, plan.adaptive_kl, plan.max_corrections, None)?
        } else {
            (
                seed.model.as_ref().clone(),
                serde_json::json!({"accepted":false,"reason":"no eligible learned proof; no substitution","candidate_full_checks":0}),
            )
        };
        if !value_equal(&seed.model, &after) {
            return Err(invalid("relay applied V differs from published seed"));
        }
        let artifact = MicroArtifact::new(
            &after,
            boundaries[bi]["updates"]
                .as_u64()
                .ok_or_else(|| invalid("relay update counter"))?,
            serde_json::json!({"diagnostic_only":true,"kind":"proof-policy-relay","block":bi,"before":seed.identity}),
        );
        let after_file = write(&case_out.join("after.json"), &artifact)?;
        let row = serde_json::json!({"block":bi,"anchor":anchor.path,"anchor_bits":parameters(&anchor.model),"before":seed.path,"before_bits":parameters(&seed.model),
            "after":after_file,"after_bits":parameters(&after),"final_actor":after_file,"final_actor_bits":parameters(&after),"seed_preserved_on_failure":detail["accepted"]==true||parameters(&after)==parameters(&seed.model),
            "guard_setup_seconds":guard_setup_seconds,"detail":detail,"successor_cache":guard.diagnostic_successor_reads()});
        write(&case_out.join("report.json"), &row)?;
        results.push(row);
        applied.push(after);
    }
    // Only after ALL four fit outcomes are fixed: descriptive learned retention,
    // never an optimizer/selection criterion. Inputs are decoded once in RAM.
    let measured = Instant::now();
    let mut retention = vec![];
    for block in source["blocks"].as_array().unwrap() {
        let panel = load_json(&block["retention"])?;
        let i = &panel["union_saved"];
        let saved: Vec<SavedMicroExample> = serde_json::from_slice(&read(
            &serde_json::json!({"path":i["path"],"sha256":i["sha256"]}),
        )?)?;
        let rows = saved
            .iter()
            .map(|s| s.example_for_rules_with_trusted_q(RULES, true))
            .collect::<Result<Vec<_>>>()?;
        retention.push((panel, rows));
    }
    if retention.iter().map(|(_, r)| r.len()).sum::<usize>() != 999 {
        return Err(invalid("relay representative cohort changed"));
    }
    let mut retention_results = vec![];
    for (bi, after) in applied.iter().enumerate() {
        let before = bound_measurement(&report, &format!("persistent-{bi}-actor"))?;
        let reused = parameters(after) == parameters(&models[bi].0.model);
        let measured_after = if reused {
            before["retention"].clone()
        } else {
            retention_measure(after, &retention)?
        };
        retention_results.push(
            serde_json::json!({"block":bi,"before":before["retention"],"after":measured_after,
            "reused_identical_before":reused,"measurement_only_not_admission":true}),
        );
    }
    let retention_output = write(&out.join("retention.json"), &retention_results)?;
    let result = serde_json::json!({"schema":"paisho-gen5-tape-policy-relay-result-v1","plan_sha256":sha256(&plan_bytes),
        "source_report":plan.tape_report,"source_plan":plan.source_plan,"cohort":cohort_native,"selection_before_fit":selection_input,
        "loading_seconds":loading_seconds,"measurement_seconds":measured.elapsed().as_secs_f64(),"retention":retention_output,"seconds":begin.elapsed().as_secs_f64(),"results":results,"bank_loads":1,"new_games":0,"searches_during_optimization":0,
        "diagnostic_only":true,"scope":"raw acquired proof relay, not MCTS strength; actor/teacher records fixed before fitting; four independent cases",
        "followup":"paired coupled and production-MCTS8/256 on same64; representative retention losses; run separately after optimization without reselection"});
    write(&out.join("report.json"), &result)?;
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn relay_can_repair_old_choices_without_a_confidence_ratchet_after_acquisition() {
        let make = |raw, gap, mass| Reading {good:0,bad:Some(1),gap,mass,raw};
        let seed = vec![make(true,-0.1,0.6),make(false,0.4,0.25)];
        let acquired = make(true,-0.008,0.334);
        let repaired = vec![seed[0].clone(),make(true,-0.000015,0.331)];
        assert!(!local_ok(&acquired,&repaired[1],&seed,&repaired));
        assert!(acquired_repair_ok(&seed,&repaired,1,0.0104,0.000001));
        assert!(!acquired_repair_ok(&seed,&seed,1,0.0104,0.));
        assert!(!acquired_repair_ok(&seed,&repaired,1,0.01,0.02));
        assert!(!acquired_repair_ok(&seed,&repaired,1,0.01,f64::NAN));
        let mut lost=repaired;lost[0].raw=false;
        assert!(!acquired_repair_ok(&seed,&lost,1,0.01,0.));
    }
    #[test]
    fn relay_admission_needs_actual_new_choice_and_retains_old_choices() {
        let make = |raw, gap, mass| Reading {
            good: 0,
            bad: Some(1),
            gap,
            mass,
            raw,
        };
        let old = vec![make(true, -0.1, 0.6), make(false, 0.1, 0.4)];
        let good = vec![old[0].clone(), make(true, -0.01, 0.51)];
        assert!(full_admission(&[], &old, &good, 1));
        assert!(!full_admission(&["KL".into()], &old, &good, 1));
        assert!(!full_admission(&[], &old, &old, 1));
        let bad = vec![make(false, 0.1, 0.4), good[1].clone()];
        assert!(!full_admission(&[], &old, &bad, 1));
        assert!(!local_ok(&old[1], &make(false, 0.01, 0.39), &old, &old));
    }
    #[test]
    fn relay_support_and_ties_follow_native_first_index() {
        let r = reading(&[0.5, 0.5], &[false, true], &[]).unwrap();
        assert!(!r.raw);
        assert_eq!(r.gap, 0.);
        assert!(reading(&[0.499, 0.501], &[false, true], &[]).unwrap().raw);
        assert!(reading(&[0., 1.], &[true, false], &[]).is_err());
        let r = reading(&[0.6, 0.4], &[false, true], &[0., 1.]).unwrap();
        assert!(r.gap < 0.);
        assert!(!r.raw);
    }
    #[test]
    fn relay_joint_new_margin_prevents_old_projection_from_erasing_acquisition() {
        let direct = vec![1., 0.];
        let old = vec![vec![-1., 1.]];
        let projected =
            super::super::super::super::protection::diagnostic_affine(&direct, &old, &[0.])
                .unwrap();
        assert!((projected[0] - 0.5).abs() < 1e-12);
        assert!(projected[0] < 1.);
        let joint = super::super::super::super::protection::diagnostic_affine(
            &direct,
            &[old[0].clone(), vec![1., 0.]],
            &[0., 1.],
        )
        .unwrap();
        assert!(joint[0] >= 1. - 1e-12);
        assert!(-joint[0] + joint[1] >= -1e-12);
        assert!((joint[0] - 1.).abs() < 1e-12 && (joint[1] - 1.).abs() < 1e-12);
    }
    #[test]
    fn relay_step_preserves_all_value_bits_and_two_candidate_budget_is_fixed() {
        let model = MicroModel::seeded(13).with_deep_value(5);
        let step = vec![0.01; model.parameters().len()];
        let after = gain_probe::shifted(&model, &step, 1.).unwrap();
        assert!(value_equal(&model, &after));
        assert!(parameters(&model) != parameters(&after));
        assert_eq!((STEPS, HALVES, CHECKS), (4, 6, 2));
    }
}
