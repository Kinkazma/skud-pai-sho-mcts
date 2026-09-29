//! Frozen ABBA root-FPU diagnostic. No training, proof import or runtime options.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const MANIFEST_SHAS: [&str; 2] = [
    "77c7d805470e6eb801562b93a20aa696aa692f8c0ec6c5c5f0112cc3198cec39",
    "488d31a70b77fad2c644d943b704d4349fcb29377f955f65cc433fbc37e3cd6b",
];
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn checked(spec: &Value) -> Result<Vec<u8>> {
    let bytes = fs::read(spec["path"].as_str().ok_or("input path")?)?;
    if sha(&bytes) != spec["sha256"].as_str().ok_or("input hash")? {
        return Err(format!("input hash mismatch: {}", spec["path"]).into());
    }
    Ok(bytes)
}
fn write_new(path: &Path, value: &Value) -> Result<()> {
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn append(f: &mut fs::File, value: &Value) -> Result<()> {
    serde_json::to_writer(&mut *f, value)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}
fn certificate_size(c: &MicroProofCertificate) -> (usize, usize) {
    c.children.iter().fold((1, 0), |(n, d), (_, child)| {
        let (nn, dd) = certificate_size(child);
        (n + nn, d.max(dd + 1))
    })
}
fn selected_in_certificate(c: &MicroProofCertificate, a: Action, sign: i8) -> bool {
    c.outcome == sign
        && c.children
            .iter()
            .any(|(s, child)| child.outcome == sign && s.parse::<Action>().ok() == Some(a))
}
fn run(manifest: Value, manifest_sha: &str, out: &Path, setup_started: Instant) -> Result<bool> {
    // The SHA freezes all roots, models and options; reject a silently edited protocol.
    let specs = manifest["models"].as_array().ok_or("models")?;
    let pos_specs = manifest["positions"].as_array().ok_or("positions")?;
    if specs.len() != 3 || pos_specs.len() != 125 {
        return Err("frozen counts".into());
    }
    let mut fixtures = Vec::new();
    for input in manifest["inputs"].as_array().ok_or("inputs")? {
        checked(input)?;
    }
    for spec in pos_specs {
        fixtures.push(serde_json::from_slice::<Value>(&checked(spec)?)?);
    }
    let input_checks_seconds = setup_started.elapsed().as_secs_f64();
    let load_started = Instant::now();
    let mut models: Vec<Arc<MicroModel>> = Vec::new();
    let mut identities = Vec::new();
    for spec in specs {
        let started = Instant::now();
        let artifact: MicroArtifact = serde_json::from_slice(&checked(spec)?)?;
        // MicroArtifact::model uses the native strong-Arc cache: only the first
        // identical bank spec reads/deserializes the bank. All model checks remain native.
        let model = Arc::new(artifact.model()?);
        if model.parameters().len() != 292363 {
            return Err("parameter count".into());
        }
        let bank = model.sequence_memory().ok_or("missing bank")?;
        if let Some(first) = models.first() {
            if !Arc::ptr_eq(first.sequence_memory().ok_or("first bank")?, bank) {
                return Err("models did not share the same immutable bank".into());
            }
        }
        identities.push(json!({"label":spec["label"], "sha256":spec["sha256"],
            "identity":artifact.identity(),"updates":artifact.updates,
            "load_seconds":started.elapsed().as_secs_f64(),"bank":artifact.sequence_memory}));
        models.push(model);
    }
    let load_seconds = load_started.elapsed().as_secs_f64();
    let options = MicroSearchOptions {
        mode: MicroSearchMode::Puct,
        seed: 37,
        dirichlet_fraction: 0.,
        dirichlet_total: 10.,
        gumbel_scale: 0.,
        considered_actions: 16,
        forced_playout_strength: 0.,
        proof_search: true,
    };
    let limit = manifest["options"]["max_active_seconds"]
        .as_f64()
        .ok_or("active limit")?;
    let mut journal = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("searches.jsonl"))?;
    let mut raw_journal = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("ungraded-searches.jsonl"))?;
    write_new(
        &out.join("started.json"),
        &json!({"manifest_sha256":manifest_sha,
        "identities":identities,"shared_bank":true,"input_checks_seconds":input_checks_seconds,
        "model_and_bank_load_seconds":load_seconds,"options":manifest["options"],
        "native_defaults":{"dirichlet_total":10.,"considered_actions":16,"exploration":1.5,
            "maximum_depth":96,"cache_entries":65536,"memory_limit_bytes":536870912},
        "order":"phases then repetitions and phase roots; model=(root_index+rotation)%3; budgets 8,32,256; false,true,true,false ABBA",
        "counter_scope":"inference_evaluations counts new state/value encodings (native Cache::get misses), not all policy/reader/neural-memory forwards",
        "active_limit_scope":"sum of native search_with_options kernel wall times; loading, replay, external grading, certificate export and output excluded"}),
    )?;
    let computation_started = Instant::now();
    let mut kernel_seconds = 0.;
    let mut grading_seconds = 0.;
    let mut replay_seconds = 0.;
    let mut export_seconds = 0.;
    let mut rows = Vec::new();
    let mut positions = Vec::new();
    let mut errors = 0;
    let mut completed = 0;
    let mut jobs = Vec::new();
    for phase in manifest["phases"].as_array().ok_or("phases")? {
        for repetition in 0..phase["repetitions"].as_u64().ok_or("repetitions")? as usize {
            for index in phase["positions"].as_array().ok_or("phase positions")? {
                jobs.push((
                    phase["name"].as_str().ok_or("phase name")?.to_owned(),
                    repetition,
                    index.as_u64().ok_or("position index")? as usize,
                ));
            }
        }
    }
    for (job, (phase, repetition, pi)) in jobs.into_iter().enumerate() {
        let spec = &pos_specs[pi];
        let replay_started = Instant::now();
        let prefix = fixtures[pi]["prefix"].as_str().ok_or("prefix")?;
        let record: GameRecord = prefix.parse()?;
        let p = record.replay()?;
        if sha(record.to_string().as_bytes())
            != spec["prefix_sha256"].as_str().ok_or("prefix hash")?
            || fixtures[pi]["rules"].as_str() != Some(p.rule_profile().as_str())
            || p.rule_profile() != RuleProfileId::SkudPaiShoGen5V1
            || p.outcome() != GameOutcome::Ongoing
        {
            return Err("invalid frozen root".into());
        }
        replay_seconds += replay_started.elapsed().as_secs_f64();
        let mut root_rows: Vec<(Value, Option<Action>, Option<MicroProofCertificate>)> = Vec::new();
        for rotation in 0..3 {
            let mi = (pi + rotation) % 3;
            for budget in [8usize, 32, 256] {
                for (abba_slot, enabled) in [false, true, true, false].into_iter().enumerate() {
                    let mut row = json!({"phase":phase,"repetition":repetition,"abba_slot":abba_slot,
                    "successor_fpu_enabled":enabled,"position":pi,"union_position":spec["union_position"],
                    "prefix_sha256":spec["prefix_sha256"],"cohort":spec["cohort"],
                    "metric_stratum":spec["metric_stratum"],"model":mi,"model_label":specs[mi]["label"],
                    "budget":budget,"search_ordinal":rows.len()+root_rows.len(),
                    "known_full_support_selected":null,"fresh_certificate_selected":null,
                    "verified_win_selected":null});
                    if kernel_seconds >= limit {
                        row["status"] = json!("not_started_active_limit");
                        row["kernel_seconds"] = json!(0.);
                        root_rows.push((row, None, None));
                        continue;
                    }
                    let mut session = MicroMctsSession::new(models[mi].clone());
                    session.set_root_value_strength(16.)?;
                    session.set_root_successor_fpu(enabled);
                    if session.retained_visits() != 0 {
                        return Err("nonempty new tree".into());
                    }
                    let allowed = limit - kernel_seconds;
                    let deadline =
                        paisho_platform::training_time::now() + Duration::from_secs_f64(allowed);
                    let started = Instant::now();
                    let searched = session.search_with_options(&p, budget, Some(deadline), options);
                    let elapsed = started.elapsed().as_secs_f64();
                    kernel_seconds += elapsed;
                    row["kernel_seconds"] = json!(elapsed);
                    match searched {
                        Err(error) => {
                            errors += 1;
                            row["status"] = json!("native_search_error");
                            row["error"] = json!(error);
                            root_rows.push((row, None, None));
                        }
                        Ok(r) => {
                            let selected =
                                *r.actions.get(r.selected_index).ok_or("selected index")?;
                            let valid_cold = r.inherited_visits == 0 && r.simulations <= budget;
                            let complete =
                                valid_cold && (r.simulations == budget || r.proven_value.is_some());
                            if !valid_cold {
                                errors += 1;
                            }
                            if complete {
                                completed += 1;
                            }
                            row["status"] = json!(if !valid_cold {
                                "cold_tree_or_budget_violation"
                            } else if complete {
                                "complete"
                            } else {
                                "incomplete_active_limit"
                            });
                            row["root_successor_fpu_cache_available"] =
                                json!(session.root_successor_fpu_values().is_some());
                            row["unvisited_root_fpu"] = json!(session
                                .root_successor_fpu_values()
                                .map(|q| q.to_vec())
                                .unwrap_or_else(|| vec![r.network_value; r.actions.len()]));
                            row["raw_priors"] = json!(r.raw_priors.as_ref().map(|q| q.as_slice()));
                            row["coupled_priors"] = json!(r.priors);
                            row["search_priors"] = json!(r.search_priors);
                            row["native_reported_action_values"] = json!(r.values);
                            if enabled
                                && r.memory_reset
                                && session.root_successor_fpu_values().is_none()
                            {
                                row["unvisited_root_fpu"] = Value::Null;
                            }
                            row["selected"] = json!(selected.to_string());
                            row["selected_index"] = json!(r.selected_index);
                            row["simulations"] = json!(r.simulations);
                            row["root_visits"] = json!(session.retained_visits());
                            row["new_root_child_visits"] =
                                json!(r.new_visits.iter().sum::<usize>());
                            row["inherited_visits"] = json!(r.inherited_visits);
                            row["tactical_evaluations"] = json!(r.tactical_evaluations);
                            row["inference_evaluations"] = json!(r.inference_evaluations);
                            row["inference_cache_hits"] = json!(r.inference_cache_hits);
                            row["retained_bytes"] = json!(r.retained_bytes);
                            row["memory_reset"] = json!(r.memory_reset);
                            row["proven_value"] = json!(r.proven_value);
                            row["selected_proven_value"] =
                                json!(r.proven_action_values[r.selected_index]);
                            row["network_value"] = json!(r.network_value);
                            row["selected_search_value_estimate"] =
                                json!(r.values[r.selected_index]);
                            row["actions"] = json!(r
                                .actions
                                .iter()
                                .map(ToString::to_string)
                                .collect::<Vec<_>>());
                            row["visits"] = json!(r.visits);
                            row["new_visits"] = json!(r.new_visits);
                            row["new_forced_visits"] = json!(r.new_forced_visits);
                            row["pruned_visits"] = json!(r.pruned_visits);
                            row["proven_action_values"] = json!(r.proven_action_values);
                            let started = Instant::now();
                            let fresh = session.certificate(10000);
                            export_seconds += started.elapsed().as_secs_f64();
                            root_rows.push((row, Some(selected), fresh));
                        }
                    }
                }
            }
        }
        for (row, _, _) in &root_rows {
            append(&mut raw_journal, row)?;
        }
        // Only now, after all choices for this root, inspect/verify its reference
        // certificate and enumerate external support. Neither reaches a search session.
        let grade_started = Instant::now();
        let cert: MicroProofCertificate =
            serde_json::from_value(fixtures[pi]["certificate"].clone())?;
        if cert.verify(&p)? != GameOutcome::Win(p.to_move()) {
            return Err("reference not winning".into());
        }
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let legal = legal_actions(&p);
        let mut immediate = Vec::new();
        let mut support = Vec::new();
        for a in &legal {
            let mut next = p.clone();
            next.apply(*a)?;
            if next.outcome() == GameOutcome::Win(p.to_move()) {
                immediate.push(*a);
            }
            if immediate.contains(a) || selected_in_certificate(&cert, *a, sign) {
                support.push(*a);
            }
        }
        if (immediate.is_empty()) != (spec["metric_stratum"] == "nonimmediate_primary") {
            return Err("frozen support stratum mismatch".into());
        }
        let (nodes, depth) = certificate_size(&cert);
        positions.push(json!({"phase":phase,"repetition":repetition,"position":pi,"union_position":spec["union_position"],
            "legal":legal.len(),"immediate_actions":immediate.len(),"full_support_actions":support.len(),
            "reference_certificate_nodes":nodes,"reference_certificate_depth":depth}));
        for (row, selected, fresh) in &mut root_rows {
            let Some(selected) = selected else {
                continue;
            };
            row["known_full_support_selected"] = json!(support.contains(selected));
            row["immediate_selected"] = json!(immediate.contains(selected));
            let mut fresh_support = false;
            if let Some(c) = fresh {
                match c.verify(&p) {
                    Ok(_) => {
                        row["fresh_certificate_valid"] = json!(true);
                        fresh_support = selected_in_certificate(c, *selected, sign);
                    }
                    Err(error) => {
                        errors += 1;
                        row["fresh_certificate_valid"] = json!(false);
                        row["fresh_certificate_error"] = json!(error);
                    }
                }
                row["fresh_certificate"] = serde_json::to_value(c)?;
            } else {
                row["fresh_certificate_valid"] = Value::Null;
            }
            row["fresh_certificate_selected"] = json!(fresh_support);
            row["verified_win_selected"] = json!(support.contains(selected) || fresh_support);
        }
        grading_seconds += grade_started.elapsed().as_secs_f64();
        for (row, _, _) in root_rows {
            append(&mut journal, &row)?;
            rows.push(row);
        }
        if job % 16 == 0 {
            eprintln!(
                "roots {}, complete searches {}, kernel {:.3}s",
                job + 1,
                completed,
                kernel_seconds
            );
        }
    }
    let mut scores = Vec::new();
    for phase in ["full_cohort", "focused_repeated"] {
        for enabled in [false, true] {
            for stratum in ["all", "nonimmediate_primary", "immediate_guard_control"] {
                for mi in 0..3 {
                    for budget in [8usize, 32, 256] {
                        let group: Vec<_> = rows
                            .iter()
                            .filter(|r| {
                                r["phase"] == phase
                                    && r["successor_fpu_enabled"] == enabled
                                    && r["model"] == mi
                                    && r["budget"] == budget
                                    && (stratum == "all" || r["metric_stratum"] == stratum)
                            })
                            .collect();
                        scores.push(json!({"phase":phase,"successor_fpu_enabled":enabled,"stratum":stratum,
            "model":mi,"budget":budget,"observations":group.len(),
            "complete":group.iter().filter(|r|r["status"]=="complete").count(),
            "native_own_verified_wins":group.iter().filter(|r|r["verified_win_selected"]==true).count(),
            "kernel_seconds":group.iter().filter_map(|r|r["kernel_seconds"].as_f64()).sum::<f64>()}));
                    }
                }
            }
        }
    }
    let successful = errors == 0 && completed == 5364;
    write_new(
        &out.join("report.json"),
        &json!({"schema":"paisho-gen5-root-fpu-probe-result-v1",
        "manifest_sha256":manifest_sha,"complete":successful,"planned_searches":5364,
        "completed_searches":completed,"errors":errors,"identities":identities,"shared_bank":true,
        "timings":{"input_checks_seconds":input_checks_seconds,"model_and_bank_load_seconds":load_seconds,
            "kernel_seconds":kernel_seconds,"prefix_replay_seconds":replay_seconds,
            "certificate_export_seconds":export_seconds,"external_grading_seconds":grading_seconds,
            "post_load_wall_seconds":computation_started.elapsed().as_secs_f64()},
        "phases":manifest["phases"],"positions":positions,"scores":scores,"results":rows,"new_games":0,"learning_updates":0,
        "installed_evaluation_certificates":0,"reserved_final_sources_read":0,
        "unknown_action_is_not_proven_losing":true,"counter_scope":"See started.json"}),
    )?;
    Ok(successful)
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("MANIFEST NEW_OUTPUT_DIRECTORY".into());
    }
    let setup_started = Instant::now();
    let bytes = fs::read(&args[1])?;
    let manifest_sha = sha(&bytes);
    if !MANIFEST_SHAS.contains(&manifest_sha.as_str()) {
        return Err("manifest differs from frozen protocol".into());
    }
    let manifest: Value = serde_json::from_slice(&bytes)?;
    let out = Path::new(&args[2]);
    fs::create_dir(out)?;
    fs::write(out.join("manifest.json"), &bytes)?;
    let exe = std::env::current_exe()?;
    write_new(
        &out.join("runtime.json"),
        &json!({
        "binary_path":exe,"binary_sha256":sha(&fs::read(&exe)?),
        "source_sha256":sha(include_bytes!("gen5_root_fpu_probe.rs")),
        "arch":std::env::consts::ARCH,"os":std::env::consts::OS,
        "rayon_threads":1,"VECLIB_MAXIMUM_THREADS":std::env::var("VECLIB_MAXIMUM_THREADS").ok()}),
    )?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    match run(manifest, &manifest_sha, out, setup_started) {
        Ok(true) => Ok(()),
        Ok(false) => Err("incomplete or erroneous result preserved in report.json".into()),
        Err(error) => {
            write_new(
                &out.join("fatal.json"),
                &json!({"error":error.to_string(),
                "manifest_sha256":manifest_sha,"completed_roots_preserved_in":"searches.jsonl",
                "pre_grading_choices_preserved_in":"ungraded-searches.jsonl"}),
            )?;
            Err(error)
        }
    }
}
