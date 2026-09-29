//! End-to-end diagnostic publication, with original C1 actors as prerequisites.
use super::*;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransferPlan {
    schema: String,
    #[serde(default)]
    family: Family,
    #[serde(default)]
    prefilter_abba: bool,
    capture_plan: Input,
    resume: Input,
    journal: Input,
    composed: Input,
    initial_actor: Input,
    expected_actors: Vec<ExpectedActor>,
    variants: Vec<Vec<Input>>,
    roles: Vec<String>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum ExpectedActor {
    Native(Input),
    Captured(CapturedActor),
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapturedActor {
    captured_anchor: Input,
    binding_census: Input,
}
impl ExpectedActor {
    fn input(&self) -> &Input {
        match self {
            Self::Native(i) => i,
            Self::Captured(c) => &c.captured_anchor,
        }
    }
    fn verify(
        &self,
        artifact: &MicroArtifact,
        bytes: usize,
        actor: &serde_json::Value,
    ) -> Result<()> {
        match self {
            Self::Native(_) => {
                if artifact.identity() != actor["identity"] {
                    return Err(invalid(
                        "native expected actor identity differs from journal",
                    ));
                }
            }
            Self::Captured(c) => {
                let census: serde_json::Value = serde_json::from_slice(&c.binding_census.bytes()?)?;
                if !captured_binding_matches(
                    &census,
                    &c.captured_anchor,
                    &artifact.provenance,
                    artifact.updates,
                    bytes,
                    actor,
                ) {
                    return Err(invalid(
                        "captured expected actor lacks explicit census/native snapshot provenance",
                    ));
                }
            }
        }
        Ok(())
    }
    fn recheck(&self) -> Result<()> {
        self.input().bytes()?;
        if let Self::Captured(c) = self {
            c.binding_census.bytes()?;
        }
        Ok(())
    }
}
fn captured_binding_matches(
    census: &serde_json::Value,
    input: &Input,
    provenance: &serde_json::Value,
    updates: u64,
    bytes: usize,
    actor: &serde_json::Value,
) -> bool {
    census["schema"] == "paisho-gen5-guard-model-census-plan-v1"
        && provenance["diagnostic_only"] == true
        && provenance["kind"] == "gen5-consolidation-capture"
        && provenance["role"] == "anchor"
        && provenance["context"]["accepted_actor_identity"] == actor["identity"]
        && provenance["context"]["accepted_actor_version"] == actor["version"]
        && provenance["context"]["accepted_actor_updates"] == actor["artifact_updates"]
        && actor["artifact_updates"] == updates
        && census["models"].as_array().is_some_and(|rows| {
            rows.iter()
                .filter(|v| {
                    v["kind"] == "actor"
                        && v["native_actor_identity"] == actor["identity"]
                        && v["path"] == input.path.to_string_lossy().as_ref()
                        && v["sha256"] == input.sha256
                        && v["file_bytes"] == bytes
                        && v["version"] == actor["version"]
                        && v["updates"] == updates
                        && v["artifact_identity_may_differ"] == true
                })
                .count()
                == 1
        })
}
#[derive(Clone, Copy, Default, Deserialize, Serialize)]
enum Family {
    #[default]
    #[serde(rename = "fresh-active-vs-continuous")]
    FreshActiveVsContinuous,
    #[serde(rename = "continuous-interior-vs-top2")]
    ContinuousInteriorVsTop2,
}
impl Family {
    fn roles(self) -> [&'static str; 3] {
        match self {
            Self::FreshActiveVsContinuous => ["original", "fresh-active", "fresh-continuous"],
            Self::ContinuousInteriorVsTop2 => [
                "original",
                "fresh-continuous-interior",
                "fresh-continuous-interior-top2",
            ],
        }
    }
    fn final_family(self) -> bool {
        matches!(self, Self::ContinuousInteriorVsTop2)
    }
    fn optional_legacy_false(self, value: &serde_json::Value, key: &str, expected: bool) -> bool {
        match value.get(key) {
            Some(v) => v.as_bool() == Some(expected),
            None => !self.final_family() && !expected,
        }
    }
    fn matches_leg(self, value: &serde_json::Value, second: bool) -> bool {
        value["fresh_active"] == true
            && value["continuous_choices"] == (self.final_family() || second)
            && self.optional_legacy_false(value, "fresh_interior", self.final_family())
            && self.optional_legacy_false(
                value,
                "two_choice_constraints",
                self.final_family() && second,
            )
    }
    fn matches_report(self, value: &serde_json::Value) -> bool {
        value["schema"] == "paisho-gen5-consolidation-capture-replay-v1"
            && value["all_four_originals_verified_before_any_variant"] == true
            && value["continuous_choice_comparison"] == true
            && self.optional_legacy_false(value, "fresh_interior_comparison", self.final_family())
            && self.optional_legacy_false(value, "two_choice_comparison", self.final_family())
            && value["cases"].as_array().is_some_and(|cases| {
                cases.len() == 4
                    && cases.iter().all(|case| {
                        case["legs"].as_array().is_some_and(|legs| {
                            legs.len() == 4
                                && legs
                                    .iter()
                                    .zip([false, true, true, false])
                                    .all(|(v, second)| self.matches_leg(v, second))
                        })
                    })
            })
    }
}

fn prefilter_signature(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Object(map) => serde_json::Value::Object(
            map.iter()
                .filter(|(k, _)| k.as_str() != "seconds" && k.as_str() != "policy_rows_computed")
                .map(|(k, v)| (k.clone(), prefilter_signature(v)))
                .collect(),
        ),
        serde_json::Value::Array(a) => {
            serde_json::Value::Array(a.iter().map(prefilter_signature).collect())
        }
        _ => value.clone(),
    }
}
fn transaction_signature(value: &serde_json::Value) -> Result<String> {
    let mut state = value["guard_state"].clone();
    state
        .as_object_mut()
        .ok_or_else(|| invalid("missing benchmark Guard state"))?
        .remove("accepted_path");
    let mut result = serde_json::Map::new();
    for key in [
        "observed",
        "candidate_parameter_bits",
        "adopted_parameter_bits",
        "candidate_to_adopted",
        "anchor_to_adopted",
        "before_publication_finite",
        "adopted_finite",
        "fresh_support_reads",
        "consolidation_pass_then_adopted_fresh_fail",
        "pre_feedback_rows",
        "returned_focus_rows",
        "pre_feedback_sha256",
        "returned_focus_sha256",
        "adopted_true_guard_against_original_anchor",
    ] {
        result.insert(key.into(), value[key].clone());
    }
    result.insert("guard_state".into(), state);
    result.insert(
        "logical_prefilter_trace".into(),
        prefilter_signature(&value["prefilter_profile"]),
    );
    Ok(sha256(&serde_json::to_vec(&result)?))
}

fn state_guard(
    case: &Case,
    plan: &Plan,
    o: &Options,
    out: &Path,
    pools: &[Arc<rayon::ThreadPool>],
    state: &serde_json::Value,
) -> Result<publication::Guard> {
    let identity = state["accepted_identity"]
        .as_str()
        .ok_or_else(|| invalid("publication state lacks actor identity"))?;
    let version = state["accepted_version"]
        .as_u64()
        .ok_or_else(|| invalid("publication state lacks actor version"))?;
    let path = Path::new(
        state["accepted_path"]
            .as_str()
            .ok_or_else(|| invalid("publication state lacks actor path"))?,
    );
    let initial = publication::load_snapshot(path, identity, version, &case.anchor)?;
    if !bits_equal(&initial.model, &case.anchor) {
        return Err(invalid(
            "pre-publication accepted actor differs from captured Protection anchor",
        ));
    }
    let mut g = publication::Guard::open(
        &plan.primary.path,
        out,
        o.value_policy_strength,
        initial,
        state,
    )?;
    g.enable_v2(&plan.validation.path)?;
    g.enable_v3()?;
    g.enable_parallel(pools);
    if encoded_rows(&g.reference_examples())? != encoded_rows(&case.references)? {
        return Err(invalid("publication corrective panel differs from capture"));
    }
    let panel = validation_value::ValidationValue::new(
        g.diagnostic_validation_examples()?,
        &case.anchor,
        Some(&cpu::Ordered::new(pools)),
    )?;
    if encoded_rows(&panel.rows)? != case.validation_bytes {
        return Err(invalid("publication secondary panel differs from capture"));
    }
    Ok(g)
}
fn candidate(case: &Case, model: &MicroModel, role: &str, out: &Path) -> Arc<Snapshot> {
    let version = case.report["context"]["version"].as_u64().unwrap();
    let artifact = Arc::new(MicroArtifact::new(
        model,
        case.report["context"]["updates_consumed"].as_u64().unwrap(),
        serde_json::json!({"diagnostic_only":true,"kind":"captured-publication-replay-candidate","role":role,"context":case.report["context"]}),
    ));
    Arc::new(Snapshot {
        identity: artifact.identity(),
        artifact: Some(artifact),
        version,
        model: Arc::new(model.clone()),
        path: out.join("candidate.json"),
    })
}
fn observed(state: &serde_json::Value) -> serde_json::Value {
    let decision = &state["last_decision"];
    let accepted = decision == "accepted-transaction";
    // Stale repair metadata is explicitly ignored on unchanged/rejected calls.
    serde_json::json!({"decision":decision,"checks":state["checks"],"rejected":state["rejected"],
        "accepted_branch":if accepted {state["accepted_branch"].clone()}else{serde_json::Value::Null},
        "accepted_fraction":if accepted {state["accepted_fraction"].clone()}else{serde_json::Value::Null},
        "repair":if accepted {state["repair"].clone()}else{serde_json::Value::Null}})
}
struct Published {
    actor: Arc<Snapshot>,
    report: serde_json::Value,
}
fn publish_one(
    g: &mut publication::Guard,
    case: &Case,
    model: &MicroModel,
    role: &str,
    out: &Path,
    pools: &[Arc<rayon::ThreadPool>],
) -> Result<Published> {
    let snapshot = candidate(case, model, role, out);
    let start = Instant::now();
    let pre_feedback = g.observe_candidate(&case.incoming)?;
    let observation_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let focus = g.consider(snapshot, true)?;
    let publication_seconds = start.elapsed().as_secs_f64();
    let actor = g.accepted();
    let state = g.progress();
    let verify = Instant::now();
    let ordered = cpu::Ordered::new(pools);
    let before_finite = finite(case, model, Some(&ordered))?;
    let after_finite = finite(case, &actor.model, Some(&ordered))?;
    // A fresh independent Guard anchored at the captured actor is checked by caller;
    // g is now reanchored, so its own finite method would hide relative regression.
    let support = fresh_support(case, model, &actor.model)?;
    let path = out.join("adopted.json");
    let artifact = MicroArtifact::new(
        &actor.model,
        case.report["context"]["updates_consumed"].as_u64().unwrap(),
        serde_json::json!({"diagnostic_only":true,"kind":"capture-publication-adopted","role":role,"context":case.report["context"],"snapshot_identity":actor.identity}),
    );
    save_json_new(&path, &artifact)?;
    let report = serde_json::json!({"role":role,"observed":observed(&state),"guard_state":state,
        "candidate_parameter_bits":bits_hash(model),"adopted_parameter_bits":bits_hash(&actor.model),
        "candidate_to_adopted":changed(model,&actor.model),"anchor_to_adopted":changed(&case.anchor,&actor.model),
        "before_publication_finite":before_finite,"adopted_finite":after_finite,"fresh_support_reads":support,
        "consolidation_pass_then_adopted_fresh_fail":before_finite["all_original_consolidation_criteria"]==true&&after_finite["fresh_pass"]==false,
        "pre_feedback_rows":pre_feedback.len(),"returned_focus_rows":focus.as_ref().map(|f|f.len()),
        "pre_feedback_sha256":sha256(&encoded_rows(&pre_feedback)?),
        "returned_focus_sha256":focus.as_ref().map(|f|encoded_rows(f).map(|b|sha256(&b))).transpose()?,
        "observation_seconds":observation_seconds,"publication_seconds":publication_seconds,
        "observation_plus_publication_seconds":observation_seconds+publication_seconds,
        "verification_and_save_seconds":verify.elapsed().as_secs_f64(),
        "adopted_model":{"path":path,"sha256":sha256(&fs::read(&path)?)},
        "private_candidate_identity_not_runtime_identity":true});
    Ok(Published { actor, report })
}
/// All four original publications must reproduce native adopted bits/decisions first.
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("publication replay output must be new"));
    }
    let start = Instant::now();
    let plan_bytes = fs::read(plan_path)?;
    let p: TransferPlan = serde_json::from_slice(&plan_bytes)?;
    if p.schema != "paisho-gen5-captured-publication-transfer-v1"
        || p.expected_actors.len() != 4
        || p.variants.len() != 4
        || p.variants.iter().any(|v| v.len() != 2)
        || p.roles != p.family.roles()
        || (p.prefilter_abba && !p.family.final_family())
    {
        return Err(invalid(
            "publication replay requires all fixed roles and four cases",
        ));
    }
    let capture_plan: Plan = serde_json::from_slice(&p.capture_plan.bytes()?)?;
    if capture_plan.reports.len() != 4
        || capture_plan.selection != "first-four-captures-by-check-no-replacement"
    {
        return Err(invalid("original capture selection changed"));
    }
    let options: Options = serde_json::from_slice(&capture_plan.config.bytes()?)?;
    if !options.learning_loop_v3
        || options.value_policy_strength != 16.
        || options.main_threads() != 10
        || options.search_pool_shards != 5
        || options.publication_guard.as_ref() != Some(&capture_plan.primary.path)
        || options.publication_validation.as_ref() != Some(&capture_plan.validation.path)
    {
        return Err(invalid("publication replay configuration changed"));
    }
    capture_plan.primary.bytes()?;
    capture_plan.validation.bytes()?;
    let resume: serde_json::Value = serde_json::from_slice(&p.resume.bytes()?)?;
    let initial_state = resume["publication_guard"].clone();
    let a: MicroArtifact = serde_json::from_slice(&p.initial_actor.bytes()?)?;
    if a.identity() != initial_state["accepted_identity"]
        || p.initial_actor.path
            != PathBuf::from(
                initial_state["accepted_path"]
                    .as_str()
                    .ok_or_else(|| invalid("missing restored actor path"))?,
            )
    {
        return Err(invalid("B1 initial actor binding changed"));
    }
    let journal_bytes = p.journal.bytes()?;
    let journal: Vec<serde_json::Value> = std::str::from_utf8(&journal_bytes)
        .map_err(|e| invalid(e.to_string()))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let boundaries = journal
        .iter()
        .filter(|r| r["kind"] == "boundary")
        .take(4)
        .collect::<Vec<_>>();
    if boundaries.len() != 4 {
        return Err(invalid("native first four publication boundaries absent"));
    }
    let composed: serde_json::Value = serde_json::from_slice(&p.composed.bytes()?)?;
    if !p.family.matches_report(&composed) {
        return Err(invalid("completed composed consolidation replay required"));
    }
    let mut cases = vec![];
    let mut expected = vec![];
    let mut variants = vec![];
    for (i, source) in capture_plan.reports.iter().enumerate() {
        let c = load(source, cases.first().map(|c: &Case| &c.anchor))?;
        if c.report["context"]["receipt_id"] != boundaries[i]["receipt_id"]
            || c.report["context"]["check"] != boundaries[i]["consolidation_checks"]
            || boundaries[i]["event_in_process"] != i
            || c.report["context"] != composed["cases"][i]["context"]
        {
            return Err(invalid(
                "captured and actual publication chronology mismatch",
            ));
        }
        let expected_bytes = p.expected_actors[i].input().bytes()?;
        let artifact: MicroArtifact = serde_json::from_slice(&expected_bytes)?;
        p.expected_actors[i].verify(
            &artifact,
            expected_bytes.len(),
            &boundaries[i]["actor_after"],
        )?;
        expected.push(model(&artifact, Some(&c.anchor))?);
        let mut vv = vec![];
        for (j, input) in p.variants[i].iter().enumerate() {
            let a: MicroArtifact = serde_json::from_slice(&input.bytes()?)?;
            let m = model(&a, Some(&c.anchor))?;
            let leg = &composed["cases"][i]["legs"][j];
            if a.provenance["kind"] != "capture-consolidation-replay"
                || a.provenance["case"] != i
                || a.provenance["leg"] != j
                || !p.family.matches_leg(leg, j == 1)
                || !p.family.matches_leg(&a.provenance, j == 1)
                || a.provenance["diagnostic_only"] != true
                || bits_hash(&m) != leg["anchor_to_applied"]["after"]
            {
                return Err(invalid(
                    "variant applied role does not match completed replay",
                ));
            }
            vv.push(m);
        }
        variants.push(vv);
        cases.push(c);
    }
    let pools = cpu::build_search_pools(10, 5, None)?;
    fs::create_dir(out)?;
    let setup = Instant::now();
    let mut original = state_guard(
        &cases[0],
        &capture_plan,
        &options,
        &out.join("original-guard"),
        &pools,
        &initial_state,
    )?;
    let original_guard_setup_seconds = setup.elapsed().as_secs_f64();
    let mut before_states = vec![];
    let mut baseline = vec![];
    // NO variant inference starts until every original transaction matches C1.
    for (i, case) in cases.iter().enumerate() {
        if !bits_equal(&original.accepted().model, &case.anchor) {
            return Err(invalid(
                "original replay anchor drifted from next captured block",
            ));
        }
        before_states.push(original.progress());
        let applied = match case.report["applied_role"].as_str() {
            Some("anchor") => &case.anchor,
            Some("attempted") => &case.attempted,
            _ => return Err(invalid("invalid original applied role")),
        };
        let leg = publish_one(
            &mut original,
            case,
            applied,
            "original",
            &out.join(format!("original-{i:02}")),
            &pools,
        )?;
        if !bits_equal(&leg.actor.model, &expected[i]) {
            return Err(invalid("original true publisher does not reproduce native C1 actor bits; no variant allowed"));
        }
        let observed = &leg.report["observed"];
        for key in [
            "decision",
            "checks",
            "rejected",
            "accepted_branch",
            "accepted_fraction",
            "repair",
        ] {
            if observed[key] != boundaries[i]["publication"][key] {
                return Err(invalid(format!(
                    "original publisher field {key} does not match native C1"
                )));
            }
        }
        save_json_new(
            &out.join(format!("original-verified-{i:02}.json")),
            &leg.report,
        )?;
        baseline.push(leg.report);
    }
    let mut blocks = vec![];
    for (i, case) in cases.iter().enumerate() {
        let fixed_start = Instant::now();
        let fixed = super::guard(
            case,
            &capture_plan,
            &options,
            &out.join(format!("fixed-validation-{i:02}")),
            &pools,
        )?;
        let fixed_guard_setup_seconds = fixed_start.elapsed().as_secs_f64();
        let mut original_report = baseline[i].clone();
        original_report["adopted_true_guard_against_original_anchor"] =
            fixed.diagnostic_finite_v3(&expected[i])?;
        if original_report["adopted_true_guard_against_original_anchor"]["accepted"] != true {
            return Err(invalid(
                "reproduced original actor failed original finite guard",
            ));
        }
        let mut legs = vec![original_report];
        let trials: Vec<(usize, bool)> = if p.prefilter_abba {
            vec![(1, false), (1, true), (1, true), (1, false)]
        } else {
            vec![(0, false), (1, false)]
        };
        let mut expected_signature = None;
        for (trial, (j, enabled)) in trials.into_iter().enumerate() {
            let model = &variants[i][j];
            let role = p.roles[j + 1].as_str();
            let path = out.join(if p.prefilter_abba {
                format!("case-{i:02}-abba-{trial}-{role}")
            } else {
                format!("case-{i:02}-{role}")
            });
            let setup = Instant::now();
            let mut guard = state_guard(
                case,
                &capture_plan,
                &options,
                &path.join("guard"),
                &pools,
                &before_states[i],
            )?;
            guard.diagnostic_parallel_prefilter(enabled, p.prefilter_abba);
            let setup_seconds = setup.elapsed().as_secs_f64();
            let mut leg = publish_one(&mut guard, case, model, role, &path, &pools)?;
            // Check relative to the initial captured anchor, not the newly adopted one.
            leg.report["adopted_true_guard_against_original_anchor"] =
                fixed.diagnostic_finite_v3(&leg.actor.model)?;
            if leg.report["adopted_true_guard_against_original_anchor"]["accepted"] != true {
                return Err(invalid(
                    "true publisher adopted a model failing its original finite criteria",
                ));
            }
            leg.report["guard_setup_seconds"] = serde_json::json!(setup_seconds);
            if p.prefilter_abba {
                // Profile hashes are computed after both transaction timers.
                leg.report["prefilter_profile"] = guard.diagnostic_prefilter_profile();
                let signature = transaction_signature(&leg.report)?;
                if let Some(expected) = &expected_signature {
                    if expected != &signature {
                        return Err(invalid(
                            "parallel prefilter changed full transaction/state/score/logical trace",
                        ));
                    }
                } else {
                    expected_signature = Some(signature.clone());
                }
                leg.report["transaction_signature"] = serde_json::json!(signature);
                leg.report["parallel_prefilter"] = serde_json::json!(enabled);
                leg.report["setup_observation_publication_seconds"] = serde_json::json!(
                    setup_seconds
                        + leg.report["observation_plus_publication_seconds"]
                            .as_f64()
                            .unwrap()
                );
            }
            legs.push(leg.report);
        }
        blocks.push(serde_json::json!({"check":case.report["context"]["check"],"capture":case.input,"legs":legs,
            "before_state_sha256":sha256(&serde_json::to_vec(&before_states[i])?),"fixed_guard_setup_seconds":fixed_guard_setup_seconds}));
    }
    p.capture_plan.bytes()?;
    p.resume.bytes()?;
    p.journal.bytes()?;
    p.composed.bytes()?;
    p.initial_actor.bytes()?;
    capture_plan.config.bytes()?;
    capture_plan.primary.bytes()?;
    capture_plan.validation.bytes()?;
    for input in &p.expected_actors {
        input.recheck()?;
    }
    for input in p.variants.iter().flatten() {
        input.bytes()?;
    }
    for c in &cases {
        c.input.bytes()?;
        for input in &c.dependencies {
            input.bytes()?;
        }
    }
    if fs::read(plan_path)? != plan_bytes {
        return Err(invalid("publication transfer plan changed"));
    }
    let report = serde_json::json!({"schema":if p.prefilter_abba {"paisho-gen5-prefilter-transaction-abba-v1"}else{"paisho-gen5-captured-publication-transfer-result-v1"},"prefilter_abba":p.prefilter_abba,"prefilter_order":if p.prefilter_abba {serde_json::json!([false,true,true,false])}else{serde_json::Value::Null},"all_prefilter_abba_transaction_signatures_exact":p.prefilter_abba,"plan_sha256":sha256(&plan_bytes),"family":p.family,"roles":p.roles,"blocks":blocks,
        "all_four_native_original_actors_and_decisions_exact_before_variants":true,"expected_actor_sources":p.expected_actors.iter().map(|s| match s {ExpectedActor::Native(i)=>serde_json::json!({"native_artifact":i}),ExpectedActor::Captured(c)=>serde_json::json!({"captured_anchor":c.captured_anchor,"binding_census":c.binding_census,"artifact_identity_is_not_snapshot_identity":true})}).collect::<Vec<_>>(),"original_guard_setup_seconds":original_guard_setup_seconds,
        "seconds_including_load_setup_verification_io":start.elapsed().as_secs_f64(),"one_bank_arc":true,"new_sgd_or_games":false,
        "scope":"single controlled diagnostic per fixed role; native weights and decisions reproduced, private candidate identities; no recovered in-RAM LRUs, no causal timing or campaign strength claim"});
    save_json_new(&out.join("report.json"), &report)?;
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_final_family_does_not_accept_old_comparison_or_missing_flags() {
        let legacy = serde_json::json!({"fresh_active":true,"continuous_choices":false});
        assert!(Family::FreshActiveVsContinuous.matches_leg(&legacy, false));
        assert!(!Family::ContinuousInteriorVsTop2.matches_leg(&legacy, false));
        let a = serde_json::json!({"fresh_active":true,"continuous_choices":true,"fresh_interior":true,"two_choice_constraints":false});
        let b = serde_json::json!({"fresh_active":true,"continuous_choices":true,"fresh_interior":true,"two_choice_constraints":true});
        assert!(Family::ContinuousInteriorVsTop2.matches_leg(&a, false));
        assert!(Family::ContinuousInteriorVsTop2.matches_leg(&b, true));
        assert!(!Family::ContinuousInteriorVsTop2.matches_leg(&a, true));
        assert!(!Family::FreshActiveVsContinuous.matches_leg(&a, true));
        let mut incomplete = a;
        incomplete
            .as_object_mut()
            .unwrap()
            .remove("two_choice_constraints");
        assert!(!Family::ContinuousInteriorVsTop2.matches_leg(&incomplete, false));
    }
    #[test]
    fn captured_actor_requires_native_anchor_provenance_and_census_hash() {
        let actor = serde_json::json!({"identity":"native","version":7,"artifact_updates":9});
        let input = Input {
            path: PathBuf::from("anchor.json"),
            sha256: "sha".into(),
        };
        let provenance = serde_json::json!({"diagnostic_only":true,"kind":"gen5-consolidation-capture","role":"anchor",
            "context":{"accepted_actor_identity":"native","accepted_actor_version":7,"accepted_actor_updates":9}});
        let mut census = serde_json::json!({"schema":"paisho-gen5-guard-model-census-plan-v1","models":[{
            "kind":"actor","native_actor_identity":"native","path":"anchor.json","sha256":"sha","file_bytes":42,
            "version":7,"updates":9,"artifact_identity_may_differ":true}]});
        assert!(captured_binding_matches(
            &census,
            &input,
            &provenance,
            9,
            42,
            &actor
        ));
        assert!(!captured_binding_matches(
            &census,
            &input,
            &provenance,
            10,
            42,
            &actor
        ));
        census["models"][0]["sha256"] = "other".into();
        assert!(!captured_binding_matches(
            &census,
            &input,
            &provenance,
            9,
            42,
            &actor
        ));
    }
    #[test]
    fn unchanged_never_reports_stale_repair() {
        let s = serde_json::json!({"last_decision":"unchanged","checks":7,"rejected":4,"accepted_branch":"full","accepted_fraction":0.5,"repair":{"steps":2}});
        let d = observed(&s);
        assert!(d["repair"].is_null());
        assert!(d["accepted_fraction"].is_null());
        assert!(d["accepted_branch"].is_null());
    }
}
