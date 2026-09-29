//! Independent bounded artifact verification. No model forward, bank load or game generation.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{gen5::RULES, MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn object(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    v.as_array().ok_or_else(|| "array missing".into())
}
fn number(v: &Value) -> Result<usize> {
    Ok(v.as_u64().ok_or("integer missing")? as usize)
}
fn string(v: &Value) -> Result<&str> {
    v.as_str().ok_or_else(|| "string missing".into())
}
fn checked_input(v: &Value) -> Result<Vec<u8>> {
    let b = fs::read(string(&v["path"])?)?;
    if hash(&b) != string(&v["sha256"])? {
        return Err("input hash mismatch".into());
    }
    Ok(b)
}
fn prefix(r: &GameRecord, n: usize) -> Result<GameRecord> {
    if n > r.actions().len() {
        return Err("prefix bounds".into());
    }
    let mut p = GameRecord::with_rules(r.setup(), r.rules());
    for a in &r.actions()[..n] {
        p.push(*a);
    }
    Ok(p)
}
fn flag(v: &Value) -> Result<bool> {
    match v {
        Value::Null => Ok(false),
        Value::Bool(b) => Ok(*b),
        _ => Err("boolean flag expected".into()),
    }
}
fn spec_source(v: &Value) -> Result<(GameRecord, GameOutcome)> {
    let b = checked_input(&v["input"])?;
    let r: GameRecord = std::str::from_utf8(&b)?.parse()?;
    let end = r.replay()?;
    let (r, end) = if r.rules() == RULES {
        (r, end)
    } else {
        r.replay_prefix_with_rules(RULES)?
    };
    Ok((r, end.outcome()))
}
fn checked_prefix(r: &GameRecord, v: &Value) -> Result<GameRecord> {
    let p = prefix(&r, number(&v["decisions"])?)?;
    if p.replay()?.outcome() != GameOutcome::Ongoing {
        return Err("terminal prefix".into());
    }
    Ok(p)
}
fn spec_prefix(v: &Value) -> Result<GameRecord> {
    checked_prefix(&spec_source(v)?.0, v)
}
fn host_observation(outcome: GameOutcome, source_hash: &str) -> Option<(f64, String)> {
    let value = match outcome {
        GameOutcome::Win(Player::Host) => 1.,
        GameOutcome::Win(Player::Guest) => -1.,
        GameOutcome::Draw => 0.,
        GameOutcome::Ongoing => return None,
    };
    Some((value, source_hash.to_owned()))
}
fn oriented_value(observation: &Option<(f64, String)>, player: Player) -> Option<f64> {
    observation.as_ref().map(|(value, _)| {
        if player == Player::Host {
            *value
        } else {
            -*value
        }
    })
}
fn verify_reanalysis_origin(
    receipt: &Value,
    spec: &Value,
    source: &GameRecord,
    outcome: GameOutcome,
    budget: usize,
) -> Result<Option<(f64, String)>> {
    let meta = &receipt["reanalysis_protocol"];
    let source_hash = hash(source.to_string().as_bytes());
    let expected = host_observation(outcome, &source_hash);
    if budget == 0
        || meta["reanalysis"] != true
        || meta["budgets"] != json!([[budget, 1.]])
        || meta["source_input_sha256"] != spec["input"]["sha256"]
        || meta["source_gen5_psr_sha256"] != source_hash
        || number(&meta["source_decisions"])? != source.actions().len()
        || meta["source_terminal_outcome"] != format!("{outcome:?}")
        || meta["observed_origin"] != serde_json::to_value(&expected)?
    {
        return Err("reanalysis observation differs from verified full source".into());
    }
    Ok(expected)
}
fn inside(run: &Path, path: &str) -> Result<PathBuf> {
    let p = Path::new(path).canonicalize()?;
    if !p.starts_with(run) {
        return Err("artifact escapes diagnostic run".into());
    }
    Ok(p)
}
fn replay(v: &Value, hash_field: &str, run: &Path) -> Result<GameRecord> {
    let path = inside(run, string(&v["psr"])?)?;
    let b = fs::read(path)?;
    if hash(&b) != string(&v[hash_field])? {
        return Err("PSR hash mismatch".into());
    }
    let r: GameRecord = std::str::from_utf8(&b)?.parse()?;
    if r.rules() != RULES {
        return Err("PSR rules mismatch".into());
    }
    let end = r.replay()?;
    if format!("{:?}", end.outcome()) != string(&v["outcome"])? {
        return Err("PSR result mismatch".into());
    }
    Ok(r)
}
fn verify_proof(v: &Value) -> Result<String> {
    if v["rules"] != RULES.as_str() {
        return Err("proof rules".into());
    }
    let text = string(&v["prefix"])?;
    let r: GameRecord = text.parse()?;
    if r.rules() != RULES {
        return Err("proof record rules".into());
    }
    let p = r.replay()?;
    let c: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
    c.verify(&p)?;
    Ok(hash(text.as_bytes()))
}
fn weights(path: &Path) -> Result<Vec<f64>> {
    let a: MicroArtifact = serde_json::from_slice(&fs::read(path)?)?;
    if a.parameters.len() != 292363 {
        return Err("unexpected diagnostic model shape".into());
    }
    Ok(a.parameters)
}
fn equal_bits(a: &[f64], b: &[f64]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 {
        return Err("RUN_DIR NEW_OUTPUT_JSON".into());
    }
    if Path::new(&a[2]).exists() {
        return Err("output exists".into());
    }
    let run = Path::new(&a[1]).canonicalize()?;
    let plan_bytes = fs::read(run.join("plan.json"))?;
    let report_bytes = fs::read(run.join("report.json"))?;
    let config_bytes = fs::read(run.join("input-config.json"))?;
    let plan: Value = serde_json::from_slice(&plan_bytes)?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let native_receipts = flag(&plan["immediate_lesson_admission"])?;
    if flag(&report["immediate_lesson_admission"])? != native_receipts {
        return Err("receipt protocol flag mismatch".into());
    }
    if native_receipts
        && (report["receipt_protocol"] != "all-fresh-native-memory-admission-no-warmup-v1"
            || report["recall_warmup_draws"] != 0)
    {
        return Err("native receipt protocol identity".into());
    }
    if plan["schema"] != "paisho-gen5-loop-cycle-probe-v1"
        || report["schema"] != "paisho-gen5-loop-cycle-result-v1"
        || plan["repaired"] != true
    {
        return Err("requires repaired loop diagnostic".into());
    }
    let cycles = array(&report["cycles"])?;
    let planned = array(&plan["cycles"])?;
    if !(2..=24).contains(&cycles.len()) || cycles.len() != planned.len() {
        return Err("cycle bounds".into());
    }
    for v in array(&report["inputs"])? {
        checked_input(v)?;
    }
    // State features depend on architecture, not learned weights or bank reads.
    let artifact: MicroArtifact = serde_json::from_slice(&checked_input(&plan["initial_model"])?)?;
    let model = MicroModel::from_parameters(artifact.parameters.clone())?;
    if !model.has_spatial() {
        return Err("spatial model required".into());
    }
    let mut initial_proofs = BTreeSet::new();
    for input in array(&plan["recall_proofs"])? {
        let v: Value = serde_json::from_slice(&checked_input(input)?)?;
        let key = verify_proof(&v)?;
        if Path::new(string(&input["path"])?)
            .file_stem()
            .and_then(|s| s.to_str())
            != Some(&key)
            || !initial_proofs.insert(key)
        {
            return Err("initial proof identity".into());
        }
    }
    let mut bundles = BTreeMap::new();
    let mut bundle_hashes = vec![];
    for entry in fs::read_dir(run.join("archive"))? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("gz") {
            continue;
        }
        let bytes = fs::read(&path)?;
        let h = hash(&bytes);
        if path.file_name().and_then(|s| s.to_str()) != Some(format!("{h}.json.gz").as_str()) {
            return Err("bundle filename/hash".into());
        }
        let b: Value = serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
        if b["schema"] != "paisho-gen5-durable-lessons-v1"
            || b["rules"] != RULES.as_str()
            || b["source_run"] != "gen5-loop-probe-training"
        {
            return Err("fresh bundle provenance".into());
        }
        let id = number(&b["game_id"])?;
        if bundles.insert(id, b).is_some() {
            return Err("duplicate bundle game".into());
        }
        bundle_hashes.push(json!({"path":path,"sha256":h}));
    }
    let mut receipts = 0;
    let mut target_count = 0;
    let mut selected_count = 0;
    let mut observed = 0;
    let mut support_count = 0;
    let mut support_actions = 0;
    let mut unknown_q = 0;
    let mut known_q = 0;
    let mut collection_proofs = 0;
    let mut collection_keys = BTreeSet::new();
    let mut receipt_rows = vec![];
    let mut reanalysis_origins = vec![];
    for (ci, cycle) in cycles.iter().enumerate() {
        if number(&cycle["cycle"])? != ci + 1 {
            return Err("cycle ordering".into());
        }
        let rows = array(&cycle["receipts"])?;
        let specs = array(&planned[ci])?;
        if rows.len() != specs.len() || rows.len() > 64 {
            return Err("receipt count".into());
        }
        for (ri, r) in rows.iter().enumerate() {
            let id = number(&r["id"])?;
            if id != ci * 1000 + ri {
                return Err("receipt id ordering".into());
            }
            let record = replay(r, "psr_sha256", &run)?;
            let (source, source_outcome) = spec_source(&specs[ri])?;
            let pref = checked_prefix(&source, &specs[ri])?;
            let reanalysis = flag(&r["reanalysis"])?;
            if reanalysis != flag(&specs[ri]["reanalysis"])? {
                return Err("planned reanalysis flag mismatch".into());
            }
            let reanalysis_observation = if native_receipts && reanalysis {
                let budget = number(&config["case_curriculum"]["reanalysis_budget"])?;
                let observed =
                    verify_reanalysis_origin(r, &specs[ri], &source, source_outcome, budget)?;
                reanalysis_origins.push(json!({"receipt":id,"budget":budget,
                    "source_input":specs[ri]["input"],"source_gen5_psr_sha256":hash(source.to_string().as_bytes()),
                    "source_outcome":format!("{source_outcome:?}"),"observed_origin":observed,"native_rules_verified":true}));
                observed
            } else {
                None
            };
            if prefix(&record, pref.actions().len())? != pref
                || r["source_group"] != specs[ri]["source_group"]
                || number(&r["new_decisions"])? != record.actions().len() - pref.actions().len()
            {
                return Err("training prefix binding".into());
            }
            let target_path = inside(&run, string(&r["targets"])?)?;
            if target_path != run.join(format!("cycle-{:03}/game-{ri:03}.targets.json.gz", ci + 1))
            {
                return Err("target cycle binding".into());
            }
            let bytes = fs::read(&target_path)?;
            if hash(&bytes) != string(&r["targets_sha256"])? {
                return Err("target hash".into());
            }
            let saved: Vec<SavedMicroExample> =
                serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
            if saved.len() != number(&r["all_fresh"])? {
                return Err("saved count".into());
            }
            let stride = saved
                .len()
                .div_ceil(number(&plan["max_fresh_per_game"])?)
                .max(1);
            let selected = saved.iter().step_by(stride).count();
            if stride != number(&r["stride"])? || selected != number(&r["selected_fresh"])? {
                return Err("fresh selection".into());
            }
            selected_count += selected;
            let b = bundles.remove(&id).ok_or("missing compact bundle")?;
            if b["psr"] != record.to_string()
                || b["psr_sha256"] != r["psr_sha256"]
                || b["source"] != r["actor"]
                || b["case"]["human_source"] != r["source_group"]
            {
                return Err("bundle receipt binding".into());
            }
            let lessons = array(&b["lessons"])?;
            if lessons.len() != saved.len() {
                return Err("compact lesson count".into());
            }
            let mut p = record.initial_position();
            let mut states = vec![p.clone()];
            for action in record.actions() {
                p.apply(*action)?;
                states.push(p.clone());
            }
            let outcome = p.outcome();
            if r["terminal"] != (outcome != GameOutcome::Ongoing) {
                return Err("terminal classification".into());
            }
            let mut certs = BTreeMap::new();
            for pair in array(&b["proofs"])? {
                let d = number(&pair[0])?;
                let state = states
                    .get(d.checked_sub(1).ok_or("proof decision zero")?)
                    .ok_or("proof decision bounds")?;
                let c: MicroProofCertificate = serde_json::from_value(pair[1].clone())?;
                c.verify(state)?;
                if certs.insert(d, c).is_some() {
                    return Err("duplicate proof decision".into());
                }
                collection_keys.insert(hash(prefix(&record, d - 1)?.to_string().as_bytes()));
                collection_proofs += 1;
            }
            if certs.len() != number(&r["certificates"])? {
                return Err("certificate count".into());
            }
            let mut game_observed = 0;
            let mut teachers = BTreeMap::<String, usize>::new();
            for (s, l) in saved.iter().zip(lessons) {
                let p = states
                    .get(s.decision.checked_sub(1).ok_or("target decision zero")?)
                    .ok_or("target decision bounds")?;
                let ex = s.example_for_rules_with_trusted_q(RULES, true)?;
                if s.source_run != "gen5-loop-probe-training"
                    || s.game_id != id.to_string()
                    || s.collector != string(&r["actor"])?
                    || !equal_bits(&s.state, &model.state_features(p))
                {
                    return Err("saved source/features".into());
                }
                let legal = legal_actions(p);
                let names = legal.iter().map(ToString::to_string).collect::<Vec<_>>();
                if !s.actions.is_empty() && s.actions != names {
                    return Err("legal action order".into());
                }
                for (action, features) in legal.iter().zip(&s.action_features) {
                    if !equal_bits(features, &micro_action_features(p, *action)) {
                        return Err("action features".into());
                    }
                }
                let e = s.evidence.as_ref().ok_or("evidence missing")?;
                if e.player
                    != if p.to_move() == Player::Host {
                        "H"
                    } else {
                        "G"
                    }
                {
                    return Err("evidence player differs from native position".into());
                }
                *teachers.entry(e.policy_source.clone()).or_default() += 1;
                let expected_observation = if reanalysis {
                    reanalysis_observation.clone()
                } else {
                    host_observation(outcome, string(&r["psr_sha256"])?)
                };
                let observed_value = oriented_value(&expected_observation, p.to_move());
                if e.observed_value != observed_value
                    || e.observed_psr != expected_observation.map(|(_, hash)| hash)
                {
                    return Err("observed outcome/source".into());
                }
                if observed_value.is_some() {
                    observed += 1;
                    game_observed += 1;
                }
                let dense = s
                    .actions
                    .iter()
                    .zip(&s.policy)
                    .filter(|(_, q)| **q > 0.)
                    .map(|(a, q)| json!([a, q]))
                    .collect::<Vec<_>>();
                if l["decision"] != s.decision
                    || l["policy"] != json!(dense)
                    || l["value"] != s.value
                    || l["policy_weight"] != s.policy_weight
                    || l["reason"] != s.reason
                    || l["collector"] != s.collector
                    || l.get("evidence").unwrap_or(&Value::Null)
                        != &serde_json::to_value(&s.evidence)?
                    || l.get("tactical").unwrap_or(&Value::Null)
                        != &serde_json::to_value(&s.tactical)?
                {
                    return Err("compact target differs from saved".into());
                }
                let proof_win = s.tactical.as_ref().is_some_and(|t| t.root_value == Some(1))
                    && s.policy_weight > 0.;
                if ex.policy_support != proof_win {
                    return Err("winning support admission".into());
                }
                if ex.policy_support {
                    let c = certs
                        .get(&s.decision)
                        .ok_or("support lacks collection certificate")?;
                    let win = if p.to_move() == Player::Host { 1 } else { -1 };
                    if c.outcome != win {
                        return Err("support root not a win".into());
                    }
                    for (i, action) in legal.iter().enumerate() {
                        let certified = c
                            .children
                            .iter()
                            .any(|(a, c)| a == &names[i] && c.outcome == win);
                        let mut next = p.clone();
                        next.apply(*action)?;
                        let proved = certified || next.outcome() == GameOutcome::Win(p.to_move());
                        if (s.policy[i] > 0.) != proved
                            || (proved && ex.action_values.get(i) != Some(&Some(1.)))
                        {
                            return Err("full regulatory support/Q mismatch".into());
                        }
                        support_actions += usize::from(proved);
                    }
                    support_count += 1;
                }
                // Reconstruct the V3 Q mask independently of the converter.
                let mut q = vec![None; s.actions.len()];
                if e.policy_source == "full-search-estimate"
                    && e.completed_action_values.len() == q.len()
                {
                    let visits = if e.action_value_visits.len() == q.len() {
                        &e.action_value_visits
                    } else {
                        &s.new_visits
                    };
                    if visits.len() == q.len() {
                        for i in 0..q.len() {
                            if visits[i] > 0 && !e.excluded_actions.get(i).copied().unwrap_or(false)
                            {
                                q[i] = Some(e.completed_action_values[i]);
                            }
                        }
                    }
                }
                if let Some(t) = s
                    .tactical
                    .as_ref()
                    .filter(|t| t.action_values.len() == q.len())
                {
                    for (q, p) in q.iter_mut().zip(&t.action_values) {
                        if let Some(p) = p {
                            *q = Some(*p as f64);
                        }
                    }
                }
                known_q += q.iter().flatten().count();
                unknown_q += q.iter().filter(|q| q.is_none()).count();
                if q.iter().all(Option::is_none) {
                    q.clear();
                }
                if q != ex.action_values {
                    return Err("trusted Q provenance mismatch".into());
                }
                target_count += 1;
            }
            if game_observed != number(&r["observed_targets"])?
                || serde_json::to_value(teachers)? != r["teachers"]
            {
                return Err("teacher/observed counts".into());
            }
            receipt_rows.push(json!({"id":id,"psr_sha256":r["psr_sha256"],"targets_sha256":r["targets_sha256"],"targets":saved.len(),"selected":selected}));
            receipts += 1;
        }
    }
    if !bundles.is_empty() {
        return Err("unaccounted fresh bundles".into());
    }
    let mut archive_proofs = vec![];
    let mut final_keys = BTreeSet::new();
    for entry in fs::read_dir(run.join("archive/proofs"))? {
        let path = entry?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let bytes = fs::read(&path)?;
        let v: Value = serde_json::from_slice(&bytes)?;
        let key = verify_proof(&v)?;
        if path.file_stem().and_then(|s| s.to_str()) != Some(&key)
            || !collection_keys.contains(&key)
            || !final_keys.insert(key.clone())
        {
            return Err("new proof identity/origin".into());
        }
        archive_proofs.push(
            json!({"key":key,"sha256":hash(&bytes),"also_initial":initial_proofs.contains(&key)}),
        );
    }
    if final_keys != collection_keys {
        return Err("collection proof not durable".into());
    }
    let specs = array(&plan["evaluation"])?;
    let mut prior = &report["initial_frozen_games"];
    let mut previous_weights = artifact.parameters;
    let mut physical = BTreeSet::new();
    let mut logical = 0;
    let mut reused = 0;
    for index in 0..=cycles.len() {
        let panel = if index == 0 {
            prior
        } else {
            &cycles[index - 1]["frozen_games"]
        };
        let is_reused =
            index > 0 && cycles[index - 1]["frozen_games_reused_unchanged_actor"] == true;
        if index > 0 {
            let current = weights(&run.join(format!("actor-{index:03}.json")))?;
            let changed = !equal_bits(&current, &previous_weights);
            if cycles[index - 1]["actor_weights_changed"] != changed || is_reused == changed {
                return Err("frozen reuse/weight consistency".into());
            }
            previous_weights = current;
        }
        if is_reused {
            if panel != prior {
                return Err("reused frozen panel changed".into());
            }
            reused += 1;
        }
        if panel["learned"] != 0 || array(&panel["rows"])?.len() != specs.len() * 2 {
            return Err("frozen evaluation shape".into());
        }
        let mut wdlu = [0usize; 4];
        for (i, row) in array(&panel["rows"])?.iter().enumerate() {
            let record = replay(row, "sha256", &run)?;
            let pref = spec_prefix(&specs[i / 2])?;
            let seat = if i % 2 == 0 {
                Player::Host
            } else {
                Player::Guest
            };
            if prefix(&record, pref.actions().len())? != pref
                || row["candidate_seat"] != format!("{seat:?}")
                || row["source_group"] != specs[i / 2]["source_group"]
                || number(&row["decisions"])? != record.actions().len() - pref.actions().len()
            {
                return Err("frozen prefix/seat".into());
            }
            let result = match record.replay()?.outcome() {
                GameOutcome::Win(w) if w == seat => 0,
                GameOutcome::Draw => 1,
                GameOutcome::Win(_) => 2,
                _ => 3,
            };
            if number(&row["result_index_wdlu"])? != result {
                return Err("frozen result classification".into());
            }
            wdlu[result] += 1;
            logical += 1;
            physical.insert(string(&row["psr"])?.to_owned());
        }
        if json!(wdlu) != panel["wdlu"] {
            return Err("frozen aggregate".into());
        }
        prior = panel;
    }
    for v in array(&report["inputs"])? {
        checked_input(v)?;
    }
    if fs::read(run.join("plan.json"))? != plan_bytes
        || fs::read(run.join("report.json"))? != report_bytes
        || fs::read(run.join("input-config.json"))? != config_bytes
    {
        return Err("run changed while verifying".into());
    }
    let out = json!({"verified":true,"run":run,"plan_sha256":hash(&plan_bytes),"report_sha256":hash(&report_bytes),"config_sha256":hash(&config_bytes),"native_receipt_protocol":native_receipts,"verified_reanalysis_origins":reanalysis_origins,"cycles":cycles.len(),"training_psr":receipts,"targets":target_count,"selected_fresh":selected_count,"observed_outcomes":observed,"support_examples":support_count,"verified_winning_actions":support_actions,"known_auxiliary_q":known_q,"unknown_auxiliary_q":unknown_q,"collection_certificates":collection_proofs,"new_unique_proofs":archive_proofs.len(),"initial_proofs":initial_proofs.len(),"initial_new_key_overlap":initial_proofs.intersection(&final_keys).count(),"frozen_logical_observations":logical,"frozen_unique_psr_paths":physical.len(),"frozen_reused_panels":reused,"receipts":receipt_rows,"bundles":bundle_hashes,"new_proofs":archive_proofs,"inputs_unchanged":true,"native_rule_replays":true,"model_forwards":0,"sequence_bank_loads":0,"new_games":0,"optimizer_updates":0,"limitations":["does not recompute MCTS estimates or optimizer trajectory","frozen reuse verified by exact model weights and copied panel, not regenerated matches","initial proof corpus is historical, not heldout"]});
    fs::write(&a[2], serde_json::to_vec_pretty(&out)?)?;
    println!(
        "{}",
        json!({"verified":true,"training_psr":receipts,"targets":target_count,"support_examples":support_count,"new_proofs":archive_proofs.len(),"initial_proofs":initial_proofs.len(),"frozen_logical":logical,"frozen_paths":physical.len(),"frozen_reused":reused})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empirical_origin_uses_the_full_verified_record_and_rejects_forged_metadata() {
        let text = include_str!("../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr");
        let dir = std::env::temp_dir().join(format!(
            "gen5-origin-verifier-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("source.psr");
        fs::write(&path, text).unwrap();
        let spec = json!({"input":{"path":path,"sha256":hash(text.as_bytes())},"decisions":0,"reanalysis":true});
        let (source, outcome) = spec_source(&spec).unwrap();
        assert_eq!(source.rules(), RULES);
        assert_ne!(outcome, GameOutcome::Ongoing);
        assert_eq!(
            checked_prefix(&source, &spec)
                .unwrap()
                .replay()
                .unwrap()
                .outcome(),
            GameOutcome::Ongoing
        );
        let source_hash = hash(source.to_string().as_bytes());
        let expected = host_observation(outcome, &source_hash);
        let receipt = json!({"reanalysis_protocol":{"reanalysis":true,"budgets":[[512,1.]],
            "source_input_sha256":spec["input"]["sha256"],"source_gen5_psr_sha256":source_hash,
            "source_decisions":source.actions().len(),"source_terminal_outcome":format!("{outcome:?}"),
            "observed_origin":expected}});
        assert_eq!(
            verify_reanalysis_origin(&receipt, &spec, &source, outcome, 512).unwrap(),
            expected
        );
        assert_eq!(
            oriented_value(&expected, Player::Guest),
            oriented_value(&expected, Player::Host).map(|v| -v)
        );
        for field in [
            "source_input_sha256",
            "source_gen5_psr_sha256",
            "observed_origin",
            "budgets",
        ] {
            let mut wrong = receipt.clone();
            wrong["reanalysis_protocol"][field] = Value::Null;
            assert!(
                verify_reanalysis_origin(&wrong, &spec, &source, outcome, 512).is_err(),
                "{field}"
            );
        }
        let mut wrong_input = spec.clone();
        wrong_input["input"]["sha256"] = json!("invalid");
        assert!(spec_source(&wrong_input).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn absent_legacy_flag_remains_false_and_nonterminal_sources_remain_unobserved() {
        assert!(!flag(&Value::Null).unwrap());
        assert!(!flag(&json!(false)).unwrap());
        assert!(flag(&json!("true")).is_err());
        let observation = host_observation(GameOutcome::Ongoing, "source");
        assert!(observation.is_none());
        assert!(oriented_value(&observation, Player::Host).is_none());
        assert!(oriented_value(&observation, Player::Guest).is_none());
        assert_eq!(
            host_observation(GameOutcome::Draw, "source"),
            Some((0., "source".into()))
        );
    }
}
