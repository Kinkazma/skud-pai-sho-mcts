//! Authenticated historical consumption for the bounded recurrent diagnostic.
//! Importing a consumed proof creates pending work, never an acquired choice.
use super::*;
use std::{collections::BTreeSet, io::Read};

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn bytes(&self) -> Result<Vec<u8>> {
        let bytes = fs::read(&self.path)?;
        if sha256(&bytes) != self.sha256 {
            return Err(invalid(format!("changed transfer bootstrap source {}", self.path.display())));
        }
        Ok(bytes)
    }
    fn json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::from_slice(&self.bytes()?)?)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Capsule {
    schema: String,
    cohort: Input,
    source_plan: Input,
    tape_report: Input,
    tape: Input,
    initial_actor: Input,
    initial_updates: u64,
    entries: Vec<Entry>,
    scope: String,
    native_rules_verification_by_this_preparer: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    index: usize,
    key: String,
    first_block: usize,
    proof: learned_choices::ConsumedProof,
    sources: Sources,
    native_proof: Input,
    saved_index: usize,
    source_consumption_after_initial: bool,
    source_measurement: bool,
    eligible_for_initial_registry: bool,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Sources {
    receipt: Input,
    psr: Input,
    targets: Input,
    bundle: Input,
}

fn consumption_matches(receipt: &serde_json::Value, updates: u64, cutoff: u64,
    measurement: bool, after_initial: bool, eligible: bool) -> bool
{
    updates > 0 && receipt["updates"].as_u64() == Some(updates)
        && receipt["fully_learned"] == true && receipt["error"].is_null()
        && receipt["learned_batches"].as_u64().is_some_and(|n| n > 0)
        && receipt["measurement"].as_bool() == Some(measurement)
        && after_initial == (updates > cutoff)
        && eligible == (updates <= cutoff)
}

impl Guard {
    pub fn diagnostic_bootstrap_transfer(&mut self, path: &Path, digest: &str,
        initial_updates: u64) -> Result<()>
    {
        let started = Instant::now();
        if !self.state.publication_transfer {
            return Err(invalid("transfer bootstrap requires enabled native transfer"));
        }
        let original = self.learned_choices.as_ref()
            .ok_or_else(|| invalid("transfer bootstrap has no registry"))?;
        let progress = original.progress();
        if progress["publications"] != 0 || progress["pending"] != 0 || progress["active"] != 0
            || original.last_validated_actor() != self.accepted.identity
        {
            return Err(invalid("transfer bootstrap requires the untouched initial registry"));
        }
        let input = Input { path: path.to_owned(), sha256: digest.to_owned() };
        let capsule: Capsule = serde_json::from_slice(&input.bytes()?)?;
        if capsule.schema != "paisho-gen5-authenticated-consumed-proof-bootstrap-v1"
            || capsule.initial_updates != initial_updates || capsule.entries.len() != 64
            || capsule.scope.is_empty() || capsule.native_rules_verification_by_this_preparer
        {
            return Err(invalid("transfer bootstrap schema, cohort or cutoff changed"));
        }
        let actor: MicroArtifact = serde_json::from_slice(&capsule.initial_actor.bytes()?)?;
        let accepted = self.accepted.artifact.as_ref()
            .ok_or_else(|| invalid("transfer bootstrap accepted actor lacks native artifact"))?;
        // The loop helper rewraps this exact model as actor-000. Its provenance,
        // and thus artifact identity, change; its parameters and memory may not.
        let same_bits = |a: &[f64], b: &[f64]| a.len() == b.len()
            && a.iter().zip(b).all(|(a,b)| a.to_bits() == b.to_bits());
        if actor.updates != initial_updates || accepted.updates != initial_updates
            || accepted.identity() != self.accepted.identity
            || actor.schema != accepted.schema || actor.feature_schema != accepted.feature_schema
            || actor.schema != self.accepted.model.schema()
            || actor.feature_schema != self.accepted.model.feature_schema()
            || actor.sequence_memory != accepted.sequence_memory
            || actor.sequence_memory != self.accepted.model.sequence_memory().map(|m| m.spec.clone())
            || !same_bits(&actor.parameters, self.accepted.model.parameters())
            || !same_bits(&actor.parameters, &accepted.parameters)
        {
            return Err(invalid("transfer bootstrap initial actor differs from accepted weights, memory or updates"));
        }
        let cohort = capsule.cohort.json()?;
        let source_plan = capsule.source_plan.json()?;
        let tape_report = capsule.tape_report.json()?;
        let tape = capsule.tape.json()?;
        let historical = cohort.as_array().ok_or_else(|| invalid("bootstrap cohort is not an array"))?;
        let tape_cohort = tape["proof_cohort"].as_array()
            .ok_or_else(|| invalid("bootstrap tape lacks fixed cohort"))?;
        if historical.len() != 64 || tape_cohort.len() != 64
            || tape_report["tape"] != serde_json::to_value(&capsule.tape)?
            || tape_report["plan_sha256"].as_str() != Some(capsule.source_plan.sha256.as_str())
            || tape["plan_sha256"].as_str() != Some(capsule.source_plan.sha256.as_str())
            || tape["schema"] != "paisho-gen5-native-learning-tape-v1"
        {
            return Err(invalid("transfer bootstrap source plan/tape/cohort binding differs"));
        }
        let blocks = source_plan["blocks"].as_array()
            .ok_or_else(|| invalid("transfer bootstrap source plan lacks blocks"))?;
        let mut keys = BTreeSet::new();
        let mut book = original.clone();
        let mut verified_rows = vec![];
        let mut entries = vec![];
        let mut pending = 0;
        let mut measurement_pending = vec![];
        for (i, entry) in capsule.entries.iter().enumerate() {
            let h = &historical[i];
            let t = &tape_cohort[i];
            let sources = serde_json::to_value(&entry.sources)?;
            let receipt_source = serde_json::to_value(&entry.sources.receipt)?;
            let block = blocks.get(entry.first_block)
                .ok_or_else(|| invalid("bootstrap proof first block missing"))?;
            let source_receipts = block["receipts"].as_array()
                .ok_or_else(|| invalid("bootstrap proof block lacks receipts"))?;
            if entry.index != i || !keys.insert(entry.key.clone())
                || entry.key != sha256(entry.proof.prefix.as_bytes())
                || h["index"].as_u64() != Some(i as u64) || h["key"] != entry.key
                || t["key"] != entry.key
                || h["first_block"].as_u64() != Some(entry.first_block as u64)
                || t["first_block"].as_u64() != Some(entry.first_block as u64)
                || h["sources"] != sources || h["receipt"] != receipt_source
                || t["receipt"] != receipt_source
                || h["native_proof"] != serde_json::to_value(&entry.native_proof)?
                || t["saved_index"].as_u64() != Some(entry.saved_index as u64)
                || t["decision"].as_u64() != Some(entry.proof.decision as u64)
                || source_receipts.iter().filter(|r| **r == sources).count() != 1
            {
                return Err(invalid(format!("bootstrap changed chronological source selection at {i}")));
            }
            let receipt = entry.sources.receipt.json()?;
            let saved = decode_examples(&entry.sources.targets.bytes()?)?;
            let row = saved.get(entry.saved_index)
                .ok_or_else(|| invalid("bootstrap selected Saved missing"))?;
            let evidence = row.evidence.as_ref()
                .ok_or_else(|| invalid("bootstrap Saved lacks decision evidence"))?;
            let native = entry.native_proof.json()?;
            let certificate = serde_json::to_value(&entry.proof.certificate)?;
            let psr = entry.sources.psr.bytes()?;
            let bundle_bytes = entry.sources.bundle.bytes()?;
            let mut bundle_json = vec![];
            flate2::read::GzDecoder::new(bundle_bytes.as_slice()).read_to_end(&mut bundle_json)?;
            let bundle: serde_json::Value = serde_json::from_slice(&bundle_json)?;
            let proofs = bundle["proofs"].as_array().ok_or_else(|| invalid("bootstrap bundle lacks proofs"))?;
            let certificates = proofs.iter().filter(|p| p[0].as_u64() == Some(entry.proof.decision as u64))
                .collect::<Vec<_>>();
            let source_run = entry.sources.receipt.path.parent().and_then(Path::parent)
                .ok_or_else(|| invalid("bootstrap receipt has no training source"))?;
            if !consumption_matches(&receipt, entry.proof.updates_consumed, initial_updates,
                entry.source_measurement, entry.source_consumption_after_initial, entry.eligible_for_initial_registry)
                || receipt["fresh_used"].as_u64() != Some(saved.len() as u64)
                || receipt["id"].as_u64() != Some(entry.proof.game_id)
                || receipt["collector"].as_str() != Some(entry.proof.collector.as_str())
                || receipt["prefix_decisions"].as_u64().map_or(true, |d| entry.proof.decision as u64 <= d)
                || row.game_id != entry.proof.game_id.to_string() || row.decision != entry.proof.decision
                || row.source_run != entry.proof.source_run || row.collector != entry.proof.collector
                || Path::new(&row.source_run).canonicalize()? != source_run.canonicalize()?
                || evidence.actor != entry.proof.decision_actor || evidence.player != entry.proof.player
                || !matches!(evidence.policy_source.as_str(), "verified-search" | "verified-regulatory-win")
                || row.policy_weight <= 0. || !row.policy_weight.is_finite()
                || native["prefix"].as_str() != Some(entry.proof.prefix.as_str())
                || native["certificate"] != certificate
                || bundle["schema"] != "paisho-gen5-durable-lessons-v1" || bundle["rules"] != RULES.as_str()
                || bundle["psr"].as_str().map(str::as_bytes) != Some(psr.as_slice())
                || bundle["psr_sha256"].as_str() != Some(entry.sources.psr.sha256.as_str())
                || bundle["game_id"].as_u64() != Some(entry.proof.game_id)
                || bundle["source"].as_str() != Some(entry.proof.collector.as_str())
                || certificates.len() != 1 || certificates[0][1] != certificate
            {
                return Err(invalid(format!("bootstrap consumption/certificate provenance mismatch at {i}")));
            }
            let adapter = serde_json::json!({"targets":entry.sources.targets.path,
                "targets_sha256":entry.sources.targets.sha256,"stride":1,
                "actor":receipt["collector"],"source_group":receipt["psr_sha256"]});
            let spec = serde_json::json!({"key":entry.key,"receipt_order":i,
                "target_index":entry.saved_index,"decision":entry.proof.decision,
                "proof":entry.native_proof,"targets":entry.sources.targets,
                "collector_identity":receipt["collector"],"source_group":receipt["psr_sha256"],
                "native_source":{"receipt":entry.sources.receipt,"psr":entry.sources.psr}});
            // Native replay checks full rule-winning support, Saved features/Q,
            // collector versus actual decision actor, seat/lane, PSR and hashes.
            let (witness, verified) = dynamic_proof_probe::transfer_witness(&spec, &self.accepted.model, &adapter)?;
            if verified["actions"] != h["verified"]["actions"]
                || verified["valid"] != h["verified"]["valid"]
                || verified["support_sha256"] != h["verified"]["support_sha256"]
            {
                return Err(invalid(format!("bootstrap native support changed at {i}")));
            }
            if entry.eligible_for_initial_registry {
                let admission = book.admit_consumed(entry.proof.clone(), true, &self.accepted.model)?;
                if !admission.inserted || admission.evicted_pending.is_some() {
                    return Err(invalid("bootstrap duplicated or evicted pending consumption"));
                }
                pending += 1;
                if entry.source_measurement { measurement_pending.push(i); }
            }
            verified_rows.push(witness);
            entries.push(serde_json::json!({"index":i,"key":entry.key,"pending":entry.eligible_for_initial_registry,
                "measurement":entry.source_measurement,"updates_consumed":entry.proof.updates_consumed,
                "proof":entry.proof,"sources":entry.sources,"native_proof":entry.native_proof,
                "saved_index":entry.saved_index,"native_verification":verified}));
        }
        // The complete old cohort stays a measurement. In particular its nine
        // later historical receipts never become pending on this new branch.
        let readings = policy_transfer::readings(&self.accepted.model, &verified_rows)?;
        let checkpoint = book.checkpoint()?;
        let report = serde_json::json!({"schema":"paisho-gen5-native-transfer-bootstrap-result-v1",
            "input":input,"initial_actor":capsule.initial_actor,"initial_identity":actor.identity(),
            "accepted_identity":self.accepted.identity,"initial_parameters_memory_updates_exact":true,
            "initial_updates":initial_updates,"verified":entries,"verified_count":entries.len(),
            "pending":pending,"excluded_later_consumption":entries.len()-pending,
            "measurement_pending_indices":measurement_pending,"publications":0,"acquisitions":0,
            "initial_raw64":readings.iter().filter(|r|r.raw).count(),"readings":readings,
            "registry":book.progress(),"seconds":started.elapsed().as_secs_f64(),
            "scope":"Authenticated prior source consumption only; pending proofs are not acquired choices; no SGD or publication performed."});
        durable::write(&self.out.join("transfer-bootstrap.json"), &report)?;
        // Commit only after every source, rule proof and exact actor binding has
        // passed. No failed import can leave a partially populated live registry.
        self.state.learned_choices = checkpoint;
        self.state.transfer = serde_json::json!({"bootstrap":report});
        self.learned_choices = Some(book);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_digest_rejects_changed_bytes() {
        let path = std::env::temp_dir().join(format!("gen5-transfer-auth-{}", std::process::id()));
        fs::write(&path, b"{\"consumed\":true}").unwrap();
        let input = Input { sha256: sha256(&fs::read(&path).unwrap()), path: path.clone() };
        assert_eq!(input.json().unwrap()["consumed"], true);
        fs::write(&path, b"{\"consumed\":false}").unwrap();
        assert!(input.bytes().is_err());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn actual_consumption_controls_cutoff_without_relabelling_measurement() {
        let receipt = serde_json::json!({"updates":19,"fully_learned":true,"error":null,
            "learned_batches":2,"measurement":true});
        // A measurement label cannot erase consumption that really happened.
        assert!(consumption_matches(&receipt, 19, 19, true, false, true));
        assert!(!consumption_matches(&receipt, 19, 19, false, false, true));
        // A later source belongs to the external cohort, never this bootstrap.
        assert!(consumption_matches(&receipt, 19, 18, true, true, false));
        assert!(!consumption_matches(&receipt, 19, 18, true, false, true));
        assert!(!consumption_matches(&receipt, 18, 19, true, false, true));
        for (field, wrong) in [("fully_learned", serde_json::json!(false)),
            ("error", serde_json::json!("interrupted")), ("learned_batches", serde_json::json!(0)),
            ("updates", serde_json::Value::Null)] {
            let mut changed = receipt.clone();
            changed[field] = wrong;
            assert!(!consumption_matches(&changed, 19, 19, true, false, true));
        }
    }
}
