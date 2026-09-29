//! Opt-in diagnostic persistence only: no extra prediction, loss or gradient.
//! The existing consolidation callback retains one immutable attempted model.
use super::*;
use std::io::Write;

fn write_new(path: &Path, value: &impl Serialize) -> Result<serde_json::Value> {
    let bytes = serde_json::to_vec(value)?;
    let hash = sha256(&bytes);
    let mut file = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(serde_json::json!({"file":path.file_name().unwrap().to_string_lossy(),
        "sha256":hash,"bytes":bytes.len()}))
}
fn save_model(
    directory: &Path, role: &str, model: &MicroModel, updates: u64,
    context: &serde_json::Value,
) -> Result<serde_json::Value> {
    let artifact = MicroArtifact::new(model, updates, serde_json::json!({
        "diagnostic_only":true,"kind":"gen5-consolidation-capture","role":role,
        "context":context,"not_a_promoted_or_resume_selected_model":true}));
    // MicroArtifact::save calls model(), which can reload its external bank.
    // This already validated native model needs only serialization here.
    write_new(&directory.join(format!("{role}.json")), &artifact)
}
fn resume_rows<'a>(rows: impl Iterator<Item = &'a Arc<MicroExample>>) -> Vec<resume_example::ResumeExample> {
    rows.map(|e| resume_example::ResumeExample::from_with_trusted_q(e.as_ref(), true)).collect()
}

impl Protection {
    /// Called only by a due V3 boundary when the diagnostic option is enabled.
    /// No diagnostic files are read during resume; normal checkpoints stay intact.
    pub fn consolidate_diagnostic(
        &mut self, model: &mut MicroModel, output: &Path, version: u64,
        updates: u64, receipt_id: usize, accepted: &Snapshot,
    ) -> Result<()> {
        if !self.loop_v3 {
            return Err(invalid("consolidation capture requires transactional V3"));
        }
        let directory = output.join(format!("check-{:06}-v{version:07}-u{updates:010}-r{receipt_id:010}", self.checks + 1));
        // Created lazily, never during preload, zero-game preflight or idle checks.
        fs::create_dir_all(output)?;
        fs::create_dir(&directory)?;
        let anchor_updates = accepted.artifact.as_ref().map(|a| a.updates);
        let context = serde_json::json!({"rules":RULES.as_str(),"version":version,
            "updates_consumed":updates,"receipt_id":receipt_id,
            "check":self.checks+1,"accepted_actor_identity":accepted.identity,
            "accepted_actor_version":accepted.version,"accepted_actor_updates":anchor_updates});
        let anchor_file = save_model(&directory,"anchor",&self.anchor,
            anchor_updates.unwrap_or(0),&context)?;
        let incoming_file = save_model(&directory,"incoming",model,updates,&context)?;
        let fresh = write_new(&directory.join("fresh.json"), &resume_rows(self.fresh.iter()))?;
        let references = write_new(&directory.join("references.json"), &resume_rows(self.rows.iter()))?;
        let validation = self.validation_value.as_ref().map(|panel| {
            write_new(&directory.join("validation-value.json"), &resume_rows(panel.rows.iter()))
        }).transpose()?;
        let prepared = serde_json::json!({"schema":"paisho-gen5-consolidation-capture-v1",
            "status":"prepared","diagnostic_only":true,"context":context,
            "models":{"anchor":anchor_file,"incoming":incoming_file},
            "examples":{"fresh":fresh,"references":references,"validation_value":validation},
            "counts":{"fresh":self.fresh.len(),"references":self.rows.len(),
                "validation_value":self.validation_value.as_ref().map_or(0,|p|p.rows.len())},
            "example_format":"ResumeExample::from_with_trusted_q(example,true)",
            "example_provenance_limit":"Exact native optimizer inputs, not fabricated SavedMicroExample provenance; archive receipt linkage is separate.",
            "sequence_bank":"External immutable specification is included in each model; not reloaded or copied by capture.",
            "limits":{"finite_loss_tolerance":FINITE_LOSS_TOLERANCE,
                "correction_target_tolerance_fraction":0.5,
                "fresh_gain_retained_fraction":0.05,"fresh_loss_tolerance":1e-12,
                "fresh_ceiling":"old - 0.05*(old-incoming) if incoming<old; otherwise incoming",
                "maximum_correction_iterations":6,"maximum_line_search_halvings":5,
                "policy_gradient_constraint":self.policy_gradient_constraint},
            "anchor_state":{"losses":self.anchor_losses,"raw_choices":self.anchor_choices,
                "validation_value":self.validation_value.as_ref().map(|p|serde_json::json!({
                    "loss":p.anchor_loss,"class_counts":p.counts})),
                "reference_evaluations":self.reference_evaluations,
                "updates":self.updates,"checks":self.checks,"accepted":self.accepted}});
        write_new(&directory.join("prepared.json"), &prepared)?;
        fs::File::open(&directory)?.sync_all()?;
        let mut attempted = None;
        let result = self.consolidate_target(model, 0.5, |candidate| {
            // MicroModel clone shares its parameter, weight-view and bank Arcs.
            attempted = Some(candidate.clone());
        });
        let attempted_file = attempted.as_ref().map(|candidate| {
            save_model(&directory,"attempted",candidate,updates,&context)
        }).transpose()?;
        let mut report = prepared;
        report["status"] = if result.is_ok() {"complete"} else {"consolidation-error"}.into();
        report["models"]["attempted"] = attempted_file.unwrap_or(serde_json::Value::Null);
        report["result"] = if result.is_ok() {self.last.clone()} else {serde_json::Value::Null};
        report["applied_role"] = if result.is_err() {serde_json::Value::Null}
            else if self.last["accepted"] == true {"attempted".into()} else {"anchor".into()};
        report["error"] = result.as_ref().err().map(|e|e.to_string()).into();
        report["counters_after"] = serde_json::json!({"updates":self.updates,
            "checks":self.checks,"accepted":self.accepted,
            "reference_evaluations":self.reference_evaluations});
        report["capture_io_in_active_wall_time"] = true.into();
        write_new(&directory.join("report.json"), &report)?;
        fs::File::open(&directory)?.sync_all()?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bits(model: &MicroModel) -> Vec<u64> {
        model.parameters().iter().map(|v|v.to_bits()).collect()
    }
    fn value_example(value: f64) -> Arc<MicroExample> {
        Arc::new(MicroExample { structured: Vec::new(),state:vec![0.;128],actions:vec![],policy:vec![],
            action_values:vec![],policy_support:false,policy_weight:0.,value_weight:1.,
            value,sequence_source:91})
    }
    fn capture_case(reject: bool) {
        let mut parameters = vec![0.;MicroModel::seeded(1).parameters().len()];
        parameters[4160] = 0.3_f64.atanh();
        let anchor = MicroModel::from_parameters(parameters.clone()).unwrap();
        parameters[4160] = 0.4_f64.atanh();
        let incoming = MicroModel::from_parameters(parameters).unwrap();
        let rows = vec![value_example(if reject {0.} else {1.})];
        let fresh = vec![value_example(1.)];
        let mut a = Protection::new(&anchor,rows.clone()).unwrap();a.enable_loop_v3();a.observe(&fresh);
        let mut b = Protection::new(&anchor,rows).unwrap();b.enable_loop_v3();b.observe(&fresh);
        let mut direct = incoming.clone();let mut captured = incoming.clone();let mut last_attempt = None;
        a.consolidate_target(&mut direct,0.5,|model|last_attempt=Some(model.clone())).unwrap();
        let directory = std::env::temp_dir().join(format!("paisho-consolidation-capture-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let artifact = Arc::new(MicroArtifact::new(&anchor,73,serde_json::json!({"test":true})));
        let actor = Snapshot {identity:artifact.identity(),artifact:Some(artifact),version:11,
            model:Arc::new(anchor.clone()),path:directory.join("unwritten-actor.json")};
        b.consolidate_diagnostic(&mut captured,&directory,12,77,123,&actor).unwrap();
        assert_eq!(bits(&direct),bits(&captured));assert_eq!(bits(&a.anchor),bits(&b.anchor));
        assert_eq!(a.last,b.last);
        assert_eq!((a.updates,a.checks,a.accepted,a.reference_evaluations),
            (b.updates,b.checks,b.accepted,b.reference_evaluations));
        assert_eq!(a.last["accepted"],!reject);
        let paths=fs::read_dir(&directory).unwrap().map(|e|e.unwrap().path()).collect::<Vec<_>>();
        assert_eq!(paths.len(),1);let dir=&paths[0];
        let report:serde_json::Value=serde_json::from_slice(&fs::read(dir.join("report.json")).unwrap()).unwrap();
        assert_eq!(report["status"],"complete");assert_eq!(report["result"],a.last);
        assert_eq!(report["applied_role"],if reject {"anchor"} else {"attempted"});
        for (role,expected) in [("anchor",&anchor),("incoming",&incoming),("attempted",last_attempt.as_ref().unwrap())] {
            let bytes=fs::read(dir.join(format!("{role}.json"))).unwrap();
            assert_eq!(sha256(&bytes),report["models"][role]["sha256"].as_str().unwrap());
            let saved:MicroArtifact=serde_json::from_slice(&bytes).unwrap();
            assert_eq!(saved.parameters.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),bits(expected));
        }
        let saved:Vec<resume_example::ResumeExample>=serde_json::from_slice(&fs::read(dir.join("fresh.json")).unwrap()).unwrap();
        let restored=saved.into_iter().map(|e|e.example_with_trusted_q(true).unwrap()).collect::<Vec<_>>();
        assert_eq!(serde_json::to_vec(&resume_rows(restored.iter())).unwrap(),serde_json::to_vec(&resume_rows(fresh.iter())).unwrap());
        assert_eq!(a.fresh.len(),b.fresh.len());
        if reject {assert_eq!(report["result"]["fresh_after_measured"],true);assert_ne!(bits(last_attempt.as_ref().unwrap()),bits(&anchor));}
        // Diagnostics are not part of the native checkpoint or restore contract.
        let checkpoints=directory.join("checkpoints");fs::create_dir(&checkpoints).unwrap();
        b.checkpoint(&checkpoints).unwrap();
        let mut resumed=Protection::new(&captured,b.rows.clone()).unwrap();resumed.enable_loop_v3();
        resumed.restore(&b.progress()).unwrap();assert_eq!(bits(&resumed.anchor),bits(&b.anchor));
        assert_eq!(resumed.checks,b.checks);assert_eq!(resumed.fresh.len(),b.fresh.len());
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn diagnostic_capture_keeps_accepted_math_and_inputs_exact() {capture_case(false);}
    #[test]
    fn diagnostic_capture_retains_the_failed_fresh_gate_attempt_before_rollback() {capture_case(true);}
    #[test]
    fn diagnostic_capture_requires_v3_before_creating_files() {
        let model=MicroModel::seeded(3);let mut protection=Protection::new(&model,vec![value_example(0.)]).unwrap();
        let directory=std::env::temp_dir().join(format!("paisho-disabled-capture-{}",std::process::id()));
        let snapshot=Snapshot {identity:"unused".into(),artifact:None,version:0,model:Arc::new(model.clone()),path:directory.clone()};
        assert!(protection.consolidate_diagnostic(&mut model.clone(),&directory,0,0,0,&snapshot).is_err());
        assert!(!directory.exists());
    }

    #[test]
    fn diagnostic_capture_roundtrip_preserves_support_sparse_q_and_signed_zero() {
        let mut state=vec![0.;417];state[0]=-0.;state[127]=0.125;
        let mut a=[0.;32];a[7]=-0.;let mut b=[0.;32];b[2]=0.5;
        let example=Arc::new(MicroExample { structured: Vec::new(),state,actions:vec![a,b],policy:vec![0.5,0.5],
            action_values:vec![Some(-0.),None],policy_support:true,policy_weight:0.25,
            value_weight:0.75,value:-0.,sequence_source:u64::MAX-1});
        let original=resume_rows(std::iter::once(&example));
        let bytes=serde_json::to_vec(&original).unwrap();
        let rows:Vec<resume_example::ResumeExample>=serde_json::from_slice(&bytes).unwrap();
        let restored=rows.into_iter().next().unwrap().example_with_trusted_q(true).unwrap();
        assert!(restored.policy_support);assert_eq!(restored.sequence_source,example.sequence_source);
        assert_eq!(restored.action_values.len(),2);assert!(restored.action_values[1].is_none());
        assert_eq!(restored.action_values[0].unwrap().to_bits(),(-0_f64).to_bits());
        let inputs=|e:&MicroExample|e.state.iter().chain(e.actions.iter().flatten()).chain(e.policy.iter())
            .copied().chain([e.policy_weight,e.value_weight,e.value]).map(f64::to_bits).collect::<Vec<_>>();
        assert_eq!(inputs(&restored),inputs(&example));
        assert_eq!(serde_json::to_vec(&resume_rows(std::iter::once(&restored))).unwrap(),bytes);
    }
}
