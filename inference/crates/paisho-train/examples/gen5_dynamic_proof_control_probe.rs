use std::path::Path;
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("usage: gen5_dynamic_proof_control_probe PLAN.json NEW_OUTPUT_DIRECTORY".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::probe_dynamic_proof_control(Path::new(&args[1]),Path::new(&args[2]))?;
    println!("{}",serde_json::to_string(&serde_json::json!({"report":Path::new(&args[2]).join("report.json"),
        "six_proofs":report["all_six_actor000_to_actor001"],"old_guard_unchanged":report["old_guard_unchanged"]}))?);
    Ok(())
}
