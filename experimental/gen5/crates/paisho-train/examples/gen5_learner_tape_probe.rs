//! Isolated native tape; never launches a collector or a production campaign.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let preflight = args.len() == 4 && args[1] == "--preflight";
    if args.len() != 3 && !preflight {
        return Err("[--preflight] PLAN NEW_OUTPUT_DIRECTORY".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let offset = usize::from(preflight);
    let plan = std::path::Path::new(&args[1 + offset]);
    let out = std::path::Path::new(&args[2 + offset]);
    let result = if preflight {
        paisho_train::micro_learning::gen5::preflight_learner_tape(plan, out)?
    } else {
        paisho_train::micro_learning::gen5::probe_learner_tape(plan, out)?
    };
    println!(
        "{}",
        serde_json::json!({"diagnostic_only":true,"preflight":preflight,
        "same_tape_all_arms":result["same_tape_all_arms"],"reset_replay_every_sgd_and_boundary_exact":result["reset_replay_every_sgd_and_boundary_exact"],
        "seconds":result["seconds"]})
    );
    Ok(())
}
