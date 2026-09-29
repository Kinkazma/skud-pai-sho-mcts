//! Fixed single-SGD displacement readings, no campaign or grid learning.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("PLAN NEW_OUTPUT_DIRECTORY".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::probe_tape_step_fractions(std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))?;
    println!("{}",serde_json::json!({"exact_replayed_steps":report["replayed_sgd_steps"],"seconds":report["total_seconds"],"grid_sgd_steps":0}));
    Ok(())
}
