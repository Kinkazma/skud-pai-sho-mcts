//! Frozen tape base-policy reads only: PLAN NEW_OUTPUT_DIRECTORY.
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("PLAN NEW_OUTPUT_DIRECTORY".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::measure_tape_base_policy(
        std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))?;
    println!("{}",serde_json::json!({"diagnostic_only":true,"unique_parameter_sets":report["unique_parameter_sets"],
        "computed_base_rows":report["computed_base_rows"],"base_kernel_seconds":report["base_kernel_seconds"],
        "bank_loads":report["bank_loads"],"neural_forwards":report["neural_forwards"]}));
    Ok(())
}
