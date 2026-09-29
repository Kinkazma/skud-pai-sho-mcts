//! Input-bound diagnostic only; no games, SGD or production promotion.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("FROZEN_PLAN NEW_OUTPUT_DIRECTORY".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::benchmark_lazy_publication(std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))?;
    println!("{}",serde_json::json!({"all_exact":report["all_exact"],"total_seconds":report["total_seconds"]}));
    Ok(())
}
