//! Diagnostic only: same frontier, optional balanced validation MSE, then actual Guard.
use std::path::Path;
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=2 {return Err("FRONTIER_MANIFEST NEW_OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::verify_validation_value_frontier(Path::new(&args[0]),Path::new(&args[1]))?;
    println!("{}",serde_json::json!({"runs":report["runs"].as_array().unwrap().iter().map(|r|serde_json::json!({
        "extra_validation_value":r["diagnostic_validation_value"],"consolidation_accepted":r["progress"]["last"]["accepted"],
        "consolidation_seconds":r["seconds"],"publication":r["publication"]})).collect::<Vec<_>>() }));
    Ok(())
}
