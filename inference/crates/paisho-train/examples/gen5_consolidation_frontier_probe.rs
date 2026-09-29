//! Exact saved cycle12: vary only the numerical consolidation target.
use std::path::Path;
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=2 {return Err("FRONTIER_MANIFEST NEW_OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::verify_consolidation_frontier(Path::new(&args[0]),Path::new(&args[1]))?;
    println!("{}",serde_json::json!({"runs":report["runs"].as_array().unwrap().iter().map(|r|serde_json::json!({
        "target_tolerance_fraction":r["target_tolerance_fraction"],"accepted":r["progress"]["last"]["accepted"],
        "fresh_attempted":r["fresh_attempted_measured"],"fresh_applied":r["fresh_applied_measured"],
        "attempted_choices":r["attempted_choices"],"applied_choices":r["applied_choices"],"seconds":r["seconds"]})).collect::<Vec<_>>() }));
    Ok(())
}
