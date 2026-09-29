//! Native read-only KL decomposition of the frozen frontier policy axis.
use std::path::Path;
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=3 {return Err("FRONTIER_MANIFEST VALIDATION339_REPORT NEW_OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::verify_kl_frontier(Path::new(&args[0]),Path::new(&args[1]),Path::new(&args[2]))?;
    println!("{}",serde_json::json!({"report":Path::new(&args[2]).join("report.json"),"fractions":report["runs"].as_array().unwrap().iter().map(|r|r["policy_fraction"].clone()).collect::<Vec<_>>()}));
    Ok(())
}
