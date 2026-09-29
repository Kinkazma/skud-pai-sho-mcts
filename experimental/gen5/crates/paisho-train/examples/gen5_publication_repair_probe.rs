//! No production writes: compare publication decisions for a frozen candidate.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    if a.len()!=5 {return Err("INITIAL CANDIDATE GUARD VALIDATION NEW_OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let r=paisho_train::micro_learning::gen5::verify_transactional_candidate(Path::new(&a[0]),Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]))?;
    println!("{}",serde_json::json!({"old_seconds":r["old_seconds"],"corrected_seconds":r["corrected_seconds"],"old_decision":r["old"]["last_decision"],"new_decision":r["corrected"]["last_decision"]}));
    Ok(())
}
