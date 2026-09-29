//! Frozen ABBA consolidation benchmark; output must be a new diagnostic folder.
use std::path::Path;
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a=std::env::args().skip(1).collect::<Vec<_>>();
    if a.len()!=5 {return Err("INITIAL CANDIDATE GUARD FRESH NEW_OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let r=paisho_train::micro_learning::gen5::verify_consolidation_cycle(Path::new(&a[0]),Path::new(&a[1]),Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]))?;
    println!("{}",serde_json::json!({"runs":r["runs"].as_array().unwrap().iter().map(|r|serde_json::json!({"corrected":r["corrected"],"seconds":r["seconds"],"accepted":r["progress"]["last"]["accepted"]})).collect::<Vec<_>>(),"deterministic":r["corrected_repeated_bits_exact"]}));
    Ok(())
}
