//! Four frozen proof-relay cases; no campaign or search.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("PLAN NEW_OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let r = paisho_train::micro_learning::gen5::probe_policy_relay(
        std::path::Path::new(&args[1]),
        std::path::Path::new(&args[2]),
    )?;
    println!(
        "{}",
        serde_json::json!({"seconds":r["seconds"],"results":r["results"].as_array().map(|v|v.iter().map(|v|&v["detail"]["accepted"]).collect::<Vec<_>>())})
    );
    Ok(())
}
