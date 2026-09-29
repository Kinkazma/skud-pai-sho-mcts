//! Explicit isolated native learning cycles; never a production campaign.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("CONFIG PLAN NEW_OUTPUT_DIRECTORY".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let result = paisho_train::micro_learning::gen5::probe_learning_cycles(
        std::path::Path::new(&args[1]),
        std::path::Path::new(&args[2]),
        std::path::Path::new(&args[3]),
    )?;
    println!(
        "{}",
        serde_json::json!({"completed_cycles": result["cycles"].as_array().map(Vec::len), "active_seconds":result["active_seconds"], "diagnostic_only":true})
    );
    Ok(())
}
