//! Export hashed C1 fresh sources into native trusted recovery examples, no models.
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 {
        return Err("BLOCK_MANIFEST NEW_OUTPUT".into());
    }
    let report = paisho_train::micro_learning::gen5::export_block_retention(
        Path::new(&args[0]),
        Path::new(&args[1]),
    )?;
    println!(
        "{}",
        serde_json::json!({"rows":report["blocks"].as_array().unwrap().iter().map(|b|b["rows"].clone()).collect::<Vec<_>>(),"model_calculations":0})
    );
    Ok(())
}
