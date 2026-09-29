//! Weighted reads only; no learning, search, FIFO or model promotion.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 {
        return Err("FROZEN_PLAN NEW_OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let r = paisho_train::micro_learning::gen5::measure_final_block_retention(
        std::path::Path::new(&a[1]),
        std::path::Path::new(&a[2]),
    )?;
    println!(
        "{}",
        serde_json::json!({"blocks":r["blocks"].as_array().map(|a|a.len()),"new_sgd_or_games":false})
    );
    Ok(())
}
