//! Four preselected native consolidation captures; no FIFO/SGD/game/promotion.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 {
        return Err("FROZEN_PLAN NEW_OUTPUT_DIRECTORY".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let result = paisho_train::micro_learning::gen5::replay_captured_consolidation(
        std::path::Path::new(&a[1]),
        std::path::Path::new(&a[2]),
    )?;
    println!(
        "{}",
        serde_json::json!({"all_originals_exact":result["all_four_originals_verified_before_any_variant"],"cases":4,"diagnostic_only":true})
    );
    Ok(())
}
