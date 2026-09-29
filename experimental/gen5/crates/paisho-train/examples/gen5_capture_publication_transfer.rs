//! True publisher replay in private snapshots; no campaign, learning or promotion.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 {
        return Err("FROZEN_PLAN NEW_OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let r = paisho_train::micro_learning::gen5::replay_capture_publication(
        std::path::Path::new(&a[1]),
        std::path::Path::new(&a[2]),
    )?;
    println!(
        "{}",
        serde_json::json!({"four_native_original_actors_exact":r["all_four_native_original_actors_and_decisions_exact_before_variants"],"new_sgd_or_games":false})
    );
    Ok(())
}
