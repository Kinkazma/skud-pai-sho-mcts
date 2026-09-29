fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 {
        return Err("CONFIG OUTPUT_DIRECTORY".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let v = paisho_train::micro_learning::gen5::verify_v34_integration(
        std::path::Path::new(&a[1]),
        std::path::Path::new(&a[2]),
    )?;
    println!(
        "{}",
        serde_json::json!({"verified":true,"seconds":v["seconds"],"next_update_exact":v["next_protected_update_exact_after_reload"]})
    );
    Ok(())
}
