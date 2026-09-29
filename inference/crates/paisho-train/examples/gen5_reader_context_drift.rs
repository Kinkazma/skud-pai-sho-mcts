fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() != 3 { return Err("FROZEN_PLAN NEW_OUTPUT".into()); }
    let r = paisho_train::micro_learning::gen5::measure_reader_context_drift(
        std::path::Path::new(&a[1]), std::path::Path::new(&a[2]))?;
    println!("{}", serde_json::json!({"blocks":r["blocks"].as_array().map(|v|v.len()),"new_sgd_or_games":false}));
    Ok(())
}
