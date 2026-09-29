//! Offline neutral recovery proposal. Never activates a campaign.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("CONFIG DURABLE_PROGRESS SEED_DIRECTORY NEW_OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let result = paisho_train::micro_learning::gen5::prepare_relational_recovery(
        std::path::Path::new(&args[1]),
        std::path::Path::new(&args[2]),
        std::path::Path::new(&args[3]),
        std::path::Path::new(&args[4]),
    )?;
    println!("{}", result);
    Ok(())
}
