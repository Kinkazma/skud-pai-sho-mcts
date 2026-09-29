//! Run only after the final candidate AND protocol have been frozen.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("FROZEN_FINAL_PLAN.json NEW_OUTPUT_DIRECTORY".into());
    }
    std::env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let result = paisho_train::micro_learning::gen5::probe_final_evaluation(
        std::path::Path::new(&args[1]), std::path::Path::new(&args[2]),
    )?;
    println!("{}", serde_json::json!({"paired_games":result["paired_games"],
        "source_blocks":result["source_blocks"],"wdlu":result["games"]["wdlu"],
        "unresolved_games":result["unresolved_games"],"learned_examples":0}));
    Ok(())
}
