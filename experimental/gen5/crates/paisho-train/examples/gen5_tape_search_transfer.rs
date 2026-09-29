//! Same frozen roots, cold native search and independently reported coupled rank.
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().collect::<Vec<_>>();if args.len()!=3 {return Err("PLAN NEW_OUTPUT_DIRECTORY".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let report=paisho_train::micro_learning::gen5::measure_tape_search_transfer(
        std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))?;
    println!("{}",serde_json::json!({"complete":report["complete"],"queries":report["completed_query_records"],
        "new_queries":report["fresh_queries"],"kernel_seconds":report["native_kernel_seconds"],"new_games":0}));
    if report["complete"]!=true {return Err("Incomplete diagnostic preserved in report and query journal".into());}Ok(())
}
