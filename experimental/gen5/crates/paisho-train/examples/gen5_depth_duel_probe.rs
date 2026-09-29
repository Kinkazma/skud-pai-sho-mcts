fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();if args.len()!=3 {return Err("PLAN NEW_OUTPUT_DIR".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let r=paisho_train::micro_learning::gen5::run_depth_duel_probe(std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))?;
    println!("{}",serde_json::json!({"complete":r["complete"],"games":r["games"].as_array().map(Vec::len),"active_seconds":r["active_seconds"]}));Ok(())
}
