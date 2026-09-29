fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().collect();
    if args.len()!=3 {return Err("PLAN NEW_OUTPUT_DIR".into());}
    paisho_train::micro_learning::gen5::run_depth_confirmation(std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))
}
