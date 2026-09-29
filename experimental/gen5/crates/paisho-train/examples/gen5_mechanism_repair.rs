fn main()->Result<(),Box<dyn std::error::Error>> {
    std::env::set_var("VECLIB_MAXIMUM_THREADS","1");
    let args=std::env::args().collect::<Vec<_>>();
    if args.len()!=3 {return Err("PLAN NEW_OUTPUT".into());}
    paisho_train::micro_learning::gen5::probe_mechanism_repair(std::path::Path::new(&args[1]),std::path::Path::new(&args[2]))?;
    Ok(())
}
