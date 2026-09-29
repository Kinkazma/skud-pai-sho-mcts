fn main()->Result<(),Box<dyn std::error::Error>> {
    std::env::set_var("VECLIB_MAXIMUM_THREADS","1");
    let a=std::env::args().collect::<Vec<_>>();if a.len()!=3{return Err("PLAN NEW_OUTPUT".into());}
    let r=paisho_train::micro_learning::gen5::probe_recall_repair(std::path::Path::new(&a[1]),std::path::Path::new(&a[2]))?;
    println!("{}",serde_json::to_string(&r)?);Ok(())
}
