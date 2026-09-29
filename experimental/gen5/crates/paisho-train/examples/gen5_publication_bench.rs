fn main() -> Result<(),Box<dyn std::error::Error>> {
    let a=std::env::args().collect::<Vec<_>>();
    if a.len()!=3 {return Err("CONFIG OUTPUT_DIRECTORY".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let r=paisho_train::micro_learning::gen5::benchmark_publication(std::path::Path::new(&a[1]),std::path::Path::new(&a[2]))?;
    println!("{r}");
    Ok(())
}
