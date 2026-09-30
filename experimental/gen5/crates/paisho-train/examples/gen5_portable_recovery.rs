fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=std::env::args().collect();
    rayon::ThreadPoolBuilder::new().num_threads(4).build_global()?;
    let result=match a.get(1).map(String::as_str) {
        Some("finalize") if a.len()==3 => paisho_train::micro_learning::gen5::finalize_portable_recovery(std::path::Path::new(&a[2]))?,
        Some("verify") if a.len()==4 => paisho_train::micro_learning::gen5::verify_portable_recovery(std::path::Path::new(&a[2]),std::path::Path::new(&a[3]))?,
        _=>return Err("finalize CONFIG | verify CONFIG NEW_OUTPUT".into()),
    };
    println!("{result}");Ok(())
}
