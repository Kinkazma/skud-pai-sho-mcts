fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = std::env::args().skip(1).collect::<Vec<_>>();
    if a.len() != 2 {
        return Err("CONFIG OUTPUT".into());
    }
    println!(
        "{}",
        paisho_train::micro_learning::gen5::verify_coverage_prefetch(
            std::path::Path::new(&a[0]),
            std::path::Path::new(&a[1])
        )?
    );
    Ok(())
}
