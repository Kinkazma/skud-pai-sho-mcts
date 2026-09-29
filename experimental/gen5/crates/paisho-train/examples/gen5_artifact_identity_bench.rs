fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 2 { return Err("MODEL OUTPUT".into()); }
    paisho_train::micro_learning::gen5::benchmark_artifact_identity(
        std::path::Path::new(&args[0]), std::path::Path::new(&args[1]),
    )
}
