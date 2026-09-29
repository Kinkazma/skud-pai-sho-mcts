//! Scalar recall-priority comparison on immutable archived native predictions.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if !(3..=4).contains(&args.len()) {
        return Err("FIXTURES OUTPUT [REPETITIONS=4096]".into());
    }
    let repetitions = args.get(3).map(|v| v.parse()).transpose()?.unwrap_or(4096);
    let result = paisho_train::micro_learning::gen5::verify_recall_policy_distance(
        std::path::Path::new(&args[1]), repetitions)?;
    std::fs::write(&args[2], serde_json::to_vec_pretty(&result)?)?;
    println!("{}", serde_json::json!({"rows":result["rows"].as_array().unwrap().len(),
        "abba":result["abba"],"model_evaluations":0}));
    Ok(())
}
