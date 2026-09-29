//! Assimilation diagnostic on source-separated durable tactical targets.
use paisho_ai::MicroExample;
use paisho_train::{gen32::Artifact, micro_learning::SavedMicroExample};
use std::{collections::HashSet, fs};
fn load_examples(
    path: &std::path::Path,
) -> Result<Vec<SavedMicroExample>, Box<dyn std::error::Error>> {
    Ok(serde_json::from_reader(flate2::read::GzDecoder::new(
        fs::File::open(path)?,
    ))?)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let parent = Artifact::load(std::path::Path::new(&a[1]))?;
    let mut model = parent.model()?;
    let index: serde_json::Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let mut paths = HashSet::new();
    let mut train = Vec::<MicroExample>::new();
    let mut test = vec![];
    for row in index["corrections"].as_array().unwrap() {
        let path = row["path"].as_str().unwrap();
        if !paths.insert(path.to_string()) {
            continue;
        }
        let destination = if paths.len() % 3 == 0 {
            &mut test
        } else {
            &mut train
        };
        for s in load_examples(std::path::Path::new(path))? {
            destination.push(s.example()?);
        }
    }
    if train.is_empty() || test.is_empty() {
        return Err("not enough independent correction games".into());
    }
    let loss = |m: &paisho_ai::Gen32Model, x: &[MicroExample]| -> f64 {
        x.iter()
            .map(|s| m.policy.loss_gradient(s).unwrap().0.policy)
            .sum::<f64>()
            / x.len() as f64
    };
    let before = [loss(&model, &train), loss(&model, &test)];
    for i in 0..1000 {
        model
            .policy
            .train_policy_step(&train[i % train.len()], 0.01)?;
    }
    let after = [loss(&model, &train), loss(&model, &test)];
    let result = serde_json::json!({"train_positions":train.len(),"test_positions":test.len(),"source_games":paths.len(),"updates":1000,"policy_cross_entropy_before":before,"policy_cross_entropy_after":after,"value_unchanged":model.value.weights()==parent.model()?.value.weights(),"installed":false});
    fs::write(&a[3], serde_json::to_vec_pretty(&result)?)?;
    println!("{result}");
    Ok(())
}
