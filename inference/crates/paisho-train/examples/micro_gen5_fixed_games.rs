use std::{fs,path::Path};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=std::env::args().skip(1).collect();
    if a.len()!=3 {return Err("CONFIG MANIFEST OUTPUT".into());}
    paisho_train::micro_learning::gen5::benchmark_frozen_games(serde_json::from_slice(&fs::read(&a[0])?)?,Path::new(&a[1]),Path::new(&a[2]))
}
