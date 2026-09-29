//! Frozen opponent inference/search parity probe, no learner or campaign.
use std::{fs,path::Path};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let a:Vec<_>=std::env::args().collect();
    if a.len()!=4{return Err("SPECS RECORD_DIRECTORY OUTPUT".into());}
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global()?;
    let specs:Vec<paisho_train::micro_learning::gen5::OpponentSpec>=serde_json::from_slice(&fs::read(&a[1])?)?;
    let mut paths=fs::read_dir(&a[2])?.map(|e|e.map(|e|e.path())).collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|p|p.extension().is_some_and(|x|x=="psr"));paths.sort();
    let records=paths.iter().map(|p|fs::read_to_string(p)?.parse().map_err(|e|Box::<dyn std::error::Error>::from(e))).collect::<Result<Vec<_>,_>>()?;
    let result=paisho_train::micro_learning::gen5::probe_opponents(&specs,&records)?;
    fs::write(Path::new(&a[3]),serde_json::to_vec_pretty(&result)?)?;Ok(())
}
