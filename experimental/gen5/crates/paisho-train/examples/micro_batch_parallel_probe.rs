//! Frozen ordered minibatches; compare private learner pools, never publish.
use paisho_ai::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::json;
use sha2::{Digest,Sha256};
use std::{fs,io::Read,path::Path,time::Instant};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args:Vec<_>=std::env::args().skip(1).collect();
    if args.len()!=3 {return Err("MODEL MANIFEST OUTPUT".into());}
    let model=MicroArtifact::load(Path::new(&args[0]))?.model()?;
    let manifest:serde_json::Value=serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut examples=vec![];
    for row in manifest["rows"].as_array().ok_or("rows")? {
        let raw=fs::read(row["targets"].as_str().ok_or("targets")?)?;
        assert_eq!(format!("{:x}",Sha256::digest(&raw)),row["targets_sha256"]);
        let mut bytes=vec![];flate2::read::GzDecoder::new(&raw[..]).read_to_end(&mut bytes)?;
        let saved:Vec<SavedMicroExample>=serde_json::from_slice(&bytes)?;
        examples.push(saved[row["index"].as_u64().ok_or("index")? as usize].example_for_rules(paisho_train::micro_learning::gen5::RULES)?);
    }
    for ex in &examples {model.memory_context(&ex.state,ex.sequence_source)?;}
    let mut results=vec![];
    for size in [5,16,32,64] {
        let mut expected=None;
        for threads in [1,2,4,4,2,1] {
            let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
            let mut candidate=model.clone();let mut outcomes=vec![];
            let started=Instant::now();
            for step in 0..40 {
                let batch:Vec<_>=(0..size).map(|j|&examples[(step*7+j)%examples.len()]).collect();
                let rate=0.001*size as f64/64.;
                let loss=if threads==1 {candidate.train_batch_inline(&batch,rate,1e-5)?}
                    else {pool.install(||candidate.train_batch(&batch,rate,1e-5))?};
                outcomes.push([loss.value.to_bits(),loss.policy.to_bits()]);
            }
            let seconds=started.elapsed().as_secs_f64();
            let bits:Vec<_>=candidate.parameters().iter().map(|x|x.to_bits()).collect();
            let digest=format!("{:x}",Sha256::digest(serde_json::to_vec(&json!([bits,outcomes]))?));
            if let Some(old)=&expected {assert_eq!(&digest,old);}else{expected=Some(digest.clone());}
            let row=json!({"batch":size,"threads":threads,"seconds":seconds,"digest":digest});
            println!("{row}");results.push(row);
        }
    }
    fs::write(&args[2],serde_json::to_vec_pretty(&results)?)?;
    Ok(())
}
