//! Frozen real examples: allocation reuse only, no optimizer step or publication.
use paisho_train::micro_learning::{MicroArtifact,SavedMicroExample};
use paisho_core::RuleProfileId;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,io::Read,path::Path,time::Instant};
fn main()->Result<(),Box<dyn std::error::Error>> {
 let a=std::env::args().skip(1).collect::<Vec<_>>();if !(3..=4).contains(&a.len()) {return Err("MODEL MANIFEST OUTPUT [SCALING]".into());}
 let model=MicroArtifact::load(Path::new(&a[0]))?.model()?;
 let manifest:Value=serde_json::from_slice(&fs::read(&a[1])?)?;let mut rows=vec![];
 for r in manifest["rows"].as_array().ok_or("rows")? {
  let bytes=fs::read(r["targets"].as_str().ok_or("targets")?)?;
  if format!("{:x}",Sha256::digest(&bytes))!=r["targets_sha256"] {return Err("target hash".into());}
  let mut raw=vec![];flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut raw)?;
  let all:Vec<SavedMicroExample>=serde_json::from_slice(&raw)?;
  rows.push(all[r["index"].as_u64().ok_or("index")? as usize].example_for_rules(RuleProfileId::SkudPaiShoGen5V1)?);
 }
 for e in &rows {
  let (a,g)=model.loss_gradient(e)?;
  let buffer=vec![f64::NAN;model.parameters().len()];let pointer=buffer.as_ptr();
  let (b,h)=model.loss_gradient_reusing(e,buffer)?;
  assert_eq!(pointer,h.as_ptr());assert_eq!((a.value.to_bits(),a.policy.to_bits()),(b.value.to_bits(),b.policy.to_bits()));
  assert!(g.iter().zip(&h).all(|(a,b)|a.to_bits()==b.to_bits()));
 }
 let mut results=vec![];
 for reuse in [false,true,true,false] {
  let mut buffer=vec![];let t=Instant::now();
  for _ in 0..16 {for e in &rows {
   let (loss,g)=if reuse {model.loss_gradient_reusing(e,std::mem::take(&mut buffer))?} else {model.loss_gradient(e)?};
   std::hint::black_box(loss);if reuse {buffer=g;}else{std::hint::black_box(g);}
  }}
  results.push(json!({"reuse":reuse,"seconds":t.elapsed().as_secs_f64(),"examples":rows.len()*16}));
 }
 let mut scaling=vec![];
 if a.len()==4 {
  use rayon::prelude::*;
  let jobs=(0..32).flat_map(|_|rows.iter()).collect::<Vec<_>>();
  for threads in [1,2,4,8,10,10,8,4,2,1] {
   let pool=rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
   let start=Instant::now();
   pool.install(||jobs.par_iter().with_min_len(4).for_each_init(Vec::new,|buffer,e| {
    let (loss,g)=model.loss_gradient_reusing(e,std::mem::take(buffer)).unwrap();std::hint::black_box(loss);*buffer=g;
   }));
   let seconds=start.elapsed().as_secs_f64();scaling.push(json!({"threads":threads,"examples":jobs.len(),"seconds":seconds,"examples_per_second":jobs.len() as f64/seconds}));
  }
 }
 let report=json!({"rows":rows.len(),"parameters":model.parameters().len(),"gradient_bits_exact":true,"buffer_address_preserved":true,"results":results,"scaling":scaling});
 fs::write(&a[2],serde_json::to_vec_pretty(&report)?)?;println!("{report}");Ok(())
}
