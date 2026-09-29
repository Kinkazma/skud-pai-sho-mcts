//! Real archived states: shard occupancy only, no model update or cache mutation.
use paisho_ai::*;
use paisho_core::GameRecord;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{fs,io::Read,hash::{Hash,Hasher},collections::{HashSet,hash_map::DefaultHasher}};
fn main()->Result<(),Box<dyn std::error::Error>> {
    let args=std::env::args().skip(1).collect::<Vec<_>>();
    if args.len()!=3 {return Err("MODEL BUNDLE_MANIFEST OUTPUT".into());}
    let artifact:MicroArtifact=serde_json::from_slice(&fs::read(&args[0])?)?;
    let model=artifact.model()?;
    let bank=model.sequence_memory().ok_or("sequence bank")?;
    let sources=bank.entries.iter().map(|e|e.source).collect::<HashSet<_>>();
    // nearest_index explicitly defines zero as no source exclusion.
    let absent=0u64;
    let manifest:Value=serde_json::from_slice(&fs::read(&args[1])?)?;
    let mut rows=vec![]; let mut unique=HashSet::new();
    for row in manifest["rows"].as_array().ok_or("rows")? {
        let bytes=fs::read(row["path"].as_str().ok_or("path")?)?;
        if format!("{:x}",Sha256::digest(&bytes))!=row["sha256"] {return Err("bundle hash".into());}
        let mut raw=vec![];flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut raw)?;
        let bundle:Value=serde_json::from_slice(&raw)?;
        let record:GameRecord=bundle["psr"].as_str().ok_or("psr")?.parse()?;
        let game_id=bundle["game_id"].as_u64().ok_or("game id")?;
        let excluded=bundle["source_run"].as_str().map_or(u64::MAX,|s|sequence_source(&format!("{s}/{game_id}")));
        for lesson in bundle["lessons"].as_array().ok_or("lessons")? {
            let n=lesson["decision"].as_u64().ok_or("decision")? as usize;
            let mut prefix=GameRecord::with_rules(record.setup(),record.rules());
            for &action in record.actions().iter().take(n.checked_sub(1).ok_or("decision zero")?) { prefix.push(action); }
            let state=model.state_features(&prefix.replay()?);
            let key=sequence_key(state[..128].try_into()?);
            let vector=key.map(f32::to_bits);let phase=u8::from(state[125]>0.5);
            let geometry=Some(SequenceGeometry::from_state(&state)?);
            if unique.insert((vector,phase,excluded,geometry)) {rows.push((vector,phase,excluded,geometry,state));}
        }
    }
    let mut panels=vec![];
    for zero in [false,true] {
        let mut old=[0usize;16];let mut balanced=[0usize;16];
        for (v,p,e,g,_) in &rows {
            let e=if zero {0} else {*e};
            old[(v[0] as usize ^ v[17] as usize ^ e as usize)%16]+=1;
            let mut hash=DefaultHasher::new();v.hash(&mut hash);p.hash(&mut hash);e.hash(&mut hash);g.hash(&mut hash);
            balanced[hash.finish() as usize%16]+=1;
        }
        panels.push(json!({"excluded_zero":zero,"old":old,"full_key_hash":balanced}));
    }
    let mut absent_sources=0;let mut normalized=HashSet::new();let mut pairs=0;let mut contexts=Sha256::new();
    for (v,p,e,g,state) in &rows {
        let present=sources.contains(e);let canonical=if present {*e} else {absent};
        normalized.insert((*v,*p,canonical,*g));
        let original=bank.try_context(state,*e)?;
        contexts.update((original.neighbors.len() as u64).to_le_bytes());
        for &i in &original.neighbors {contexts.update((i as u64).to_le_bytes());}
        contexts.update(original.confidence.to_bits().to_le_bytes());
        for v in original.patterns.iter().flatten() {contexts.update(v.to_bits().to_le_bytes());}
        if !present {
            absent_sources+=1;
            let a=bank.try_context(state,*e)?;let b=bank.try_context(state,canonical)?;
            if a.neighbors!=b.neighbors || a.confidence.to_bits()!=b.confidence.to_bits() ||
                !a.patterns.iter().flatten().zip(b.patterns.iter().flatten()).all(|(a,b)|a.to_bits()==b.to_bits()) {
                return Err("absent source normalization changed context".into());
            }
            pairs+=1;
        }
    }
    let report=json!({"real_unique_states":rows.len(),"bank_sources":sources.len(),"absent_sources":absent_sources,"canonical_absent_source":absent,"normalized_unique_queries":normalized.len(),"exact_context_pairs":pairs,"context_sha256":format!("{:x}",contexts.finalize()),"panels":panels,"scope":"frozen archive queries only; zero-source panel represents proof/search queries, not measured production proportions"});
    fs::write(&args[2],serde_json::to_vec_pretty(&report)?)?;println!("{report}");Ok(())
}
