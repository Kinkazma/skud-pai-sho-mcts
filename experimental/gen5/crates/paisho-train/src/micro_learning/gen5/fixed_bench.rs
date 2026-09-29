//! Explicit diagnostic: fixed models, prefixes and seeds; no learning or campaign writes.
use super::*;
use std::sync::atomic::{AtomicUsize,Ordering};

pub fn benchmark_frozen_games(o:Options, manifest:&Path, output:&Path)->Result<()> {
    let panel:serde_json::Value=serde_json::from_slice(&fs::read(manifest)?)?;
    let rows=panel["rows"].as_array().ok_or_else(||invalid("fixed game rows missing"))?;
    if rows.is_empty() || rows.len()>256 {return Err(invalid("fixed game panel must contain 1..256 rows"));}
    fs::create_dir(output)?;
    let artifact=MicroArtifact::load(&o.model)?;
    let model=Arc::new(artifact.model()?);
    let reference=crate::compact_learning::load_model(&o.reference)?.model()?;
    let snapshot=Arc::new(Snapshot {artifact:None,version:0,identity:artifact.identity(),model,path:o.model.clone()});
    let mut inputs=vec![];
    for row in rows {
        let path=Path::new(row["psr"].as_str().ok_or_else(||invalid("fixed PSR"))?);
        let bytes=fs::read(path)?;
        if sha256(&bytes)!=row["sha256"] {return Err(invalid("fixed PSR hash"));}
        let record:GameRecord=std::str::from_utf8(&bytes)?.parse()?;
        record.replay()?;
        let prefix=cases::prefix(&record,row["prefix"].as_u64().ok_or_else(||invalid("prefix"))? as usize);
        let seat=if row["candidate_seat"]=="Host" {Player::Host}else{Player::Guest};
        inputs.push((row["id"].as_u64().ok_or_else(||invalid("id"))? as usize,prefix,seat,row["reference_budget"].as_u64().ok_or_else(||invalid("reference budget"))? as usize));
    }
    let private=panel["private_workers"].as_u64().unwrap_or(0) as usize;
    if private>0 && private!=o.threads {return Err(invalid("private workers must preserve the configured CPU count"));}
    let shards=panel["private_pool_shards"].as_u64().unwrap_or(private as u64) as usize;
    if private>0 && (shards==0 || private%shards!=0) {return Err(invalid("invalid private pool shards"));}
    let actors=if private>0 {panel["private_actors"].as_u64().unwrap_or(private as u64) as usize}else{o.actors};
    let mut executors=vec![];
    for _ in 0..shards.max(1) {
        let (pool,_)=cpu::build_pool(if private>0 {private/shards}else{o.threads},None)?;
        executors.push(cpu::Executor::direct(pool));
    }
    let next=AtomicUsize::new(0);
    let started=paisho_platform::training_time::now();let end=started+Duration::from_secs(300);
    let all=std::thread::scope(|scope| {
        let mut handles=vec![];
        for actor in 0..actors.min(inputs.len()) {
            let executor=&executors[if private>0 {actor%shards}else{0}];
            let (inputs,next,o,snapshot,reference)=(&inputs,&next,&o,&snapshot,&reference);
            handles.push(scope.spawn(move || {
                let mut out=vec![];
                loop {
                    let i=next.fetch_add(1,Ordering::Relaxed);
                    if i>=inputs.len() {break;}
                    let (id,prefix,seat,budget)=&inputs[i];let mut options=o.clone();options.candidate_seat=Some(*seat);
                    let mut game=collector::play_from(*id,snapshot.clone(),snapshot.clone(),Some((&reference,*budget)),&options,end,&executor,"gen5-fixed-throughput-panel-v1",Some(prefix),false,None);
                    if let Some(e)=&game.error {return Err(e.clone());}
                    let psr=game.record.to_string();
                    let saved=collector::targets(&mut game);
                    fs::write(output.join(format!("game-{i:04}.psr")),&psr).map_err(|e|e.to_string())?;
                    out.push(serde_json::json!({"index":i,"id":id,"outcome":format!("{:?}",game.outcome),"decisions":game.record.actions().len(),"continuation_decisions":game.record.actions().len()-game.prefix_decisions,"psr_sha256":sha256(psr.as_bytes()),"targets_sha256":sha256(&serde_json::to_vec(&saved).unwrap()),"proofs_sha256":sha256(&serde_json::to_vec(&game.certificates).unwrap()),"seconds":game.seconds,"search_seconds":game.search_seconds,"simulations":game.simulations,"tactical":game.tactical_evaluations,"eligible":saved.len()}));
                }
                Ok(out)
            }));
        }
        handles.into_iter().map(|h|h.join().map_err(|_|"fixed game worker panic".to_string())?).collect::<std::result::Result<Vec<Vec<serde_json::Value>>,String>>()
    }).map_err(invalid)?;
    let seconds=paisho_platform::training_time::elapsed(started).as_secs_f64();
    let mut rows:Vec<_>=all.into_iter().flatten().collect();rows.sort_by_key(|r|r["index"].as_u64());
    let terminals=rows.iter().filter(|r|r["outcome"]!="Ongoing").count();
    let report=serde_json::json!({"model":snapshot.identity,"frozen_games":rows.len(),"terminals":terminals,"seconds":seconds,"terminal_per_second":terminals as f64/seconds,"actors":actors,"threads":o.threads,"private_workers":private,"pool_shards":shards,"no_learning":true,"rows":rows});
    fs::write(output.join("report.json"),serde_json::to_vec_pretty(&report)?)?;
    println!("{}",serde_json::json!({"frozen_games":inputs.len(),"terminal":terminals,"seconds":seconds,"terminal_per_second":terminals as f64/seconds}));
    Ok(())
}
