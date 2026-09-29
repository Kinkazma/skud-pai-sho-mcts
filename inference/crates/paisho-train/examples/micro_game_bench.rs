//! Frozen whole-game throughput; censored games stay separate from terminal yield.
use paisho_ai::{MctsConfig, MctsSession, MicroMctsSession};
use paisho_core::{legal_actions, GameOutcome, GameRecord, Position, StandardSetup, BASIC_FLOWERS};
use paisho_train::{compact_learning::load_model, micro_learning::MicroArtifact};
use rayon::prelude::*;
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    if !(7..=8).contains(&a.len()) {
        return Err("MODEL4 MODEL3 OUTPUT THREADS GAMES BUDGET MAX_SECONDS".into());
    }
    let micro = Arc::new(MicroArtifact::load(Path::new(&a[0]))?.model()?);
    let old = load_model(Path::new(&a[1]))?.model()?;
    let out = Path::new(&a[2]);
    std::fs::create_dir(out)?;
    let threads: usize = a[3].parse()?;
    let games: usize = a[4].parse()?;
    let budget: usize = a[5].parse()?;
    let seconds: f64 = a[6].parse()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    // Each generation gets the complete pool in alternating blocks; no cross-generation contention.
    for gen in if a.get(7).is_some_and(|v| v == "gen4-only") {
        vec![4]
    } else {
        vec![4, 3]
    } {
        let start = Instant::now();
        let rows:Vec<_>=pool.install(||(0..games).into_par_iter().map(|id| -> Result<serde_json::Value,String> {
   let t=Instant::now();let deadline=t+Duration::from_secs_f64(seconds);
   let setup=StandardSetup::balanced(BASIC_FLOWERS[id%BASIC_FLOWERS.len()]);
   let mut p=Position::from_standard_setup(setup);let mut record=GameRecord::new(setup);
   let mut m=MicroMctsSession::new(micro.clone());
   let mut l=MctsSession::new(47000+id as u64,MctsConfig{simulations:budget,..Default::default()},&old)?;
   let mut search=0.0;let mut maintenance=0.0;let mut sims=0;
   while p.outcome()==GameOutcome::Ongoing && record.actions().len()<600 && Instant::now()<deadline {
    let b=Instant::now();
    let action=if gen==4 {let r=m.search_until(&p,budget,Some(deadline))?;sims+=r.simulations;if r.simulations==0 {break;}r.actions[r.selected_index]}
      else {let actions=legal_actions(&p);let r=l.search_until(&p,&actions,Some(deadline))?;sims+=r.simulations;actions[r.selected_index]};
    search+=b.elapsed().as_secs_f64();p.apply(action).map_err(|e|e.to_string())?;record.push(action);
    let b=Instant::now();if gen==4 {m.advance(action)?;}else{l.advance(action);}maintenance+=b.elapsed().as_secs_f64();
   }
   let elapsed=t.elapsed().as_secs_f64();let terminal=p.outcome()!=GameOutcome::Ongoing;
   let psr=record.to_string();std::fs::write(out.join(format!("gen{gen}-{id}.psr")),psr).map_err(|e|e.to_string())?;
   Ok(serde_json::json!({"generation":gen,"id":id,"budget":budget,"threads":threads,"seconds":elapsed,"terminal":terminal,"outcome":format!("{:?}",p.outcome()),"decisions":record.actions().len(),"search_seconds":search,"maintenance_seconds":maintenance,"simulations":sims,"termination":if terminal{"terminal"}else if record.actions().len()>=600{"decision-limit"}else{"wall-limit"}}))
  }).collect());
        let rows = rows
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .map_err(std::io::Error::other)?;
        let report = serde_json::json!({"generation":gen,"budget":budget,"threads":threads,"wall_seconds":start.elapsed().as_secs_f64(),"games":rows});
        std::fs::write(
            out.join(format!("gen{gen}.json")),
            serde_json::to_vec_pretty(&report)?,
        )?;
        println!("{report}");
    }
    Ok(())
}
