//! Larger current-state rows and successor-only supervision, with frozen groups.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json,Value};
use sha2::{Digest,Sha256};
use std::{collections::{BTreeMap,BTreeSet},fs,io::{BufRead,Write},path::Path,time::Instant};
#[path="gen5_structured_corpus/facts.rs"] mod facts;
type Result<T> = std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(b:&[u8])->String {format!("{:x}",Sha256::digest(b))}
struct Root {position:Position, record:GameRecord, group:String, split:String, key:String, symmetry:String, human_reply:Option<(Action,Action)>}

fn main()->Result<()> {
    let args:Vec<_>=std::env::args().collect();if args.len()!=3 {return Err("PLAN NEW_OUT".into());}
    let bytes=fs::read(&args[1])?;let plan:Value=serde_json::from_slice(&bytes)?;
    let out=Path::new(&args[2]);fs::create_dir(out)?;fs::create_dir(out.join("psr"))?;
    let start=Instant::now();let mut roots=BTreeMap::<String,Root>::new();let mut sym=BTreeMap::<String,BTreeSet<String>>::new();
    for s in plan["sources"].as_array().ok_or("sources")? {
        let b=fs::read(s["psr"].as_str().ok_or("psr")?)?;if hash(&b)!=s["sha256"] {return Err("source changed".into());}
        let source:GameRecord=std::str::from_utf8(&b)?.parse()?;
        let record=source.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)?.0;
        let human=s["kind"]=="human";let n=record.actions().len();
        let indices:BTreeSet<usize>=if human {[n/2,n.saturating_sub(1)].into_iter().collect()} else {[n].into_iter().collect()};
        let mut p=record.initial_position();let mut prefix=GameRecord::with_rules(record.setup(),record.rules());
        for i in 0..=n {
            if indices.contains(&i) && p.outcome()==GameOutcome::Ongoing {
                let key=facts::position_hash(&p);let symmetry=facts::symmetry_hash(&p);
                let split=s["split"].as_str().unwrap().to_string();
                sym.entry(symmetry.clone()).or_default().insert(split.clone());
                let group=s["group"].as_str().unwrap().to_string();
                if let Some(old)=roots.get(&key) {
                    // First deterministic provenance owns same-split duplicates.
                    if old.split!=split {sym.entry(symmetry).or_default().insert(old.split.clone());}
                } else {
                    let human_reply=(human && i+1<n).then(||(record.actions()[i],record.actions()[i+1]));
                    roots.insert(key.clone(),Root {position:p.clone(),record:prefix.clone(),group,split,key,symmetry,human_reply});
                }
            }
            if i<n {p.apply(record.actions()[i])?;prefix.push(record.actions()[i]);}
        }
    }
    let before_count=roots.len();roots.retain(|_,r|sym[&r.symmetry].len()==1);
    // Frozen complete R1 reply censuses are re-used after successor identity checks.
    let rb=fs::read(plan["legacy_roots"].as_str().unwrap())?;
    if hash(&rb)!=plan["legacy_roots_sha256"] {return Err("legacy roots changed".into());}
    let legacy_rows:Vec<Value>=serde_json::from_slice(&rb)?;
    let legacy_base=Path::new(plan["legacy_roots"].as_str().unwrap()).parent().unwrap();
    let mut root_keys=BTreeMap::new();
    for r in legacy_rows {let b=fs::read(legacy_base.join(r["psr"].as_str().unwrap()))?;
        if hash(&b)!=r["sha256"] {return Err("legacy PSR changed".into());}
        let rec:GameRecord=std::str::from_utf8(&b)?.parse()?;
        root_keys.insert(r["key"].as_str().unwrap().to_owned(),facts::position_hash(&rec.replay()?));}
    let bb=fs::read(plan["legacy_branches"].as_str().unwrap())?;
    if hash(&bb)!=plan["legacy_branches_sha256"] {return Err("legacy branches changed".into());}
    let mut cached=BTreeMap::new();
    for line in std::io::BufReader::new(bb.as_slice()).lines() {
        let v:Value=serde_json::from_str(&line?)?;let key=&root_keys[v["root"].as_str().unwrap()];
        if roots.contains_key(key) {cached.insert((key.clone(),v["action"].as_str().unwrap().to_string()),v);}
    }
    let ab=fs::read(plan["actor"].as_str().unwrap())?;if hash(&ab)!=plan["actor_sha256"] {return Err("actor changed".into());}
    let actor:MicroArtifact=serde_json::from_slice(&ab)?;let t=Instant::now();let model=actor.model()?;let load=t.elapsed().as_secs_f64();
    let mut writer=std::io::BufWriter::new(fs::File::create(out.join("rows.jsonl"))?);
    let mut counts=BTreeMap::<String,usize>::new();let mut total_actions=0;let mut positive=[0usize;10];let mut known=[0usize;10];
    let mut label_seconds=0.;
    for (index,r) in roots.values().enumerate() {
        let p=&r.position;let state=model.state_features(p);let graph=MicroPieceGraph::extract(p,p.to_move());
        let actions=legal_actions(p);let af:Vec<_>=actions.iter().map(|&a|micro_action_features(p,a)).collect();
        let prior=model.memory_priors(&state,&af,&micro_softmax(&MicroModel::logits(&model.embed(&state),&af))?,0)?;
        let t=Instant::now();let before=MicroRelations::extract(p,p.to_move());
        let mut targets=vec![];let mut wins=vec![];let mut threat_sources=vec![];
        for &a in &actions {
            let mut q=p.clone();q.apply(a)?;wins.push(q.outcome()==GameOutcome::Win(p.to_move()));
            let mut threat=micro_immediate_threat(&q,p.to_move(),0)?;let mut provenance="bounded-zero";
            if let Some(c)=cached.get(&(r.key.clone(),a.to_string())) {
                if c["position_sha256"]!=facts::position_hash(&q) {return Err("cached successor differs".into());}
                if c["threat"]["status"]=="present" {
                    let witness=c["threat"]["winning_replies"][0].as_str().ok_or("witness")?.parse()?;
                    threat=micro_verify_threat_witness(&q,p.to_move(),witness)?;provenance="R1-replayed-witness";
                } else if c["threat"]["status"]=="absent" {
                    let legal=legal_actions(&q).len();
                    if !c["threat"]["complete"].as_bool().unwrap_or(false) || c["threat"]["examined"]!=legal || c["threat"]["legal"]!=legal
                        || q.to_move()==p.to_move() || q.outcome()!=GameOutcome::Ongoing {return Err("incomplete cached absence".into());}
                    threat=MicroImmediateThreat::Absent {examined:legal};provenance="R1-frozen-complete-census";
                }
            } else if let Some((played,reply))=r.human_reply {
                if played==a {
                    if let Ok(t)=micro_verify_threat_witness(&q,p.to_move(),reply) {
                        if t.label()==Some(true) {threat=t;provenance="human-replayed-witness";}
                    }
                }
            }
            let y=MicroStructuredTarget::from_successor(p,a,&q,&before,&threat);
            for i in 0..10 {if let Some(v)=y.events[i] {known[i]+=1;positive[i]+=usize::from(v);}}
            targets.push(y);threat_sources.push(provenance);
        }
        label_seconds+=t.elapsed().as_secs_f64();total_actions+=actions.len();
        *counts.entry(r.split.clone()).or_default()+=1;
        let psr=format!("psr/{}.psr",r.key);let text=r.record.to_string();fs::write(out.join(&psr),&text)?;
        let row=json!({"key":r.key,"group":r.group,"split":r.split,"task":"policy","held_out":r.split=="final",
            "psr":psr,"psr_sha256":hash(text.as_bytes()),"symmetry":r.symmetry,"state":state,
            "nodes":graph.nodes.iter().map(|n|n.features.to_vec()).collect::<Vec<_>>(),
            "edges":graph.messages.iter().map(|e|json!({"source":e.source,"destination":e.destination,"features":e.features})).collect::<Vec<_>>(),
            "actions":af.iter().map(|a|a.to_vec()).collect::<Vec<_>>(),"action_names":actions.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "prior":prior,"winning":wins,"structured":targets,"threat_provenance":threat_sources});
        serde_json::to_writer(&mut writer,&row)?;writeln!(writer)?;
        if index%200==0 {println!("roots {} actions {}",index+1,total_actions);}
    }
    writer.flush()?;
    fs::write(out.join("summary.json"),serde_json::to_vec_pretty(&json!({"schema":MICRO_STRUCTURED_SCHEMA,"plan_sha256":hash(&bytes),
        "counts":counts,"roots":roots.len(),"cross_split_symmetric_roots_removed":before_count-roots.len(),"actions":total_actions,
        "events_positive":positive,"events_known":known,"label_seconds":label_seconds,"bank_load_seconds":load,
        "total_seconds":start.elapsed().as_secs_f64(),"rows_sha256":hash(&fs::read(out.join("rows.jsonl"))?)}))?)?;
    Ok(())
}
