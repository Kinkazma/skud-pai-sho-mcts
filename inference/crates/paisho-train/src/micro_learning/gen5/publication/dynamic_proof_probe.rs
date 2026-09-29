//! Isolated diagnostic only: no runtime registration, publication, or campaign.
//! Frozen native certificates -> accepted-actor bits -> additional finite choices.
use super::*;
mod registry;
mod fractions;
use registry::{Bits,Entry,Registry};
const EPS:f64=1e-6;
const REQUIRED_KEY:&str="7ca1ea18081fbcb9e95b12a1a8b4c249989d083dbbc77cb18945b7a38b6cc986";

#[derive(Clone,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Input {path:PathBuf,sha256:String}
impl Input {
    fn bytes(&self)->Result<Vec<u8>> {let b=fs::read(&self.path)?;if sha256(&b)!=self.sha256 {return Err(invalid(format!("changed dynamic diagnostic input {}",self.path.display())));}Ok(b)}
}
#[derive(Clone,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct ProofSpec {
    key:String,receipt_order:usize,target_index:usize,decision:usize,proof:Input,targets:Input,
    // Optional native episode provenance; absent keeps the historical diagnostic contract.
    #[serde(default,skip_serializing_if="Option::is_none")]
    native_source:Option<NativeSource>,
}
#[derive(Clone,Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct NativeSource {receipt:Input,psr:Input}
#[derive(Deserialize,Serialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema:String,initial:Input,candidate:Input,config:Input,guard:Input,validation:Input,
    report:Input,loop_plan:Input,provenance:Input,proofs:Vec<ProofSpec>,benchmark_repetitions:usize,
    #[serde(default)] fraction_control:bool,
}
struct Loaded {spec:ProofSpec,witness:Arc<Witness>,support_sha256:String,actions:Vec<String>}
fn best(p:&[f64],keep:impl Fn(usize)->bool)->Option<usize> {
    (0..p.len()).filter(|&i|keep(i)).max_by(|&a,&b|p[a].total_cmp(&p[b]).then_with(||b.cmp(&a)))
}
fn bits(s:&Score,i:usize)->Bits {Bits {raw:s.raw[i],coupled:s.coupled[i]}}
fn same_parameters(a:&MicroModel,b:&MicroModel)->bool {
    a.parameters().len()==b.parameters().len()&&a.parameters().iter().zip(b.parameters()).all(|(a,b)|a.to_bits()==b.to_bits())
}
fn parameter_sha(m:&MicroModel)->String {sha256(&m.parameters().iter().flat_map(|x|x.to_bits().to_le_bytes()).collect::<Vec<_>>())}
fn score_sha(scores:&[Score])->Result<String> {
    Ok(sha256(&serde_json::to_vec(&scores.iter().map(|s|serde_json::json!({"raw":s.raw,"coupled":s.coupled,
        "mass_bits":s.mass.to_bits(),"value_mse_bits":s.value_mse.to_bits(),
        "priors_bits":s.priors.iter().map(|r|r.iter().map(|x|x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "coupled_logits_bits":s.coupled_logits.iter().map(|r|r.iter().map(|x|x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "errors_bits":s.errors.iter().map(|x|x.to_bits()).collect::<Vec<_>>()})).collect::<Vec<_>>())?))
}
fn value_bits(a:&MicroModel,b:&MicroModel)->Result<usize> {
    if a.parameters().len()!=b.parameters().len() {return Err(invalid("dynamic model shape differs"));}
    let mut count=0;for (i,(a,b)) in a.parameters().iter().zip(b.parameters()).enumerate() {
        if branches::value_parameter(i) {count+=1;if a.to_bits()!=b.to_bits() {return Err(invalid("dynamic repair changed a value coefficient"));}}
    }Ok(count)
}
fn independent_copy(m:&MicroModel)->Result<MicroModel> {
    let mut out=MicroModel::from_parameters(m.parameters().to_vec()).map_err(invalid)?;
    if let Some(bank)=m.sequence_memory() {out=out.with_sequence_memory_owned(bank.clone());}Ok(out)
}
fn paired_choices(loaded:&[Loaded],old:&Score,new:&Score)->serde_json::Value {
    let rows=loaded.iter().enumerate().map(|(i,l)| {
        let selected=|s:&Score,coupled:bool| {let logits=ActivePanel::logits(s,i,coupled);best(&logits,|_|true).map(|i|l.actions[i].clone())};
        serde_json::json!({"key":l.spec.key,"before":bits(old,i),"after":bits(new,i),
            "raw_before":selected(old,false),"raw_after":selected(new,false),"coupled_before":selected(old,true),"coupled_after":selected(new,true)})
    }).collect::<Vec<_>>();
    let summary=|a:&[bool],b:&[bool]|serde_json::json!({"before":a.iter().filter(|x|**x).count(),"after":b.iter().filter(|x|**x).count(),
        "gains":a.iter().zip(b).filter(|(a,b)|!**a&&**b).count(),"losses":a.iter().zip(b).filter(|(a,b)|**a&&!**b).count()});
    serde_json::json!({"rows":rows,"raw":summary(&old.raw,&new.raw),"coupled":summary(&old.coupled,&new.coupled)})
}

fn native_decision_actor(r:&serde_json::Value,player:Player)->Result<String> {
    let text=|field:&str|r[field].as_str().filter(|s|!s.is_empty())
        .ok_or_else(||invalid(format!("native decision provenance lacks {field}")));
    let collector=text("collector")?;
    let opponent=text("opponent")?;
    let seat=match text("candidate_seat")? {
        "Host"=>Player::Host,"Guest"=>Player::Guest,
        _=>return Err(invalid("native decision provenance invalid candidate_seat")),
    };
    let lane=text("lane")?;
    let reanalysis=r["reanalysis"].as_bool().ok_or_else(||invalid("native decision provenance lacks reanalysis"))?;
    if reanalysis!=(lane=="Reanalysis") {
        return Err(invalid("native decision provenance lane/reanalysis mismatch"));
    }
    let opponent_actor=if !r["reference_budget"].is_null() {
        let identity=text("reference_identity")?;
        let generation=r["case"]["opponent_generation"].as_str()
            .ok_or_else(||invalid("native reference lacks case generation"))?;
        if !matches!(lane,"Historical"|"Reanalysis")
            ||r["reference_budget"].as_u64().unwrap_or(0)==0
            ||identity.len()!=64||!identity.bytes().all(|c|c.is_ascii_hexdigit())
            ||opponent!=format!("Gen{generation}") {
            return Err(invalid("native decision provenance reference/lane/generation mismatch"));
        }
        format!("{opponent}:{identity}")
    } else {
        if !r["reference_identity"].is_null()
            ||!matches!(lane,"Selfplay"|"Checkpoint"|"Reanalysis")
            ||lane=="Selfplay"&&collector!=opponent
            ||lane=="Checkpoint"&&collector==opponent {
            return Err(invalid("native decision provenance opponent/lane mismatch"));
        }
        opponent.to_string()
    };
    Ok(if player==seat {collector.to_string()}else{opponent_actor})
}

fn native_actor(spec:&ProofSpec,adapter:&serde_json::Value,prefix:&str,player:Player,saved:&SavedMicroExample)->Result<Option<String>> {
    let Some(source)=&spec.native_source else {return Ok(None)};
    let r:serde_json::Value=serde_json::from_slice(&source.receipt.bytes()?)?;
    let psr=source.psr.bytes()?;
    let record:GameRecord=std::str::from_utf8(&psr)?.parse()?;
    let targets_file=r["targets_file"].as_str().ok_or_else(||invalid("native receipt lacks targets_file"))?;
    let targets=source.receipt.path.parent().ok_or_else(||invalid("native receipt lacks parent"))?.join(targets_file);
    if r["rules"]!=RULES.as_str()||record.rules()!=RULES||record.to_string().as_bytes()!=psr
        ||r["psr_sha256"].as_str()!=Some(source.psr.sha256.as_str())
        ||r["psr_sha256"]!=adapter["source_group"]
        ||r["targets_sha256"].as_str()!=Some(spec.targets.sha256.as_str())
        ||targets.canonicalize()?!=spec.targets.path.canonicalize()?
        ||r["collector"]!=adapter["actor"]||r["collector"].as_str()!=Some(saved.collector.as_str())
        ||r["id"].as_u64().map(|id|id.to_string()).as_deref()!=Some(saved.game_id.as_str())
        ||r["fully_learned"]!=true||!r["error"].is_null()
        ||spec.decision==0||spec.decision>record.actions().len()+1
        ||cases::prefix(&record,spec.decision-1).to_string()!=prefix {
        return Err(invalid(format!("native receipt/PSR/target binding mismatch for {} decision {}",spec.key,spec.decision)));
    }
    Ok(Some(native_decision_actor(&r,player)?))
}

fn decision_identity_matches(evidence:Option<&TargetEvidence>,actor:&str,player:Option<Player>)->bool {
    evidence.is_some_and(|e|e.actor==actor&&player.map_or(true,|p|e.player==p.code().to_string()))
}

fn load_proof(spec:&ProofSpec,model:&MicroModel,receipt:&serde_json::Value)->Result<Loaded> {
    let receipt_path=Path::new(receipt["targets"].as_str().ok_or_else(||invalid("dynamic receipt lacks target path"))?).canonicalize()?;
    if receipt_path!=spec.targets.path.canonicalize()?||receipt["targets_sha256"].as_str()!=Some(spec.targets.sha256.as_str()) {
        return Err(invalid("dynamic target is not bound to cycle-1 receipt"));
    }
    let stride=receipt["stride"].as_u64().ok_or_else(||invalid("missing actual stride"))? as usize;
    if stride==0||spec.target_index%stride!=0 {return Err(invalid("dynamic proof was not selected by the actual stride"));}
    let all=decode_examples(&spec.targets.bytes()?)?;
    let saved=all.get(spec.target_index).ok_or_else(||invalid("dynamic selected target missing"))?;
    let v:serde_json::Value=serde_json::from_slice(&spec.proof.bytes()?)?;
    let prefix=v["prefix"].as_str().ok_or_else(||invalid("dynamic proof prefix missing"))?;
    if sha256(prefix.as_bytes())!=spec.key||spec.proof.path.file_stem().and_then(|s|s.to_str())!=Some(spec.key.as_str())||v["rules"]!=RULES.as_str() {
        return Err(invalid("dynamic proof identity/rules mismatch"));
    }
    let record:GameRecord=prefix.parse()?;let position=record.replay()?;
    if record.rules()!=RULES||position.outcome()!=GameOutcome::Ongoing||record.actions().len()+1!=spec.decision {
        return Err(invalid("dynamic proof decision is not the declared ongoing Gen5 prefix"));
    }
    let certificate:MicroProofCertificate=serde_json::from_value(v["certificate"].clone())?;
    // Full native verification is inside this helper. No inference of proof from Q.
    let legal=paisho_core::legal_actions(&position);
    let policy=action_values::verified_winning_policy(&position,&legal,&certificate)?;
    let actions=legal.iter().map(ToString::to_string).collect::<Vec<_>>();
    let mut example=saved.example_for_rules_with_trusted_q(RULES,true)?;
    let native_actor=native_actor(spec,receipt,prefix,position.to_move(),saved)?;
    let expected_actor=native_actor.as_deref().unwrap_or_else(||receipt["actor"].as_str().unwrap_or(""));
    if !decision_identity_matches(saved.evidence.as_ref(),expected_actor,native_actor.as_ref().map(|_|position.to_move())) {
        return Err(invalid(format!("dynamic decision provenance mismatch key={} game={} decision={} expected_actor={} actual_actor={:?} expected_player={} actual_player={:?} native={}",
            spec.key,saved.game_id,spec.decision,expected_actor,saved.evidence.as_ref().map(|e|e.actor.as_str()),
            position.to_move().code(),saved.evidence.as_ref().map(|e|e.player.as_str()),native_actor.is_some())));
    }
    if saved.rules!=RULES.as_str()||saved.decision!=spec.decision||saved.collector!=receipt["actor"].as_str().unwrap_or("")
        ||saved.actions!=actions||example.state!=model.state_features(&position)
        ||example.actions!=legal.iter().map(|a|micro_action_features(&position,*a)).collect::<Vec<_>>()
        ||!example.policy_support||example.policy.iter().map(|p|*p>0.).collect::<Vec<_>>() != policy.iter().map(|p|*p>0.).collect::<Vec<_>>()
        ||policy.iter().enumerate().any(|(i,p)|*p>0.&&example.action_values.get(i)!=Some(&Some(1.))) {
        return Err(invalid(format!("dynamic proof/Saved feature, support, Q or collector mismatch key={} game={} decision={}",spec.key,saved.game_id,spec.decision)));
    }
    example.policy=policy;example.value=1.;example.policy_weight=1.;example.value_weight=0.;example.sequence_source=0;example.action_values.clear();
    let valid=example.policy.iter().map(|p|*p>0.).collect::<Vec<_>>();
    let support_sha256=sha256(&serde_json::to_vec(&actions.iter().zip(&valid).filter(|(_,v)|**v).map(|(a,_)|a).collect::<Vec<_>>())?);
    Ok(Loaded {spec:spec.clone(),witness:Arc::new(Witness {position,valid,example:Arc::new(example),successors:Default::default()}),support_sha256,actions})
}

// Reads is tied to this immutable row order. A changed registry requires a NEW
// ActivePanel/Reads, because the native value cache key does not contain rows.
struct ActivePanel {registry:Registry,rows:Vec<Arc<Witness>>,reads:read_cache::Reads}
impl ActivePanel {
    fn new(registry:Registry,loaded:&[Loaded])->Result<Self> {
        let rows=registry.entries.iter().map(|e| {
            let l=loaded.iter().find(|l|l.spec.key==e.key).ok_or_else(||invalid("restored dynamic proof unavailable"))?;
            if e.proof_sha256!=l.spec.proof.sha256||e.support_sha256!=l.support_sha256 {return Err(invalid("restored dynamic verified support changed"));}
            Ok(l.witness.clone())
        }).collect::<Result<Vec<_>>>()?;
        Ok(Self {registry,rows,reads:Default::default()})
    }
    fn measure(&self,m:&MicroModel,guard:&Guard)->Result<Score> {self.reads.evaluate(&self.rows,m,guard.beta,guard.parallel.as_ref())}
    fn losses(&self,s:&Score)->Result<Vec<serde_json::Value>> {
        if s.raw.len()!=self.rows.len()||s.coupled.len()!=self.rows.len() {return Err(invalid("dynamic score shape differs"));}
        Ok(self.registry.entries.iter().enumerate().filter_map(|(i,e)| {
            (!e.bits.retained(bits(s,i))).then(||serde_json::json!({"key":e.key,"raw_lost":e.bits.raw&&!s.raw[i],"coupled_lost":e.bits.coupled&&!s.coupled[i]}))
        }).collect())
    }
    fn logits(s:&Score,i:usize,coupled:bool)->Vec<f64> {
        if coupled {s.coupled_logits[i].to_vec()}else{s.priors[i].iter().map(|p|p.max(1e-300).ln()).collect()}
    }
    fn margins(&self,s:&Score)->Result<(f64,Option<(usize,usize,usize,f64)>)> {
        let mut sum=0.;let mut n=0;let mut worst=None;let mut largest=0.;
        for (i,e) in self.registry.entries.iter().enumerate() {
            for (enabled,coupled) in [(e.bits.raw,false),(e.bits.coupled,true)] {
                if !enabled {continue;}n+=1;let logits=Self::logits(s,i,coupled);
                if logits.len()!=self.rows[i].valid.len()||logits.iter().any(|x|!x.is_finite()) {return Err(invalid("invalid dynamic margin logits"));}
                let good=best(&logits,|a|self.rows[i].valid[a]).ok_or_else(||invalid("dynamic support empty"))?;
                if let Some(bad)=best(&logits,|a|!self.rows[i].valid[a]) {
                    let gap=(logits[bad]-logits[good]+EPS).max(0.);sum+=gap;
                    if gap>largest {largest=gap;worst=Some((i,good,bad,gap));}
                }
            }
        }Ok((sum/n.max(1) as f64,worst))
    }
}

fn margin_gradient(model:&MicroModel,row:&Witness,good:usize,bad:usize)->Result<Vec<f64>> {
    let mut ex=(*row.example).clone();ex.policy_support=false;ex.policy_weight=1.;ex.value_weight=0.;ex.action_values.clear();
    ex.policy.fill(0.);ex.policy[good]=1.;
    let mut g=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).map_err(invalid)?.1;
    ex.policy[good]=0.;ex.policy[bad]=1.;
    let b=model.loss_gradient_loop_v3_reusing(&ex,Vec::new()).map_err(invalid)?.1;
    for (i,(g,b)) in g.iter_mut().zip(b).enumerate() {*g=if branches::value_parameter(i) {0.}else{*g-b};}Ok(g)
}
fn repair(guard:&Guard,panel:&ActivePanel,initial:&MicroModel,candidate:&MicroModel,max_steps:usize)->Result<(MicroModel,serde_json::Value)> {
    let t=Instant::now();let mut model=candidate.clone();let mut checks=0;let mut attempts=vec![];
    let (_,before_reasons)=guard.checked_v3(&model)?;checks+=1;
    if !before_reasons.is_empty() {return Err(invalid("historical actor-001 does not pass the unchanged initial Guard"));}
    let mut score=panel.measure(&model,guard)?;let before=panel.losses(&score)?;
    let mut accepted_steps=0;
    for step in 0..max_steps {
        if panel.losses(&score)?.is_empty() {break;}
        let (before_merit,worst)=panel.margins(&score)?;
        let Some((row,good,bad,gap))=worst else {break;};
        if score.priors[row][good]<=1e-300||score.priors[row][bad]<=1e-300 {break;}
        let gradient=margin_gradient(&model,&panel.rows[row],good,bad)?;
        let result=margin_step::policy_step(&model,&gradient,gap,before_merit,|trial| {
            let (old_scores,reasons)=guard.checked_v3(trial)?;checks+=1;
            let next=panel.measure(trial,guard)?;let losses=panel.losses(&next)?;
            let retained_current=panel.registry.entries.iter().enumerate().all(|(i,e)|
                (!e.bits.raw||!score.raw[i]||next.raw[i])&&(!e.bits.coupled||!score.coupled[i]||next.coupled[i]));
            let merit=if reasons.is_empty()&&retained_current {panel.margins(&next)?.0}else{f64::INFINITY};
            let valid=reasons.is_empty()&&losses.is_empty()&&retained_current;
            attempts.push(serde_json::json!({"step":step+1,"check":checks,"old_reasons":reasons,"dynamic_lost":losses,
                "current_dynamic_choices_retained":retained_current,"merit":if merit.is_finite(){Some(merit)}else{None},"fully_valid":valid,
                "old_scores":old_scores,"value_bits_equal_candidate":value_bits(candidate,trial)?}));
            Ok(margin_step::Assessment {merit,fully_valid:valid,evidence:next})
        })?;
        let Some(result)=result else {break;};model=result.model;score=result.assessment.evidence;accepted_steps+=1;
    }
    let repaired=panel.losses(&score)?.is_empty();
    // A private partial repair is never an accepted actor. Exact original fallback.
    let chosen=if repaired {model}else{initial.clone()};
    let (old_scores,reasons)=guard.checked_v3(&chosen)?;checks+=1;
    let final_dynamic=panel.measure(&chosen,guard)?;
    if !reasons.is_empty()||!panel.losses(&final_dynamic)?.is_empty() {return Err(invalid("dynamic final fallback/repair is not admissible"));}
    if !repaired&&!same_parameters(&chosen,initial) {return Err(invalid("dynamic fallback lost original bits"));}
    Ok((chosen,serde_json::json!({"max_policy_steps":max_steps,"accepted_private_steps":accepted_steps,"checks":checks,
        "max_checks":2+6*max_steps,"repaired":repaired,"fallback_initial_exact":!repaired,"initial_losses":before,
        "final_dynamic":final_dynamic,"final_old_scores":old_scores,"attempts":attempts,"seconds":t.elapsed().as_secs_f64(),
        "scope":"extra choice checks only; original Guard unchanged; shadow repairs private; no new dynamic KL/MSE"})))
}

/// A standalone helper calls this only after the parent explicitly schedules it.
pub fn run(plan_path:&Path,out:&Path)->Result<serde_json::Value> {
    if out.exists() {return Err(invalid("dynamic diagnostic output must be new"));}
    let load_start=Instant::now();let plan_bytes=fs::read(plan_path)?;let plan:Plan=serde_json::from_slice(&plan_bytes)?;
    if plan.schema!="paisho-gen5-dynamic-proof-control-v1"||plan.proofs.len()!=6||plan.benchmark_repetitions>16
        ||plan.fraction_control&&plan.benchmark_repetitions!=0 {
        return Err(invalid("dynamic diagnostic requires the six fixed cycle-1 proofs and a bounded benchmark"));
    }
    let fixed_inputs=[&plan.initial,&plan.candidate,&plan.config,&plan.guard,&plan.validation,&plan.report,&plan.loop_plan,&plan.provenance];
    for input in fixed_inputs {input.bytes()?;}
    let provenance:serde_json::Value=serde_json::from_slice(&plan.provenance.bytes()?)?;
    let expected=provenance["all_new_proofs"].as_array().ok_or_else(||invalid("dynamic source provenance missing"))?
        .iter().filter(|p|p["first_selected_teaching"]["cycle"]==1&&p["target_value"]==1)
        .map(|p| {
            if p["known_corrective_prefix"]!=false||p["initial_fifo_state_seen"]!=false {return Err(invalid("cycle-1 proof was already corrective/in initial FIFO"));}
            let f=&p["first_selected_teaching"];
            if f["selected_fresh"]!=true||f["cycle"]!=1 {return Err(invalid("proof not actually selected in cycle1"));}
            Ok((f["receipt_order"].as_u64().ok_or_else(||invalid("missing receipt order"))? as usize,
                f["target_index"].as_u64().ok_or_else(||invalid("missing selected index"))? as usize,
                p["prefix_sha256"].as_str().ok_or_else(||invalid("missing prefix hash"))?.to_string(),
                f["proof_native"].clone(),f["targets"].clone(),f["decision"].clone()))
        }).collect::<Result<Vec<_>>>()?;
    let mut expected=expected;expected.sort_by(|a,b|(a.0,a.1,&a.2).cmp(&(b.0,b.1,&b.2)));
    if expected.len()!=plan.proofs.len() {return Err(invalid("dynamic plan dropped/added a selected cycle-1 proof"));}
    for (e,p) in expected.iter().zip(&plan.proofs) {
        if (e.0,e.1,e.2.as_str())!=(p.receipt_order,p.target_index,p.key.as_str())
            ||e.3!=serde_json::to_value(&p.proof)?||e.4!=serde_json::to_value(&p.targets)?||e.5!=p.decision {
            return Err(invalid("dynamic order/selected evidence changed"));
        }
    }
    if !plan.proofs.iter().any(|p|p.key==REQUIRED_KEY) {return Err(invalid("required 7ca diagnostic absent"));}
    let report:serde_json::Value=serde_json::from_slice(&plan.report.bytes()?)?;
    let lp:serde_json::Value=serde_json::from_slice(&plan.loop_plan.bytes()?)?;
    let receipts=report["cycles"][0]["receipts"].as_array().ok_or_else(||invalid("cycle1 receipts missing"))?;
    if report["cycles"][0]["cycle"]!=1||report["repaired_publication"]!=true {
        return Err(invalid("dynamic source was not the repaired baseline cycle1"));
    }
    let source_inputs=provenance["inputs"].as_array().ok_or_else(||invalid("missing source hashes"))?;
    if !source_inputs.contains(&serde_json::to_value(&plan.report)?)||!source_inputs.contains(&serde_json::to_value(&plan.loop_plan)?) {
        return Err(invalid("dynamic source report/plan differ from native provenance"));
    }
    let o:Options=serde_json::from_slice(&plan.config.bytes()?)?;
    if o.publication_guard.as_ref()!=Some(&plan.guard.path)||o.publication_validation.as_ref()!=Some(&plan.validation.path)
        ||!o.learning_loop_v2||o.value_policy_strength!=16. {
        return Err(invalid("dynamic source Guard/beta/config differs"));
    }
    let a:MicroArtifact=serde_json::from_slice(&plan.initial.bytes()?)?;
    let initial=a.model()?;let c:MicroArtifact=serde_json::from_slice(&plan.candidate.bytes()?)?;
    let candidate=load_snapshot(&plan.candidate.path,&c.identity(),1,&initial)?;
    if !initial.has_deep_value()||initial.parameters().len()!=292363||candidate.model.parameters().len()!=292363
        ||report["cycles"][0]["old_actor"].as_str()!=Some(a.identity().as_str())
        ||report["cycles"][0]["actor"].as_str()!=Some(candidate.identity.as_str()) {
        return Err(invalid("dynamic actor-000/001 provenance or architecture differs"));
    }
    let mut loaded=vec![];
    for p in &plan.proofs {
        if lp["recall_proofs"].as_array().ok_or_else(||invalid("initial proof list missing"))?.iter()
            .any(|q|q["path"].as_str().and_then(|s|Path::new(s).file_stem()).and_then(|s|s.to_str())==Some(p.key.as_str())) {
            return Err(invalid("dynamic new proof is in initial durable proof set"));
        }
        let receipt=receipts.get(p.receipt_order).ok_or_else(||invalid("missing proof receipt"))?;
        if receipt["actor"].as_str()!=Some(a.identity().as_str()) {return Err(invalid("cycle1 proof collector differs from initial actor"));}
        loaded.push(load_proof(p,&initial,receipt)?);
    }
    let initial_examples:Input=serde_json::from_value(lp["initial_examples"].clone())?;
    let fifo=decode_examples(&initial_examples.bytes()?)?;
    if loaded.iter().any(|l|fifo.iter().any(|s|s.state==l.witness.example.state)) {return Err(invalid("dynamic proof was already in initial FIFO"));}
    for input in [&plan.guard,&plan.validation] {
        let panel:Manifest=serde_json::from_slice(&input.bytes()?)?;
        if panel.rows.iter().any(|r|plan.proofs.iter().any(|p|sha256(r.prefix.as_bytes())==p.key)) {return Err(invalid("dynamic proof is inside original panels"));}
    }
    fs::create_dir(out)?;
    let snapshot=Arc::new(Snapshot {identity:a.identity(),artifact:Some(Arc::new(a.clone())),model:Arc::new(initial.clone()),version:0,path:plan.initial.path.clone()});
    let mut guard=Guard::open(&plan.guard.path,&out.join("guard"),o.value_policy_strength,snapshot,&serde_json::Value::Null)?;
    guard.enable_v2(&plan.validation.path)?;guard.enable_v3()?;
    if guard.rows.len()!=57||guard.validation.as_ref().map(|v|v.rows.len())!=Some(339) {return Err(invalid("expected unchanged 57/339 Guard panels"));}
    let pool=Arc::new(rayon::ThreadPoolBuilder::new().num_threads(10).build()?);guard.enable_parallel(&[pool]);
    let guard_before=guard.progress();let all_rows=loaded.iter().map(|l|l.witness.clone()).collect::<Vec<_>>();
    let admission_start=Instant::now();let all_reads=read_cache::Reads::default();
    let initial_scores=all_reads.evaluate(&all_rows,&initial,guard.beta,guard.parallel.as_ref())?;
    let mut registry=Registry::new(a.identity(),guard.beta)?;registry.begin_epoch(1)?;
    let mut admission=vec![];
    for (i,l) in loaded.iter().enumerate() {
        let succeeded=bits(&initial_scores,i);
        let active=registry.admit(Entry {key:l.spec.key.clone(),proof_sha256:l.spec.proof.sha256.clone(),support_sha256:l.support_sha256.clone(),
            acquired_actor:a.identity(),bits:succeeded,epoch:1})?;
        admission.push(serde_json::json!({"key":l.spec.key,"initial":succeeded,"active":active,
            "support":l.actions.iter().zip(&l.witness.valid).filter(|(_,v)|**v).map(|(a,_)|a).collect::<Vec<_>>(),"legal_actions":l.actions.len()}));
    }
    let admission_seconds=admission_start.elapsed().as_secs_f64();
    if registry.entries.is_empty() {return Err(invalid("no accepted successes to protect"));}
    let checkpoint=registry.checkpoint()?;fs::write(out.join("registry.json"),serde_json::to_vec_pretty(&checkpoint)?)?;
    let checkpoint_disk:serde_json::Value=serde_json::from_slice(&fs::read(out.join("registry.json"))?)?;
    let resumed=Registry::restore(&checkpoint_disk,&a.identity(),guard.beta)?;
    if resumed!=registry {return Err(invalid("dynamic registry resume differs"));}
    let panel=ActivePanel::new(resumed,&loaded)?;
    let restored_initial=panel.measure(&initial,&guard)?;
    if !panel.losses(&restored_initial)?.is_empty() {return Err(invalid("dynamic restored accepted bits differ"));}
    let candidate_dynamic=panel.measure(&candidate.model,&guard)?;
    let candidate_losses=panel.losses(&candidate_dynamic)?;
    let all_candidate_scores=all_reads.evaluate(&all_rows,&candidate.model,guard.beta,guard.parallel.as_ref())?;
    let (_,old_reasons)=guard.checked_v3(&candidate.model)?;
    if !old_reasons.is_empty() {return Err(invalid("baseline actor001 no longer satisfies unchanged Guard"));}
    if !candidate_losses.iter().any(|x|x["key"]==REQUIRED_KEY&&x["coupled_lost"]==true) {return Err(invalid("expected native 7ca coupled loss not reproduced"));}
    let loading_seconds=load_start.elapsed().as_secs_f64();let mut repairs=vec![];
    for steps in if plan.fraction_control {Vec::new()}else{vec![2,4]} {
        let (chosen,result)=repair(&guard,&panel,&initial,&candidate.model,steps)?;
        let unchanged=parameter_sha(&chosen);let artifact=MicroArtifact::new(&chosen,if result["repaired"]==true {c.updates}else{a.updates},
            serde_json::json!({"diagnostic_only":true,"dynamic_proof_control":true,"max_steps":steps,"no_campaign_or_publication":true}));
        let path=out.join(format!("diagnostic-candidate-{steps}.json"));fs::write(&path,serde_json::to_vec_pretty(&artifact)?)?;
        let all_final_scores=all_reads.evaluate(&all_rows,&chosen,guard.beta,guard.parallel.as_ref())?;
        repairs.push(serde_json::json!({"repair":result,"model":path,"parameter_sha256":unchanged,
            "initial_parameters_preserved_if_fallback":same_parameters(&chosen,&initial),
            "all_six_vs_actor001":paired_choices(&loaded,&all_candidate_scores,&all_final_scores),
            "all_six_vs_actor000":paired_choices(&loaded,&initial_scores,&all_final_scores),
            "candidate_value_bits_if_repaired":if result["repaired"]==true {Some(value_bits(&candidate.model,&chosen)?)}else{None}}));
    }
    let fraction_result=if plan.fraction_control {Some(fractions::run(&guard,&panel,&loaded,&all_reads,
        &initial,&candidate.model,receipts,a.updates,c.updates,out)?)}else{None};
    // Optional causal comparison measures just the additional finite check.
    // Model copying stays OUTSIDE timers; fresh storage avoids score-cache hits.
    // Successor/value tables are resident in both modes. Admission is separate.
    let mut benchmark=vec![];
    for mode in [false,true,true,false] {
        let mut elapsed=0.;let mut scores_digest=None;let mut losses_digest=None;
        for _ in 0..plan.benchmark_repetitions {
            let fresh=independent_copy(&candidate.model)?;let t=Instant::now();
            let (scores,reasons)=guard.checked_v3(&fresh)?;
            let losses=if mode {Some(panel.losses(&panel.measure(&fresh,&guard)?)?)}else{None};
            elapsed+=t.elapsed().as_secs_f64();
            if !reasons.is_empty() {return Err(invalid("benchmark changed original finite result"));}
            let digest=score_sha(&scores)?;
            if scores_digest.as_ref().is_some_and(|old|old!=&digest) {return Err(invalid("benchmark original scores changed"));}scores_digest=Some(digest);
            if let Some(losses)=losses {if losses!=candidate_losses {return Err(invalid("benchmark dynamic losses changed"));}losses_digest=Some(sha256(&serde_json::to_vec(&losses)?));}
        }
        benchmark.push(serde_json::json!({"dynamic_control":mode,"repetitions":plan.benchmark_repetitions,"seconds":elapsed,
            "seconds_per_check":if plan.benchmark_repetitions>0 {Some(elapsed/plan.benchmark_repetitions as f64)}else{None},"old_scores_sha256":scores_digest,"dynamic_losses_sha256":losses_digest}));
    }
    if plan.benchmark_repetitions>0&&benchmark.iter().any(|b|b["old_scores_sha256"]!=benchmark[0]["old_scores_sha256"]) {
        return Err(invalid("ABBA changed original score fields"));
    }
    if guard.progress()!=guard_before {return Err(invalid("dynamic diagnostic mutated original Guard"));}
    for input in fixed_inputs {input.bytes()?;}initial_examples.bytes()?;
    for p in &plan.proofs {p.proof.bytes()?;p.targets.bytes()?;}
    if fs::read(plan_path)?!=plan_bytes {return Err(invalid("dynamic plan changed during diagnostic"));}
    let result=serde_json::json!({"schema":"paisho-gen5-dynamic-proof-control-result-v1","diagnostic_only":true,
        "plan":plan_path,"plan_sha256":sha256(&plan_bytes),"inputs":plan,"loading_seconds_excluded":loading_seconds,
        "admission_seconds":admission_seconds,"admissions":admission,"registry_checkpoint":checkpoint,
        "all_six_actor000_to_actor001":paired_choices(&loaded,&initial_scores,&all_candidate_scores),
        "registry_resumed_exact":true,"actor001_dynamic":candidate_dynamic,"actor001_losses":candidate_losses,
        "fraction_control":fraction_result,"old_guard_unchanged":true,"initial_actor_admissible":true,"actor001_old_guard_admissible":true,"repairs":repairs,"abba":benchmark,
        "limitations":["known six cycle1 development proofs only","no promotion or runtime integration","no acquisition of unproved or merely estimated targets",
            "fallback may preserve7ca by refusing progress rather than repairing it","bounded active entries only; no global or permanent retention guarantee",
            "coupled prior ranking is not full MCTS strength","initial proof/FIFO/panel exclusion is not absence from lineage or sequence bank",
            "ABBA compares extra finite-control overhead, not complete campaign throughput or repair efficiency"]});
    fs::write(out.join("report.json"),serde_json::to_vec_pretty(&result)?)?;Ok(result)
}

/// Bounded diagnostic adapter sharing the existing native certificate/Saved
/// verifier; additional capsule metadata never substitutes for those checks.
pub(super) fn transfer_witness(spec:&serde_json::Value,model:&MicroModel,receipt:&serde_json::Value)->Result<(Arc<Witness>,serde_json::Value)> {
    if spec["collector_identity"]!=receipt["actor"] || spec["source_group"]!=receipt["source_group"] {
        return Err(invalid("transfer capsule collector/source differs from receipt"));
    }
    let core=serde_json::json!({"key":spec["key"],"receipt_order":spec["receipt_order"],"target_index":spec["target_index"],
        "decision":spec["decision"],"proof":spec["proof"],"targets":spec["targets"],"native_source":spec["native_source"]});
    let core:ProofSpec=serde_json::from_value(core)?;let loaded=load_proof(&core,model,receipt)?;
    let meta=serde_json::json!({"key":loaded.spec.key,"actions":loaded.actions,"support_sha256":loaded.support_sha256,
        "valid":loaded.witness.valid,"native_certificate_verified":true,"saved_support_q_features_collector_verified":true,
        "native_decision_provenance":loaded.spec.native_source});
    Ok((loaded.witness,meta))
}

#[cfg(test)]
mod native_provenance_tests {
    use super::*;

    fn historical()->serde_json::Value {
        serde_json::json!({"collector":"a".repeat(64),"opponent":"Gen3.3","candidate_seat":"Host",
            "lane":"Historical","reanalysis":false,"reference_budget":32,"reference_identity":"b".repeat(64),
            "case":{"opponent_generation":"3.3"}})
    }
    fn evidence(actor:String,player:&str)->TargetEvidence {
        TargetEvidence {policy_support:false,policy_coordinates:String::new(),search_prior:vec![],coupling_strength:None,
            observed_value:None,observed_psr:None,estimated_value:None,value_weight:0.,policy_source:"verified-regulatory-win".into(),
            completed_action_values:vec![],action_value_visits:vec![],target_prior:vec![],excluded_actions:vec![],player:player.into(),actor}
    }
    #[test]
    fn native_historical_opponent_actor_is_distinct_from_collector() {
        let r=historical();
        let guest=native_decision_actor(&r,Player::Guest).unwrap();
        assert_eq!(guest,format!("Gen3.3:{}","b".repeat(64)));
        assert_eq!(native_decision_actor(&r,Player::Host).unwrap(),"a".repeat(64));
        let e=evidence(guest.clone(),"G");
        assert!(decision_identity_matches(Some(&e),&guest,Some(Player::Guest)));
        assert!(!decision_identity_matches(Some(&e),&"a".repeat(64),Some(Player::Guest)));
        assert!(!decision_identity_matches(Some(&e),&guest,Some(Player::Host)));
        let wrong=evidence(format!("Gen3.3:{}","c".repeat(64)),"G");
        assert!(!decision_identity_matches(Some(&wrong),&guest,Some(Player::Guest)));
    }
    #[test]
    fn native_provenance_rejects_seat_lane_reference_and_generation_changes() {
        let mut r=historical();r["candidate_seat"]="Guest".into();
        assert_eq!(native_decision_actor(&r,Player::Guest).unwrap(),"a".repeat(64));
        for (field,value) in [("candidate_seat",serde_json::json!("bad")),("lane",serde_json::json!("Selfplay")),
            ("reference_identity",serde_json::Value::Null),("reference_budget",serde_json::json!(0)),
            ("opponent",serde_json::json!("Gen3.4")),("reanalysis",serde_json::json!(true))] {
            let mut changed=historical();changed[field]=value;
            assert!(native_decision_actor(&changed,Player::Guest).is_err(),"{field}");
        }
    }
    #[test]
    fn native_selfplay_checkpoint_and_reanalysis_follow_actual_seat() {
        let mut r=serde_json::json!({"collector":"a","opponent":"a","candidate_seat":"Guest",
            "lane":"Selfplay","reanalysis":false,"reference_budget":null,"reference_identity":null});
        assert_eq!(native_decision_actor(&r,Player::Host).unwrap(),"a");
        r["opponent"]="b".into();assert!(native_decision_actor(&r,Player::Host).is_err());
        r["lane"]="Checkpoint".into();
        assert_eq!(native_decision_actor(&r,Player::Host).unwrap(),"b");
        assert_eq!(native_decision_actor(&r,Player::Guest).unwrap(),"a");
        r["lane"]="Reanalysis".into();r["reanalysis"]=true.into();
        assert_eq!(native_decision_actor(&r,Player::Host).unwrap(),"b");
    }
    #[test]
    fn historical_proof_spec_and_actor_contract_remain_unchanged_without_native_source() {
        let v=serde_json::json!({"key":"k","receipt_order":1,"target_index":0,"decision":1,
            "proof":{"path":"p","sha256":"h"},"targets":{"path":"t","sha256":"j"}});
        let s:ProofSpec=serde_json::from_value(v.clone()).unwrap();
        assert!(s.native_source.is_none());assert_eq!(serde_json::to_value(s).unwrap(),v);
        let e=evidence("legacy".into(),"unrecorded");
        assert!(decision_identity_matches(Some(&e),"legacy",None));
        assert!(!decision_identity_matches(Some(&e),"other",None));
        assert!(!decision_identity_matches(None,"legacy",None));
    }
    #[test]
    #[ignore = "requires frozen C1 diagnostic artifacts; schedule explicitly with --include-ignored"]
    fn native_receipt_1107311_authenticates_real_opponent_decision_without_model() {
        let dir=Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/results/gen5-loop-repair-2026-09-13/full-throughput-ab/runs/c1/training/games");
        let receipt=Input {path:dir.join("game-1107311.json"),sha256:"0bc9fcfaaf4b13a4cdc11f08430c983ea9e6a13820526abaf02c213be3417800".into()};
        let psr=Input {path:dir.join("game-1107311.psr"),sha256:"be810f47d0cbeefb2b2e68535a7edc1f6e9fafbf0eb9c1506c38c34afc2c3901".into()};
        let targets=Input {path:dir.join("game-1107311.targets.json.gz"),sha256:"7a311c95578fffdcfed48a65611912b943c68db61630aed7948fdc889c662a25".into()};
        let r:serde_json::Value=serde_json::from_slice(&receipt.bytes().unwrap()).unwrap();
        let record:GameRecord=std::str::from_utf8(&psr.bytes().unwrap()).unwrap().parse().unwrap();
        let prefix=cases::prefix(&record,36).to_string();
        let position:GameRecord=prefix.parse().unwrap();let player=position.replay().unwrap().to_move();
        assert_eq!(player,Player::Guest);
        let rows=decode_examples(&targets.bytes().unwrap()).unwrap();let saved=&rows[2];
        assert_eq!(saved.game_id,"1107311");assert_eq!(saved.decision,37);
        assert_eq!(r["candidate_seat"],"Host");assert_eq!(r["targets_file"],"game-1107311.targets.json.gz");
        let adapter=serde_json::json!({"actor":r["collector"],"source_group":r["psr_sha256"]});
        let spec=ProofSpec {key:sha256(prefix.as_bytes()),receipt_order:0,target_index:2,decision:37,proof:targets.clone(),targets,
            native_source:Some(NativeSource {receipt,psr})};
        let actor=native_actor(&spec,&adapter,&prefix,player,saved).unwrap().unwrap();
        assert_eq!(actor,"Gen3.3:f71cf7dfc125611dd3c9097c412ab7c798a6529431db553fc8d02c94014e38e1");
        assert!(decision_identity_matches(saved.evidence.as_ref(),&actor,Some(player)));
        assert!(!decision_identity_matches(saved.evidence.as_ref(),&saved.collector,None));
        let mut bad_saved=saved.clone();bad_saved.game_id="1107312".into();
        assert!(native_actor(&spec,&adapter,&prefix,player,&bad_saved).is_err());
        assert!(native_actor(&spec,&adapter,&cases::prefix(&record,35).to_string(),player,saved).is_err());
        let mut bad_hash=spec.clone();bad_hash.native_source.as_mut().unwrap().receipt.sha256="0".repeat(64);
        assert!(native_actor(&bad_hash,&adapter,&prefix,player,saved).is_err());
    }
}
