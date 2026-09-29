//! Offline Gen5 structural audit. Reads frozen evidence; writes only a new output directory.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{Read, Write},
    path::Path,
    sync::Arc,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[path = "micro_structural_probe/repair.rs"]
mod repair;

struct Case {
    source: String,
    group: String,
    prefix: GameRecord,
    position: Position,
    example: MicroExample,
    wins: Vec<usize>,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn prefix(record: &GameRecord, count: usize) -> GameRecord {
    let mut p = GameRecord::with_rules(record.setup(), record.rules());
    for &a in record.actions().iter().take(count) {
        p.push(a);
    }
    p
}
fn load(manifest: &Value) -> Result<Vec<Case>> {
    let mut cases = vec![];
    for row in manifest["rows"].as_array().ok_or("rows")? {
        let bytes = fs::read(row["path"].as_str().ok_or("path")?)?;
        if hash(&bytes) != row["sha256"].as_str().ok_or("hash")? {
            return Err("bundle hash".into());
        }
        let mut decoded = String::new();
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_string(&mut decoded)?;
        let b: Value = serde_json::from_str(&decoded)?;
        let psr = b["psr"].as_str().ok_or("psr")?;
        if hash(psr.as_bytes()) != b["psr_sha256"].as_str().ok_or("psr hash")? {
            return Err("PSR hash".into());
        }
        let record: GameRecord = psr.parse()?;
        record.replay()?;
        let lessons: Vec<_> = b["lessons"]
            .as_array()
            .ok_or("lessons")?
            .iter()
            .filter(|l| l["policy_weight"].as_f64().unwrap_or(0.) > 0.)
            .collect();
        let mut chosen: Vec<_> = (0..4.min(lessons.len()))
            .map(|i| i * (lessons.len() - 1) / 3.min(lessons.len() - 1).max(1))
            .collect();
        chosen.sort();
        chosen.dedup();
        for i in chosen {
            let l = lessons[i];
            let d = l["decision"].as_u64().ok_or("decision")? as usize;
            if d == 0 || d > record.actions().len() + 1 {
                return Err("decision bounds".into());
            }
            let p = prefix(&record, d - 1);
            let position = p.replay()?;
            if position.outcome() != GameOutcome::Ongoing {
                return Err("terminal lesson".into());
            }
            let legal = legal_actions(&position);
            let ids: HashMap<_, _> = legal
                .iter()
                .enumerate()
                .map(|(i, a)| (a.to_string(), i))
                .collect();
            let mut target = vec![0.; legal.len()];
            for pair in l["policy"].as_array().ok_or("policy")? {
                target[*ids
                    .get(pair[0].as_str().ok_or("action")?)
                    .ok_or("illegal target")?] = pair[1].as_f64().ok_or("mass")?;
            }
            let example = MicroExample { policy_support: false, action_values: vec![], 
                value_weight: 1.0, sequence_source: sequence_source(&format!(
                    "{}/{}",
                    b["source"].as_str().ok_or("source")?,
                    b["game_id"]
                )),
                state: micro_state_features(&position).to_vec(),
                actions: legal
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect(),
                policy: target,
                value: l["value"].as_f64().ok_or("value")?,
                policy_weight: 1.,
            };
            example.validate()?;
            let wins = immediate(&position, &legal)?;
            cases.push(Case {
                source: row["source"].as_str().ok_or("source")?.into(),
                group: row["group"].as_str().ok_or("group")?.into(),
                prefix: p,
                position,
                example,
                wins,
            });
        }
    }
    Ok(cases)
}
fn immediate(p: &Position, actions: &[Action]) -> Result<Vec<usize>> {
    let mut wins = vec![];
    for (i, &a) in actions.iter().enumerate() {
        let mut q = p.clone();
        q.apply(a)?;
        if q.outcome() == GameOutcome::Win(p.to_move()) {
            wins.push(i);
        }
    }
    Ok(wins)
}
fn argmax(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn probabilities(m: &MicroModel, e: &MicroExample) -> Result<Vec<f64>> {
    let base = micro_softmax(&MicroModel::logits(&m.embed(&e.state), &e.actions))?;
    Ok(m.memory_priors(&e.state, &e.actions, &base, 0)?)
}
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let aa = a.iter().map(|x| x * x).sum::<f64>();
    let bb = b.iter().map(|x| x * x).sum::<f64>();
    if aa * bb == 0. {
        0.
    } else {
        a.iter().zip(b).map(|(a, b)| a * b).sum::<f64>() / (aa * bb).sqrt()
    }
}
fn panel(cases: &[Case], model: &Arc<MicroModel>, out: &Path) -> Result<()> {
    let mut file = fs::File::create(out.join("panel.jsonl"))?;
    let mut zero = model.parameters().to_vec();
    zero[MICRO_RESIDUAL_PARAMETERS..].fill(0.);
    let no_reader = MicroModel::from_parameters(zero)?
        .with_sequence_memory(model.sequence_memory().unwrap().clone());
    let mut gradients = vec![];
    for (i, c) in cases.iter().enumerate() {
        let e = &c.example;
        let emb = model.embed(&e.state);
        let base = micro_softmax(&MicroModel::logits(&emb, &e.actions))?;
        let p = probabilities(model, e)?;
        let context = model.memory_context(&e.state, 0)?.unwrap();
        let mut searches = vec![];
        for budget in [64, 256, 512] {
            let r = MicroMctsSession::new(model.clone()).search_with_options(
                &c.position,
                budget,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    ..Default::default()
                },
            )?;
            searches.push(json!({"budget":budget,"selected":r.selected_index,"simulations":r.simulations,"proven":r.proven_value,"winning_target_mass":c.wins.iter().map(|&j|r.policy_target[j]).sum::<f64>(),"winning_prior_mass":c.wins.iter().map(|&j|r.priors[j]).sum::<f64>(),"coverage":r.new_visits.iter().filter(|n|**n>0).count(),"actions":r.actions.len(),"new_visits":r.new_visits,"policy":r.policy_target}));
        }
        let mut noise = vec![];
        if !c.wins.is_empty() {
            for seed in 0..8 {
                let r = MicroMctsSession::new(model.clone()).search_with_options(
                    &c.position,
                    512,
                    None,
                    MicroSearchOptions {
                        proof_search: true,
                        seed,
                        dirichlet_fraction: 0.25,
                        forced_playout_strength: 2.,
                        ..Default::default()
                    },
                )?;
                noise.push(json!({"seed":seed,"selected_wins":c.wins.contains(&r.selected_index),"winning_target_mass":c.wins.iter().map(|&j|r.policy_target[j]).sum::<f64>(),"coverage":r.new_visits.iter().filter(|n|**n>0).count(),"simulations":r.simulations}));
            }
        }
        let (loss, g) = model.loss_gradient(e)?;
        let mut v = e.clone();
        v.actions.clear();
        v.policy.clear();
        v.policy_weight = 0.;
        let (_, gv) = model.loss_gradient(&v)?;
        let gp: Vec<_> = g.iter().zip(&gv).map(|(a, b)| a - b).collect();
        gradients.push(g);
        let no_memory = probabilities(&no_reader, e)?;
        let row = json!({"index":i,"source":c.source,"group":c.group,"decision":c.prefix.actions().len()+1,"phase":format!("{:?}",c.position.phase()),"state":e.state.as_slice(),"action_features":e.actions,"action_names":legal_actions(&c.position).iter().map(ToString::to_string).collect::<Vec<_>>(),"target":e.policy,"value_target":e.value,"value":emb.value,"loss_value":loss.value,"loss_policy":loss.policy,"hidden":emb.hidden,"policy":p,"base_policy":base,"policy_pick":argmax(&p),"no_reader_pick":argmax(&no_memory),"memory_l1":p.iter().zip(&no_memory).map(|(a,b)|(a-b).abs()).sum::<f64>(),"memory_neighbors":context.neighbors.len(),"memory_confidence":context.confidence,"wins":c.wins,"searches":searches,"training_searches":noise,"value_policy_trunk_cosine":cosine(&gv[..4128],&gp[..4128]),"value_trunk_norm":gv[..4128].iter().map(|v|v*v).sum::<f64>().sqrt(),"policy_trunk_norm":gp[..4128].iter().map(|v|v*v).sum::<f64>().sqrt()});
        writeln!(file, "{row}")?;
        if i % 20 == 0 {
            eprintln!("panel {i}/{}", cases.len());
        }
    }
    fs::write(
        out.join("gradient-cosines.json"),
        serde_json::to_vec(&json!((0..cases.len())
            .map(|i| (0..cases.len())
                .map(|j| cosine(&gradients[i], &gradients[j]))
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()))?,
    )?;
    Ok(())
}
fn aliases(cases: &[Case], model: &MicroModel, out: &Path) -> Result<()> {
    let mut groups = 0;
    let mut children = 0;
    let mut witnesses = vec![];
    for (ci, c) in cases.iter().enumerate() {
        let mut seen: BTreeMap<Vec<u64>, Vec<(Action, Position)>> = BTreeMap::new();
        for a in legal_actions(&c.position) {
            let mut p = c.position.clone();
            p.apply(a)?;
            if p.outcome() != GameOutcome::Ongoing {
                continue;
            }
            children += 1;
            let x = micro_state_features(&p);
            seen.entry(
                x.iter()
                    .map(|x| if *x == 0. { 0 } else { x.to_bits() })
                    .collect(),
            )
            .or_default()
            .push((a, p));
        }
        for group in seen.values().filter(|g| g.len() > 1) {
            groups += 1;
            if witnesses.len() >= 8 {
                continue;
            }
            let a = &group[0];
            let aa = legal_actions(&a.1);
            let aw = immediate(&a.1, &aa)?;
            for b in &group[1..] {
                let ba = legal_actions(&b.1);
                let bw = immediate(&b.1, &ba)?;
                let mut differences = vec![];
                for (ai, action) in aa.iter().enumerate() {
                    if let Some(bi) = ba.iter().position(|b| b == action) {
                        if aw.contains(&ai) != bw.contains(&bi)
                            && micro_action_features(&a.1, *action)
                                == micro_action_features(&b.1, *action)
                        {
                            differences.push(json!({"action":action.to_string(),"wins_a":aw.contains(&ai),"wins_b":bw.contains(&bi)}));
                        }
                    }
                }
                if !differences.is_empty() {
                    let n = witnesses.len();
                    let mut ar = c.prefix.clone();
                    ar.push(a.0);
                    let mut br = c.prefix.clone();
                    br.push(b.0);
                    assert_eq!(ar.replay()?, a.1);
                    assert_eq!(br.replay()?, b.1);
                    let ax = micro_state_features(&a.1);
                    let bx = micro_state_features(&b.1);
                    assert_eq!(ax, bx);
                    assert_eq!(model.embed(&ax).value, model.embed(&bx).value);
                    fs::write(out.join(format!("alias-{n}-a.psr")), ar.to_string())?;
                    fs::write(out.join(format!("alias-{n}-b.psr")), br.to_string())?;
                    witnesses.push(json!({"case":ci,"source":c.source,"prefix_a":format!("alias-{n}-a.psr"),"prefix_b":format!("alias-{n}-b.psr"),"sibling_a":a.0.to_string(),"sibling_b":b.0.to_string(),"same_state_features":true,"network_value":model.embed(&ax).value,"actions_a":aa.len(),"actions_b":ba.len(),"winning_actions_a":aw.iter().map(|&i|aa[i].to_string()).collect::<Vec<_>>(),"winning_actions_b":bw.iter().map(|&i|ba[i].to_string()).collect::<Vec<_>>(),"same_action_input_different_immediate_result":differences}));
                    break;
                }
            }
        }
        if ci % 20 == 0 {
            eprintln!("aliases {ci}/{} witnesses {}", cases.len(), witnesses.len());
        }
    }
    fs::write(
        out.join("aliases.json"),
        serde_json::to_vec_pretty(
            &json!({"children":children,"colliding_sibling_groups":groups,"witnesses":witnesses}),
        )?,
    )?;
    Ok(())
}
fn metrics(model: &MicroModel, cases: &[Case]) -> Result<Value> {
    let mut groups = serde_json::Map::new();
    for group in ["learn", "interference", "heldout"] {
        let mut loss = 0.;
        let mut value = 0.;
        let mut mass = 0.;
        let mut wins = 0;
        let mut n = 0;
        let mut tn = 0;
        for c in cases.iter().filter(|c| c.group == group) {
            let l = model.loss_gradient(&c.example)?.0;
            loss += l.policy;
            value += l.value;
            n += 1;
            if !c.wins.is_empty() {
                let p = probabilities(model, &c.example)?;
                mass += c.wins.iter().map(|&i| p[i]).sum::<f64>();
                wins += usize::from(c.wins.contains(&argmax(&p)));
                tn += 1;
            }
        }
        groups.insert(group.into(),json!({"n":n,"policy_ce":loss/n as f64,"half_mse":value/n as f64,"tactics":tn,"tactical_hits":wins,"tactical_mass":if tn>0 {mass/tn as f64}else{0.}}));
    }
    Ok(Value::Object(groups))
}
fn retention(cases: &[Case], model: &MicroModel, out: &Path) -> Result<()> {
    let mut result = vec![];
    let a: Vec<_> = cases.iter().filter(|c| c.group == "learn").collect();
    let b: Vec<_> = cases.iter().filter(|c| c.group == "interference").collect();
    for seed in 0..3 {
        let mut rng = StableRng::new(seed);
        let mut learned = model.clone();
        let initial = metrics(&learned, cases)?;
        for _ in 0..1000 {
            let batch: Vec<_> = (0..16).map(|_| &a[rng.index(a.len())].example).collect();
            learned.train_batch_inline(&batch, 0.02, 1e-5)?;
        }
        let after_a = metrics(&learned, cases)?;
        for fraction in [0., 0.1, 0.5] {
            let mut m = learned.clone();
            let mut rng = StableRng::new(seed + 100);
            for _ in 0..1000 {
                let batch: Vec<_> = (0..16)
                    .map(|_| {
                        let source = if rng.next_f64() < fraction { &a } else { &b };
                        &source[rng.index(source.len())].example
                    })
                    .collect();
                m.train_batch_inline(&batch, 0.02, 1e-5)?;
            }
            result.push(json!({"seed":seed,"recall_fraction":fraction,"initial":initial,"after_a":after_a,"after_b":metrics(&m,cases)?}));
            eprintln!("retention seed {seed} fraction {fraction}");
        }
    }
    fs::write(
        out.join("retention.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
fn tactical_metrics(model: &MicroModel, cases: &[&Case]) -> Result<Value> {
    let mut hits = 0;
    let mut mass = 0.;
    let mut search_hits = 0;
    for c in cases {
        let p = probabilities(model, &c.example)?;
        hits += usize::from(c.wins.contains(&argmax(&p)));
        mass += c.wins.iter().map(|&j| p[j]).sum::<f64>();
        let r = MicroMctsSession::new(Arc::new(model.clone())).search_with_options(
            &c.position,
            512,
            None,
            MicroSearchOptions {
                proof_search: true,
                ..Default::default()
            },
        )?;
        search_hits += usize::from(c.wins.contains(&r.selected_index));
    }
    Ok(
        json!({"positions":cases.len(),"policy_hits":hits,"winning_mass":mass/cases.len() as f64,"mcts512_hits":search_hits}),
    )
}
fn certified_fit(cases: &[Case], model: &MicroModel, out: &Path) -> Result<()> {
    let tactics: Vec<_> = cases.iter().filter(|c| !c.wins.is_empty()).collect();
    let sources: std::collections::HashSet<_> = tactics.iter().map(|c| &c.source).collect();
    let background: Vec<_> = cases
        .iter()
        .filter(|c| !sources.contains(&c.source))
        .map(|c| &c.example)
        .collect();
    let targets: Vec<_> = tactics
        .iter()
        .map(|c| {
            let mut e = c.example.clone();
            e.value = 1.;
            e.policy.fill(0.);
            for &i in &c.wins {
                e.policy[i] = 1. / c.wins.len() as f64;
            }
            e
        })
        .collect();
    let initial = tactical_metrics(model, &tactics)?;
    let mut runs = vec![];
    for seed in 0..3 {
        let mut rng = StableRng::new(seed);
        let mut fit = model.clone();
        for _ in 0..2000 {
            let batch: Vec<_> = (0..16)
                .map(|_| &targets[rng.index(targets.len())])
                .collect();
            fit.train_batch_inline(&batch, 0.02, 1e-5)?;
        }
        let fitted = tactical_metrics(&fit, &tactics)?;
        MicroArtifact::new(&fit,0,json!({"kind":"diagnostic-only-certified-fit","seed":seed,"not_a_production_checkpoint":true})).save(&out.join(format!("diagnostic-fit-{seed}.json")))?;
        for recall in [0., 0.1, 0.5] {
            let mut m = fit.clone();
            let mut rng = StableRng::new(seed + 100);
            for _ in 0..1000 {
                let batch: Vec<_> = (0..16)
                    .map(|_| {
                        if rng.next_f64() < recall {
                            &targets[rng.index(targets.len())]
                        } else {
                            background[rng.index(background.len())]
                        }
                    })
                    .collect();
                m.train_batch_inline(&batch, 0.02, 1e-5)?;
            }
            runs.push(json!({"seed":seed,"recall":recall,"initial":initial,"fitted":fitted,"after_background":tactical_metrics(&m,&tactics)?}));
            eprintln!("certified fit {seed} recall {recall}");
        }
    }
    fs::write(
        out.join("certified-fit.json"),
        serde_json::to_vec_pretty(
            &json!({"tactics":tactics.len(),"background_positions":background.len(),"source_disjoint_background":true,"steps_fit":2000,"steps_background":1000,"batch":16,"rate":0.02,"runs":runs}),
        )?,
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len())
        || args
            .get(4)
            .is_some_and(|s| s != "certified" && s != "aliases" && s != "reader" && s != "repair")
    {
        return Err(
            "micro_structural_probe MANIFEST MODEL NEW_OUTPUT [certified|aliases|reader]".into(),
        );
    }
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let bytes = fs::read(&args[2])?;
    let artifact = MicroArtifact::load(Path::new(&args[2]))?;
    let model = Arc::new(artifact.model()?);
    let mut cases = load(&manifest)?;
    if args.get(4).is_some_and(|s| s == "repair") {
        for case in &mut cases {
            case.example.state = model.state_features(&case.position);
            // Frozen audit bundles predate source_run: match repaired rehearsal.
            case.example.sequence_source = u64::MAX;
        }
    }
    let bank = model
        .sequence_memory()
        .ok_or("current model must have sequence memory")?;
    fs::write(
        out.join("identity.json"),
        serde_json::to_vec_pretty(
            &json!({"model_sha256":hash(&bytes),"model":args[2],"updates":artifact.updates,"parameters":model.parameters().len(),"manifest_sha256":hash(&fs::read(&args[1])?),"cases":cases.len(),"bank":bank.spec,"bank_entries":bank.entries.len(),"bank_main_entries":bank.entries.iter().filter(|e|e.phase==0).count(),"bank_bonus_entries":bank.entries.iter().filter(|e|e.phase==1).count(),"reader_weights":&model.parameters()[MICRO_RESIDUAL_PARAMETERS..MICRO_MEMORY_PARAMETERS]}),
        )?,
    )?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    if args.get(4).is_some_and(|s| s == "repair") {
        return pool.install(|| repair::run(&cases, &model, out).map_err(|e|e.to_string())).map_err(Into::into);
    }
    if args.get(4).is_some_and(|s| s == "reader") {
        let mut zero = model.parameters().to_vec();
        zero[MICRO_RESIDUAL_PARAMETERS..].fill(0.);
        let no_reader =
            Arc::new(MicroModel::from_parameters(zero)?.with_sequence_memory(bank.clone()));
        let mut rows = vec![];
        for (i, c) in cases.iter().enumerate() {
            let mut results = vec![];
            for m in [&model, &no_reader] {
                let r = pool.install(|| {
                    MicroMctsSession::new(m.clone()).search_with_options(
                        &c.position,
                        512,
                        None,
                        MicroSearchOptions {
                            proof_search: true,
                            ..Default::default()
                        },
                    )
                })?;
                results.push(json!({"selected":r.selected_index,"wins":c.wins.contains(&r.selected_index),"simulations":r.simulations}));
            }
            rows.push(json!({"index":i,"with_reader":results[0],"zero_reader":results[1]}));
        }
        fs::write(
            out.join("reader-ablation.json"),
            serde_json::to_vec_pretty(&rows)?,
        )?;
        return Ok(());
    }
    if args.get(4).is_some_and(|s| s == "aliases") {
        return aliases(&cases, &model, out);
    }
    if args.get(4).is_some_and(|s| s == "certified") {
        pool.install(|| certified_fit(&cases, &model, out).map_err(|e| e.to_string()))?;
        return Ok(());
    }
    pool.install(|| panel(&cases, &model, out).map_err(|e| e.to_string()))?;
    aliases(&cases, &model, out)?;
    retention(&cases, &model, out)?;
    Ok(())
}
