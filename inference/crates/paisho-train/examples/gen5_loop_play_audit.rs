//! Frozen, paired behavior intervention. No learning, no production writes.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::{compact_learning, micro_learning::MicroArtifact};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs,
    path::Path,
    sync::Arc,
    time::Instant,
};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn same(a: &Position, b: &Position) -> bool {
    a.rule_profile() == b.rule_profile()
        && a.board() == b.board()
        && a.reserve(Player::Host) == b.reserve(Player::Host)
        && a.reserve(Player::Guest) == b.reserve(Player::Guest)
        && a.to_move() == b.to_move()
        && a.phase() == b.phase()
        && a.outcome() == b.outcome()
}
fn cycle(history: &VecDeque<Position>) -> bool {
    (2..=32).any(|period| {
        let span = 4 * period;
        span >= 24
            && history.len() > span
            && (history.len() - span - 1 + period..history.len())
                .all(|i| same(&history[i], &history[i - period]))
    })
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("MANIFEST OUTPUT_DIRECTORY".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let output = Path::new(&args[2]);
    fs::create_dir(output)?;
    let artifact: MicroArtifact =
        serde_json::from_slice(&fs::read(manifest["model"].as_str().ok_or("model")?)?)?;
    let model = Arc::new(artifact.model()?);
    let reference = compact_learning::load_model(Path::new(
        manifest["reference"].as_str().ok_or("reference")?,
    ))?
    .model()?;
    let proof_dir = Path::new(manifest["proofs"].as_str().ok_or("proofs")?);
    // Freeze the visible catalogue once. A concurrently resumed campaign can add
    // files, but these new proofs cannot enter either arm of this experiment.
    let names: HashSet<_> = fs::read_dir(proof_dir)?
        .map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect::<Result<_, _>>()?;
    let mut proofs: HashMap<String, MicroProofCertificate> = HashMap::new();
    let mut proof_hashes = HashMap::new();
    let mut rows = vec![];
    let started = Instant::now();
    for (pair, spec) in manifest["games"]
        .as_array()
        .ok_or("games")?
        .iter()
        .enumerate()
    {
        let prefix: GameRecord = spec["prefix"].as_str().ok_or("prefix")?.parse()?;
        prefix.replay()?;
        for seat in [Player::Host, Player::Guest] {
            // Alternate order between pairs. Timings are descriptive, never a speed benchmark.
            for training in if pair % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let mut record = GameRecord::with_rules(prefix.setup(), prefix.rules());
                let mut p = record.initial_position();
                let mut history = VecDeque::from([p.clone()]);
                for a in prefix.actions() {
                    p.apply(*a)?;
                    record.push(*a);
                    history.push_back(p.clone());
                    if history.len() > 129 {
                        history.pop_front();
                    }
                }
                let mut micro = MicroMctsSession::new(model.clone());
                let mut old = MctsSession::new(
                    913 + pair as u64,
                    MctsConfig {
                        simulations: 8,
                        ..Default::default()
                    },
                    &reference,
                )?;
                let mut decisions = vec![];
                let t = Instant::now();
                let mut termination = "decision-limit";
                for step in 0..800 {
                    if p.outcome() != GameOutcome::Ongoing {
                        termination = "rules-terminal";
                        break;
                    }
                    // Independent per-decision RNG isolates behavior while holding
                    // budgets/seeds equal on shared prefixes across the two arms.
                    let mut rng =
                        StableRng::new(913 + (pair as u64) * 100_003 + (step as u64) * 997);
                    let action = if p.to_move() == seat {
                        let key = hash(record.to_string().as_bytes());
                        let filename = format!("{key}.json");
                        if names.contains(&filename) {
                            if !proofs.contains_key(&key) {
                                let b = fs::read(proof_dir.join(&filename))?;
                                let v: Value = serde_json::from_slice(&b)?;
                                assert_eq!(
                                    hash(v["prefix"].as_str().ok_or("proof prefix")?.as_bytes()),
                                    key
                                );
                                let c: MicroProofCertificate =
                                    serde_json::from_value(v["certificate"].clone())?;
                                c.verify(&p)?;
                                proof_hashes.insert(filename, hash(&b));
                                proofs.insert(key.clone(), c);
                            }
                            micro.install_certificate(&p, &proofs[&key])?;
                        }
                        let budget = if rng.next_f64() < 0.5 { 256 } else { 512 };
                        let r = micro.search_with_options(
                            &p,
                            budget,
                            None,
                            MicroSearchOptions {
                                proof_search: true,
                                seed: rng.next_u64(),
                                dirichlet_fraction: if training { 0.25 } else { 0. },
                                forced_playout_strength: if training { 2. } else { 0. },
                                ..Default::default()
                            },
                        )?;
                        if let Some(c) = micro.certificate(10000) {
                            c.verify(&p)?;
                        }
                        let mut selected = r.selected_index;
                        let sampling = training && step < 40;
                        if sampling {
                            if r.proven_action_values.iter().any(Option::is_some) {
                                let mut draw = rng.next_f64();
                                for (i, v) in r.policy_target.iter().enumerate() {
                                    if draw < *v {
                                        selected = i;
                                        break;
                                    }
                                    draw -= v;
                                }
                            } else {
                                let mut draw = rng.index(r.visits.iter().sum());
                                for (i, n) in r.visits.iter().enumerate() {
                                    if draw < *n {
                                        selected = i;
                                        break;
                                    }
                                    draw -= n;
                                }
                            }
                        }
                        decisions.push(json!({"step":step,"absolute_decision":record.actions().len()+1,"budget":budget,"sampling":sampling,
                            "selected":r.actions[selected].to_string(),"greedy":r.actions[r.selected_index].to_string(),"different":selected!=r.selected_index,"root_proof":r.proven_value}));
                        r.actions[selected]
                    } else {
                        let actions = legal_actions(&p);
                        let r = old.search_until(&p, &actions, None)?;
                        actions[r.selected_index]
                    };
                    p.apply(action)?;
                    record.push(action);
                    micro.advance(action)?;
                    old.advance(action);
                    history.push_back(p.clone());
                    if history.len() > 129 {
                        history.pop_front();
                    }
                    if p.outcome() != GameOutcome::Ongoing {
                        termination = "rules-terminal";
                        break;
                    }
                    if cycle(&history) {
                        termination = "repetition-unresolved";
                        break;
                    }
                }
                assert_eq!(record.replay()?, p);
                let id = rows.len();
                let psr = record.to_string();
                fs::write(output.join(format!("game-{id:03}.psr")), &psr)?;
                let row = json!({"id":id,"pair":pair,"source":spec["source"],"from_zero":prefix.actions().is_empty(),"prefix_decisions":prefix.actions().len(),
                    "seat":format!("{:?}",seat),"training_behavior":training,"outcome":format!("{:?}",p.outcome()),"win":p.outcome()==GameOutcome::Win(seat),
                    "termination":termination,"new_decisions":record.actions().len()-prefix.actions().len(),"seconds":t.elapsed().as_secs_f64(),"psr_sha256":hash(psr.as_bytes()),"decisions":decisions});
                fs::write(
                    output.join(format!("game-{id:03}.json")),
                    serde_json::to_vec(&row)?,
                )?;
                eprintln!(
                    "game {id} pair {pair} {seat:?} training={training} {:?} {:.1}s",
                    p.outcome(),
                    t.elapsed().as_secs_f64()
                );
                rows.push(row);
            }
        }
    }
    fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"rows":rows,"proof_hashes":proof_hashes,"catalogue_size_at_start":names.len(),
        "model_identity":artifact.identity(),"reference_budget":8,"production_writes":0,"learning_updates":0,"seconds":started.elapsed().as_secs_f64()}),
        )?,
    )?;
    Ok(())
}
