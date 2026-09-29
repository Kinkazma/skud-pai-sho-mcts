//! Offline intervention: same frozen V5 value, different search operator.
//! Never writes model weights, production controls or training examples.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::{compact_learning, micro_learning::MicroArtifact};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, sync::Arc, time::Instant};

struct ValueOnly(Arc<MicroModel>);
impl ValueOnly {
    fn at(&self, p: &Position, player: Player) -> f64 {
        match p.outcome() {
            GameOutcome::Win(w) => {
                if w == player {
                    1.
                } else {
                    -1.
                }
            }
            GameOutcome::Draw => 0.,
            GameOutcome::Ongoing => {
                let v = self.0.embed(&self.0.state_features(p)).value;
                if p.to_move() == player {
                    v
                } else {
                    -v
                }
            }
        }
    }
}
impl MctsEvaluator for ValueOnly {
    fn ordering_matches_leaf(&self) -> bool {
        true
    }
    fn evaluate(
        &self,
        p: &[Position],
        player: Player,
        _: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        Ok(p.iter().map(|p| self.at(p, player) as f32).collect())
    }
    fn evaluate_leaf(
        &self,
        p: &Position,
        player: Player,
        _: HeuristicWeights,
    ) -> Result<f32, String> {
        Ok(self.at(p, player) as f32)
    }
}
fn digest(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn best(v: &[f64]) -> usize {
    (0..v.len())
        .max_by(|a, b| v[*a].total_cmp(&v[*b]).then_with(|| b.cmp(a)))
        .unwrap()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let artifact: MicroArtifact =
        serde_json::from_slice(&fs::read(manifest["model"].as_str().ok_or("model")?)?)?;
    let model = Arc::new(artifact.model()?);
    let evaluator = ValueOnly(model.clone());
    let legacy = compact_learning::load_model(std::path::Path::new(
        manifest["reference"].as_str().ok_or("reference")?,
    ))?
    .model()?;
    let mut sources = HashSet::new();
    let mut rows = vec![];
    let started = Instant::now();
    for (index, spec) in manifest["positions"]
        .as_array()
        .ok_or("positions")?
        .iter()
        .enumerate()
    {
        let bytes = fs::read(spec["path"].as_str().ok_or("path")?)?;
        if let Some(h) = spec["sha256"].as_str() {
            assert_eq!(h, digest(&bytes));
        }
        let is_case = spec["kind"] == "case";
        let (record, cert, source): (GameRecord, Option<MicroProofCertificate>, String) =
            if spec["kind"] == "certificate" {
                let data: Value = serde_json::from_slice(&bytes)?;
                (
                    data["prefix"].as_str().ok_or("prefix")?.parse()?,
                    Some(serde_json::from_value(data["certificate"].clone())?),
                    data["human_source"].as_str().unwrap().to_owned(),
                )
            } else {
                let full: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
                if is_case {
                    // Independently replay the whole original game before keeping its start.
                    full.replay()?;
                    let mut prefix = GameRecord::with_rules(full.setup(), full.rules());
                    for a in full
                        .actions()
                        .iter()
                        .take(spec["prefix_decisions"].as_u64().ok_or("prefix count")? as usize)
                    {
                        prefix.push(*a);
                    }
                    (prefix, None, spec["source"].as_str().unwrap().to_owned())
                } else {
                    (full, None, format!("alias-{index}"))
                }
            };
        let p = record.replay()?;
        assert_eq!(p.outcome(), GameOutcome::Ongoing);
        let actions = legal_actions(&p);
        let successors: Vec<_> = actions
            .iter()
            .map(|a| {
                let mut n = p.clone();
                n.apply(*a).unwrap();
                n
            })
            .collect();
        let immediate: Vec<_> = successors
            .iter()
            .map(|n| n.outcome() == GameOutcome::Win(p.to_move()))
            .collect();
        let mut winners = immediate.clone();
        if let Some(c) = &cert {
            let outcome = c.verify(&p)?;
            if outcome != GameOutcome::Win(p.to_move())
                || immediate.iter().any(|v| *v)
                || sources.len() >= 64
                || !sources.insert(source.clone())
            {
                continue;
            }
            let sign = if p.to_move() == Player::Host { 1 } else { -1 };
            for (a, c) in &c.children {
                if c.outcome == sign {
                    let a: Action = a.parse()?;
                    winners[actions.iter().position(|x| *x == a).unwrap()] = true;
                }
            }
        } else if !is_case && !immediate.iter().any(|v| *v) {
            continue;
        }
        let x = model.state_features(&p);
        let f: Vec<_> = actions
            .iter()
            .map(|a| micro_action_features(&p, *a))
            .collect();
        let prior = model.memory_priors(
            &x,
            &f,
            &micro_softmax(&MicroModel::logits(&model.embed(&x), &f))?,
            0,
        )?;
        let t = Instant::now();
        let values: Vec<_> = successors
            .iter()
            .map(|n| evaluator.at(n, p.to_move()))
            .collect();
        let greedy_seconds = t.elapsed().as_secs_f64();
        // Assert bonus-turn perspective and exact terminal semantics for every successor.
        for n in &successors {
            assert_eq!(
                evaluator.at(n, Player::Host),
                -evaluator.at(n, Player::Guest)
            );
        }
        let mut searches = vec![];
        for (name, ev, budget, solver) in [
            (
                "gen3-value/gen3-search8",
                &legacy as &dyn MctsEvaluator,
                8,
                false,
            ),
            (
                "v5-value/gen3-search8",
                &evaluator as &dyn MctsEvaluator,
                8,
                false,
            ),
            (
                "v5-value/gen3-search32",
                &evaluator as &dyn MctsEvaluator,
                32,
                false,
            ),
        ] {
            let t = Instant::now();
            let mut s = MctsSession::new(
                37,
                MctsConfig {
                    simulations: budget,
                    ..Default::default()
                },
                ev,
            )?;
            s.set_solver(solver);
            let r = s.search_until(&p, &actions, None)?;
            searches.push(json!({"arm":name,"selected":r.selected_index,"certified":winners[r.selected_index],"immediate":immediate[r.selected_index],
                "seconds":t.elapsed().as_secs_f64(),"evaluated_actions":r.evaluated_actions,"simulations":r.simulations,"depth":r.maximum_depth,
                "visited":r.actions.iter().filter(|a|a.visits>0).count()}));
        }
        for (noise, force) in [(0., 0.), (0.25, 2.)] {
            let t = Instant::now();
            let mut s = MicroMctsSession::new(model.clone());
            let r = s.search_with_options(
                &p,
                512,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    seed: 37,
                    dirichlet_fraction: noise,
                    forced_playout_strength: force,
                    ..Default::default()
                },
            )?;
            assert_eq!(r.actions, actions);
            if let Some(c) = s.certificate(10000) {
                c.verify(&p)?;
            }
            searches.push(json!({"arm":if noise==0. {"v5-puct512"} else {"v5-training512"},"selected":r.selected_index,
                "certified":winners[r.selected_index],"immediate":immediate[r.selected_index],"seconds":t.elapsed().as_secs_f64(),
                "inference_evaluations":r.inference_evaluations,"tactical_evaluations":r.tactical_evaluations,"simulations":r.simulations,
                "visited":r.visits.iter().filter(|n|**n>0).count(),"proven":r.proven_value,
                "proved_actions":r.proven_action_values,"visits":r.visits,"target":r.policy_target,"q":r.values,
                "certified_target_mass":r.policy_target.iter().zip(&winners).filter(|(_,w)|**w).map(|(p,_)|p).sum::<f64>(),
                "greedy_value_visits":r.visits[best(&values)]}));
        }
        rows.push(json!({"index":index,"kind":spec["kind"],"source":source,"original":spec,"prefix":record.to_string(),
            "prefix_sha256":digest(record.to_string().as_bytes()),"chooser":format!("{:?}",p.to_move()),"legal":actions.len(),
            "actions":actions.iter().map(ToString::to_string).collect::<Vec<_>>(),"certified":winners,"immediate":immediate,
            "prior":prior,"successor_values":values,"policy_pick":best(&prior),"value_pick":best(&values),"value_seconds":greedy_seconds,"searches":searches}));
        eprintln!(
            "roots {} elapsed {:.1}s",
            rows.len(),
            started.elapsed().as_secs_f64()
        );
    }
    fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"rows":rows,"model_identity":artifact.identity(),"total_seconds":started.elapsed().as_secs_f64(),"production_writes":0}),
        )?,
    )?;
    Ok(())
}
