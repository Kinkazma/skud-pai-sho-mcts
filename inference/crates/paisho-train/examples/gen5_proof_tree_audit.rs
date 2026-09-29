//! Inspect all already-certified branches, including positions never played.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path, sync::Arc};
struct Audit<'a> {
    model: &'a MicroModel,
    proofs: &'a Path,
    rows: Vec<Value>,
    seen: HashSet<String>,
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
impl Audit<'_> {
    fn walk(
        &mut self,
        p: &Position,
        prefix: &GameRecord,
        c: &MicroProofCertificate,
        root: usize,
        depth: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let key = hash(prefix.to_string().as_bytes());
        if !self.seen.insert(key.clone()) {
            return Ok(());
        }
        if p.outcome() != GameOutcome::Ongoing {
            return Ok(());
        }
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let value = (sign * c.outcome) as f64;
        let state = self.model.state_features(p);
        let embedding = self.model.embed(&state);
        let actions = legal_actions(p);
        let features: Vec<_> = actions
            .iter()
            .map(|a| micro_action_features(p, *a))
            .collect();
        let base = micro_softmax(&MicroModel::logits(&embedding, &features))?;
        let prior = self.model.memory_priors(&state, &features, &base, 0)?;
        let mut allowed = vec![false; actions.len()];
        if value >= 0. {
            for (name, child) in &c.children {
                if child.outcome == c.outcome {
                    let a: Action = name.parse()?;
                    allowed[actions.iter().position(|x| *x == a).ok_or("proof action")?] = true;
                }
            }
        }
        let best = (0..prior.len())
            .max_by(|a, b| prior[*a].total_cmp(&prior[*b]).then_with(|| b.cmp(a)))
            .unwrap();
        let mass = prior
            .iter()
            .zip(&allowed)
            .filter(|(_, ok)| **ok)
            .map(|(x, _)| x)
            .sum::<f64>();
        self.rows.push(json!({"root":root,"depth":depth,"prefix":key,"target":value,"value":embedding.value,
            "root_catalogue_contains":self.proofs.join(format!("{key}.json")).exists(),"legal":actions.len(),
            "certified_mass":mass,"certified_top":allowed[best],"base_certified_mass":base.iter().zip(&allowed).filter(|(_,ok)|**ok).map(|(x,_)|x).sum::<f64>()}));
        for (a, child) in &c.children {
            let a: Action = a.parse()?;
            let mut n = p.clone();
            n.apply(a)?;
            let mut next = prefix.clone();
            next.push(a);
            self.walk(&n, &next, child, root, depth + 1)?;
        }
        Ok(())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("MODEL PANEL_MANIFEST PROOF_DIR OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&args[1])?)?;
    let model = Arc::new(artifact.model()?);
    let manifest: Value = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut audit = Audit {
        model: &model,
        proofs: Path::new(&args[3]),
        rows: vec![],
        seen: HashSet::new(),
    };
    let mut searches = vec![];
    let mut selected_sources = HashSet::new();
    for (index, spec) in manifest["positions"]
        .as_array()
        .ok_or("positions")?
        .iter()
        .enumerate()
    {
        if spec["kind"] != "certificate" {
            continue;
        }
        let data: Value = serde_json::from_slice(&fs::read(spec["path"].as_str().ok_or("path")?)?)?;
        let record: GameRecord = data["prefix"].as_str().ok_or("prefix")?.parse()?;
        let p = record.replay()?;
        let c: MicroProofCertificate = serde_json::from_value(data["certificate"].clone())?;
        c.verify(&p)?;
        audit.walk(&p, &record, &c, index, 0)?;
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let actions = legal_actions(&p);
        let immediate = actions.iter().any(|a| {
            let mut next = p.clone();
            next.apply(*a).unwrap();
            next.outcome() == GameOutcome::Win(p.to_move())
        });
        let source = data["human_source"].as_str().ok_or("source")?;
        if c.outcome == sign
            && !immediate
            && selected_sources.len() < 64
            && !selected_sources.contains(source)
        {
            selected_sources.insert(source.to_string());
            let winners = c
                .children
                .iter()
                .filter(|(_, c)| c.outcome == sign)
                .map(|(a, _)| a.parse::<Action>())
                .collect::<Result<Vec<_>, _>>()?;
            for (noise, seed) in [(false, 0), (true, 37), (true, 913)] {
                let mut session = MicroMctsSession::new(model.clone());
                let r = session.search_with_options(
                    &p,
                    512,
                    None,
                    MicroSearchOptions {
                        proof_search: true,
                        seed,
                        dirichlet_fraction: if noise { 0.25 } else { 0. },
                        forced_playout_strength: if noise { 2. } else { 0. },
                        ..Default::default()
                    },
                )?;
                let original_best = (0..r.priors.len())
                    .max_by(|a, b| r.priors[*a].total_cmp(&r.priors[*b]).then_with(|| b.cmp(a)))
                    .unwrap();
                searches.push(json!({"position":index,"source":source,"noise":noise,"seed":seed,
                    "proven":r.proven_value,"simulations":r.simulations,"visited":r.new_visits.iter().filter(|n|**n>0).count(),"legal":r.actions.len(),
                    "raw_certified_top":winners.contains(&r.actions[original_best]),"selected_certified":winners.contains(&r.actions[r.selected_index]),
                    "certified_prior_mass":r.priors.iter().zip(&r.actions).filter(|(_,a)|winners.contains(a)).map(|(v,_)|v).sum::<f64>(),
                    "certified_target_mass":r.policy_target.iter().zip(&r.actions).filter(|(_,a)|winners.contains(a)).map(|(v,_)|v).sum::<f64>(),
                    "certified_new_visits":r.new_visits.iter().zip(&r.actions).filter(|(_,a)|winners.contains(a)).map(|(v,_)|v).sum::<usize>()}));
                if let Some(new) = session.certificate(10000) {
                    new.verify(&p)?;
                }
            }
        }
        if index % 64 == 0 {
            eprintln!("proof roots {}", index + 1);
        }
    }
    fs::write(
        &args[4],
        serde_json::to_vec(&json!({"positions":audit.rows,"searches":searches,
        "search_scope":"first 64 source-distinct known wins without immediate victory; fresh searches without persisted certificate installation",
        "all_root_certificates_verified":true,"production_writes":0}))?,
    )?;
    Ok(())
}
