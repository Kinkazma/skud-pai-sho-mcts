//! Read-only sample of the actual retained FIFO: labels, proofs and action aliases.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::SavedMicroExample;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, io::Read, path::Path};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("MANIFEST OUTPUT".into());
    }
    let manifest: Value = serde_json::from_slice(&fs::read(&args[1])?)?;
    let proof_dir = Path::new(manifest["proofs"].as_str().ok_or("proof directory")?);
    let mut rows = vec![];
    for (si, source) in manifest["rows"]
        .as_array()
        .ok_or("rows")?
        .iter()
        .enumerate()
    {
        let path = Path::new(source["source"]["path"].as_str().ok_or("path")?);
        let bytes = fs::read(path)?;
        assert_eq!(source["source"]["sha256"], hash(&bytes));
        let mut raw = vec![];
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut raw)?;
        let saved: Vec<SavedMicroExample> = serde_json::from_slice(&raw)?;
        let base = path
            .to_str()
            .unwrap()
            .strip_suffix(".targets.json.gz")
            .ok_or("suffix")?;
        let record_bytes = fs::read(format!("{base}.psr"))?;
        let receipt: Value = serde_json::from_slice(&fs::read(format!("{base}.json"))?)?;
        assert_eq!(receipt["psr_sha256"], hash(&record_bytes));
        let record: GameRecord = std::str::from_utf8(&record_bytes)?.parse()?;
        let mut prefix = GameRecord::with_rules(record.setup(), record.rules());
        let mut p = record.initial_position();
        let mut next = 0;
        let mut indices = source["indices"]
            .as_array()
            .ok_or("indices")?
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        indices.sort_by_key(|&i| saved[i].decision);
        for index in indices {
            let s = &saved[index];
            s.example_for_rules(record.rules())?;
            while next < s.decision - 1 {
                let a = record.actions()[next];
                p.apply(a)?;
                prefix.push(a);
                next += 1;
            }
            assert_eq!(s.state, micro_spatial_state_features(&p));
            let key = hash(prefix.to_string().as_bytes());
            let state_bytes = s
                .state
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<_>>();
            let legal = legal_actions(&p);
            let actions = legal
                .iter()
                .map(|a| micro_action_features(&p, *a))
                .collect::<Vec<_>>();
            if !s.actions.is_empty() {
                assert_eq!(
                    s.actions,
                    legal.iter().map(ToString::to_string).collect::<Vec<_>>()
                );
                assert!(s.action_features.iter().zip(&actions).all(|(a, b)| a == b));
            }
            let immediate = legal
                .iter()
                .map(|a| {
                    let mut n = p.clone();
                    n.apply(*a).unwrap();
                    n.outcome() == GameOutcome::Win(p.to_move())
                })
                .collect::<Vec<_>>();
            let mut aliases: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
            for (i, a) in actions.iter().enumerate() {
                aliases
                    .entry(a.iter().map(|v| v.to_bits()).collect())
                    .or_default()
                    .push(i);
            }
            let bad_aliases = aliases
                .values()
                .filter(|v| v.iter().any(|&i| immediate[i]) && v.iter().any(|&i| !immediate[i]))
                .count();
            let cert_path = proof_dir.join(format!("{key}.json"));
            let mut proof = None;
            let mut known_before = false;
            let mut proof_mass = None;
            if cert_path.exists() {
                let c: Value = serde_json::from_slice(&fs::read(&cert_path)?)?;
                assert_eq!(hash(c["prefix"].as_str().ok_or("prefix")?.as_bytes()), key);
                let cert: MicroProofCertificate = serde_json::from_value(c["certificate"].clone())?;
                cert.verify(&p)?;
                let sign = if p.to_move() == Player::Host { 1 } else { -1 };
                proof = Some(cert.outcome * sign);
                known_before =
                    fs::metadata(&cert_path)?.modified()? <= fs::metadata(path)?.modified()?;
                if cert.outcome == sign && !s.policy.is_empty() {
                    let mut mass = 0.;
                    for (a, child) in &cert.children {
                        if child.outcome == sign {
                            let a: Action = a.parse()?;
                            mass += s.policy
                                [legal.iter().position(|x| *x == a).ok_or("proof action")?];
                        }
                    }
                    proof_mass = Some(mass);
                }
            }
            let immediate_mass = if s.policy.is_empty() {
                None
            } else {
                Some(
                    s.policy
                        .iter()
                        .zip(&immediate)
                        .filter(|(_, ok)| **ok)
                        .map(|(v, _)| v)
                        .sum::<f64>(),
                )
            };
            rows.push(json!({"source_index":si,"target_index":index,"prefix":key,"state":hash(&state_bytes),
                "lane":source["lane"],"decision":s.decision,"value":s.value,"reason":s.reason,
                "policy_weight":s.policy_weight,"proof":proof,"proof_known_before":known_before,
                "proof_target_mass":proof_mass,"immediate_actions":immediate.iter().filter(|v|**v).count(),
                "immediate_target_mass":immediate_mass,"legal":legal.len(),"action_alias_groups":aliases.values().filter(|v|v.len()>1).count(),
                "action_alias_with_different_immediate_outcome":bad_aliases,"human_source":receipt["case"]["human_source"],
                "collector":s.collector,"budget":s.budget}));
        }
        if si % 256 == 0 {
            eprintln!(
                "sources {}/{}",
                si + 1,
                manifest["rows"].as_array().unwrap().len()
            );
        }
    }
    assert_eq!(rows.len(), manifest["selected"].as_u64().unwrap() as usize);
    fs::write(
        &args[2],
        serde_json::to_vec(
            &json!({"rows":rows,"source_hashes_checked":manifest["rows"].as_array().unwrap().len(),"exact_features":true}),
        )?,
    )?;
    Ok(())
}
