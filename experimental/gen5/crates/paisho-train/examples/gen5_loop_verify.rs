//! Offline verification for the integrated loop; never writes a campaign.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Arc};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() < 4 {
        return Err("manifest PROOF_DIR OUTPUT | targets MODEL RUN_DIR OUTPUT | kernel MODEL MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    match a[1].as_str() {
        "candidate" => { let v=paisho_train::micro_learning::gen5::verify_learning_loop_candidate(Path::new(&a[2]),Path::new(&a[3]),Path::new(&a[4]),Path::new(&a[5]))?;println!("{}",v); },
        "guard" => {
            let v = paisho_train::micro_learning::gen5::verify_learning_loop_guard(
                Path::new(&a[2]),
                Path::new(&a[3]),
                Path::new(&a[4]),
            )?;
            fs::write(
                Path::new(&a[4]).join("verification.json"),
                serde_json::to_vec_pretty(&v)?,
            )?;
            println!("{}", v);
        }
        "manifest" => {
            let mut files = fs::read_dir(&a[2])?
                .map(|x| x.map(|x| x.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            files.sort();
            let mut rows = vec![];
            let mut counts = [0usize; 3];
            let limits = [16, 16, 32];
            let mut seen = std::collections::BTreeSet::new();
            let mut scanned = 0;
            for file in files.into_iter().take(4096) {
                let v: Value = serde_json::from_slice(&fs::read(&file)?)?;
                let text = v["prefix"].as_str().ok_or("prefix")?;
                if file.file_stem().unwrap() != hash(text.as_bytes()).as_str() {
                    return Err("proof hash".into());
                }
                let record: GameRecord = text.parse()?;
                let p = record.replay()?;
                let c: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
                c.verify(&p)?;
                scanned += 1;
                if p.outcome() != GameOutcome::Ongoing {
                    continue;
                }
                let z = c.outcome * if p.to_move() == Player::Host { 1 } else { -1 };
                let k = (z + 1) as usize;
                let source = v["human_source"].as_str().unwrap_or(text).to_string();
                if counts[k] >= limits[k] || !seen.insert((k, source)) {
                    continue;
                }
                rows.push(json!({"prefix":text,"certificate":c}));
                counts[k] += 1;
                if counts == limits {
                    break;
                }
            }
            fs::write(
                &a[3],
                serde_json::to_vec(
                    &json!({"schema":"paisho-gen5-publication-guard-v1","rows":rows}),
                )?,
            )?;
            println!(
                "{}",
                json!({"counts_loss_draw_win":counts,"scanned":scanned,"sha256":hash(&fs::read(&a[3])?)})
            );
        }
        "targets" => {
            let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&a[2])?)?;
            let model = artifact.model()?;
            let run = Path::new(&a[3]);
            let mut files = fs::read_dir(run.join("games"))?
                .map(|x| x.map(|x| x.path()))
                .collect::<std::io::Result<Vec<_>>>()?;
            files.sort();
            let mut games = 0;
            let mut targets = 0;
            let mut opponent = 0;
            let mut opponent_wins = 0;
            let mut teachers = 0;
            let mut observed = 0;
            let mut terminal = 0;
            for file in files
                .into_iter()
                .filter(|p| p.extension().is_some_and(|s| s == "psr"))
            {
                let bytes = fs::read(&file)?;
                let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
                let end = record.replay()?;
                games += 1;
                terminal += usize::from(end.outcome() != GameOutcome::Ongoing);
                let receipt: Value =
                    serde_json::from_slice(&fs::read(file.with_extension("json"))?)?;
                assert_eq!(receipt["psr_sha256"], hash(&bytes));
                let target = file.with_extension("targets.json.gz");
                if !target.exists() {
                    continue;
                }
                let bytes = fs::read(target)?;
                assert_eq!(receipt["targets_sha256"], hash(&bytes));
                let rows: Vec<SavedMicroExample> =
                    serde_json::from_reader(flate2::read::GzDecoder::new(bytes.as_slice()))?;
                let mut p = record.initial_position();
                let mut n = 0;
                for s in rows {
                    while n < s.decision - 1 {
                        p.apply(record.actions()[n])?;
                        n += 1;
                    }
                    s.example_for_rules(record.rules())?;
                    assert_eq!(s.state, model.state_features(&p));
                    let legal = legal_actions(&p);
                    let actions: Vec<Action> = s
                        .actions
                        .iter()
                        .map(|s| s.parse())
                        .collect::<std::result::Result<_, _>>()?;
                    if !actions.is_empty() {
                        assert_eq!(actions, legal);
                    }
                    for (a, f) in actions.iter().zip(&s.action_features) {
                        assert_eq!(f, &micro_action_features(&p, *a).to_vec());
                    }
                    let e = s.evidence.as_ref().ok_or("missing evidence")?;
                    if !receipt["reanalysis"].as_bool().unwrap()
                        && end.outcome() != GameOutcome::Ongoing
                    {
                        let z = match end.outcome() {
                            GameOutcome::Win(w) => {
                                if w == p.to_move() {
                                    1.
                                } else {
                                    -1.
                                }
                            }
                            _ => 0.,
                        };
                        assert_eq!(e.observed_value, Some(z));
                        assert_eq!(e.observed_psr, Some(hash(record.to_string().as_bytes())));
                        observed += 1;
                    }
                    if e.policy_source == "verified-regulatory-win" {
                        for (a, q) in actions.iter().zip(&s.policy) {
                            if *q > 0. {
                                let mut next = p.clone();
                                next.apply(*a)?;
                                assert_eq!(next.outcome(), GameOutcome::Win(p.to_move()));
                            }
                        }
                        opponent_wins += 1;
                    }
                    if e.policy_source == "observed-action-value-only"
                        || e.policy_source == "verified-regulatory-win"
                    {
                        opponent += 1;
                    }
                    if e.policy_source == "full-search-estimate" {
                        teachers += 1;
                    }
                    targets += 1;
                }
            }
            fs::write(
                &a[4],
                serde_json::to_vec_pretty(
                    &json!({"games":games,"terminal":terminal,"targets":targets,"opponent_targets":opponent,"opponent_proven_wins":opponent_wins,"full_search_teachers":teachers,"observed_targets":observed,"verified":true}),
                )?,
            )?;
        }
        "kernel" => {
            let artifact: MicroArtifact = serde_json::from_slice(&fs::read(&a[2])?)?;
            let model = artifact.model()?;
            let manifest: Value = serde_json::from_slice(&fs::read(&a[3])?)?;
            let mut rows = vec![];
            let mut max_error = 0f64;
            let mut derivatives = 0;
            for row in manifest["rows"].as_array().ok_or("rows")?.iter().take(8) {
                let record: GameRecord = row["prefix"].as_str().unwrap().parse()?;
                let p = record.replay()?;
                let actions = legal_actions(&p);
                let features: Vec<_> = actions
                    .iter()
                    .map(|a| micro_action_features(&p, *a))
                    .collect();
                let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
                    state: model.state_features(&p),
                    actions: features,
                    policy: vec![1. / actions.len() as f64; actions.len()],
                    value: -0.75,
                    policy_weight: 0.4,
                    value_weight: 1.,
                    sequence_source: u64::MAX,
                };
                let (_, g1) = model.loss_gradient(&ex)?;
                let mut zero = ex.clone();
                zero.value_weight = 0.;
                let (_, g0) = model.loss_gradient(&zero)?;
                let mut quarter = ex.clone();
                quarter.value_weight = 0.25;
                let (_, gq) = model.loss_gradient(&quarter)?;
                for i in 0..gq.len() {
                    assert!((gq[i] - (g0[i] + 0.25 * (g1[i] - g0[i]))).abs() < 1e-12);
                }
                for i in (0..gq.len()).step_by(997) {
                    let mut plus = model.parameters().to_vec();
                    plus[i] += 1e-5;
                    let mut minus = model.parameters().to_vec();
                    minus[i] -= 1e-5;
                    let l = |w| -> Result<f64> {
                        Ok(MicroModel::from_parameters(w)?
                            .loss_gradient(&quarter)?
                            .0
                            .total(quarter.policy_weight))
                    };
                    let numerical = (l(plus)? - l(minus)?) / 2e-5;
                    max_error = max_error.max((numerical - gq[i]).abs());
                    derivatives += 1;
                }
                let mut a = MicroMctsSession::new(Arc::new(model.clone()));
                let mut b = MicroMctsSession::new(Arc::new(model.clone()));
                b.set_root_value_strength(0.)?;
                let opts = MicroSearchOptions {
                    proof_search: true,
                    ..Default::default()
                };
                let x = a.search_with_options(&p, 512, None, opts)?;
                let y = b.search_with_options(&p, 512, None, opts)?;
                assert_eq!(x.priors, y.priors);
                assert_eq!(x.visits, y.visits);
                assert_eq!(x.values, y.values);
                assert_eq!(x.policy_target, y.policy_target);
                let mut c = MicroMctsSession::new(Arc::new(model.clone()));
                c.set_root_value_strength(16.)?;
                let coupled = c.search_with_options(&p, 512, None, opts)?;
                if let Some(cert) = c.certificate(10000) {
                    cert.verify(&p)?;
                }
                rows.push(json!({"neutral_exact":true,"proof":coupled.proven_value,"evaluations":coupled.inference_evaluations,"tactical_evaluations":coupled.tactical_evaluations}));
            }
            assert!(max_error < 1e-7);
            let restored =
                MicroArtifact::new(&model, artifact.updates, artifact.provenance.clone());
            assert_eq!(restored.parameters, artifact.parameters);
            assert_eq!(restored.identity(), artifact.identity());
            fs::write(
                &a[4],
                serde_json::to_vec_pretty(
                    &json!({"derivatives":derivatives,"max_error":max_error,"parameters":model.parameters().len(),"migration_identity_exact":true,"searches":rows}),
                )?,
            )?;
        }
        _ => return Err("unknown mode".into()),
    }
    Ok(())
}
