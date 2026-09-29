//! Fixed-input cost and neutral-search parity. Not asynchronous campaign games/s.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::{MicroArtifact, SavedMicroExample};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, hint::black_box, path::Path, sync::Arc, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("OLD_ACTOR TRAINED_V7 MIGRATION_INPUTS NEW_OUTPUT".into());
    }
    let out = Path::new(&args[4]);
    fs::create_dir(out)?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let t = Instant::now();
    let old: MicroArtifact = serde_json::from_slice(&fs::read(&args[1])?)?;
    let trained: MicroArtifact = serde_json::from_slice(&fs::read(&args[2])?)?;
    let base = old.model()?;
    let neutral = base.with_relational(20260926);
    let trained_model = trained.model()?;
    if !trained_model.has_relational() {
        return Err("trained model is not V7".into());
    }
    let loading = t.elapsed().as_secs_f64();
    let input = Path::new(&args[3]);
    let sources: Vec<Value> = serde_json::from_slice(&fs::read(input.join("seed-sources.json"))?)?;
    let saved: Vec<SavedMicroExample> =
        serde_json::from_slice(&fs::read(input.join("seed-examples.json"))?)?;
    let mut positions = vec![];
    let mut examples = vec![];
    for (source, s) in sources.iter().zip(&saved).take(32) {
        let bytes = fs::read(source["psr"].as_str().ok_or("psr")?)?;
        if hash(&bytes) != source["sha256"] {
            return Err("source changed".into());
        }
        let r: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        let p = r.replay()?;
        let e = s.example_for_rules_with_trusted_q(RuleProfileId::SkudPaiShoGen5V1, true)?;
        if e.state != base.state_features(&p) {
            return Err("seed state mismatch".into());
        }
        positions.push(p);
        examples.push(e);
    }
    let models = [
        ("baseline", Arc::new(base)),
        ("neutral", Arc::new(neutral)),
        ("trained", Arc::new(trained_model)),
    ];
    let mut rows = vec![];
    let mut parity = vec![];
    for round in 0..4 {
        for index in if round % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
            let (name, m) = &models[index];
            let mut checksum = 0.;
            let t = Instant::now();
            for _ in 0..32 {
                for p in &positions {
                    black_box(m.state_features(p));
                }
            }
            let features = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..32 {
                for e in &examples {
                    checksum += black_box(m.value(&e.state));
                }
            }
            let value = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..4 {
                for e in &examples {
                    let (_, prior) = m.policy_value_priors(&e.state, &e.actions, 0)?;
                    checksum += black_box(prior[0]);
                }
            }
            let root = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for e in examples.iter().take(8) {
                let (loss, g) = m.loss_gradient_loop_v3_reusing(e, vec![])?;
                checksum += black_box(loss.total(e.policy_weight) + g.iter().sum::<f64>());
            }
            let gradient = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..32 {
                black_box(MicroStructuredBatchBalance::new(examples.iter().take(8)));
            }
            let balance = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let weights = MicroStructuredBatchBalance::new(examples.iter().take(8));
            let mut balanced_checksum = 0.;
            for e in examples.iter().take(8) {
                let (loss, g) =
                    m.loss_gradient_structured_batch_reusing(e, &weights, true, vec![])?;
                balanced_checksum += black_box(loss.total(e.policy_weight) + g.iter().sum::<f64>());
            }
            let balanced_gradient = t.elapsed().as_secs_f64();
            let t = Instant::now();
            for _ in 0..4 {
                for p in &positions {
                    let a = legal_actions(p)[0];
                    let mut q = p.clone();
                    q.apply(a)?;
                    let before = MicroRelations::extract(p, p.to_move());
                    let threat = micro_immediate_threat(&q, p.to_move(), 0)?;
                    black_box(MicroStructuredTarget::from_successor(
                        p, a, &q, &before, &threat,
                    ));
                }
            }
            let labels = t.elapsed().as_secs_f64();
            let mut searches = vec![];
            for p in positions.iter().take(4) {
                for budget in [256, 512] {
                    let mut search = MicroMctsSession::new(m.clone());
                    search.set_minimum_search_depth(5)?;
                    search.set_root_value_strength(2.)?;
                    let t = Instant::now();
                    let r = search.search_with_options(
                        p,
                        budget,
                        None,
                        MicroSearchOptions {
                            proof_search: true,
                            ..Default::default()
                        },
                    )?;
                    let seconds = t.elapsed().as_secs_f64();
                    let fingerprint = hash(&serde_json::to_vec(
                        &json!({"selected":r.selected_index,"visits":r.visits,
                    "policy":r.policy_target.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"value":r.network_value.to_bits(),
                    "priors":r.priors.iter().map(|v|v.to_bits()).collect::<Vec<_>>()}),
                    )?);
                    searches
                        .push(json!({"budget":budget,"seconds":seconds,"fingerprint":fingerprint}));
                }
            }
            if round == 0 {
                parity.push(
                    searches
                        .iter()
                        .map(|v| v["fingerprint"].clone())
                        .collect::<Vec<_>>(),
                );
            }
            rows.push(json!({"round":round,"model":name,"features_seconds_1024":features,"value_seconds_1024":value,
                "root_seconds_128":root,"gradient_seconds_8":gradient,"labels_seconds_128":labels,"searches":searches,"checksum":checksum,
                "balance_seconds_32_batches":balance,"balanced_gradient_seconds_8":balanced_gradient,"balanced_gradient_checksum":balanced_checksum}));
            fs::write(out.join("rows.json"), serde_json::to_vec_pretty(&rows)?)?;
        }
    }
    if parity[0] != parity[1] {
        return Err("neutral migration changes fixed full-search results".into());
    }
    let result = json!({"rows":rows,"loading_seconds":loading,"neutral_search_parity":true,
        "scope":"32 fixed roots; 8 fixed searches per leg; does not measure campaign throughput or strength",
        "initial_model_sha256":hash(&fs::read(&args[1])?),"trained_model_sha256":hash(&fs::read(&args[2])?)});
    fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("neutral search parity passed; 12 fixed cost legs complete");
    Ok(())
}
