//! Isolated proposals on immutable V33 snapshots. No campaign or production writes.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, sync::Arc, time::Instant};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[path = "gen5_v33_solution_probe/data.rs"]
mod data;
#[path = "gen5_v33_solution_probe/repair.rs"]
mod repair;
#[path = "gen5_v33_solution_probe/teaching.rs"]
mod teaching;
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn best(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn prior(m: &MicroModel, e: &MicroExample) -> Vec<f64> {
    m.memory_priors(
        &e.state,
        &e.actions,
        &micro_softmax(&MicroModel::logits(&m.embed(&e.state), &e.actions)).unwrap(),
        e.sequence_source,
    )
    .unwrap()
}
fn clone_weights(m: &MicroModel, w: Vec<f64>) -> MicroModel {
    MicroModel::from_parameters(w)
        .unwrap()
        .with_sequence_memory(m.sequence_memory().unwrap().clone())
}
fn mean_gradient(m: &MicroModel, xs: &[MicroExample]) -> Vec<f64> {
    let mut g = vec![0.; m.parameters().len()];
    for ex in xs {
        let (_, v) = m.loss_gradient(ex).unwrap();
        for (g, v) in g.iter_mut().zip(v) {
            *g += v / xs.len() as f64;
        }
    }
    g
}
fn mean_loss(m: &MicroModel, xs: &[MicroExample]) -> f64 {
    xs.iter()
        .map(|x| m.loss_gradient(x).unwrap().0.total(x.policy_weight) / xs.len() as f64)
        .sum()
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn step(m: &MicroModel, g: &[f64], rate: f64) -> MicroModel {
    let scale = (10. / dot(g, g).sqrt()).min(1.);
    clone_weights(
        m,
        m.parameters()
            .iter()
            .zip(g)
            .map(|(w, g)| w - rate * scale * g)
            .collect(),
    )
}
fn value_parameter(i: usize) -> bool {
    let start = (MICRO_INPUTS + 1) * MICRO_HIDDEN;
    (start..=start + MICRO_HIDDEN).contains(&i) || i >= MICRO_VALUE_TRUNK
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 4 {
        return Err("FLOW_MANIFEST REVIEW_MANIFEST OUTPUT".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let manifest: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let review: Value = serde_json::from_slice(&fs::read(&a[2])?)?;
    let t = Instant::now();
    let d = data::load(&manifest)?;
    let load = t.elapsed().as_secs_f64();
    eprintln!(
        "loaded {} fresh, {} teachers, {} verified proofs in {:.3}s",
        d.fresh.len(),
        d.teachers.len(),
        d.proofs.len(),
        load
    );
    let teaching = teaching::run(&d)?;
    eprintln!("finite policy learning complete");
    let repair = repair::run(&d, &review, std::path::Path::new(&a[3]).parent().unwrap())?;
    eprintln!("retention and publication proposals complete");
    fs::write(
        &a[3],
        serde_json::to_vec(
            &json!({"load_seconds":load,"teaching":teaching,"repair":repair,"production_writes":0,"new_games":0,"elapsed_seconds":t.elapsed().as_secs_f64()}),
        )?,
    )?;
    Ok(())
}
