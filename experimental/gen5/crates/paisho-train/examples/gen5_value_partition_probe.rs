//! Offline value-only supervision; neither policy nor neural weights are updated.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::MicroArtifact;
use serde_json::{json, Value};
use std::{fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()?;
    let a = std::env::args().collect::<Vec<_>>();
    let manifest: Value = serde_json::from_slice(&fs::read(&a[1])?)?;
    let out = Path::new(&a[2]);
    fs::create_dir_all(out)?;
    let base =
        MicroArtifact::load(Path::new(manifest["models"][2]["path"].as_str().unwrap()))?.model()?;
    let mut groups = vec![vec![]; 6];
    for spec in manifest["positions"].as_array().unwrap() {
        let group = match spec["group"].as_str().unwrap() {
            "guard" => 0,
            "validation" => 1,
            _ => continue,
        };
        let v: Value = serde_json::from_slice(&fs::read(spec["path"].as_str().unwrap())?)?;
        let p = v["prefix"]
            .as_str()
            .unwrap()
            .parse::<GameRecord>()?
            .replay()?;
        let c: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
        c.verify(&p)?;
        let value = (c.outcome * if p.to_move() == Player::Host { 1 } else { -1 }) as f64;
        groups[group * 3 + (value as i8 + 1) as usize].push(MicroExample { structured: Vec::new(), policy_support: false,
            state: base.state_features(&p),
            actions: vec![],
            policy: vec![],
            policy_weight: 0.,
            value_weight: 1.,
            value,
            sequence_source: 0,
            action_values: vec![],
        });
    }
    let mut models = vec![manifest["models"][2].clone()];
    for rate in [0.005, 0.0005, 0.00005] {
        let mut m = base.clone();
        for step in 0..32 {
            let rows = groups
                .iter()
                .flat_map(|g| (0..8).map(move |i| &g[(step * 8 + i) % g.len()]))
                .collect::<Vec<_>>();
            let mut gradient = vec![0.; 292363];
            for e in &rows {
                let part = m.loss_gradient(e)?.1;
                for (g, p) in gradient.iter_mut().zip(part) {
                    *g += p / rows.len() as f64;
                }
            }
            let w = m
                .parameters()
                .iter()
                .zip(gradient)
                .map(|(w, g)| w - rate * g)
                .collect();
            let mut next = MicroModel::from_parameters(w)?;
            next = next.with_sequence_memory_owned(m.sequence_memory().unwrap().clone());
            m = next;
        }
        assert!(base
            .parameters()
            .iter()
            .zip(m.parameters())
            .enumerate()
            .filter(|(i, _)| !(4128..4161).contains(i) && !(15822..93071).contains(i))
            .all(|(_, (a, b))| a.to_bits() == b.to_bits()));
        let path = out.join(format!("value-{rate}.json"));
        MicroArtifact::new(
            &m,
            0,
            json!({"diagnostic":true,"rate":rate,"steps":32,"value_supervision_only":true}),
        )
        .save(&path)?;
        models.push(json!({"path":path,"label":format!("value_{rate}"),"coupled":true}));
    }
    fs::write(
        out.join("manifest.json"),
        serde_json::to_vec(&json!({"models":models,"positions":manifest["positions"]}))?,
    )?;
    Ok(())
}
