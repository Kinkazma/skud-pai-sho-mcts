//! Bounded scalar scoring diagnostic. Archived priors only; no model or bank.
use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    position: usize,
    policy_support: bool,
    prediction: Vec<f64>,
    policy: Vec<f64>,
    old: f64,
    new: f64,
}

fn legacy(prediction: &[f64], ex: &MicroExample) -> f64 {
    prediction.iter().zip(&ex.policy).map(|(p,t)| (p-t).abs()).sum::<f64>()
}

pub fn verify(path: &Path, repetitions: usize) -> Result<serde_json::Value> {
    if !(1..=65_536).contains(&repetitions) {
        return Err(invalid("scoring repetitions must be 1..65536"));
    }
    let init = Instant::now();
    let bytes = fs::read(path)?;
    let fixtures: Vec<Fixture> = serde_json::from_slice(&bytes)?;
    if fixtures.is_empty() || fixtures.len() > 64 {
        return Err(invalid("scoring fixture must contain 1..64 rows"));
    }
    let mut jobs = Vec::new();
    let mut rows = Vec::new();
    for f in fixtures {
        let ex = MicroExample {
            state: vec![0.; 417], actions: vec![[0.; 32]; f.policy.len()],
            policy: f.policy, policy_support: f.policy_support, value: 1.,
            policy_weight: 1., value_weight: 0., action_values: vec![], sequence_source: 0,
        };
        ex.validate().map_err(invalid)?;
        if f.prediction.len() != ex.policy.len()
            || f.prediction.iter().any(|p| !p.is_finite() || *p < 0.)
            || (f.prediction.iter().sum::<f64>() - 1.).abs() > 1e-12 {
            return Err(invalid("invalid archived prediction"));
        }
        let old = legacy(&f.prediction, &ex);
        let new = policy_distance(&f.prediction, &ex);
        if old.to_bits() != f.old.to_bits() || new.to_bits() != f.new.to_bits() {
            return Err(invalid(format!("native archived score mismatch at position {}", f.position)));
        }
        rows.push(serde_json::json!({"position":f.position,"old":old,"new":new,
            "old_bits":old.to_bits(),"new_bits":new.to_bits(),"reference_bits_match":true}));
        jobs.push((f.prediction, ex));
    }
    let initialization_seconds = init.elapsed().as_secs_f64();
    let mut timings = Vec::new();
    for (name, kernel) in [
        ("legacy", legacy as fn(&[f64], &MicroExample)->f64),
        ("support", policy_distance), ("support", policy_distance), ("legacy", legacy),
    ] {
        let kernel = std::hint::black_box(kernel);
        let start = Instant::now();
        let mut checksum = 0.;
        for _ in 0..repetitions {
            for (prediction, ex) in &jobs {
                checksum += std::hint::black_box(kernel(
                    std::hint::black_box(prediction.as_slice()), std::hint::black_box(ex)));
            }
        }
        let seconds = start.elapsed().as_secs_f64();
        timings.push(serde_json::json!({"mode":name,"seconds":seconds,
            "scorings":repetitions*jobs.len(),
            "microseconds_per_scoring":seconds*1e6/(repetitions*jobs.len()) as f64,
            "checksum":std::hint::black_box(checksum)}));
    }
    Ok(serde_json::json!({"fixture_sha256":sha256(&bytes),"rows":rows,
        "initialization_seconds_excluded":initialization_seconds,"abba":timings,
        "new_games":0,"model_evaluations":0,"learning_updates":0,
        "scope":"Scalar policy priority only; archived predictions, no neural inference, bank, sampler or campaign throughput measurement."}))
}
