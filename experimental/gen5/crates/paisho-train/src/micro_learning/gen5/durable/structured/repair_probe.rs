//! Bounded native recovery/overflow/cost test on hashed production snapshots.
use super::*;
#[derive(Deserialize)]
struct Plan {
    before: StructuredRecallAnchors,
    after: StructuredRecallAnchors,
    actor: StructuredRecallAnchors,
}
fn open(source: &StructuredRecallAnchors) -> Result<Structured> {
    let mut b = Structured::default();
    b.restore(
        &serde_json::json!({"checkpoint":{"path":source.path,"sha256":source.sha256}}),
        true,
    )?;
    Ok(b)
}
fn same(a: &MicroExample, b: &MicroExample) -> bool {
    a.state == b.state
        && a.actions == b.actions
        && a.policy == b.policy
        && a.structured == b.structured
        && a.action_values == b.action_values
        && a.value == b.value
        && a.value_weight == b.value_weight
        && a.policy_weight == b.policy_weight
        && a.policy_support == b.policy_support
        && a.sequence_source == b.sequence_source
}
fn annotations(bank: &Structured) -> usize {
    bank.rows
        .iter()
        .map(|r| r.example.structured.iter().flatten().count())
        .sum()
}
fn verify_seeds(bank: &Structured, seeds: &Structured) -> Result<()> {
    for r in &seeds.rows {
        let current = bank
            .rows
            .iter()
            .find(|x| x.key == r.key)
            .ok_or_else(|| invalid("seed lost under pressure"))?;
        if !same(&current.example, &r.example) || !bank.anchors.contains(&r.key) {
            return Err(invalid("seed teaching changed"));
        }
    }
    Ok(())
}
pub fn run(plan_path: &Path, out: &Path) -> Result<serde_json::Value> {
    if out.exists() {
        return Err(invalid("diagnostic output exists"));
    }
    let plan: Plan = serde_json::from_slice(&fs::read(plan_path)?)?;
    let seeds = open(&plan.before)?;
    let before = open(&plan.after)?;
    let mut bank = before.clone();
    fs::create_dir(out)?;
    let t = Instant::now();
    bank.import_anchors(&plan.before)?;
    let recovery_seconds = t.elapsed().as_secs_f64();
    verify_seeds(&bank, &seeds)?;
    let repaired = bank.clone();
    let recovered = bank.progress();
    let keys = bank.rows.iter().map(|r| r.key.clone()).collect::<Vec<_>>();
    bank.import_anchors(&plan.before)?;
    if keys != bank.rows.iter().map(|r| r.key.clone()).collect::<Vec<_>>() {
        return Err(invalid("seed import not idempotent"));
    }
    let t = Instant::now();
    for round in 0..8 {
        for (i, row) in before.rows.iter().enumerate() {
            bank.admit(
                format!("diagnostic-{round}-{i}"),
                row.group.clone(),
                row.example.clone(),
            );
        }
        verify_seeds(&bank, &seeds)?;
    }
    let overflow_seconds = t.elapsed().as_secs_f64();
    let sampled = bank.draw(3200);
    if sampled.len() != 800 {
        return Err(invalid("structured draw exceeded existing recall quota"));
    }
    let seed_draws = sampled
        .iter()
        .filter(|e| seeds.rows.iter().any(|r| same(&r.example, e)))
        .count();
    bank.draw(3);
    bank.checkpoint(out, true, &mut vec![])?;
    let mut restored = Structured::default();
    restored.restore(&bank.progress().unwrap(), true)?;
    for _ in 0..64 {
        if !bank
            .draw(7)
            .iter()
            .zip(restored.draw(7))
            .all(|(a, b)| same(a, &b))
        {
            return Err(invalid("checkpoint continuation changed"));
        }
    }
    verify_seeds(&restored, &seeds)?;
    let bytes = fs::read(&plan.actor.path)?;
    if sha256(&bytes) != plan.actor.sha256 {
        return Err(invalid("actor hash changed"));
    }
    let model = serde_json::from_slice::<MicroArtifact>(&bytes)?.model()?;
    let mut timings = vec![];
    // ABBA, same sampler clocks. Composition intentionally changes: no bit-parity claim.
    for corrected in [false, true, true, false] {
        let mut source = if corrected {
            repaired.clone()
        } else {
            before.clone()
        };
        let t = Instant::now();
        let examples = source.draw(256);
        let draw_seconds = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut checksum = 0.;
        let mut buffer = vec![];
        for batch in examples.chunks(8) {
            let balance = MicroStructuredBatchBalance::new(batch.iter().map(AsRef::as_ref));
            for e in batch {
                let (loss, g) = model
                    .loss_gradient_structured_batch_reusing(e, &balance, true, buffer)
                    .map_err(invalid)?;
                checksum += loss.total(e.policy_weight);
                buffer = g;
            }
        }
        timings.push(serde_json::json!({"corrected":corrected,"examples":examples.len(),"draw_seconds":draw_seconds,
            "gradient_seconds":t.elapsed().as_secs_f64(),"checksum":checksum,
            "annotated_actions":examples.iter().map(|e|e.structured.iter().flatten().count()).sum::<usize>()}));
    }
    let report = serde_json::json!({"before":before.progress(),"recovered":recovered,"after_pressure":bank.progress(),
        "before_annotations":annotations(&before),"recovered_annotations":annotations(&repaired),"after_pressure_annotations":annotations(&bank),
        "anchors_verified":seeds.rows.len(),"arrivals":before.rows.len()*8,"recovery_seconds":recovery_seconds,
        "overflow_seconds":overflow_seconds,"structured_draws":800,"seed_draws":seed_draws,"checkpoint_continuation_exact":true,"cost_abba":timings});
    write(&out.join("report.json"), &report)?;
    Ok(report)
}
