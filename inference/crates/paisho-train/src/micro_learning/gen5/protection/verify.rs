//! Controlled decision-boundary regression in the actual model.
use super::*;
pub(crate) fn boundary(
    model: &MicroModel,
    rows: &[Arc<MicroExample>],
) -> Result<serde_json::Value> {
    let mut selected = None;
    for r in rows
        .iter()
        .filter(|r| r.value == 1. && !r.policy.is_empty())
    {
        let p = prior(model, r)?;
        let good = best(&p);
        if r.policy[good] == 0. {
            continue;
        }
        let Some(bad) = (0..p.len())
            .filter(|&i| r.policy[i] == 0.)
            .max_by(|&a, &b| p[a].total_cmp(&p[b]))
        else {
            continue;
        };
        let gap = p[good].ln() - p[bad].max(1e-300).ln();
        if selected.as_ref().map_or(true, |(_, _, g)| gap < *g) {
            selected = Some((r.clone(), bad, gap));
        }
    }
    let (row, bad, _) =
        selected.ok_or_else(|| invalid("boundary control needs a known winning choice"))?;
    let mut target = row.as_ref().clone();
    target.value_weight = 0.;
    target.policy_weight = 1.;
    target.policy.fill(0.);
    target.policy[bad] = 1.;
    let mut changed = model.clone();
    let mut steps = 0;
    while row.policy[best(&prior(&changed, &row)?)] > 0. && steps < 256 {
        changed
            .train_batch_inline(&[&target], 0.002, 0.)
            .map_err(invalid)?;
        steps += 1;
    }
    if row.policy[best(&prior(&changed, &row)?)] > 0. {
        return Err(invalid("could not construct boundary regression"));
    }
    let mut p = Protection::new(model, rows.to_vec())?;
    p.consolidate(&mut changed)?;
    if !p.last["accepted"].as_bool().unwrap_or(false)
        || p.last["choice_constraints"].as_u64().unwrap_or(0) == 0
    {
        return Err(invalid(format!("boundary correction failed: {}", p.last)));
    }
    Ok(serde_json::json!({"adverse_steps":steps,"result":p.last,"seconds":p.seconds}))
}
