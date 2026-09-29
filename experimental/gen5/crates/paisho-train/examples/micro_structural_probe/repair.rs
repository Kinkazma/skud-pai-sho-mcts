//! Read-only before/after panel; no fitting, selection or parameter sweep.
use super::*;
pub(super) fn run(cases: &[Case], model: &Arc<MicroModel>, out: &Path) -> Result<()> {
    let mut rows = vec![];
    let mut probes = vec![];
    for (i, c) in cases.iter().enumerate() {
        let e = &c.example;
        let base = micro_softmax(&MicroModel::logits(&model.embed(&e.state), &e.actions))?;
        let p = model.memory_priors(&e.state, &e.actions, &base, 0)?;
        let (loss, _) = model.loss_gradient(e)?;
        if !c.wins.is_empty() {
            for seed in 0..8 {
                let t = std::time::Instant::now();
                let mut search = MicroMctsSession::new(model.clone());
                let r = search.search_with_options(
                    &c.position,
                    512,
                    None,
                    MicroSearchOptions {
                        proof_search: true,
                        seed,
                        dirichlet_fraction: 0.25,
                        forced_playout_strength: 2.,
                        ..Default::default()
                    },
                )?;
                let example = r.example(1.0, 1.0)?;
                example.validate()?;
                probes.push(json!({"index":i,"seed":seed,"selected_wins":c.wins.contains(&r.selected_index),"winning_target_mass":c.wins.iter().map(|&j|r.policy_target[j]).sum::<f64>(),"proven":r.proven_value,"simulations":r.simulations,"tactical_evaluations":r.tactical_evaluations,"seconds":t.elapsed().as_secs_f64()}));
            }
        }
        rows.push(json!({"index":i,"source":c.source,"group":c.group,"state_len":e.state.len(),"phase":format!("{:?}",c.position.phase()),"wins":c.wins,"policy_wins":c.wins.contains(&argmax(&p)),"winning_mass":c.wins.iter().map(|&j|p[j]).sum::<f64>(),"loss_policy":loss.policy,"loss_value":loss.value,"value":model.embed(&e.state).value}));
    }
    fs::write(
        out.join("positions.json"),
        serde_json::to_vec_pretty(&rows)?,
    )?;
    fs::write(
        out.join("tactics.json"),
        serde_json::to_vec_pretty(&probes)?,
    )?;
    Ok(())
}
