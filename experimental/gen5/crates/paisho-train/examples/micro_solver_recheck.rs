//! Read-only audit of simulation reuse on already proved drawing actions.
use paisho_ai::{MicroMctsSession, MicroSearchOptions};
use paisho_core::{legal_actions, GameOutcome, GameRecord};
use paisho_train::micro_learning::MicroArtifact;
use serde_json::json;
use std::{fs, path::Path, sync::Arc};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 3 {
        return Err("micro_solver_recheck MODEL PSR_LIST.json".into());
    }
    let model = Arc::new(MicroArtifact::load(Path::new(&a[1]))?.model()?);
    let paths: Vec<String> = serde_json::from_slice(&fs::read(&a[2])?)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    for path in paths {
        let r: GameRecord = fs::read_to_string(&path)?.parse()?;
        assert_eq!(r.replay()?.outcome(), GameOutcome::Draw);
        let mut p = r.initial_position();
        for action in &r.actions()[..r.actions().len() - 1] {
            p.apply(*action)?;
        }
        let actions = legal_actions(&p);
        let known_draw: Vec<_> = actions
            .iter()
            .map(|a| {
                let mut q = p.clone();
                q.apply(*a).unwrap();
                q.outcome() == GameOutcome::Draw
            })
            .collect();
        let mut search = MicroMctsSession::new(model.clone());
        let started = std::time::Instant::now();
        let report = pool.install(|| {
            search.search_with_options(
                &p,
                512,
                None,
                MicroSearchOptions {
                    proof_search: true,
                    ..Default::default()
                },
            )
        })?;
        assert_eq!(report.actions, actions);
        let wasted: usize = report
            .visits
            .iter()
            .zip(&known_draw)
            .filter(|(_, yes)| **yes)
            .map(|(n, _)| *n)
            .sum();
        let chosen = report.selected_index;
        let safe_choice = report.proven_value.is_some()
            || known_draw[chosen]
            || (report.visits[chosen] > 0 && report.values[chosen] > 0.);
        println!(
            "{}",
            json!({"path":path,"elapsed_seconds":started.elapsed().as_secs_f64(),"immediate_draw_actions":known_draw.iter().filter(|x|**x).count(),"legal_actions":actions.len(),"root_proven":report.proven_value,"simulations":report.simulations,"new_visits":report.new_visits.iter().sum::<usize>(),"visits_to_preexisting_proved_draws":wasted,"selected_draw":known_draw[chosen],"selected_action":actions[chosen].to_string(),"selected_value":report.values[chosen],"draw_floor_choice":safe_choice,"policy_mass":report.policy_target.iter().sum::<f64>(),"selected_policy":report.policy_target[chosen]})
        );
    }
    Ok(())
}
