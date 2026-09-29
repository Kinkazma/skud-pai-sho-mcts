//! Read-only, fixed-budget tactical diagnostic of a frozen Gen3.2 artifact.
//! No training or Elo claim; fixture prefixes are explicitly revalidated as V2.
use paisho_ai::{gen32_tactical_guard, prove_forced_win, MctsConfig, MctsSession};
use paisho_core::{legal_actions, GameOutcome, GameRecord, Player, RuleProfileId};
use paisho_train::gen32::Artifact;
use serde_json::json;
use std::{fs, path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let model_path = Path::new(args.get(1).expect("model output"));
    let artifact: Artifact = serde_json::from_slice(&fs::read(model_path)?)?;
    let model = artifact.model()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let directory = Path::new("crates/paisho-ai/tests/fixtures");
    let mut rows = vec![];
    let cases_path = args
        .get(3)
        .map(|p| Path::new(p).to_path_buf())
        .unwrap_or_else(|| directory.join("tactical_cases.tsv"));
    for line in fs::read_to_string(cases_path)?.lines().skip(1) {
        let c: Vec<_> = line.split('\t').collect();
        let record: GameRecord = fs::read_to_string(directory.join(c[1]))?.parse()?;
        let mut prefix = GameRecord::with_rules(record.setup(), RuleProfileId::SkudPaiSho2022V2);
        for action in &record.actions()[..c[2].parse::<usize>()?] {
            prefix.push(*action);
        }
        let position = prefix.replay()?;
        assert_eq!(position.outcome(), GameOutcome::Ongoing);
        let attacker = if c[4] == "Host" {
            Player::Host
        } else {
            Player::Guest
        };
        let horizon: usize = c[5].parse()?;
        let initial = prove_forced_win(&position, attacker, horizon);
        for (name, root, internal, solver, guard_limit) in [
            ("baseline", 1., 1., false, 0),
            ("solver", 1., 1., true, 0),
            ("guard", 1., 1., true, 32768),
            ("guard_narrow_inside", 1., 0.5, true, 32768),
        ] {
            for seed in [17, 731] {
                let config = MctsConfig {
                    simulations: 512,
                    root_widening_factor: root,
                    progressive_widening_factor: internal,
                    ..MctsConfig::default()
                };
                let mut session = MctsSession::new(seed, config, &model)?;
                session.set_solver(solver);
                let mut p = position.clone();
                let mut played = vec![];
                let mut searches = vec![];
                let began = Instant::now();
                let chooser = p.to_move();
                // A complete chooser turn can include a harmony bonus decision.
                while p.outcome() == GameOutcome::Ongoing
                    && p.to_move() == chooser
                    && played.len() < 4
                {
                    let legal = legal_actions(&p);
                    let report = pool.install(|| session.search_until(&p, &legal, None))?;
                    let guard = gen32_tactical_guard(&p, &legal, &report, guard_limit, None);
                    let selected = if session.action_proofs()[report.selected_index]
                        == Some(GameOutcome::Win(p.to_move()))
                    {
                        report.selected_index
                    } else {
                        guard.selected
                    };
                    let chosen = report.actions[selected];
                    searches.push(json!({"maximum_depth":report.maximum_depth,"guard_nodes":guard.visited,"guard_exhausted":guard.exhausted,"guard_refuted":guard.refuted,"guard_safe":guard.safe,"root_legal":legal.len(),"root_visited":report.actions.iter().filter(|a|a.visits>0).count(),"chosen_visits":chosen.visits,"chosen_value":chosen.mean_value(),"expanded":report.expanded_nodes,"evaluated_candidates":report.evaluated_actions}));
                    played.push(chosen.action.to_string());
                    p.apply(chosen.action)?;
                    session.advance(chosen.action);
                }
                let after = if c[3] == "attack" {
                    horizon.saturating_sub(played.len())
                } else {
                    horizon
                };
                let proof = prove_forced_win(&p, attacker, after);
                let success = if c[3] == "attack" {
                    proof.is_forced_win()
                } else {
                    !proof.is_forced_win()
                };
                let row = json!({"case":c[0],"kind":c[3],"horizon":horizon,"initial_forced":initial.is_forced_win(),"initial_oracle_nodes":initial.visited_positions,"config":name,"seed":seed,"success":success,"actions":played,"searches":searches,"oracle_nodes":proof.visited_positions,"elapsed_seconds_contended":began.elapsed().as_secs_f64()});
                println!("{}", row);
                rows.push(row);
            }
        }
    }
    fs::write(
        &args[2],
        serde_json::to_vec_pretty(
            &json!({"model":model_path,"updates":artifact.updates,"rules":"skud-pai-sho-2022-03-14-v2","rows":rows}),
        )?,
    )?;
    Ok(())
}
