//! Frozen post-fit diagnostics, fixed seed/budget. No matches or production writes.
use paisho_ai::{prove_forced_win, MctsConfig, MctsSession};
use paisho_core::{legal_actions, GameOutcome, GameRecord, Player, RuleProfileId};
use paisho_train::gen32::Artifact;
use serde_json::json;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
struct Case {
    name: String,
    path: PathBuf,
    prefix: usize,
    kind: String,
    attacker: Player,
    horizon: usize,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let out = Path::new(&a[2]);
    let fixtures = Path::new("crates/paisho-ai/tests/fixtures");
    let mut cases = vec![];
    for n in [5, 9, 13] {
        cases.push(Case {
            name: format!("r5-{}", n + 1),
            path: fixtures.join("gen34-r5-capture.psr"),
            prefix: n,
            kind: "capture".into(),
            attacker: Player::Guest,
            horizon: 2,
        });
    }
    let human = Path::new("benchmarks/results/gen32-ring-foresight-2026-09-11/human");
    for (stem, n, attacker) in [
        (
            "partie-20260910T220511Z-2a420b3f40456a1708381a23c2f02c57",
            25,
            Player::Guest,
        ),
        (
            "partie-20260910T220624Z-a370f0c673ddb3b4fffdeec0c7243d5f",
            20,
            Player::Guest,
        ),
        (
            "partie-20260910T220926Z-3886efd9cc814b2046259f42c3214d36",
            19,
            Player::Host,
        ),
    ] {
        cases.push(Case {
            name: format!("duat-{n}"),
            path: human.join(format!("{stem}.psr")),
            prefix: n,
            kind: "defend".into(),
            attacker,
            horizon: 2,
        });
    }
    for line in fs::read_to_string(fixtures.join("tactical_cases.tsv"))?
        .lines()
        .skip(1)
    {
        let c: Vec<_> = line.split('\t').collect();
        let horizon = c[5].parse()?;
        if horizon > 2 {
            continue;
        }
        cases.push(Case {
            name: c[0].into(),
            path: fixtures.join(c[1]),
            prefix: c[2].parse()?,
            kind: c[3].into(),
            attacker: if c[4] == "Host" {
                Player::Host
            } else {
                Player::Guest
            },
            horizon,
        });
    }
    let models: Vec<_> = [
        ("parent", PathBuf::from(&a[1])),
        ("linear", out.join("linear.json")),
        ("residual7", out.join("residual7.json")),
        ("residual71", out.join("residual71.json")),
        ("residual731", out.join("residual731.json")),
    ]
    .into_iter()
    .map(|(n, p)| Ok((n, Artifact::load(&p)?.model()?)))
    .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let mut file = fs::File::create(out.join("search.jsonl"))?;
    for case in cases {
        let record: GameRecord = fs::read_to_string(&case.path)?.parse()?;
        let mut prefix = GameRecord::with_rules(record.setup(), RuleProfileId::SkudPaiSho2022V2);
        for action in &record.actions()[..case.prefix] {
            prefix.push(*action);
        }
        let position = prefix.replay()?;
        assert_eq!(position.outcome(), GameOutcome::Ongoing);
        for (name, model) in &models {
            let mut p = position.clone();
            let chooser = p.to_move();
            let mut played = vec![];
            let mut searches = vec![];
            let mut session = MctsSession::new(
                17,
                MctsConfig {
                    simulations: 512,
                    ..Default::default()
                },
                model,
            )?;
            session.set_solver(true);
            let began = Instant::now();
            let mut recorded_trace = vec![];
            while p.outcome() == GameOutcome::Ongoing && p.to_move() == chooser && played.len() < 4
            {
                let legal = legal_actions(&p);
                let report = pool.install(|| session.search_until(&p, &legal, None))?;
                if played.is_empty() && case.kind == "capture" {
                    recorded_trace=session.trace_path(&record.actions()[case.prefix..case.prefix+2],chooser).iter().map(|s|json!({"action":s.action.to_string(),"stage":format!("{:?}",s.stage),"visits":s.visits,"mean":s.mean_value,"cached_leaf":s.cached_leaf_value})).collect();
                }
                let chosen = report.actions[report.selected_index];
                searches.push(json!({"simulations":report.simulations,"depth":report.maximum_depth,"chosen_visits":chosen.visits,"chosen_mean":chosen.mean_value(),"evaluated_candidates":report.evaluated_actions}));
                played.push(chosen.action.to_string());
                p.apply(chosen.action)?;
                session.advance(chosen.action);
            }
            assert!(
                p.outcome() != GameOutcome::Ongoing || p.to_move() != chooser,
                "incomplete chooser turn"
            );
            let mut captures = vec![];
            if p.outcome() == GameOutcome::Ongoing {
                for action in legal_actions(&p) {
                    let mut q = p.clone();
                    if let Some(tile) = q.apply(action)?.captured {
                        if tile.owner == chooser {
                            captures.push(json!({"action":action.to_string(),"piece":format!("{:?}",tile.kind)}));
                        }
                    }
                }
            }
            let h = if case.kind == "attack" {
                case.horizon.saturating_sub(played.len())
            } else {
                case.horizon
            };
            let proof = prove_forced_win(&p, case.attacker, h);
            let mut after_recorded = position.clone();
            let after_capture_value = if case.kind == "capture" {
                after_recorded.apply(record.actions()[case.prefix])?;
                after_recorded.apply(record.actions()[case.prefix + 1])?;
                Some(model.value_at(&after_recorded, chooser))
            } else {
                None
            };
            let row = json!({"case":case.name,"path":case.path,"prefix":case.prefix,"kind":case.kind,"model":name,"seed":17,"budget":512,"actions":played,"searches":searches,"root_value":model.value_at(&position,chooser),"after_recorded_capture_value":after_capture_value,"recorded_trace":recorded_trace,"direct_capture_replies":captures,"attacker":format!("{:?}",case.attacker),"remaining_horizon":h,"attacker_forced_win":proof.is_forced_win(),"oracle_nodes":proof.visited_positions,"final_outcome":format!("{:?}",p.outcome()),"elapsed_seconds_contended":began.elapsed().as_secs_f64()});
            writeln!(file, "{row}")?;
            file.flush()?;
            println!(
                "{} {} {:?} forced={}",
                case.name,
                name,
                played,
                proof.is_forced_win()
            );
        }
    }
    Ok(())
}
