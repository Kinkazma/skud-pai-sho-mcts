//! Portable frozen Gen3 MCTS opponent, with the same evaluator and retained tree.
use paisho_ai::{MctsConfig, MctsSession};
use paisho_core::{
    legal_actions, Action, GameOutcome, GameRecord, Position, RuleProfileId, StandardSetup,
    BASIC_FLOWERS,
};
use paisho_train::micro_learning::gen5::{Frozen, OpponentSpec};
use serde_json::{json, Value};
use std::{env, error::Error, fs, io::{self, BufRead, Write}, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    io::Error::new(io::ErrorKind::InvalidInput, message.into()).into()
}

fn rules(arg: Option<&String>) -> Result<RuleProfileId> {
    match arg.map(String::as_str).unwrap_or("gen3") {
        "gen3" => Ok(RuleProfileId::SkudPaiSho2022V2),
        "gen5" => Ok(RuleProfileId::SkudPaiShoGen5V1),
        _ => Err(invalid("rules must be gen3 or gen5")),
    }
}

fn specs(path: &str) -> Result<Vec<OpponentSpec>> {
    let rows: Vec<OpponentSpec> = serde_json::from_slice(&fs::read(path)?)?;
    if rows.len() != 5 || rows.iter().enumerate().any(|(i, row)| row.generation != format!("3.{}", i + 1)) {
        return Err(invalid("opponents.json must list Gen3.1 through Gen3.5 in order"));
    }
    Ok(rows)
}

fn model(rows: &[OpponentSpec], generation: &str) -> Result<Frozen> {
    let row = rows.iter().find(|row| row.generation == generation)
        .ok_or_else(|| invalid(format!("unknown generation: {generation}")))?;
    Ok(Frozen::load(row)?)
}

fn session<'a>(model: &'a Frozen, budget: usize, seed: u64) -> Result<MctsSession<'a>> {
    if !(1..=2048).contains(&budget) {
        return Err(invalid("budget must be 1..2048"));
    }
    let mut search = MctsSession::new(seed, MctsConfig { simulations: budget, ..Default::default() }, model)?;
    search.set_solver(model.spec.solver);
    Ok(search)
}

fn report_state(record: &GameRecord, position: &Position) -> Value {
    json!({"rules":record.rules().as_str(), "decisions":record.actions().len(),
        "to_move":format!("{:?}", position.to_move()),
        "outcome":format!("{:?}", position.outcome())})
}

fn play(args: &[String]) -> Result<()> {
    if args.len() != 9 && args.len() != 10 {
        return Err(invalid("usage: paisho-gen3-suite play OPPONENTS.json HOST_GEN GUEST_GEN HOST_BUDGET GUEST_BUDGET SEED SETUP_INDEX MAX_DECISIONS OUTPUT.psr [gen3|gen5]"));
    }
    let rows = specs(&args[0])?;
    let host = model(&rows, &args[1])?;
    let guest = model(&rows, &args[2])?;
    let host_budget = args[3].parse()?;
    let guest_budget = args[4].parse()?;
    let seed: u64 = args[5].parse()?;
    let setup_index: usize = args[6].parse()?;
    let limit: usize = args[7].parse()?;
    if setup_index >= BASIC_FLOWERS.len() || limit == 0 {
        return Err(invalid("setup index must be 0..5 and decision limit positive"));
    }
    let output = Path::new(&args[8]);
    if output.exists() { return Err(invalid("output already exists")); }
    let mut record = GameRecord::with_rules(StandardSetup::balanced(BASIC_FLOWERS[setup_index]), rules(args.get(9))?);
    let mut position = record.initial_position();
    let mut sessions = [session(&host, host_budget, seed)?, session(&guest, guest_budget, seed.wrapping_add(1))?];
    for _ in 0..limit {
        if position.outcome() != GameOutcome::Ongoing { break; }
        let legal = legal_actions(&position);
        if legal.is_empty() { break; }
        let player = position.to_move();
        let selected = sessions[player.index()].search_until(&position, &legal, None)?.selected_index;
        let action = legal[selected];
        position.apply(action)?;
        record.push(action);
        for search in &mut sessions { search.advance(action); }
    }
    fs::write(output, record.to_string())?;
    println!("{}", json!({"record":output, "host":host.spec, "guest":guest.spec,
        "host_budget":host_budget, "guest_budget":guest_budget, "seed":seed,
        "setup_index":setup_index, "state":report_state(&record,&position)}));
    Ok(())
}

fn serve(args: &[String]) -> Result<()> {
    if args.len() != 4 && args.len() != 5 {
        return Err(invalid("usage: paisho-gen3-suite serve OPPONENTS.json GENERATION BUDGET SEED [gen3|gen5]"));
    }
    let rows = specs(&args[0])?;
    let model = model(&rows, &args[1])?;
    let budget = args[2].parse()?;
    let seed: u64 = args[3].parse()?;
    let rules = rules(args.get(4))?;
    let mut search = session(&model, budget, seed)?;
    let mut game: Option<(GameRecord, Position)> = None;
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{}", json!({"ready":true,"generation":model.spec.generation,
        "model_sha256":model.spec.sha256,"budget":budget,"solver":model.spec.solver,
        "rules":rules.as_str()}))?;
    stdout.flush()?;
    for line in stdin.lock().lines() {
        let result = (|| -> Result<Value> {
            let request: Value = serde_json::from_str(&line?)?;
            match request["cmd"].as_str().ok_or_else(|| invalid("cmd required"))? {
                "start" => {
                    let index = request["setup_index"].as_u64().ok_or_else(|| invalid("setup_index required"))? as usize;
                    let flower = BASIC_FLOWERS.get(index).ok_or_else(|| invalid("setup_index must be 0..5"))?;
                    let record = GameRecord::with_rules(StandardSetup::balanced(*flower), rules);
                    let position = record.initial_position();
                    search = session(&model, budget, request["seed"].as_u64().unwrap_or(seed))?;
                    game = Some((record, position));
                    let (record, position) = game.as_ref().unwrap();
                    Ok(report_state(record, position))
                }
                "choose" => {
                    let (_, position) = game.as_ref().ok_or_else(|| invalid("start a game first"))?;
                    if position.outcome() != GameOutcome::Ongoing { return Err(invalid("game is terminal")); }
                    let actions = legal_actions(position);
                    if actions.is_empty() { return Err(invalid("no legal action")); }
                    let report = search.search_until(position, &actions, None)?;
                    Ok(json!({"action":actions[report.selected_index].to_string(),
                        "simulations":report.simulations, "inherited_visits":search.reuse_statistics().inherited_root_visits}))
                }
                "apply" => {
                    let action: Action = request["action"].as_str().ok_or_else(|| invalid("action required"))?.parse()?;
                    let (record, position) = game.as_mut().ok_or_else(|| invalid("start a game first"))?;
                    position.apply(action)?;
                    record.push(action);
                    search.advance(action);
                    Ok(report_state(record, position))
                }
                "state" => {
                    let (record, position) = game.as_ref().ok_or_else(|| invalid("start a game first"))?;
                    Ok(report_state(record, position))
                }
                "record" => {
                    let (record, _) = game.as_ref().ok_or_else(|| invalid("start a game first"))?;
                    Ok(json!({"psr":record.to_string()}))
                }
                "quit" => Ok(json!({"bye":true})),
                _ => Err(invalid("unknown cmd")),
            }
        })();
        let done = result.as_ref().is_ok_and(|value| value["bye"] == true);
        writeln!(stdout, "{}", match result { Ok(value) => value, Err(error) => json!({"error":error.to_string()}) })?;
        stdout.flush()?;
        if done { break; }
    }
    Ok(())
}

fn main() -> Result<()> {
    env::set_var("VECLIB_MAXIMUM_THREADS", "1");
    let args: Vec<_> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("play") => play(&args[1..]),
        Some("serve") => serve(&args[1..]),
        _ => Err(invalid("usage: paisho-gen3-suite play|serve ...")),
    }
}
