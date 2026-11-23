//! Rules-only, batched JSONL environment. No search, neural inference or rewards.
use paisho_core::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Describe,
    Reset { rules: String, openings: Vec<u8> },
    ResetEmpty { rules: String, slots: usize },
    ResetEmptySlots { rules: String, slots: Vec<usize> },
    ResetSlots { rules: String, openings: Vec<SlotOpening> },
    Step { moves: Vec<Move>, #[serde(default)] compact: bool },
    Export,
    Restore { games: Vec<SavedGame> },
    LoadRecord { record: String },
    AuditRecords { records: Vec<String> },
    ReplayExamples { record: String, rules: String, start: usize, count: usize },
    ReplayBatch { chunks: Vec<ReplayChunk> },
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayChunk { record: String, rules: String, start: usize, count: usize }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Move { slot: usize, index: usize }
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SlotOpening { slot: usize, opening: u8 }
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SavedGame { record: String }
#[derive(Clone)]
struct Game { position: Position, record: GameRecord, history: Vec<[usize; 5]>, actions: Vec<Action> }

// Only independent rule computations run concurrently. Result order and the
// all-or-nothing validation of a step remain identical to the serial bridge.
fn parallel_map<T: Sync, R: Send, F: Fn(&T) -> R + Sync>(items: &[T], f: F) -> Vec<R> {
    let workers = std::env::var("GPT_SKUD_ENGINE_WORKERS").ok()
        .and_then(|s| s.parse::<usize>().ok()).unwrap_or(1).clamp(1, 64).min(items.len());
    if workers <= 1 { return items.iter().map(f).collect(); }
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    let pool=POOL.get_or_init(||rayon::ThreadPoolBuilder::new()
        .num_threads(std::env::var("GPT_SKUD_ENGINE_WORKERS").ok()
            .and_then(|s|s.parse().ok()).unwrap_or(1).clamp(1,64))
        .build().expect("cannot create rules workers"));
    // Indexed parallel collect preserves input order, with persistent workers
    // and work stealing rather than a new OS thread group on every decision.
    pool.install(||items.par_iter().map(&f).collect())
}

fn replay_examples(chunk: &ReplayChunk) -> Result<Value, String> {
    let ReplayChunk {record: text, rules, start, count} = chunk;
    if *count==0 || *count>1024 {return Err("example chunks require 1..1024 decisions".into());}
    let record:GameRecord=text.parse().map_err(|e|format!("{e}"))?;
    if record.rules().as_str()!=rules {return Err("human record rule identity differs".into());}
    let mut position=record.initial_position();
    let mut history=Vec::new();
    let mut examples=Vec::new();
    for (index,&action) in record.actions().iter().enumerate().take(start.saturating_add(*count)) {
        let player=position.to_move().index();
        let legal=legal_actions(&position);
        let target=legal.iter().position(|&a|a==action).ok_or("human next action is illegal")?;
        if index>=*start {
            let mut observation=observe_with_actions(&position,record.setup(),&legal,record.uses_empty_start());
            observation["history_length"]=json!(history.len());
            examples.push(json!({"observation":observation,"target":target,"index":index}));
        }
        position.apply(action).map_err(|e|format!("{e}"))?;
        let fields=action_fields(action);
        history.push([fields[0],fields[1],fields[2],fields[3],player]);
    }
    Ok(json!({"examples":examples,"history":history,"decisions":record.actions().len()}))
}

fn point_kind(p: PointType) -> u8 {
    match p {
        PointType::NonPlayable => 0, PointType::Gate => 1, PointType::Neutral => 2,
        PointType::Red => 3, PointType::White => 4, PointType::RedWhite => 5,
        PointType::RedNeutral => 6, PointType::WhiteNeutral => 7,
        PointType::RedWhiteNeutral => 8,
    }
}
fn action_fields(action: Action) -> [usize; 4] {
    // family, tile category (0 absent), source, destination; 289 = absent point.
    let none = CELL_COUNT;
    match action {
        Action::Plant { flower, gate } => [0, flower as usize + 1, none, gate.dense_index()],
        Action::Arrange { from, to } => [1, 0, from.dense_index(), to.dense_index()],
        Action::SkipHarmonyBonus => [2, 0, none, none],
        Action::PlayAccent { accent, placement: AccentPlacement::At(at) } =>
            [3, TileKind::Accent(accent).index()+1, none, at.dense_index()],
        Action::PlayAccent { accent, placement: AccentPlacement::BoatMove { flower, destination } } =>
            [4, TileKind::Accent(accent).index()+1, flower.dense_index(), destination.dense_index()],
        Action::PlantSpecial { flower, gate } => [5, flower.kind().index()+1, none, gate.dense_index()],
        Action::BonusPlantBasic { flower, gate } => [6, flower as usize+1, none, gate.dense_index()],
    }
}
fn observe(game: &Game) -> Value {
    let mut value=observe_with_actions(&game.position,game.record.setup(),&game.actions,game.record.uses_empty_start());
    value["history"]=json!(game.history);
    value
}
#[cfg(test)]
fn observe_state(p: &Position, setup: StandardSetup) -> Value {
    observe_with_actions(p,setup,&legal_actions(p),false)
}
fn observe_with_actions(p: &Position, setup: StandardSetup, actions: &[Action], empty_start: bool) -> Value {
    let player = p.to_move();
    let board: Vec<u8> = playable_coordinates().map(|at| match p.board().get(at) {
        None => 0,
        Some(tile) => (1+tile.kind.index()+if tile.owner == player {0} else {12}) as u8,
    }).collect();
    let reserves: Vec<u8> = [player, player.opponent()].into_iter().flat_map(|owner|
        STANDARD_TILE_KINDS.into_iter().map(move |kind| p.reserve(owner).count(kind))).collect();
    json!({
        "board": board, "reserves": reserves, "player": player.index(),
        "phase": if p.phase()==TurnPhase::Main {0} else {1},
        "turns": p.completed_turns(), "rules": p.rule_profile().as_str(),
        "setup": [if empty_start {-1} else {setup.starting_flower as i32},
            setup.host_accents.count(Accent::Rock) as usize,setup.host_accents.count(Accent::Wheel) as usize,
            setup.host_accents.count(Accent::Knotweed) as usize,setup.host_accents.count(Accent::Boat) as usize,
            setup.guest_accents.count(Accent::Rock) as usize,setup.guest_accents.count(Accent::Wheel) as usize,
            setup.guest_accents.count(Accent::Knotweed) as usize,setup.guest_accents.count(Accent::Boat) as usize],
        "outcome": match p.outcome() { GameOutcome::Ongoing => -2, GameOutcome::Draw => -1,
            GameOutcome::Win(winner) => winner.index() as i8 },
        "actions": actions.iter().copied().map(action_fields).collect::<Vec<_>>()
    })
}
fn from_record(text: &str) -> Result<Game, String> {
    let record: GameRecord = text.parse().map_err(|e| format!("invalid record: {e}"))?;
    let mut position=record.initial_position();
    let mut history=Vec::new();
    for &action in record.actions() {
        let player=position.to_move().index();
        position.apply(action).map_err(|e|format!("illegal record: {e}"))?;
        let fields=action_fields(action);
        history.push([fields[0],fields[1],fields[2],fields[3],player]);
    }
    let actions=legal_actions(&position);
    Ok(Game {position, record,history,actions})
}
fn handle(request: Request, games: &mut Vec<Game>) -> Result<Value, String> {
    match request {
        Request::Describe => Ok(json!({"protocol":1,"points":playable_coordinates().map(|at|
            [at.dense_index(),point_kind(point_type_at(at)) as usize,at.row(),at.column()])
            .collect::<Vec<_>>(),"rules":[RuleProfileId::CURRENT.as_str(),
            RuleProfileId::SkudPaiShoGen5V1.as_str()]})),
        Request::ResetEmpty {rules, slots} => {
            if slots == 0 || slots > 128 { return Err("use 1..128 empty slots".into()); }
            let rules: RuleProfileId = rules.parse().map_err(|e|format!("{e}"))?;
            let record = GameRecord::with_empty_start(AccentLoadout::balanced(), AccentLoadout::balanced(), rules);
            let position = record.initial_position();
            let actions = legal_actions(&position);
            *games = (0..slots).map(|_|Game {position:position.clone(), record:record.clone(), history:Vec::new(), actions:actions.clone()}).collect();
            Ok(json!(parallel_map(games, observe)))
        },
        Request::ResetEmptySlots {rules, slots} => {
            let rules: RuleProfileId = rules.parse().map_err(|e|format!("{e}"))?;
            let mut unique=std::collections::HashSet::new();
            if slots.is_empty() || slots.iter().any(|s|*s>=games.len() || !unique.insert(*s)) {
                return Err("invalid empty replacement slots".into());
            }
            for &slot in &slots {
                let record=GameRecord::with_empty_start(AccentLoadout::default(),AccentLoadout::default(),rules);
                let position=record.initial_position();
                let actions=legal_actions(&position);
                games[slot]=Game {position,record,history:Vec::new(),actions};
            }
            Ok(json!(parallel_map(&slots,|s|json!({"slot":s,"observation":observe(&games[*s])}))))
        },
        Request::Reset {rules, openings} => {
            if openings.is_empty() || openings.len()>128 || openings.iter().any(|x|*x>=6) {
                return Err("use 1..128 openings with flower indices 0..5".into());
            }
            let rules: RuleProfileId = rules.parse().map_err(|e|format!("{e}"))?;
            if rules==RuleProfileId::SkudPaiSho2022 {return Err("legacy incomplete ring rules are disabled".into());}
            *games = openings.into_iter().map(|n| {
                let record = GameRecord::with_rules(StandardSetup::balanced(BASIC_FLOWERS[n as usize]),rules);
                let position=record.initial_position();
                let actions=legal_actions(&position);
                Game {position,record,history:Vec::new(),actions}
            }).collect();
            Ok(json!(parallel_map(games, observe)))
        },
        Request::ResetSlots {rules, openings} => {
            let rules:RuleProfileId=rules.parse().map_err(|e|format!("{e}"))?;
            if rules==RuleProfileId::SkudPaiSho2022 {return Err("legacy incomplete ring rules are disabled".into());}
            let mut slots=std::collections::HashSet::new();
            if openings.is_empty() || openings.iter().any(|x| x.slot>=games.len() || x.opening>=6 || !slots.insert(x.slot)) {
                return Err("invalid replacement slot/opening".into());
            }
            for item in &openings {
                let record=GameRecord::with_rules(StandardSetup::balanced(BASIC_FLOWERS[item.opening as usize]),rules);
                let position=record.initial_position();
                let actions=legal_actions(&position);
                games[item.slot]=Game {position,record,history:Vec::new(),actions};
            }
            Ok(json!(parallel_map(&openings, |x|json!({"slot":x.slot,"observation":observe(&games[x.slot])}))))
        },
        Request::Step {moves, compact} => {
            // Validate the complete batch before changing any slot.
            let mut slots = std::collections::HashSet::new();
            if moves.iter().any(|m| !slots.insert(m.slot)) {return Err("duplicate slot".into());}
            let staged = parallel_map(&moves, |m| {
                let game = games.get(m.slot).ok_or("invalid slot")?;
                let mut position = game.position.clone();
                let mover=position.to_move().index();
                let action = *game.actions.get(m.index).ok_or("invalid legal-action index")?;
                position.apply(action).map_err(|e|format!("{e}"))?;
                let actions=legal_actions(&position);
                Ok((m.slot,position,action,mover,actions))
            }).into_iter().collect::<Result<Vec<_>, String>>()?;
            for (slot,position,action,mover,actions) in &staged {
                games[*slot].position=position.clone();
                games[*slot].actions=actions.clone();
                games[*slot].record.push(*action);
                let fields=action_fields(*action);
                games[*slot].history.push([fields[0],fields[1],fields[2],fields[3],*mover]);
            }
            let result = parallel_map(&staged, |(slot,_,_,_,_)| {
                let game=&games[*slot];
                let observation=if compact {observe_with_actions(&game.position,game.record.setup(),&game.actions,game.record.uses_empty_start())} else {observe(game)};
                if compact {json!({"slot":slot,"observation":observation,"history_append":game.history.last()})}
                else {json!({"slot":slot,"observation":observation})}
            });
            Ok(json!(result))
        },
        Request::Export => Ok(json!(games.iter().map(|g|SavedGame {record:g.record.to_string()}).collect::<Vec<_>>())),
        Request::Restore {games: saved} => {
            if saved.is_empty() || saved.len()>128 {return Err("invalid slot count".into());}
            let restored = parallel_map(&saved, |s|from_record(&s.record)).into_iter().collect::<Result<Vec<_>,_>>()?;
            *games = restored;
            Ok(json!(parallel_map(games, observe)))
        },
        Request::LoadRecord {record} => {
            let game = from_record(&record)?;
            let result = observe(&game);
            *games = vec![game];
            Ok(result)
        },
        Request::AuditRecords {records} => {
            let results:Vec<_>=records.iter().map(|text| {
                let checked=(|| -> Result<Value,String> {
                    let record:GameRecord=text.parse().map_err(|e|format!("{e}"))?;
                    let mut position=record.initial_position();
                    for (index,&action) in record.actions().iter().enumerate() {
                        if !legal_actions(&position).contains(&action) {
                            return Err(format!("action {} is absent from legal actions",index+1));
                        }
                        position.apply(action).map_err(|e|format!("{e}"))?;
                    }
                    Ok(json!({"valid":true,"rules":record.rules().as_str(),"decisions":record.actions().len(),
                        "outcome":match position.outcome() {GameOutcome::Ongoing=>-2,GameOutcome::Draw=>-1,
                            GameOutcome::Win(p)=>p.index() as i8}}))
                })();
                checked.unwrap_or_else(|error|json!({"valid":false,"error":error}))
            }).collect();
            Ok(json!(results))
        },
        Request::ReplayExamples {record: text,rules,start,count} => {
            replay_examples(&ReplayChunk {record:text, rules, start, count})
        },
        Request::ReplayBatch {chunks} => {
            if chunks.is_empty() || chunks.len()>256 {return Err("use 1..256 replay chunks".into());}
            let results=parallel_map(&chunks, replay_examples).into_iter().collect::<Result<Vec<_>,_>>()?;
            Ok(json!(results))
        },
    }
}
fn main() {
    let stdin = io::stdin();
    let mut output = io::BufWriter::new(io::stdout().lock());
    let mut games = Vec::new();
    for line in stdin.lock().lines() {
        let result = line.map_err(|e|e.to_string()).and_then(|line| {
            let request = serde_json::from_str(&line).map_err(|e|format!("invalid request: {e}"))?;
            handle(request,&mut games)
        });
        let response = match result {Ok(data)=>json!({"ok":true,"data":data}),
            Err(error)=>json!({"ok":false,"error":error})};
        if serde_json::to_writer(&mut output,&response).is_err() || writeln!(output).is_err() || output.flush().is_err() {break;}
    }
}
