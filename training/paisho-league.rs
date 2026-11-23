//! Resident published Gen3 opponents. All five models load once per process.
use paisho_ai::{MctsConfig, MctsSession};
use paisho_core::{legal_actions, AccentLoadout, Action, GameRecord, Player, Position, RuleProfileId};
use paisho_train::micro_learning::gen5::{Frozen, OpponentSpec};
use serde_json::{json, Value};
use std::{env, error::Error, fs, io::{self, BufRead, Write}};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
fn invalid(message: &str) -> Box<dyn Error> { io::Error::new(io::ErrorKind::InvalidInput, message).into() }
struct Match<'a> { position: Position, search: MctsSession<'a>, player: Player }
fn create<'a>(models: &'a [Frozen], request: &Value) -> Result<Match<'a>> {
    let generation=request["generation"].as_str().ok_or_else(||invalid("generation required"))?;
    let model=models.iter().find(|m|m.spec.generation==generation).ok_or_else(||invalid("unknown generation"))?;
    let budget=request["budget"].as_u64().ok_or_else(||invalid("budget required"))? as usize;
    if ![1,2,3,4,6,8,16,32,64,128].contains(&budget) { return Err(invalid("unsupported curriculum budget")); }
    let seed=request["seed"].as_u64().ok_or_else(||invalid("seed required"))?;
    let player=match request["mcts_player"].as_u64() {Some(0)=>Player::Host,Some(1)=>Player::Guest,_=>return Err(invalid("player required"))};
    let record=GameRecord::with_empty_start(AccentLoadout::balanced(),AccentLoadout::balanced(),RuleProfileId::SkudPaiSho2022V2);
    let mut search=MctsSession::new(seed,MctsConfig {simulations:budget,..Default::default()},model)?;
    search.set_solver(model.spec.solver);
    Ok(Match {position:record.initial_position(),search,player})
}
fn choose(game: &mut Match<'_>) -> Result<Value> {
    if game.position.to_move()!=game.player {return Err(invalid("not the MCTS turn"));}
    let legal=legal_actions(&game.position);
    if legal.is_empty() {return Err(invalid("no legal move"));}
    let report=game.search.search_until(&game.position,&legal,None)?;
    Ok(json!({"action":legal[report.selected_index].to_string(),"simulations":report.simulations,
        "inherited_visits":game.search.reuse_statistics().inherited_root_visits}))
}
fn apply(game: &mut Match<'_>, action: Action) -> Result<()> {
    game.position.apply(action)?;game.search.advance(action);Ok(())
}
fn main() -> Result<()> {
    env::set_var("RAYON_NUM_THREADS","1");env::set_var("VECLIB_MAXIMUM_THREADS","1");
    let path=env::args().nth(1).ok_or_else(||invalid("opponents manifest required"))?;
    let specs:Vec<OpponentSpec>=serde_json::from_slice(&fs::read(path)?)?;
    if specs.len()!=5 || specs.iter().enumerate().any(|(i,s)|s.generation!=format!("3.{}",i+1)) {return Err(invalid("five ordered generations required"));}
    let models=specs.iter().map(Frozen::load).collect::<Result<Vec<_>>>()?;
    let stdin=io::stdin();let mut stdout=io::BufWriter::new(io::stdout().lock());
    writeln!(stdout,"{}",json!({"ready":true,"model_loads":5,"opponents":specs}))?;stdout.flush()?;
    let mut game:Option<Match<'_>>=None;
    for line in stdin.lock().lines() {
        let request:Value=serde_json::from_str(&line?)?;
        let result=(|| -> Result<Value> {
            match request["op"].as_str().unwrap_or("") {
                "reset" => { game=Some(create(&models,&request)?);Ok(json!({"empty_start":true,"pieces":0})) },
                "restore" => {
                    let record:GameRecord=request["record"].as_str().ok_or_else(||invalid("record required"))?.parse()?;
                    if !record.uses_empty_start() || record.rules()!=RuleProfileId::SkudPaiSho2022V2 {return Err(invalid("wrong initial state or rules"));}
                    let mut restored=create(&models,&request)?;
                    for &action in record.actions() {
                        if restored.position.to_move()==restored.player && choose(&mut restored)?["action"]!=action.to_string() {
                            return Err(invalid("replayed MCTS decision differs from saved game"));
                        }
                        apply(&mut restored,action)?;
                    }
                    let replayed_choice=if request["choice"].is_null() {Value::Null} else {
                        let choice=choose(&mut restored)?;
                        if choice["action"]!=request["choice"]["action"] {return Err(invalid("saved pending choice differs"));}
                        choice
                    };
                    game=Some(restored);Ok(json!({"restored":true,"decisions":record.actions().len(),"choice":replayed_choice}))
                },
                "choose" => choose(game.as_mut().ok_or_else(||invalid("reset required"))?),
                "apply" => {
                    let action:Action=request["action"].as_str().ok_or_else(||invalid("action required"))?.parse()?;
                    apply(game.as_mut().ok_or_else(||invalid("reset required"))?,action)?;Ok(json!({"applied":true}))
                },
                "quit" => Ok(json!({"bye":true})),
                _ => Err(invalid("unknown operation")),
            }
        })();
        let quit=request["op"]=="quit";
        let reply=match result {Ok(data)=>json!({"ok":true,"data":data}),Err(error)=>json!({"ok":false,"error":error.to_string()})};
        writeln!(stdout,"{reply}")?;stdout.flush()?;if quit {break;}
    }
    Ok(())
}
