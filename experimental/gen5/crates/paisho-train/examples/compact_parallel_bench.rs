//! Fixed-position scalar/parallel ABBA; run through the paused-training wrapper.
use paisho_ai::{CompactValueModel, HeuristicWeights, MctsAgent, MctsConfig, MctsEvaluator};
use paisho_core::{legal_actions, GameRecord, Player, Position};
use paisho_train::compact_learning::load_model;
use std::{error::Error, fs, path::Path, time::Instant};

struct Scalar<'a>(&'a CompactValueModel);
impl MctsEvaluator for Scalar<'_> {
    fn ordering_matches_leaf(&self) -> bool {
        true
    }
    fn evaluate(
        &self,
        positions: &[Position],
        player: Player,
        _: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        Ok(positions
            .iter()
            .map(|p| self.0.evaluate(p, player))
            .collect())
    }
    fn evaluate_leaf(
        &self,
        position: &Position,
        player: Player,
        _: HeuristicWeights,
    ) -> Result<f32, String> {
        Ok(self.0.evaluate(position, player))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: compact_parallel_bench MODEL.json GAME.psr".into());
    }
    let model = load_model(Path::new(&args[0]))?.model()?;
    let scalar = Scalar(&model);
    let record: GameRecord = fs::read_to_string(&args[1])?.parse()?;
    let mut position = record.initial_position();
    let mut panel = Vec::new();
    for (i, &action) in record.actions().iter().enumerate() {
        if [record.actions().len() / 4, record.actions().len() * 3 / 4].contains(&i) {
            panel.push((i, position.clone(), legal_actions(&position)));
        }
        position.apply(action)?;
    }
    if panel.is_empty() {
        return Err("empty panel".into());
    }
    let global_start = Instant::now();
    let mut identities = std::collections::BTreeMap::new();
    for threads in [1, 4, 10] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        for budget in [32, 512] {
            let config = MctsConfig {
                simulations: budget,
                independent_trees: 1,
                ..MctsConfig::default()
            };
            for (index, position, actions) in &panel {
                if global_start.elapsed().as_secs() >= 180 {
                    return Err("bounded benchmark incomplete after 180s".into());
                }
                // One untimed warmup per position/config/pool.
                pool.install(|| {
                    MctsAgent::new(900 + *index as u64, config)
                        .unwrap()
                        .search_with_evaluator(position, actions, &model)
                })?;
                for (pass, parallel) in [false, true, true, false].into_iter().enumerate() {
                    let evaluator: &dyn MctsEvaluator = if parallel { &model } else { &scalar };
                    let start = Instant::now();
                    let report = pool.install(|| {
                        MctsAgent::new(900 + *index as u64, config)
                            .unwrap()
                            .search_with_evaluator(position, actions, evaluator)
                    })?;
                    let seconds = start.elapsed().as_secs_f64();
                    let identity = (
                        report.selected_index,
                        report.simulations,
                        report
                            .actions
                            .iter()
                            .map(|a| (a.action.to_string(), a.visits, a.value_sum.to_bits()))
                            .collect::<Vec<_>>(),
                    );
                    if let Some(expected) = identities.get(&(budget, index)) {
                        assert_eq!(&identity, expected, "fixed-budget search changed");
                    } else {
                        identities.insert((budget, index), identity);
                    }
                    println!(
                        "{}",
                        serde_json::json!({"threads":threads,"budget":budget,"decision":index,"candidates":actions.len(),"pass":pass,"parallel":parallel,"seconds":seconds,"evaluated_actions":report.evaluated_actions,"simulations":report.simulations,"exact_parity":true})
                    );
                }
            }
        }
    }
    Ok(())
}
