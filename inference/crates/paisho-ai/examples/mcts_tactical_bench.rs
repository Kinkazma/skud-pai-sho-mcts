use std::collections::BTreeMap;
use std::time::Instant;

use paisho_ai::{
    evaluate_position, prove_forced_win, Agent, AgentTelemetry, HeuristicWeights, MctsAgent,
    MctsConfig, EXHAUSTIVE_ACTION_RANKING,
};
use paisho_core::{legal_actions, GameOutcome, GameRecord, Player, Position};
use rayon::prelude::*;

const CASES: &str = include_str!("../tests/fixtures/tactical_cases.tsv");
const OFFICIAL_RING: &str = include_str!("../tests/fixtures/site_bot_v1_ring_finish.psr");
const GUEST_WHEEL: &str = include_str!("../tests/fixtures/tactical_guest_wheel_finish.psr");
const HOST_WHEEL: &str = include_str!("../tests/fixtures/tactical_host_wheel_finish.psr");
const GUEST_THREE: &str = include_str!("../tests/fixtures/tactical_guest_forced_three.psr");
const HOST_THREE: &str = include_str!("../tests/fixtures/tactical_host_forced_three.psr");

fn main() {
    let mut arguments = std::env::args().skip(1);
    let simulations = positive_argument(&mut arguments, "simulations", 128);
    let repetitions = positive_argument(&mut arguments, "repetitions", 16);
    let policies = arguments
        .next()
        .map(|text| parse_policies(&text))
        .unwrap_or_else(|| {
            vec![
                WideningPolicy::uniform(0.5),
                WideningPolicy::uniform(1.0),
                WideningPolicy::uniform(2.0),
                WideningPolicy::uniform(4.0),
                WideningPolicy::new(2.0, 1.0),
                WideningPolicy::new(4.0, 1.0),
            ]
        });
    if arguments.next().is_some() {
        eprintln!("usage: mcts_tactical_bench [simulations] [repetitions] [root/internal,...]");
        std::process::exit(2);
    }

    let scenarios = scenarios();
    println!("Exact short-horizon tactical corpus");
    for scenario in &scenarios {
        let profile = tactical_profile(scenario);
        println!(
            "case={} kind={} horizon={} legal={} tactical={} best_static_rank={}..{}",
            scenario.id,
            scenario.kind.label(),
            scenario.horizon,
            profile.legal_actions,
            profile.tactical_actions,
            profile.best_rank,
            profile.worst_tied_rank,
        );
    }

    let jobs: Vec<_> = policies
        .iter()
        .copied()
        .flat_map(|policy| {
            (0..scenarios.len()).flat_map(move |case_index| {
                (0..repetitions).map(move |repetition| Job {
                    policy,
                    case_index,
                    repetition,
                })
            })
        })
        .collect();
    let started = Instant::now();
    let results: Vec<_> = jobs
        .into_par_iter()
        .map(|job| run_job(job, &scenarios, simulations))
        .collect();
    let elapsed = started.elapsed();

    let mut workers = results
        .iter()
        .filter_map(|result| result.worker)
        .collect::<Vec<_>>();
    workers.sort_unstable();
    workers.dedup();
    println!("\nEqual-budget MCTS widening comparison");
    println!("simulations per decision: {simulations}");
    println!("repetitions per case: {repetitions}");
    println!(
        "observed benchmark workers: {}/{}",
        workers.len().max(1),
        rayon::current_num_threads()
    );

    for policy in policies {
        let selected: Vec<_> = results
            .iter()
            .filter(|result| result.policy == policy)
            .collect();
        let successes = selected.iter().filter(|result| result.success).count();
        let attacks = selected
            .iter()
            .filter(|result| result.kind == CaseKind::Attack)
            .collect::<Vec<_>>();
        let defenses = selected
            .iter()
            .filter(|result| result.kind == CaseKind::Defend)
            .collect::<Vec<_>>();
        let telemetry = selected
            .iter()
            .fold(Totals::default(), |mut total, result| {
                total.add(result.telemetry);
                total
            });
        println!(
            "root/internal={}/{}: total={successes}/{} attack={}/{} defend={}/{} eval/sim={:.2} expanded/sim={:.3} max_depth={}",
            policy.root,
            policy.internal,
            selected.len(),
            attacks.iter().filter(|result| result.success).count(),
            attacks.len(),
            defenses.iter().filter(|result| result.success).count(),
            defenses.len(),
            telemetry.per_simulation(telemetry.evaluated_actions),
            telemetry.per_simulation(telemetry.expanded_nodes),
            telemetry.maximum_depth,
        );

        let mut per_case = BTreeMap::new();
        for result in selected {
            let entry = per_case.entry(result.case_id).or_insert((0, 0));
            entry.0 += usize::from(result.success);
            entry.1 += 1;
        }
        println!(
            "  {}",
            per_case
                .into_iter()
                .map(|(id, (success, total))| format!("{id}={success}/{total}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    println!("elapsed: {:.3}s", elapsed.as_secs_f64());
}

fn run_job(job: Job, scenarios: &[Scenario<'static>], simulations: usize) -> JobResult<'static> {
    let scenario = &scenarios[job.case_index];
    let config = MctsConfig {
        simulations,
        independent_trees: 1,
        maximum_tree_depth: 16,
        action_rank_batch_size: EXHAUSTIVE_ACTION_RANKING,
        root_widening_factor: job.policy.root,
        progressive_widening_factor: job.policy.internal,
        rollout_depth: 0,
        exploration: core::f32::consts::SQRT_2,
        heuristic_weights: HeuristicWeights::default(),
    };
    let mut agent = MctsAgent::new(job_seed(job.case_index, job.repetition), config).unwrap();
    let success = evaluate_agent(scenario, &mut agent);
    JobResult {
        policy: job.policy,
        case_id: scenario.id,
        kind: scenario.kind,
        success,
        telemetry: agent.telemetry(),
        worker: rayon::current_thread_index(),
    }
}

fn evaluate_agent(scenario: &Scenario<'_>, agent: &mut MctsAgent) -> bool {
    let mut position = scenario.position.clone();
    match scenario.kind {
        CaseKind::Attack => {
            let mut decisions = 0;
            while position.outcome() == GameOutcome::Ongoing
                && position.to_move() == scenario.attacker
                && decisions < scenario.horizon
            {
                apply_agent_decision(&mut position, agent);
                decisions += 1;
            }
            match position.outcome() {
                GameOutcome::Win(winner) => winner == scenario.attacker,
                GameOutcome::Draw => false,
                GameOutcome::Ongoing => {
                    decisions < scenario.horizon
                        && prove_forced_win(
                            &position,
                            scenario.attacker,
                            scenario.horizon - decisions,
                        )
                        .is_forced_win()
                }
            }
        }
        CaseKind::Defend => {
            let defender = scenario.attacker.opponent();
            while position.outcome() == GameOutcome::Ongoing && position.to_move() == defender {
                apply_agent_decision(&mut position, agent);
            }
            match position.outcome() {
                GameOutcome::Win(winner) => winner == defender,
                GameOutcome::Draw => true,
                GameOutcome::Ongoing => {
                    !prove_forced_win(&position, scenario.attacker, scenario.horizon)
                        .is_forced_win()
                }
            }
        }
    }
}

fn apply_agent_decision(position: &mut Position, agent: &mut MctsAgent) {
    let actions = legal_actions(position);
    let selected = agent
        .select_action(position, &actions)
        .expect("MCTS selection is infallible");
    position
        .apply(actions[selected])
        .expect("MCTS selected an engine-generated action");
}

struct TacticalProfile {
    legal_actions: usize,
    tactical_actions: usize,
    best_rank: usize,
    worst_tied_rank: usize,
}

fn tactical_profile(scenario: &Scenario<'_>) -> TacticalProfile {
    let player = scenario.position.to_move();
    let weights = HeuristicWeights::default();
    let mut actions: Vec<_> = legal_actions(&scenario.position)
        .into_par_iter()
        .map(|action| {
            let mut child = scenario.position.clone();
            child.apply(action).unwrap();
            let tactical = match scenario.kind {
                CaseKind::Attack => match child.outcome() {
                    GameOutcome::Win(winner) => winner == scenario.attacker,
                    GameOutcome::Draw => false,
                    GameOutcome::Ongoing => prove_forced_win(
                        &child,
                        scenario.attacker,
                        scenario.horizon.saturating_sub(1),
                    )
                    .is_forced_win(),
                },
                CaseKind::Defend => match child.outcome() {
                    GameOutcome::Win(winner) => winner != scenario.attacker,
                    GameOutcome::Draw => true,
                    GameOutcome::Ongoing => {
                        !prove_forced_win(&child, scenario.attacker, scenario.horizon)
                            .is_forced_win()
                    }
                },
            };
            let value = evaluate_position(&child, player, weights);
            (action, tactical, value)
        })
        .collect();
    actions.sort_by(|left, right| right.2.total_cmp(&left.2));
    let best_value = actions
        .iter()
        .filter(|entry| entry.1)
        .map(|entry| entry.2)
        .max_by(f32::total_cmp)
        .expect("validated tactical cases have at least one successful action");
    let strictly_better = actions.iter().filter(|entry| entry.2 > best_value).count();
    let tied = actions.iter().filter(|entry| entry.2 == best_value).count();
    TacticalProfile {
        legal_actions: actions.len(),
        tactical_actions: actions.iter().filter(|entry| entry.1).count(),
        best_rank: strictly_better + 1,
        worst_tied_rank: strictly_better + tied,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaseKind {
    Attack,
    Defend,
}

impl CaseKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Attack => "attack",
            Self::Defend => "defend",
        }
    }
}

struct Scenario<'a> {
    id: &'a str,
    kind: CaseKind,
    attacker: Player,
    horizon: usize,
    position: Position,
}

fn scenarios() -> Vec<Scenario<'static>> {
    let mut lines = CASES.lines();
    assert_eq!(
        lines.next(),
        Some("case_id\trecord\tprefix\tkind\tattacker\thorizon\tsource_game")
    );
    lines
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 7);
            let record: GameRecord = record_text(fields[1]).parse().unwrap();
            let prefix = fields[2].parse().unwrap();
            Scenario {
                id: fields[0],
                kind: match fields[3] {
                    "attack" => CaseKind::Attack,
                    "defend" => CaseKind::Defend,
                    other => panic!("unknown tactical case kind: {other}"),
                },
                attacker: match fields[4] {
                    "Host" => Player::Host,
                    "Guest" => Player::Guest,
                    other => panic!("unknown tactical attacker: {other}"),
                },
                horizon: fields[5].parse().unwrap(),
                position: replay_prefix(&record, prefix),
            }
        })
        .collect()
}

fn replay_prefix(record: &GameRecord, action_count: usize) -> Position {
    let mut position = record.initial_position();
    for action in &record.actions()[..action_count] {
        position.apply(*action).unwrap();
    }
    position
}

fn record_text(name: &str) -> &'static str {
    match name {
        "site_bot_v1_ring_finish.psr" => OFFICIAL_RING,
        "tactical_guest_wheel_finish.psr" => GUEST_WHEEL,
        "tactical_host_wheel_finish.psr" => HOST_WHEEL,
        "tactical_guest_forced_three.psr" => GUEST_THREE,
        "tactical_host_forced_three.psr" => HOST_THREE,
        other => panic!("unknown tactical record: {other}"),
    }
}

#[derive(Clone, Copy)]
struct Job {
    policy: WideningPolicy,
    case_index: usize,
    repetition: usize,
}

struct JobResult<'a> {
    policy: WideningPolicy,
    case_id: &'a str,
    kind: CaseKind,
    success: bool,
    telemetry: AgentTelemetry,
    worker: Option<usize>,
}

#[derive(Default)]
struct Totals {
    simulations: usize,
    evaluated_actions: usize,
    expanded_nodes: usize,
    maximum_depth: usize,
}

impl Totals {
    fn add(&mut self, telemetry: AgentTelemetry) {
        self.simulations += telemetry.simulations;
        self.evaluated_actions += telemetry.evaluated_actions;
        self.expanded_nodes += telemetry.expanded_nodes;
        self.maximum_depth = self.maximum_depth.max(telemetry.maximum_search_depth);
    }

    fn per_simulation(&self, value: usize) -> f64 {
        if self.simulations == 0 {
            0.0
        } else {
            value as f64 / self.simulations as f64
        }
    }
}

fn job_seed(case_index: usize, repetition: usize) -> u64 {
    0x5441_4354_4943_414c_u64
        ^ (case_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (repetition as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

fn positive_argument(
    arguments: &mut impl Iterator<Item = String>,
    name: &str,
    default: usize,
) -> usize {
    arguments
        .next()
        .map(|text| {
            let value = text
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be an integer"));
            assert!(value > 0, "{name} must be positive");
            value
        })
        .unwrap_or(default)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WideningPolicy {
    root: f32,
    internal: f32,
}

impl WideningPolicy {
    const fn new(root: f32, internal: f32) -> Self {
        Self { root, internal }
    }

    const fn uniform(factor: f32) -> Self {
        Self::new(factor, factor)
    }
}

fn parse_policies(text: &str) -> Vec<WideningPolicy> {
    let mut policies = Vec::new();
    for field in text.split(',') {
        let mut values = field.split('/');
        let root = parse_factor(values.next().expect("policy cannot be empty"));
        let internal = values.next().map(parse_factor).unwrap_or(root);
        assert!(values.next().is_none(), "a policy is root/internal");
        let policy = WideningPolicy::new(root, internal);
        if !policies.contains(&policy) {
            policies.push(policy);
        }
    }
    assert!(
        !policies.is_empty(),
        "at least one widening policy is required"
    );
    policies
}

fn parse_factor(text: &str) -> f32 {
    let factor = text
        .parse::<f32>()
        .expect("widening factors must be numbers");
    assert!(factor.is_finite() && factor > 0.0);
    factor
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_policy_arguments_are_measured_only_once() {
        assert_eq!(
            parse_policies("1,1/1,2/1"),
            vec![WideningPolicy::uniform(1.0), WideningPolicy::new(2.0, 1.0)]
        );
    }
}
