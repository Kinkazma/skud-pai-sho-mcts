use paisho_ai::{
    play_match, play_match_from_record, play_parallel, Agent, AgentError, MatchConfig, MatchError,
    MatchTask, MatchTermination, RandomAgent, StableRng,
};
use paisho_core::{
    legal_actions, Action, BasicFlower, Coordinate, GameOutcome, GameRecord, Position,
    StandardSetup, TurnPhase,
};

#[test]
fn stable_rng_has_a_pinned_sequence() {
    let mut rng = StableRng::new(0);
    assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
    assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
}

#[test]
fn a_random_match_is_reproducible_and_its_record_replays() {
    let task = MatchTask {
        id: 7,
        setup: StandardSetup::balanced(BasicFlower::Red3),
    };
    let config = MatchConfig {
        decision_soft_limit: 64,
    };
    let run = || {
        play_match(
            task,
            config,
            &mut RandomAgent::new(11),
            &mut RandomAgent::new(29),
        )
        .unwrap()
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
    assert_eq!(first.record.replay().unwrap(), first.final_position);
    assert!(matches!(
        first.termination,
        MatchTermination::Rules(_) | MatchTermination::DecisionLimit
    ));
}

#[test]
fn a_parallel_batch_uses_the_runtime_pool_and_preserves_task_order() {
    let tasks: Vec<_> = (0_u64..8)
        .map(|id| MatchTask {
            id,
            setup: StandardSetup::balanced(BasicFlower::White4),
        })
        .collect();
    let batch = play_parallel(
        &tasks,
        MatchConfig {
            decision_soft_limit: 24,
        },
        |task| RandomAgent::new(task.id ^ 0x1111),
        |task| RandomAgent::new(task.id ^ 0x2222),
    );

    assert!(batch.workers >= 1);
    assert!(batch.workers <= batch.worker_capacity);
    assert_eq!(batch.matches.len(), tasks.len());
    for (expected_id, result) in (0_u64..).zip(batch.matches) {
        let result = result.unwrap();
        assert_eq!(result.task_id, expected_id);
        assert_eq!(result.record.replay().unwrap(), result.final_position);
    }
}

struct BrokenAgent;

impl Agent for BrokenAgent {
    fn select_action(
        &mut self,
        _position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        Ok(legal_actions.len())
    }

    fn reset_telemetry(&mut self) {}
}

struct FailingAgent;

impl Agent for FailingAgent {
    fn select_action(
        &mut self,
        _position: &Position,
        _legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        Err(AgentError::new("backend disappeared"))
    }

    fn reset_telemetry(&mut self) {}
}

#[test]
fn the_harness_rejects_an_out_of_range_agent_choice() {
    let error = play_match(
        MatchTask {
            id: 0,
            setup: StandardSetup::balanced(BasicFlower::Red3),
        },
        MatchConfig {
            decision_soft_limit: 1,
        },
        &mut RandomAgent::new(0),
        &mut BrokenAgent,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        MatchError::AgentChoiceOutOfRange {
            player: paisho_core::Player::Guest,
            ..
        }
    ));
}

#[test]
fn the_harness_reports_agent_failures_without_inventing_a_move() {
    let error = play_match(
        MatchTask {
            id: 1,
            setup: StandardSetup::balanced(BasicFlower::Red3),
        },
        MatchConfig {
            decision_soft_limit: 1,
        },
        &mut RandomAgent::new(0),
        &mut FailingAgent,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        MatchError::AgentFailure {
            player: paisho_core::Player::Guest,
            ref source,
        } if source.message() == "backend disappeared"
    ));
}

#[test]
fn a_zero_decision_limit_is_unrated_and_does_not_mutate_the_game() {
    let result = play_match(
        MatchTask {
            id: 9,
            setup: StandardSetup::balanced(BasicFlower::White5),
        },
        MatchConfig {
            decision_soft_limit: 0,
        },
        &mut RandomAgent::new(0),
        &mut RandomAgent::new(1),
    )
    .unwrap();

    assert_eq!(result.termination, MatchTermination::DecisionLimit);
    assert_eq!(result.scored_outcome(), None);
    assert!(result.record.actions().is_empty());
    assert_eq!(result.final_position.completed_turns(), 0);
}

#[test]
fn a_prefixed_match_preserves_the_prefix_and_limits_only_new_decisions() {
    let setup = StandardSetup::balanced(BasicFlower::White3);
    let position = Position::from_standard_setup(setup);
    let action = legal_actions(&position)
        .into_iter()
        .find(|action| {
            let mut next = position.clone();
            next.apply(*action).is_ok()
                && next.outcome() == GameOutcome::Ongoing
                && next.phase() == TurnPhase::Main
        })
        .expect("the opening has a main-phase continuation");
    let mut prefix = GameRecord::new(setup);
    prefix.push(action);

    let result = play_match_from_record(
        10,
        prefix.clone(),
        MatchConfig {
            decision_soft_limit: 0,
        },
        &mut RandomAgent::new(0),
        &mut RandomAgent::new(1),
    )
    .unwrap();

    assert_eq!(result.termination, MatchTermination::DecisionLimit);
    assert_eq!(result.record, prefix);
    assert_eq!(result.final_position, result.record.replay().unwrap());
    assert_eq!(result.host_telemetry.decisions, 0);
    assert_eq!(result.guest_telemetry.decisions, 0);
}

#[test]
fn reused_agents_report_only_the_current_match() {
    let task = MatchTask {
        id: 17,
        setup: StandardSetup::balanced(BasicFlower::Red5),
    };
    let config = MatchConfig {
        decision_soft_limit: 8,
    };
    let mut host = RandomAgent::new(101);
    let mut guest = RandomAgent::new(202);

    let first = play_match(task, config, &mut host, &mut guest).unwrap();
    let second = play_match(task, config, &mut host, &mut guest).unwrap();
    for result in [first, second] {
        assert_eq!(
            result.host_telemetry.decisions + result.guest_telemetry.decisions,
            result.record.actions().len()
        );
    }
}

struct ScriptedAgent {
    main_actions: Vec<Action>,
    next_main: usize,
}

impl ScriptedAgent {
    fn new(main_actions: Vec<Action>) -> Self {
        Self {
            main_actions,
            next_main: 0,
        }
    }
}

impl Agent for ScriptedAgent {
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        let expected = if position.phase() == TurnPhase::HarmonyBonus {
            Action::SkipHarmonyBonus
        } else {
            let action = self.main_actions[self.next_main];
            self.next_main += 1;
            action
        };
        Ok(legal_actions
            .iter()
            .position(|action| *action == expected)
            .unwrap_or_else(|| panic!("scripted action `{expected}` is not legal")))
    }

    fn reset_telemetry(&mut self) {}
}

#[test]
fn decision_limit_finishes_an_in_flight_harmony_bonus() {
    let mut guest = ScriptedAgent::new(vec![
        arrange(0, -8, 0, -5),
        Action::Plant {
            flower: BasicFlower::Red4,
            gate: at(0, -8),
        },
        arrange(0, -8, -1, -5),
        arrange(-1, -5, -1, -4),
        arrange(-1, -4, -1, -5),
    ]);
    let mut host = ScriptedAgent::new(vec![
        arrange(0, 8, 0, 5),
        Action::Plant {
            flower: BasicFlower::Red4,
            gate: at(0, 8),
        },
        arrange(0, 8, 1, 5),
        arrange(1, 5, 1, 4),
    ]);
    let result = play_match(
        MatchTask {
            id: 23,
            setup: StandardSetup::balanced(BasicFlower::Red3),
        },
        MatchConfig {
            decision_soft_limit: 11,
        },
        &mut host,
        &mut guest,
    )
    .unwrap();

    assert_eq!(result.termination, MatchTermination::DecisionLimit);
    assert_eq!(result.record.actions().len(), 12);
    assert_eq!(
        result.record.actions().last(),
        Some(&Action::SkipHarmonyBonus)
    );
    assert_eq!(result.final_position.phase(), TurnPhase::Main);
    assert_eq!(result.final_position.completed_turns(), 9);
}

fn arrange(from_x: i8, from_y: i8, to_x: i8, to_y: i8) -> Action {
    Action::Arrange {
        from: at(from_x, from_y),
        to: at(to_x, to_y),
    }
}

fn at(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}
