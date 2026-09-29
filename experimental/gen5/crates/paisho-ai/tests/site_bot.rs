use paisho_ai::{
    play_match, site_v1_action_score, site_v1_cycle_length, site_v1_cycle_progress_score,
    site_v1_main_actions, Agent, MatchConfig, MatchTask, MatchTermination, SiteBotV1,
    SITE_BOT_V1_SOURCE_COMMIT,
};
use paisho_core::{
    harmony_ring_owners, is_drained, legal_actions, Accent, AccentPlacement, Action, BasicFlower,
    Board, Coordinate, GameOutcome, GameRecord, Player, Position, StandardSetup, Tile, TileKind,
    TurnPhase,
};
use sha2::{Digest, Sha256};

const SCORE_FIXTURE: &str = include_str!("fixtures/site_bot_v1_scores.tsv");
const TRAJECTORY_FIXTURE: &str = include_str!("fixtures/site_bot_v1_trajectories.tsv");
const RESERVE_FINISH_FIXTURE: &str = include_str!("fixtures/site_bot_v1_reserve_finish.psr");
const RING_FINISH_FIXTURE: &str = include_str!("fixtures/site_bot_v1_ring_finish.psr");

fn reference_task() -> MatchTask {
    MatchTask {
        id: 41,
        setup: StandardSetup::balanced(BasicFlower::Red3),
    }
}

#[test]
fn site_bot_source_revision_is_pinned() {
    assert_eq!(
        SITE_BOT_V1_SOURCE_COMMIT,
        "b849dbdabb1138ff0f6d609adf38b301c2f875ae"
    );
}

#[test]
fn action_order_and_scores_match_the_javascript_oracle() {
    let opening = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    assert_scenario_matches("formal-opening", &opening);
    let developed = developed_position();
    assert_scenario_matches("developed-no-bonus", &developed);
    let developed_rock = developed_rock_position();
    assert_scenario_matches("developed-rock", &developed_rock);
    let developed_knotweed = developed_knotweed_position();
    assert_scenario_matches("developed-knotweed", &developed_knotweed);
    let developed_wheel = developed_wheel_position();
    assert_scenario_matches("developed-wheel", &developed_wheel);
    let developed_boat = developed_boat_position();
    assert_scenario_matches("developed-boat", &developed_boat);
    let capture_ready = capture_ready_position();
    assert_scenario_matches("capture-ready", &capture_ready);
    let ring_ready = ring_ready_position();
    assert_scenario_matches("ring-ready", &ring_ready);
}

#[test]
fn seeded_main_choices_match_the_javascript_oracle() {
    let opening = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    let developed = developed_position();
    let developed_rock = developed_rock_position();
    let developed_knotweed = developed_knotweed_position();
    let developed_wheel = developed_wheel_position();
    let developed_boat = developed_boat_position();
    let capture_ready = capture_ready_position();
    let ring_ready = ring_ready_position();
    for (scenario, seed, expected_main, expected_bonus) in fixture_selections() {
        let position = match scenario {
            "formal-opening" => &opening,
            "developed-no-bonus" => &developed,
            "developed-rock" => &developed_rock,
            "developed-knotweed" => &developed_knotweed,
            "developed-wheel" => &developed_wheel,
            "developed-boat" => &developed_boat,
            "capture-ready" => &capture_ready,
            "ring-ready" => &ring_ready,
            _ => panic!("unknown selection scenario {scenario}"),
        };
        let actions = legal_actions(position);
        let mut bot = SiteBotV1::new(seed);
        let selected = bot.select_action(position, &actions).unwrap();
        assert_eq!(
            actions[selected], expected_main,
            "scenario {scenario}, seed {seed:#x}"
        );
        let mut after = position.clone();
        after.apply(expected_main).unwrap();
        match (after.phase(), expected_bonus) {
            (TurnPhase::Main, None) => {}
            (TurnPhase::HarmonyBonus, Some(expected)) => {
                let bonuses = legal_actions(&after);
                let selected = bot.select_action(&after, &bonuses).unwrap();
                assert_eq!(
                    bonuses[selected], expected,
                    "bonus for scenario {scenario}, seed {seed:#x}"
                );
            }
            (TurnPhase::HarmonyBonus, None) => {
                panic!("fixture omitted bonus for scenario {scenario}, seed {seed:#x}")
            }
            (TurnPhase::Main, Some(expected)) => panic!(
                "fixture expected `{expected}` outside a bonus phase for scenario {scenario}, seed {seed:#x}"
            ),
        }
    }
}

#[test]
fn site_bot_self_play_is_seeded_legal_and_replayable() {
    let run = || {
        play_match(
            reference_task(),
            MatchConfig {
                decision_soft_limit: 64,
            },
            &mut SiteBotV1::new(0x484f_5354),
            &mut SiteBotV1::new(0x0047_5545_5354),
        )
        .unwrap()
    };

    let first = run();
    let second = run();
    assert_eq!(first, second);
    assert_eq!(first.record.replay().unwrap(), first.final_position);
    assert_eq!(
        first.host_telemetry.decisions + first.guest_telemetry.decisions,
        first.record.actions().len()
    );
    assert!(first.host_telemetry.evaluated_actions > 0);
    assert!(first.guest_telemetry.evaluated_actions > 0);
}

#[test]
fn seeded_full_trajectories_match_the_javascript_oracle() {
    let trajectories = fixture_trajectories();
    assert_eq!(trajectories.len(), 2);
    for trajectory in trajectories {
        let result = play_match(
            MatchTask {
                id: 0,
                setup: StandardSetup::balanced(trajectory.starting_flower),
            },
            MatchConfig {
                decision_soft_limit: trajectory.decision_soft_limit,
            },
            &mut SiteBotV1::new(trajectory.host_seed),
            &mut SiteBotV1::new(trajectory.guest_seed),
        )
        .unwrap();
        let actual = result.record.actions();
        if actual != trajectory.actions {
            let mismatch = actual
                .iter()
                .zip(&trajectory.actions)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or_else(|| actual.len().min(trajectory.actions.len()));
            panic!(
                "decision {} mismatch in {}: actual {:?}, expected {:?}; lengths {}/{}",
                mismatch + 1,
                trajectory.name,
                actual.get(mismatch),
                trajectory.actions.get(mismatch),
                actual.len(),
                trajectory.actions.len()
            );
        }
        assert_eq!(
            result.termination, trajectory.termination,
            "termination mismatch in {}",
            trajectory.name
        );
        assert_eq!(result.record.replay().unwrap(), result.final_position);
    }
}

#[test]
fn site_bot_reproduces_the_sources_empty_move_marker_omission() {
    let trajectory = fixture_trajectories()
        .into_iter()
        .find(|trajectory| trajectory.name == "white4-independent")
        .unwrap();
    let mut position =
        Position::from_standard_setup(StandardSetup::balanced(trajectory.starting_flower));
    for action in &trajectory.actions[..60] {
        position.apply(*action).unwrap();
    }

    let omitted: Action = "arrange -5,4 0,4".parse().unwrap();
    assert!(legal_actions(&position).contains(&omitted));
    assert!(!site_v1_main_actions(&position).contains(&omitted));
}

#[test]
fn full_source_trajectories_replay_with_identical_candidate_order() {
    for trajectory in fixture_trajectories() {
        let mut position =
            Position::from_standard_setup(StandardSetup::balanced(trajectory.starting_flower));
        let mut decision_offset = 0;

        for turn in &trajectory.turns {
            assert_eq!(
                decision_offset, turn.decision_offset,
                "decision offset before move {} in {}",
                turn.move_number, trajectory.name
            );
            assert_eq!(position.phase(), TurnPhase::Main);
            assert_eq!(
                position.to_move(),
                turn.player,
                "player before move {} in {}",
                turn.move_number,
                trajectory.name
            );

            let candidates = site_v1_main_actions(&position);
            let occupied = position
                .board()
                .occupied()
                .map(|(coordinate, tile)| {
                    format!("{coordinate}={}:{}", tile.owner.code(), tile.kind)
                })
                .collect::<Vec<_>>()
                .join(" ");
            assert_eq!(
                candidates.len(),
                turn.action_count,
                "candidate count at decision {} in {}; board: {}; Rust candidates:\n{}",
                decision_offset + 1,
                trajectory.name,
                occupied,
                candidates
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            );
            assert_eq!(
                action_digest(&candidates),
                turn.action_digest,
                "candidate order at decision {} in {}",
                decision_offset + 1,
                trajectory.name
            );
            let main = trajectory.actions[decision_offset];
            assert_eq!(
                candidates.get(turn.selected_index),
                Some(&main),
                "selected candidate at decision {} in {}",
                decision_offset + 1,
                trajectory.name
            );
            position.apply(main).unwrap_or_else(|error| {
                panic!(
                    "source main action `{main}` failed at decision {} in {}: {error}",
                    decision_offset + 1,
                    trajectory.name
                )
            });
            decision_offset += 1;

            if position.phase() == TurnPhase::HarmonyBonus {
                let bonus = trajectory.actions[decision_offset];
                position.apply(bonus).unwrap_or_else(|error| {
                    panic!(
                        "source bonus action `{bonus}` failed at decision {} in {}: {error}",
                        decision_offset + 1,
                        trajectory.name
                    )
                });
                decision_offset += 1;
            }
        }

        assert_eq!(decision_offset, trajectory.actions.len());
        assert_eq!(
            MatchTermination::Rules(position.outcome()),
            trajectory.termination,
            "replayed termination in {}",
            trajectory.name
        );
    }
}

#[test]
fn reserve_exhaustion_is_not_mistaken_for_the_site_bots_ring_win_signal() {
    let source_contract = fixture_reserve_exhaustion_contract();
    assert_eq!(source_contract, (7, 0, 2));

    let record: GameRecord = RESERVE_FINISH_FIXTURE.parse().unwrap();
    let (last, prefix) = record.actions().split_last().unwrap();
    let mut before = record.initial_position();
    for action in prefix {
        before.apply(*action).unwrap();
    }
    assert_eq!(before.outcome(), GameOutcome::Ongoing);
    let player = before.to_move();
    let mut after = before.clone();
    after.apply(*last).unwrap();
    assert_ne!(after.outcome(), GameOutcome::Ongoing);
    assert!(!harmony_ring_owners(after.board()).contains(&player));
    assert_ne!(
        site_v1_action_score(&before, *last).unwrap(),
        paisho_ai::SITE_BOT_V1_WIN_SCORE
    );
}

#[test]
fn source_terminal_basic_bonus_remains_a_separate_recorded_decision() {
    let (decisions, bonus_allowed, remaining_basics, board_winners, end_game_winners) =
        fixture_terminal_bonus_contract();
    assert_eq!(
        decisions,
        vec![
            "arrange -1,-4 -1,-5".parse::<Action>().unwrap(),
            "bonus-plant W5 0,-8".parse::<Action>().unwrap(),
        ]
    );
    assert!(bonus_allowed);
    assert_eq!(
        (remaining_basics, board_winners, end_game_winners),
        (0, 0, 2)
    );
}

#[test]
fn ring_closing_move_matches_the_site_bots_terminal_signal() {
    let record: GameRecord = RING_FINISH_FIXTURE.parse().unwrap();
    let (last, _) = record.actions().split_last().unwrap();
    let mut before = ring_ready_position();
    assert_eq!(before.outcome(), GameOutcome::Ongoing);
    assert_eq!(
        site_v1_action_score(&before, *last).unwrap(),
        paisho_ai::SITE_BOT_V1_WIN_SCORE
    );
    before.apply(*last).unwrap();
    assert_eq!(
        before.outcome(),
        GameOutcome::Win(paisho_core::Player::Guest)
    );
    assert_eq!(
        harmony_ring_owners(before.board()),
        vec![paisho_core::Player::Guest]
    );
}

#[test]
fn mixed_owner_lotus_cycle_matches_the_site_bots_quirk() {
    let source = fixture_mixed_cycle_contract();
    assert_eq!(source, (4, 5, 4, 4, 5, 0));

    let before = mixed_cycle_board(false);
    let after = mixed_cycle_board(true);
    assert_eq!(site_v1_cycle_length(&before, Player::Host), source.0);
    assert_eq!(site_v1_cycle_length(&after, Player::Host), source.1);
    assert_eq!(
        site_v1_cycle_progress_score(&before, &after, Player::Host),
        source.4
    );
    assert!(harmony_ring_owners(&after).is_empty());
}

#[test]
fn tactical_capture_probe_uses_a_legal_replayable_position() {
    let mut position = capture_ready_position();
    let capture = arrange(-5, 3, -4, 4);
    assert!(legal_actions(&position).contains(&capture));
    assert!(position.apply(capture).unwrap().captured.is_some());
}

#[test]
fn tactical_accent_positions_apply_the_intended_effects() {
    let knotweed = developed_knotweed_position();
    assert_eq!(
        knotweed.board().get(coordinate(-1, -4)).unwrap().kind,
        TileKind::Accent(Accent::Knotweed)
    );
    assert!(is_drained(knotweed.board(), coordinate(-1, -5)));
    assert!(is_drained(knotweed.board(), coordinate(0, -5)));

    let wheel = developed_wheel_position();
    assert_eq!(
        wheel.board().get(coordinate(-1, -4)).unwrap().kind,
        TileKind::Accent(Accent::Wheel)
    );
    assert_eq!(
        wheel.board().get(coordinate(-1, -5)).unwrap().kind,
        TileKind::Basic(BasicFlower::Red3)
    );
    assert_eq!(
        wheel.board().get(coordinate(-2, -5)).unwrap().kind,
        TileKind::Basic(BasicFlower::Red4)
    );

    let boat = developed_boat_position();
    assert_eq!(
        boat.board().get(coordinate(0, -5)).unwrap().kind,
        TileKind::Accent(Accent::Boat)
    );
    assert_eq!(
        boat.board().get(coordinate(0, -4)).unwrap().kind,
        TileKind::Basic(BasicFlower::Red3)
    );
}

fn fixture_reserve_exhaustion_contract() -> (i32, usize, usize) {
    let line = SCORE_FIXTURE
        .lines()
        .skip_while(|line| *line != "[reserve-exhaustion-contract]")
        .nth(1)
        .expect("fixture has a reserve-exhaustion contract");
    let mut fields = line.split('\t');
    let _action = fields.next().expect("contract has an action");
    let score = fields
        .next()
        .expect("contract has a score")
        .parse()
        .expect("contract score is an integer");
    let board_winners = fields
        .next()
        .expect("contract has a board winner count")
        .parse()
        .expect("board winner count is an integer");
    let end_game_winners = fields
        .next()
        .expect("contract has an end-game winner count")
        .parse()
        .expect("end-game winner count is an integer");
    assert!(fields.next().is_none());
    (score, board_winners, end_game_winners)
}

fn fixture_terminal_bonus_contract() -> (Vec<Action>, bool, usize, usize, usize) {
    let line = SCORE_FIXTURE
        .lines()
        .skip_while(|line| *line != "[terminal-bonus-contract]")
        .nth(1)
        .expect("fixture has a terminal-bonus contract");
    let fields: Vec<_> = line.split('\t').collect();
    assert_eq!(fields.len(), 6);
    let decisions = fields[..2]
        .iter()
        .map(|field| field.parse().expect("terminal-bonus action notation"))
        .collect();
    (
        decisions,
        fields[2] == "1",
        fields[3].parse().expect("remaining Basic count"),
        fields[4].parse().expect("board winner count"),
        fields[5].parse().expect("end-game winner count"),
    )
}

fn fixture_mixed_cycle_contract() -> (usize, usize, usize, usize, i32, usize) {
    let line = SCORE_FIXTURE
        .lines()
        .skip_while(|line| *line != "[mixed-cycle-contract]")
        .nth(1)
        .expect("fixture has a mixed-cycle contract");
    let mut fields = line.split('\t');
    let _action = fields.next().expect("contract has an action");
    let values: Vec<i32> = fields
        .map(|field| field.parse().expect("contract fields are integers"))
        .collect();
    assert_eq!(values.len(), 6);
    (
        values[0] as usize,
        values[1] as usize,
        values[2] as usize,
        values[3] as usize,
        values[4],
        values[5] as usize,
    )
}

struct FixtureTrajectory {
    name: &'static str,
    starting_flower: BasicFlower,
    host_seed: u64,
    guest_seed: u64,
    decision_soft_limit: usize,
    termination: MatchTermination,
    turns: Vec<FixtureTurn>,
    actions: Vec<Action>,
}

struct FixtureTurn {
    decision_offset: usize,
    player: Player,
    move_number: usize,
    action_count: usize,
    action_digest: &'static str,
    selected_index: usize,
}

fn fixture_trajectories() -> Vec<FixtureTrajectory> {
    let mut lines = TRAJECTORY_FIXTURE.lines().peekable();
    assert_eq!(lines.next(), Some("PAISHO-SITE-TRAJECTORIES\t1"));
    assert_eq!(
        lines.next(),
        Some(concat!(
            "# source-commit ",
            "b849dbdabb1138ff0f6d609adf38b301c2f875ae"
        ))
    );
    let mut trajectories = Vec::new();
    while let Some(heading) = lines.next() {
        let name = heading
            .strip_prefix("[trajectory ")
            .and_then(|text| text.strip_suffix(']'))
            .expect("trajectory heading");
        let starting_flower = match fixture_value(lines.next(), "starting_flower")
            .parse::<TileKind>()
            .expect("starting flower code")
        {
            TileKind::Basic(flower) => flower,
            _ => panic!("trajectory starting tile is not a Basic Flower"),
        };
        let host_seed = parse_hex_seed(fixture_value(lines.next(), "host_seed"));
        let guest_seed = parse_hex_seed(fixture_value(lines.next(), "guest_seed"));
        let decision_soft_limit = fixture_value(lines.next(), "decision_soft_limit")
            .parse()
            .expect("trajectory decision limit");
        let termination = match fixture_value(lines.next(), "termination") {
            "RING_HOST" | "RESERVE_HOST" => MatchTermination::Rules(GameOutcome::Win(Player::Host)),
            "RING_GUEST" | "RESERVE_GUEST" => {
                MatchTermination::Rules(GameOutcome::Win(Player::Guest))
            }
            "RING_DRAW" | "RESERVE_DRAW" => MatchTermination::Rules(GameOutcome::Draw),
            "DECISION_LIMIT" => MatchTermination::DecisionLimit,
            other => panic!("unknown trajectory termination {other}"),
        };
        let mut turns = Vec::new();
        while let Some(line) = lines.next_if(|line| line.starts_with("turn\t")) {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 7, "malformed trajectory turn `{line}`");
            let player = match fields[2] {
                "H" => Player::Host,
                "G" => Player::Guest,
                other => panic!("unknown trajectory player {other}"),
            };
            turns.push(FixtureTurn {
                decision_offset: fields[1].parse().expect("turn decision offset"),
                player,
                move_number: fields[3].parse().expect("turn move number"),
                action_count: fields[4].parse().expect("turn action count"),
                action_digest: fields[5],
                selected_index: fields[6].parse().expect("turn selected index"),
            });
        }
        let mut actions = Vec::new();
        while let Some(line) = lines.next_if(|line| line.starts_with("action\t")) {
            actions.push(
                fixture_value(Some(line), "action")
                    .parse()
                    .expect("trajectory action notation"),
            );
        }
        assert!(!turns.is_empty(), "trajectory {name} has no turn metadata");
        assert!(!actions.is_empty(), "trajectory {name} has no actions");
        trajectories.push(FixtureTrajectory {
            name,
            starting_flower,
            host_seed,
            guest_seed,
            decision_soft_limit,
            termination,
            turns,
            actions,
        });
    }
    trajectories
}

fn action_digest(actions: &[Action]) -> String {
    let mut digest = Sha256::new();
    for (index, action) in actions.iter().enumerate() {
        if index > 0 {
            digest.update(b"\n");
        }
        digest.update(action.to_string().as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn fixture_value<'a>(line: Option<&'a str>, key: &str) -> &'a str {
    line.and_then(|line| line.strip_prefix(key))
        .and_then(|line| line.strip_prefix('\t'))
        .unwrap_or_else(|| panic!("trajectory fixture requires {key}"))
}

fn parse_hex_seed(text: &str) -> u64 {
    u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("trajectory seed is hexadecimal")
}

fn assert_scenario_matches(name: &str, position: &Position) {
    let expected = fixture_scenario(name);
    let actual: Vec<_> = site_v1_main_actions(position)
        .into_iter()
        .map(|action| {
            let score = site_v1_action_score(position, action).unwrap();
            (action, score)
        })
        .collect();
    assert_eq!(actual, expected, "differential scenario {name}");
}

fn fixture_scenario(name: &str) -> Vec<(Action, i32)> {
    let heading = format!("[{name}]");
    let mut active = false;
    let mut entries = Vec::new();
    for line in SCORE_FIXTURE.lines() {
        if line.starts_with('[') {
            active = line == heading;
            continue;
        }
        if !active || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (action, score) = line.split_once('\t').expect("fixture line uses a tab");
        entries.push((
            action.parse().expect("fixture action uses stable notation"),
            score.parse().expect("fixture score is an integer"),
        ));
    }
    assert!(!entries.is_empty(), "unknown fixture scenario {name}");
    entries
}

fn fixture_selections() -> Vec<(&'static str, u64, Action, Option<Action>)> {
    let mut active = false;
    let mut entries = Vec::new();
    for line in SCORE_FIXTURE.lines() {
        if line.starts_with('[') {
            active = line == "[seeded-selections]";
            continue;
        }
        if !active || line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split('\t');
        let scenario = fields.next().expect("selection fixture has a scenario");
        let seed_text = fields.next().expect("selection fixture has a seed");
        let action_text = fields.next().expect("selection fixture has an action");
        let bonus = fields.next().map(|text| {
            text.parse()
                .expect("selection bonus uses stable action notation")
        });
        assert!(
            fields.next().is_none(),
            "selection fixture has at most four fields"
        );
        entries.push((
            scenario,
            u64::from_str_radix(seed_text.trim_start_matches("0x"), 16)
                .expect("selection seed is hexadecimal"),
            action_text
                .parse()
                .expect("selection action uses stable notation"),
            bonus,
        ));
    }
    entries
}

fn developed_position() -> Position {
    let mut position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    for action in [
        arrange(0, -8, 0, -5),
        arrange(0, 8, 0, 5),
        Action::Plant {
            flower: BasicFlower::Red4,
            gate: coordinate(0, -8),
        },
        Action::Plant {
            flower: BasicFlower::Red4,
            gate: coordinate(0, 8),
        },
        arrange(0, -8, -1, -5),
        arrange(0, 8, 1, 5),
        arrange(-1, -5, -1, -4),
        arrange(1, 5, 1, 4),
    ] {
        apply_turn_without_bonus(&mut position, action);
    }
    position
}

fn developed_rock_position() -> Position {
    developed_bonus_position(Action::PlayAccent {
        accent: Accent::Rock,
        placement: AccentPlacement::At(coordinate(0, 0)),
    })
}

fn developed_knotweed_position() -> Position {
    developed_bonus_position(Action::PlayAccent {
        accent: Accent::Knotweed,
        placement: AccentPlacement::At(coordinate(-1, -4)),
    })
}

fn developed_wheel_position() -> Position {
    developed_bonus_position(Action::PlayAccent {
        accent: Accent::Wheel,
        placement: AccentPlacement::At(coordinate(-1, -4)),
    })
}

fn developed_boat_position() -> Position {
    developed_bonus_position(Action::PlayAccent {
        accent: Accent::Boat,
        placement: AccentPlacement::BoatMove {
            flower: coordinate(0, -5),
            destination: coordinate(0, -4),
        },
    })
}

fn developed_bonus_position(bonus: Action) -> Position {
    let mut position = developed_position();
    position.apply(arrange(-1, -4, -1, -5)).unwrap();
    assert_eq!(position.phase(), TurnPhase::HarmonyBonus);
    assert!(
        legal_actions(&position).contains(&bonus),
        "missing `{bonus}`"
    );
    position.apply(bonus).unwrap();
    position
}

fn capture_ready_position() -> Position {
    let mut position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
    for action in [
        Action::Plant {
            flower: BasicFlower::White3,
            gate: coordinate(-8, 0),
        },
        arrange(0, 8, -2, 7),
        arrange(-8, 0, -6, 1),
        arrange(-2, 7, -4, 6),
        arrange(-6, 1, -5, 3),
        arrange(-4, 6, -4, 4),
    ] {
        apply_turn_without_bonus(&mut position, action);
    }
    position
}

fn ring_ready_position() -> Position {
    let record: GameRecord = RING_FINISH_FIXTURE.parse().unwrap();
    let (_, prefix) = record.actions().split_last().unwrap();
    let mut position = record.initial_position();
    for action in prefix {
        position.apply(*action).unwrap();
    }
    position
}

fn mixed_cycle_board(moved: bool) -> Board {
    let mut board = Board::empty();
    for (owner, kind, x, y) in [
        (Player::Host, TileKind::WhiteLotus, -4, -2),
        (Player::Host, TileKind::Basic(BasicFlower::Red3), -4, 2),
        (
            Player::Host,
            TileKind::Basic(BasicFlower::Red4),
            0,
            if moved { 2 } else { 3 },
        ),
        (Player::Host, TileKind::Orchid, 2, -3),
        (Player::Guest, TileKind::WhiteLotus, 4, 2),
        (Player::Guest, TileKind::Basic(BasicFlower::Red3), 4, -2),
        (Player::Guest, TileKind::Basic(BasicFlower::Red4), 0, -2),
    ] {
        board
            .place(coordinate(x, y), Tile::new(owner, kind))
            .unwrap();
    }
    board
}

fn apply_turn_without_bonus(position: &mut Position, action: Action) {
    position
        .apply(action)
        .unwrap_or_else(|error| panic!("failed to apply `{action}`: {error}"));
    if position.phase() == TurnPhase::HarmonyBonus {
        position.apply(Action::SkipHarmonyBonus).unwrap();
    }
}

fn arrange(from_x: i8, from_y: i8, to_x: i8, to_y: i8) -> Action {
    Action::Arrange {
        from: coordinate(from_x, from_y),
        to: coordinate(to_x, to_y),
    }
}

fn coordinate(x: i8, y: i8) -> Coordinate {
    Coordinate::new(x, y).unwrap()
}
