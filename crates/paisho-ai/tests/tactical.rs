use paisho_ai::{prove_forced_win, TacticalVerdict};
use paisho_core::{harmony_ring_owners, GameOutcome, GameRecord, Player, Position};

const CASES: &str = include_str!("fixtures/tactical_cases.tsv");
const OFFICIAL_RING: &str = include_str!("fixtures/site_bot_v1_ring_finish.psr");
const GUEST_WHEEL: &str = include_str!("fixtures/tactical_guest_wheel_finish.psr");
const HOST_WHEEL: &str = include_str!("fixtures/tactical_host_wheel_finish.psr");
const GUEST_THREE: &str = include_str!("fixtures/tactical_guest_forced_three.psr");
const HOST_THREE: &str = include_str!("fixtures/tactical_host_forced_three.psr");

#[test]
fn corpus_positions_have_the_exact_bounded_tactical_contract() {
    let cases = cases();
    assert_eq!(cases.len(), 8);
    assert!(cases.iter().any(|case| case.kind == CaseKind::Attack));
    assert!(cases.iter().any(|case| case.kind == CaseKind::Defend));
    assert!(cases.iter().any(|case| case.attacker == Player::Host));
    assert!(cases.iter().any(|case| case.attacker == Player::Guest));
    assert!(cases.iter().any(|case| case.horizon == 1));
    assert!(cases.iter().any(|case| case.horizon == 2));
    assert!(cases.iter().any(|case| case.horizon == 3));

    for case in cases {
        let record: GameRecord = record_text(case.record).parse().unwrap();
        let final_position = record.replay().unwrap();
        assert_eq!(
            final_position.outcome(),
            GameOutcome::Win(case.attacker),
            "{} source game has the wrong winner",
            case.id
        );
        assert!(
            harmony_ring_owners(final_position.board()).contains(&case.attacker),
            "{} source game does not finish with the expected Ring",
            case.id
        );

        let position = replay_prefix(&record, case.prefix);
        assert_eq!(position.outcome(), GameOutcome::Ongoing, "{}", case.id);
        match case.kind {
            CaseKind::Attack => assert_attack_contract(case, &position),
            CaseKind::Defend => assert_defense_contract(case, &record, &position),
        }
    }
}

fn assert_attack_contract(case: Case<'_>, position: &Position) {
    assert_eq!(position.to_move(), case.attacker, "{}", case.id);
    assert_eq!(
        prove_forced_win(position, case.attacker, case.horizon).verdict,
        TacticalVerdict::ForcedWin,
        "{} must be a proven win",
        case.id
    );
    assert_eq!(
        prove_forced_win(position, case.attacker, case.horizon - 1).verdict,
        TacticalVerdict::NotProven,
        "{} horizon must be minimal",
        case.id
    );
}

fn assert_defense_contract(case: Case<'_>, record: &GameRecord, position: &Position) {
    assert_eq!(position.to_move(), case.attacker.opponent(), "{}", case.id);
    assert_eq!(
        prove_forced_win(position, case.attacker, case.horizon + 1).verdict,
        TacticalVerdict::NotProven,
        "{} must still admit a defense",
        case.id
    );

    let mut after_recorded_blunder = position.clone();
    after_recorded_blunder
        .apply(record.actions()[case.prefix])
        .unwrap();
    assert_eq!(
        prove_forced_win(&after_recorded_blunder, case.attacker, case.horizon).verdict,
        TacticalVerdict::ForcedWin,
        "{} recorded move must activate the threat",
        case.id
    );
}

fn replay_prefix(record: &GameRecord, action_count: usize) -> Position {
    let mut position = record.initial_position();
    for action in &record.actions()[..action_count] {
        position.apply(*action).unwrap();
    }
    position
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaseKind {
    Attack,
    Defend,
}

#[derive(Clone, Copy)]
struct Case<'a> {
    id: &'a str,
    record: &'a str,
    prefix: usize,
    kind: CaseKind,
    attacker: Player,
    horizon: usize,
}

fn cases() -> Vec<Case<'static>> {
    let mut lines = CASES.lines();
    assert_eq!(
        lines.next(),
        Some("case_id\trecord\tprefix\tkind\tattacker\thorizon\tsource_game")
    );
    lines
        .map(|line| {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 7);
            Case {
                id: fields[0],
                record: fields[1],
                prefix: fields[2].parse().unwrap(),
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
            }
        })
        .collect()
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
