use paisho_core::{
    harmony_ring_owners_for_profile, legal_actions, BasicFlower, GameOutcome, GameRecord, Player,
    Position, RuleProfileId, StandardSetup,
};

const REPORTED: &str = include_str!("fixtures/reported-ring-v1.psr");

#[test]
fn new_games_use_corrected_rules_and_old_records_keep_their_identity() {
    let setup = StandardSetup::balanced(BasicFlower::Red5);
    assert_eq!(
        Position::from_standard_setup(setup).rule_profile(),
        RuleProfileId::CURRENT
    );
    assert_eq!(GameRecord::new(setup).rules(), RuleProfileId::CURRENT);
    assert_eq!(
        RuleProfileId::CURRENT.as_str(),
        "skud-pai-sho-2022-03-14-v2"
    );
    let old: GameRecord = REPORTED.parse().unwrap();
    assert_eq!(old.to_string(), REPORTED);
    assert_eq!(
        old.initial_position().rule_profile(),
        RuleProfileId::SkudPaiSho2022
    );
    assert_eq!(
        old.replay().unwrap().outcome(),
        GameOutcome::Win(Player::Host)
    );
}

#[test]
fn reported_boat_move_wins_immediately_under_corrected_rules() {
    let old: GameRecord = REPORTED.parse().unwrap();
    let mut position = Position::from_standard_setup(old.setup());
    for (index, action) in old.actions()[..74].iter().enumerate() {
        assert_eq!(
            position.outcome(),
            GameOutcome::Ongoing,
            "decision {}",
            index + 1
        );
        position.apply(*action).unwrap();
    }
    assert_eq!(position.completed_turns(), 53);
    assert_eq!(position.outcome(), GameOutcome::Win(Player::Guest));
    assert_eq!(paisho_core::harmonies(position.board()).len(), 18);
    assert!(
        harmony_ring_owners_for_profile(position.board(), RuleProfileId::SkudPaiSho2022).is_empty()
    );
    assert!(legal_actions(&position).is_empty());
    let before = position.clone();
    assert!(position.apply(old.actions()[74]).is_err());
    assert_eq!(position, before);
}

#[test]
fn explicit_revalidation_discards_only_the_post_terminal_suffix() {
    let old: GameRecord = REPORTED.parse().unwrap();
    let (updated, position) = old
        .replay_prefix_with_rules(RuleProfileId::CURRENT)
        .unwrap();
    assert_eq!(updated.rules(), RuleProfileId::CURRENT);
    assert_eq!(updated.actions(), &old.actions()[..74]);
    assert_eq!(updated.replay().unwrap(), position);
    assert_eq!(position.outcome(), GameOutcome::Win(Player::Guest));
    assert_eq!(old.to_string(), REPORTED);
    let (same, replayed) = old.replay_prefix_with_rules(old.rules()).unwrap();
    assert_eq!(same, old);
    assert_eq!(replayed.outcome(), GameOutcome::Win(Player::Host));
    let invalid: GameRecord = REPORTED
        .replace("arrange 0,-8 2,-5", "arrange 0,-8 0,8")
        .parse()
        .unwrap();
    assert_eq!(
        invalid
            .replay_prefix_with_rules(RuleProfileId::CURRENT)
            .unwrap_err()
            .action_number,
        1
    );
}

#[test]
fn changing_only_the_header_does_not_silently_accept_a_post_terminal_tail() {
    let changed: GameRecord = REPORTED
        .replace(
            RuleProfileId::SkudPaiSho2022.as_str(),
            RuleProfileId::CURRENT.as_str(),
        )
        .parse()
        .unwrap();
    assert_eq!(changed.replay().unwrap_err().action_number, 75);
}
