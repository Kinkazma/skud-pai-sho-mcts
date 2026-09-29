use paisho_core::*;

#[test]
fn responsible_guest_loses_in_archived_block_but_v2_is_unchanged() {
    let old: GameRecord = include_str!("fixtures/no-legal-actions-v2.psr")
        .parse()
        .unwrap();
    let blocked = old.replay().unwrap();
    assert_eq!(blocked.outcome(), GameOutcome::Ongoing);
    assert_eq!(blocked.to_move(), Player::Host);
    assert!(legal_actions(&blocked).is_empty());
    let (new, position) = old
        .replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)
        .unwrap();
    assert_eq!(new.rules(), RuleProfileId::SkudPaiShoGen5V1);
    assert_eq!(position.outcome(), GameOutcome::Win(Player::Host));
    assert!(new.actions().len() <= old.actions().len());
    assert_eq!(new.replay().unwrap(), position);
    assert_eq!(old.replay().unwrap(), blocked);
    assert_eq!(RuleProfileId::CURRENT, RuleProfileId::SkudPaiSho2022V2);
}

#[test]
fn ring_win_keeps_precedence_and_bonus_turns_replay() {
    let old: GameRecord = include_str!("fixtures/reported-ring-v1.psr")
        .parse()
        .unwrap();
    let (new, position) = old
        .replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)
        .unwrap();
    assert_eq!(new.actions().len(), 74);
    assert_eq!(position.outcome(), GameOutcome::Win(Player::Guest));
    assert_eq!(new.replay().unwrap(), position);
    let mut played = new.initial_position();
    let mut bonus = false;
    for action in new.actions() {
        let mover = played.to_move();
        played.apply(*action).unwrap();
        if played.phase() == TurnPhase::HarmonyBonus {
            bonus = true;
            assert_eq!(played.to_move(), mover);
            assert_eq!(played.outcome(), GameOutcome::Ongoing);
        }
    }
    assert!(bonus);
}
