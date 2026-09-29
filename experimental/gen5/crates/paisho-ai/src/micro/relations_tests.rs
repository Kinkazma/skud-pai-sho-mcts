use super::*;
use paisho_core::{BasicFlower, GameRecord, RuleProfileId, Tile, TileKind};

fn rectangle(dx:i8) -> Board {
    let mut b=Board::empty();
    for ((x,y),kind) in [((-2,-2),TileKind::WhiteLotus),((2,-2),TileKind::Basic(BasicFlower::White5)),
        ((2,2),TileKind::Basic(BasicFlower::Red3)),((-2,2),TileKind::Basic(BasicFlower::White5))] {
        b.place(Coordinate::new(x+dx,y).unwrap(),Tile::new(Player::Guest,kind)).unwrap();
    }
    b
}
#[test]
fn equal_cycle_rank_distinguishes_enclosing_off_centre_and_touching() {
    for (dx,kind,ring) in [(0,HarmonyCycleGeometry::EnclosingCentre,true),
        (4,HarmonyCycleGeometry::OffCentre,false),(2,HarmonyCycleGeometry::TouchingCentre,false)] {
        let board=rectangle(dx);
        let r=MicroRelations::from_board(&board,Player::Guest,RuleProfileId::SkudPaiShoGen5V1,[1,12],TurnPhase::Main);
        assert_eq!(r.global[2],1./16.);
        assert_eq!(r.global[3],f64::from(ring));
        assert!(r.cycles.iter().any(|c|c.geometry==kind));
        assert_eq!(r.edges,harmonies(&board));
        assert_eq!(r.global[6],1./54.);assert_eq!(r.global[15],12./54.);
    }
}
#[test]
fn explicit_pairs_and_categorical_endpoints_are_preserved() {
    let r=MicroRelations::from_board(&rectangle(0),Player::Guest,RuleProfileId::SkudPaiShoGen5V1,[1,12],TurnPhase::Main);
    let tokens=r.action_tokens(&[0.;32]);
    assert_eq!(tokens.len(),r.edges.len());
    for (x,h) in tokens.iter().zip(&r.edges) {
        assert_eq!(x[0],1.);assert_eq!(x[1],f64::from(h.first.x())/8.);
        assert_eq!(x[5..17].iter().sum::<f64>(),1.);
        assert_eq!(x[17..29].iter().sum::<f64>(),1.);assert_eq!(x[32],1.);
    }
}
#[test]
fn perspective_swaps_reserves_midline_and_ring_without_changing_graph() {
    let board=rectangle(0);
    let a=MicroRelations::from_board(&board,Player::Guest,RuleProfileId::SkudPaiShoGen5V1,[1,12],TurnPhase::Main);
    let b=MicroRelations::from_board(&board,Player::Host,RuleProfileId::SkudPaiShoGen5V1,[12,1],TurnPhase::Main);
    assert_eq!(a.global[..9],b.global[9..18]);assert_eq!(a.edges,b.edges);
    for (a,b) in a.action_tokens(&[0.;32]).iter().zip(b.action_tokens(&[0.;32])) {assert_eq!(a[0],-b[0]);}
}
#[test]
fn labels_replay_legal_actions_and_never_mutate_current_inputs() {
    let old:GameRecord=include_str!("../../tests/fixtures/site_bot_v1_ring_finish.psr").parse().unwrap();
    let (r,_)=old.replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1).unwrap();
    let mut p=r.initial_position();let mut observed_bonus=false;
    for &action in r.actions() {
        let before=p.clone();let rel=MicroRelations::extract(&p,p.to_move());
        let targets=micro_relation_targets(&p,action).unwrap();
        assert_eq!(p,before);
        assert_eq!(rel.global,MicroRelations::extract(&p,p.to_move()).global);
        assert!(targets.iter().all(|v|v.is_finite()));
        observed_bonus|=p.phase()==TurnPhase::HarmonyBonus;
        let perspective=p.to_move();p.apply(action).unwrap();
        let rings=harmony_ring_owners_for_profile(p.board(),p.rule_profile());
        assert_eq!(targets[8],f64::from(rings.contains(&perspective)));
        assert_eq!(targets[9],f64::from(rings.contains(&perspective.opponent())));
    }
    assert!(observed_bonus);
    let r=MicroRelations::extract(&p,p.to_move());assert!(r.global[3]+r.global[12]>0.);
}
