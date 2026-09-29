use core::fmt;

use crate::legality::legal_arrangement_destinations_with_clash_state;
use crate::{
    all_coordinates, apply_accent, apply_arrangement, legal_accent_placements, point_type_at,
    Accent, AccentPlacement, BasicFlower, Coordinate, GameOutcome, Position, SpecialFlower, Tile,
    TileKind, TurnPhase, ACCENTS, BASIC_FLOWERS, SPECIAL_FLOWERS,
};

/// One decision in the phase-based environment. Harmony Bonuses stay as a
/// separate decision so policy heads do not need a Cartesian product of moves
/// and bonus placements.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Action {
    Plant {
        flower: BasicFlower,
        gate: Coordinate,
    },
    Arrange {
        from: Coordinate,
        to: Coordinate,
    },
    SkipHarmonyBonus,
    PlayAccent {
        accent: Accent,
        placement: AccentPlacement,
    },
    PlantSpecial {
        flower: SpecialFlower,
        gate: Coordinate,
    },
    BonusPlantBasic {
        flower: BasicFlower,
        gate: Coordinate,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActionResult {
    pub captured: Option<Tile>,
    pub removed_by_accent: Option<Tile>,
    pub harmony_bonus_awarded: bool,
    pub turn_completed: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ApplyError {
    GameAlreadyOver,
    WrongPhase,
    DestinationIsNotAnOpenGate,
    TileMissingFromReserve,
    BasicBonusRequiresNoGrowingFlower,
    Arrangement(crate::ArrangementError),
    Accent(crate::AccentError),
}

pub fn legal_actions(position: &Position) -> Vec<Action> {
    if position.outcome != GameOutcome::Ongoing {
        return Vec::new();
    }
    if position.phase == TurnPhase::HarmonyBonus {
        return legal_harmony_bonuses(position);
    }

    let player = position.to_move;
    let board_has_clash = crate::has_clash(&position.board);
    let mut actions = Vec::new();
    for flower in BASIC_FLOWERS {
        if position.reserves[player.index()].count(TileKind::Basic(flower)) == 0 {
            continue;
        }
        for gate in all_coordinates().filter(|coordinate| {
            point_type_at(*coordinate).is_gate() && position.board.is_empty(*coordinate)
        }) {
            actions.push(Action::Plant { flower, gate });
        }
    }

    for (from, tile) in position.board.occupied() {
        if tile.owner != player || !tile.kind.is_flower() {
            continue;
        }
        for to in legal_arrangement_destinations_with_clash_state(
            &position.board,
            player,
            from,
            board_has_clash,
        ) {
            actions.push(Action::Arrange { from, to });
        }
    }
    actions
}

/// Existence check for Gen5's end-of-turn blocking rule. Stops at the first
/// movable flower and avoids constructing all plant/action combinations.
fn has_main_action(position: &Position) -> bool {
    if position.reserves[position.to_move.index()].basic_count() > 0
        && [
            crate::NORTH_GATE,
            crate::EAST_GATE,
            crate::SOUTH_GATE,
            crate::WEST_GATE,
        ]
        .iter()
        .any(|at| position.board.is_empty(*at))
    {
        return true;
    }
    let clash = crate::has_clash(&position.board);
    position.board.occupied().any(|(from, tile)| {
        tile.owner == position.to_move
            && tile.kind.is_flower()
            && !legal_arrangement_destinations_with_clash_state(
                &position.board,
                position.to_move,
                from,
                clash,
            )
            .is_empty()
    })
}

impl Position {
    pub fn apply(&mut self, action: Action) -> Result<ActionResult, ApplyError> {
        if self.outcome != GameOutcome::Ongoing {
            return Err(ApplyError::GameAlreadyOver);
        }

        match action {
            Action::Plant { flower, gate } => self.apply_plant(flower, gate),
            Action::Arrange { from, to } => self.apply_main_arrangement(from, to),
            Action::SkipHarmonyBonus => self.apply_skip_bonus(),
            Action::PlayAccent { accent, placement } => self.apply_bonus_accent(accent, placement),
            Action::PlantSpecial { flower, gate } => {
                self.apply_bonus_plant(flower.kind(), gate, false)
            }
            Action::BonusPlantBasic { flower, gate } => {
                self.apply_bonus_plant(TileKind::Basic(flower), gate, true)
            }
        }
    }

    fn apply_plant(
        &mut self,
        flower: BasicFlower,
        gate: Coordinate,
    ) -> Result<ActionResult, ApplyError> {
        if self.phase != TurnPhase::Main {
            return Err(ApplyError::WrongPhase);
        }
        if !point_type_at(gate).is_gate() || !self.board.is_empty(gate) {
            return Err(ApplyError::DestinationIsNotAnOpenGate);
        }

        let kind = TileKind::Basic(flower);
        if !self.reserves[self.to_move.index()].take(kind) {
            return Err(ApplyError::TileMissingFromReserve);
        }
        self.board
            .place(gate, Tile::new(self.to_move, kind))
            .expect("validated open Gate accepts a tile");
        if self.reserves[self.to_move.index()].basic_count() == 0 {
            self.conclude_by_midline_score();
        } else {
            self.finish_turn();
        }

        Ok(ActionResult {
            captured: None,
            removed_by_accent: None,
            harmony_bonus_awarded: false,
            turn_completed: true,
        })
    }

    fn apply_main_arrangement(
        &mut self,
        from: Coordinate,
        to: Coordinate,
    ) -> Result<ActionResult, ApplyError> {
        if self.phase != TurnPhase::Main {
            return Err(ApplyError::WrongPhase);
        }
        let result = apply_arrangement(&mut self.board, self.to_move, from, to)
            .map_err(ApplyError::Arrangement)?;

        let terminal = self.conclude_harmony_ring();
        if !terminal {
            if result.formed_new_harmony {
                self.phase = TurnPhase::HarmonyBonus;
            } else {
                self.finish_turn();
            }
        }

        Ok(ActionResult {
            captured: result.captured,
            removed_by_accent: None,
            harmony_bonus_awarded: result.formed_new_harmony && !terminal,
            turn_completed: terminal || !result.formed_new_harmony,
        })
    }

    fn apply_skip_bonus(&mut self) -> Result<ActionResult, ApplyError> {
        if self.phase != TurnPhase::HarmonyBonus {
            return Err(ApplyError::WrongPhase);
        }
        self.finish_turn();
        Ok(ActionResult {
            captured: None,
            removed_by_accent: None,
            harmony_bonus_awarded: false,
            turn_completed: true,
        })
    }

    fn apply_bonus_accent(
        &mut self,
        accent: Accent,
        placement: AccentPlacement,
    ) -> Result<ActionResult, ApplyError> {
        if self.phase != TurnPhase::HarmonyBonus {
            return Err(ApplyError::WrongPhase);
        }
        let kind = TileKind::Accent(accent);
        if self.reserves[self.to_move.index()].count(kind) == 0 {
            return Err(ApplyError::TileMissingFromReserve);
        }
        let effect = apply_accent(&mut self.board, self.to_move, accent, placement)
            .map_err(ApplyError::Accent)?;
        let removed = self.reserves[self.to_move.index()].take(kind);
        debug_assert!(removed);
        if !self.conclude_harmony_ring() {
            self.finish_turn();
        }

        Ok(ActionResult {
            captured: None,
            removed_by_accent: effect.removed,
            harmony_bonus_awarded: false,
            turn_completed: true,
        })
    }

    fn apply_bonus_plant(
        &mut self,
        kind: TileKind,
        gate: Coordinate,
        require_no_growing_flower: bool,
    ) -> Result<ActionResult, ApplyError> {
        if self.phase != TurnPhase::HarmonyBonus {
            return Err(ApplyError::WrongPhase);
        }
        if require_no_growing_flower && player_has_growing_flower(self, self.to_move) {
            return Err(ApplyError::BasicBonusRequiresNoGrowingFlower);
        }
        if !point_type_at(gate).is_gate() || !self.board.is_empty(gate) {
            return Err(ApplyError::DestinationIsNotAnOpenGate);
        }
        if !self.reserves[self.to_move.index()].take(kind) {
            return Err(ApplyError::TileMissingFromReserve);
        }
        self.board
            .place(gate, Tile::new(self.to_move, kind))
            .expect("validated open Gate accepts a bonus Flower");
        if kind.is_basic() && self.reserves[self.to_move.index()].basic_count() == 0 {
            self.conclude_by_midline_score();
        } else {
            self.finish_turn();
        }

        Ok(ActionResult {
            captured: None,
            removed_by_accent: None,
            harmony_bonus_awarded: false,
            turn_completed: true,
        })
    }

    fn finish_turn(&mut self) {
        self.completed_turns += 1;
        self.to_move = self.to_move.opponent();
        self.phase = TurnPhase::Main;
        // The complete turn (including its optional bonus) has finished. The
        // previous player caused this blocked turn and loses; the blocked
        // player wins. Normal ring/exhaustion wins have already been settled.
        if self.rules.profile().blocking_player_loses && !has_main_action(self) {
            self.outcome = GameOutcome::Win(self.to_move);
        }
    }

    fn conclude_harmony_ring(&mut self) -> bool {
        let owners = crate::harmony_ring_owners_for_profile(&self.board, self.rules);
        self.outcome = match owners.as_slice() {
            [] => return false,
            [winner] => GameOutcome::Win(*winner),
            _ => GameOutcome::Draw,
        };
        self.complete_terminal_turn();
        true
    }

    fn conclude_by_midline_score(&mut self) {
        let host = crate::midline_crossing_harmony_count(&self.board, crate::Player::Host);
        let guest = crate::midline_crossing_harmony_count(&self.board, crate::Player::Guest);
        self.outcome = match host.cmp(&guest) {
            core::cmp::Ordering::Greater => GameOutcome::Win(crate::Player::Host),
            core::cmp::Ordering::Less => GameOutcome::Win(crate::Player::Guest),
            core::cmp::Ordering::Equal => GameOutcome::Draw,
        };
        self.complete_terminal_turn();
    }

    fn complete_terminal_turn(&mut self) {
        self.completed_turns += 1;
        self.phase = TurnPhase::Main;
    }
}

impl fmt::Display for ApplyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GameAlreadyOver => formatter.write_str("the game is already over"),
            Self::WrongPhase => formatter.write_str("the action is not valid in this turn phase"),
            Self::DestinationIsNotAnOpenGate => {
                formatter.write_str("Planting requires an open Gate")
            }
            Self::TileMissingFromReserve => {
                formatter.write_str("the requested tile is not in the player's reserve")
            }
            Self::BasicBonusRequiresNoGrowingFlower => {
                formatter.write_str("a Basic Flower bonus requires no Growing Flower")
            }
            Self::Arrangement(error) => write!(formatter, "invalid Arrangement: {error}"),
            Self::Accent(error) => write!(formatter, "invalid Accent action: {error}"),
        }
    }
}

impl std::error::Error for ApplyError {}

fn legal_harmony_bonuses(position: &Position) -> Vec<Action> {
    let player = position.to_move;
    let reserve = &position.reserves[player.index()];
    let mut actions = vec![Action::SkipHarmonyBonus];

    for accent in ACCENTS {
        if reserve.count(TileKind::Accent(accent)) == 0 {
            continue;
        }
        for placement in legal_accent_placements(&position.board, player, accent) {
            actions.push(Action::PlayAccent { accent, placement });
        }
    }

    let open_gates: Vec<_> = all_coordinates()
        .filter(|coordinate| {
            point_type_at(*coordinate).is_gate() && position.board.is_empty(*coordinate)
        })
        .collect();
    for flower in SPECIAL_FLOWERS {
        if reserve.count(flower.kind()) > 0 {
            for gate in &open_gates {
                actions.push(Action::PlantSpecial {
                    flower,
                    gate: *gate,
                });
            }
        }
    }

    if !player_has_growing_flower(position, player) {
        for flower in BASIC_FLOWERS {
            if reserve.count(TileKind::Basic(flower)) > 0 {
                for gate in &open_gates {
                    actions.push(Action::BonusPlantBasic {
                        flower,
                        gate: *gate,
                    });
                }
            }
        }
    }
    actions
}

fn player_has_growing_flower(position: &Position, player: crate::Player) -> bool {
    position.board.occupied().any(|(coordinate, tile)| {
        tile.owner == player && tile.kind.is_flower() && point_type_at(coordinate).is_gate()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccentLoadout, Board, Player, StandardSetup, NORTH_GATE, SOUTH_GATE};

    fn position_with_board(board: Board, to_move: Player) -> Position {
        Position {
            rules: crate::RuleProfileId::SkudPaiSho2022,
            board,
            reserves: [
                crate::Reserve::standard(AccentLoadout::balanced()),
                crate::Reserve::standard(AccentLoadout::balanced()),
            ],
            to_move,
            phase: TurnPhase::Main,
            completed_turns: 0,
            outcome: GameOutcome::Ongoing,
        }
    }

    fn leave_one_basic_in_reserve(position: &mut Position, player: Player, remaining: BasicFlower) {
        for flower in BASIC_FLOWERS {
            let kind = TileKind::Basic(flower);
            let target = u8::from(flower == remaining);
            while position.reserves[player.index()].count(kind) > target {
                assert!(position.reserves[player.index()].take(kind));
            }
        }
    }

    #[test]
    fn initial_position_generates_plants_and_arrangements() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let actions = legal_actions(&position);
        assert!(actions.contains(&Action::Plant {
            flower: BasicFlower::White5,
            gate: crate::WEST_GATE,
        }));
        assert!(actions.iter().any(|action| matches!(
            action,
            Action::Arrange { from, .. } if *from == SOUTH_GATE
        )));
    }

    #[test]
    fn planting_consumes_reserve_and_finishes_the_turn() {
        let mut position =
            Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let before = position
            .reserve(Player::Guest)
            .count(TileKind::Basic(BasicFlower::White5));
        let result = position
            .apply(Action::Plant {
                flower: BasicFlower::White5,
                gate: crate::WEST_GATE,
            })
            .unwrap();

        assert!(result.turn_completed);
        assert_eq!(position.to_move(), Player::Host);
        assert_eq!(position.completed_turns(), 1);
        assert_eq!(
            position.board().get(crate::WEST_GATE),
            Some(Tile::new(
                Player::Guest,
                TileKind::Basic(BasicFlower::White5)
            ))
        );
        assert_eq!(
            position
                .reserve(Player::Guest)
                .count(TileKind::Basic(BasicFlower::White5)),
            before - 1
        );
    }

    #[test]
    fn a_new_harmony_opens_an_optional_bonus_phase() {
        let mut board = Board::empty();
        board
            .place(
                Coordinate::new(-2, 0).unwrap(),
                Tile::new(Player::Guest, TileKind::Basic(BasicFlower::Red3)),
            )
            .unwrap();
        board
            .place(
                Coordinate::new(0, 1).unwrap(),
                Tile::new(Player::Guest, TileKind::Basic(BasicFlower::Red4)),
            )
            .unwrap();
        let mut position = position_with_board(board, Player::Guest);

        let result = position
            .apply(Action::Arrange {
                from: Coordinate::new(0, 1).unwrap(),
                to: Coordinate::new(2, 0).unwrap(),
            })
            .unwrap();
        assert!(result.harmony_bonus_awarded);
        assert!(!result.turn_completed);
        assert_eq!(position.phase(), TurnPhase::HarmonyBonus);
        assert_eq!(position.to_move(), Player::Guest);
        assert!(legal_actions(&position).contains(&Action::SkipHarmonyBonus));

        position.apply(Action::SkipHarmonyBonus).unwrap();
        assert_eq!(position.phase(), TurnPhase::Main);
        assert_eq!(position.to_move(), Player::Host);
        assert_eq!(position.completed_turns(), 1);
    }

    #[test]
    fn phase_mismatches_do_not_mutate_the_position() {
        let mut position =
            Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let before = position.clone();
        assert_eq!(
            position.apply(Action::SkipHarmonyBonus),
            Err(ApplyError::WrongPhase)
        );
        assert_eq!(position, before);
        assert!(position.board().get(NORTH_GATE).is_some());
    }

    #[test]
    fn bonus_generator_is_complete_and_every_generated_action_applies() {
        let mut position = position_with_board(Board::empty(), Player::Guest);
        position.phase = TurnPhase::HarmonyBonus;
        let actions = legal_actions(&position);

        assert!(actions.contains(&Action::SkipHarmonyBonus));
        assert!(actions.contains(&Action::PlantSpecial {
            flower: SpecialFlower::WhiteLotus,
            gate: NORTH_GATE,
        }));
        assert!(actions.contains(&Action::BonusPlantBasic {
            flower: BasicFlower::Red3,
            gate: NORTH_GATE,
        }));
        assert!(actions.iter().any(|action| matches!(
            action,
            Action::PlayAccent {
                accent: Accent::Rock,
                ..
            }
        )));

        for action in actions {
            let mut copy = position.clone();
            assert!(
                copy.apply(action).is_ok(),
                "generated invalid action: {action:?}"
            );
            assert_eq!(copy.phase(), TurnPhase::Main);
            assert_eq!(copy.to_move(), Player::Host);
        }
    }

    #[test]
    fn basic_bonus_is_hidden_while_player_has_a_growing_flower() {
        let mut board = Board::empty();
        board
            .place(
                NORTH_GATE,
                Tile::new(Player::Guest, TileKind::Basic(BasicFlower::Red3)),
            )
            .unwrap();
        let mut position = position_with_board(board, Player::Guest);
        position.phase = TurnPhase::HarmonyBonus;

        let actions = legal_actions(&position);
        assert!(!actions
            .iter()
            .any(|action| matches!(action, Action::BonusPlantBasic { .. })));
        assert!(actions
            .iter()
            .any(|action| matches!(action, Action::PlantSpecial { .. })));
    }

    #[test]
    fn playing_an_accent_consumes_it_and_finishes_the_turn() {
        let mut position = position_with_board(Board::empty(), Player::Guest);
        position.phase = TurnPhase::HarmonyBonus;
        let kind = TileKind::Accent(Accent::Rock);
        let before = position.reserve(Player::Guest).count(kind);

        position
            .apply(Action::PlayAccent {
                accent: Accent::Rock,
                placement: AccentPlacement::At(Coordinate::new(0, 0).unwrap()),
            })
            .unwrap();

        assert_eq!(position.reserve(Player::Guest).count(kind), before - 1);
        assert_eq!(position.to_move(), Player::Host);
        assert_eq!(position.phase(), TurnPhase::Main);
    }

    #[test]
    fn completing_a_harmony_ring_ends_the_game_without_a_bonus() {
        let mut board = Board::empty();
        for (coordinate, kind) in [
            ((-2, -2), TileKind::WhiteLotus),
            ((2, -2), TileKind::Basic(BasicFlower::White5)),
            ((2, 2), TileKind::Basic(BasicFlower::Red3)),
            ((-2, 1), TileKind::Basic(BasicFlower::White5)),
        ] {
            board
                .place(
                    Coordinate::new(coordinate.0, coordinate.1).unwrap(),
                    Tile::new(Player::Guest, kind),
                )
                .unwrap();
        }
        let mut position = position_with_board(board, Player::Guest);

        let result = position
            .apply(Action::Arrange {
                from: Coordinate::new(-2, 1).unwrap(),
                to: Coordinate::new(-2, 2).unwrap(),
            })
            .unwrap();

        assert_eq!(position.outcome(), GameOutcome::Win(Player::Guest));
        assert_eq!(position.phase(), TurnPhase::Main);
        assert_eq!(position.to_move(), Player::Guest);
        assert_eq!(position.completed_turns(), 1);
        assert!(!result.harmony_bonus_awarded);
        assert!(result.turn_completed);
        assert!(legal_actions(&position).is_empty());
    }

    #[test]
    fn planting_the_last_basic_flower_uses_midline_harmony_score() {
        let mut board = Board::empty();
        for (coordinate, flower) in [((-2, 4), BasicFlower::Red3), ((2, 4), BasicFlower::Red4)] {
            board
                .place(
                    Coordinate::new(coordinate.0, coordinate.1).unwrap(),
                    Tile::new(Player::Guest, TileKind::Basic(flower)),
                )
                .unwrap();
        }
        let mut position = position_with_board(board, Player::Guest);
        leave_one_basic_in_reserve(&mut position, Player::Guest, BasicFlower::White3);

        position
            .apply(Action::Plant {
                flower: BasicFlower::White3,
                gate: NORTH_GATE,
            })
            .unwrap();

        assert_eq!(position.outcome(), GameOutcome::Win(Player::Guest));
        assert_eq!(position.completed_turns(), 1);
        assert_eq!(position.to_move(), Player::Guest);
        assert!(legal_actions(&position).is_empty());
    }

    #[test]
    fn an_equal_midline_score_ends_in_a_draw() {
        let mut position = position_with_board(Board::empty(), Player::Guest);
        leave_one_basic_in_reserve(&mut position, Player::Guest, BasicFlower::White3);

        position
            .apply(Action::Plant {
                flower: BasicFlower::White3,
                gate: NORTH_GATE,
            })
            .unwrap();

        assert_eq!(position.outcome(), GameOutcome::Draw);
        assert_eq!(position.completed_turns(), 1);
    }
}
