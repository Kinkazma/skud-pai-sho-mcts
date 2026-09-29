use crate::{
    AccentLoadout, BasicFlower, Board, Player, Reserve, RuleProfileId, Tile, TileKind, NORTH_GATE,
    SOUTH_GATE,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StandardSetup {
    pub host_accents: AccentLoadout,
    pub guest_accents: AccentLoadout,
    pub starting_flower: BasicFlower,
}

impl StandardSetup {
    pub const fn balanced(starting_flower: BasicFlower) -> Self {
        Self {
            host_accents: AccentLoadout::balanced(),
            guest_accents: AccentLoadout::balanced(),
            starting_flower,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GameOutcome {
    Ongoing,
    Win(Player),
    Draw,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnPhase {
    Main,
    HarmonyBonus,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Position {
    pub(crate) rules: RuleProfileId,
    pub(crate) board: Board,
    pub(crate) reserves: [Reserve; 2],
    pub(crate) to_move: Player,
    pub(crate) phase: TurnPhase,
    pub(crate) completed_turns: u32,
    pub(crate) outcome: GameOutcome,
}

impl Position {
    /// Creates the position after the formal opening placement. The Guest chose
    /// the flower, both players received it in opposite Gates, and Guest moves.
    pub fn from_standard_setup(setup: StandardSetup) -> Self {
        Self::from_standard_setup_with_rules(setup, RuleProfileId::CURRENT)
    }

    /// Explicit profile for historical replay. New games should use the default.
    pub fn from_standard_setup_with_rules(setup: StandardSetup, rules: RuleProfileId) -> Self {
        let mut board = Board::empty();
        let mut reserves = [
            Reserve::standard(setup.host_accents),
            Reserve::standard(setup.guest_accents),
        ];
        let kind = TileKind::Basic(setup.starting_flower);

        let host_removed = reserves[Player::Host.index()].take(kind);
        let guest_removed = reserves[Player::Guest.index()].take(kind);
        debug_assert!(host_removed && guest_removed);
        board
            .place(NORTH_GATE, Tile::new(Player::Host, kind))
            .expect("north gate is empty and playable");
        board
            .place(SOUTH_GATE, Tile::new(Player::Guest, kind))
            .expect("south gate is empty and playable");

        Self {
            rules,
            board,
            reserves,
            to_move: Player::Guest,
            phase: TurnPhase::Main,
            completed_turns: 0,
            outcome: GameOutcome::Ongoing,
        }
    }

    pub const fn rule_profile(&self) -> RuleProfileId {
        self.rules
    }

    pub const fn board(&self) -> &Board {
        &self.board
    }

    pub const fn reserve(&self, player: Player) -> &Reserve {
        &self.reserves[player.index()]
    }

    pub const fn to_move(&self) -> Player {
        self.to_move
    }

    pub const fn phase(&self) -> TurnPhase {
        self.phase
    }

    pub const fn completed_turns(&self) -> u32 {
        self.completed_turns
    }

    pub const fn outcome(&self) -> GameOutcome {
        self.outcome
    }
}
