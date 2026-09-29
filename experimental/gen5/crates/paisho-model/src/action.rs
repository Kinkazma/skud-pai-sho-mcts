use core::fmt;

use paisho_core::{
    legal_actions, point_type_at, Accent, AccentPlacement, Action, BasicFlower, Coordinate, Player,
    Position, SpecialFlower, TileKind,
};

use crate::{
    canonical_coordinate_v1, native_coordinate_v1, special_slot_v1, tile_kind_v1, tile_slot_v1,
    BOARD_CELL_COUNT_V1, BOARD_SIZE_V1, TILE_KIND_COUNT_V1,
};

pub const ACTION_FAMILY_COUNT_V1: usize = 7;
pub const ACTION_SLOT_COUNT_V1: usize = 4;
pub const NO_TILE_V1: u16 = TILE_KIND_COUNT_V1 as u16;
pub const NO_COORDINATE_V1: u16 = BOARD_CELL_COUNT_V1 as u16;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u16)]
pub enum ActionFamilyV1 {
    PlantBasicMain = 0,
    Arrange = 1,
    SkipHarmonyBonus = 2,
    PlaceAccent = 3,
    BoatMove = 4,
    PlantSpecial = 5,
    PlantBasicBonus = 6,
}

impl ActionFamilyV1 {
    pub const fn index(self) -> usize {
        self as usize
    }

    const fn from_slot(slot: u16) -> Option<Self> {
        match slot {
            0 => Some(Self::PlantBasicMain),
            1 => Some(Self::Arrange),
            2 => Some(Self::SkipHarmonyBonus),
            3 => Some(Self::PlaceAccent),
            4 => Some(Self::BoatMove),
            5 => Some(Self::PlantSpecial),
            6 => Some(Self::PlantBasicBonus),
            _ => None,
        }
    }
}

/// A stable four-slot policy address: family, tile, source, destination.
/// Sentinels represent components that are absent for a family.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ActionEncodingV1 {
    family: ActionFamilyV1,
    tile_slot: u16,
    source_slot: u16,
    destination_slot: u16,
}

impl ActionEncodingV1 {
    pub fn from_slots(slots: [u16; ACTION_SLOT_COUNT_V1]) -> Result<Self, ActionEncodingError> {
        let family = ActionFamilyV1::from_slot(slots[0])
            .ok_or(ActionEncodingError::InvalidFamily(slots[0]))?;
        validate_tile_slot(slots[1])?;
        validate_coordinate_slot(slots[2])?;
        validate_coordinate_slot(slots[3])?;

        let encoded = Self {
            family,
            tile_slot: slots[1],
            source_slot: slots[2],
            destination_slot: slots[3],
        };
        encoded.validate_shape()?;
        Ok(encoded)
    }

    pub const fn family(self) -> ActionFamilyV1 {
        self.family
    }

    pub const fn slots(self) -> [u16; ACTION_SLOT_COUNT_V1] {
        [
            self.family as u16,
            self.tile_slot,
            self.source_slot,
            self.destination_slot,
        ]
    }

    pub fn tile_kind(self) -> Option<TileKind> {
        tile_from_slot(self.tile_slot)
    }

    pub fn source(self) -> Option<Coordinate> {
        coordinate_from_slot(self.source_slot)
    }

    pub fn destination(self) -> Option<Coordinate> {
        coordinate_from_slot(self.destination_slot)
    }

    /// Decodes the structural action. Legality in a particular position remains
    /// the responsibility of `paisho_core::legal_actions`.
    pub fn decode(self, perspective: Player) -> Result<Action, ActionEncodingError> {
        self.validate_shape()?;
        let tile = self.tile_kind();
        let source = self
            .source()
            .map(|at| native_coordinate_v1(perspective, at));
        let destination = self
            .destination()
            .map(|at| native_coordinate_v1(perspective, at));

        let action = match self.family {
            ActionFamilyV1::PlantBasicMain => Action::Plant {
                flower: basic_from_kind(tile.expect("validated family requires a tile"))
                    .expect("validated family requires a Basic Flower"),
                gate: destination.expect("validated family requires a destination"),
            },
            ActionFamilyV1::Arrange => Action::Arrange {
                from: source.expect("validated family requires a source"),
                to: destination.expect("validated family requires a destination"),
            },
            ActionFamilyV1::SkipHarmonyBonus => Action::SkipHarmonyBonus,
            ActionFamilyV1::PlaceAccent => Action::PlayAccent {
                accent: accent_from_kind(tile.expect("validated family requires a tile"))
                    .expect("validated family requires an Accent"),
                placement: AccentPlacement::At(
                    destination.expect("validated family requires a destination"),
                ),
            },
            ActionFamilyV1::BoatMove => Action::PlayAccent {
                accent: Accent::Boat,
                placement: AccentPlacement::BoatMove {
                    flower: source.expect("validated family requires a source"),
                    destination: destination.expect("validated family requires a destination"),
                },
            },
            ActionFamilyV1::PlantSpecial => Action::PlantSpecial {
                flower: special_from_kind(tile.expect("validated family requires a tile"))
                    .expect("validated family requires a Special Flower"),
                gate: destination.expect("validated family requires a destination"),
            },
            ActionFamilyV1::PlantBasicBonus => Action::BonusPlantBasic {
                flower: basic_from_kind(tile.expect("validated family requires a tile"))
                    .expect("validated family requires a Basic Flower"),
                gate: destination.expect("validated family requires a destination"),
            },
        };
        Ok(action)
    }

    fn validate_shape(self) -> Result<(), ActionEncodingError> {
        let tile = self.tile_kind();
        let source = self.source();
        let destination = self.destination();
        let is_gate = destination.is_some_and(|at| point_type_at(at).is_gate());
        let shape_is_valid = match self.family {
            ActionFamilyV1::PlantBasicMain | ActionFamilyV1::PlantBasicBonus => {
                tile.is_some_and(TileKind::is_basic)
                    && source.is_none()
                    && destination.is_some()
                    && is_gate
            }
            ActionFamilyV1::Arrange => {
                tile.is_none() && source.is_some() && destination.is_some() && source != destination
            }
            ActionFamilyV1::SkipHarmonyBonus => {
                tile.is_none() && source.is_none() && destination.is_none()
            }
            ActionFamilyV1::PlaceAccent => {
                tile.is_some_and(TileKind::is_accent)
                    && source.is_none()
                    && destination.is_some()
                    && !is_gate
            }
            ActionFamilyV1::BoatMove => {
                tile == Some(TileKind::Accent(Accent::Boat))
                    && source.is_some()
                    && destination.is_some()
                    && source != destination
            }
            ActionFamilyV1::PlantSpecial => {
                tile.is_some_and(TileKind::is_special)
                    && source.is_none()
                    && destination.is_some()
                    && is_gate
            }
        };
        if shape_is_valid {
            Ok(())
        } else {
            Err(ActionEncodingError::InvalidShape(self.family))
        }
    }
}

pub fn encode_action_v1(
    action: Action,
    perspective: Player,
) -> Result<ActionEncodingV1, ActionEncodingError> {
    let no_tile = NO_TILE_V1;
    let no_coordinate = NO_COORDINATE_V1;
    let slots = match action {
        Action::Plant { flower, gate } => [
            ActionFamilyV1::PlantBasicMain as u16,
            tile_slot_v1(TileKind::Basic(flower)) as u16,
            no_coordinate,
            coordinate_slot(perspective, gate),
        ],
        Action::Arrange { from, to } => [
            ActionFamilyV1::Arrange as u16,
            no_tile,
            coordinate_slot(perspective, from),
            coordinate_slot(perspective, to),
        ],
        Action::SkipHarmonyBonus => [
            ActionFamilyV1::SkipHarmonyBonus as u16,
            no_tile,
            no_coordinate,
            no_coordinate,
        ],
        Action::PlayAccent {
            accent,
            placement: AccentPlacement::At(at),
        } => [
            ActionFamilyV1::PlaceAccent as u16,
            tile_slot_v1(TileKind::Accent(accent)) as u16,
            no_coordinate,
            coordinate_slot(perspective, at),
        ],
        Action::PlayAccent {
            accent,
            placement:
                AccentPlacement::BoatMove {
                    flower,
                    destination,
                },
        } => {
            if accent != Accent::Boat {
                return Err(ActionEncodingError::InvalidBoatAccent(accent));
            }
            [
                ActionFamilyV1::BoatMove as u16,
                tile_slot_v1(TileKind::Accent(Accent::Boat)) as u16,
                coordinate_slot(perspective, flower),
                coordinate_slot(perspective, destination),
            ]
        }
        Action::PlantSpecial { flower, gate } => [
            ActionFamilyV1::PlantSpecial as u16,
            special_slot_v1(flower) as u16,
            no_coordinate,
            coordinate_slot(perspective, gate),
        ],
        Action::BonusPlantBasic { flower, gate } => [
            ActionFamilyV1::PlantBasicBonus as u16,
            tile_slot_v1(TileKind::Basic(flower)) as u16,
            no_coordinate,
            coordinate_slot(perspective, gate),
        ],
    };
    ActionEncodingV1::from_slots(slots)
}

pub fn encode_legal_actions_v1(
    position: &Position,
) -> Result<Vec<ActionEncodingV1>, ActionEncodingError> {
    legal_actions(position)
        .into_iter()
        .map(|action| encode_action_v1(action, position.to_move()))
        .collect()
}

const fn coordinate_slot(perspective: Player, coordinate: Coordinate) -> u16 {
    canonical_coordinate_v1(perspective, coordinate).dense_index() as u16
}

fn validate_tile_slot(slot: u16) -> Result<(), ActionEncodingError> {
    if slot <= NO_TILE_V1 {
        Ok(())
    } else {
        Err(ActionEncodingError::InvalidTile(slot))
    }
}

fn validate_coordinate_slot(slot: u16) -> Result<(), ActionEncodingError> {
    if slot > NO_COORDINATE_V1 {
        return Err(ActionEncodingError::InvalidCoordinate(slot));
    }
    if slot == NO_COORDINATE_V1 {
        return Ok(());
    }
    let coordinate = coordinate_from_slot(slot).expect("validated dense coordinate slot");
    if point_type_at(coordinate).is_playable() {
        Ok(())
    } else {
        Err(ActionEncodingError::NonPlayableCoordinate(slot))
    }
}

fn tile_from_slot(slot: u16) -> Option<TileKind> {
    tile_kind_v1(slot)
}

fn coordinate_from_slot(slot: u16) -> Option<Coordinate> {
    if slot >= NO_COORDINATE_V1 {
        return None;
    }
    let index = usize::from(slot);
    Some(
        Coordinate::from_grid(index / BOARD_SIZE_V1, index % BOARD_SIZE_V1)
            .expect("dense coordinate slot is inside the board envelope"),
    )
}

const fn basic_from_kind(kind: TileKind) -> Option<BasicFlower> {
    match kind {
        TileKind::Basic(flower) => Some(flower),
        _ => None,
    }
}

const fn accent_from_kind(kind: TileKind) -> Option<Accent> {
    match kind {
        TileKind::Accent(accent) => Some(accent),
        _ => None,
    }
}

const fn special_from_kind(kind: TileKind) -> Option<SpecialFlower> {
    match kind {
        TileKind::WhiteLotus => Some(SpecialFlower::WhiteLotus),
        TileKind::Orchid => Some(SpecialFlower::Orchid),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionEncodingError {
    InvalidFamily(u16),
    InvalidTile(u16),
    InvalidCoordinate(u16),
    NonPlayableCoordinate(u16),
    InvalidShape(ActionFamilyV1),
    InvalidBoatAccent(Accent),
}

impl fmt::Display for ActionEncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFamily(slot) => write!(formatter, "invalid V1 action family slot {slot}"),
            Self::InvalidTile(slot) => write!(formatter, "invalid V1 tile slot {slot}"),
            Self::InvalidCoordinate(slot) => {
                write!(formatter, "invalid V1 coordinate slot {slot}")
            }
            Self::NonPlayableCoordinate(slot) => {
                write!(formatter, "V1 coordinate slot {slot} is not playable")
            }
            Self::InvalidShape(family) => {
                write!(formatter, "invalid component shape for {family:?}")
            }
            Self::InvalidBoatAccent(accent) => {
                write!(
                    formatter,
                    "BoatMove placement cannot be paired with {accent:?}"
                )
            }
        }
    }
}

impl std::error::Error for ActionEncodingError {}
