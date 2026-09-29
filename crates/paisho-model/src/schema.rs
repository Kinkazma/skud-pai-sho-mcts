use paisho_core::{Accent, BasicFlower, SpecialFlower, TileKind};

pub const BOARD_SIZE_V1: usize = 17;
pub const BOARD_CELL_COUNT_V1: usize = BOARD_SIZE_V1 * BOARD_SIZE_V1;
pub const TILE_KIND_COUNT_V1: usize = 12;

pub const TILE_KINDS_V1: [TileKind; TILE_KIND_COUNT_V1] = [
    TileKind::Basic(BasicFlower::Red3),
    TileKind::Basic(BasicFlower::Red4),
    TileKind::Basic(BasicFlower::Red5),
    TileKind::Basic(BasicFlower::White3),
    TileKind::Basic(BasicFlower::White4),
    TileKind::Basic(BasicFlower::White5),
    TileKind::WhiteLotus,
    TileKind::Orchid,
    TileKind::Accent(Accent::Rock),
    TileKind::Accent(Accent::Wheel),
    TileKind::Accent(Accent::Knotweed),
    TileKind::Accent(Accent::Boat),
];

// A change in the engine catalogue or board dimensions requires a new encoding
// version instead of silently changing the V1 tensor contract.
const _: [(); BOARD_SIZE_V1] = [(); paisho_core::BOARD_SIZE];
const _: [(); BOARD_CELL_COUNT_V1] = [(); paisho_core::CELL_COUNT];
const _: [(); TILE_KIND_COUNT_V1] = [(); TileKind::COUNT];

pub const fn tile_slot_v1(kind: TileKind) -> usize {
    match kind {
        TileKind::Basic(flower) => match flower {
            BasicFlower::Red3 => 0,
            BasicFlower::Red4 => 1,
            BasicFlower::Red5 => 2,
            BasicFlower::White3 => 3,
            BasicFlower::White4 => 4,
            BasicFlower::White5 => 5,
        },
        TileKind::WhiteLotus => 6,
        TileKind::Orchid => 7,
        TileKind::Accent(accent) => match accent {
            Accent::Rock => 8,
            Accent::Wheel => 9,
            Accent::Knotweed => 10,
            Accent::Boat => 11,
        },
    }
}

pub const fn tile_kind_v1(slot: u16) -> Option<TileKind> {
    if slot < TILE_KIND_COUNT_V1 as u16 {
        Some(TILE_KINDS_V1[slot as usize])
    } else {
        None
    }
}

pub const fn special_slot_v1(flower: SpecialFlower) -> usize {
    tile_slot_v1(flower.kind())
}
