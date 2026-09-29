use paisho_core::{
    all_coordinates, playable_coordinates, point_type_at, PointType, Region, EAST_GATE, NORTH_GATE,
    PLAYABLE_POINT_COUNT, SOUTH_GATE, WEST_GATE,
};

// Exhaustive transcription of the 17 rows in the current reference implementation:
// https://github.com/thejambi/SkudPaiSho/blob/b849dbdabb1138ff0f6d609adf38b301c2f875ae/js/skud-pai-sho/SkudPaiShoBoard.js#L13-L330
// Legend: .=non-playable, n=neutral, G=gate, r/w=single garden,
// m=red+white, R/W=red/white+neutral, M=red+white+neutral.
const REFERENCE_ROWS: [&str; 17] = [
    "....nnnnGnnnn....",
    "...nnnnnMnnnnn...",
    "..nnnnnWmRnnnnn..",
    ".nnnnnWwmrRnnnnn.",
    "nnnnnWwwmrrRnnnnn",
    "nnnnWwwwmrrrRnnnn",
    "nnnWwwwwmrrrrRnnn",
    "nnWwwwwwmrrrrrRnn",
    "GMmmmmmmmmmmmmmMG",
    "nnRrrrrrmwwwwwWnn",
    "nnnRrrrrmwwwwWnnn",
    "nnnnRrrrmwwwWnnnn",
    "nnnnnRrrmwwWnnnnn",
    ".nnnnnRrmwWnnnnn.",
    "..nnnnnRmWnnnnn..",
    "...nnnnnMnnnnn...",
    "....nnnnGnnnn....",
];

#[test]
fn every_dense_position_matches_the_reference_board() {
    for (row, expected_row) in REFERENCE_ROWS.iter().enumerate() {
        assert_eq!(expected_row.chars().count(), 17, "bad fixture row {row}");
        for (column, symbol) in expected_row.chars().enumerate() {
            let coordinate = paisho_core::Coordinate::from_grid(row, column).unwrap();
            assert_eq!(
                point_type_at(coordinate),
                decode(symbol),
                "reference mismatch at {coordinate} / grid ({row},{column})"
            );
        }
    }
}

#[test]
fn playable_count_and_gates_are_exact() {
    assert_eq!(all_coordinates().count(), 289);
    assert_eq!(playable_coordinates().count(), PLAYABLE_POINT_COUNT);

    let gates: Vec<_> = all_coordinates()
        .filter(|coordinate| point_type_at(*coordinate).is_gate())
        .collect();
    assert_eq!(gates, vec![NORTH_GATE, WEST_GATE, EAST_GATE, SOUTH_GATE]);
}

#[test]
fn opposite_points_keep_the_same_regions() {
    for coordinate in all_coordinates() {
        assert_eq!(
            point_type_at(coordinate),
            point_type_at(coordinate.rotate_180())
        );
    }
}

#[test]
fn overlapping_boundaries_report_every_region() {
    let centre = point_type_at(paisho_core::Coordinate::new(0, 0).unwrap());
    assert!(centre.belongs_to(Region::Red));
    assert!(centre.belongs_to(Region::White));
    assert!(!centre.belongs_to(Region::Neutral));

    let boundary = point_type_at(paisho_core::Coordinate::new(0, 7).unwrap());
    assert!(boundary.belongs_to(Region::Red));
    assert!(boundary.belongs_to(Region::White));
    assert!(boundary.belongs_to(Region::Neutral));
}

fn decode(symbol: char) -> PointType {
    match symbol {
        '.' => PointType::NonPlayable,
        'n' => PointType::Neutral,
        'G' => PointType::Gate,
        'r' => PointType::Red,
        'w' => PointType::White,
        'm' => PointType::RedWhite,
        'R' => PointType::RedNeutral,
        'W' => PointType::WhiteNeutral,
        'M' => PointType::RedWhiteNeutral,
        other => panic!("unknown reference symbol {other}"),
    }
}
