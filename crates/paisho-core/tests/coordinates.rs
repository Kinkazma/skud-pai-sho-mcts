use std::str::FromStr;

use paisho_core::{Coordinate, CoordinateError, BOARD_SIZE};

#[test]
fn notation_and_grid_indices_round_trip() {
    for row in 0..BOARD_SIZE {
        for column in 0..BOARD_SIZE {
            let coordinate = Coordinate::from_grid(row, column).unwrap();
            assert_eq!((coordinate.row(), coordinate.column()), (row, column));
            assert_eq!(
                Coordinate::from_str(&coordinate.to_string()),
                Ok(coordinate)
            );
        }
    }
}

#[test]
fn site_orientation_is_preserved() {
    assert_eq!(Coordinate::from_grid(0, 8).unwrap().to_string(), "0,8");
    assert_eq!(Coordinate::from_grid(8, 16).unwrap().to_string(), "8,0");
    assert_eq!(Coordinate::from_grid(16, 8).unwrap().to_string(), "0,-8");
    assert_eq!(Coordinate::from_grid(8, 0).unwrap().to_string(), "-8,0");
}

#[test]
fn malformed_and_outside_coordinates_are_rejected() {
    assert_eq!(
        Coordinate::from_str("1"),
        Err(CoordinateError::InvalidNotation)
    );
    assert_eq!(
        Coordinate::from_str("a,1"),
        Err(CoordinateError::InvalidNumber)
    );
    assert_eq!(
        Coordinate::from_str("9,-12"),
        Err(CoordinateError::OutsideEnvelope { x: 9, y: -12 })
    );
    assert_eq!(
        Coordinate::from_grid(17, 0),
        Err(CoordinateError::GridIndexOutsideEnvelope { row: 17, column: 0 })
    );
}

#[test]
fn geometric_transforms_stay_in_the_envelope() {
    let point = Coordinate::new(-5, 3).unwrap();
    assert_eq!(point.rotate_clockwise(), Coordinate::new(3, 5).unwrap());
    assert_eq!(point.rotate_180(), Coordinate::new(5, -3).unwrap());
    assert_eq!(
        point.mirror_across_vertical_axis(),
        Coordinate::new(5, 3).unwrap()
    );
}
