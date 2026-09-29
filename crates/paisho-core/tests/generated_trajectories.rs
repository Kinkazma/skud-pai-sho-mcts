use paisho_core::{
    legal_actions, legal_arrangement_destinations, reachable_points, validate_arrangement, Action,
    BasicFlower, GameOutcome, GameRecord, Position, StandardSetup, TurnPhase, BASIC_FLOWERS,
};

#[derive(Clone, Copy)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn index(&mut self, upper_bound: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 as usize % upper_bound
    }
}

#[test]
fn generated_legal_trajectories_replay_exactly() {
    for seed in 1_u64..=12 {
        let starting_flower: BasicFlower = BASIC_FLOWERS[seed as usize % BASIC_FLOWERS.len()];
        let setup = StandardSetup::balanced(starting_flower);
        let mut position = Position::from_standard_setup(setup);
        let mut record = GameRecord::new(setup);
        let mut rng = DeterministicRng(seed * 0x9e37_79b9);

        for _ in 0..128 {
            if position.outcome() != GameOutcome::Ongoing {
                break;
            }
            let actions = legal_actions(&position);
            assert!(
                !actions.is_empty(),
                "seed {seed} reached an ongoing dead end"
            );
            let action = actions[rng.index(actions.len())];
            position
                .apply(action)
                .unwrap_or_else(|error| panic!("seed {seed}: generated {action:?}: {error}"));
            record.push(action);
        }

        let encoded = record.to_string();
        let decoded: GameRecord = encoded.parse().unwrap();
        assert_eq!(decoded.replay().unwrap(), position, "seed {seed}");
    }
}

#[test]
fn bulk_arrangement_generation_matches_the_reference_validator() {
    for seed in 1_u64..=6 {
        let starting_flower: BasicFlower = BASIC_FLOWERS[seed as usize % BASIC_FLOWERS.len()];
        let setup = StandardSetup::balanced(starting_flower);
        let mut position = Position::from_standard_setup(setup);
        let mut rng = DeterministicRng(seed * 0x517c_c1b7);

        for decision in 0..96 {
            if position.outcome() != GameOutcome::Ongoing {
                break;
            }

            if position.phase() == TurnPhase::Main {
                let player = position.to_move();
                let mut reference_actions = Vec::new();
                for (from, tile) in position.board().occupied() {
                    if tile.owner != player || !tile.kind.is_flower() {
                        continue;
                    }
                    let maximum = tile
                        .kind
                        .movement()
                        .expect("every Flower has a movement allowance");
                    let reference_destinations: Vec<_> =
                        reachable_points(position.board(), from, maximum)
                            .into_iter()
                            .filter(|to| {
                                validate_arrangement(position.board(), player, from, *to).is_ok()
                            })
                            .collect();
                    let bulk_destinations =
                        legal_arrangement_destinations(position.board(), player, from);
                    assert_eq!(
                        bulk_destinations, reference_destinations,
                        "seed {seed}, decision {decision}, source {from}"
                    );
                    reference_actions.extend(
                        reference_destinations
                            .into_iter()
                            .map(|to| Action::Arrange { from, to }),
                    );
                }

                let generated_actions = legal_actions(&position);
                let generated_arrangements: Vec<_> = generated_actions
                    .iter()
                    .copied()
                    .filter(|action| matches!(action, Action::Arrange { .. }))
                    .collect();
                assert_eq!(
                    generated_arrangements, reference_actions,
                    "seed {seed}, decision {decision}"
                );
            }

            let actions = legal_actions(&position);
            assert!(
                !actions.is_empty(),
                "seed {seed}, decision {decision} reached an ongoing dead end"
            );
            let action = actions[rng.index(actions.len())];
            position.apply(action).unwrap_or_else(|error| {
                panic!("seed {seed}, decision {decision}: generated {action:?}: {error}")
            });
        }
    }
}
