use paisho_core::{
    all_coordinates, harmonies, is_drained, is_trapped, legal_actions, orthogonal_neighbors,
    point_type_at, Action, ApplyError, BasicFlower, Board, Coordinate, Harmony, LineOrientation,
    Player, PointType, Position, TileKind, TurnPhase, CELL_COUNT,
};

use crate::{Agent, AgentError, AgentTelemetry, StableRng};

pub const SITE_BOT_V1_SOURCE_COMMIT: &str = "b849dbdabb1138ff0f6d609adf38b301c2f875ae";
pub const SITE_BOT_V1_WIN_SCORE: i32 = 9_999_999;

/// Behavioral reimplementation of the weak automatic opponent shipped by
/// The Garden Gate. It keeps decision-relevant quirks but uses the local,
/// deterministic rules engine and seeded randomness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SiteBotV1 {
    rng: StableRng,
    planned_bonus: Option<Action>,
    decisions: usize,
    evaluated_actions: usize,
}

impl SiteBotV1 {
    pub const fn new(seed: u64) -> Self {
        Self {
            rng: StableRng::new(seed),
            planned_bonus: None,
            decisions: 0,
            evaluated_actions: 0,
        }
    }

    fn select_main_action(&mut self, position: &Position, actions: &[Action]) -> usize {
        self.planned_bonus = None;
        let compatible = site_ordered_indices(position, actions);
        if compatible.is_empty() {
            // The pinned JavaScript returns `undefined` in this case. An
            // out-of-range index lets matchplay report the incompatibility
            // instead of inventing a legal fallback move.
            let _ = self.rng.next_f64();
            return actions.len();
        }

        let player = position.to_move();
        let mut best: Option<(usize, i32)> = None;
        for index in compatible.iter().copied() {
            let score = site_v1_action_score(position, actions[index])
                .expect("the site bot scores only engine-generated actions");
            self.evaluated_actions += 1;
            if score > 9_999 {
                return index;
            }

            if self.rng.next_f64() > 0.1
                && best
                    .map(|(_, best_score)| best_score == 0 || score > best_score)
                    .unwrap_or(true)
            {
                best = Some((index, score));
            }
        }

        if let Some((index, _)) = best {
            self.plan_basic_bonus(position, actions[index], actions, player);
            index
        } else {
            compatible[self.site_index(compatible.len())]
        }
    }

    fn select_bonus_action(&mut self, actions: &[Action]) -> usize {
        if let Some(planned) = self.planned_bonus.take() {
            if let Some(index) = actions.iter().position(|action| *action == planned) {
                return index;
            }
        }
        actions
            .iter()
            .position(|action| *action == Action::SkipHarmonyBonus)
            .unwrap_or(0)
    }

    fn plan_basic_bonus(
        &mut self,
        position: &Position,
        selected: Action,
        actions: &[Action],
        player: Player,
    ) {
        if !matches!(selected, Action::Arrange { .. })
            || player_has_growing_flower(position, player)
        {
            return;
        }

        let plants: Vec<_> = site_ordered_indices(position, actions)
            .into_iter()
            .filter_map(|index| match actions[index] {
                Action::Plant { flower, gate } => Some((flower, gate)),
                _ => None,
            })
            .collect();
        let random_value = self.rng.next_f64();
        if plants.is_empty() {
            return;
        }
        let (flower, _) = plants[index_from_unit(random_value, plants.len())];

        let mut gates = Vec::new();
        for (_, gate) in &plants {
            if !gates.contains(gate) {
                gates.push(*gate);
            }
        }
        let gate = gates[self.site_index(gates.len())];
        self.planned_bonus = Some(Action::BonusPlantBasic { flower, gate });
    }

    fn site_index(&mut self, length: usize) -> usize {
        assert!(
            length > 0,
            "site-style random choice needs a non-empty list"
        );
        index_from_unit(self.rng.next_f64(), length)
    }
}

impl Agent for SiteBotV1 {
    fn select_action(
        &mut self,
        position: &Position,
        legal_actions: &[Action],
    ) -> Result<usize, AgentError> {
        self.decisions += 1;
        Ok(match position.phase() {
            TurnPhase::Main => self.select_main_action(position, legal_actions),
            TurnPhase::HarmonyBonus => self.select_bonus_action(legal_actions),
        })
    }

    fn telemetry(&self) -> AgentTelemetry {
        AgentTelemetry {
            decisions: self.decisions,
            simulations: 0,
            evaluated_actions: self.evaluated_actions,
            ..AgentTelemetry::default()
        }
    }

    fn reset_telemetry(&mut self) {
        self.decisions = 0;
        self.evaluated_actions = 0;
    }
}

pub fn site_v1_action_score(position: &Position, action: Action) -> Result<i32, ApplyError> {
    let player = position.to_move();
    let mut candidate = position.clone();
    candidate.apply(action)?;
    Ok(site_v1_transition_score(position, &candidate, player))
}

/// Returns the engine-legal main actions in the iteration order used by the
/// pinned JavaScript opponent. The order matters because its 10% noise is
/// sampled once per candidate.
pub fn site_v1_main_actions(position: &Position) -> Vec<Action> {
    let actions = legal_actions(position);
    site_ordered_indices(position, &actions)
        .into_iter()
        .map(|index| actions[index])
        .collect()
}

pub fn site_v1_transition_score(before: &Position, after: &Position, player: Player) -> i32 {
    if player_is_winner(before, player) || player_is_winner(after, player) {
        return SITE_BOT_V1_WIN_SCORE;
    }

    let opponent = player.opponent();
    let before_features = SiteV1Features::from_position(before);
    let after_features = SiteV1Features::from_position(after);
    let mut score = 0;

    score += change_score(
        before_features.harmony_count(player),
        after_features.harmony_count(player),
        20,
        -20,
    );
    score += change_score(
        before_features.center_crossing_count(player),
        after_features.center_crossing_count(player),
        50,
        -20,
    );
    score += change_score(
        before_features.harmony_count(opponent),
        after_features.harmony_count(opponent),
        -7,
        3,
    );
    score += change_score(
        before_features.matching_garden_tiles,
        after_features.matching_garden_tiles,
        10,
        -10,
    );
    if after_features.tile_count(opponent) < before_features.tile_count(opponent) {
        score += 8;
    }

    let before_surroundness = before_features.surroundness(player);
    let after_surroundness = after_features.surroundness(player);
    if after_surroundness > before_surroundness {
        score += after_surroundness as i32;
    }
    score += site_v1_cycle_progress_score_with_surroundness(
        before.board(),
        after.board(),
        player,
        after_surroundness,
    );
    if after_features.tile_count(player) > before_features.tile_count(player) {
        score += 5;
    }
    score
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SiteV1Features {
    harmonies: [usize; 2],
    center_crossings: [usize; 2],
    tiles: [usize; 2],
    surroundness: [usize; 2],
    pub matching_garden_tiles: usize,
}

impl SiteV1Features {
    pub fn from_position(position: &Position) -> Self {
        let board = position.board();
        let board_harmonies = harmonies(board);
        let mut features = Self {
            harmonies: [0; 2],
            center_crossings: [0; 2],
            tiles: [0; 2],
            surroundness: [0; 2],
            matching_garden_tiles: matching_garden_tile_count(board),
        };
        for player in [Player::Host, Player::Guest] {
            features.harmonies[player.index()] = board_harmonies
                .iter()
                .filter(|harmony| harmony.owner == player)
                .count();
            features.center_crossings[player.index()] = board_harmonies
                .iter()
                .filter(|harmony| harmony.owner == player && crosses_center(harmony))
                .count();
            features.tiles[player.index()] = board
                .occupied()
                .filter(|(_, tile)| tile.owner == player)
                .count();
            features.surroundness[player.index()] = surroundness(board, player);
        }
        features
    }

    pub const fn harmony_count(self, player: Player) -> usize {
        self.harmonies[player.index()]
    }

    pub const fn center_crossing_count(self, player: Player) -> usize {
        self.center_crossings[player.index()]
    }

    pub const fn tile_count(self, player: Player) -> usize {
        self.tiles[player.index()]
    }

    pub const fn surroundness(self, player: Player) -> usize {
        self.surroundness[player.index()]
    }
}

fn site_ordered_indices(position: &Position, actions: &[Action]) -> Vec<usize> {
    let mut indices: Vec<_> = actions
        .iter()
        .enumerate()
        .filter(|(_, action)| matches!(action, Action::Plant { .. }))
        .map(|(index, _)| index)
        .collect();
    indices.sort_by_key(|index| {
        let Action::Plant { flower, gate } = actions[*index] else {
            unreachable!("Plant filter and ordering stay in lockstep")
        };
        let remaining = position
            .reserve(position.to_move())
            .count(TileKind::Basic(flower));
        let consumed = 3_u8.saturating_sub(remaining);
        (consumed as usize, flower as usize, gate.dense_index())
    });

    let mut distance_remaining = [None; CELL_COUNT];
    for (from, tile) in position.board().occupied() {
        if tile.owner != position.to_move()
            || !tile.kind.is_flower()
            || is_drained(position.board(), from)
            || is_trapped(position.board(), from)
        {
            continue;
        }
        let Some(movement) = tile.kind.movement() else {
            continue;
        };
        let mut action_by_destination = [None; CELL_COUNT];
        for (index, action) in actions.iter().enumerate() {
            if let Action::Arrange { from: start, to } = action {
                if *start == from {
                    action_by_destination[to.dense_index()] = Some(index);
                }
            }
        }
        let destinations = site_generated_destinations(
            position.board(),
            from,
            movement,
            &action_by_destination,
            &mut distance_remaining,
        );
        let generated_any = !destinations.is_empty();
        for to in &destinations {
            indices.push(
                action_by_destination[to.dense_index()]
                    .expect("source-compatible destination is engine legal"),
            );
        }

        // The source clears these traversal marks only inside its endpoint
        // loop. A flower with no generated move therefore affects the next
        // flower's traversal; this omission is observable in its policy.
        if generated_any {
            distance_remaining.fill(None);
        }
    }
    indices
}

fn site_generated_destinations(
    board: &Board,
    from: Coordinate,
    movement: u8,
    action_by_destination: &[Option<usize>; CELL_COUNT],
    distance_remaining: &mut [Option<u8>; CELL_COUNT],
) -> Vec<Coordinate> {
    let mut generated = [false; CELL_COUNT];
    let mut frontier = vec![from];
    let mut remaining = movement;

    while remaining > 0 && !frontier.is_empty() {
        let mut next_frontier = Vec::new();
        for current in frontier {
            for adjacent in orthogonal_neighbors(current) {
                let index = adjacent.dense_index();
                if distance_remaining[index].is_some_and(|known| known >= remaining) {
                    continue;
                }
                distance_remaining[index] = Some(remaining);
                let is_empty = board.is_empty(adjacent);
                if !is_empty {
                    distance_remaining[index] = Some(0);
                }
                if action_by_destination[index].is_some() {
                    generated[index] = true;
                }
                if is_empty {
                    next_frontier.push(adjacent);
                }
            }
        }
        frontier = next_frontier;
        remaining -= 1;
    }

    all_coordinates()
        .filter(|coordinate| generated[coordinate.dense_index()])
        .collect()
}

fn player_has_growing_flower(position: &Position, player: Player) -> bool {
    position.board().occupied().any(|(coordinate, tile)| {
        tile.owner == player && tile.kind.is_flower() && point_type_at(coordinate).is_gate()
    })
}

fn player_is_winner(position: &Position, player: Player) -> bool {
    paisho_core::harmony_ring_owners_for_profile(position.board(), position.rule_profile())
        .contains(&player)
}

const fn change_score(before: usize, after: usize, increase: i32, decrease: i32) -> i32 {
    if after > before {
        increase
    } else if after < before {
        decrease
    } else {
        0
    }
}

fn matching_garden_tile_count(board: &Board) -> usize {
    board
        .occupied()
        .filter(|(coordinate, tile)| {
            matches!(
                (point_type_at(*coordinate), tile.kind),
                (
                    PointType::Red,
                    TileKind::Basic(BasicFlower::Red3 | BasicFlower::Red4 | BasicFlower::Red5)
                ) | (
                    PointType::White,
                    TileKind::Basic(
                        BasicFlower::White3 | BasicFlower::White4 | BasicFlower::White5
                    )
                )
            )
        })
        .count()
}

fn crosses_center(harmony: &Harmony) -> bool {
    match harmony.orientation {
        LineOrientation::Horizontal => signs_straddle_zero(harmony.first.x(), harmony.second.x()),
        LineOrientation::Vertical => signs_straddle_zero(harmony.first.y(), harmony.second.y()),
    }
}

const fn signs_straddle_zero(first: i8, second: i8) -> bool {
    (first < 0 && second > 0) || (first > 0 && second < 0)
}

fn surroundness(board: &Board, player: Player) -> usize {
    let mut directions = [0_usize; 4];
    for (coordinate, tile) in board.occupied() {
        if tile.owner != player {
            continue;
        }
        if coordinate.y() > 0 {
            directions[0] += 1;
        }
        if coordinate.y() < 0 {
            directions[1] += 1;
        }
        if coordinate.x() < 0 {
            directions[2] += 1;
        }
        if coordinate.x() > 0 {
            directions[3] += 1;
        }
    }
    let minimum = directions.iter().copied().min().unwrap_or(0);
    if minimum == 0 {
        directions.iter().filter(|count| **count > 0).count()
    } else {
        minimum * 4
    }
}

pub fn site_v1_cycle_length(board: &Board, player: Player) -> usize {
    site_v1_cycle_length_from_harmonies(&harmonies(board), player)
}

/// Returns only the source bot's cycle-length contribution to a transition.
/// Cycle enumeration is deliberately deferred until surroundness can activate it.
pub fn site_v1_cycle_progress_score(before: &Board, after: &Board, player: Player) -> i32 {
    site_v1_cycle_progress_score_with_surroundness(
        before,
        after,
        player,
        surroundness(after, player),
    )
}

fn site_v1_cycle_progress_score_with_surroundness(
    before: &Board,
    after: &Board,
    player: Player,
    after_surroundness: usize,
) -> i32 {
    if after_surroundness <= 3 {
        return 0;
    }
    let before_cycle = site_v1_cycle_length(before, player);
    if before_cycle >= 5 {
        return 5;
    }
    match site_v1_cycle_length(after, player).cmp(&before_cycle) {
        std::cmp::Ordering::Greater => 5,
        std::cmp::Ordering::Less => -2,
        std::cmp::Ordering::Equal => 0,
    }
}

fn site_v1_cycle_length_from_harmonies(harmonies: &[Harmony], player: Player) -> usize {
    site_harmony_cycles(harmonies)
        .into_iter()
        .filter_map(|cycle| {
            let closing_edge = *cycle.last()?;
            (harmonies[closing_edge].owner == player).then_some(cycle.len() - 1)
        })
        .max()
        .unwrap_or(0)
}

/// Ports the source bot's deliberately order-sensitive `getHarmonyChains` /
/// `ringLengthForPlayer` behavior. Real Ring ownership remains in paisho-core.
fn site_harmony_cycles(harmonies: &[Harmony]) -> Vec<Vec<usize>> {
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for (index, harmony) in harmonies.iter().enumerate() {
        let chain = vec![index];
        let mut found = Vec::new();
        find_site_harmony_cycles(harmonies, harmony.second, harmony.first, &chain, &mut found);
        for cycle in found {
            if !cycles
                .iter()
                .any(|existing| same_harmony_cycle(existing, &cycle))
            {
                cycles.push(cycle);
            }
        }
    }
    cycles
}

fn find_site_harmony_cycles(
    harmonies: &[Harmony],
    current: Coordinate,
    target: Coordinate,
    original_chain: &[usize],
    found: &mut Vec<Vec<usize>>,
) {
    let mut continuations = Vec::new();
    for (index, harmony) in harmonies.iter().enumerate() {
        if original_chain.contains(&index) || !harmony_contains(*harmony, current) {
            continue;
        }
        let mut chain = original_chain.to_vec();
        chain.push(index);
        if harmony_contains(*harmony, target) {
            found.push(chain);
        } else {
            continuations.push((index, chain));
        }
    }
    for (index, chain) in continuations {
        let next = other_harmony_endpoint(harmonies[index], current);
        find_site_harmony_cycles(harmonies, next, target, &chain, found);
    }
}

fn harmony_contains(harmony: Harmony, coordinate: Coordinate) -> bool {
    harmony.first == coordinate || harmony.second == coordinate
}

fn other_harmony_endpoint(harmony: Harmony, coordinate: Coordinate) -> Coordinate {
    if harmony.first == coordinate {
        harmony.second
    } else {
        debug_assert_eq!(harmony.second, coordinate);
        harmony.first
    }
}

fn same_harmony_cycle(left: &[usize], right: &[usize]) -> bool {
    left.len() == right.len() && left.iter().all(|edge| right.contains(edge))
}

fn index_from_unit(random: f64, length: usize) -> usize {
    ((random * length as f64) as usize).min(length - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use paisho_core::{legal_actions, StandardSetup, NORTH_GATE, SOUTH_GATE};

    fn apply_turn(position: &mut Position, action: Action) {
        position.apply(action).unwrap();
        if position.phase() == TurnPhase::HarmonyBonus {
            position.apply(Action::SkipHarmonyBonus).unwrap();
        }
    }

    #[test]
    fn initial_gate_plant_has_the_reference_score() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let score = site_v1_action_score(
            &position,
            Action::Plant {
                flower: BasicFlower::White3,
                gate: paisho_core::WEST_GATE,
            },
        )
        .unwrap();
        assert_eq!(score, 7);
    }

    #[test]
    fn site_style_bonus_is_planned_only_when_no_flower_was_growing() {
        let mut position =
            Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        apply_turn(
            &mut position,
            Action::Arrange {
                from: SOUTH_GATE,
                to: Coordinate::new(0, -5).unwrap(),
            },
        );
        apply_turn(
            &mut position,
            Action::Arrange {
                from: NORTH_GATE,
                to: Coordinate::new(0, 5).unwrap(),
            },
        );
        apply_turn(
            &mut position,
            Action::Plant {
                flower: BasicFlower::Red4,
                gate: SOUTH_GATE,
            },
        );
        apply_turn(
            &mut position,
            Action::Plant {
                flower: BasicFlower::Red4,
                gate: NORTH_GATE,
            },
        );
        apply_turn(
            &mut position,
            Action::Arrange {
                from: SOUTH_GATE,
                to: Coordinate::new(-1, -5).unwrap(),
            },
        );
        apply_turn(
            &mut position,
            Action::Arrange {
                from: NORTH_GATE,
                to: Coordinate::new(1, 5).unwrap(),
            },
        );
        apply_turn(
            &mut position,
            Action::Arrange {
                from: Coordinate::new(-1, -5).unwrap(),
                to: Coordinate::new(-1, -4).unwrap(),
            },
        );
        apply_turn(
            &mut position,
            Action::Arrange {
                from: Coordinate::new(1, 5).unwrap(),
                to: Coordinate::new(1, 4).unwrap(),
            },
        );

        let arrangement = Action::Arrange {
            from: Coordinate::new(-1, -4).unwrap(),
            to: Coordinate::new(-1, -5).unwrap(),
        };
        let main_actions = legal_actions(&position);
        let mut bot = SiteBotV1::new(3);
        bot.plan_basic_bonus(&position, arrangement, &main_actions, Player::Guest);
        assert!(matches!(
            bot.planned_bonus,
            Some(Action::BonusPlantBasic { .. })
        ));

        position.apply(arrangement).unwrap();
        assert_eq!(position.phase(), TurnPhase::HarmonyBonus);
        let bonuses = legal_actions(&position);
        let selected = bot.select_bonus_action(&bonuses);
        assert!(matches!(bonuses[selected], Action::BonusPlantBasic { .. }));
    }

    #[test]
    fn empty_source_candidate_list_still_consumes_its_fallback_draw() {
        let position = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        let seed = 0x0045_4d50_5459;
        let mut bot = SiteBotV1::new(seed);
        assert_eq!(bot.select_main_action(&position, &[]), 0);

        let mut expected_rng = StableRng::new(seed);
        let _ = expected_rng.next_f64();
        assert_eq!(bot.rng, expected_rng);
    }
}
