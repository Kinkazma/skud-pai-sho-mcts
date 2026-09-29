//! At most four lookup anchors per non-overlapping 20-turn source segment:
//! each seat in each phase. The original PSR and exact segment end are retained.
use super::*;
type Active = (SequenceEntry, Player, u32, [[f64; 32]; 4], [usize; 4]);
#[cfg(test)]
pub(super) fn entries(
    record: &GameRecord,
    result: &str,
    source: &str,
    game: usize,
) -> Result<Vec<SequenceEntry>, String> {
    Ok(entries_with_geometry(record, result, source, game, false)?.0)
}
pub(super) fn entries_with_geometry(
    record: &GameRecord,
    result: &str,
    source: &str,
    game: usize,
    spatial: bool,
) -> Result<(Vec<SequenceEntry>, Vec<SequenceGeometry>), String> {
    let mut position = record.initial_position();
    let mut start = position.completed_turns();
    let mut active: Vec<Active> = vec![];
    let mut out = vec![];
    let mut geometry = vec![];
    for (i, action) in record.actions().iter().enumerate() {
        if position.completed_turns() - start >= 20 && position.phase() == TurnPhase::Main {
            flush(&mut active, &mut out, i);
            start = position.completed_turns();
        }
        let player = position.to_move();
        let phase = u8::from(position.phase() == TurnPhase::HarmonyBonus);
        if !active
            .iter()
            .any(|(e, p, _, _, _)| *p == player && e.phase == phase)
        {
            if spatial {
                geometry.push(SequenceGeometry::from_state(
                    &micro_spatial_state_features(&position),
                )?);
            }
            let outcome = if result == "draw" {
                0
            } else if result == player.code().to_string() {
                1
            } else {
                -1
            };
            active.push((
                SequenceEntry {
                    key: sequence_key(&micro_state_features(&position)),
                    patterns: [[0; 32]; 4],
                    source: sequence_source(source),
                    game: game as u32,
                    decision: i as u32,
                    end_decision: 0,
                    outcome,
                    phase,
                },
                player,
                position.completed_turns(),
                [[0.; 32]; 4],
                [0; 4],
            ));
        }
        let features = micro_action_features(&position, *action);
        for (_, p, turn, sums, counts) in &mut active {
            if *p == player {
                let bin = ((position.completed_turns() - *turn) / 5).min(3) as usize;
                counts[bin] += 1;
                for k in 0..32 {
                    sums[bin][k] += features[k];
                }
            }
        }
        position.apply(*action).map_err(|e| e.to_string())?;
    }
    flush(&mut active, &mut out, record.actions().len());
    Ok((out, geometry))
}
fn flush(active: &mut Vec<Active>, out: &mut Vec<SequenceEntry>, end: usize) {
    for (mut e, _, _, sums, counts) in active.drain(..) {
        e.end_decision = end as u32;
        finish(&mut e, sums, counts);
        out.push(e);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_game_has_both_seats_and_bonus_anchors_with_exact_bounds() {
        let old: GameRecord =
            include_str!("../../../paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let (r, p) = old
            .replay_prefix_with_rules(RuleProfileId::SkudPaiShoGen5V1)
            .unwrap();
        let result = match p.outcome() {
            GameOutcome::Win(p) => p.code().to_string(),
            _ => "draw".into(),
        };
        let rows = entries(&r, &result, "test", 0).unwrap();
        let mut seats = std::collections::HashSet::new();
        let mut bonus = 0;
        for e in rows {
            let mut p = r.initial_position();
            for a in &r.actions()[..e.decision as usize] {
                p.apply(*a).unwrap();
            }
            seats.insert(p.to_move());
            bonus += usize::from(e.phase == 1);
            assert_eq!(e.phase, u8::from(p.phase() == TurnPhase::HarmonyBonus));
            assert!(e.end_decision > e.decision && e.end_decision as usize <= r.actions().len());
            assert_eq!(e.key, sequence_key(&micro_state_features(&p)));
        }
        assert_eq!(seats.len(), 2);
        assert!(bonus > 0);
    }
}
