//! R3/R4 action consequences. Labels never enter current-state inference.
use super::{HarmonyCycleGeometry, MicroRelations};
use paisho_core::*;
use serde::{Deserialize, Serialize};

pub const MICRO_STRUCTURED_COUNTS: usize = 10;
pub const MICRO_STRUCTURED_EVENTS: usize = 10;
pub const MICRO_STRUCTURED_SCHEMA: &str = "paisho-action-consequences-10-10-v1";

/// Event class weights over the actual minibatch, before its mean reduction.
/// Balancing each position separately lets negative-only positions overwhelm
/// rare positives elsewhere in the same batch. Unknown labels contribute zero.
#[derive(Clone, Debug)]
pub struct MicroStructuredBatchBalance {
    weights: [[f64; 2]; MICRO_STRUCTURED_EVENTS],
}
impl MicroStructuredBatchBalance {
    pub fn new<'a>(examples: impl IntoIterator<Item = &'a super::MicroExample>) -> Self {
        let mut counts = [[0usize; 2]; MICRO_STRUCTURED_EVENTS];
        let mut annotated = 0;
        for ex in examples {
            // Preserve the original 0.2 budget per annotated example; old
            // unlabelled replay rows must not amplify the auxiliary loss.
            annotated += usize::from(
                ex.structured
                    .iter()
                    .flatten()
                    .any(|t| t.events.iter().any(Option::is_some)),
            );
            for target in ex.structured.iter().flatten() {
                for (j, value) in target.events.iter().enumerate() {
                    if let Some(value) = value {
                        counts[j][usize::from(*value)] += 1;
                    }
                }
            }
        }
        let active = counts.iter().filter(|c| c[0] + c[1] > 0).count();
        let weights = std::array::from_fn(|j| {
            let classes = counts[j].iter().filter(|n| **n > 0).count();
            std::array::from_fn(|class| {
                if counts[j][class] == 0 {
                    0.
                } else {
                    annotated as f64 / (active * classes * counts[j][class]) as f64
                }
            })
        });
        Self { weights }
    }
    pub(super) fn weight(&self, event: usize, class: bool) -> f64 {
        self.weights[event][usize::from(class)]
    }
}
pub const MICRO_STRUCTURED_COUNT_NAMES: [&str; 10] = [
    "own_created",
    "own_removed",
    "opponent_created",
    "opponent_removed",
    "own_midline_delta",
    "opponent_midline_delta",
    "own_cycle_rank_delta",
    "opponent_cycle_rank_delta",
    "own_component_delta",
    "opponent_component_delta",
];
pub const MICRO_STRUCTURED_EVENT_NAMES: [&str; 10] = [
    "own_ring",
    "opponent_ring",
    "exhaustion_win",
    "exhaustion_draw",
    "exhaustion_loss",
    "opponent_immediate_win",
    "own_offcentre_basis",
    "opponent_offcentre_basis",
    "own_touching_basis",
    "opponent_touching_basis",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MicroStructuredTarget {
    pub counts: [f64; MICRO_STRUCTURED_COUNTS],
    pub events: [Option<bool>; MICRO_STRUCTURED_EVENTS],
    #[serde(default)]
    pub threat: MicroThreatEvidence,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum MicroThreatEvidence {
    Present {
        reply: String,
    },
    Absent {
        examined: usize,
    },
    #[default]
    Unknown,
    SamePlayerBonus,
    Terminal,
}

/// Scope is one opponent decision, never an inferred multi-ply minimax loss.
#[derive(Clone, Debug, PartialEq)]
pub enum MicroImmediateThreat {
    Present {
        witness: Action,
        examined: usize,
    },
    Absent {
        examined: usize,
    },
    Unknown {
        examined: usize,
        legal: Option<usize>,
    },
    SamePlayerBonus,
    Terminal,
}
impl MicroImmediateThreat {
    pub fn label(&self) -> Option<bool> {
        match self {
            Self::Present { .. } => Some(true),
            Self::Absent { .. } => Some(false),
            _ => None,
        }
    }
}

/// A bounded audit: zero budget yields unknown, not absent. Stops on a proof.
pub fn micro_immediate_threat(
    p: &Position,
    original_mover: Player,
    limit: usize,
) -> Result<MicroImmediateThreat, String> {
    if p.outcome() != GameOutcome::Ongoing {
        return Ok(MicroImmediateThreat::Terminal);
    }
    if p.to_move() == original_mover {
        return Ok(MicroImmediateThreat::SamePlayerBonus);
    }
    if limit == 0 {
        return Ok(MicroImmediateThreat::Unknown {
            examined: 0,
            legal: None,
        });
    }
    let actions = legal_actions(p);
    for (i, &a) in actions.iter().take(limit).enumerate() {
        let mut q = p.clone();
        q.apply(a).map_err(|e| e.to_string())?;
        if q.outcome() == GameOutcome::Win(original_mover.opponent()) {
            return Ok(MicroImmediateThreat::Present {
                witness: a,
                examined: i + 1,
            });
        }
    }
    Ok(if limit >= actions.len() {
        MicroImmediateThreat::Absent {
            examined: actions.len(),
        }
    } else {
        MicroImmediateThreat::Unknown {
            examined: limit,
            legal: Some(actions.len()),
        }
    })
}

/// Recheck a retained witness without scanning any other reply.
pub fn micro_verify_threat_witness(
    p: &Position,
    mover: Player,
    witness: Action,
) -> Result<MicroImmediateThreat, String> {
    if p.outcome() != GameOutcome::Ongoing {
        return Ok(MicroImmediateThreat::Terminal);
    }
    if p.to_move() == mover {
        return Ok(MicroImmediateThreat::SamePlayerBonus);
    }
    let mut q = p.clone();
    q.apply(witness).map_err(|e| e.to_string())?;
    if q.outcome() != GameOutcome::Win(mover.opponent()) {
        return Err("retained reply is not an immediate regulatory win".into());
    }
    Ok(MicroImmediateThreat::Present {
        witness,
        examined: 1,
    })
}

fn components(p: &Position, r: &MicroRelations, owner: Player) -> usize {
    let mut parent: Vec<usize> = (0..CELL_COUNT).collect();
    let mut present = [false; CELL_COUNT];
    fn root(parent: &[usize], mut i: usize) -> usize {
        while parent[i] != i {
            i = parent[i];
        }
        i
    }
    for (at, t) in p.board().occupied() {
        if t.owner == owner {
            present[at.dense_index()] = true;
        }
    }
    for e in r.edges.iter().filter(|e| e.owner == owner) {
        let (a, b) = (e.first.dense_index(), e.second.dense_index());
        present[a] = true;
        present[b] = true;
        let (ra, rb) = (root(&parent, a), root(&parent, b));
        parent[ra] = rb;
    }
    (0..CELL_COUNT)
        .filter(|&i| present[i] && root(&parent, i) == i)
        .count()
}

impl MicroStructuredTarget {
    /// Same position/action: exact consequences must agree, unknown cannot erase a proof.
    pub fn merge_evidence(&self, newer: &Self) -> Result<Self, String> {
        self.validate()?;
        newer.validate()?;
        if self.counts != newer.counts
            || (0..10).any(|i| i != 5 && self.events[i] != newer.events[i])
        {
            return Err("conflicting exact action consequences".into());
        }
        if let (Some(a), Some(b)) = (self.events[5], newer.events[5]) {
            if a != b {
                return Err("conflicting immediate threat evidence".into());
            }
        }
        let mut out = newer.clone();
        if self.events[5].is_some() && newer.events[5].is_none() {
            out.events[5] = self.events[5];
            out.threat = self.threat.clone();
        }
        Ok(out)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.counts.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite structured target".into());
        }
        let label = match &self.threat {
            MicroThreatEvidence::Present { reply } => {
                reply.parse::<Action>().map_err(|e| e.to_string())?;
                Some(true)
            }
            MicroThreatEvidence::Absent { .. } => Some(false),
            _ => None,
        };
        if self.events[5] != label {
            return Err("threat label lacks matching immediate evidence".into());
        }
        if self.events[2..5]
            .iter()
            .filter(|v| **v == Some(true))
            .count()
            > 1
        {
            return Err("conflicting exhaustion outcomes".into());
        }
        Ok(())
    }
    /// `before` may be reused for every already-constructed successor at this root.
    /// Caller supplies the actual legal successor, never a speculative board.
    pub fn from_successor(
        p: &Position,
        action: Action,
        q: &Position,
        before: &MicroRelations,
        threat: &MicroImmediateThreat,
    ) -> Self {
        let mover = p.to_move();
        let after = MicroRelations::extract(q, mover);
        let mut counts = [0.; MICRO_STRUCTURED_COUNTS];
        let mut events = [None; MICRO_STRUCTURED_EVENTS];
        for (seat, owner) in [mover, mover.opponent()].into_iter().enumerate() {
            counts[2 * seat] = after
                .edges
                .iter()
                .filter(|e| e.owner == owner && !before.edges.contains(e))
                .count() as f64
                / 8.;
            counts[2 * seat + 1] = before
                .edges
                .iter()
                .filter(|e| e.owner == owner && !after.edges.contains(e))
                .count() as f64
                / 8.;
            counts[4 + seat] = 2. * (after.global[9 * seat + 1] - before.global[9 * seat + 1]);
            counts[6 + seat] = 2. * (after.global[9 * seat + 2] - before.global[9 * seat + 2]);
            counts[8 + seat] =
                (components(q, &after, owner) as f64 - components(p, before, owner) as f64) / 8.;
            events[seat] = Some(after.global[9 * seat + 3] != 0.);
            events[6 + seat] = Some(
                after
                    .cycles
                    .iter()
                    .any(|c| c.owner == owner && c.geometry == HarmonyCycleGeometry::OffCentre),
            );
            events[8 + seat] =
                Some(after.cycles.iter().any(|c| {
                    c.owner == owner && c.geometry == HarmonyCycleGeometry::TouchingCentre
                }));
        }
        let exhausted = matches!(
            action,
            Action::Plant { .. } | Action::BonusPlantBasic { .. }
        ) && q.reserve(mover).basic_count() == 0
            && q.outcome() != GameOutcome::Ongoing;
        events[2] = Some(exhausted && q.outcome() == GameOutcome::Win(mover));
        events[3] = Some(exhausted && q.outcome() == GameOutcome::Draw);
        events[4] = Some(exhausted && q.outcome() == GameOutcome::Win(mover.opponent()));
        events[5] = threat.label();
        let threat = match threat {
            MicroImmediateThreat::Present { witness, .. } => MicroThreatEvidence::Present {
                reply: witness.to_string(),
            },
            MicroImmediateThreat::Absent { examined } => MicroThreatEvidence::Absent {
                examined: *examined,
            },
            MicroImmediateThreat::Unknown { .. } => MicroThreatEvidence::Unknown,
            MicroImmediateThreat::SamePlayerBonus => MicroThreatEvidence::SamePlayerBonus,
            MicroImmediateThreat::Terminal => MicroThreatEvidence::Terminal,
        };
        Self {
            counts,
            events,
            threat,
        }
    }
    pub fn rare_events(&self) -> bool {
        self.events.iter().any(|x| *x == Some(true))
    }
    pub fn record_threat(&mut self, proof: &MicroImmediateThreat) {
        if let MicroImmediateThreat::Present { witness, .. } = proof {
            self.events[5] = Some(true);
            self.threat = MicroThreatEvidence::Present {
                reply: witness.to_string(),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_replies_unknown_and_bonus_are_not_negative_labels() {
        let r: GameRecord = include_str!("../../tests/fixtures/site_bot_v1_ring_finish.psr")
            .parse()
            .unwrap();
        let mut p = r.initial_position();
        for &a in &r.actions()[..r.actions().len() - 1] {
            p.apply(a).unwrap();
        }
        let mover = p.to_move().opponent();
        assert!(matches!(
            micro_immediate_threat(&p, mover, 0).unwrap(),
            MicroImmediateThreat::Unknown { .. }
        ));
        assert_eq!(micro_immediate_threat(&p, mover, 0).unwrap().label(), None);
        let present = micro_immediate_threat(&p, mover, usize::MAX).unwrap();
        assert_eq!(present.label(), Some(true));
        if let MicroImmediateThreat::Present { witness, .. } = present {
            assert_eq!(
                micro_verify_threat_witness(&p, mover, witness)
                    .unwrap()
                    .label(),
                Some(true)
            );
        }
        assert_eq!(
            micro_immediate_threat(&p, p.to_move(), 0).unwrap(),
            MicroImmediateThreat::SamePlayerBonus
        );
        p.apply(*r.actions().last().unwrap()).unwrap();
        assert_eq!(
            micro_immediate_threat(&p, mover, usize::MAX).unwrap(),
            MicroImmediateThreat::Terminal
        );
    }
    #[test]
    fn complete_nonwinning_census_alone_proves_absence() {
        let r: GameRecord = include_str!("../../tests/fixtures/site_bot_v1_ring_finish.psr")
            .parse()
            .unwrap();
        let p = r.initial_position();
        let mover = p.to_move().opponent();
        assert_eq!(
            micro_immediate_threat(&p, mover, usize::MAX)
                .unwrap()
                .label(),
            Some(false)
        );
    }
}
