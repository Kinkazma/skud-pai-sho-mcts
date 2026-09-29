//! Human cases: three strata, exact prefixes, and explicit alternating outcomes.
use super::*;
use crate::compact_selfplay::reuse::Repetitions;
use paisho_core::Position;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CaseOptions {
    pub archive: PathBuf,
    pub reversals: usize,
    pub losses: usize,
    pub unresolved_attempts: usize,
    pub reanalysis_positions: usize,
    pub reanalysis_budget: usize,
    pub durable_fraction: f64,
}
impl Default for CaseOptions {
    fn default() -> Self {
        Self {
            archive: PathBuf::new(),
            reversals: 3,
            losses: 10,
            unresolved_attempts: 10,
            reanalysis_positions: 4,
            reanalysis_budget: 512,
            durable_fraction: 0.1,
        }
    }
}
#[derive(Clone)]
pub(super) struct Case {
    pub identity: String,
    pub source: String,
    pub zone: usize,
    pub record: GameRecord,
}
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub(super) struct State {
    #[serde(default)]
    pub legacy_budget: Option<usize>,
    #[serde(default)]
    pub pending: Vec<usize>,
    #[serde(default)]
    pub pending_record: Option<(PathBuf, String)>,
    pub ticket: usize,
    pub attempts: usize,
    pub alternatives: Vec<String>,
    pub reversals: usize,
    pub losses: usize,
    pub unresolved: usize,
    pub focus: Option<String>,
    pub rotate_reason: Option<String>,
}
impl State {
    pub fn observe(&mut self, outcome: GameOutcome, trajectory: &str, o: &CaseOptions) {
        self.attempts += 1;
        if !self.alternatives.iter().any(|x| x == trajectory) {
            self.alternatives.push(trajectory.into());
        }
        match outcome {
            GameOutcome::Win(winner) => {
                let winner = winner.code().to_string();
                if self.focus.as_ref() == Some(&winner) {
                    self.reversals += 1;
                    self.losses = 1;
                } else {
                    self.losses += 1;
                }
                self.focus = Some(
                    if winner == Player::Host.code().to_string() {
                        Player::Guest
                    } else {
                        Player::Host
                    }
                    .code()
                    .to_string(),
                );
                self.unresolved = 0;
            }
            _ => {
                // A draw/truncation never fabricates a loss or a role reversal.
                self.unresolved += 1;
            }
        }
        self.rotate_reason = if self.reversals >= o.reversals {
            Some("reversal-target".into())
        } else if self.losses >= o.losses {
            Some("ten-losses-without-reversal".into())
        } else if self.unresolved >= o.unresolved_attempts {
            Some("unresolved-case".into())
        }
        // Alternating draws and losses must not monopolize one actor forever.
        else if self.attempts >= (o.losses + o.unresolved_attempts) * (o.reversals + 1) {
            Some("case-attempt-budget".into())
        } else {
            None
        };
    }
    pub fn advance(&mut self, stride: usize) {
        *self = State {
            ticket: self.ticket + stride,
            ..Default::default()
        };
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Attempt {
    pub actor: usize,
    pub case: String,
    pub human_source: String,
    pub zone: usize,
    pub prefix_decisions: usize,
    pub before: State,
    pub after: State,
    pub kind: String,
}
/// Reconstruct every historical action, including bonus phases and repetition context.
pub(super) fn context(record: &GameRecord) -> Result<(Position, Repetitions)> {
    let mut p = record.initial_position();
    let mut repetition = Repetitions::new(&p, 4);
    for (i, a) in record.actions().iter().enumerate() {
        let mover = p.to_move();
        p.apply(*a)?;
        repetition.observe(&p, mover, i + 1);
    }
    Ok((p, repetition))
}
pub(super) fn prefix(record: &GameRecord, count: usize) -> GameRecord {
    let mut out = GameRecord::with_rules(record.setup(), record.rules());
    for a in record.actions().iter().take(count) {
        out.push(*a);
    }
    out
}
pub(super) fn load(path: &Path, seed: u64, out: &Path) -> Result<Vec<Case>> {
    let data = crate::compact_learning::load_dataset(path)?;
    let mut cases = vec![];
    let mut rng = StableRng::new(seed ^ 0x6361736573);
    for game in data.games.iter().filter(|g| !g.held_out) {
        let original = game
            .originals
            .first()
            .ok_or_else(|| invalid("human original missing"))?;
        let bytes = fs::read(&original.path)?;
        if sha256(&bytes) != original.sha256 {
            return Err(invalid("human source changed"));
        }
        let old: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        if sha256(old.to_string().as_bytes()) != game.game_sha256
            || old.rules().as_str() != data.rules
        {
            return Err(invalid("human identity mismatch"));
        }
        let (record, _) = old.replay_prefix_with_rules(RULES)?;
        let n = record.actions().len();
        if n < 3 {
            continue;
        }
        for zone in 0..3 {
            let low = n * zone / 3;
            let high = n * (zone + 1) / 3;
            let start = low + rng.index(high - low);
            let record = prefix(&record, start);
            let p = record.replay()?;
            if p.outcome() != GameOutcome::Ongoing {
                return Err(invalid("terminal human case"));
            }
            cases.push(Case {
                identity: sha256(record.to_string().as_bytes()),
                source: game
                    .split_identity_sha256
                    .clone()
                    .unwrap_or_else(|| game.game_sha256.clone()),
                zone,
                record,
            });
        }
    }
    if cases.is_empty() {
        return Err(invalid("no human training cases"));
    }
    shuffle(&mut cases, &mut rng);
    save_json_new(
        &out.join("human-cases.json"),
        &serde_json::json!({"rules":RULES.as_str(),"dataset_sha256":sha256(&fs::read(path)?),"seed":seed,"zones":3,"held_out_used":0,"cases":cases.iter().map(|c|serde_json::json!({"identity":c.identity,"source":c.source,"zone":c.zone,"prefix":c.record.actions().len()})).collect::<Vec<_>>()}),
    )?;
    Ok(cases)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ten_losses_rotate_but_three_attempts_are_not_three_reversals() {
        let o = CaseOptions::default();
        let mut s = State::default();
        for n in 0..9 {
            s.observe(GameOutcome::Win(Player::Host), &format!("{n}"), &o);
            assert!(s.rotate_reason.is_none());
        }
        assert_eq!(s.reversals, 0);
        assert_eq!(s.alternatives.len(), 9);
        s.observe(GameOutcome::Win(Player::Host), "10", &o);
        assert_eq!(
            s.rotate_reason.as_deref(),
            Some("ten-losses-without-reversal")
        );
        let mut s = State::default();
        for w in [Player::Host, Player::Guest, Player::Host] {
            s.observe(GameOutcome::Win(w), &w.code().to_string(), &o);
        }
        assert_eq!(s.reversals, 2);
        assert!(s.rotate_reason.is_none());
        s.observe(GameOutcome::Win(Player::Guest), "fourth", &o);
        assert_eq!(s.reversals, 3);
        assert_eq!(s.focus.as_deref(), Some("H"));
        assert_eq!(s.rotate_reason.as_deref(), Some("reversal-target"));
    }
    #[test]
    fn draw_and_timeout_are_neither_losses_nor_reversals() {
        let o = CaseOptions::default();
        let mut s = State::default();
        s.observe(GameOutcome::Win(Player::Guest), "a", &o);
        for _ in 0..9 {
            s.observe(GameOutcome::Draw, "b", &o);
        }
        assert_eq!(s.focus.as_deref(), Some("H"));
        assert_eq!(s.losses, 1);
        assert_eq!(s.reversals, 0);
        s.observe(GameOutcome::Ongoing, "c", &o);
        assert_eq!(s.rotate_reason.as_deref(), Some("unresolved-case"));
    }
    #[test]
    fn exact_prefix_keeps_bonus_and_record_action_indices() {
        let record: GameRecord = include_str!(
            "../../../../../crates/paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr"
        )
        .parse()
        .unwrap();
        let mut position = record.initial_position();
        for (i, a) in record.actions().iter().enumerate() {
            position.apply(*a).unwrap();
            let selected = prefix(&record, i + 1);
            assert_eq!(context(&selected).unwrap().0, position);
        }
    }
}
