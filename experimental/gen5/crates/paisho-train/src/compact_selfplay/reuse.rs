//! Training-only repetition adjudication and bounded, auditable RAM replay.
use super::*;
use std::collections::VecDeque;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct RepetitionLoss {
    pub loser: Seat,
    pub first_decision: usize,
    pub last_decision: usize,
    pub period: usize,
    pub cycles: usize,
}

pub(crate) struct Repetitions {
    positions: VecDeque<Position>,
    cycles: usize,
    maximum_period: usize,
    minimum_decisions: usize,
}

fn same_state(a: &Position, b: &Position) -> bool {
    // Turn counters do not change legal choices. Every rule-relevant field does.
    a.rule_profile() == b.rule_profile()
        && a.board() == b.board()
        && a.reserve(Player::Host) == b.reserve(Player::Host)
        && a.reserve(Player::Guest) == b.reserve(Player::Guest)
        && a.to_move() == b.to_move()
        && a.phase() == b.phase()
        && a.outcome() == b.outcome()
}

impl Repetitions {
    pub fn new(position: &Position, cycles: usize) -> Self {
        Self::with_limits(position, cycles, 32, 24)
    }
    // Explicit limits for retrospective audits; runtime callers keep `new`.
    pub(super) fn with_limits(
        position: &Position,
        cycles: usize,
        maximum_period: usize,
        minimum_decisions: usize,
    ) -> Self {
        Self {
            positions: VecDeque::from([position.clone()]),
            cycles,
            maximum_period,
            minimum_decisions,
        }
    }
    pub fn observe(
        &mut self,
        position: &Position,
        mover: Player,
        decision: usize,
    ) -> Option<RepetitionLoss> {
        if self.cycles == 0 || position.outcome() != GameOutcome::Ongoing {
            return None;
        }
        self.positions.push_back(position.clone());
        while self.positions.len() > self.maximum_period * self.cycles + 1 {
            self.positions.pop_front();
        }
        for period in 2..=self.maximum_period {
            let span = period * self.cycles;
            if span < self.minimum_decisions || self.positions.len() <= span {
                continue;
            }
            let start = self.positions.len() - span - 1;
            if (start + period..self.positions.len())
                .all(|i| same_state(&self.positions[i], &self.positions[i - period]))
            {
                return Some(RepetitionLoss {
                    loser: Seat::from_player(mover),
                    first_decision: decision - span + 1,
                    last_decision: decision,
                    period,
                    cycles: self.cycles,
                });
            }
        }
        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ReplayExample {
    #[serde(default = "crate::compact_learning::legacy_training_rules")]
    pub rules: String,
    pub source_run: String,
    pub game_id: usize,
    pub actor_version: u64,
    pub actor_weights_sha256: String,
    pub sample: RootSample,
    pub target: f64,
    pub reason: String,
}

pub(super) struct ReplayMemory {
    entries: VecDeque<ReplayExample>,
    capacity: usize,
    rng: StableRng,
    pub draws: u64,
}
impl ReplayMemory {
    pub fn new(capacity: usize, seed: u64) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity,
            rng: StableRng::new(seed),
            draws: 0,
        }
    }
    pub fn load(&mut self, path: &Path) -> Result<()> {
        let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
        if value["schema"] != "paisho-compact-ram-replay-v1" {
            return Err(invalid("unsupported replay memory schema"));
        }
        let entries: Vec<ReplayExample> = serde_json::from_value(value["entries"].clone())?;
        for entry in entries {
            crate::compact_learning::require_current_training_rules(&entry.rules)?;
            entry.sample.features()?;
            if !entry.target.is_finite()
                || entry.target.abs() > 1.0
                || entry.source_run.is_empty()
                || entry.actor_weights_sha256.len() != 64
                || !matches!(
                    entry.reason.as_str(),
                    "rules-terminal-q-mix" | "repetition-training-loss"
                )
            {
                return Err(invalid("invalid persisted replay target or provenance"));
            }
            self.push(entry);
        }
        Ok(())
    }
    pub fn push(&mut self, entry: ReplayExample) {
        if self.capacity == 0 {
            return;
        }
        if self.entries.len() == self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }
    pub fn draw(&mut self) -> Option<ReplayExample> {
        if self.entries.is_empty() {
            return None;
        }
        self.draws += 1;
        Some(self.entries[self.rng.index(self.entries.len())].clone())
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn persist(&self, output: &Path) -> Result<()> {
        save_json_new(
            &output.join("replay-final.json"),
            &serde_json::json!({
                "schema":"paisho-compact-ram-replay-v1", "capacity":self.capacity,
                "rules":paisho_core::RuleProfileId::CURRENT.as_str(),
                "draws":self.draws,"entries":self.entries,
                "resume":"examples and immutable targets preserved; continuation seeds a new recorded RNG stream"
            }),
        )
    }
}

pub(super) fn training_example(
    game: &PlayedGame,
    sample: &RootSample,
    lambda: f64,
    source_run: &str,
) -> Option<ReplayExample> {
    if game.error.is_some() {
        return None;
    }
    let (target, reason) = if let Some(loss) = &game.repetition_loss {
        // Penalize the player closing the repeated cycle. Never teach a fabricated
        // victory to its opponent, or contradict both perspectives of one state.
        if sample.perspective != loss.loser || sample.decision < loss.first_decision {
            return None;
        }
        (-1.0, "repetition-training-loss")
    } else {
        (
            mixed_target(sample, game.outcome, lambda)?,
            "rules-terminal-q-mix",
        )
    };
    Some(ReplayExample {
        rules: game.record.rules().to_string(),
        source_run: source_run.into(),
        game_id: game.id,
        actor_version: game.version,
        actor_weights_sha256: game.weights_sha256.clone(),
        sample: sample.clone(),
        target,
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn previous_profile_replay_is_refused_before_any_target_is_loaded() {
        let path =
            std::env::temp_dir().join(format!("paisho-rules-replay-{}.json", std::process::id()));
        let example = ReplayExample {
            rules: paisho_core::RuleProfileId::CURRENT.to_string(),
            source_run: "old-run".into(),
            game_id: 1,
            actor_version: 1,
            actor_weights_sha256: "a".repeat(64),
            sample: super::super::tests::sample(Player::Guest, 1),
            target: 1.0,
            reason: "rules-terminal-q-mix".into(),
        };
        let mut old = serde_json::to_value(example).unwrap();
        old.as_object_mut().unwrap().remove("rules");
        fs::write(
            &path,
            serde_json::json!({"schema":"paisho-compact-ram-replay-v1", "entries":[old]})
                .to_string(),
        )
        .unwrap();
        let mut replay = ReplayMemory::new(8, 1);
        assert!(replay
            .load(&path)
            .unwrap_err()
            .to_string()
            .contains("current training requires"));
        assert_eq!(replay.len(), 0);
        fs::remove_file(path).unwrap();
    }
    #[test]
    fn replay_is_bounded_and_preserves_targets_and_provenance() {
        let mut replay = ReplayMemory::new(2, 7);
        for game_id in 0..3 {
            replay.push(ReplayExample {
                rules: paisho_core::RuleProfileId::CURRENT.to_string(),
                source_run: "test-run".into(),
                game_id,
                actor_version: 10 + game_id as u64,
                actor_weights_sha256: format!("weights-{game_id}"),
                sample: super::super::tests::sample(Player::Guest, 1),
                target: game_id as f64 / 3.0,
                reason: "rules-terminal-q-mix".into(),
            });
        }
        assert_eq!(replay.len(), 2);
        for _ in 0..20 {
            let item = replay.draw().unwrap();
            assert!(item.game_id > 0);
            assert_eq!(item.target, item.game_id as f64 / 3.0);
            assert_eq!(item.actor_version, 10 + item.game_id as u64);
        }
        assert_eq!(replay.draws, 20);
    }
    #[test]
    fn retrospective_six_cycles_requires_the_whole_sequence_and_ignores_longer_periods() {
        // Synthetic distinct state streams isolate sequence matching from move legality.
        let states: Vec<_> = BASIC_FLOWERS
            .iter()
            .map(|flower| Position::from_standard_setup(StandardSetup::balanced(*flower)))
            .collect();
        let mut short = Repetitions::with_limits(&states[0], 6, 4, 0);
        for decision in 1..24 {
            assert!(short
                .observe(&states[decision % 4], Player::Host, decision)
                .is_none());
        }
        let hit = short.observe(&states[0], Player::Host, 24).unwrap();
        assert_eq!((hit.period, hit.cycles, hit.first_decision), (4, 6, 1));
        let mut longer = Repetitions::with_limits(&states[0], 6, 4, 0);
        for decision in 1..100 {
            assert!(longer
                .observe(&states[decision % 5], Player::Host, decision)
                .is_none());
        }
        // Returning to one state is insufficient when the intervening sequence differs.
        let mut broken = Repetitions::with_limits(&states[0], 6, 4, 0);
        for decision in 1..=24 {
            let index = if decision == 13 { 4 } else { decision % 4 };
            assert!(broken
                .observe(&states[index], Player::Host, decision)
                .is_none());
        }
    }

    #[test]
    fn archived_long_loop_is_detected_with_legal_unchanged_psr() {
        let record: GameRecord = include_str!("../../../../benchmarks/results/compact-final-evaluation-2026-09-08/final32-vs-old32/records/game-00000005.psr").parse().unwrap();
        let mut position = record.initial_position();
        let mut detector = Repetitions::new(&position, 4);
        let mut detected = None;
        for (index, action) in record.actions().iter().enumerate() {
            let mover = position.to_move();
            position.apply(*action).unwrap();
            if let Some(loss) = detector.observe(&position, mover, index + 1) {
                detected = Some(loss);
                break;
            }
        }
        let loss = detected.expect("known long loop");
        assert!(loss.last_decision < 300);
        assert!(loss.last_decision - loss.first_decision + 1 >= 24);
        assert_eq!(position.outcome(), GameOutcome::Ongoing);
    }

    #[test]
    fn v2_loop_is_exact_despite_advancing_turn_counter_and_disabled_mode_keeps_playing() {
        let old: GameRecord = include_str!("../../../../benchmarks/results/compact-final-evaluation-2026-09-08/final32-vs-old32/records/game-00000005.psr").parse().unwrap();
        let (record, _) = old
            .replay_prefix_with_rules(paisho_core::RuleProfileId::CURRENT)
            .unwrap();
        let mut position = record.initial_position();
        let mut detector = Repetitions::new(&position, 4);
        let mut disabled = Repetitions::new(&position, 0);
        let mut states = vec![position.clone()];
        let mut found = None;
        for (index, action) in record.actions().iter().enumerate() {
            let mover = position.to_move();
            position.apply(*action).unwrap();
            assert!(disabled.observe(&position, mover, index + 1).is_none());
            states.push(position.clone());
            if let Some(loss) = detector.observe(&position, mover, index + 1) {
                assert_eq!(loss.loser.player(), mover);
                assert_eq!(loss.last_decision, index + 1);
                let earlier = &states[states.len() - 1 - loss.period];
                assert!(same_state(earlier, &position));
                assert!(earlier.completed_turns() < position.completed_turns());
                assert!(loss.last_decision - loss.first_decision + 1 >= 24);
                assert_eq!(position.outcome(), GameOutcome::Ongoing);
                found = Some(loss);
                break;
            }
        }
        assert!(
            found
                .expect("known loop still exists under V2")
                .last_decision
                < 300
        );
    }

    #[test]
    fn same_board_is_not_a_repetition_with_different_rules_reserves_or_bonus_phase() {
        let setup = StandardSetup::balanced(BASIC_FLOWERS[0]);
        let current = Position::from_standard_setup(setup);
        let historical = Position::from_standard_setup_with_rules(
            setup,
            paisho_core::RuleProfileId::SkudPaiSho2022,
        );
        assert_eq!(current.board(), historical.board());
        assert!(!same_state(&current, &historical));
        let other = Position::from_standard_setup(StandardSetup {
            host_accents: paisho_core::AccentLoadout::new(2, 2, 0, 0).unwrap(),
            ..setup
        });
        assert_eq!(current.board(), other.board());
        assert!(!same_state(&current, &other));

        let record: GameRecord =
            include_str!("../../../../crates/paisho-ai/tests/fixtures/site_bot_v1_ring_finish.psr")
                .parse()
                .unwrap();
        let (record, _) = record
            .replay_prefix_with_rules(paisho_core::RuleProfileId::CURRENT)
            .unwrap();
        let mut position = record.initial_position();
        let mut checked_bonus = false;
        for action in record.actions() {
            position.apply(*action).unwrap();
            if position.phase() == paisho_core::TurnPhase::HarmonyBonus {
                let mut skipped = position.clone();
                skipped
                    .apply(paisho_core::Action::SkipHarmonyBonus)
                    .unwrap();
                assert_eq!(position.board(), skipped.board());
                assert!(!same_state(&position, &skipped));
                checked_bonus = true;
                break;
            }
        }
        assert!(checked_bonus);
    }
}
