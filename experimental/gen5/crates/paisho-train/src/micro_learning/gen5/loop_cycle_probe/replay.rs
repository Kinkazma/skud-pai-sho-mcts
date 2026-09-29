//! Diagnostic adapter over the production FIFO/correction queues. No duplicate
//! sampling logic or model evaluation lives here. Old plans do not construct it.
use super::*;

pub(super) struct Replay {
    memory: memory::Memory,
    historical_fraction: f64,
    correction_fraction: f64,
    structural_repair: bool,
    initial_rows: usize,
    correction_attempts: usize,
    correction_used: usize,
    fifo_used: usize,
    fresh_fallback_used: usize,
    admitted_rows: usize,
    admitted_corrections: usize,
    obsolete_sources_retained: usize,
    lane_draws: [usize; 4],
}

impl Replay {
    pub(super) fn open(
        options: &Options,
        input: &Input,
        count: usize,
        capacity: usize,
        repaired: bool,
        spatial: bool,
        out: &Path,
    ) -> Result<Self> {
        // Saved examples have no original Lane. At fraction zero the initial
        // Selfplay tag affects reporting, not the native draw distribution.
        if count == 0 || capacity == 0 || options.historical_replay_fraction != 0. {
            return Err(invalid("diagnostic native replay requires nonempty input and zero historical preference (initial lanes unknown)"));
        }
        let mut options = options.clone();
        options.replay_capacity = capacity;
        options.learning_loop_v3 = repaired;
        let index = out.join("initial-native-replay-index.json");
        save_json_new(
            &index,
            &serde_json::json!({
                "schema":"paisho-gen5-replay-index-v1", "rules":RULES.as_str(),
                "rows":[{"source":{"path":fs::canonicalize(&input.path)?,"sha256":input.sha256},
                    "indices":(0..count).collect::<Vec<_>>(),"lane":"Selfplay"}]
            }),
        )?;
        let mut memory = memory::Memory::new(&options);
        memory.load_for_model(&index, spatial)?;
        if memory.len() == 0 {
            return Err(invalid("diagnostic native replay empty after loading"));
        }
        Ok(Self {
            memory,
            historical_fraction: options.historical_replay_fraction,
            correction_fraction: options.correction_replay_fraction,
            structural_repair: options.structural_repair,
            initial_rows: count,
            correction_attempts: 0,
            correction_used: 0,
            fifo_used: 0,
            fresh_fallback_used: 0,
            admitted_rows: 0,
            admitted_corrections: 0,
            obsolete_sources_retained: 0,
            lane_draws: [0; 4],
        })
    }

    /// Preserve runtime's RNG calls, including the history preference draw at
    /// fraction zero, the correction short circuit, and its fallback behavior.
    pub(super) fn draw(
        &mut self,
        rng: &mut StableRng,
        owned: &[Arc<MicroExample>],
        lane: Lane,
    ) -> Result<(Arc<MicroExample>, u8)> {
        self.draw_with_lane(rng, owned, lane)
            .map(|(e, k, _)| (e, k))
    }
    /// Diagnostic tape provenance; same native draw and RNG sequence.
    pub(super) fn draw_with_lane(
        &mut self,
        rng: &mut StableRng,
        owned: &[Arc<MicroExample>],
        lane: Lane,
    ) -> Result<(Arc<MicroExample>, u8, Lane)> {
        let prefer = rng.next_f64() < self.historical_fraction;
        let correction = self.correction_fraction > 0.
            && self.memory.correction_len() > 0
            && rng.next_f64() < self.correction_fraction;
        self.correction_attempts += usize::from(correction);
        let prioritized = if correction {
            self.memory.draw_correction(rng)
        } else {
            None
        };
        let kind = if prioritized.is_some() { 3 } else { 1 };
        let mut fresh_fallback = false;
        let selected = prioritized
            .or_else(|| self.memory.draw(rng, prefer))
            .or_else(|| {
                if self.structural_repair && !owned.is_empty() {
                    fresh_fallback = true;
                    Some((owned[rng.index(owned.len())].clone(), lane))
                } else {
                    None
                }
            });
        let (example, selected_lane) =
            selected.ok_or_else(|| invalid("diagnostic native replay has no trainable draw"))?;
        if kind == 3 {
            self.correction_used += 1;
        } else if fresh_fallback {
            self.fresh_fallback_used += 1;
        } else {
            self.fifo_used += 1;
        }
        let i = match selected_lane {
            Lane::Selfplay => 0,
            Lane::Checkpoint => 1,
            Lane::Historical => 2,
            Lane::Reanalysis => 3,
        };
        self.lane_draws[i] += 1;
        Ok((example, kind, selected_lane))
    }

    /// Call after this receipt's SGD and before publication/Archive::admit, as
    /// in runtime. Native dense source indices require every fresh row in order.
    pub(super) fn admit(
        &mut self,
        owned: &[Arc<MicroExample>],
        targets: &Path,
        hash: &str,
        lane: Lane,
        saved: &[SavedMicroExample],
    ) -> Result<()> {
        if owned.len() != saved.len()
            || saved.iter().zip(owned).any(|(s, e)| {
                s.state != e.state
                    || s.action_features.len() != e.actions.len()
                    || s.action_features.iter().zip(&e.actions).any(|(a, b)| {
                        a.len() != b.len()
                            || a.iter().zip(b).any(|(x, y)| x.to_bits() != y.to_bits())
                    })
            })
        {
            return Err(invalid(
                "native replay requires aligned, unstrided fresh targets",
            ));
        }
        let corrections = saved
            .iter()
            .map(|s| s.correction_priority)
            .collect::<Vec<_>>();
        self.memory
            .add(owned.to_vec(), targets, hash, lane, &corrections);
        self.admitted_rows += owned.len();
        self.admitted_corrections += corrections.iter().filter(|x| **x).count();
        // Forensic artifacts stay on disk even after the native FIFO releases
        // their RAM entries. This does not retain evicted examples or weak refs.
        self.obsolete_sources_retained += self.memory.take_obsolete().len();
        Ok(())
    }

    pub(super) fn progress(&self) -> serde_json::Value {
        serde_json::json!({
            "implementation":"production Memory", "initial_rows":self.initial_rows,
            "initial_lane_provenance":"unknown; Selfplay tag; historical preference zero",
            "entries":self.memory.len(), "bytes":self.memory.bytes,
            "evicted":self.memory.evicted, "correction_queue_entries":self.memory.correction_len(),
            "correction_reference_bytes":self.memory.correction_reference_bytes(),
            "historical_replay_fraction":self.historical_fraction,
            "correction_replay_fraction":self.correction_fraction,
            "correction_attempts":self.correction_attempts, "correction_used":self.correction_used,
            "fifo_used":self.fifo_used, "fresh_fallback_used":self.fresh_fallback_used,
            "admitted_rows":self.admitted_rows, "admitted_corrections":self.admitted_corrections,
            "obsolete_source_files_retained":self.obsolete_sources_retained,
            "lane_draws_selfplay_checkpoint_historical_reanalysis":self.lane_draws,
            "counts_are_cumulative":true
        })
    }
}

fn terminal_host_value(outcome: GameOutcome) -> Option<f64> {
    match outcome {
        GameOutcome::Win(Player::Host) => Some(1.),
        GameOutcome::Win(Player::Guest) => Some(-1.),
        GameOutcome::Draw => Some(0.),
        GameOutcome::Ongoing => None,
    }
}

/// Use the production reanalysis budget and only a rules-derived, hash-bound
/// terminal observation. A truncated/nonterminal source remains an estimate.
/// Call before collection, only for the opt-in receipt-faithful protocol.
pub(super) fn reanalysis_options(
    options: &Options,
    prefix: &Prefix,
) -> Result<(Options, serde_json::Value)> {
    let mut result = options.clone();
    result.observed_origin = None;
    if !prefix.reanalysis {
        let metadata = serde_json::json!({"reanalysis":false,"budgets":result.budgets,
            "observed_origin":null,"origin_kind":"fresh self-play"});
        return Ok((result, metadata));
    }
    let budget = options
        .case_curriculum
        .as_ref()
        .ok_or_else(|| invalid("native reanalysis requires case curriculum budget"))?
        .reanalysis_budget;
    if budget == 0 {
        return Err(invalid("zero native reanalysis budget"));
    }
    result.budgets = vec![(budget, 1.)];
    let bytes = prefix.input.bytes()?;
    let original: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
    original.replay()?;
    let all = if original.rules() == RULES {
        original
    } else {
        original.replay_prefix_with_rules(RULES)?.0
    };
    if prefix.decisions > all.actions().len() {
        return Err(invalid(
            "reanalysis observation prefix exceeds verified Gen5 source",
        ));
    }
    let outcome = all.replay()?.outcome();
    let canonical_hash = sha256(all.to_string().as_bytes());
    result.observed_origin =
        terminal_host_value(outcome).map(|value| (value, canonical_hash.clone()));
    let metadata = serde_json::json!({
        "reanalysis":true,"budgets":result.budgets,"observed_origin":result.observed_origin,
        "source_input_sha256":prefix.input.sha256,"source_gen5_psr_sha256":canonical_hash,
        "source_terminal_outcome":format!("{outcome:?}"),"source_decisions":all.actions().len(),
        "origin_kind":if result.observed_origin.is_some() {"verified full source terminal"} else {"no terminal observation"}
    });
    Ok((result, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_terminal_observations_supply_host_values() {
        assert_eq!(
            terminal_host_value(GameOutcome::Win(Player::Host)),
            Some(1.)
        );
        assert_eq!(
            terminal_host_value(GameOutcome::Win(Player::Guest)),
            Some(-1.)
        );
        assert_eq!(terminal_host_value(GameOutcome::Draw), Some(0.));
        assert_eq!(terminal_host_value(GameOutcome::Ongoing), None);
    }

    #[test]
    fn native_correction_draws_keep_runtime_rng_and_dense_admission() {
        let root = std::env::temp_dir().join(format!(
            "gen5-native-replay-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&root).unwrap();
        let mut saved = vec![crate::micro_learning::tactics::fixture(); 3];
        saved[0].correction_priority = true;
        saved[1].correction_priority = false;
        saved[2].correction_priority = true;
        let targets = root.join("targets.json.gz");
        save_examples_new(&targets, &saved).unwrap();
        let input = Input {
            path: targets.clone(),
            sha256: sha256(&fs::read(&targets).unwrap()),
        };
        let options = Options {
            replay_capacity: 3,
            correction_capacity: 2,
            historical_replay_fraction: 0.,
            correction_replay_fraction: 0.5,
            ..Default::default()
        };
        let mut actual = Replay::open(&options, &input, 3, 3, false, false, &root).unwrap();
        let mut expected = memory::Memory::new(&options);
        expected
            .load_for_model(&root.join("initial-native-replay-index.json"), false)
            .unwrap();
        assert_eq!(actual.memory.correction_len(), 2);
        let mut a = StableRng::new(8187);
        let mut b = StableRng::new(8187);
        for _ in 0..64 {
            let prefer = b.next_f64() < options.historical_replay_fraction;
            let correction = options.correction_replay_fraction > 0.
                && expected.correction_len() > 0
                && b.next_f64() < options.correction_replay_fraction;
            let prioritized = if correction {
                expected.draw_correction(&mut b)
            } else {
                None
            };
            let kind = if prioritized.is_some() { 3 } else { 1 };
            let (expected, _) = prioritized
                .or_else(|| expected.draw(&mut b, prefer))
                .unwrap();
            let (actual, actual_kind) = actual.draw(&mut a, &[], Lane::Selfplay).unwrap();
            assert_eq!(actual_kind, kind);
            assert_eq!(actual.state, expected.state);
            assert_eq!(actual.sequence_source, expected.sequence_source);
            assert_eq!(a.next_u64(), b.next_u64());
        }
        let owned = saved
            .iter()
            .map(|s| Arc::new(s.example_for_rules(RULES).unwrap()))
            .collect::<Vec<_>>();
        assert!(actual
            .admit(
                &owned[..2],
                &targets,
                &input.sha256,
                Lane::Reanalysis,
                &saved
            )
            .is_err());
        actual
            .admit(&owned, &targets, &input.sha256, Lane::Reanalysis, &saved)
            .unwrap();
        assert_eq!(actual.memory.len(), 3);
        assert_eq!(actual.memory.evicted, 3);
        assert_eq!(actual.admitted_corrections, 2);
        assert_eq!(actual.correction_used + actual.fifo_used, 64);
        drop(actual);
        drop(expected);
        fs::remove_dir_all(root).unwrap();
    }
}
