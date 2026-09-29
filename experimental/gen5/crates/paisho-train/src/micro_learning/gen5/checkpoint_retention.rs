//! Bounded passive measurements and temporally distributed checkpoint retention.
//! No new matches, inference or learning; only exact collector snapshots are scored.
use super::*;
mod selection;
#[cfg(test)]
mod tests;

#[derive(Clone, Default, Serialize)]
pub(super) struct Results {
    attempts: [usize; 2],
    wins: [usize; 2],
    cases: std::collections::BTreeSet<(String, usize)>,
}
impl Results {
    fn score(&self) -> Option<f64> {
        (self.attempts.iter().sum::<usize>() > 0).then(|| {
            // Beta(1,1) shrinkage per seat; an unobserved seat stays neutral.
            // A single game is an explicitly provisional observation, not Elo.
            (0..2)
                .map(|i| (self.wins[i] as f64 + 1.) / (self.attempts[i] as f64 + 2.))
                .sum::<f64>()
                / 2.
        })
    }
    fn record(&mut self, case: &str, seat: usize, win: bool) {
        if self.cases.insert((case.to_owned(), seat)) {
            self.attempts[seat] += 1;
            self.wins[seat] += usize::from(win);
        }
    }
}
#[derive(Clone, Serialize)]
pub(super) struct Entry {
    pub path: PathBuf,
    pub version: u64,
    pub ordinal: usize,
    /// Monotonic observation/save time in this writer's lifetime, not update count.
    pub elapsed_millis: u64,
    pub identity: String,
    pub results: std::collections::BTreeMap<String, Results>,
}
pub(super) struct Pending {
    pub snapshot: Arc<Snapshot>,
    pub entry: Entry,
}
pub(super) struct Retention {
    pub entries: Vec<Entry>,
    pending: Vec<Pending>,
    next: usize,
    started: Instant,
}
impl Default for Retention {
    fn default() -> Self {
        Self {
            entries: vec![],
            pending: vec![],
            next: 0,
            started: paisho_platform::training_time::now(),
        }
    }
}
impl Retention {
    const PENDING_LIMIT: usize = 16;
    fn entry(&mut self, path: PathBuf, snapshot: &Snapshot) -> Entry {
        let entry = Entry {
            path,
            version: snapshot.version,
            ordinal: self.next,
            elapsed_millis: paisho_platform::training_time::elapsed(self.started).as_millis().min(u64::MAX as u128) as u64,
            identity: snapshot.identity.clone(),
            results: Default::default(),
        };
        self.next += 1;
        entry
    }
    pub fn add(&mut self, path: PathBuf, snapshot: &Snapshot) -> Result<()> {
        let entry = self.entry(path, snapshot);
        self.entries.push(entry);
        Ok(())
    }
    pub fn take_candidates(&mut self) -> Vec<Pending> {
        std::mem::take(&mut self.pending)
    }
    pub fn observe(&mut self, game: &collector::Played, specs: &[OpponentSpec]) {
        let Some(case) = game.case.as_ref() else {
            return;
        };
        if game.reanalysis
            || case.before.attempts != 0
            || game.error.is_some()
            || game.reference_budget.is_none()
            || game.outcome == GameOutcome::Ongoing
        {
            return;
        }
        let Some(spec) = specs
            .iter()
            .find(|s| game.opponent == format!("Gen{}", s.generation))
        else {
            return;
        };
        let key = format!(
            "{}:{}:MCTS{}",
            game.opponent,
            spec.sha256,
            game.reference_budget.unwrap()
        );
        self.observe_result(
            &game.snapshot,
            key,
            &case.case,
            usize::from(game.candidate_seat == Player::Guest),
            game.outcome == GameOutcome::Win(game.candidate_seat),
        );
    }
    pub(super) fn observe_result(
        &mut self,
        snapshot: &Arc<Snapshot>,
        key: String,
        case: &str,
        seat: usize,
        win: bool,
    ) {
        let matches = |e: &Entry| e.version == snapshot.version && e.identity == snapshot.identity;
        if let Some(entry) = self.entries.iter_mut().find(|e| matches(e)) {
            entry
                .results
                .entry(key)
                .or_default()
                .record(case, seat, win);
            return;
        }
        if let Some(candidate) = self.pending.iter_mut().find(|p| matches(&p.entry)) {
            candidate
                .entry
                .results
                .entry(key)
                .or_default()
                .record(case, seat, win);
            return;
        }
        let mut entry = self.entry(PathBuf::new(), snapshot);
        entry
            .results
            .entry(key)
            .or_default()
            .record(case, seat, win);
        self.pending.push(Pending {
            snapshot: snapshot.clone(),
            entry,
        });
        if self.pending.len() > Self::PENDING_LIMIT {
            let entries: Vec<_> = self.pending.iter().map(|p| &p.entry).collect();
            let chosen = selection::select(&entries, Self::PENDING_LIMIT, entries.len() - 1);
            let mut index = 0;
            self.pending.retain(|_| {
                let keep = chosen.contains(&index);
                index += 1;
                keep
            });
        }
    }
    pub fn select(&self, keep: usize, recovery: &Path) -> Vec<usize> {
        let entries: Vec<_> = self.entries.iter().collect();
        if entries.is_empty() {
            return vec![];
        }
        let current = entries
            .iter()
            .position(|e| e.path == recovery)
            // An existing recovery input is outside this writer's ownership
            // and therefore cannot be pruned, even when it is absent here.
            .unwrap_or(entries.len() - 1);
        selection::select(&entries, keep, current)
    }
    pub fn metadata(&self, chosen: &[usize], keep: usize) -> serde_json::Value {
        serde_json::json!({"limit":keep,"policy":"time-bins_spaced-passive-champions_v2",
            "score":"provisional seat-balanced Beta(1,1) actual-win rate; exact collector, first case/seat attempts, terminal only; rank only within same opponent hash/budget; not Elo or a promotion gate",
            "spacing":"monotonic elapsed observation/save time; aligned power-of-two bins only merge as history grows; recovery checkpoint always protected",
            "pending_limit":Self::PENDING_LIMIT,
            "models":chosen.iter().map(|&i| &self.entries[i]).collect::<Vec<_>>()})
    }
}
