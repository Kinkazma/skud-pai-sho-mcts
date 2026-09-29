//! Cross-budget FIFO game permits: independent archives, one global CPU capacity.
use super::*;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

pub(super) struct GameSlots {
    state: Mutex<SlotState>,
    ready: Condvar,
    pub capacity: usize,
}
struct SlotState {
    free: usize,
    next: u64,
    waiting: VecDeque<u64>,
}
struct Permit<'a>(&'a GameSlots);
impl Drop for Permit<'_> {
    fn drop(&mut self) {
        self.0.state.lock().expect("game slot lock").free += 1;
        self.0.ready.notify_all();
    }
}
impl GameSlots {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(SlotState {
                free: capacity,
                next: 0,
                waiting: VecDeque::new(),
            }),
            ready: Condvar::new(),
            capacity,
        }
    }
    pub(super) fn acquire(&self, deadline: Instant) -> Option<impl Drop + '_> {
        let mut state = self.state.lock().expect("game slot lock");
        let id = state.next;
        state.next += 1;
        state.waiting.push_back(id);
        loop {
            let now = Instant::now();
            if now >= deadline {
                state.waiting.retain(|&ticket| ticket != id);
                self.ready.notify_all();
                return None;
            }
            if state.free > 0 && state.waiting.front() == Some(&id) {
                state.free -= 1;
                state.waiting.pop_front();
                self.ready.notify_all();
                return Some(Permit(self));
            }
            state = self
                .ready
                .wait_timeout(state, deadline - now)
                .expect("game slot wait")
                .0;
        }
    }
}

pub(super) fn run(args: &[String]) -> Result<()> {
    let mut flags = BTreeMap::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 || flags.insert(pair[0].clone(), pair[1].clone()).is_some() {
            return Err(invalid("expected distinct --name value pairs"));
        }
    }
    let budgets: Vec<usize> = flags
        .remove("--budgets")
        .unwrap_or_else(|| "32,64,128,256,512".into())
        .split(',')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()?;
    if budgets.is_empty()
        || budgets.iter().any(|&b| b == 0 || b > 8192)
        || budgets
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != budgets.len()
        || flags.contains_key("--simulations")
    {
        return Err(invalid(
            "suite requires distinct budgets up to8192; optional reference-simulations is shared",
        ));
    }
    let output = PathBuf::from(
        flags
            .remove("--output")
            .ok_or_else(|| invalid("missing --output"))?,
    );
    let seconds: u64 = flags
        .get("--seconds")
        .map_or("600", String::as_str)
        .parse()?;
    let workers: usize = flags.get("--workers").map_or("8", String::as_str).parse()?;
    if seconds == 0 || seconds > 3600 || workers == 0 || workers > 64 {
        return Err(invalid("invalid suite time/workers"));
    }
    let first_pair: u64 = flags
        .get("--first-pair")
        .map_or("4000000", String::as_str)
        .parse()?;
    let mut jobs = Vec::new();
    for (index, budget) in budgets.iter().enumerate() {
        let mut job = flags.clone();
        job.insert("--simulations".into(), budget.to_string());
        job.insert("--workers".into(), workers.to_string());
        job.insert("--seconds".into(), seconds.to_string());
        job.insert(
            "--output".into(),
            output
                .join(format!("budget-{budget}"))
                .to_string_lossy()
                .into_owned(),
        );
        job.insert(
            "--first-pair".into(),
            first_pair
                .checked_add(index as u64 * 10000)
                .ok_or_else(|| invalid("pair overflow"))?
                .to_string(),
        );
        let argv: Vec<_> = job
            .into_iter()
            .flat_map(|(key, value)| [key, value])
            .collect();
        let parsed = Options::parse(&argv)?;
        FrozenModel::load(&parsed.candidate)?;
        if let Some(path) = parsed.reference_model {
            FrozenModel::load(&path)?;
        }
        jobs.push(argv);
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)?;
    let started = Instant::now();
    let deadline = started + Duration::from_secs(seconds);
    save_json_new(
        &output.join("suite-plan.json"),
        &json!({"schema":"compact-comparison-suite-v1",
        "budgets":budgets,"capacity":workers,"seconds":seconds,"jobs":jobs,
        "scheduling":"FIFO permits between individual games across independent budget archives; one shared deadline; maximum total active games equals capacity",
        "threading":"each budget retains its worker threads; waiting threads hold no game permit; each active search is monothread"}),
    )?;
    let slots = GameSlots::new(workers);
    let results = std::thread::scope(|scope| {
        let handles: Vec<_> = jobs
            .iter()
            .map(|args| {
                let slots = &slots;
                scope.spawn(move || {
                    super::run_limited(args, Some(slots), Some(deadline)).map_err(|e| e.to_string())
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .unwrap_or_else(|_| Err("comparison panicked".into()))
            })
            .collect::<Vec<_>>()
    });
    let failed = results.iter().any(std::result::Result::is_err);
    save_json_new(
        &output.join("suite-summary.json"),
        &json!({"status":if failed {"failed"}else{"completed"},
        "wall_seconds":started.elapsed().as_secs_f64(),"capacity":workers,"results":results}),
    )?;
    if failed {
        return Err(invalid(
            "suite had comparison errors; see each immutable archive",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn permits_cap_concurrency_release_and_expire() {
        let slots = GameSlots::new(2);
        let deadline = Instant::now() + Duration::from_secs(2);
        let one = slots.acquire(deadline).unwrap();
        let two = slots.acquire(deadline).unwrap();
        assert!(slots.acquire(Instant::now()).is_none());
        drop(one);
        let three = slots.acquire(deadline).unwrap();
        assert_eq!(slots.state.lock().unwrap().free, 0);
        drop(two);
        drop(three);
        assert_eq!(slots.state.lock().unwrap().free, 2);
        assert!(slots.state.lock().unwrap().waiting.is_empty());
    }
    #[test]
    fn two_budget_suite_publishes_separate_complete_archives() {
        let root = std::env::temp_dir().join(format!("compact-suite-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let model = root.join("parent.json");
        crate::compact_learning::save_model_new(
            &model,
            &crate::compact_learning::ModelArtifact::legacy(),
        )
        .unwrap();
        let argv = vec![
            "--candidate".into(),
            model.to_string_lossy().into_owned(),
            "--reference-model".into(),
            model.to_string_lossy().into_owned(),
            "--output".into(),
            root.join("suite").to_string_lossy().into_owned(),
            "--budgets".into(),
            "1,2".into(),
            "--pairs".into(),
            "1".into(),
            "--seconds".into(),
            "5".into(),
            "--workers".into(),
            "2".into(),
            "--decision-limit".into(),
            "1".into(),
        ];
        run(&argv).unwrap();
        for budget in [1, 2] {
            let base = root.join(format!("suite/budget-{budget}"));
            let summary: Value =
                serde_json::from_slice(&fs::read(base.join("summary.json")).unwrap()).unwrap();
            let plan: Value =
                serde_json::from_slice(&fs::read(base.join("plan.json")).unwrap()).unwrap();
            assert_eq!(summary["complete_archive"], true);
            assert_eq!(plan["suite_capacity"], 2);
            assert_eq!(plan["reference_kind"], "compact-model");
            assert_eq!(plan["reference_simulations"], budget);
            assert_eq!(fs::read_dir(base.join("records")).unwrap().count(), 2);
        }
        assert!(run(&argv).is_err());
        let mut fixed = argv.clone();
        let output_index = fixed.iter().position(|x| x == "--output").unwrap() + 1;
        fixed[output_index] = root.join("fixed-reference").to_string_lossy().into_owned();
        fixed.extend(["--reference-simulations".into(), "3".into()]);
        run(&fixed).unwrap();
        for budget in [1, 2] {
            let base = root.join(format!("fixed-reference/budget-{budget}"));
            let plan: Value = serde_json::from_slice(&fs::read(base.join("plan.json")).unwrap()).unwrap();
            assert_eq!(plan["mcts"]["simulations"], budget);
            assert_eq!(plan["reference_simulations"], 3);
            for entry in fs::read_dir(base.join("games")).unwrap() {
                let game: Value = serde_json::from_slice(&fs::read(entry.unwrap().path()).unwrap()).unwrap();
                assert_eq!(game["reference"]["simulations"].as_u64().unwrap(),
                           3 * game["reference"]["completed_searches"].as_u64().unwrap());
            }
        }
        fs::remove_dir_all(root).unwrap();
    }
}
