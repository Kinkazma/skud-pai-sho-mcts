//! Read-only repetition audit over archived legal game records.
use super::*;

pub(super) fn run(args: &[String]) -> Result<()> {
    let flags: BTreeMap<_, _> = args
        .chunks(2)
        .map(|pair| {
            if pair.len() != 2 {
                return Err(invalid("expected --source DIR --output NEW_JSON"));
            }
            Ok((pair[0].as_str(), pair[1].as_str()))
        })
        .collect::<Result<_>>()?;
    let source = PathBuf::from(
        *flags
            .get("--source")
            .ok_or_else(|| invalid("missing source"))?,
    );
    let output = PathBuf::from(
        *flags
            .get("--output")
            .ok_or_else(|| invalid("missing output"))?,
    );
    let number = |key, default| -> Result<usize> {
        Ok(flags
            .get(key)
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let cycles = number("--cycles", 4)?;
    let maximum_period = number("--maximum-period", 32)?;
    let minimum_decisions = number("--minimum-decisions", 24)?;
    if !(1..=100).contains(&cycles)
        || !(2..=32).contains(&maximum_period)
        || minimum_decisions > cycles * maximum_period
    {
        return Err(invalid(
            "cycles must be 1..100, maximum period 2..32 and minimum decisions <= their product",
        ));
    }
    let mut paths = Vec::new();
    fn collect(path: &Path, paths: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                collect(&entry.path(), paths)?;
            } else if entry.path().extension().is_some_and(|e| e == "psr") {
                paths.push(entry.path());
            }
        }
        Ok(())
    }
    collect(&source, &mut paths)?;
    paths.sort();
    let started = Instant::now();
    let mut rows = Vec::new();
    for path in paths {
        let bytes = fs::read(&path)?;
        let record: GameRecord = std::str::from_utf8(&bytes)?.parse()?;
        let mut position = record.initial_position();
        let mut detector =
            Repetitions::with_limits(&position, cycles, maximum_period, minimum_decisions);
        let mut first = None;
        let mut last = None;
        let mut detections = 0;
        let mut trailing_detection_streak = 0;
        for (index, action) in record.actions().iter().enumerate() {
            let mover = position.to_move();
            position.apply(*action)?;
            if let Some(hit) = detector.observe(&position, mover, index + 1) {
                first.get_or_insert_with(|| hit.clone());
                last = Some(hit);
                detections += 1;
                trailing_detection_streak += 1;
            } else {
                trailing_detection_streak = 0;
            }
        }
        rows.push(serde_json::json!({"path":path,"rules":record.rules().as_str(),"record_sha256":sha256(&bytes),
            "original_decisions":record.actions().len(),"original_outcome":outcome_name(position.outcome()),
            "first_training_loss":first,"last_detection":last,"detected_decisions":detections,
            "trailing_detection_streak":trailing_detection_streak}));
    }
    save_json_new(
        &output,
        &serde_json::json!({"schema":"compact-repetition-audit-v2",
        "cycles":cycles,"minimum_decisions":minimum_decisions,"maximum_period":maximum_period,"elapsed_seconds":started.elapsed().as_secs_f64(),
        "interpretation":"counterfactual first training stop on recorded trajectory; no speed or force claim; some rules-terminal games may be adjudicated earlier",
        "games":rows}),
    )
}
