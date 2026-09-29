//! Frozen two-reference assessments share only the historical CPU lane.
//! Marks are provisional internal score-Elo; they never stop learning or alter
//! the public agent. Gen3.1 results remain separate from the fixed Gen4 anchor.
use super::*;
#[derive(Clone)]
pub(super) struct Ranked {
    pub snapshot: Arc<Snapshot>,
    pub score: f64,
}
pub(super) fn assess(
    snapshot: Arc<Snapshot>,
    initial: &Arc<Snapshot>,
    o: &Options,
    end: Instant,
    pool: &rayon::ThreadPool,
    index: usize,
) -> Result<Option<Ranked>> {
    let started = Instant::now();
    let cap = end.min(started + Duration::from_secs_f64(o.history_seconds));
    let dir = o.output.join("history").join(format!("sweep-{index:04}"));
    fs::create_dir_all(&dir)?;
    let snapshot = if o.case_curriculum.is_some() {
        let path = dir.join("model.json");
        if let Some(artifact) = &snapshot.artifact {
            artifact.save(&path)?;
        } else {
            fs::copy(&snapshot.path, &path)?;
        }
        Arc::new(Snapshot {
            path,
            ..(*snapshot).clone()
        })
    } else {
        snapshot
    };
    let mut reports = vec![];
    for (name, reference, pairs, seconds) in [
        (
            "initial",
            initial.path.as_path(),
            o.history_initial_pairs,
            o.game_seconds.max(30.0),
        ),
        (
            "gen3-1",
            o.reference.as_path(),
            2,
            o.historical_seconds
                .iter()
                .find(|(b, _)| *b == 64)
                .unwrap()
                .1
                .max(60.0),
        ),
    ] {
        let remaining = cap.saturating_duration_since(Instant::now()).as_secs_f64();
        if remaining < 0.05 {
            break;
        }
        let path = dir.join(name);
        compare_micro_profile(
            &snapshot.path,
            reference,
            &path,
            MicroCompareOptions {
                pairs,
                workers: 1,
                varied_setups: true,
                simulations: 64,
                move_ms: 0,
                game_seconds: seconds,
                seconds: remaining,
                decisions: o.history_decision_limit,
                seed: o.seed ^ 0x6576616c,
            },
            pool,
            RULES,
            o.search(0, false).map_err(invalid)?,
            if name == "initial" {
                o.search(0, false).map_err(invalid)?
            } else {
                MicroSearchOptions::default()
            },
            64,
        )?;
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(path.join("report.json"))?)?;
        if report["games"]
            .as_array()
            .is_some_and(|rows| rows.iter().any(|r| !r["error"].is_null()))
        {
            return Err(invalid("history comparison engine error"));
        }
        reports.push((name, report));
    }
    let mut eligible = reports.len() == 2;
    let mut score = 0.0;
    for (_, r) in &reports {
        let w = r["wins"].as_u64().unwrap() as f64;
        let d = r["draws"].as_u64().unwrap() as f64;
        let l = r["losses"].as_u64().unwrap() as f64;
        eligible &= r["unknown"].as_u64() == Some(0);
        score += 0.5 * (w + 0.5 * d + 1.0) / (w + d + l + 2.0);
    }
    let internal = reports
        .first()
        .filter(|(name, _)| *name == "initial")
        .map(|(_, r)| {
            let w = r["wins"].as_u64().unwrap() as f64;
            let d = r["draws"].as_u64().unwrap() as f64;
            let l = r["losses"].as_u64().unwrap() as f64;
            400.0 * ((w + 0.5 * d + 1.0) / (l + 0.5 * d + 1.0)).log10()
        });
    save_json_new(
        &dir.join("assessment.json"),
        &serde_json::json!({"rules":RULES.as_str(),"model":snapshot.identity,"version":snapshot.version,"references":reports,"complete":eligible,"regularized_combined_score":score,"provisional_internal_elo_vs_initial_64":internal,"site_elo":null,"seconds":started.elapsed().as_secs_f64(),"claim":"small frozen paired panel; selection is provisional, not confirmed progress"}),
    )?;
    if eligible {
        if let Some(elo) = internal {
            let mark = (elo / 100.0).floor() as i64;
            for level in 1..=mark {
                let path = o
                    .output
                    .join("milestones")
                    .join(format!("plus-{}.json", level * 100));
                if !path.exists() {
                    save_json_new(
                        &path,
                        &serde_json::json!({"model":snapshot.identity,"path":snapshot.path,"version":snapshot.version,"internal_elo":elo,"anchor":initial.identity,"budget":64,"evidence":dir,"provisional":true,"site_elo":null}),
                    )?;
                }
            }
        }
        Ok(Some(Ranked { snapshot, score }))
    } else {
        Ok(None)
    }
}
