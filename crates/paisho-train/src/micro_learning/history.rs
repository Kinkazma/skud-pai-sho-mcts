//! Bounded in-process frozen evaluations; all searches share selfplay's CPU pool.
//! Milestones are provisional internal score-Elo changes, never site conversions.
use super::selfplay::Snapshot;
use super::*;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
    time::{Duration, Instant},
};

// Normal(0,300 Elo) regularization keeps all-win small samples finite.
// Draws contribute half a point. Unfinished pairs cannot qualify for milestones.
fn estimate(scores: &[f64]) -> (f64, f64) {
    let k = std::f64::consts::LN_10 / 400.0;
    let mut elo = 0.0;
    for _ in 0..50 {
        let p = 1.0 / (1.0 + (-k * elo).exp());
        let curvature = k * k * scores.len() as f64 * p * (1.0 - p) + 1.0 / 90000.0;
        let step = (k * (scores.iter().sum::<f64>() - scores.len() as f64 * p) - elo / 90000.0)
            / curvature;
        elo += step;
        if step.abs() < 1e-9 {
            break;
        }
    }
    let p = 1.0 / (1.0 + (-k * elo).exp());
    (
        elo,
        (k * k * scores.len() as f64 * p * (1.0 - p) + 1.0 / 90000.0)
            .sqrt()
            .recip(),
    )
}
fn thresholds(elo: f64, previous: usize, complete: bool) -> Vec<usize> {
    if !complete || !elo.is_finite() || elo < 100.0 {
        return vec![];
    }
    ((previous / 100 + 1)..=(elo / 100.0).floor() as usize)
        .map(|x| x * 100)
        .collect()
}
pub(super) fn run(
    output: PathBuf,
    shared: Arc<RwLock<Arc<Snapshot>>>,
    pool: Arc<rayon::ThreadPool>,
    stop: Arc<AtomicBool>,
    end: Instant,
    interval: Duration,
) -> std::result::Result<(), String> {
    let root = output.join("history");
    std::fs::create_dir(&root).map_err(|e| e.to_string())?;
    let budgets = [32, 64, 128, 256, 512];
    let mut marks = [0usize; 5];
    let mut round = 0;
    let mut next = Instant::now() + interval.min(Duration::from_secs(300));
    while !stop.load(Ordering::Relaxed) && Instant::now() < end {
        if Instant::now() < next {
            std::thread::sleep(Duration::from_millis(100));
            continue;
        }
        let snapshot = shared.read().map_err(|e| e.to_string())?.clone();
        if snapshot.version == 0 {
            next = Instant::now() + Duration::from_secs(1);
            continue;
        }
        let directory = root.join(format!("round-{round:04}"));
        std::fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let model = directory.join("model.json");
        std::fs::copy(
            output
                .join("models")
                .join(format!("model-{:06}.json", snapshot.version)),
            &model,
        )
        .map_err(|e| e.to_string())?;
        for (index, budget) in budgets.into_iter().enumerate() {
            if stop.load(Ordering::Relaxed) || Instant::now() >= end {
                break;
            }
            let compare = directory.join(format!("mcts{budget}"));
            compare_micro(
                &model,
                &output.join("parent.json"),
                &compare,
                MicroCompareOptions {
                    pairs: 16,
                    workers: 2,
                    varied_setups: true,
                    simulations: budget,
                    move_ms: 0,
                    game_seconds: 60.0,
                    seconds: end
                        .saturating_duration_since(Instant::now())
                        .as_secs_f64()
                        .min(180.0),
                    decisions: 600,
                    seed: 87000 + round * 100,
                },
                &pool,
            )
            .map_err(|e| e.to_string())?;
            let report: serde_json::Value = serde_json::from_slice(
                &fs::read(compare.join("report.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let games = report["games"]
                .as_array()
                .ok_or("missing comparison games")?;
            let scores: Vec<_> = games
                .chunks_exact(2)
                .filter(|pair| {
                    pair.iter()
                        .all(|g| g["score"].as_f64().is_some() && g["error"].is_null())
                })
                .flat_map(|pair| pair.iter().map(|g| g["score"].as_f64().unwrap()))
                .collect();
            let complete = scores.len() == 32;
            let (elo, sd) = estimate(&scores);
            let summary = serde_json::json!({"rules":paisho_core::RuleProfileId::CURRENT.as_str(),"model_version":snapshot.version,"model_identity":snapshot.identity,"budget":budget,"internal_elo_delta":elo,"approximate_interval":[elo-1.96*sd,elo+1.96*sd],"complete_pairs":scores.len()/2,"eligible_for_milestone":complete,"scale":"internal score-Elo, logistic 400; fixed starting Gen4 at same budget = zero delta","canonical_league_rating":false,"site_conversion":null,"provisional":true,"uncertainty":"curvature approximation, correlated paired outcomes and repeated checkpoint selection can widen uncertainty; no promotion claim","wins":report["wins"],"draws":report["draws"],"losses":report["losses"],"unknown":report["unknown"]});
            save_json_new(&compare.join("internal-elo.json"), &summary)
                .map_err(|e| e.to_string())?;
            for threshold in thresholds(elo, marks[index], complete) {
                let mark = root.join(format!("mcts{budget}-plus-{threshold:04}"));
                fs::create_dir(&mark).map_err(|e| e.to_string())?;
                fs::copy(&model, mark.join("model.json")).map_err(|e| e.to_string())?;
                save_json_new(&mark.join("mark.json"),&serde_json::json!({"threshold":threshold,"comparison":compare,"observation":summary,"stops_training":false})).map_err(|e|e.to_string())?;
                marks[index] = threshold;
            }
        }
        round += 1;
        next = Instant::now() + interval;
    }
    save_json_new(&root.join("report.json"),&serde_json::json!({"rounds":round,"budgets":budgets,"milestones":marks,"automatic_resume":false})).map_err(|e|e.to_string())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn finite_symmetric_internal_scale() {
        let (a, _) = estimate(&[1.0; 32]);
        let (b, _) = estimate(&[0.0; 32]);
        assert!(a > 100.0 && a.is_finite());
        assert!((a + b).abs() < 1e-8);
        assert_eq!(estimate(&[0.5; 32]).0, 0.0);
    }
    #[test]
    fn marks_require_complete_data_and_are_idempotent() {
        assert_eq!(thresholds(320.0, 100, true), vec![200, 300]);
        assert!(thresholds(99.0, 0, true).is_empty());
        assert!(thresholds(400.0, 0, false).is_empty());
        assert!(thresholds(200.0, 300, true).is_empty());
    }
}
