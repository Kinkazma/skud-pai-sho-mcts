//! Streaming-safe coverage: aligned time bins merge, never slide or split.
use super::Entry;

pub(super) fn select(entries: &[&Entry], keep: usize, recovery: usize) -> Vec<usize> {
    let n = entries.len();
    if n <= keep {
        return (0..n).collect();
    }
    if keep == 0 {
        return vec![];
    }
    let mut order: Vec<_> = (0..n).collect();
    order.sort_by_key(|&i| (entries[i].elapsed_millis, entries[i].ordinal));
    if keep < 8 {
        let mut chosen = vec![recovery];
        for &i in order.iter().rev() {
            if chosen.len() == keep {
                break;
            }
            if !chosen.contains(&i) {
                chosen.push(i);
            }
        }
        chosen.sort_unstable();
        return chosen;
    }
    let first = order[0];
    let last = order[n - 1];
    let origin = entries[first].elapsed_millis;
    let span = entries[last].elapsed_millis - origin;
    let bins = (keep / 2).saturating_sub(2).max(1) as u64;
    let minimum_width = (span / bins).saturating_add(1);
    let width = minimum_width
        .checked_next_power_of_two()
        .unwrap_or(u64::MAX);
    let distance = |a: usize, b: usize| {
        entries[a]
            .elapsed_millis
            .abs_diff(entries[b].elapsed_millis)
    };
    let mut chosen = vec![first];
    for i in [recovery, last] {
        if !chosen.contains(&i) {
            chosen.push(i);
        }
    }
    // Quality nominations remain separated by reference and budget. A dense
    // cluster of strong neighbours contributes one spaced nominee, not all.
    let keys: std::collections::BTreeSet<_> =
        entries.iter().flat_map(|e| e.results.keys()).collect();
    let mut rankings = vec![];
    for key in keys {
        let mut ranked: Vec<_> = (0..n)
            .filter_map(|i| {
                let result = entries[i].results.get(key)?;
                let score = result.score()?;
                (score > 0.5).then_some((i, score, result.attempts.iter().sum::<usize>()))
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| b.2.cmp(&a.2))
                .then_with(|| entries[a.0].ordinal.cmp(&entries[b.0].ordinal))
        });
        rankings.push(ranked);
    }
    let mut champions = vec![];
    let depth = rankings.iter().map(Vec::len).max().unwrap_or(0);
    'rank: for d in 0..depth {
        for ranking in &rankings {
            if let Some(&(i, _, _)) = ranking.get(d) {
                if !champions.contains(&i) && champions.iter().all(|&j| distance(i, j) >= width) {
                    champions.push(i);
                    if !chosen.contains(&i) {
                        chosen.push(i);
                    }
                    if champions.len() == keep / 4 {
                        break 'rank;
                    }
                }
            }
        }
    }
    let mut buckets = std::collections::BTreeMap::<u64, Vec<usize>>::new();
    for &i in &order {
        buckets
            .entry((entries[i].elapsed_millis - origin) / width)
            .or_default()
            .push(i);
    }
    for (bucket, indices) in buckets {
        if indices.iter().any(|i| chosen.contains(i)) {
            continue;
        }
        let middle = origin
            .saturating_add(bucket.saturating_mul(width))
            .saturating_add(width / 2);
        let i = *indices
            .iter()
            .min_by_key(|&&i| {
                (
                    entries[i].elapsed_millis.abs_diff(middle),
                    entries[i].ordinal,
                )
            })
            .unwrap();
        chosen.push(i);
    }
    assert!(chosen.len() <= keep);
    // Balance remaining places across occupied bins, then maximize spacing.
    // A global farthest-first refill alone locks too many early checkpoints in.
    // Every occupied bin above stays protected after repeated prior deletions.
    let mut used = vec![false; n];
    let bucket_of: Vec<_> = entries
        .iter()
        .map(|e| (e.elapsed_millis - origin) / width)
        .collect();
    let mut occupancy = std::collections::BTreeMap::<u64, usize>::new();
    for &i in &chosen {
        used[i] = true;
        *occupancy.entry(bucket_of[i]).or_default() += 1;
    }
    let mut nearest: Vec<_> = (0..n)
        .map(|i| chosen.iter().map(|&j| distance(i, j)).min().unwrap())
        .collect();
    while chosen.len() < keep {
        let i = (0..n)
            .filter(|&i| !used[i])
            .max_by_key(|&i| {
                (
                    usize::MAX - occupancy.get(&bucket_of[i]).copied().unwrap_or(0),
                    nearest[i],
                    usize::MAX - entries[i].ordinal,
                )
            })
            .unwrap();
        chosen.push(i);
        used[i] = true;
        *occupancy.entry(bucket_of[i]).or_default() += 1;
        for j in 0..n {
            nearest[j] = nearest[j].min(distance(i, j));
        }
    }
    chosen.sort_unstable();
    chosen
}
