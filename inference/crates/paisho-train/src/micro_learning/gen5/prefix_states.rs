//! A coverage reader keeps exact states from its current four records in RAM.
//! No weights, hash approximation, shared lock, or eager validation of later moves.
use paisho_core::{GameRecord, Position, ReplayError};
use std::collections::VecDeque;

pub(super) struct PrefixStates<'a> {
    record: &'a GameRecord,
    index: usize,
    last: Position,
    recent: VecDeque<(usize, Position)>,
    applications: usize,
}
impl<'a> PrefixStates<'a> {
    pub fn new(record: &'a GameRecord) -> Self {
        let last = record.initial_position();
        Self {
            record,
            index: 0,
            recent: VecDeque::from([(0, last.clone())]),
            last,
            applications: 0,
        }
    }
    pub fn at(&mut self, end: usize) -> Result<Position, ReplayError> {
        assert!(end <= self.record.actions().len());
        if let Some((_, p)) = self.recent.iter().find(|(i, _)| *i == end) {
            return Ok(p.clone());
        }
        if end < self.index {
            // Unusually reordered lessons outside the bounded window keep the
            // original replay path, without invalidating the forward cursor.
            self.applications += end;
            return super::cases::prefix(self.record, end).replay();
        }
        while self.index < end {
            let action = self.record.actions()[self.index];
            let mut next = self.last.clone();
            self.applications += 1;
            next.apply(action).map_err(|source| ReplayError {
                action_number: self.index + 1,
                action,
                source,
            })?;
            self.index += 1;
            self.last = next;
            if self.recent.len() == 64 {
                self.recent.pop_front();
            }
            self.recent.push_back((self.index, self.last.clone()));
        }
        Ok(self.last.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_order_eviction_and_late_errors_are_exact() {
        let mut r: GameRecord =
            include_str!("../../../../paisho-ai/tests/fixtures/micro-alias-1-a.psr")
                .parse()
                .unwrap();
        let length = r.actions().len();
        assert!(length > 64);
        r.push(paisho_core::Action::Arrange {
            from: paisho_core::NORTH_GATE,
            to: paisho_core::NORTH_GATE,
        });
        let mut cached = PrefixStates::new(&r);
        for n in (0..=length)
            .chain((0..=length).rev())
            .chain([length + 1, length, length + 1, 0])
        {
            assert_eq!(cached.at(n), super::super::cases::prefix(&r, n).replay());
            assert!(cached.recent.len() <= 64);
        }
        let mut ordered = PrefixStates::new(&r);
        for n in 0..=length {
            ordered.at(n).unwrap();
        }
        assert_eq!(ordered.applications, length);
    }
}

pub fn benchmark(
    manifest: &std::path::Path,
    out: &std::path::Path,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    use std::{fs, io::Read, time::Instant};
    let m: serde_json::Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let mut rows = vec![];
    for row in m["rows"].as_array().ok_or("rows")? {
        let bytes = fs::read(row["path"].as_str().ok_or("path")?)?;
        if format!("{:x}", Sha256::digest(&bytes)) != row["sha256"] {
            return Err("bundle hash".into());
        }
        let mut raw = vec![];
        flate2::read::GzDecoder::new(bytes.as_slice()).read_to_end(&mut raw)?;
        let b: serde_json::Value = serde_json::from_slice(&raw)?;
        let r: GameRecord = b["psr"].as_str().ok_or("psr")?.parse()?;
        let indices = b["lessons"]
            .as_array()
            .ok_or("lessons")?
            .iter()
            .map(|l| {
                l["decision"]
                    .as_u64()
                    .ok_or("decision")
                    .and_then(|i| i.checked_sub(1).ok_or("decision zero"))
                    .map(|i| i as usize)
            })
            .collect::<Result<Vec<_>, _>>()?;
        rows.push((r, indices));
    }
    let expected = rows
        .iter()
        .map(|(r, indices)| {
            indices
                .iter()
                .map(|&i| super::cases::prefix(r, i).replay())
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut results = vec![];
    for mode in ["original", "cached", "cached", "original"] {
        let start = Instant::now();
        let mut applications = 0;
        let actual = rows
            .iter()
            .map(|(r, indices)| {
                let mut cache = PrefixStates::new(r);
                let v = indices
                    .iter()
                    .map(|&i| {
                        if mode == "cached" {
                            cache.at(i)
                        } else {
                            applications += i;
                            super::cases::prefix(r, i).replay()
                        }
                    })
                    .collect::<Result<Vec<_>, _>>();
                applications += cache.applications;
                v
            })
            .collect::<Result<Vec<_>, _>>()?;
        let seconds = start.elapsed().as_secs_f64();
        assert_eq!(actual, expected);
        results.push(serde_json::json!({"mode":mode,"seconds":seconds,"applications":applications,"states_exact":true}));
    }
    let report = serde_json::json!({"bundles":rows.len(),"lessons":rows.iter().map(|(_,i)|i.len()).sum::<usize>(),"results":results,"maximum_states_per_record":65,"maximum_resident_records":4});
    fs::write(out, serde_json::to_vec_pretty(&report)?)?;
    Ok(report)
}
