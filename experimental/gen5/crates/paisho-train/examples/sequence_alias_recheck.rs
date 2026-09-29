//! Verify previously observed tactical aliases against both full frozen banks.
use paisho_ai::*;
use paisho_core::*;
use paisho_train::micro_learning::sequence_memory_from_bytes;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Arc};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    if a.len() != 5 {
        return Err("sequence_alias_recheck OLD_BANK NEW_BANK ALIAS_DIRECTORY OUTPUT.json".into());
    }
    let load = |p: &str| {
        let bytes = fs::read(p)?;
        sequence_memory_from_bytes(
            SequenceMemorySpec {
                path: p.into(),
                sha256: format!("{:x}", Sha256::digest(&bytes)),
            },
            &bytes,
        )
    };
    let old = load(&a[1])?;
    let new = load(&a[2])?;
    let dir = Path::new(&a[3]);
    let manifest: Value = serde_json::from_slice(&fs::read(dir.join("aliases.json"))?)?;
    let mut rows = vec![];
    for w in manifest["witnesses"].as_array().ok_or("alias witnesses")? {
        let load = |field: &str| -> Result<_, Box<dyn std::error::Error>> {
            let r: GameRecord =
                fs::read_to_string(dir.join(w[field].as_str().ok_or("prefix")?))?.parse()?;
            Ok(micro_spatial_state_features(&r.replay()?))
        };
        let a = load("prefix_a")?;
        let b = load("prefix_b")?;
        assert_eq!(&a[..128], &b[..128]);
        let ga = SequenceGeometry::from_state(&a)?;
        let gb = SequenceGeometry::from_state(&b)?;
        assert_ne!(ga, gb);
        let oa = old.context(&a, 0);
        let ob = old.context(&b, 0);
        let na = new.context(&a, 0);
        let nb = new.context(&b, 0);
        assert!(Arc::ptr_eq(&oa, &ob));
        assert!(!Arc::ptr_eq(&na, &nb));
        let differences = na
            .patterns
            .iter()
            .flatten()
            .zip(nb.patterns.iter().flatten())
            .map(|(a, b)| (a - b).abs())
            .fold(0f64, f64::max);
        rows.push(json!({"prefix_a":w["prefix_a"],"prefix_b":w["prefix_b"],"geometry_distance":ga.distance(&gb),"old_same_cache_entry":true,"new_distinct_cache_entries":true,"old_neighbors":oa.neighbors,"new_neighbors_a":na.neighbors,"new_neighbors_b":nb.neighbors,"new_neighbors_differ":na.neighbors!=nb.neighbors,"new_pattern_max_difference":differences,"confidence_a":na.confidence,"confidence_b":nb.confidence}));
    }
    fs::write(&a[4], serde_json::to_vec_pretty(&rows)?)?;
    println!("{} alias pairs verified in the full banks", rows.len());
    Ok(())
}
