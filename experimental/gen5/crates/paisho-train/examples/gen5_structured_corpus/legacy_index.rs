//! Audit old corpus labels and index exact/symmetric states without rewriting it.
use super::{
    branches,
    facts::{self, Facts},
    hash, write_json, Result,
};
use paisho_core::*;
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, Write},
    path::Path,
    time::Instant,
};

pub fn index(plan: &str, out: &Path) -> Result<()> {
    fs::create_dir(out)?;
    let start = Instant::now();
    let plan: Value = serde_json::from_slice(&fs::read(plan)?)?;
    let census = Path::new(plan["census"].as_str().ok_or("census")?);
    let branches = Path::new(plan["branches"].as_str().ok_or("branches")?);
    let roots: Vec<Value> = serde_json::from_slice(&fs::read(census.join("roots.json"))?)?;
    let mut index = std::io::BufWriter::new(fs::File::create(out.join("positions.jsonl"))?);
    let mut corrections =
        std::io::BufWriter::new(fs::File::create(out.join("component-corrections.jsonl"))?);
    let mut cached = BTreeMap::new();
    let mut root_changes = 0;
    let mut branch_changes = 0;
    let mut rows = 0;
    for (line, r) in roots.iter().enumerate() {
        let record = branches::root_record(census, r)?;
        let p = record.replay()?;
        let f = Facts::new(&p);
        let components = f.components(&p);
        if r["position"]["components_host_guest"] != json!(components) {
            root_changes += 1;
            serde_json::to_writer(
                &mut corrections,
                &json!({"file":"roots.json","line":line,
                "root":r["key"],"old":r["position"]["components_host_guest"],"corrected":components}),
            )?;
            writeln!(corrections)?;
        }
        write_position(&mut index, &p, r, "roots.json", line, "root")?;
        cached.insert(r["key"].as_str().unwrap().to_owned(), (p, components, r));
    }
    for (line, text) in std::io::BufReader::new(fs::File::open(branches.join("branches.jsonl"))?)
        .lines()
        .enumerate()
    {
        let row: Value = serde_json::from_str(&text?)?;
        let (p, before, meta) = &cached[row["root"].as_str().ok_or("root")?];
        if meta["group"] != row["group"] || meta["held_out"] != row["held_out"] {
            return Err("branch provenance changed".into());
        }
        let mut q = p.clone();
        q.apply(row["action"].as_str().ok_or("action")?.parse()?)?;
        if row["position_sha256"] != facts::position_hash(&q)
            || row["outcome"] != facts::result(q.outcome(), p.to_move())
        {
            return Err("branch outcome changed".into());
        }
        let after = Facts::new(&q).components(&q);
        if row["components_before_host_guest"] != json!(before)
            || row["components_after_host_guest"] != json!(after)
        {
            branch_changes += 1;
            serde_json::to_writer(
                &mut corrections,
                &json!({"file":"branches.jsonl","line":line,
                "root":row["root"],"action":row["action"],"old_before":row["components_before_host_guest"],
                "old_after":row["components_after_host_guest"],"corrected_before":before,"corrected_after":after}),
            )?;
            writeln!(corrections)?;
        }
        write_position(&mut index, &q, &row, "branches.jsonl", line, "successor")?;
        rows += 1;
    }
    index.flush()?;
    corrections.flush()?;
    write_json(
        out.join("summary.json"),
        &json!({"roots":roots.len(),"branches":rows,"corrected_root_components":root_changes,
        "corrected_branch_components":branch_changes,"all_outcomes_and_states_unchanged":true,"old_files_not_modified":true,
        "roots_sha256":hash(&fs::read(census.join("roots.json"))?),"branches_sha256":hash(&fs::read(branches.join("branches.jsonl"))?),
        "seconds":start.elapsed().as_secs_f64()}),
    )
}
fn write_position(
    w: &mut impl Write,
    p: &Position,
    row: &Value,
    file: &str,
    line: usize,
    role: &str,
) -> Result<()> {
    serde_json::to_writer(
        &mut *w,
        &json!({"file":file,"line":line,"role":role,"group":row["group"],"held_out":row["held_out"],
        "exact":facts::position_hash(p),"symmetry":facts::symmetry_hash(p)}),
    )?;
    writeln!(w)?;
    Ok(())
}
