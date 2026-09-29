//! Only completed block commits are recovery points. Mutable live status is not.
use super::BoxError;
use paisho_train::CurriculumTierV1;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub generation: u64,
    pub checkpoint: PathBuf,
    pub checkpoint_sha256: String,
    pub training_step: u64,
    pub champion: PathBuf,
    pub tier: CurriculumTierV1,
    pub games: u64,
    pub examples: u64,
    pub attempt_directory: PathBuf,
}

pub fn write_new(path: &Path, value: &impl Serialize) -> Result<(), BoxError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("pending-{}", unique_id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(&serde_json::to_vec_pretty(value)?)?;
    file.sync_all()?;
    // A hard link publishes atomically without replacing an existing artifact.
    fs::hard_link(&temp, path)?;
    fs::remove_file(&temp)?;
    if let Some(parent) = path.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}

pub fn status(root: &Path, value: &impl Serialize) -> Result<(), BoxError> {
    let temp = root.join("live-status.pending");
    fs::write(&temp, serde_json::to_vec(value)?)?;
    fs::rename(temp, root.join("live-status.json"))?;
    Ok(())
}

pub fn unique_id() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after epoch")
        .as_nanos()
}

pub fn plan(root: &Path, identity: Value) -> Result<(), BoxError> {
    fs::create_dir_all(root)?;
    let path = root.join("live-plan.json");
    if path.exists() {
        let stored: Value = serde_json::from_slice(&fs::read(path)?)?;
        if stored != identity {
            return Err(
                "live campaign settings differ from its saved plan; use a new campaign directory"
                    .into(),
            );
        }
    } else {
        write_new(&path, &identity)?;
    }
    Ok(())
}

pub fn commit(root: &Path, block: &Block) -> Result<(), BoxError> {
    write_new(
        &root
            .join("blocks")
            .join(format!("block-{:020}.json", block.generation)),
        block,
    )
}

pub fn latest(root: &Path) -> Result<Option<Block>, BoxError> {
    let dir = root.join("blocks");
    if !dir.exists() {
        return Ok(None);
    }
    let mut paths = fs::read_dir(dir)?
        .map(|r| r.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()?;
    paths.retain(|p| {
        p.extension().is_some_and(|x| x == "json")
            && p.file_name()
                .is_some_and(|x| x.to_string_lossy().starts_with("block-"))
    });
    paths.sort();
    let Some(path) = paths.last() else {
        return Ok(None);
    };
    let block: Block = serde_json::from_slice(&fs::read(path)?)?;
    let metadata = paisho_mpsgraph_client::read_checkpoint_metadata(&block.checkpoint)?;
    if paisho_train::live_learner::hex(metadata.content_sha256()) != block.checkpoint_sha256
        || metadata.generation() != block.generation
        || metadata.training_step() != block.training_step
    {
        return Err("durable block checkpoint does not match its commit".into());
    }
    Ok(Some(block))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publication_never_overwrites_and_live_status_is_not_a_commit() {
        let root = std::env::temp_dir().join(format!("paisho-live-journal-{}", unique_id()));
        fs::create_dir(&root).unwrap();
        let path = root.join("record.json");
        write_new(&path, &serde_json::json!({"n":1})).unwrap();
        assert!(write_new(&path, &serde_json::json!({"n":2})).is_err());
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(path).unwrap()).unwrap()["n"],
            1
        );
        status(&root, &serde_json::json!({"generation":4})).unwrap();
        assert!(latest(&root).unwrap().is_none());
        fs::remove_dir_all(root).unwrap();
    }
}
