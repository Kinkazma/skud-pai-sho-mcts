use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{invalid, EvaluationArchiveError};

pub(super) const MANIFEST_FILE: &str = "MANIFEST.sha256";

pub(super) fn write_json_new<T: Serialize>(
    path: &Path,
    value: &T,
) -> Result<(), EvaluationArchiveError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_new_synced(path, &bytes)
}

pub(super) fn write_new_synced(path: &Path, bytes: &[u8]) -> Result<(), EvaluationArchiveError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn write_manifest(directory: &Path) -> Result<(), EvaluationArchiveError> {
    let files = archive_files(directory)?;
    let mut text = String::new();
    use core::fmt::Write as _;
    for relative in files {
        writeln!(
            text,
            "{}  {}",
            sha256_hex(&directory.join(&relative))?,
            relative.display()
        )
        .expect("writing a manifest to String cannot fail");
    }
    write_new_synced(&directory.join(MANIFEST_FILE), text.as_bytes())
}

pub(super) fn verify_manifest(directory: &Path) -> Result<(), EvaluationArchiveError> {
    let manifest_path = directory.join(MANIFEST_FILE);
    if !fs::symlink_metadata(&manifest_path)?.file_type().is_file() {
        return Err(invalid("evaluation manifest must be a regular file"));
    }
    let manifest = fs::read_to_string(manifest_path)?;
    let mut expected = BTreeSet::new();
    for line in manifest.lines() {
        let (digest, relative_text) = line
            .split_once("  ")
            .ok_or_else(|| invalid("malformed evaluation manifest row"))?;
        if !is_lower_hex_digest(digest) || !safe_relative_path(relative_text) {
            return Err(invalid("unsafe or malformed evaluation manifest entry"));
        }
        let relative = PathBuf::from(relative_text);
        if !expected.insert(relative.clone()) || sha256_hex(&directory.join(relative))? != digest {
            return Err(invalid("evaluation manifest checksum or path mismatch"));
        }
    }
    let actual: BTreeSet<_> = archive_files(directory)?.into_iter().collect();
    if actual != expected {
        return Err(invalid(
            "evaluation manifest file set differs from directory",
        ));
    }
    Ok(())
}

fn archive_files(directory: &Path) -> Result<Vec<PathBuf>, EvaluationArchiveError> {
    let mut files = Vec::new();
    collect_files(directory, directory, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), EvaluationArchiveError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(invalid("evaluation archives cannot contain symbolic links"));
        }
        if file_type.is_dir() {
            collect_files(root, &path, files)?;
        } else if file_type.is_file() && entry.file_name() != MANIFEST_FILE {
            files.push(
                path.strip_prefix(root)
                    .map_err(|_| invalid("evaluation archive path escaped its root"))?
                    .to_owned(),
            );
        } else if !file_type.is_file() {
            return Err(invalid("evaluation archive contains a non-regular entry"));
        }
    }
    Ok(())
}

pub(super) fn sha256_hex(path: &Path) -> Result<String, EvaluationArchiveError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1_024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let mut text = String::with_capacity(64);
    for byte in hasher.finalize() {
        use core::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing a digest to String cannot fail");
    }
    Ok(text)
}

pub(super) fn is_lower_hex_digest(text: &str) -> bool {
    text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn safe_relative_path(text: &str) -> bool {
    let path = Path::new(text);
    !text.is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

pub(super) fn sync_tree(directory: &Path) -> Result<(), EvaluationArchiveError> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            sync_tree(&entry.path())?;
        } else {
            File::open(entry.path())?.sync_all()?;
        }
    }
    File::open(directory)?.sync_all()?;
    Ok(())
}

pub(super) struct TemporaryDirectory {
    pub path: PathBuf,
    published: bool,
}

impl TemporaryDirectory {
    pub fn publish(mut self, destination: &Path) -> Result<(), EvaluationArchiveError> {
        fs::rename(&self.path, destination)?;
        self.published = true;
        File::open(destination.parent().unwrap_or_else(|| Path::new(".")))?.sync_all()?;
        Ok(())
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        if !self.published {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

pub(super) fn temporary_directory(
    parent: &Path,
    stem: &str,
) -> Result<TemporaryDirectory, EvaluationArchiveError> {
    for attempt in 0..1_000_u32 {
        let path = parent.join(format!("{stem}-{}-{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => {
                return Ok(TemporaryDirectory {
                    path,
                    published: false,
                })
            }
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => return Err(source.into()),
        }
    }
    Err(invalid("cannot reserve a temporary evaluation directory"))
}
