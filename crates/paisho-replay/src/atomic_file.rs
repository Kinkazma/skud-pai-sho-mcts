use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn write_new(destination: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = parent_directory(destination);
    fs::create_dir_all(parent)?;
    let temporary = temporary_path(destination)?;
    let result = write_and_publish(&temporary, destination, bytes);
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn temporary_path(destination: &Path) -> io::Result<PathBuf> {
    let file_name = destination.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{} has no file name", destination.display()),
        )
    })?;
    let counter = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    Ok(destination.with_file_name(format!(
        ".{}.partial-{}-{counter}",
        file_name.to_string_lossy(),
        std::process::id()
    )))
}

fn write_and_publish(temporary: &Path, destination: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::hard_link(temporary, destination)?;
    fs::remove_file(temporary)?;
    File::open(parent_directory(destination))?.sync_all()?;
    Ok(())
}

fn parent_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}
