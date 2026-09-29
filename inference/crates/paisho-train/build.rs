use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

const SOURCE_INPUTS: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "crates/paisho-core/Cargo.toml",
    "crates/paisho-core/src",
    "crates/paisho-ai/Cargo.toml",
    "crates/paisho-ai/src",
    "crates/paisho-model/Cargo.toml",
    "crates/paisho-model/src",
    "crates/paisho-mpsgraph-client/Cargo.toml",
    "crates/paisho-mpsgraph-client/src",
    "crates/paisho-rating/Cargo.toml",
    "crates/paisho-rating/src",
    "crates/paisho-replay/Cargo.toml",
    "crates/paisho-replay/src",
    "crates/paisho-train/Cargo.toml",
    "crates/paisho-train/build.rs",
    "crates/paisho-train/src",
];

fn main() {
    let manifest =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR"));
    let workspace = manifest.join("../..");
    for relative in SOURCE_INPUTS {
        println!(
            "cargo:rerun-if-changed={}",
            workspace.join(relative).display()
        );
    }
    let source_fingerprint = fingerprint_sources(&workspace)
        .expect("paisho-train build inputs must be readable for fingerprinting");
    println!("cargo:rustc-env=PAISHO_BUILD_SOURCE_SHA256={source_fingerprint}");

    let revision = git(&manifest, &["rev-parse", "HEAD"]);
    println!("cargo:rustc-env=PAISHO_BUILD_GIT_REVISION={revision}");
    let dirty = !git(&manifest, &["status", "--porcelain"]).is_empty();
    println!("cargo:rustc-env=PAISHO_BUILD_GIT_DIRTY={dirty}");

    let head_path = git(&manifest, &["rev-parse", "--git-path", "HEAD"]);
    println!(
        "cargo:rerun-if-changed={}",
        absolute(&manifest, &head_path).display()
    );
    if let Some(reference) = git_optional(&manifest, &["symbolic-ref", "-q", "HEAD"]) {
        let reference_path = git(&manifest, &["rev-parse", "--git-path", &reference]);
        println!(
            "cargo:rerun-if-changed={}",
            absolute(&manifest, &reference_path).display()
        );
    }
}

fn fingerprint_sources(workspace: &Path) -> io::Result<String> {
    let mut files = Vec::new();
    for relative in SOURCE_INPUTS {
        collect_files(&workspace.join(relative), &mut files)?;
    }
    files.sort();
    files.dedup();

    let mut hasher = Sha256::new();
    hasher.update(b"PAISHO-BUILD-SOURCE-V1\0");
    for path in files {
        let relative = path
            .strip_prefix(workspace)
            .map_err(|source| io::Error::new(io::ErrorKind::InvalidInput, source))?;
        let name = relative.to_string_lossy().replace('\\', "/");
        let bytes = fs::read(&path)?;
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    Ok(hex_digest(hasher.finalize().into()))
}

fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    if path.is_file() {
        files.push(path.to_owned());
        return Ok(());
    }
    let mut entries = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(&entry.path(), files)?;
        } else if file_type.is_file() {
            files.push(entry.path());
        }
    }
    Ok(())
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing to a String cannot fail");
    }
    text
}

fn git(directory: &Path, arguments: &[&str]) -> String {
    git_optional(directory, arguments)
        .unwrap_or_else(|| panic!("git {:?} failed while stamping paisho-train", arguments))
}

fn git_optional(directory: &Path, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .output()
        .expect("git is required to build paisho-train");
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).expect("git output is UTF-8"))
        .map(|text| text.trim().to_owned())
}

fn absolute(manifest: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_owned()
    } else {
        manifest.join(path)
    }
}
