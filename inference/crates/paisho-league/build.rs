use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo sets CARGO_MANIFEST_DIR");
    for relative in [
        "../../Cargo.toml",
        "../../Cargo.lock",
        "../paisho-core/Cargo.toml",
        "../paisho-core/src",
        "../paisho-ai/Cargo.toml",
        "../paisho-ai/src",
        "../paisho-rating/Cargo.toml",
        "../paisho-rating/src",
        "Cargo.toml",
        "build.rs",
        "src",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            Path::new(&manifest).join(relative).display()
        );
    }
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

fn git(directory: &str, arguments: &[&str]) -> String {
    git_optional(directory, arguments)
        .unwrap_or_else(|| panic!("git {:?} failed while stamping paisho-league", arguments))
}

fn git_optional(directory: &str, arguments: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(directory)
        .args(arguments)
        .output()
        .expect("git is required to build paisho-league");
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).expect("git output is UTF-8"))
        .map(|text| text.trim().to_owned())
}

fn absolute(manifest: &str, path: &str) -> std::path::PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_owned()
    } else {
        Path::new(manifest).join(path)
    }
}
