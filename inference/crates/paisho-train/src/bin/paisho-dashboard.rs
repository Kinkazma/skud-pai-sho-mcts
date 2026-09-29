use std::error::Error;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use paisho_train::{
    build_dashboard_snapshot, render_dashboard_html, write_dashboard_html, DashboardVerificationV1,
};

type BoxError = Box<dyn Error + Send + Sync>;

struct Options {
    curriculum_directory: PathBuf,
    output: PathBuf,
    watch_seconds: Option<u64>,
    verification: DashboardVerificationV1,
}

impl Options {
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, BoxError> {
        let arguments = arguments.collect::<Vec<_>>();
        if arguments
            .iter()
            .any(|argument| argument == "--help" || argument == "-h")
        {
            print_usage();
            std::process::exit(0);
        }
        let mut curriculum_directory = None;
        let mut output = None;
        let mut watch_seconds = None;
        let mut verification = DashboardVerificationV1::Integrity;
        let mut index = 0;
        while index < arguments.len() {
            let flag = &arguments[index];
            if flag == "--verify" {
                verification = DashboardVerificationV1::Semantic;
                index += 1;
                continue;
            }
            let value = arguments
                .get(index + 1)
                .ok_or_else(|| format!("valeur manquante après {flag}"))?;
            match flag.as_str() {
                "--curriculum-dir" => curriculum_directory = Some(PathBuf::from(value)),
                "--output" => output = Some(PathBuf::from(value)),
                "--watch-seconds" => {
                    let seconds = value.parse::<u64>()?;
                    if seconds == 0 {
                        return Err("--watch-seconds doit être strictement positif".into());
                    }
                    watch_seconds = Some(seconds);
                }
                _ => return Err(format!("option inconnue {flag}").into()),
            }
            index += 2;
        }
        if verification == DashboardVerificationV1::Semantic && watch_seconds.is_some() {
            return Err("--verify et --watch-seconds sont incompatibles : le rejeu complet n’est pas un rafraîchissement léger".into());
        }
        let curriculum_directory =
            curriculum_directory.ok_or("option requise : --curriculum-dir DOSSIER")?;
        let output = output.unwrap_or_else(|| default_output(&curriculum_directory));
        Ok(Self {
            curriculum_directory,
            output,
            watch_seconds,
            verification,
        })
    }
}

fn main() -> Result<(), BoxError> {
    let options = Options::parse(std::env::args().skip(1))?;
    let mut previous_summary = None;
    loop {
        let snapshot =
            build_dashboard_snapshot(&options.curriculum_directory, options.verification)?;
        let summary = (
            snapshot.completed_generations,
            snapshot.games.total_attempted(),
            snapshot.current_stage.clone(),
        );
        let html = render_dashboard_html(&snapshot, options.watch_seconds);
        write_dashboard_html(&options.output, &html)?;
        if previous_summary.as_ref() != Some(&summary) {
            println!("dashboard={}", options.output.display());
            println!("completed_generations={}", snapshot.completed_generations);
            println!("attempted_games={}", snapshot.games.total_attempted());
            println!("current_stage={}", snapshot.current_stage);
            previous_summary = Some(summary);
        }
        let Some(seconds) = options.watch_seconds else {
            return Ok(());
        };
        thread::sleep(Duration::from_secs(seconds));
    }
}

fn default_output(curriculum_directory: &Path) -> PathBuf {
    let campaign_directory = curriculum_directory
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let campaign_name = campaign_directory
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("paisho-campaign"))
        .to_string_lossy();
    campaign_directory
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{campaign_name}-dashboard.html"))
}

fn print_usage() {
    eprintln!(
        "usage: paisho-dashboard --curriculum-dir DOSSIER [--output FICHIER] \
         [--watch-seconds N | --verify]\n\
         \n\
         Sans option supplémentaire, produit une vue HTML ponctuelle.\n\
         --watch-seconds régénère cette vue périodiquement sans modifier les archives.\n\
         --verify rejoue intégralement les preuves avant de produire la vue."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_a_sibling_dashboard() {
        let options = Options::parse(
            ["--curriculum-dir", "/tmp/campaign/curriculum"]
                .into_iter()
                .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(
            options.output,
            PathBuf::from("/tmp/campaign-dashboard.html")
        );
        assert_eq!(options.watch_seconds, None);
    }

    #[test]
    fn semantic_replay_is_not_repeated_as_a_watch_loop() {
        let result = Options::parse(
            [
                "--curriculum-dir",
                "/tmp/campaign/curriculum",
                "--verify",
                "--watch-seconds",
                "5",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        assert!(result.is_err());
    }
}
