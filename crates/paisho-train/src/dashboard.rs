use core::fmt;
use core::fmt::Write as _;
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use crate::{
    ContextualEloPointV1, CurriculumArchiveError, CurriculumCampaignArchive, CurriculumDecisionV1,
    CurriculumEvidenceV1, CurriculumTierV1, EvaluationArchiveError, EvaluationCampaignArchive,
    EvaluationEstimateV1, EvaluationRunIdentityV1, GenerationArchiveError,
    GenerationCampaignArchive, GenerationOutcomeV1, PromotionArchiveError,
    PromotionCampaignArchive, PromotionCampaignConclusion,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DashboardVerificationV1 {
    Integrity,
    Semantic,
}

impl DashboardVerificationV1 {
    const fn label(self) -> &'static str {
        match self {
            Self::Integrity => "Intégrité des archives vérifiée",
            Self::Semantic => "Archives intégralement rejouées et vérifiées",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DashboardCheckpointV1 {
    pub generation: u64,
    pub training_step: u64,
    pub sha256: String,
    pub path: PathBuf,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DashboardGameCountsV1 {
    pub actor_attempted: u64,
    pub actor_retained: u64,
    pub promotion_attempted: u64,
    pub promotion_eligible: u64,
    pub evaluation_attempted: u64,
    pub evaluation_eligible: u64,
}

impl DashboardGameCountsV1 {
    pub fn total_attempted(self) -> u64 {
        self.actor_attempted
            .saturating_add(self.promotion_attempted)
            .saturating_add(self.evaluation_attempted)
    }

    pub fn total_retained_or_eligible(self) -> u64 {
        self.actor_retained
            .saturating_add(self.promotion_eligible)
            .saturating_add(self.evaluation_eligible)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DashboardEloObservationV1 {
    pub generation: u64,
    pub opponent: String,
    pub protocol_key: String,
    pub protocol_label: String,
    pub contextual_elo: ContextualEloPointV1,
    pub davidson: Option<EvaluationEstimateV1>,
    pub attempted_pairs: u64,
    pub eligible_pairs: u64,
    pub excluded_pairs: u64,
    pub wins: u64,
    pub draws: u64,
    pub losses: u64,
}

impl DashboardEloObservationV1 {
    pub const fn finite_elo(&self) -> Option<f64> {
        match self.contextual_elo {
            ContextualEloPointV1::Finite { elo } => Some(elo),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DashboardEloSeriesV1 {
    pub protocol_key: String,
    pub protocol_label: String,
    pub observations: Vec<DashboardEloObservationV1>,
    pub best: Option<DashboardEloObservationV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DashboardGenerationV1 {
    pub generation: u64,
    pub opponent: String,
    pub stage: String,
    pub actor_attempts: Option<u64>,
    pub actor_retained: Option<u64>,
    pub actor_interrupted: Option<u64>,
    pub training_examples: Option<u64>,
    pub training_step: Option<u64>,
    pub promotion_attempted_pairs: Option<u64>,
    pub promotion_eligible_pairs: Option<u64>,
    pub promotion_score: Option<f64>,
    pub promotion_conclusion: Option<PromotionCampaignConclusion>,
    pub evaluation: Option<DashboardEloObservationV1>,
    pub outcome: Option<GenerationOutcomeV1>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DashboardSnapshotV1 {
    pub generated_unix_seconds: u64,
    pub verification: DashboardVerificationV1,
    pub campaign_path: PathBuf,
    pub curriculum_path: PathBuf,
    pub active_generation: u64,
    pub completed_generations: u64,
    pub next_generation: u64,
    pub current_stage: String,
    pub current_tier: CurriculumTierV1,
    pub training_examples: u64,
    pub games: DashboardGameCountsV1,
    pub disk_bytes: u64,
    pub training_checkpoint: DashboardCheckpointV1,
    pub champion_checkpoint: DashboardCheckpointV1,
    pub latest_checkpoint: Option<DashboardCheckpointV1>,
    pub latest_evaluation: Option<DashboardEloObservationV1>,
    pub best_comparable_evaluation: Option<DashboardEloObservationV1>,
    pub elo_series: Vec<DashboardEloSeriesV1>,
    pub generations: Vec<DashboardGenerationV1>,
}

pub fn build_dashboard_snapshot(
    curriculum_path: impl AsRef<Path>,
    verification: DashboardVerificationV1,
) -> Result<DashboardSnapshotV1, DashboardError> {
    let curriculum = CurriculumCampaignArchive::open_existing(curriculum_path.as_ref())?;
    if verification == DashboardVerificationV1::Semantic {
        curriculum.verify_all_evidence()?;
    }
    let state = curriculum.load_state()?;
    let decisions = curriculum.load_decisions()?;
    let campaign_path = PathBuf::from(&curriculum.identity().generation_campaign_path);
    let campaign = GenerationCampaignArchive::open_existing(campaign_path)?;
    if verification == DashboardVerificationV1::Semantic {
        campaign.verify_all_artifacts()?;
    }
    let chain = campaign.load_chain()?;
    let decision_by_generation = decisions
        .iter()
        .map(|decision| (decision.generation, decision))
        .collect::<BTreeMap<_, _>>();
    let observations = load_elo_observations(&curriculum, &decisions)?;
    let observation_by_generation = observations
        .iter()
        .map(|observation| (observation.generation, observation.clone()))
        .collect::<BTreeMap<_, _>>();

    let first_generation = campaign
        .identity()
        .genesis_checkpoint
        .generation
        .checked_add(1)
        .ok_or_else(|| invalid("le numéro de génération initial déborde"))?;
    let last_completed = chain.next_generation.saturating_sub(1);
    let last_visible = chain.in_progress_generation.unwrap_or(last_completed);
    let mut games = DashboardGameCountsV1::default();
    let mut training_examples = 0_u64;
    let mut generations = Vec::new();
    if first_generation <= last_visible {
        for generation in first_generation..=last_visible {
            let plan = campaign.read_plan(generation)?;
            let actor = campaign.read_actor_stage(generation)?;
            let learner = campaign.read_learner_stage(generation)?;
            let promotion_stage = campaign.read_promotion_stage(generation)?;
            let outcome = campaign.read_outcome(generation)?;
            let decision = decision_by_generation.get(&generation).copied();
            let actor_counts = actor
                .as_ref()
                .map(|stage| read_actor_counts(&campaign, &stage.run_directory))
                .transpose()?;
            let promotion = read_promotion_progress(
                &campaign,
                generation,
                promotion_stage
                    .as_ref()
                    .map(|stage| stage.archive_directory.as_str()),
            )?;
            let evaluation = observation_by_generation.get(&generation).cloned();

            if let Some(counts) = actor_counts {
                games.actor_attempted = checked_add(games.actor_attempted, counts.attempts)?;
                games.actor_retained = checked_add(games.actor_retained, counts.retained)?;
            }
            if let Some(stage) = &actor {
                training_examples = checked_add(training_examples, stage.training_examples)?;
            }
            if let Some(progress) = &promotion {
                games.promotion_attempted = checked_add(
                    games.promotion_attempted,
                    checked_double(progress.attempted_pairs)?,
                )?;
                games.promotion_eligible = checked_add(
                    games.promotion_eligible,
                    checked_double(progress.eligible_pairs)?,
                )?;
            }
            if let Some(evaluation) = &evaluation {
                games.evaluation_attempted = checked_add(
                    games.evaluation_attempted,
                    checked_double(evaluation.attempted_pairs)?,
                )?;
                games.evaluation_eligible = checked_add(
                    games.evaluation_eligible,
                    checked_double(evaluation.eligible_pairs)?,
                )?;
            }

            generations.push(DashboardGenerationV1 {
                generation,
                opponent: plan.actor.opponent,
                stage: generation_stage(
                    actor.is_some(),
                    learner.is_some(),
                    promotion_stage.is_some(),
                    outcome.is_some(),
                    decision.is_some(),
                )
                .to_owned(),
                actor_attempts: actor_counts.map(|counts| counts.attempts),
                actor_retained: actor_counts.map(|counts| counts.retained),
                actor_interrupted: actor_counts.map(|counts| counts.interrupted),
                training_examples: actor.map(|stage| stage.training_examples),
                training_step: learner.map(|stage| stage.checkpoint.training_step),
                promotion_attempted_pairs: promotion
                    .as_ref()
                    .map(|progress| progress.attempted_pairs),
                promotion_eligible_pairs: promotion
                    .as_ref()
                    .map(|progress| progress.eligible_pairs),
                promotion_score: promotion.as_ref().and_then(promotion_score),
                promotion_conclusion: promotion_stage.map(|stage| stage.conclusion),
                evaluation,
                outcome,
            });
        }
    }

    let elo_series = group_elo_series(observations);
    let latest_evaluation = elo_series
        .iter()
        .flat_map(|series| series.observations.iter())
        .max_by_key(|observation| observation.generation)
        .cloned();
    let best_comparable_evaluation = latest_evaluation.as_ref().and_then(|latest| {
        elo_series
            .iter()
            .find(|series| series.protocol_key == latest.protocol_key)
            .and_then(|series| series.best.clone())
    });
    let active_generation = active_generation(&chain, state.last_decided_generation);
    let current_stage = campaign_stage(&chain, state.last_decided_generation, &generations);
    let disk_bytes = checked_add(
        directory_size(campaign.root())?,
        directory_size(curriculum.root())?,
    )?;

    Ok(DashboardSnapshotV1 {
        generated_unix_seconds: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        verification,
        campaign_path: campaign.root().to_owned(),
        curriculum_path: curriculum.root().to_owned(),
        active_generation,
        completed_generations: chain.completed_generations,
        next_generation: chain.next_generation,
        current_stage,
        current_tier: state.current_tier,
        training_examples,
        games,
        disk_bytes,
        training_checkpoint: checkpoint(&campaign, &chain.roles.training)?,
        champion_checkpoint: checkpoint(&campaign, &chain.roles.champion)?,
        latest_checkpoint: chain
            .roles
            .latest
            .as_ref()
            .map(|reference| checkpoint(&campaign, reference))
            .transpose()?,
        latest_evaluation,
        best_comparable_evaluation,
        elo_series,
        generations,
    })
}

pub fn render_dashboard_html(
    snapshot: &DashboardSnapshotV1,
    refresh_seconds: Option<u64>,
) -> String {
    let mut html = String::with_capacity(32_768);
    write!(
        html,
        "<!doctype html><html lang=\"fr\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    )
    .unwrap();
    if let Some(seconds) = refresh_seconds {
        write!(html, "<meta http-equiv=\"refresh\" content=\"{seconds}\">").unwrap();
    }
    html.push_str(STYLE_AND_TITLE);
    write!(
        html,
        "</head><body><main><header><div><p class=\"eyebrow\">PAI SHO · RÉSEAU PUR</p><h1>Tableau de bord d’entraînement</h1><p class=\"subtitle\">{} · palier <strong>{}</strong></p></div><div class=\"status\"><span></span>{}</div></header>",
        escape_html(&snapshot.current_stage),
        tier_label(snapshot.current_tier),
        snapshot.verification.label(),
    )
    .unwrap();

    html.push_str("<section class=\"cards\">");
    card(
        &mut html,
        "Génération active",
        &format!("G{}", snapshot.active_generation),
        &snapshot.current_stage,
    );
    card(
        &mut html,
        "Générations terminées",
        &french_integer(snapshot.completed_generations),
        &format!("prochaine : {}", snapshot.next_generation),
    );
    card(
        &mut html,
        "Combats lancés",
        &french_integer(snapshot.games.total_attempted()),
        &format!(
            "{} conservés ou classables",
            french_integer(snapshot.games.total_retained_or_eligible())
        ),
    );
    card(
        &mut html,
        "Dernier écart Elo mesuré",
        &snapshot
            .latest_evaluation
            .as_ref()
            .map_or_else(|| "—".to_owned(), elo_gap_card_value),
        &snapshot.latest_evaluation.as_ref().map_or_else(
            || "aucune évaluation".to_owned(),
            |value| {
                format!(
                    "G{} contre {} · ce n’est pas une cote absolue",
                    value.generation, value.opponent
                )
            },
        ),
    );
    card(
        &mut html,
        "Meilleur agent comparable",
        &snapshot
            .best_comparable_evaluation
            .as_ref()
            .map_or_else(|| "—".to_owned(), |value| format!("G{}", value.generation)),
        &snapshot.best_comparable_evaluation.as_ref().map_or_else(
            || "aucune série".to_owned(),
            |value| {
                format!(
                    "{} contre {} · même protocole",
                    elo_gap_card_value(value),
                    value.opponent
                )
            },
        ),
    );
    card(
        &mut html,
        "Cote Elo interne canonique",
        "Non raccordée",
        "les checkpoints doivent encore jouer des matchs classés sans exploration dans la ligue commune",
    );
    card(
        &mut html,
        "Exemples PPO",
        &french_integer(snapshot.training_examples),
        &format!(
            "pas global {}",
            french_integer(snapshot.training_checkpoint.training_step)
        ),
    );
    card(
        &mut html,
        "Elo The Garden Gate",
        "Non calibré",
        "les matchs-ponts humains restent à faire",
    );
    html.push_str("</section>");

    render_ladder(&mut html, snapshot.current_tier);
    render_current_series(&mut html, snapshot);
    render_game_counts(&mut html, snapshot.games);
    render_generation_table(&mut html, &snapshot.generations);
    render_checkpoints(&mut html, snapshot);
    render_series_table(&mut html, &snapshot.elo_series);

    write!(
        html,
        "<footer><p>Vue dérivée en lecture seule des archives scellées. « Combats lancés » compte les tentatives acteurs et les deux manches de chaque paire de promotion ou d’évaluation. Les écarts Elo restent relatifs à l’adversaire et au protocole affichés. La cote Elo interne canonique exige des matchs classés sans exploration dans la ligue commune ; elle n’est pas encore calculée pour ces checkpoints. Aucun de ces nombres n’est un Elo The Garden Gate.</p><p>Volume des archives : {} · actualisé <time data-unix=\"{}\">{}</time></p></footer></main><script>for(const e of document.querySelectorAll('time[data-unix]')){{e.textContent=new Date(Number(e.dataset.unix)*1000).toLocaleString('fr-FR')}};</script></body></html>",
        human_bytes(snapshot.disk_bytes),
        snapshot.generated_unix_seconds,
        snapshot.generated_unix_seconds,
    )
    .unwrap();
    html
}

pub fn write_dashboard_html(path: impl AsRef<Path>, html: &str) -> Result<(), DashboardError> {
    let path = path.as_ref();
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .ok_or_else(|| invalid("le chemin du tableau de bord n’a pas de nom de fichier"))?;
    let temporary = path.with_file_name(format!(
        ".{}.partial-{}",
        name.to_string_lossy(),
        std::process::id()
    ));
    let result = (|| -> io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(html.as_bytes())?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        File::open(parent)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(Into::into)
}

#[derive(Clone, Copy, Debug)]
struct ActorCounts {
    attempts: u64,
    retained: u64,
    interrupted: u64,
}

fn read_actor_counts(
    campaign: &GenerationCampaignArchive,
    relative_run_directory: &str,
) -> Result<ActorCounts, DashboardError> {
    let directory = campaign.root().join(relative_run_directory);
    let metadata_path = directory.join("actors.txt");
    let bytes = fs::read(metadata_path)?;
    verify_manifest_entry(&directory.join("MANIFEST.sha256"), "actors.txt", &bytes)?;
    parse_actor_counts(&bytes)
}

fn parse_actor_counts(bytes: &[u8]) -> Result<ActorCounts, DashboardError> {
    let text = core::str::from_utf8(bytes)
        .map_err(|_| invalid("le rapport acteur n’est pas un texte UTF-8"))?;
    let without_final_newline = text
        .strip_suffix('\n')
        .ok_or_else(|| invalid("le rapport acteur n’a pas de fin de ligne canonique"))?;
    let (body, checksum_line) = without_final_newline
        .rsplit_once('\n')
        .ok_or_else(|| invalid("le rapport acteur est tronqué"))?;
    let checksum = checksum_line
        .strip_prefix("sha256\t")
        .ok_or_else(|| invalid("le rapport acteur n’a pas de checksum final"))?;
    let mut checked_body = body.as_bytes().to_vec();
    checked_body.push(b'\n');
    if checksum != hex_digest(Sha256::digest(&checked_body).into()) {
        return Err(invalid(
            "le checksum interne du rapport acteur est invalide",
        ));
    }
    if !body.starts_with("PAISHO-ACTOR-RUN\t") {
        return Err(invalid("le rapport acteur a un en-tête inconnu"));
    }
    Ok(ActorCounts {
        attempts: report_u64(body, "attempts")?,
        retained: report_u64(body, "retained-games")?,
        interrupted: report_u64(body, "interrupted")?,
    })
}

fn report_u64(text: &str, field: &str) -> Result<u64, DashboardError> {
    let prefix = format!("{field}\t");
    let mut values = text.lines().filter_map(|line| line.strip_prefix(&prefix));
    let value = values
        .next()
        .ok_or_else(|| invalid(format!("le rapport acteur ne contient pas {field}")))?;
    if values.next().is_some() {
        return Err(invalid(format!(
            "le rapport acteur contient plusieurs champs {field}"
        )));
    }
    value
        .parse()
        .map_err(|_| invalid(format!("le champ acteur {field} n’est pas un entier")))
}

fn verify_manifest_entry(
    manifest_path: &Path,
    file_name: &str,
    bytes: &[u8],
) -> Result<(), DashboardError> {
    let manifest = fs::read_to_string(manifest_path)?;
    let expected_digest = hex_digest(Sha256::digest(bytes).into());
    let expected = format!("{expected_digest}  {file_name}");
    let matches = manifest.lines().filter(|line| *line == expected).count();
    if matches != 1 {
        return Err(invalid(format!(
            "le manifeste ne scelle pas exactement une fois {file_name}"
        )));
    }
    Ok(())
}

fn read_promotion_progress(
    campaign: &GenerationCampaignArchive,
    generation: u64,
    archived_directory: Option<&str>,
) -> Result<Option<crate::PromotionCampaignProgress>, DashboardError> {
    let path = archived_directory.map_or_else(
        || campaign.generation_directory(generation).join("promotion"),
        |relative| campaign.root().join(relative),
    );
    if !path.join("run.json").is_file() {
        return Ok(None);
    }
    let (_, progress) = PromotionCampaignArchive::open_existing_integrity(path)?;
    Ok(Some(progress))
}

fn promotion_score(progress: &crate::PromotionCampaignProgress) -> Option<f64> {
    if progress.eligible_pairs == 0 {
        return None;
    }
    let weights = [0.0, 0.25, 0.5, 0.75, 1.0];
    let score = progress
        .pentanomial
        .bins()
        .into_iter()
        .zip(weights)
        .map(|(count, weight)| count as f64 * weight)
        .sum::<f64>();
    Some(score / progress.eligible_pairs as f64)
}

fn load_elo_observations(
    curriculum: &CurriculumCampaignArchive,
    decisions: &[CurriculumDecisionV1],
) -> Result<Vec<DashboardEloObservationV1>, DashboardError> {
    let mut observations = Vec::new();
    for decision in decisions {
        let CurriculumEvidenceV1::FixedOpponent(evidence) = &decision.evidence else {
            continue;
        };
        let evaluation = EvaluationCampaignArchive::open_existing_integrity(
            curriculum.root().join(&evidence.relative_directory),
        )?;
        let analysis = evaluation.published_analysis_integrity()?;
        let (protocol_key, protocol_label) = evaluation_protocol(evaluation.identity());
        let opponent = opponent_label(&evaluation.identity().opponent.display_label());
        observations.push(DashboardEloObservationV1 {
            generation: decision.generation,
            opponent,
            protocol_key,
            protocol_label,
            contextual_elo: analysis.contextual_elo_point,
            davidson: analysis
                .mle
                .as_ref()
                .map(|mle| mle.candidate_minus_opponent),
            attempted_pairs: analysis.attempted_pairs,
            eligible_pairs: analysis.eligible_pairs,
            excluded_pairs: analysis.excluded_pairs,
            wins: analysis.candidate_wins,
            draws: analysis.draws,
            losses: analysis.candidate_losses,
        });
    }
    observations.sort_by_key(|observation| observation.generation);
    Ok(observations)
}

fn evaluation_protocol(identity: &EvaluationRunIdentityV1) -> (String, String) {
    let opponent = identity.opponent.display_label();
    let start_key = identity.neutral_start.map_or_else(
        || "standard".to_owned(),
        |start| {
            format!(
                "horizon:{}:source-limit:{}:source-attempts:{}",
                start.target_remaining_decisions,
                start.source_decision_limit,
                start.maximum_source_attempts
            )
        },
    );
    let start_label = identity.neutral_start.map_or_else(
        || "partie standard".to_owned(),
        |start| format!("horizon {}", start.target_remaining_decisions),
    );
    let policy_key = identity.sampling_policy.map_or_else(
        || "argmax".to_owned(),
        |policy| {
            format!(
                "sample:{:08x}:{:08x}",
                policy.temperature_bits, policy.uniform_mix_bits
            )
        },
    );
    let policy_label = identity.sampling_policy.map_or_else(
        || "argmax".to_owned(),
        |policy| {
            format!(
                "T={} + {} % uniforme",
                compact_float(f32::from_bits(policy.temperature_bits) as f64),
                compact_float(100.0 * f32::from_bits(policy.uniform_mix_bits) as f64)
            )
        },
    );
    let classes = identity
        .inference_classes
        .iter()
        .map(|class| format!("{}:{}", class.legal_action_capacity, class.batch_size))
        .collect::<Vec<_>>()
        .join(",");
    let key = format!(
        "opponent={};opponent-sha={};source={};service={};preset={};level={};classes={};limit={};start={};policy={};max-attempted={};max-eligible={};elo={:016x}:{:016x}:{:?};errors={:016x}:{:016x}",
        identity.opponent.display_label(),
        identity.opponent_sha256,
        identity.source_sha256,
        identity.service_sha256,
        identity.preset,
        identity.optimization_level,
        classes,
        identity.decision_soft_limit,
        start_key,
        policy_key,
        identity.maximum_attempted_pairs,
        identity.maximum_eligible_pairs,
        identity.elo0.to_bits(),
        identity.elo1.to_bits(),
        identity.lower_elo.map(f64::to_bits),
        identity.alpha.to_bits(),
        identity.beta.to_bits(),
    );
    let label = format!(
        "{} · {} · limite {} décisions · {} · jusqu’à {} paires admissibles",
        opponent_label(&opponent),
        start_label,
        identity.decision_soft_limit,
        policy_label,
        identity.maximum_eligible_pairs,
    );
    (key, label)
}

fn group_elo_series(observations: Vec<DashboardEloObservationV1>) -> Vec<DashboardEloSeriesV1> {
    let mut grouped = BTreeMap::<String, Vec<DashboardEloObservationV1>>::new();
    for observation in observations {
        grouped
            .entry(observation.protocol_key.clone())
            .or_default()
            .push(observation);
    }
    let mut series = grouped
        .into_iter()
        .map(|(protocol_key, mut observations)| {
            observations.sort_by_key(|observation| observation.generation);
            let protocol_label = observations
                .first()
                .map(|observation| observation.protocol_label.clone())
                .unwrap_or_default();
            let best = observations
                .iter()
                .filter_map(|observation| {
                    observation
                        .finite_elo()
                        .map(|elo| (elo, observation.clone()))
                })
                .max_by(|left, right| left.0.total_cmp(&right.0))
                .map(|(_, observation)| observation);
            DashboardEloSeriesV1 {
                protocol_key,
                protocol_label,
                observations,
                best,
            }
        })
        .collect::<Vec<_>>();
    series.sort_by_key(|item| {
        item.observations
            .last()
            .map_or(0, |observation| observation.generation)
    });
    series
}

fn generation_stage(
    actor: bool,
    learner: bool,
    promotion: bool,
    outcome: bool,
    curriculum_decision: bool,
) -> &'static str {
    if curriculum_decision {
        "Terminée"
    } else if outcome {
        "Évaluation Elo"
    } else if promotion {
        "Publication"
    } else if learner {
        "Promotion"
    } else if actor {
        "Apprentissage"
    } else {
        "Production des parties"
    }
}

fn campaign_stage(
    chain: &crate::GenerationCampaignChainV1,
    last_decided_generation: u64,
    generations: &[DashboardGenerationV1],
) -> String {
    if last_decided_generation < chain.next_generation.saturating_sub(1) {
        return format!(
            "Évaluation Elo de la génération {}",
            last_decided_generation.saturating_add(1)
        );
    }
    if let Some(generation) = chain.in_progress_generation {
        let stage = generations
            .iter()
            .find(|row| row.generation == generation)
            .map_or("En cours", |row| row.stage.as_str());
        return format!("Génération {generation} · {stage}");
    }
    format!(
        "{} générations terminées · prochaine éventuelle : {}",
        chain.next_generation.saturating_sub(1),
        chain.next_generation
    )
}

fn active_generation(
    chain: &crate::GenerationCampaignChainV1,
    last_decided_generation: u64,
) -> u64 {
    if last_decided_generation < chain.next_generation.saturating_sub(1) {
        last_decided_generation.saturating_add(1)
    } else {
        chain
            .in_progress_generation
            .unwrap_or(chain.next_generation)
    }
}

fn checkpoint(
    campaign: &GenerationCampaignArchive,
    reference: &crate::CheckpointReferenceV1,
) -> Result<DashboardCheckpointV1, DashboardError> {
    Ok(DashboardCheckpointV1 {
        generation: reference.generation,
        training_step: reference.training_step,
        sha256: reference.sha256.clone(),
        path: reference.path(campaign.root())?,
    })
}

fn directory_size(path: &Path) -> Result<u64, DashboardError> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Ok(0);
    }
    if metadata.is_file() {
        return Ok(metadata.len());
    }
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        total = checked_add(total, directory_size(&entry?.path())?)?;
    }
    Ok(total)
}

fn checked_add(left: u64, right: u64) -> Result<u64, DashboardError> {
    left.checked_add(right)
        .ok_or_else(|| invalid("un compteur du tableau de bord déborde"))
}

fn checked_double(value: u64) -> Result<u64, DashboardError> {
    value
        .checked_mul(2)
        .ok_or_else(|| invalid("un compteur de parties appariées déborde"))
}

fn card(html: &mut String, label: &str, value: &str, detail: &str) {
    write!(
        html,
        "<article class=\"card\"><p>{}</p><strong>{}</strong><small>{}</small></article>",
        escape_html(label),
        escape_html(value),
        escape_html(detail),
    )
    .unwrap();
}

fn render_ladder(html: &mut String, current: CurriculumTierV1) {
    html.push_str("<section class=\"panel\"><div class=\"panel-head\"><div><p class=\"eyebrow\">CURRICULUM</p><h2>Échelle des adversaires</h2></div></div><div class=\"ladder\">");
    for tier in [
        CurriculumTierV1::Random,
        CurriculumTierV1::SiteBotV1,
        CurriculumTierV1::Mcts8,
        CurriculumTierV1::Mcts32,
        CurriculumTierV1::Mcts128,
        CurriculumTierV1::Mcts512,
        CurriculumTierV1::SelfPlay,
    ] {
        let class = match tier.cmp(&current) {
            std::cmp::Ordering::Equal => "active",
            std::cmp::Ordering::Less => "passed",
            std::cmp::Ordering::Greater => "future",
        };
        write!(
            html,
            "<div class=\"rung {class}\"><span></span><b>{}</b></div>",
            tier_label(tier)
        )
        .unwrap();
    }
    html.push_str("</div></section>");
}

fn render_current_series(html: &mut String, snapshot: &DashboardSnapshotV1) {
    let Some(latest) = &snapshot.latest_evaluation else {
        return;
    };
    let Some(series) = snapshot
        .elo_series
        .iter()
        .find(|series| series.protocol_key == latest.protocol_key)
    else {
        return;
    };
    html.push_str("<section class=\"panel\"><div class=\"panel-head\"><div><p class=\"eyebrow\">ELO INTERNE CONTEXTUEL</p><h2>Progression comparable</h2></div>");
    write!(
        html,
        "<p class=\"protocol\">{}</p></div>",
        escape_html(&series.protocol_label)
    )
    .unwrap();
    html.push_str(&elo_chart(&series.observations));
    html.push_str("</section>");
}

fn elo_chart(observations: &[DashboardEloObservationV1]) -> String {
    let points = observations
        .iter()
        .filter_map(|observation| {
            observation
                .finite_elo()
                .map(|elo| (observation.generation, elo))
        })
        .collect::<Vec<_>>();
    if points.is_empty() {
        return "<p class=\"empty\">Aucune valeur Elo finie dans cette série.</p>".to_owned();
    }
    let width = 760.0_f64;
    let height = 230.0_f64;
    let left = 48.0;
    let right = 22.0;
    let top = 28.0;
    let bottom = 42.0;
    let minimum = points
        .iter()
        .map(|(_, elo)| *elo)
        .fold(0.0_f64, f64::min)
        .min(-20.0);
    let maximum = points
        .iter()
        .map(|(_, elo)| *elo)
        .fold(0.0_f64, f64::max)
        .max(20.0);
    let padding = ((maximum - minimum) * 0.18).max(8.0);
    let low = minimum - padding;
    let high = maximum + padding;
    let usable_width = width - left - right;
    let usable_height = height - top - bottom;
    let x = |index: usize| {
        if points.len() == 1 {
            left + usable_width / 2.0
        } else {
            left + usable_width * index as f64 / (points.len() - 1) as f64
        }
    };
    let y = |elo: f64| top + (high - elo) * usable_height / (high - low);
    let coordinates = points
        .iter()
        .enumerate()
        .map(|(index, (_, elo))| format!("{:.1},{:.1}", x(index), y(*elo)))
        .collect::<Vec<_>>()
        .join(" ");
    let mut svg = format!(
        "<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Évolution de l’Elo interne\"><line class=\"zero\" x1=\"{left}\" y1=\"{:.1}\" x2=\"{}\" y2=\"{:.1}\"/><text class=\"axis-label\" x=\"8\" y=\"{:.1}\">0</text><polyline points=\"{coordinates}\"/>",
        y(0.0),
        width - right,
        y(0.0),
        y(0.0) + 4.0,
    );
    for (index, (generation, elo)) in points.iter().enumerate() {
        write!(
            svg,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"5\"/><text class=\"value-label\" x=\"{:.1}\" y=\"{:.1}\">{}</text><text class=\"generation-label\" x=\"{:.1}\" y=\"{}\">G{}</text>",
            x(index),
            y(*elo),
            x(index),
            y(*elo) - 12.0,
            signed_elo(*elo),
            x(index),
            height - 12.0,
            generation,
        )
        .unwrap();
    }
    svg.push_str("</svg>");
    svg
}

fn render_game_counts(html: &mut String, games: DashboardGameCountsV1) {
    html.push_str("<section class=\"panel\"><div class=\"panel-head\"><div><p class=\"eyebrow\">VOLUME</p><h2>Parties par fonction</h2></div></div><div class=\"counts\">");
    count_block(
        html,
        "Apprentissage",
        games.actor_attempted,
        games.actor_retained,
        "tentées",
        "conservées",
    );
    count_block(
        html,
        "Promotion",
        games.promotion_attempted,
        games.promotion_eligible,
        "jouées",
        "classables",
    );
    count_block(
        html,
        "Évaluation Elo",
        games.evaluation_attempted,
        games.evaluation_eligible,
        "jouées",
        "classables",
    );
    html.push_str("</div></section>");
}

fn count_block(
    html: &mut String,
    title: &str,
    primary: u64,
    secondary: u64,
    primary_label: &str,
    secondary_label: &str,
) {
    write!(
        html,
        "<article class=\"count\"><h3>{}</h3><div><strong>{}</strong><span>{}</span></div><div><strong>{}</strong><span>{}</span></div></article>",
        escape_html(title),
        french_integer(primary),
        escape_html(primary_label),
        french_integer(secondary),
        escape_html(secondary_label),
    )
    .unwrap();
}

fn render_generation_table(html: &mut String, generations: &[DashboardGenerationV1]) {
    html.push_str("<section class=\"panel table-panel\"><div class=\"panel-head\"><div><p class=\"eyebrow\">HISTORIQUE</p><h2>Générations</h2></div></div><div class=\"table-wrap\"><table><thead><tr><th>Gén.</th><th>Adversaire</th><th>Apprentissage</th><th>Promotion</th><th>Évaluation</th><th>État</th></tr></thead><tbody>");
    for row in generations.iter().rev() {
        let actor = match (
            row.actor_attempts,
            row.actor_retained,
            row.training_examples,
        ) {
            (Some(attempts), Some(retained), Some(examples)) => format!(
                "{} / {} parties<small>{} exemples · {} interrompues</small>",
                french_integer(retained),
                french_integer(attempts),
                french_integer(examples),
                french_integer(row.actor_interrupted.unwrap_or(0)),
            ),
            _ => "—".to_owned(),
        };
        let promotion = match (
            row.promotion_score,
            row.promotion_eligible_pairs,
            row.promotion_conclusion,
        ) {
            (Some(score), Some(pairs), conclusion) => format!(
                "{:.1} %<small>{} paires · {}</small>",
                score * 100.0,
                french_integer(pairs),
                conclusion.map_or("en cours", promotion_label),
            ),
            _ => "—".to_owned(),
        };
        let evaluation = row.evaluation.as_ref().map_or_else(
            || "—".to_owned(),
            |evaluation| {
                let davidson = evaluation.davidson.map_or_else(String::new, |estimate| {
                    let uncertainty = estimate.paired_cluster.unwrap_or(estimate.model);
                    format!(
                        "<small>Davidson {} · IC 95 % [{}, {}]</small>",
                        signed_elo(estimate.estimate),
                        signed_elo(uncertainty.interval_95_lower),
                        signed_elo(uncertainty.interval_95_upper),
                    )
                });
                format!(
                    "{}<small>{}/{}/{} · {} paires</small>{}",
                    elo_gap_card_value(evaluation),
                    evaluation.wins,
                    evaluation.draws,
                    evaluation.losses,
                    french_integer(evaluation.eligible_pairs),
                    davidson,
                )
            },
        );
        write!(
            html,
            "<tr><td><b>G{}</b><small>pas {}</small></td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><span class=\"pill\">{}</span></td></tr>",
            row.generation,
            row.training_step.map_or_else(|| "—".to_owned(), french_integer),
            escape_html(&opponent_label(&row.opponent)),
            actor,
            promotion,
            evaluation,
            escape_html(&row.stage),
        )
        .unwrap();
    }
    html.push_str("</tbody></table></div></section>");
}

fn render_checkpoints(html: &mut String, snapshot: &DashboardSnapshotV1) {
    html.push_str("<section class=\"panel\"><div class=\"panel-head\"><div><p class=\"eyebrow\">REPRISE</p><h2>Checkpoints actifs</h2></div></div><div class=\"checkpoints\">");
    checkpoint_block(html, "Parent d’entraînement", &snapshot.training_checkpoint);
    checkpoint_block(html, "Champion officiel", &snapshot.champion_checkpoint);
    if let Some(latest) = &snapshot.latest_checkpoint {
        checkpoint_block(html, "Dernier produit", latest);
    }
    html.push_str("</div></section>");
}

fn checkpoint_block(html: &mut String, title: &str, checkpoint: &DashboardCheckpointV1) {
    write!(
        html,
        "<article class=\"checkpoint\"><p>{}</p><strong>G{} · pas {}</strong><code>{}</code><small>{}</small></article>",
        escape_html(title),
        checkpoint.generation,
        french_integer(checkpoint.training_step),
        &checkpoint.sha256[..12.min(checkpoint.sha256.len())],
        escape_html(&checkpoint.path.display().to_string()),
    )
    .unwrap();
}

fn render_series_table(html: &mut String, series: &[DashboardEloSeriesV1]) {
    html.push_str("<section class=\"panel\"><div class=\"panel-head\"><div><p class=\"eyebrow\">COMPARABILITÉ</p><h2>Records par protocole</h2></div></div><div class=\"series-list\">");
    for item in series.iter().rev() {
        let latest = item.observations.last();
        write!(
            html,
            "<article><div><strong>{}</strong><small>{} mesure(s)</small></div><div><span>Dernier {}</span><span>Record {}</span></div></article>",
            escape_html(&item.protocol_label),
            item.observations.len(),
            latest.map_or_else(|| "—".to_owned(), elo_gap_card_value),
            item.best.as_ref().map_or_else(|| "—".to_owned(), elo_gap_card_value),
        )
        .unwrap();
    }
    if series.is_empty() {
        html.push_str("<p class=\"empty\">Aucune série Elo publiée.</p>");
    }
    html.push_str("</div></section>");
}

fn elo_gap_card_value(observation: &DashboardEloObservationV1) -> String {
    match observation.contextual_elo {
        ContextualEloPointV1::NoRatedGames => "indisponible".to_owned(),
        ContextualEloPointV1::NegativeInfinity => "−∞ points".to_owned(),
        ContextualEloPointV1::Finite { elo } => format!("{} points", signed_elo(elo)),
        ContextualEloPointV1::PositiveInfinity => "+∞ points".to_owned(),
    }
}

fn signed_elo(value: f64) -> String {
    if value >= 0.0 {
        format!("+{value:.1}")
    } else {
        format!("{value:.1}")
    }
}

fn french_integer(value: u64) -> String {
    let digits = value.to_string();
    let mut formatted = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            formatted.push('\u{202f}');
        }
        formatted.push(character);
    }
    formatted
}

fn compact_float(value: f64) -> String {
    if value.fract().abs() < 1.0e-6 {
        format!("{value:.0}")
    } else {
        format!("{value:.2}")
    }
}

fn human_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} Gio", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} Mio", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.1} Kio", bytes / KIB)
    } else {
        format!("{bytes:.0} octets")
    }
}

fn tier_label(tier: CurriculumTierV1) -> &'static str {
    match tier {
        CurriculumTierV1::Random => "Aléatoire",
        CurriculumTierV1::SiteBotV1 => "Bot du site",
        CurriculumTierV1::Mcts8 => "MCTS 8",
        CurriculumTierV1::Mcts32 => "MCTS 32",
        CurriculumTierV1::Mcts128 => "MCTS 128",
        CurriculumTierV1::Mcts512 => "MCTS 512",
        CurriculumTierV1::SelfPlay => "Auto-jeu pur",
    }
}

fn opponent_label(opponent: &str) -> String {
    CurriculumTierV1::parse(opponent)
        .map(tier_label)
        .map(str::to_owned)
        .unwrap_or_else(|_| opponent.to_owned())
}

fn promotion_label(conclusion: PromotionCampaignConclusion) -> &'static str {
    match conclusion {
        PromotionCampaignConclusion::PromoteCandidate => "promu",
        PromotionCampaignConclusion::RejectCandidate => "rejeté",
        PromotionCampaignConclusion::InconclusiveMaximumEligiblePairs
        | PromotionCampaignConclusion::InconclusiveMaximumAttemptedPairs => "indéterminé",
    }
}

fn escape_html(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        write!(text, "{byte:02x}").unwrap();
    }
    text
}

fn invalid(message: impl Into<String>) -> DashboardError {
    DashboardError::InvalidData(message.into())
}

#[derive(Debug)]
pub enum DashboardError {
    Io(io::Error),
    Curriculum(CurriculumArchiveError),
    Generation(GenerationArchiveError),
    Promotion(PromotionArchiveError),
    Evaluation(EvaluationArchiveError),
    InvalidData(String),
}

impl fmt::Display for DashboardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => source.fmt(formatter),
            Self::Curriculum(source) => source.fmt(formatter),
            Self::Generation(source) => source.fmt(formatter),
            Self::Promotion(source) => source.fmt(formatter),
            Self::Evaluation(source) => source.fmt(formatter),
            Self::InvalidData(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for DashboardError {}

impl From<io::Error> for DashboardError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<CurriculumArchiveError> for DashboardError {
    fn from(source: CurriculumArchiveError) -> Self {
        Self::Curriculum(source)
    }
}

impl From<GenerationArchiveError> for DashboardError {
    fn from(source: GenerationArchiveError) -> Self {
        Self::Generation(source)
    }
}

impl From<PromotionArchiveError> for DashboardError {
    fn from(source: PromotionArchiveError) -> Self {
        Self::Promotion(source)
    }
}

impl From<EvaluationArchiveError> for DashboardError {
    fn from(source: EvaluationArchiveError) -> Self {
        Self::Evaluation(source)
    }
}

const STYLE_AND_TITLE: &str = r#"<title>Pai Sho · Suivi d’entraînement</title><style>
:root{color-scheme:dark;--bg:#101812;--panel:#17231b;--panel2:#1e2e23;--ink:#f1eedf;--muted:#aebbac;--line:#34483a;--jade:#79c995;--gold:#e7bd67;--red:#ef8d75}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#243a2b 0,transparent 34rem),var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{width:min(1180px,calc(100% - 32px));margin:auto;padding:44px 0 64px}header{display:flex;align-items:flex-start;justify-content:space-between;gap:24px;margin-bottom:28px}h1{font-family:Georgia,serif;font-size:clamp(2.2rem,5vw,4.5rem);line-height:.95;margin:8px 0 14px;letter-spacing:-.04em}h2{font-family:Georgia,serif;font-size:1.65rem;margin:3px 0}.eyebrow{font-size:.72rem;letter-spacing:.18em;color:var(--gold);font-weight:800;margin:0}.subtitle,.protocol,footer{color:var(--muted)}.status{border:1px solid var(--line);background:#142018;border-radius:999px;padding:11px 16px;white-space:nowrap;font-size:.82rem}.status span{display:inline-block;width:8px;height:8px;background:var(--jade);border-radius:50%;margin-right:9px;box-shadow:0 0 14px var(--jade)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card,.panel{background:linear-gradient(145deg,rgba(31,48,37,.96),rgba(21,32,25,.98));border:1px solid var(--line);box-shadow:0 18px 40px rgba(0,0,0,.18)}.card{border-radius:18px;padding:20px;min-height:135px;display:flex;flex-direction:column}.card p,.checkpoint p{margin:0;color:var(--muted);font-size:.8rem}.card strong{font-family:Georgia,serif;font-size:2rem;margin:auto 0 7px}.card small,.checkpoint small,.series-list small,td small{display:block;color:var(--muted);font-size:.76rem}.panel{border-radius:22px;padding:24px;margin-top:14px}.panel-head{display:flex;justify-content:space-between;align-items:flex-end;gap:24px;margin-bottom:22px}.protocol{font-size:.78rem;max-width:58%;text-align:right;margin:0}.ladder{display:grid;grid-template-columns:repeat(7,1fr);gap:7px}.rung{position:relative;padding-top:24px;text-align:center;color:#77867a;font-size:.76rem}.rung:before{content:"";position:absolute;top:6px;left:-4px;right:-4px;height:2px;background:var(--line)}.rung:first-child:before{left:50%}.rung:last-child:before{right:50%}.rung span{position:absolute;top:0;left:calc(50% - 7px);width:14px;height:14px;border-radius:50%;background:#435247;border:3px solid var(--panel)}.rung.passed{color:var(--muted)}.rung.passed span{background:var(--jade)}.rung.active{color:var(--ink)}.rung.active span{background:var(--gold);box-shadow:0 0 18px rgba(231,189,103,.7)}.chart{display:block;width:100%;height:auto;max-height:310px;overflow:visible}.chart .zero{stroke:#56685a;stroke-dasharray:5 6}.chart polyline{fill:none;stroke:var(--jade);stroke-width:3;stroke-linejoin:round}.chart circle{fill:var(--gold);stroke:var(--panel);stroke-width:3}.chart text{fill:var(--muted);font-size:12px}.chart .value-label{fill:var(--ink);text-anchor:middle;font-weight:700}.chart .generation-label{text-anchor:middle}.counts{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.count{background:rgba(9,15,11,.24);border:1px solid var(--line);border-radius:15px;padding:17px}.count h3{margin:0 0 18px;font-size:.9rem}.count div{display:flex;align-items:baseline;justify-content:space-between;border-top:1px solid rgba(255,255,255,.06);padding-top:10px;margin-top:10px}.count strong{font-size:1.25rem}.count span{font-size:.75rem;color:var(--muted)}.table-panel{padding-left:0;padding-right:0}.table-panel .panel-head{padding:0 24px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:900px}th,td{text-align:left;padding:15px 16px;border-top:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.09em}td{font-size:.86rem}td small{margin-top:5px}.pill{display:inline-block;border:1px solid #47604e;color:var(--jade);border-radius:999px;padding:5px 9px;font-size:.7rem;white-space:nowrap}.checkpoints{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.checkpoint{min-width:0;background:rgba(9,15,11,.24);border:1px solid var(--line);border-radius:15px;padding:16px}.checkpoint strong,.checkpoint code{display:block;margin-top:10px}.checkpoint code{color:var(--gold)}.checkpoint small{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;margin-top:7px}.series-list article{display:grid;grid-template-columns:1fr auto;gap:20px;padding:14px 0;border-top:1px solid var(--line)}.series-list article>div:last-child{display:flex;gap:18px;color:var(--muted);font-size:.8rem}.empty{color:var(--muted)}footer{font-size:.77rem;line-height:1.55;padding:20px 6px 0}footer p{margin:8px 0}@media(max-width:850px){header{display:block}.status{display:inline-block;margin-top:12px}.cards{grid-template-columns:repeat(2,1fr)}.counts,.checkpoints{grid-template-columns:1fr}.panel-head{display:block}.protocol{max-width:none;text-align:left;margin-top:10px}.ladder{grid-template-columns:1fr}.rung{padding:5px 0 5px 24px;text-align:left}.rung:before{top:-7px;bottom:-7px;left:6px!important;right:auto!important;width:2px;height:auto}.rung span{left:0;top:7px}.series-list article{grid-template-columns:1fr}.series-list article>div:last-child{justify-content:space-between}}@media(max-width:520px){main{width:min(100% - 20px,1180px);padding-top:26px}.cards{grid-template-columns:1fr}.panel{padding:18px}.card{min-height:115px}}
</style>"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actor_report_counts_require_the_embedded_checksum() {
        let body = "PAISHO-ACTOR-RUN\t4\nattempts\t12\nretained-games\t10\ninterrupted\t2\n";
        let checksum = hex_digest(Sha256::digest(body.as_bytes()).into());
        let report = format!("{body}sha256\t{checksum}\n");
        let counts = parse_actor_counts(report.as_bytes()).unwrap();
        assert_eq!(counts.attempts, 12);
        assert_eq!(counts.retained, 10);
        assert_eq!(counts.interrupted, 2);

        let damaged = report.replace("attempts\t12", "attempts\t13");
        assert!(parse_actor_counts(damaged.as_bytes()).is_err());
    }

    #[test]
    fn elo_records_never_cross_protocol_boundaries() {
        let observation = |generation, key: &str, elo| DashboardEloObservationV1 {
            generation,
            opponent: "random".to_owned(),
            protocol_key: key.to_owned(),
            protocol_label: key.to_owned(),
            contextual_elo: ContextualEloPointV1::Finite { elo },
            davidson: None,
            attempted_pairs: 10,
            eligible_pairs: 10,
            excluded_pairs: 0,
            wins: 10,
            draws: 0,
            losses: 10,
        };
        let series = group_elo_series(vec![
            observation(1, "horizon-64", 20.0),
            observation(2, "horizon-64", 15.0),
            observation(3, "standard", 200.0),
        ]);
        assert_eq!(series.len(), 2);
        let contextual = series
            .iter()
            .find(|item| item.protocol_key == "horizon-64")
            .unwrap();
        assert_eq!(contextual.best.as_ref().unwrap().generation, 1);
    }

    #[test]
    fn html_escaping_and_french_grouping_are_stable() {
        assert_eq!(escape_html("<a&\"'>"), "&lt;a&amp;&quot;&#39;&gt;");
        assert_eq!(french_integer(2_822), "2\u{202f}822");
        assert_eq!(human_bytes(395 * 1024 * 1024), "395.0 Mio");
    }

    #[test]
    fn mutable_dashboard_is_replaced_without_leaving_a_partial_file() {
        let path = std::env::temp_dir().join(format!(
            "paisho-dashboard-write-test-{}.html",
            std::process::id()
        ));
        let partial = path.with_file_name(format!(
            ".{}.partial-{}",
            path.file_name().unwrap().to_string_lossy(),
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let _ = fs::remove_file(&partial);
        write_dashboard_html(&path, "premier").unwrap();
        write_dashboard_html(&path, "second").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "second");
        assert!(!partial.exists());
        fs::remove_file(path).unwrap();
    }
}
