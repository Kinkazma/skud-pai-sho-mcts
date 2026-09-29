use core::fmt;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    assess_curriculum_evaluation, atomic_file, deferred_evaluation_assessment,
    self_play_assessment, CheckpointReferenceV1, ContextualEloPointV1, CurriculumAssessmentV1,
    CurriculumDecisionError, CurriculumEloWindowV1, CurriculumTierV1, EvaluationArchiveError,
    EvaluationCampaignArchive, EvaluationConclusion, EvaluationEstimateV1, GenerationArchiveError,
    GenerationCampaignArchive, PromotionCampaignConclusion,
};

const IDENTITY_FORMAT: &str = "paisho-curriculum-campaign-v1";
const DECISION_FORMAT: &str = "paisho-curriculum-decision-v1";
const IDENTITY_FILE: &str = "curriculum.json";
const DECISIONS_DIRECTORY: &str = "decisions";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CurriculumCampaignIdentityV1 {
    pub generation_campaign_path: String,
    pub generation_campaign_sha256: String,
    pub baseline_generation: u64,
    pub baseline_checkpoint: CheckpointReferenceV1,
    pub baseline_champion: CheckpointReferenceV1,
    pub initial_tier: CurriculumTierV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CurriculumEvaluationEvidenceV1 {
    pub relative_directory: String,
    pub run_sha256: String,
    pub result_sha256: String,
    pub conclusion: EvaluationConclusion,
    pub elo_window: Option<CurriculumEloWindowV1>,
    pub contextual_elo_point: ContextualEloPointV1,
    pub davidson_candidate_minus_opponent: Option<EvaluationEstimateV1>,
    pub eligible_pairs: u64,
    pub excluded_pairs: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum CurriculumEvidenceV1 {
    FixedOpponent(Box<CurriculumEvaluationEvidenceV1>),
    DeferredEvaluation,
    SelfPlayPromotion {
        conclusion: Option<PromotionCampaignConclusion>,
    },
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct CurriculumDecisionV1 {
    pub sequence: u64,
    pub generation: u64,
    pub checkpoint: CheckpointReferenceV1,
    pub tier_before: CurriculumTierV1,
    pub evidence: CurriculumEvidenceV1,
    pub assessment: CurriculumAssessmentV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CurriculumStateV1 {
    pub decisions: u64,
    pub last_decided_generation: u64,
    pub current_checkpoint: CheckpointReferenceV1,
    pub current_champion: CheckpointReferenceV1,
    pub current_tier: CurriculumTierV1,
}

pub struct CurriculumCampaignArchive {
    root: PathBuf,
    identity: CurriculumCampaignIdentityV1,
    generation_campaign: GenerationCampaignArchive,
}

impl CurriculumCampaignArchive {
    pub fn open_or_create(
        root: impl AsRef<Path>,
        generation_campaign: &GenerationCampaignArchive,
        initial_tier: CurriculumTierV1,
    ) -> Result<Self, CurriculumArchiveError> {
        let root = root.as_ref().to_owned();
        if root.join(IDENTITY_FILE).is_file() {
            let archive = Self::open_existing(&root)?;
            if fs::canonicalize(generation_campaign.root())?
                != PathBuf::from(&archive.identity.generation_campaign_path)
            {
                return Err(invalid(
                    "existing curriculum belongs to another generation campaign",
                ));
            }
            return Ok(archive);
        }
        fs::create_dir_all(root.join(DECISIONS_DIRECTORY))?;
        let generation_root = fs::canonicalize(generation_campaign.root())?;
        let chain = generation_campaign.load_chain()?;
        let baseline_generation = chain
            .in_progress_generation
            .unwrap_or(chain.next_generation)
            .checked_sub(1)
            .ok_or_else(|| invalid("generation campaign has no baseline generation"))?;
        let identity = CurriculumCampaignIdentityV1 {
            generation_campaign_path: utf8_path(&generation_root)?,
            generation_campaign_sha256: sha256_file(&generation_root.join("campaign.json"))?,
            baseline_generation,
            baseline_checkpoint: chain.roles.training,
            baseline_champion: chain.roles.champion,
            initial_tier,
        };
        write_document(&root.join(IDENTITY_FILE), IDENTITY_FORMAT, &identity)?;
        let archive = Self {
            root,
            identity,
            generation_campaign: GenerationCampaignArchive::open_existing(generation_root)?,
        };
        archive.load_state_with_evidence(false)?;
        Ok(archive)
    }

    pub fn open_existing(root: impl AsRef<Path>) -> Result<Self, CurriculumArchiveError> {
        let root = root.as_ref().to_owned();
        let identity: CurriculumCampaignIdentityV1 =
            read_document(&root.join(IDENTITY_FILE), IDENTITY_FORMAT)?;
        validate_digest(
            &identity.generation_campaign_sha256,
            "generation campaign SHA-256",
        )?;
        let generation_root = PathBuf::from(&identity.generation_campaign_path);
        if !generation_root.is_absolute()
            || sha256_file(&generation_root.join("campaign.json"))?
                != identity.generation_campaign_sha256
        {
            return Err(invalid(
                "curriculum generation campaign is unavailable or changed",
            ));
        }
        let archive = Self {
            root,
            identity,
            generation_campaign: GenerationCampaignArchive::open_existing(generation_root)?,
        };
        archive.load_state_with_evidence(false)?;
        Ok(archive)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub const fn identity(&self) -> &CurriculumCampaignIdentityV1 {
        &self.identity
    }

    pub fn load_state(&self) -> Result<CurriculumStateV1, CurriculumArchiveError> {
        self.load_state_with_evidence(false)
    }

    pub fn load_decisions(&self) -> Result<Vec<CurriculumDecisionV1>, CurriculumArchiveError> {
        self.load_state_with_evidence(false)?;
        decision_paths(&self.root.join(DECISIONS_DIRECTORY))?
            .into_iter()
            .map(|path| read_document(&path, DECISION_FORMAT))
            .collect()
    }

    pub fn verify_all_evidence(&self) -> Result<CurriculumStateV1, CurriculumArchiveError> {
        self.load_state_with_evidence(true)
    }

    pub fn evaluation_directory(
        &self,
        generation: u64,
        tier: CurriculumTierV1,
        checkpoint_sha256: &str,
        protocol_sha256: &str,
    ) -> Result<PathBuf, CurriculumArchiveError> {
        validate_digest(checkpoint_sha256, "checkpoint SHA-256")?;
        validate_digest(protocol_sha256, "evaluation protocol SHA-256")?;
        Ok(self.root.join("evaluations").join(format!(
            "generation-{generation:020}-{}-{}-{}",
            tier.label(),
            &checkpoint_sha256[..12],
            &protocol_sha256[..12]
        )))
    }

    pub fn publish_evaluation_decision(
        &self,
        generation: u64,
        checkpoint: CheckpointReferenceV1,
        evaluation: &EvaluationCampaignArchive,
    ) -> Result<CurriculumDecisionV1, CurriculumArchiveError> {
        let state = self.load_state()?;
        require_next_generation(&state, generation, &checkpoint)?;
        if !self
            .generation_campaign
            .read_plan(generation)?
            .curriculum_evaluation_due
        {
            return Err(invalid(
                "an Elo evaluation cannot be attached to a generation whose plan defers it",
            ));
        }
        let identity = evaluation.identity();
        if identity.candidate_checkpoint_sha256 != checkpoint.sha256 {
            return Err(invalid(
                "evaluation candidate differs from the generation checkpoint",
            ));
        }
        let analysis = evaluation.analysis()?;
        let assessment =
            assess_curriculum_evaluation(state.current_tier, evaluation.identity(), &analysis)?;
        let relative_directory = relative_existing_path(&self.root, evaluation.root())?;
        let evidence = CurriculumEvaluationEvidenceV1 {
            run_sha256: sha256_file(&evaluation.root().join("run.json"))?,
            result_sha256: sha256_file(&evaluation.root().join("result/result.json"))?,
            relative_directory,
            conclusion: analysis.conclusion,
            elo_window: CurriculumEloWindowV1::from_evaluation(identity)?,
            contextual_elo_point: analysis.contextual_elo_point,
            davidson_candidate_minus_opponent: analysis
                .mle
                .as_ref()
                .map(|mle| mle.candidate_minus_opponent),
            eligible_pairs: analysis.eligible_pairs,
            excluded_pairs: analysis.excluded_pairs,
        };
        let decision = CurriculumDecisionV1 {
            sequence: state.decisions,
            generation,
            checkpoint,
            tier_before: state.current_tier,
            evidence: CurriculumEvidenceV1::FixedOpponent(Box::new(evidence)),
            assessment,
        };
        self.publish_decision(&decision)?;
        Ok(decision)
    }

    pub fn publish_deferred_evaluation_decision(
        &self,
        generation: u64,
        checkpoint: CheckpointReferenceV1,
    ) -> Result<CurriculumDecisionV1, CurriculumArchiveError> {
        let state = self.load_state()?;
        require_next_generation(&state, generation, &checkpoint)?;
        let plan = self.generation_campaign.read_plan(generation)?;
        if plan.curriculum_evaluation_due {
            return Err(invalid(
                "an Elo evaluation is due for this generation and cannot be deferred",
            ));
        }
        let outcome = self
            .generation_campaign
            .read_outcome(generation)?
            .ok_or_else(|| invalid("a deferred decision requires a completed generation"))?;
        if &checkpoint != training_checkpoint(&outcome) {
            return Err(invalid(
                "deferred decision cites the wrong active training checkpoint",
            ));
        }
        let decision = CurriculumDecisionV1 {
            sequence: state.decisions,
            generation,
            checkpoint,
            tier_before: state.current_tier,
            evidence: CurriculumEvidenceV1::DeferredEvaluation,
            assessment: deferred_evaluation_assessment(state.current_tier),
        };
        self.publish_decision(&decision)?;
        Ok(decision)
    }

    pub fn publish_self_play_decision(
        &self,
        generation: u64,
        checkpoint: CheckpointReferenceV1,
    ) -> Result<CurriculumDecisionV1, CurriculumArchiveError> {
        let state = self.load_state()?;
        require_next_generation(&state, generation, &checkpoint)?;
        if state.current_tier != CurriculumTierV1::SelfPlay {
            return Err(invalid(
                "a self-play decision requires the active self-play tier",
            ));
        }
        let outcome = self
            .generation_campaign
            .read_outcome(generation)?
            .ok_or_else(|| invalid("self-play decision requires a completed generation"))?;
        let decision = CurriculumDecisionV1 {
            sequence: state.decisions,
            generation,
            checkpoint,
            tier_before: state.current_tier,
            evidence: CurriculumEvidenceV1::SelfPlayPromotion {
                conclusion: outcome.promotion,
            },
            assessment: self_play_assessment(),
        };
        self.publish_decision(&decision)?;
        Ok(decision)
    }

    fn publish_decision(
        &self,
        decision: &CurriculumDecisionV1,
    ) -> Result<(), CurriculumArchiveError> {
        let path = self
            .root
            .join(DECISIONS_DIRECTORY)
            .join(decision_name(decision.sequence));
        write_document(&path, DECISION_FORMAT, decision)?;
        let state = self.load_state()?;
        if state.decisions != decision.sequence + 1
            || state.last_decided_generation != decision.generation
            || state.current_tier != decision.assessment.tier_after
        {
            return Err(invalid("published curriculum decision failed verification"));
        }
        Ok(())
    }

    fn load_state_with_evidence(
        &self,
        semantic_evidence: bool,
    ) -> Result<CurriculumStateV1, CurriculumArchiveError> {
        if read_document::<CurriculumCampaignIdentityV1>(
            &self.root.join(IDENTITY_FILE),
            IDENTITY_FORMAT,
        )? != self.identity
        {
            return Err(invalid("curriculum identity changed after opening"));
        }
        if sha256_file(
            &PathBuf::from(&self.identity.generation_campaign_path).join("campaign.json"),
        )? != self.identity.generation_campaign_sha256
        {
            return Err(invalid("generation campaign identity changed"));
        }
        self.generation_campaign.load_chain()?;
        let (baseline_checkpoint, baseline_champion) =
            roles_after_generation(&self.generation_campaign, self.identity.baseline_generation)?;
        if baseline_checkpoint != self.identity.baseline_checkpoint
            || baseline_champion != self.identity.baseline_champion
        {
            return Err(invalid(
                "curriculum baseline differs from the generation history",
            ));
        }

        let paths = decision_paths(&self.root.join(DECISIONS_DIRECTORY))?;
        let mut state = CurriculumStateV1 {
            decisions: 0,
            last_decided_generation: self.identity.baseline_generation,
            current_checkpoint: self.identity.baseline_checkpoint.clone(),
            current_champion: self.identity.baseline_champion.clone(),
            current_tier: self.identity.initial_tier,
        };
        for (expected_sequence, path) in paths.into_iter().enumerate() {
            let expected_sequence = u64::try_from(expected_sequence)
                .map_err(|_| invalid("curriculum decision count does not fit u64"))?;
            let decision: CurriculumDecisionV1 = read_document(&path, DECISION_FORMAT)?;
            if decision.sequence != expected_sequence {
                return Err(invalid("curriculum decision sequence is not contiguous"));
            }
            validate_decision(self, &state, &decision, semantic_evidence)?;
            let outcome = self
                .generation_campaign
                .read_outcome(decision.generation)?
                .expect("decision validation requires a completed generation");
            state.decisions += 1;
            state.last_decided_generation = decision.generation;
            state.current_checkpoint = decision.checkpoint;
            state.current_champion = outcome.champion_after;
            state.current_tier = decision.assessment.tier_after;
        }
        Ok(state)
    }
}

fn validate_decision(
    archive: &CurriculumCampaignArchive,
    state: &CurriculumStateV1,
    decision: &CurriculumDecisionV1,
    semantic_evidence: bool,
) -> Result<(), CurriculumArchiveError> {
    require_next_generation(state, decision.generation, &decision.checkpoint)?;
    if decision.sequence != state.decisions || decision.tier_before != state.current_tier {
        return Err(invalid(
            "curriculum decision sequence or incoming tier is inconsistent",
        ));
    }
    let outcome = archive
        .generation_campaign
        .read_outcome(decision.generation)?
        .ok_or_else(|| invalid("curriculum decision cites an unfinished generation"))?;
    let plan = archive.generation_campaign.read_plan(decision.generation)?;
    if outcome.parent_champion != state.current_champion {
        return Err(invalid(
            "curriculum decision does not follow the champion lineage",
        ));
    }
    if plan.training_parent() != &state.current_checkpoint
        || plan.actor.opponent != state.current_tier.training_opponent()
    {
        return Err(invalid(
            "generation training parent or opponent differs from the curriculum state",
        ));
    }
    let expected_checkpoint = training_checkpoint(&outcome);
    if &decision.checkpoint != expected_checkpoint {
        return Err(invalid(
            "curriculum decision cites the wrong active training checkpoint",
        ));
    }
    decision.checkpoint.verify(
        archive.generation_campaign.root(),
        &archive.generation_campaign.identity().network_preset,
    )?;

    let expected_assessment = match &decision.evidence {
        CurriculumEvidenceV1::FixedOpponent(evidence) => validate_evaluation_evidence(
            archive,
            state.current_tier,
            &decision.checkpoint,
            evidence,
            semantic_evidence,
        )?,
        CurriculumEvidenceV1::DeferredEvaluation => {
            if plan.curriculum_evaluation_due {
                return Err(invalid(
                    "curriculum evidence defers an Elo evaluation required by the generation plan",
                ));
            }
            deferred_evaluation_assessment(state.current_tier)
        }
        CurriculumEvidenceV1::SelfPlayPromotion { conclusion } => {
            if state.current_tier != CurriculumTierV1::SelfPlay || *conclusion != outcome.promotion
            {
                return Err(invalid("self-play evidence differs from its generation"));
            }
            self_play_assessment()
        }
    };
    if decision.assessment != expected_assessment {
        return Err(invalid(
            "curriculum action differs from its immutable evidence",
        ));
    }
    Ok(())
}

fn validate_evaluation_evidence(
    archive: &CurriculumCampaignArchive,
    tier: CurriculumTierV1,
    checkpoint: &CheckpointReferenceV1,
    evidence: &CurriculumEvaluationEvidenceV1,
    semantic: bool,
) -> Result<CurriculumAssessmentV1, CurriculumArchiveError> {
    validate_relative_path(&evidence.relative_directory)?;
    validate_digest(&evidence.run_sha256, "evaluation run SHA-256")?;
    validate_digest(&evidence.result_sha256, "evaluation result SHA-256")?;
    let path = archive.root.join(&evidence.relative_directory);
    if sha256_file(&path.join("run.json"))? != evidence.run_sha256
        || sha256_file(&path.join("result/result.json"))? != evidence.result_sha256
    {
        return Err(invalid("curriculum evaluation evidence changed"));
    }
    let evaluation = if semantic {
        EvaluationCampaignArchive::open_existing(&path)?
    } else {
        EvaluationCampaignArchive::open_existing_integrity(&path)?
    };
    let analysis = if semantic {
        evaluation.analysis()?
    } else {
        evaluation.published_analysis_integrity()?
    };
    if evaluation.identity().candidate_checkpoint_sha256 != checkpoint.sha256
        || evidence.conclusion != analysis.conclusion
        || evidence.elo_window != CurriculumEloWindowV1::from_evaluation(evaluation.identity())?
        || evidence.contextual_elo_point != analysis.contextual_elo_point
        || evidence.davidson_candidate_minus_opponent
            != analysis
                .mle
                .as_ref()
                .map(|mle| mle.candidate_minus_opponent)
        || evidence.eligible_pairs != analysis.eligible_pairs
        || evidence.excluded_pairs != analysis.excluded_pairs
    {
        return Err(invalid(
            "curriculum evaluation summary differs from its archive",
        ));
    }
    Ok(assess_curriculum_evaluation(
        tier,
        evaluation.identity(),
        &analysis,
    )?)
}

fn require_next_generation(
    state: &CurriculumStateV1,
    generation: u64,
    checkpoint: &CheckpointReferenceV1,
) -> Result<(), CurriculumArchiveError> {
    if generation != state.last_decided_generation.saturating_add(1)
        || checkpoint.generation > generation
    {
        return Err(invalid(
            "curriculum decisions must follow completed generations one by one",
        ));
    }
    Ok(())
}

fn roles_after_generation(
    campaign: &GenerationCampaignArchive,
    generation: u64,
) -> Result<(CheckpointReferenceV1, CheckpointReferenceV1), CurriculumArchiveError> {
    let mut champion = campaign.identity().genesis_checkpoint.clone();
    let mut training = champion.clone();
    if generation < champion.generation {
        return Err(invalid("curriculum baseline predates campaign genesis"));
    }
    let first = champion
        .generation
        .checked_add(1)
        .ok_or_else(|| invalid("campaign genesis has no successor"))?;
    for current in first..=generation {
        let outcome = campaign
            .read_outcome(current)?
            .ok_or_else(|| invalid("curriculum baseline includes an unfinished generation"))?;
        training = training_checkpoint(&outcome).clone();
        champion = outcome.champion_after;
    }
    Ok((training, champion))
}

fn training_checkpoint(outcome: &crate::GenerationOutcomeV1) -> &CheckpointReferenceV1 {
    if outcome.promotion == Some(PromotionCampaignConclusion::RejectCandidate) {
        &outcome.champion_after
    } else {
        &outcome.candidate
    }
}

fn decision_paths(directory: &Path) -> Result<Vec<PathBuf>, CurriculumArchiveError> {
    if !directory.is_dir() {
        return Err(invalid("curriculum decisions directory is missing"));
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        if !entry.file_type()?.is_file() || parse_decision_name(&name).is_none() {
            return Err(invalid(format!(
                "unexpected curriculum decision entry {name}"
            )));
        }
        paths.push((
            parse_decision_name(&name).expect("checked above"),
            entry.path(),
        ));
    }
    paths.sort_by_key(|(sequence, _)| *sequence);
    Ok(paths.into_iter().map(|(_, path)| path).collect())
}

fn decision_name(sequence: u64) -> String {
    format!("decision-{sequence:020}.json")
}

fn parse_decision_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("decision-")?.strip_suffix(".json")?;
    (digits.len() == 20 && digits.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

#[derive(Deserialize, Serialize)]
struct SealedDocument<T> {
    format: String,
    payload: T,
    sha256: String,
}

fn write_document<T: Serialize>(
    path: &Path,
    format: &str,
    payload: &T,
) -> Result<(), CurriculumArchiveError> {
    let payload_bytes = serde_json::to_vec(payload)?;
    let mut hasher = Sha256::new();
    hasher.update(format.as_bytes());
    hasher.update([0]);
    hasher.update(&payload_bytes);
    let document = SealedDocument {
        format: format.to_owned(),
        payload,
        sha256: hex_digest(hasher.finalize().into()),
    };
    let mut bytes = serde_json::to_vec_pretty(&document)?;
    bytes.push(b'\n');
    atomic_file::write_idempotently(path, &bytes)?;
    Ok(())
}

fn read_document<T: DeserializeOwned + Serialize>(
    path: &Path,
    expected_format: &str,
) -> Result<T, CurriculumArchiveError> {
    let bytes = fs::read(path)?;
    let document: SealedDocument<T> = serde_json::from_slice(&bytes)?;
    if document.format != expected_format {
        return Err(invalid("unsupported curriculum document format"));
    }
    let payload_bytes = serde_json::to_vec(&document.payload)?;
    let mut hasher = Sha256::new();
    hasher.update(expected_format.as_bytes());
    hasher.update([0]);
    hasher.update(&payload_bytes);
    let recomputed = hex_digest(hasher.finalize().into());
    if document.sha256 != recomputed {
        return Err(invalid(format!(
            "curriculum document checksum mismatch: stored {}, recomputed {recomputed}",
            document.sha256
        )));
    }
    Ok(document.payload)
}

fn relative_existing_path(root: &Path, path: &Path) -> Result<String, CurriculumArchiveError> {
    let root = fs::canonicalize(root)?;
    let path = fs::canonicalize(path)?;
    let relative = path
        .strip_prefix(&root)
        .map_err(|_| invalid("evaluation archive is outside the curriculum campaign"))?;
    let text = utf8_path(relative)?;
    validate_relative_path(&text)?;
    Ok(text)
}

fn validate_relative_path(text: &str) -> Result<(), CurriculumArchiveError> {
    let path = Path::new(text);
    if text.is_empty()
        || path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        Err(invalid("curriculum path is not a safe relative path"))
    } else {
        Ok(())
    }
}

fn validate_digest(text: &str, name: &str) -> Result<(), CurriculumArchiveError> {
    if text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(invalid(format!("{name} is not a lowercase SHA-256")))
    }
}

fn sha256_file(path: &Path) -> Result<String, CurriculumArchiveError> {
    Ok(hex_digest(Sha256::digest(fs::read(path)?).into()))
}

fn hex_digest(digest: [u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in digest {
        use core::fmt::Write as _;
        write!(text, "{byte:02x}").expect("writing a digest to String cannot fail");
    }
    text
}

fn utf8_path(path: &Path) -> Result<String, CurriculumArchiveError> {
    path.to_str()
        .map(str::to_owned)
        .ok_or_else(|| invalid(format!("{} is not valid UTF-8", path.display())))
}

fn invalid(message: impl Into<String>) -> CurriculumArchiveError {
    CurriculumArchiveError::InvalidData(message.into())
}

#[derive(Debug)]
pub enum CurriculumArchiveError {
    Io(io::Error),
    Json(serde_json::Error),
    Generation(GenerationArchiveError),
    Evaluation(EvaluationArchiveError),
    Decision(CurriculumDecisionError),
    InvalidData(String),
}

impl fmt::Display for CurriculumArchiveError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => source.fmt(formatter),
            Self::Json(source) => source.fmt(formatter),
            Self::Generation(source) => source.fmt(formatter),
            Self::Evaluation(source) => source.fmt(formatter),
            Self::Decision(source) => source.fmt(formatter),
            Self::InvalidData(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for CurriculumArchiveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(source) => Some(source),
            Self::Json(source) => Some(source),
            Self::Generation(source) => Some(source),
            Self::Evaluation(source) => Some(source),
            Self::Decision(source) => Some(source),
            Self::InvalidData(_) => None,
        }
    }
}

impl From<io::Error> for CurriculumArchiveError {
    fn from(source: io::Error) -> Self {
        Self::Io(source)
    }
}

impl From<serde_json::Error> for CurriculumArchiveError {
    fn from(source: serde_json::Error) -> Self {
        Self::Json(source)
    }
}

impl From<GenerationArchiveError> for CurriculumArchiveError {
    fn from(source: GenerationArchiveError) -> Self {
        Self::Generation(source)
    }
}

impl From<EvaluationArchiveError> for CurriculumArchiveError {
    fn from(source: EvaluationArchiveError) -> Self {
        Self::Evaluation(source)
    }
}

impl From<CurriculumDecisionError> for CurriculumArchiveError {
    fn from(source: CurriculumDecisionError) -> Self {
        Self::Decision(source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paisho_core::RuleProfileId;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn decision_names_are_fixed_width_and_round_trip() {
        for sequence in [0, 1, 42, u64::MAX] {
            let name = decision_name(sequence);
            assert_eq!(parse_decision_name(&name), Some(sequence));
        }
        assert_eq!(parse_decision_name("decision-1.json"), None);
        assert_eq!(parse_decision_name("batch-00000000000000000001"), None);
    }

    #[test]
    fn relative_evidence_paths_cannot_escape_the_campaign() {
        assert!(validate_relative_path("evaluations/run-1").is_ok());
        assert!(validate_relative_path("../outside").is_err());
        assert!(validate_relative_path("/absolute").is_err());
        assert!(validate_relative_path("").is_err());
    }

    #[test]
    fn sealed_documents_reject_payload_tampering() {
        let root =
            std::env::temp_dir().join(format!("paisho-curriculum-document-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let path = root.join("document.json");
        write_document(&path, "test", &vec![1_u64, 2, 3]).unwrap();
        assert_eq!(
            read_document::<Vec<u64>>(&path, "test").unwrap(),
            vec![1, 2, 3]
        );
        let mut text = fs::read_to_string(&path).unwrap();
        text = text.replacen('2', "9", 1);
        fs::write(&path, text).unwrap();
        assert!(read_document::<Vec<u64>>(&path, "test").is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sealed_documents_round_trip_rating_floats_exactly() {
        let temporary = TemporaryDirectory::new();
        let path = temporary.path.join("rating-floats.json");
        let values = [
            110.506_056_664_702_17_f64,
            116.946_274_704_335_27,
            -118.704_429_881_922_51,
        ];
        write_document(&path, "test", &values).unwrap();
        let restored = read_document::<[f64; 3]>(&path, "test").unwrap();
        assert_eq!(restored.map(f64::to_bits), values.map(f64::to_bits));
    }

    #[test]
    fn campaign_initialization_captures_a_resumable_generation_baseline() {
        let temporary = TemporaryDirectory::new();
        let campaign_root = temporary.path.join("generations");
        fs::create_dir(&campaign_root).unwrap();
        let checkpoint_path = campaign_root.join("genesis.psckpt");
        write_test_checkpoint(&checkpoint_path);
        let checkpoint =
            CheckpointReferenceV1::from_checkpoint(&campaign_root, &checkpoint_path).unwrap();
        let generation = GenerationCampaignArchive::open_or_create(
            &campaign_root,
            crate::CampaignIdentityV1 {
                rules: RuleProfileId::SkudPaiSho2022.as_str().to_owned(),
                network_preset: "micro".to_owned(),
                genesis_checkpoint: checkpoint.clone(),
                genesis_kind: "imported".to_owned(),
                genesis_source_revision: "test".to_owned(),
                genesis_source_dirty: false,
                genesis_source_sha256: "11".repeat(32),
                genesis_service_sha256: "22".repeat(32),
                genesis_model_seed: 17,
                genesis_learning_rate_bits: 1.0e-4_f32.to_bits(),
            },
        )
        .unwrap();
        let root = temporary.path.join("curriculum");
        let curriculum =
            CurriculumCampaignArchive::open_or_create(&root, &generation, CurriculumTierV1::Random)
                .unwrap();
        assert_eq!(
            curriculum.load_state().unwrap(),
            CurriculumStateV1 {
                decisions: 0,
                last_decided_generation: 0,
                current_checkpoint: checkpoint.clone(),
                current_champion: checkpoint,
                current_tier: CurriculumTierV1::Random,
            }
        );
        assert_eq!(
            CurriculumCampaignArchive::open_existing(root)
                .unwrap()
                .load_state()
                .unwrap()
                .current_tier,
            CurriculumTierV1::Random
        );
    }

    fn write_test_checkpoint(path: &Path) {
        let metadata = serde_json::json!({
            "formatVersion": 2,
            "tensorSchema": "paisho-neural-encoding-v1",
            "ruleProfile": "skud-pai-sho-2022-03-14",
            "configuration": {
                "trunkChannels": 20,
                "residualBlocks": 3,
                "policyEmbeddingChannels": 8,
                "valueHiddenChannels": 32,
                "normalizationEpsilon": 1.0e-5
            },
            "optimization": "level1",
            "trainingStep": 0,
            "progress": {
                "generation": 0,
                "replayIndex": 0,
                "replaySnapshotSHA256": "00".repeat(32),
                "scheduler": {
                    "learningRate": 1.0e-4,
                    "completedSteps": 0
                }
            }
        });
        let metadata = serde_json::to_vec(&metadata).unwrap();
        let mut bytes = b"PAISHO-CKPT-V2\n".to_vec();
        bytes.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&metadata);
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        fs::write(path, bytes).unwrap();
    }

    struct TemporaryDirectory {
        path: PathBuf,
    }

    impl TemporaryDirectory {
        fn new() -> Self {
            let ordinal = TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "paisho-curriculum-{}-{ordinal}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self { path }
        }
    }

    impl Drop for TemporaryDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}
