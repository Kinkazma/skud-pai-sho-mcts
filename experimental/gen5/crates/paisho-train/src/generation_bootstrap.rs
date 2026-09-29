use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use paisho_core::RuleProfileId;
use paisho_model::{CheckpointRandomStateV1, CheckpointRequestV1};
use paisho_mpsgraph_client::{
    read_checkpoint_metadata, MpsGraphProcess, NetworkPreset, OptimizationLevel,
    ServiceConfiguration,
};
use paisho_replay::ReplaySnapshotV1;

use crate::{
    CampaignIdentityV1, CheckpointReferenceV1, GenerationArchiveError, GenerationCampaignArchive,
};

static BOOTSTRAP_ATTEMPT: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct GenerationBootstrapConfiguration {
    pub campaign_directory: PathBuf,
    pub service_executable: PathBuf,
    pub initial_checkpoint: Option<PathBuf>,
    pub network_preset: NetworkPreset,
    pub optimization: OptimizationLevel,
    pub batch_size: usize,
    pub legal_action_capacity: usize,
    pub model_seed: u64,
    pub learning_rate: f32,
    pub source_revision: String,
    pub source_dirty: bool,
    pub source_sha256: String,
    pub service_sha256: String,
}

pub fn open_or_initialize_generation_campaign(
    configuration: &GenerationBootstrapConfiguration,
) -> Result<GenerationCampaignArchive, GenerationArchiveError> {
    if configuration
        .campaign_directory
        .join("campaign.json")
        .exists()
    {
        let archive = GenerationCampaignArchive::open_existing(&configuration.campaign_directory)?;
        validate_existing_identity(configuration, archive.identity())?;
        return Ok(archive);
    }
    validate_configuration(configuration)?;
    fs::create_dir_all(&configuration.campaign_directory)?;
    let genesis_directory = configuration.campaign_directory.join("genesis");
    let identity = if genesis_directory.exists() {
        crate::generation::read_campaign_identity_from_genesis(&genesis_directory)?
    } else {
        publish_genesis(configuration, &genesis_directory)?
    };
    validate_existing_identity(configuration, &identity)?;
    GenerationCampaignArchive::open_or_create(&configuration.campaign_directory, identity)
}

fn validate_existing_identity(
    configuration: &GenerationBootstrapConfiguration,
    identity: &CampaignIdentityV1,
) -> Result<(), GenerationArchiveError> {
    if identity.network_preset != preset_name(configuration.network_preset) {
        return Err(GenerationArchiveError::Invalid(
            "existing campaign genesis uses a different network preset".to_owned(),
        ));
    }
    if let Some(initial) = &configuration.initial_checkpoint {
        let metadata = read_checkpoint_metadata(initial)?;
        if crate::generation::hex_digest_for_internal(metadata.content_sha256())
            != identity.genesis_checkpoint.sha256
        {
            return Err(GenerationArchiveError::Invalid(
                "supplied initial checkpoint differs from existing campaign genesis".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_configuration(
    configuration: &GenerationBootstrapConfiguration,
) -> Result<(), GenerationArchiveError> {
    if configuration.batch_size == 0 || configuration.legal_action_capacity == 0 {
        return Err(GenerationArchiveError::Invalid(
            "bootstrap execution shape must be positive".to_owned(),
        ));
    }
    if !configuration.learning_rate.is_finite() || configuration.learning_rate <= 0.0 {
        return Err(GenerationArchiveError::Invalid(
            "bootstrap learning rate must be positive and finite".to_owned(),
        ));
    }
    Ok(())
}

fn publish_genesis(
    configuration: &GenerationBootstrapConfiguration,
    destination: &Path,
) -> Result<CampaignIdentityV1, GenerationArchiveError> {
    let temporary = temporary_directory(&configuration.campaign_directory)?;
    let checkpoint_path = temporary.path.join("champion.psckpt");
    let kind = match &configuration.initial_checkpoint {
        Some(initial) => {
            copy_new_synced(initial, &checkpoint_path)?;
            "imported"
        }
        None => {
            publish_generated_checkpoint(configuration, &temporary.path, &checkpoint_path)?;
            "generated"
        }
    };
    let metadata = read_checkpoint_metadata(&checkpoint_path)?;
    if metadata.network_preset() != configuration.network_preset {
        return Err(GenerationArchiveError::Invalid(
            "genesis checkpoint uses a different network preset".to_owned(),
        ));
    }
    let final_checkpoint = destination.join("champion.psckpt");
    let reference = CheckpointReferenceV1 {
        relative_path: campaign_relative_path_for_unpublished(
            &configuration.campaign_directory,
            &final_checkpoint,
        )?,
        sha256: crate::generation::hex_digest_for_internal(metadata.content_sha256()),
        generation: metadata.generation(),
        training_step: metadata.training_step(),
    };
    let identity = CampaignIdentityV1 {
        rules: RuleProfileId::SkudPaiSho2022.as_str().to_owned(),
        network_preset: preset_name(configuration.network_preset).to_owned(),
        genesis_checkpoint: reference,
        genesis_kind: kind.to_owned(),
        genesis_source_revision: configuration.source_revision.clone(),
        genesis_source_dirty: configuration.source_dirty,
        genesis_source_sha256: configuration.source_sha256.clone(),
        genesis_service_sha256: configuration.service_sha256.clone(),
        genesis_model_seed: configuration.model_seed,
        genesis_learning_rate_bits: configuration.learning_rate.to_bits(),
    };
    identity.validate()?;
    crate::generation::write_campaign_identity_to_genesis(&temporary.path, &identity)?;
    sync_directory(&temporary.path)?;
    match fs::rename(&temporary.path, destination) {
        Ok(()) => {
            temporary.disarm();
            sync_directory(&configuration.campaign_directory)?;
            Ok(identity)
        }
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let stored = crate::generation::read_campaign_identity_from_genesis(destination)?;
            if stored == identity {
                Ok(stored)
            } else {
                Err(GenerationArchiveError::Invalid(
                    "another process published a different campaign genesis".to_owned(),
                ))
            }
        }
        Err(error) => Err(error.into()),
    }
}

fn publish_generated_checkpoint(
    configuration: &GenerationBootstrapConfiguration,
    directory: &Path,
    checkpoint_path: &Path,
) -> Result<(), GenerationArchiveError> {
    let snapshot = ReplaySnapshotV1::with_rules(Vec::new(), RuleProfileId::SkudPaiSho2022)?;
    let snapshot_path = directory.join("empty-snapshot.psrsnap");
    let snapshot_digest = snapshot.write_new(&snapshot_path)?;
    let service = ServiceConfiguration {
        executable: configuration.service_executable.clone(),
        preset: configuration.network_preset,
        batch_size: configuration.batch_size,
        legal_action_capacity: configuration.legal_action_capacity,
        inference_slots: 1,
        optimization: configuration.optimization,
        seed: configuration.model_seed,
        checkpoint: None,
    };
    let mut process = MpsGraphProcess::launch(service).map_err(|source| {
        GenerationArchiveError::Invalid(format!("genesis MPSGraph launch failed: {source}"))
    })?;
    let destination = checkpoint_path.to_str().ok_or_else(|| {
        GenerationArchiveError::Invalid("genesis checkpoint path is not UTF-8".to_owned())
    })?;
    let request = CheckpointRequestV1::new(
        0,
        0,
        *snapshot_digest.as_bytes(),
        0,
        0,
        configuration.learning_rate,
        vec![
            CheckpointRandomStateV1::new("model-seed", configuration.model_seed)
                .map_err(|source| GenerationArchiveError::Invalid(source.to_string()))?,
            CheckpointRandomStateV1::new("generation-bootstrap", 0)
                .map_err(|source| GenerationArchiveError::Invalid(source.to_string()))?,
        ],
        destination,
    )
    .map_err(|source| GenerationArchiveError::Invalid(source.to_string()))?;
    let publication = process.publish_checkpoint(&request).map_err(|source| {
        GenerationArchiveError::Invalid(format!("genesis checkpoint publication failed: {source}"))
    });
    let shutdown = process.shutdown().map_err(|source| {
        GenerationArchiveError::Invalid(format!("genesis MPSGraph shutdown failed: {source}"))
    });
    publication?;
    let status = shutdown?;
    if !status.success() {
        return Err(GenerationArchiveError::Invalid(format!(
            "genesis MPSGraph service exited with {status}"
        )));
    }
    Ok(())
}

fn copy_new_synced(source: &Path, destination: &Path) -> Result<(), GenerationArchiveError> {
    let mut input = File::open(source)?;
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(destination)?;
    io::copy(&mut input, &mut output)?;
    output.sync_all()?;
    Ok(())
}

fn campaign_relative_path_for_unpublished(
    root: &Path,
    path: &Path,
) -> Result<String, GenerationArchiveError> {
    let relative = path.strip_prefix(root).map_err(|_| {
        GenerationArchiveError::Invalid("genesis path is outside campaign root".to_owned())
    })?;
    relative
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| GenerationArchiveError::Invalid("genesis path is not UTF-8".to_owned()))
}

struct TemporaryDirectory {
    path: PathBuf,
    armed: std::cell::Cell<bool>,
}

impl TemporaryDirectory {
    fn disarm(&self) {
        self.armed.set(false);
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        if self.armed.get() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

fn temporary_directory(parent: &Path) -> Result<TemporaryDirectory, GenerationArchiveError> {
    loop {
        let attempt = BOOTSTRAP_ATTEMPT.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".genesis.partial-{}-{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => {
                return Ok(TemporaryDirectory {
                    path,
                    armed: std::cell::Cell::new(true),
                })
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

fn sync_directory(directory: &Path) -> io::Result<()> {
    File::open(directory)?.sync_all()
}

fn preset_name(preset: NetworkPreset) -> &'static str {
    match preset {
        NetworkPreset::Micro => "micro",
        NetworkPreset::Pure => "pure",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn configuration(network_preset: NetworkPreset) -> GenerationBootstrapConfiguration {
        GenerationBootstrapConfiguration {
            campaign_directory: PathBuf::from("campaign"),
            service_executable: PathBuf::from("paisho-mpsgraph-service"),
            initial_checkpoint: None,
            network_preset,
            optimization: OptimizationLevel::Level1,
            batch_size: 8,
            legal_action_capacity: 128,
            model_seed: 7,
            learning_rate: 1.0e-3,
            source_revision: "revision".to_owned(),
            source_dirty: false,
            source_sha256: "11".repeat(32),
            service_sha256: "22".repeat(32),
        }
    }

    fn identity(network_preset: &str) -> CampaignIdentityV1 {
        CampaignIdentityV1 {
            rules: RuleProfileId::SkudPaiSho2022.as_str().to_owned(),
            network_preset: network_preset.to_owned(),
            genesis_checkpoint: CheckpointReferenceV1 {
                relative_path: "genesis/champion.psckpt".to_owned(),
                sha256: "33".repeat(32),
                generation: 0,
                training_step: 0,
            },
            genesis_kind: "generated".to_owned(),
            genesis_source_revision: "revision".to_owned(),
            genesis_source_dirty: false,
            genesis_source_sha256: "11".repeat(32),
            genesis_service_sha256: "22".repeat(32),
            genesis_model_seed: 7,
            genesis_learning_rate_bits: 1.0e-3_f32.to_bits(),
        }
    }

    #[test]
    fn reopening_campaign_rejects_a_different_network_preset() {
        let existing = identity("micro");

        validate_existing_identity(&configuration(NetworkPreset::Micro), &existing).unwrap();
        let error =
            validate_existing_identity(&configuration(NetworkPreset::Pure), &existing).unwrap_err();

        assert!(error.to_string().contains("different network preset"));
    }

    #[test]
    fn legacy_genesis_snapshot_keeps_v1_before_service_launch() {
        let directory = std::env::temp_dir().join(format!(
            "paisho-legacy-genesis-rules-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let mut config = configuration(NetworkPreset::Micro);
        config.service_executable = directory.join("missing-service");
        let checkpoint = directory.join("checkpoint.psckpt");
        let error = publish_generated_checkpoint(&config, &directory, &checkpoint).unwrap_err();
        assert!(error.to_string().contains("genesis MPSGraph launch failed"));
        let snapshot = ReplaySnapshotV1::read(&directory.join("empty-snapshot.psrsnap")).unwrap();
        assert_eq!(snapshot.rule_profile(), RuleProfileId::SkudPaiSho2022);
        assert!(!checkpoint.exists());
        fs::remove_dir_all(directory).unwrap();
    }
}
