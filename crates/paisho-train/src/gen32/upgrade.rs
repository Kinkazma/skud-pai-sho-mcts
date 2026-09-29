//! Explicit, lossless-parent migration; never mutates a running campaign.
use super::*;
pub fn upgrade(
    parent: &Path,
    out: &Path,
    directory: &Path,
    scope: &str,
    value128: bool,
) -> Result<()> {
    if out.exists() {
        return Err(invalid("output already exists"));
    }
    let mut artifact = Artifact::load(parent)?;
    if artifact.value_residual.is_some() {
        return Err(invalid(
            "legacy memory upgrade cannot rewrite a residual artifact",
        ));
    }
    let mut model = artifact.model()?;
    model.memory_scope = Gen3MemoryScope::parse(scope).map_err(invalid)?;
    if value128 && model.value_extra.is_none() {
        model = model.with_value128([0.; 64]).map_err(invalid)?;
    }
    let manifest = directory.join("summary.json").canonicalize()?;
    let bytes = fs::read(&manifest)?;
    let meta: serde_json::Value = serde_json::from_slice(&bytes)?;
    if meta["rules"] != RULES.as_str() {
        return Err(invalid("memory must use Gen3 V2 rules"));
    }
    let spec = SequenceMemorySpec {
        path: directory
            .join("memory.bin")
            .canonicalize()?
            .to_string_lossy()
            .into_owned(),
        sha256: meta["sha256"]
            .as_str()
            .ok_or_else(|| invalid("missing bank hash"))?
            .into(),
    };
    model = model.with_memory(crate::micro_learning::load_sequence_memory(&spec)?);
    artifact.schema = "paisho-gen34-value128-memory-v1".into();
    artifact.generation = "3.4".into();
    artifact.parent_sha256 = sha256(&fs::read(parent)?);
    artifact.memory_manifest = Some(manifest);
    artifact.memory_manifest_sha256 = Some(sha256(&bytes));
    artifact.updated(&model, artifact.updates).save(out)
}

/// Preserve generation, old weights, update counters and memory dependencies.
/// Output is separate; this does not stage or start a campaign.
pub fn upgrade_value_residual(parent: &Path, out: &Path, seed: u64) -> Result<()> {
    if out.exists() {
        return Err(invalid("output already exists"));
    }
    let mut artifact = Artifact::load(parent)?;
    if artifact.value_residual.is_some() {
        return Err(invalid("model already has a value residual"));
    }
    let model = artifact.model()?.with_value_residual(seed);
    artifact.parent_sha256 = sha256(&fs::read(parent)?);
    artifact.updated(&model, artifact.updates).save(out)
}
