//! Hash-bound immutable memory dependencies, shared across model publications.
use super::*;
use std::sync::{Arc, Mutex, OnceLock};
pub fn load_sequence_memory(spec: &SequenceMemorySpec) -> Result<Arc<SequenceBank>> {
    static BANKS: OnceLock<Mutex<Vec<(String, Arc<SequenceBank>)>>> = OnceLock::new();
    let mut banks = BANKS.get_or_init(Default::default).lock().unwrap();
    let key = format!("{}:{}", spec.path, spec.sha256);
    if let Some(bank) = banks
        .iter()
        .find(|(k, _)| k == &key)
        .map(|(_, b)| b.clone())
    {
        return Ok(bank);
    }
    let bytes = fs::read(&spec.path)?;
    let bank = sequence_memory_from_bytes(spec.clone(), &bytes)?;
    if banks.len() >= 2 {
        banks.remove(0);
    }
    banks.push((key, bank.clone()));
    Ok(bank)
}
pub fn sequence_memory_from_bytes(
    spec: SequenceMemorySpec,
    bytes: &[u8],
) -> Result<Arc<SequenceBank>> {
    if sha256(bytes) != spec.sha256 {
        return Err(invalid("sequence bank SHA256 mismatch"));
    }
    Ok(Arc::new(
        SequenceBank::read_from(&mut &*bytes, spec).map_err(invalid)?,
    ))
}
