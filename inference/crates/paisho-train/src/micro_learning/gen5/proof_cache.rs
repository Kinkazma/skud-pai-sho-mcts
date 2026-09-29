//! Bounded RAM for verified exact proofs, with permanent content-addressed disk lookup.
use super::*;
use std::{
    collections::{HashMap, VecDeque},
    sync::RwLock,
};
pub(super) type Proofs = Arc<RwLock<Cache>>;
#[derive(Default)]
pub(super) struct Cache {
    pub disk_count: usize,
    pub generation: u64,
    directory: Option<PathBuf>,
    read_only: Vec<PathBuf>,
    entries: HashMap<String, MicroProofCertificate>,
    order: VecDeque<String>,
}
impl Cache {
    pub fn open(root: &Path) -> Result<Self> {
        let directory = root.join("proofs");
        fs::create_dir_all(&directory)?;
        let disk_count = fs::read_dir(&directory)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|s| s == "json"))
            .count();
        Ok(Self {
            disk_count,
            directory: Some(directory),
            ..Default::default()
        })
    }
    pub fn add_read_only(&mut self, root: &Path) -> Result<()> {
        let dir = root.join("proofs");
        if dir.exists() {
            self.disk_count += fs::read_dir(&dir)?
                .filter_map(|e| e.ok())
                .filter(|e| e.path().extension().is_some_and(|s| s == "json"))
                .count();
            self.read_only.push(dir);
        }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.directory.is_none()
    }
    pub fn insert(&mut self, key: String, certificate: MicroProofCertificate) {
        if self.entries.get(&key) != Some(&certificate) {
            self.generation = self.generation.wrapping_add(1);
        }
        self.insert_cached(key, certificate);
    }
    fn insert_cached(&mut self, key: String, certificate: MicroProofCertificate) {
        if !self.entries.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.entries.insert(key, certificate);
        while self.entries.len() > 8192 {
            if let Some(key) = self.order.pop_front() {
                self.entries.remove(&key);
            }
        }
    }
}
pub(super) fn lookup(shared: &Proofs, key: &str) -> Result<Option<MicroProofCertificate>> {
    let directories = {
        let cache = shared.read().unwrap();
        if let Some(c) = cache.entries.get(key) {
            return Ok(Some(c.clone()));
        }
        cache
            .directory
            .iter()
            .cloned()
            .chain(cache.read_only.iter().cloned())
            .collect::<Vec<_>>()
    };
    let Some(path) = directories
        .iter()
        .map(|dir| dir.join(format!("{key}.json")))
        .find(|p| p.exists())
    else {
        return Ok(None);
    };
    let value: serde_json::Value = serde_json::from_slice(&fs::read(path)?)?;
    if value["rules"] != RULES.as_str() {
        return Err(invalid("proof rules mismatch"));
    }
    let prefix = value["prefix"]
        .as_str()
        .ok_or_else(|| invalid("proof prefix missing"))?;
    if sha256(prefix.as_bytes()) != key {
        return Err(invalid("proof prefix identity mismatch"));
    }
    let record: GameRecord = prefix.parse()?;
    if record.rules() != RULES {
        return Err(invalid("proof record rules mismatch"));
    }
    let certificate: MicroProofCertificate = serde_json::from_value(value["certificate"].clone())?;
    certificate.verify(&record.replay()?).map_err(invalid)?;
    let mut cache = shared.write().unwrap();
    // A concurrently registered proof is newer than this disk read.
    if let Some(current) = cache.entries.get(key) {
        return Ok(Some(current.clone()));
    }
    cache.insert_cached(key.into(), certificate.clone());
    Ok(Some(certificate))
}
