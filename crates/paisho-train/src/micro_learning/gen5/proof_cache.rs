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
    directory: Option<PathBuf>,
    entries: HashMap<String, MicroProofCertificate>,
    order: VecDeque<String>,
}
impl Cache {
    pub fn open(root: &Path) -> Result<Self> {
        let directory = root.join("proofs");
        fs::create_dir_all(&directory)?;
        let disk_count=fs::read_dir(&directory)?.filter_map(|e|e.ok()).filter(|e|e.path().extension().is_some_and(|s|s=="json")).count();
        Ok(Self {
            disk_count,
            directory: Some(directory),
            ..Default::default()
        })
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.directory.is_none()
    }
    pub fn insert(&mut self, key: String, certificate: MicroProofCertificate) {
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
pub(super) fn lookup(shared:&Proofs, key:&str) -> Result<Option<MicroProofCertificate>> {
        let directory={
            let cache=shared.read().unwrap();
            if let Some(c)=cache.entries.get(key) { return Ok(Some(c.clone())); }
            cache.directory.clone()
        };
        let Some(dir)=directory else { return Ok(None); };
        let path = dir.join(format!("{key}.json"));
        if !path.exists() {
            return Ok(None);
        }
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
        let certificate: MicroProofCertificate =
            serde_json::from_value(value["certificate"].clone())?;
        certificate.verify(&record.replay()?).map_err(invalid)?;
        shared.write().unwrap().insert(key.into(), certificate.clone());
        Ok(Some(certificate))
    }

