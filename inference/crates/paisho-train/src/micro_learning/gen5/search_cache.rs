//! Exact, bounded reanalysis results. Observed game outcomes never enter this cache.
use super::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
#[cfg(test)]
mod tests;

#[derive(Clone)]
struct Entry {
    report: MicroSearchReport,
    certificate: Option<MicroProofCertificate>,
    bytes: usize,
}
#[derive(Default)]
struct Inner {
    entries: HashMap<String, Arc<Entry>>,
    order: VecDeque<String>,
    bytes: usize,
    hits: usize,
    misses: usize,
    avoided: usize,
}
/// One cache is shared by all independent actors; locks cover lookup only, never search.
pub struct ReanalysisCache {
    inner: Mutex<Inner>,
    max_bytes: usize,
}
impl std::fmt::Debug for ReanalysisCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReanalysisCache")
            .field("max_bytes", &self.max_bytes)
            .finish()
    }
}
impl ReanalysisCache {
    pub(super) fn new(max_bytes: usize) -> Self {
        Self {
            inner: Mutex::default(),
            max_bytes,
        }
    }
    fn get(&self, key: &str) -> Option<Arc<Entry>> {
        let mut c = self.inner.lock().unwrap();
        let found = c.entries.get(key).cloned();
        if let Some(e) = &found {
            c.hits += 1;
            c.avoided += e.report.simulations;
            if let Some(i) = c.order.iter().position(|k| k == key) {
                c.order.remove(i);
            }
            c.order.push_back(key.into());
        } else {
            c.misses += 1;
        }
        found
    }
    fn insert(
        &self,
        key: String,
        report: &MicroSearchReport,
        certificate: Option<MicroProofCertificate>,
    ) -> std::result::Result<(), String> {
        let n = report.actions.len();
        let bytes = 2048
            + report.raw_priors.as_ref().map_or(0,|v|v.capacity()*8)
            + report.state.capacity() * 8
            + n * 640
            + certificate
                .as_ref()
                .map_or(Ok(0), |c| serde_json::to_vec(c).map(|v| v.len()))
                .map_err(|e| e.to_string())?;
        if bytes > self.max_bytes {
            return Ok(());
        }
        let entry = Arc::new(Entry {
            report: report.clone(),
            certificate,
            bytes,
        });
        let mut c = self.inner.lock().unwrap();
        if c.entries.contains_key(&key) {
            return Ok(());
        }
        while c.bytes + bytes > self.max_bytes || c.entries.len() >= 4096 {
            let old = c.order.pop_front().ok_or("cache accounting mismatch")?;
            if let Some(e) = c.entries.remove(&old) {
                c.bytes -= e.bytes;
            }
        }
        c.bytes += bytes;
        c.order.push_back(key.clone());
        c.entries.insert(key, entry);
        Ok(())
    }
    pub(super) fn progress(&self) -> serde_json::Value {
        let c = self.inner.lock().unwrap();
        serde_json::json!({"hits":c.hits,"misses":c.misses,"avoided_simulations":c.avoided,"entries":c.entries.len(),"bytes":c.bytes,"max_bytes":self.max_bytes,"outcomes_cached":false})
    }
}

pub(super) fn key(
    model: &str,
    prefix: &str,
    budget: usize,
    beta: f64,
    options: MicroSearchOptions,
    certificate: &Option<MicroProofCertificate>,
) -> std::result::Result<String, String> {
    // PUCT without noise/forced playouts does not use the seed. Gumbel is excluded.
    let mut canonical = options;
    canonical.seed = 0;
    Ok(sha256(&serde_json::to_vec(&serde_json::json!({"schema":"gen5-reanalysis-cache-v1","model_and_bank":model,"rules":RULES.as_str(),"prefix":prefix,"budget":budget,"beta":beta,"options":format!("{canonical:?}"),"source_exclusion":0,"root_certificate":certificate})).map_err(|e|e.to_string())?))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn search(
    session: &mut MicroMctsSession,
    position: &paisho_core::Position,
    prefix: &GameRecord,
    model: &str,
    budget: usize,
    beta: f64,
    options: MicroSearchOptions,
    deadline: Instant,
    proofs: Option<&durable::Proofs>,
    cache: Option<&ReanalysisCache>,
) -> std::result::Result<
    (
        MicroSearchReport,
        Option<MicroProofCertificate>,
        bool,
        Option<String>,
    ),
    String,
> {
    let text = prefix.to_string();
    let certificate = match proofs {
        Some(p) => proof_cache::lookup(p, &sha256(text.as_bytes())).map_err(|e| e.to_string())?,
        None => None,
    };
    let cache = cache.filter(|_| {
        options.mode == MicroSearchMode::Puct
            && options.dirichlet_fraction == 0.
            && options.forced_playout_strength == 0.
    });
    let cache_key = if cache.is_some() {
        Some(key(model, &text, budget, beta, options, &certificate)?)
    } else {
        None
    };
    if let (Some(cache), Some(key)) = (cache, &cache_key) {
        if let Some(e) = cache.get(key) {
            return Ok((e.report.clone(), e.certificate.clone(), true, cache_key));
        }
    }
    if let Some(c) = &certificate {
        session.install_certificate(position, c)?;
    }
    let report = session.search_with_options(position, budget, Some(deadline), options)?;
    let proof = if report.proven_value.is_some() {
        session.certificate(10_000)
    } else {
        None
    };
    // Only a completed fresh-root request is reusable; partial deadline searches are not.
    if report.inherited_visits == 0
        && (report.simulations == budget || report.proven_value.is_some())
    {
        if let (Some(cache), Some(key)) = (cache, &cache_key) {
            cache.insert(key.clone(), &report, proof.clone())?;
        }
    }
    Ok((report, proof, false, cache_key))
}
