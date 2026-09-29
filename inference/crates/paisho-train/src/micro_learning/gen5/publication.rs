//! Local regression guard. This is a known-position retention check, not Elo.
use super::*;
use paisho_core::Position;
mod branches;
mod read_cache;
mod benchmark;
mod transaction_bench;
#[doc(hidden)]
pub use transaction_bench::run as benchmark_lazy_publication;
mod repair;
mod prefilter;
mod cached_margin_verify;
mod margin_step;
mod kl_decomposition;
mod gain_probe;
mod gain_transfer_probe;
mod policy_transfer;
mod learned_choices;
mod transmission;
mod transfer_bootstrap;
#[doc(hidden)]
pub use gain_transfer_probe::{run as probe_gain_transfer,run_relinearized as probe_gain_transfer_relinearized,run_policy_relay as probe_policy_relay};
mod dynamic_proof_probe;
pub use dynamic_proof_probe::run as probe_dynamic_proof_control;
pub(super) use gain_probe::GainTarget;
#[doc(hidden)]
pub use repair::verify_candidate as verify_transactional_candidate;
#[doc(hidden)]
pub use repair::verify_consolidation as verify_consolidation_cycle;
#[doc(hidden)]
pub use benchmark::run as benchmark_publication;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    rows: Vec<Row>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    prefix: String,
    certificate: MicroProofCertificate,
}
#[derive(Clone)]
struct Witness {
    position: Position,
    valid: Vec<bool>,
    example: Arc<MicroExample>,
    successors: std::sync::OnceLock<Vec<(f64, Vec<f64>)>>,
}
#[derive(Clone, Serialize, Deserialize)]
struct Score {
    raw: Vec<bool>,
    coupled: Vec<bool>,
    mass: f64,
    value_mse: f64,
    #[serde(skip)]
    priors: Arc<Vec<Vec<f64>>>,
    // The measured ordering is reused by margin repair. Empty for scores
    // reconstructed from older serialized diagnostics; never persisted.
    #[serde(skip)]
    coupled_logits: Arc<Vec<read_cache::CoupledLogits>>,
    #[serde(skip)]
    errors: Arc<Vec<f64>>,
}
#[derive(Clone, Default, Serialize, Deserialize)]
struct State {
    #[serde(default)]
    publication_transfer: bool,
    #[serde(default)]
    learned_choices: serde_json::Value,
    #[serde(default)]
    transfer: serde_json::Value,
    #[serde(default)]
    learning_loop_v3: bool,
    #[serde(default)]
    feedback: Vec<repair::Feedback>,
    #[serde(default)]
    feedback_cursor: usize,
    #[serde(default)]
    repair: serde_json::Value,
    #[serde(default)]
    validation_manifest: String,
    #[serde(default)]
    branch_attempts: Vec<serde_json::Value>,
    #[serde(default)]
    accepted_branch: String,
    manifest: String,
    accepted_path: PathBuf,
    accepted_identity: String,
    accepted_version: u64,
    #[serde(default)]
    last_decision: String,
    #[serde(default)]
    accepted_fraction: Option<f64>,
    checks: usize,
    rejected: usize,
    #[serde(default)]
    reasons: Vec<String>,
}
pub(super) struct Guard {
    learned_choices: Option<learned_choices::Registry>,
    publication_fresh: Option<policy_transfer::FreshLimit>,
    publication_cost: Option<Arc<transaction_bench::Cost>>,
    fast_interpolation: bool,
    dynamic_interpolation_prefilter: bool,
    separate_step_scales: bool,
    repair_interpolations: bool,
    cached_margin_reads: bool,
    adaptive_margin: bool,
    parallel: Option<cpu::Ordered>,
    reads: read_cache::Reads,
    validation: Option<Box<Guard>>,
    rows: Vec<Arc<Witness>>,
    state: State,
    score: Score,
    beta: f64,
    accepted: Arc<Snapshot>,
    out: PathBuf,
    last: Instant,
    retired: Vec<PathBuf>,
}
pub(super) fn save_snapshot(s: &Snapshot, path: &Path) -> Result<()> {
    if let Some(a) = &s.artifact {
        durable::write(path, a.as_ref())
    } else {
        let a: MicroArtifact = serde_json::from_slice(&fs::read(&s.path)?)?;
        if a.identity() != s.identity {
            return Err(invalid("snapshot source identity mismatch"));
        }
        durable::write(path, &a)
    }
}
pub(super) fn load_snapshot(
    path: &Path,
    identity: &str,
    version: u64,
    base: &MicroModel,
) -> Result<Arc<Snapshot>> {
    let artifact: MicroArtifact = serde_json::from_slice(&fs::read(path)?)?;
    if artifact.identity() != identity
        || serde_json::to_value(&artifact.sequence_memory)?
            != serde_json::to_value(base.sequence_memory().map(|b| &b.spec))?
    {
        return Err(invalid("frozen model/bank identity mismatch"));
    }
    let mut model = MicroModel::from_parameters(artifact.parameters.clone()).map_err(invalid)?;
    if model.schema() != artifact.schema || model.feature_schema() != artifact.feature_schema {
        return Err(invalid("frozen model schema mismatch"));
    }
    if let Some(bank) = base.sequence_memory() {
        model = model.with_sequence_memory_owned(bank.clone());
    }
    Ok(Arc::new(Snapshot {
        artifact: Some(Arc::new(artifact)),
        model: Arc::new(model),
        version,
        identity: identity.into(),
        path: path.into(),
    }))
}
impl Guard {
    pub fn open(
        path: &Path,
        out: &Path,
        beta: f64,
        initial: Arc<Snapshot>,
        restored: &serde_json::Value,
    ) -> Result<Self> {
        let bytes = fs::read(path)?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        if manifest.schema != "paisho-gen5-publication-guard-v1"
            || !(16..=512).contains(&manifest.rows.len())
        {
            return Err(invalid("invalid publication guard manifest"));
        }
        let hash = sha256(&bytes);
        let mut state: State = if restored.is_null() {
            State::default()
        } else {
            serde_json::from_value(restored.clone())?
        };
        if !state.manifest.is_empty() && state.manifest != hash {
            return Err(invalid("publication guard changed on resume"));
        }
        let accepted = if state.accepted_identity.is_empty() {
            initial
        } else {
            load_snapshot(
                &state.accepted_path,
                &state.accepted_identity,
                state.accepted_version,
                &initial.model,
            )?
        };
        let mut rows = vec![];
        let mut seen = std::collections::BTreeSet::new();
        for row in manifest.rows {
            if !seen.insert(sha256(row.prefix.as_bytes())) {
                return Err(invalid("duplicate publication witness"));
            }
            let record: GameRecord = row.prefix.parse()?;
            if record.rules() != RULES {
                return Err(invalid("publication witness rules mismatch"));
            }
            let position = record.replay()?;
            row.certificate.verify(&position).map_err(invalid)?;
            if position.outcome() != GameOutcome::Ongoing {
                return Err(invalid("terminal publication witness"));
            }
            let sign = if position.to_move() == Player::Host {
                1
            } else {
                -1
            };
            let value = (row.certificate.outcome * sign) as f64;
            let legal = paisho_core::legal_actions(&position);
            let valid: Vec<_> = legal
                .iter()
                .map(|a| {
                    row.certificate
                        .children
                        .iter()
                        .any(|(s, c)| s == &a.to_string() && c.outcome == row.certificate.outcome)
                })
                .collect();
            let n = valid.iter().filter(|v| **v).count();
            if n == 0 {
                return Err(invalid("publication proof lacks action"));
            }
            let example = Arc::new(MicroExample { action_values: vec![], 
                state: accepted.model.state_features(&position),
                actions: legal
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect(),
                policy: valid
                    .iter()
                    .map(|v| if *v { 1. / n as f64 } else { 0. })
                    .collect(),
                value,
                policy_weight: if value == 1. { 1. } else { 0. },
                value_weight: 0.,
                sequence_source: 0,
                policy_support: false,
            });
            rows.push(Arc::new(Witness {
                successors: Default::default(),
                position,
                valid,
                example,
            }));
        }
        state.manifest = hash;
        fs::create_dir_all(out.join("accepted"))?;
        if state.accepted_identity.is_empty() {
            state.accepted_path = out
                .join("accepted")
                .join(format!("model-{}.json", accepted.identity));
            save_snapshot(&accepted, &state.accepted_path)?;
            state.accepted_identity = accepted.identity.clone();
            state.accepted_version = accepted.version;
        }
        let score = measure(&rows, &accepted.model, beta)?;
        Ok(Self {
            learned_choices: None,
            publication_fresh: None,
            publication_cost: None,
            fast_interpolation: true,
            // Diagnostic candidate; fixed ABBA did not establish a cost gain.
            dynamic_interpolation_prefilter: false,
            separate_step_scales: true,
            repair_interpolations: true,
            cached_margin_reads: true,
            adaptive_margin: true,
            parallel: None,
            reads: Default::default(),
            validation: None,
            rows,
            state,
            score,
            beta,
            accepted,
            out: out.into(),
            last: paisho_platform::training_time::now(),
            retired: vec![],
        })
    }
    /// Read-only native finite V3 criterion; no interpolation, repair or publication.
    #[doc(hidden)]
    pub fn diagnostic_finite_v3(&self, model: &MicroModel) -> Result<serde_json::Value> {
        if !self.state.learning_loop_v3 {return Err(invalid("finite V3 diagnostic requires V3"));}
        let (scores, reasons) = self.checked_v3(model)?;
        Ok(serde_json::json!({"accepted":reasons.is_empty(),"reasons":reasons,"scores":scores}))
    }
    pub fn accepted(&self) -> Arc<Snapshot> {
        self.accepted.clone()
    }
    pub fn consider(
        &mut self,
        candidate: Arc<Snapshot>,
        force: bool,
    ) -> Result<Option<Vec<Arc<MicroExample>>>> {
        if self.state.learning_loop_v3 {
            return self.consider_v3(candidate, force);
        }
        if self.validation.is_some() {
            return self.consider_branches(candidate, force);
        }
        if candidate.identity == self.accepted.identity
            || (!force && paisho_platform::training_time::elapsed(self.last).as_secs_f64() < 30.)
        {
            return Ok(None);
        }
        self.last = paisho_platform::training_time::now();
        let score = self.evaluate(&candidate.model)?;
        let lost: Vec<_> = (0..self.rows.len())
            .filter(|&i| {
                (self.score.raw[i] && !score.raw[i]) || (self.score.coupled[i] && !score.coupled[i])
            })
            .collect();
        let mut reasons = vec![];
        if !lost.is_empty() {
            reasons.push(format!("{} known proof choices lost", lost.len()));
        }
        // Avoid treating rounding as regression; no score tradeoff can hide a lost proof choice.
        if score.mass + 1e-12 < self.score.mass {
            reasons.push("mean verified policy mass decreased".into());
        }
        if score.value_mse > self.score.value_mse + 1e-12 {
            reasons.push("balanced proof value error increased".into());
        }
        self.state.checks += 1;
        self.state.reasons = reasons;
        if self.state.reasons.is_empty() {
            self.publish(candidate, score, 1.)?;
            self.state.last_decision = "accepted".into();
            Ok(Some(vec![]))
        } else {
            self.state.rejected += 1;
            self.state.last_decision = "rejected".into();
            // A smaller, explicitly recorded actor step can retain the old choices.
            // Every proposed interpolation is measured; parameter interpolation
            // does not imply interpolation of this nonlinear network's predictions.
            if score.mass + 1e-12 >= self.score.mass
                && score.value_mse <= self.score.value_mse + 1e-12
            {
                if let Some((projected, score, fraction)) = self.project(&candidate)? {
                    self.publish(projected, score, fraction)?;
                    self.state.last_decision = "projected".into();
                }
            }
            // Focus is a bounded slice of the existing policy-only recall quota.
            let mut focus = lost;
            for i in 0..self.rows.len() {
                if self.rows[i].example.value == 1. && !focus.contains(&i) {
                    focus.push(i);
                }
            }
            Ok(Some(
                focus
                    .into_iter()
                    .filter(|&i| self.rows[i].example.value == 1.)
                    .take(32)
                    .map(|i| self.rows[i].example.clone())
                    .collect(),
            ))
        }
    }
    fn publish(&mut self, candidate: Arc<Snapshot>, score: Score, fraction: f64) -> Result<()> {
        let path = self
            .out
            .join("accepted")
            .join(format!("model-{}.json", candidate.identity));
        save_snapshot(&candidate, &path)?;
        self.retired.push(self.state.accepted_path.clone());
        self.state.accepted_path = path.clone();
        self.state.accepted_identity = candidate.identity.clone();
        self.state.accepted_version = candidate.version;
        self.state.accepted_fraction = Some(fraction);
        self.accepted = Arc::new(Snapshot {path, artifact:candidate.artifact.clone(),
            model:candidate.model.clone(), identity:candidate.identity.clone(),version:candidate.version});
        self.score = score;
        Ok(())
    }
    fn project(&self, candidate: &Snapshot) -> Result<Option<(Arc<Snapshot>, Score, f64)>> {
        for fraction in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625] {
            let weights = self
                .accepted
                .model
                .parameters()
                .iter()
                .zip(candidate.model.parameters())
                .map(|(old, new)| old + fraction * (new - old))
                .collect();
            let mut model = MicroModel::from_parameters(weights).map_err(invalid)?;
            if let Some(bank) = candidate.model.sequence_memory() {
                model = model.with_sequence_memory_owned(bank.clone());
            }
            let score = self.evaluate(&model)?;
            let kept = (0..self.rows.len()).all(|i| {
                (!self.score.raw[i] || score.raw[i]) && (!self.score.coupled[i] || score.coupled[i])
            });
            if kept
                && score.mass + 1e-12 >= self.score.mass
                && score.value_mse <= self.score.value_mse + 1e-12
            {
                if let Some(v) = &self.validation {
                    let external = v.evaluate(&model)?;
                    let kept = (0..v.rows.len()).all(|i| {
                        (!v.score.raw[i] || external.raw[i])
                            && (!v.score.coupled[i] || external.coupled[i])
                    });
                    if !kept
                        || external.mass + 1e-12 < v.score.mass
                        || external.value_mse > v.score.value_mse + 1e-12
                    {
                        continue;
                    }
                }
                let artifact = Arc::new(MicroArtifact::new(
                    &model,
                    candidate
                        .artifact
                        .as_ref()
                        .ok_or_else(|| invalid("projection requires learner artifact"))?
                        .updates,
                    serde_json::json!({"kind":"gen5-guarded-interpolation","learner":candidate.identity,"accepted_from":self.accepted.identity,"fraction":fraction,"version":candidate.version,"no_extra_optimizer_step":true}),
                ));
                let identity = artifact.identity();
                let path = self
                    .out
                    .join("accepted")
                    .join(format!("model-{identity}.json"));
                return Ok(Some((
                    Arc::new(Snapshot {
                        artifact: Some(artifact),
                        model: Arc::new(model),
                        version: candidate.version,
                        identity,
                        path,
                    }),
                    score,
                    fraction,
                )));
            }
        }
        Ok(None)
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::to_value(&self.state).unwrap()
    }
    pub fn committed(&mut self) -> Result<()> {
        for p in self.retired.drain(..) {
            if p.starts_with(&self.out) {
                fs::remove_file(p)?;
            }
        }
        Ok(())
    }
}
fn measure(rows: &[Arc<Witness>], model: &MicroModel, beta: f64) -> Result<Score> {
    measure_cached(rows, model, beta, true)
}
fn successor_inputs(r: &Witness, model: &MicroModel) -> Result<Vec<(f64, Vec<f64>)>> {
    paisho_core::legal_actions(&r.position)
        .into_iter()
        .map(|a| {
            let mut next = r.position.clone();
            next.apply(a)?;
            Ok(match next.outcome() {
                GameOutcome::Win(w) => (if w == r.position.to_move() { 1. } else { -1. }, vec![]),
                GameOutcome::Draw => (0., vec![]),
                _ => (
                    if next.to_move() == r.position.to_move() {
                        1.
                    } else {
                        -1.
                    },
                    model.state_features(&next),
                ),
            })
        })
        .collect()
}
#[derive(Clone)]
struct RowScore { raw: bool, coupled: bool, mass: f64, error: f64, class: usize, priors: Vec<f64>, coupled_logits: read_cache::CoupledLogits }
fn measure_one(r: &Witness, model: &MicroModel, beta: f64, cached: bool, values: Option<Arc<read_cache::ValueRow>>, lazy:bool, reuse_root_embedding:bool) -> Result<RowScore> {
    let class=(r.example.value as i8+1) as usize;
    if cached && r.example.value!=1. {
        let error=(model.embed(&r.example.state).value-r.example.value).powi(2);
        return Ok(RowScore {raw:false,coupled:false,mass:0.,error,class,priors:vec![],coupled_logits:vec![].into()});
    }
    let (value,p)=if reuse_root_embedding {
        model.policy_value_priors(&r.example.state,&r.example.actions,0).map_err(invalid)?
    } else {
        let e=model.embed(&r.example.state);
        let base=micro_softmax(&MicroModel::logits(&e,&r.example.actions)).map_err(invalid)?;
        (e.value,model.memory_priors(&r.example.state,&r.example.actions,&base,0).map_err(invalid)?)
    };
    let error=(value-r.example.value).powi(2);
    let best=|p:&[f64]| (0..p.len()).max_by(|&a,&b|p[a].total_cmp(&p[b]).then_with(||b.cmp(&a))).unwrap();
    let raw=r.example.value==1. && r.valid[best(&p)];
    let mass=if r.example.value==1. { p.iter().zip(&r.valid).filter(|(_,v)|**v).map(|(p,_)|*p).sum::<f64>() } else {0.};
    let mut logits:Vec<f64>=p.iter().map(|p|p.max(1e-300).ln()).collect();
    if let Some(values)=values {
        let coupled_logits=read_cache::CoupledLogits::pending(logits,beta,values,lazy);
        let selected=coupled_logits.winner(&r.valid);
        return Ok(RowScore {raw,coupled:r.example.value==1.&&r.valid[selected],mass,error,class,priors:p,coupled_logits});
    }
    let rebuilt;
    let inputs=if cached {
        if r.successors.get().is_none() { let _=r.successors.set(successor_inputs(r,model)?); }
        r.successors.get().unwrap()
    } else { rebuilt=successor_inputs(r,model)?; &rebuilt };
    for (l,(sign,state)) in logits.iter_mut().zip(inputs) {
        let q=if state.is_empty() {*sign} else {let v=model.value(state);if *sign==1. {v}else{-v}};
        *l+=beta*q;
    }
    Ok(RowScore {raw,coupled:r.example.value==1. && r.valid[best(&logits)],mass,error,class,priors:p,coupled_logits:logits.into()})
}
fn measure_ordered(rows:&[Arc<Witness>],model:&MicroModel,beta:f64,cached:bool,parallel:Option<&cpu::Ordered>,values:Option<Arc<read_cache::ValueTable>>,lazy:bool,reuse_root_embedding:bool)->Result<Score> {
    let indexed:Vec<_>=rows.iter().cloned().enumerate().collect();
    let model=model.clone();
    let f=move |(i,r): &(usize,Arc<Witness>)|measure_one(r,&model,beta,cached,values.as_ref().and_then(|v|v[*i].clone()),lazy,reuse_root_embedding).map_err(|e|e.to_string());
    let parts:Vec<_>=match parallel {Some(p)=>p.map_owned(indexed,|(_,r)|r.example.actions.len(),f),None=>indexed.iter().map(f).collect()};
    let mut raw=vec![];let mut coupled=vec![];let mut mass=0.;let mut errors=[0.;3];let mut counts=[0usize;3];let mut priors=vec![];let mut row_errors=vec![];let mut coupled_logits=vec![];
    for (r,part) in rows.iter().zip(parts) {
        let part=part.map_err(invalid)?;
        raw.push(part.raw);coupled.push(part.coupled);
        if r.example.value==1. {mass+=part.mass;}
        errors[part.class]+=part.error;counts[part.class]+=1;
        priors.push(part.priors);row_errors.push(part.error);coupled_logits.push(part.coupled_logits);
    }
    let value_mse=errors.iter().zip(counts).filter(|(_,n)|*n>0).map(|(e,n)|e/n as f64).sum::<f64>() / counts.iter().filter(|n|**n>0).count() as f64;
    Ok(Score {raw,coupled,mass:mass/rows.iter().filter(|r|r.example.value==1.).count().max(1) as f64,value_mse,priors:Arc::new(priors),coupled_logits:Arc::new(coupled_logits),errors:Arc::new(row_errors)})
}
fn measure_cached(rows:&[Arc<Witness>],model:&MicroModel,beta:f64,cached:bool)->Result<Score> {
    measure_ordered(rows,model,beta,cached,None,None,false,false)
}
impl Guard {
    pub fn enable_parallel(&mut self,pools:&[Arc<rayon::ThreadPool>]) {
        self.parallel=Some(cpu::Ordered::new(pools));
        if let Some(v)=&mut self.validation {v.enable_parallel(pools);}
    }
    /// Diagnostic ABBA switch; resets both caches, leaves all guard criteria,
    /// anchors and exact successor inputs unchanged. Set before each leg.
    pub(super) fn diagnostic_lazy_successor_values(&mut self, lazy:bool) {
        self.reads=read_cache::Reads::new(lazy);
        if let Some(v)=&mut self.validation {v.diagnostic_lazy_successor_values(lazy);}
    }
    /// Isolate the duplicate-embedding optimization with empty score caches.
    pub(super) fn diagnostic_reuse_root_embedding(&mut self, enabled:bool) {
        self.reads=read_cache::Reads::new(true);
        self.reads.reuse_root_embedding=enabled;
        if let Some(v)=&mut self.validation {v.diagnostic_reuse_root_embedding(enabled);}
    }
    pub(super) fn diagnostic_successor_reads(&self)->serde_json::Value {
        serde_json::json!({"primary":{"value_forwards":self.reads.value_forwards(),"loaded_cells_including_terminal":self.reads.loaded_value_cells()},
            "validation":self.validation.as_deref().map(|v|v.diagnostic_successor_reads())})
    }
    fn evaluate(&self,model:&MicroModel)->Result<Score> {
        self.reads.evaluate(&self.rows,model,self.beta,self.parallel.as_ref())
    }
}

mod verify;
pub use verify::{candidate as verify_candidate, run as verify_guard};
