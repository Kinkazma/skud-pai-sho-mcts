//! Project minibatch updates against recalled proofs, then verify finite changes periodically.
use super::*;
mod projection;
mod verify;
mod benchmark;
mod profile;
mod gradient_flow;
mod frontier;
mod validation_value;
mod fresh_barrier;
mod fresh_interior;
mod choice_merit;
mod capture;
mod replay_capture;
#[doc(hidden)]
pub use replay_capture::run as replay_captured_consolidation;
pub use replay_capture::replay_capture_publication;
#[doc(hidden)]
pub use replay_capture::measure_block_retention;
#[doc(hidden)]
pub use replay_capture::measure_reader_context_drift;
#[doc(hidden)]
pub use replay_capture::measure_final_block_retention;
#[doc(hidden)]
pub use frontier::run as verify_frontier;
#[doc(hidden)]
pub use frontier::run_kl_probe as verify_kl_frontier;
#[doc(hidden)]
pub use frontier::run_gain_probe as verify_fresh_gain_frontier;
pub(super) fn diagnostic_affine(g:&[f64], refs:&[Vec<f64>], rhs:&[f64])->Option<Vec<f64>> {
    projection::affine(g,refs,rhs)
}
#[doc(hidden)]
pub use frontier::run_validation_probe as verify_validation_value_frontier;
#[cfg(test)]
mod tests;
pub(super) use benchmark::run as benchmark_parallel;
pub(super) use benchmark::transaction as benchmark_transaction;
pub(super) use verify::boundary as verify_boundary;
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn weights(base: &MicroModel, w: Vec<f64>) -> Result<MicroModel> {
    let mut m = MicroModel::from_parameters(w).map_err(invalid)?;
    if let Some(bank) = base.sequence_memory() {
        m = m.with_sequence_memory_owned(bank.clone());
    }
    Ok(m)
}
fn prior(m: &MicroModel, e: &MicroExample) -> Result<Vec<f64>> {
    let base =
        micro_softmax(&MicroModel::logits(&m.embed(&e.state), &e.actions)).map_err(invalid)?;
    m.memory_priors(&e.state, &e.actions, &base, e.sequence_source)
        .map_err(invalid)
}
fn best(p: &[f64]) -> usize {
    (0..p.len())
        .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .unwrap()
}
fn losses(m: &MicroModel, rows: &[Arc<MicroExample>], parallel: Option<&cpu::Ordered>) -> Result<([f64; 4], Vec<bool>)> {
    let m=m.clone();
    let score=move |e:&Arc<MicroExample>| -> std::result::Result<_,String> {
        let k=(e.value as i8+1) as usize+1;
        let error=0.5*(m.value(&e.state)-e.value).powi(2);
        let policy=if e.value==1. && !e.policy.is_empty() {
            let p=prior(&m,e).map_err(|e|e.to_string())?;
            let mass:f64=p.iter().zip(&e.policy).filter(|(_,t)|**t>0.).map(|(p,_)|p).sum();
            Some((mass.max(1e-300).ln(),e.policy[best(&p)]>0.))
        } else {None};
        Ok((k,error,policy))
    };
    let values:Vec<_>=match parallel {Some(p)=>p.map_owned(rows.to_vec(),|e|e.actions.len(),score),None=>rows.iter().map(score).collect()};
    let mut loss=[0.;4];let mut ns=[0usize;4];let mut choices=vec![];
    for value in values {
        let (k,error,policy)=value.map_err(invalid)?;ns[k]+=1;loss[k]+=error;
        if let Some((ln,good))=policy {loss[0]-=ln;ns[0]+=1;choices.push(good);}
    }
    for k in 0..4 {loss[k]/=ns[k].max(1) as f64;}
    Ok((loss,choices))
}
fn reference_one(m: &MicroModel, r: &Arc<MicroExample>) -> Result<(usize, Vec<f64>, Option<Vec<f64>>)> {
        let mut ex = r.as_ref().clone();
        ex.policy_weight = 0.;
        ex.value_weight = 1.;
        let k = (ex.value as i8 + 1) as usize + 1;
        let mut value_only=ex.clone();
        if value_only.action_values.is_empty() {value_only.actions.clear();value_only.policy.clear();}
        let (_, g) = m.loss_gradient(&value_only).map_err(invalid)?;
        let value_gradient = g;
        let mut policy_gradient = None;
        if ex.value == 1. && !ex.policy.is_empty() {
            let g=m.policy_mass_gradient(&ex).map_err(invalid)?;
            policy_gradient = Some(g);
        }
    Ok((k, value_gradient, policy_gradient))
}
fn references(m: &MicroModel, rows: &[Arc<MicroExample>], parallel: Option<&cpu::Ordered>) -> Result<Vec<Vec<f64>>> {
    let mut gs = vec![vec![0.; m.parameters().len()]; 4];
    let mut ns = [0usize; 4];
    let m=m.clone();
    let f = move |r: &Arc<MicroExample>| reference_one(&m,r).map_err(|e|e.to_string());
    // Bound temporary gradients independently of the size of a proof catalogue.
    for chunk in rows.chunks(32) {
        let parts: Vec<_> = match parallel {Some(p)=>p.map_owned(chunk.to_vec(),|e|e.actions.len(),f.clone()),None=>chunk.iter().map(&f).collect()};
        for part in parts {
            let (k,g,p)=part.map_err(invalid)?;
            ns[k]+=1;
            for (a,b) in gs[k].iter_mut().zip(g) {*a+=b;}
            if let Some(g)=p {ns[0]+=1;for (a,b) in gs[0].iter_mut().zip(g) {*a+=b;}}
        }
    }
    for k in 0..4 {for g in &mut gs[k] {*g/=ns[k].max(1) as f64;}}
    Ok(gs)
}
// A mean can improve while its best action crosses a decision boundary.
// Repair the worst lost verified choice as one additional halfspace, keeping
// the active-set solve bounded to five constraints and checking all choices after it.
fn lost_choice_constraint(
    m: &MicroModel,
    rows: &[Arc<MicroExample>],
    old: &[bool],
) -> Result<Option<(Vec<f64>, f64)>> {
    let mut worst = None;
    let mut winner = 0;
    for r in rows
        .iter()
        .filter(|r| r.value == 1. && !r.policy.is_empty())
    {
        let protected = old[winner];
        winner += 1;
        if !protected {
            continue;
        }
        let p = prior(m, r)?;
        let bad = best(&p);
        if r.policy[bad] > 0. {
            continue;
        }
        let good = (0..p.len())
            .filter(|&i| r.policy[i] > 0.)
            .max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
            .unwrap();
        let gap = p[bad].max(1e-300).ln() - p[good].max(1e-300).ln() + 1e-6;
        if worst.as_ref().map_or(true, |(_, _, _, g)| gap > *g) {
            worst = Some((r.clone(), good, bad, gap));
        }
    }
    let Some((r, good, bad, gap)) = worst else {
        return Ok(None);
    };
    let mut ex = r.as_ref().clone();
    ex.value_weight = 0.;
    ex.policy_weight = 1.;
    ex.policy.fill(0.);
    ex.policy[good] = 1.;
    let (_, mut g) = m.loss_gradient(&ex).map_err(invalid)?;
    ex.policy[good] = 0.;
    ex.policy[bad] = 1.;
    let (_, b) = m.loss_gradient(&ex).map_err(invalid)?;
    for (g, b) in g.iter_mut().zip(b) {
        *g -= b;
    }
    Ok(Some((g, gap)))
}
fn fresh_loss(m: &MicroModel, rows: &[Arc<MicroExample>], parallel: Option<&cpu::Ordered>) -> Result<f64> {
    fresh_loss_mode(m, rows, parallel, false)
}
fn fresh_loss_mode(m: &MicroModel, rows: &[Arc<MicroExample>], parallel: Option<&cpu::Ordered>, loop_v3: bool) -> Result<f64> {
    let m=m.clone();let count=rows.len().max(1);
    let f=move |e: &Arc<MicroExample>| {
        let loss=if loop_v3 {m.loss_loop_v3(e)} else {m.loss(e)};
        loss.map(|l|l.total(e.policy_weight)/count as f64)
    };
    let parts:Vec<_>=match parallel {Some(p)=>p.map_owned(rows.to_vec(),|e|e.actions.len(),f),None=>rows.iter().map(f).collect()};
    let mut s=0.;
    for p in parts {s+=p.map_err(invalid)?;}
    Ok(s)
}
const FINITE_LOSS_TOLERANCE: f64 = 1e-9;
fn repair_rhs(current: &[f64; 4], limits: &[f64; 4]) -> Vec<f64> {
    // A Newton correction towards the exact outer boundary can remain just
    // outside a convex squared loss forever (or by one last floating-point bit).
    // Aim inside the SAME accepted interval; zero reference losses still have
    // a positive target, unlike the former incorrect `limit - tolerance` RHS.
    current.iter().zip(limits).map(|(a, b)| a - (b + 0.5 * FINITE_LOSS_TOLERANCE)).collect()
}
fn violation_merit(current: &[f64; 4], limits: &[f64; 4], choices: &[bool], old: &[bool], loop_v3: bool) -> f64 {
    // Rank trial steps by the actual outer acceptance boundary, not the stricter
    // correction target: every finitely admissible trial has zero violation.
    let losses = current.iter().zip(limits).enumerate().filter(|(i, _)| !loop_v3 || *i != 0).map(|(_, (value, limit))| {
        let excess = value - (limit + FINITE_LOSS_TOLERANCE);
        (excess.max(0.) / (limit.abs() + FINITE_LOSS_TOLERANCE)).powi(2)
    }).sum::<f64>();
    losses + old.iter().zip(choices).filter(|(before, after)| **before && !**after).count() as f64
}
pub(super) struct Protection {
    gradient_buffers: Arc<std::sync::Mutex<Vec<Vec<f64>>>>,
    profile:profile::Profile,
    parallel: Option<cpu::Ordered>,
    loop_v3: bool,
    rows: Vec<Arc<MicroExample>>,
    gradients: Vec<Vec<f64>>,
    gram: Vec<Vec<f64>>,
    value_gram: Vec<Vec<f64>>,
    policy_gradient_constraint: bool,
    validation_value: Option<validation_value::ValidationValue>,
    anchor: MicroModel,
    anchor_losses: [f64; 4],
    anchor_choices: Vec<bool>,
    fresh: std::collections::VecDeque<Arc<MicroExample>>,
    updates: usize,
    retained_norm_sum: f64,
    zero_steps: usize,
    checks: usize,
    accepted: usize,
    reference_evaluations: usize,
    last: serde_json::Value,
    seconds: f64,
    saved: serde_json::Value,
    retired: Vec<PathBuf>,
}
impl Protection {
    pub fn new(model: &MicroModel, rows: Vec<Arc<MicroExample>>) -> Result<Self> {
        if rows.is_empty() {
            return Err(invalid("empty protection references"));
        }
        let gradients = references(model, &rows, None)?;
        let gram = projection::gram(&gradients);
        let value_gram = gram.iter().skip(1).map(|row| row[1..].to_vec()).collect();
        let (anchor_losses, anchor_choices) = losses(model, &rows, None)?;
        Ok(Self {
            gradient_buffers: Arc::new(std::sync::Mutex::new(Vec::new())),
            profile:profile::Profile::default(),
            parallel: None,
            loop_v3: false,
            rows,
            gram,
            value_gram,
            policy_gradient_constraint: true,
            validation_value: None,
            gradients,
            anchor: model.clone(),
            anchor_losses,
            anchor_choices,
            fresh: Default::default(),
            updates: 0,
            retained_norm_sum: 0.,
            zero_steps: 0,
            checks: 0,
            accepted: 0,
            reference_evaluations: 0,
            last: serde_json::Value::Null,
            seconds: 0.,
            saved: serde_json::Value::Null,
            retired: vec![],
        })
    }
    pub fn enable_parallel(&mut self, pools: &[Arc<rayon::ThreadPool>]) {
        self.parallel=Some(cpu::Ordered::new(pools));
    }
    /// V3 keeps individual decisions and value classes as finite constraints;
    /// policy confidence remains a local gradient objective, not a hard ratchet.
    pub fn enable_loop_v3(&mut self) { self.loop_v3 = true; }
    /// Bounded diagnostic ablation only. The runtime default remains enabled;
    /// the option is not persisted or inferred when restoring a campaign.
    pub fn diagnostic_set_policy_gradient_constraint(&mut self, active: bool) {
        self.policy_gradient_constraint = active;
    }
    /// Match the secondary publication panel with one balanced value constraint.
    /// Checkpoints bind the exact ordered value examples; gradients are rebuilt.
    pub fn enable_validation_value(&mut self, rows: Vec<Arc<MicroExample>>) -> Result<()> {
        if !self.loop_v3 || self.validation_value.is_some() {
            return Err(invalid("validation-value requires V3 and can be enabled only once"));
        }
        let panel = validation_value::ValidationValue::new(rows, &self.anchor, self.parallel.as_ref())?;
        let gradient = panel.gradient(&self.anchor, self.parallel.as_ref())?;
        self.reference_evaluations += panel.rows.len();
        self.gradients.push(gradient);
        self.gram = projection::gram(&self.gradients);
        self.value_gram = self.gram.iter().skip(1).map(|row| row[1..].to_vec()).collect();
        self.validation_value = Some(panel);
        Ok(())
    }
    /// Fixed-panel diagnostic alias; the mathematical constraint is identical.
    pub fn diagnostic_enable_validation_value(&mut self, rows: Vec<Arc<MicroExample>>) -> Result<()> {
        self.enable_validation_value(rows)
    }
    fn validation_checkpoint_identity(&self) -> Result<serde_json::Value> {
        self.validation_value.as_ref().map(|panel| {
            let rows = panel.rows.iter().map(|e| resume_example::ResumeExample::from(e.as_ref())).collect::<Vec<_>>();
            Ok(serde_json::json!({"schema":"balanced-value-panel-v1","rows":rows.len(),"references":sha256(&serde_json::to_vec(&rows)?)}))
        }).unwrap_or(Ok(serde_json::Value::Null))
    }
    fn diagnostic_validation_loss(&self, model: &MicroModel) -> Result<Option<f64>> {
        self.validation_value.as_ref().map(|p|p.loss(model,self.parallel.as_ref())).transpose()
    }
    fn diagnostic_validation_accepts(&self, loss: Option<f64>) -> bool {
        self.validation_value.as_ref().map_or(true,|p|loss.is_some_and(|v|p.accepts(v)))
    }
    fn diagnostic_validation_merit(&self, loss: Option<f64>) -> f64 {
        self.validation_value.as_ref().map_or(0.,|p|loss.map_or(f64::INFINITY,|v|p.merit(v)))
    }
    /// Adopt an independently accepted model without discarding replay tickets,
    /// recent examples or recovery counters. Call only after finite validation.
    pub fn adopt_accepted(&mut self, model: &MicroModel) -> Result<()> {
        let same_bank = match (model.sequence_memory(), self.anchor.sequence_memory()) {
            (None, None) => true,
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        };
        if model.shares_storage_with(&self.anchor) || (same_bank
            && model.parameters().len() == self.anchor.parameters().len()
            && model.parameters().iter().zip(self.anchor.parameters()).all(|(a,b)| a.to_bits() == b.to_bits())) {
            return Ok(());
        }
        let (anchor_losses, anchor_choices) = losses(model, &self.rows, self.parallel.as_ref())?;
        self.rebuild_anchor(model, anchor_losses, anchor_choices)
    }
    fn rebuild_anchor(&mut self, model: &MicroModel, anchor_losses: [f64; 4], anchor_choices: Vec<bool>) -> Result<()> {
        let mut gradients = references(model, &self.rows, self.parallel.as_ref())?;
        // Compute everything before replacing the accepted anchor or its caches.
        let validation_loss = self.diagnostic_validation_loss(model)?;
        if let Some(panel) = &self.validation_value {
            gradients.push(panel.gradient(model,self.parallel.as_ref())?);
            self.reference_evaluations += panel.rows.len();
        }
        if let Some(panel) = &mut self.validation_value {
            panel.anchor_loss = validation_loss.ok_or_else(||invalid("missing diagnostic validation anchor loss"))?;
        }
        self.gram = projection::gram(&gradients);
        self.value_gram = self.gram.iter().skip(1).map(|row| row[1..].to_vec()).collect();
        self.gradients = gradients;
        self.anchor = model.clone();
        self.anchor_losses = anchor_losses;
        self.anchor_choices = anchor_choices;
        self.reference_evaluations += self.rows.len();
        Ok(())
    }
    fn reference_hash(&self) -> Result<String> {
        Ok(sha256(&serde_json::to_vec(
            &self
                .rows
                .iter()
                .map(|e| resume_example::ResumeExample::from(e.as_ref()))
                .collect::<Vec<_>>(),
        )?))
    }
    pub fn restore(&mut self, v: &serde_json::Value) -> Result<()> {
        let state = &v["checkpoint"];
        if state.is_null() {
            return Ok(());
        }
        let bytes = fs::read(
            state["path"]
                .as_str()
                .ok_or_else(|| invalid("protection checkpoint path"))?,
        )?;
        if sha256(&bytes) != state["sha256"].as_str().unwrap_or("") {
            return Err(invalid("protection checkpoint changed"));
        }
        let s: serde_json::Value = serde_json::from_slice(&bytes)?;
        if s["references"] != self.reference_hash()? {
            return Err(invalid("protection references changed on resume"));
        }
        if s["anchor"].as_array().map(Vec::len)!=Some(self.anchor.parameters().len()) {
            return Err(invalid("protection anchor requires an explicit architecture migration"));
        }
        let configured_validation = self.validation_checkpoint_identity()?;
        // Old checkpoints precede this additional V3 constraint. Adding it is a
        // deterministic protocol migration; dropping/changing a saved panel is not.
        if !s["validation_value"].is_null() && s["validation_value"] != configured_validation {
            return Err(invalid("protection validation panel changed or disabled on resume"));
        }
        let restored_anchor = weights(&self.anchor, serde_json::from_value(s["anchor"].clone())?)?;
        self.anchor = restored_anchor;
        self.gradients = references(&self.anchor, &self.rows, self.parallel.as_ref())?;
        if let Some(panel) = &mut self.validation_value {
            panel.anchor_loss = panel.loss(&self.anchor,self.parallel.as_ref())?;
            self.gradients.push(panel.gradient(&self.anchor,self.parallel.as_ref())?);
        }
        self.gram=projection::gram(&self.gradients);
        self.value_gram = self.gram.iter().skip(1).map(|row| row[1..].to_vec()).collect();
        (self.anchor_losses, self.anchor_choices) = losses(&self.anchor, &self.rows, self.parallel.as_ref())?;
        let rows: Vec<resume_example::ResumeExample> = serde_json::from_value(s["fresh"].clone())?;
        if rows.len() > 64 {
            return Err(invalid("oversized protection recovery"));
        }
        self.fresh = rows
            .into_iter()
            .map(|e| e.example_with_trusted_q(self.loop_v3))
            .collect::<Result<_>>()?;
        self.retained_norm_sum = s["retained_norm_sum"].as_f64().unwrap_or(0.);
        self.zero_steps = s["zero_steps"].as_u64().unwrap_or(0) as usize;
        self.updates = s["updates"].as_u64().unwrap_or(0) as usize;
        self.checks = s["checks"].as_u64().unwrap_or(0) as usize;
        self.accepted = s["accepted"].as_u64().unwrap_or(0) as usize;
        self.reference_evaluations = s["reference_evaluations"].as_u64().unwrap_or(0) as usize;
        self.seconds = s["seconds"].as_f64().unwrap_or(0.);
        self.last = s["last"].clone();
        self.saved = state.clone();
        Ok(())
    }
    pub fn checkpoint(&mut self, out: &Path) -> Result<()> {
        let s = serde_json::json!({"references":self.reference_hash()?,"validation_value":self.validation_checkpoint_identity()?,"anchor":self.anchor.parameters(),"fresh":self.fresh.iter().map(|e|resume_example::ResumeExample::from_with_trusted_q(e.as_ref(),self.loop_v3)).collect::<Vec<_>>(),"updates":self.updates,"retained_norm_sum":self.retained_norm_sum,"zero_steps":self.zero_steps,"checks":self.checks,"accepted":self.accepted,"reference_evaluations":self.reference_evaluations,"seconds":self.seconds,"last":self.last});
        let hash = sha256(&serde_json::to_vec(&s)?);
        let path = out.join(format!("protection-{hash}.json"));
        if !path.exists() {
            durable::write(&path, &s)?;
        }
        if let Some(old) = self.saved["path"].as_str() {
            let old = PathBuf::from(old);
            if old != path && old.starts_with(out) {
                self.retired.push(old);
            }
        }
        self.saved = serde_json::json!({"path":path,"sha256":hash});
        Ok(())
    }
    pub fn committed(&mut self) -> Result<()> {
        for p in self.retired.drain(..) {
            if p.exists() {
                fs::remove_file(p)?;
            }
        }
        Ok(())
    }
    pub fn observe(&mut self, rows: &[Arc<MicroExample>]) {
        for e in rows {
            if self.fresh.len() == 64 {
                self.fresh.pop_front();
            }
            self.fresh.push_back(e.clone());
        }
    }
    pub fn train(
        &mut self,
        model: &mut MicroModel,
        batch: &[&MicroExample],
        rate: f64,
        l2: f64,
    ) -> Result<()> {
        self.train_shared(model, &batch.iter().map(|e|Arc::new((*e).clone())).collect::<Vec<_>>(), rate, l2)
    }
    pub fn train_shared(
        &mut self,
        model: &mut MicroModel,
        batch: &[Arc<MicroExample>],
        rate: f64,
        l2: f64,
    ) -> Result<()> {
        if batch.is_empty() || !rate.is_finite() || rate <= 0. || !l2.is_finite() || l2 < 0. {
            return Err(invalid("invalid protected minibatch"));
        }
        let mut g = vec![0.; model.parameters().len()];
        self.profile.observe(&batch.iter().map(AsRef::as_ref).collect::<Vec<_>>());
        let frozen_model = model.clone();
        let buffers = self.gradient_buffers.clone();
        let loop_v3 = self.loop_v3;
        let allocations=self.profile.gradient_buffer_allocations.clone();
        let f=move |e: &Arc<MicroExample>| {
            let t=paisho_platform::training_time::now();
            // Drop the pool lock before neural computation. At most one maximum
            // minibatch (64 gradients) stays resident between learner steps.
            let buffer=buffers.lock().unwrap().pop().unwrap_or_default();
            if buffer.capacity()<frozen_model.parameters().len() {allocations.fetch_add(1,std::sync::atomic::Ordering::Relaxed);}
            let result=if loop_v3 { frozen_model.loss_gradient_loop_v3_reusing(e,buffer) }
                else { frozen_model.loss_gradient_reusing(e,buffer) }.map_err(|e|e.to_string());
            (paisho_platform::training_time::elapsed(t).as_secs_f64(),result)
        };
        // The runtime minibatch has at most 64 examples. Compute its independent
        // gradients together (at most 143 MiB for the 292k model) instead of
        // waiting for one half before admitting the other. Reduction stays ordered.
        for chunk in batch.chunks(64) {
            let t=paisho_platform::training_time::now();
            let old_reduction=self.profile.reduction_seconds;
            let profile=&mut self.profile;let buffers=&self.gradient_buffers;
            let mut failure=None;
            let mut reduce=|(),(seconds,part):(f64,std::result::Result<(MicroLoss,Vec<f64>),String>)| {
                if failure.is_some() {return;}
                profile.gradient_worker_wall_seconds+=seconds;
                let (_,part)=match part {Ok(p)=>p,Err(e)=>{failure=Some(e);return;}};
                let t=paisho_platform::training_time::now();
                // A successful loss_gradient_reusing has already checked every
                // coefficient for finiteness. Re-scanning here only rereads RAM.
                if batch.len().is_power_of_two() {
                    let scale=1. / batch.len() as f64;
                    for (g,v) in g.iter_mut().zip(&part) {*g+=v*scale;}
                } else {
                    for (g,v) in g.iter_mut().zip(&part) {*g+=v/batch.len() as f64;}
                }
                buffers.lock().unwrap().push(part);
                profile.reduction_seconds+=paisho_platform::training_time::elapsed(t).as_secs_f64();
            };
            match &self.parallel {
                Some(p)=>p.fold_owned(chunk.to_vec(),|e|e.actions.len(),f.clone(),(),|_,part|reduce((),part)),
                None=>for e in chunk {reduce((),f(e));},
            }
            let elapsed=paisho_platform::training_time::elapsed(t).as_secs_f64();
            // Caller phases remain disjoint in these counters; worker task wall
            // times overlap the ordered reductions now performed during dispatch.
            self.profile.gradient_wall_seconds+=elapsed-(self.profile.reduction_seconds-old_reduction);
            if let Some(error)=failure {return Err(invalid(error));}
        }
        let t=paisho_platform::training_time::now();
        for (g, w) in g.iter_mut().zip(model.parameters()) {
            *g += l2 * w;
        }
        let (references, gram) = if self.policy_gradient_constraint {
            (self.gradients.as_slice(), self.gram.as_slice())
        } else {
            (&self.gradients[1..], self.value_gram.as_slice())
        };
        let projected = projection::homogeneous(&g, references, gram)
            .ok_or_else(|| invalid("non-finite homogeneous protection projection"))?;
        let norm = dot(&projected, &projected).sqrt();
        if !norm.is_finite() {
            return Err(invalid("non-finite protected gradient"));
        }
        let scale = if norm > 10. { 10. / norm } else { 1. };
        *model = weights(
            model,
            model
                .parameters()
                .iter()
                .zip(projected)
                .map(|(w, g)| w - rate * scale * g)
                .collect(),
        )?;
        let original_norm = dot(&g, &g).sqrt();
        self.retained_norm_sum += if original_norm > 0. {
            norm / original_norm
        } else {
            1.
        };
        self.zero_steps += usize::from(norm < 1e-12);
        self.updates += 1;
        self.profile.update_seconds+=paisho_platform::training_time::elapsed(t).as_secs_f64();
        Ok(())
    }
    /// Numerical correction of the accumulated finite step; no extra replay tickets.
    /// Its reference evaluations are counted separately from SGD presentations.
    pub fn consolidate(&mut self, model: &mut MicroModel) -> Result<()> {
        self.consolidate_target(model, 0.5, |_| {})
    }
    /// The validated composed repair: failed fresh constraints remain active,
    /// with two critical choice margins and a numerical interior target.
    pub fn consolidate_for_publication(&mut self, model: &mut MicroModel) -> Result<()> {
        self.consolidate_mode(model, 0.5, true, true, true, true, |_| {})
    }
    /// Carry the SAME finite fresh contract to the final actor. A rejected
    /// consolidation supplies no accepted new gain for publication to preserve.
    pub fn publication_fresh_contract(&self) -> Result<Option<(Vec<Arc<MicroExample>>, f64)>> {
        if self.last["accepted"] != true { return Ok(None); }
        let old=self.last["fresh_anchor"].as_f64().ok_or_else(||invalid("missing fresh anchor"))?;
        let learned=self.last["fresh_before"].as_f64().ok_or_else(||invalid("missing learned fresh loss"))?;
        let ceiling=if learned<old {old-0.05*(old-learned)} else {learned};
        if !ceiling.is_finite() {return Err(invalid("nonfinite publication fresh ceiling"));}
        Ok(Some((self.fresh.iter().cloned().collect(),ceiling)))
    }
    // Diagnostic comparisons use this SAME implementation with a different
    // correction target only. The finite acceptance interval never changes.
    fn consolidate_target(
        &mut self,
        model: &mut MicroModel,
        target_tolerance_fraction: f64,
        capture_attempt: impl FnOnce(&MicroModel),
    ) -> Result<()> {
        self.consolidate_mode(model, target_tolerance_fraction, false, false, false, false, capture_attempt)
    }
    /// Isolated diagnostic only: optimize the same finite fresh criterion too.
    /// No option is persisted, and no runtime caller selects this path.
    pub fn diagnostic_consolidate_fresh_active(
        &mut self,
        model: &mut MicroModel,
        capture_attempt: impl FnOnce(&MicroModel),
    ) -> Result<()> {
        if !self.loop_v3 {return Err(invalid("fresh-active diagnostic requires V3"));}
        self.consolidate_mode(model, 0.5, true, false, false, false, capture_attempt)
    }
    /// Separate opt-in diagnostic: continue reducing a violated choice margin
    /// even while its discrete success has not changed. No runtime caller.
    pub fn diagnostic_consolidate_continuous_choices(
        &mut self, model: &mut MicroModel, fresh_active: bool,
        capture_attempt: impl FnOnce(&MicroModel),
    ) -> Result<()> {
        if !self.loop_v3 {return Err(invalid("continuous choice diagnostic requires V3"));}
        self.consolidate_mode(model, 0.5, fresh_active, true, false, false, capture_attempt)
    }
    /// Separate diagnostic margin only; the finite ceiling and six-by-six
    /// budget remain unchanged. No runtime caller or persisted option.
    pub fn diagnostic_consolidate_fresh_interior(
        &mut self, model: &mut MicroModel, continuous_choices: bool,
        capture_attempt: impl FnOnce(&MicroModel),
    ) -> Result<()> {
        if !self.loop_v3 {return Err(invalid("fresh-interior diagnostic requires V3"));}
        self.consolidate_mode(model, 0.5, true, continuous_choices, true, false, capture_attempt)
    }
    /// Diagnostic only: jointly protect the two most critical old margins.
    /// Fresh interior remains independently selectable; no runtime caller.
    pub fn diagnostic_consolidate_two_choices(
        &mut self, model: &mut MicroModel, fresh_active: bool, fresh_interior: bool,
        capture_attempt: impl FnOnce(&MicroModel),
    ) -> Result<()> {
        if !self.loop_v3 || (fresh_interior && !fresh_active) {
            return Err(invalid("invalid V3 two-choice diagnostic configuration"));
        }
        self.consolidate_mode(model, 0.5, fresh_active, true, fresh_interior, true, capture_attempt)
    }
    fn consolidate_mode(
        &mut self,
        model: &mut MicroModel,
        target_tolerance_fraction: f64,
        fresh_active: bool,
        continuous_choices: bool,
        fresh_interior: bool,
        two_choice_constraints: bool,
        capture_attempt: impl FnOnce(&MicroModel),
    ) -> Result<()> {
        if ![0.5, 1.].contains(&target_tolerance_fraction) {
            return Err(invalid("unsupported diagnostic consolidation target"));
        }
        let t = paisho_platform::training_time::now();
        self.checks += 1;
        let limits = self.anchor_losses;
        let old_choices = self.anchor_choices.clone();
        let mut candidate = model.clone();
        let (mut measured, mut decision_margins) = choice_merit::read(&candidate, &self.rows, self.parallel.as_ref(), continuous_choices)?;
        let mut decision_trace = Vec::new();
        let mut joint_choice_trace = Vec::new();
        let mut joint_choice_gradient_seconds = 0.;
        let mut joint_choice_gradient_calls = 0usize;
        let mut joint_maximum_constraints = 0usize;
        let mut validation_measured = self.diagnostic_validation_loss(&candidate)?;
        let validation_anchor = self.validation_value.as_ref().map(|p|p.anchor_loss);
        let fresh = self.fresh.iter().cloned().collect::<Vec<_>>();
        let fresh_start = fresh_active.then(paisho_platform::training_time::now);
        let old_fresh = fresh_loss_mode(&self.anchor, &fresh, self.parallel.as_ref(), self.loop_v3)?;
        let learned_fresh = fresh_loss_mode(model, &fresh, self.parallel.as_ref(), self.loop_v3)?;
        let mut barrier = if fresh_active {
            Some(fresh_barrier::Barrier::new(&fresh, old_fresh, learned_fresh,
                paisho_platform::training_time::elapsed(fresh_start.unwrap()).as_secs_f64())?)
        } else {None};
        let mut candidate_fresh = fresh_active.then_some(learned_fresh);
        let mut iterations = 0;
        let mut choice_constraints = 0;
        let mut final_choices = old_choices.clone();
        let mut success = false;
        let mut final_losses = limits;
        let mut final_fresh = None;
        let mut line_search_trials = 0;
        let mut interior_trace = vec![];
        for i in 0..=6 {
            let (current, choices) = &measured;
            final_losses = *current;
            final_choices = choices.clone();
            iterations = i;
            if let Some(margins) = &decision_margins {
                decision_trace.push(serde_json::json!({"kind":"candidate","iteration":i,
                    "margins":margins.trace(&old_choices)}));
            }
            let finite = current.iter().zip(limits).enumerate().all(|(k, (a, b))| a.is_finite() && ((self.loop_v3 && k == 0) || *a <= b + FINITE_LOSS_TOLERANCE));
            let retained = old_choices.iter().zip(choices).all(|(a, b)| !*a || *b);
            if let Some(barrier) = barrier.as_mut() {
                let value = candidate_fresh.ok_or_else(|| invalid("missing fresh-active scalar"))?;
                final_fresh = Some(value);
                barrier.observe(i, value, finite && retained && self.diagnostic_validation_accepts(validation_measured));
            }
            if finite && retained && self.diagnostic_validation_accepts(validation_measured) {
                let measured_fresh = match candidate_fresh {
                    Some(value) => value,
                    None => fresh_loss_mode(&candidate, &fresh, self.parallel.as_ref(), self.loop_v3)?,
                };
                final_fresh = Some(measured_fresh);
                let ceiling = if learned_fresh < old_fresh {
                    old_fresh - 0.05 * (old_fresh - learned_fresh)
                } else {
                    learned_fresh
                };
                success = measured_fresh <= ceiling + 1e-12;
                // Historical mode retains its exact early exit. Diagnostic mode
                // spends only the remaining corrections on the failed criterion.
                if success || !fresh_active {break;}
            }
            if i == 6 {
                break;
            }
            let mut refs = references(&candidate, &self.rows, self.parallel.as_ref())?;
            self.reference_evaluations += self.rows.len();
            let mut rhs = if target_tolerance_fraction == 0.5 {
                repair_rhs(current, &limits)
            } else {
                current.iter().zip(limits).map(|(a, b)| a - (b + FINITE_LOSS_TOLERANCE)).collect()
            };
            if self.loop_v3 { refs.remove(0); rhs.remove(0); }
            if let Some(panel) = &self.validation_value {
                refs.push(panel.gradient(&candidate,self.parallel.as_ref())?);
                rhs.push(validation_measured.ok_or_else(||invalid("missing diagnostic validation loss"))?
                    - (panel.anchor_loss + target_tolerance_fraction * FINITE_LOSS_TOLERANCE));
                self.reference_evaluations += panel.rows.len();
            }
            if two_choice_constraints {
                let started_choices = paisho_platform::training_time::now();
                let selected = decision_margins.as_ref().ok_or_else(||invalid("missing joint choice reads"))?
                    .constraints(&candidate,&self.rows,&old_choices)?;
                joint_choice_gradient_seconds += paisho_platform::training_time::elapsed(started_choices).as_secs_f64();
                joint_choice_gradient_calls += 2 * selected.len();
                for (gradient,gap,detail) in selected {
                    choice_constraints += 1;
                    refs.push(gradient);rhs.push(gap);self.reference_evaluations += 2;
                    joint_choice_trace.push(serde_json::json!({"iteration":i,"selected":detail}));
                }
            } else if let Some((gradient, gap)) =
                lost_choice_constraint(&candidate, &self.rows, &old_choices)?
            {
                choice_constraints += 1;
                refs.push(gradient);
                rhs.push(gap);
                self.reference_evaluations += 2;
            }
            let mut fresh_normal = None;
            if let Some(barrier) = barrier.as_mut() {
                let value = candidate_fresh.ok_or_else(|| invalid("missing fresh-active scalar"))?;
                if barrier.active {
                    match barrier.gradient(&candidate, self.parallel.as_ref()) {
                        Ok(gradient) => refs.push(gradient),
                        Err(_) => break,
                    }
                    rhs.push(value - barrier.ceiling);
                    fresh_normal = Some((refs.len()-1, value, barrier.ceiling));
                    self.reference_evaluations += fresh.len();
                }
                barrier.maximum_constraints = barrier.maximum_constraints.max(refs.len());
                if refs.len() > if two_choice_constraints {7} else {6} {return Err(invalid("fresh-active constraint budget exceeded"));}
            }
            if two_choice_constraints {
                joint_maximum_constraints=joint_maximum_constraints.max(refs.len());
                if refs.len()>7 {return Err(invalid("joint choice constraint budget exceeded"));}
            }
            let origin = vec![0.; model.parameters().len()];
            let correction = if fresh_interior && fresh_normal.is_some() {
                let (index, value, ceiling) = fresh_normal.unwrap();
                let step = fresh_interior::solve(&origin, &refs, &rhs, index, value, ceiling)?;
                interior_trace.push(serde_json::json!({"iteration":i,"step":step.diagnostic}));
                step.correction
            } else {projection::affine(&origin, &refs, &rhs)};
            let Some(correction) = correction else {break;};
            let mut before = if let Some(margins) = &decision_margins {
                violation_merit(current, &limits, &[], &[], self.loop_v3) + margins.merit(&old_choices)?
            } else { violation_merit(current, &limits, choices, &old_choices, self.loop_v3) };
            before += self.diagnostic_validation_merit(validation_measured);
            if let Some(barrier) = &barrier {before += barrier.merit(candidate_fresh.unwrap());}
            let mut accepted_step = None;
            for halve in 0..=5 {
                line_search_trials += 1;
                let scale = 0.5_f64.powi(halve);
                let Ok(trial) = weights(&candidate, candidate.parameters().iter()
                    .zip(&correction).map(|(w, g)| w - scale * g).collect()) else { continue; };
                let Ok((score, trial_margins)) = choice_merit::read(&trial, &self.rows, self.parallel.as_ref(), continuous_choices) else { continue; };
                let Ok(validation_score) = self.diagnostic_validation_loss(&trial) else {continue;};
                let trial_fresh = if let Some(barrier) = barrier.as_mut() {
                    match barrier.loss(&trial, self.parallel.as_ref(), self.loop_v3) {
                        Ok(value) => Some(value),
                        Err(error) => {
                            barrier.rejected_error(i, halve, &error.to_string());
                            continue;
                        }
                    }
                } else {None};
                let mut after = if let Some(margins) = &trial_margins {
                    violation_merit(&score.0, &limits, &[], &[], self.loop_v3) + margins.merit(&old_choices)?
                } else { violation_merit(&score.0, &limits, &score.1, &old_choices, self.loop_v3) };
                after += self.diagnostic_validation_merit(validation_score);
                if let Some(barrier) = &barrier {after += barrier.merit(trial_fresh.unwrap());}
                let admitted = after.is_finite() && (after == 0. || after < before);
                if let Some(barrier) = barrier.as_mut() {
                    barrier.trial(i, halve, trial_fresh.unwrap(), before, after, admitted);
                }
                if let Some(margins) = &trial_margins {
                    decision_trace.push(serde_json::json!({"kind":"trial","iteration":i,"halve":halve,
                        "merit_before":before,"merit_after":after,"admitted":admitted,
                        "margins":margins.trace(&old_choices)}));
                }
                if admitted {
                    accepted_step = Some((trial, score, validation_score, trial_fresh, trial_margins));
                    break;
                }
            }
            let Some((trial, score, validation_score, trial_fresh, trial_margins)) = accepted_step else { break; };
            candidate = trial;
            measured = score;
            validation_measured = validation_score;
            candidate_fresh = trial_fresh;
            decision_margins = trial_margins;
        }
        capture_attempt(&candidate);
        if success {
            self.rebuild_anchor(&candidate, final_losses, final_choices.clone())?;
            *model = candidate;
            self.accepted += 1;
        } else {
            // SGD presentations and all replay data remain consumed and durable.
            // Only the unvalidated weight transaction is rolled back. Its old
            // validated anchor and gradients remain the reference for the next block.
            *model = self.anchor.clone();
        }
        self.last = serde_json::json!({"accepted":success,"rolled_back":!success,"iterations":iterations,"line_search_trials":line_search_trials,"choice_constraints":choice_constraints,"raw_before":old_choices.iter().filter(|x|**x).count(),"raw_after":final_choices.iter().filter(|x|**x).count(),"lost_choices":old_choices.iter().zip(&final_choices).filter(|(a,b)|**a && !**b).count(),"reference_before":limits,"reference_after":final_losses,"applied_reference_after":self.anchor_losses,"fresh_anchor":old_fresh,"fresh_before":learned_fresh,"fresh_after":final_fresh,"fresh_after_measured":final_fresh.is_some(),"constraint_scope":if self.loop_v3 {"mean value class losses and individual raw verified choices; confidence diagnostic; publication separately verifies coupled choices"} else {"mean class losses and individual raw verified choices; publication separately verifies coupled choices"}});
        if let Some(barrier) = &barrier {
            self.last["fresh_barrier"] = barrier.progress();
        }
        if two_choice_constraints {
            self.last["joint_choice_constraints"] = serde_json::json!({"diagnostic_only":true,
                "maximum_choice_halfspaces":2,"selection":"largest current log-probability gaps among all old protected choices; ascending row breaks ties",
                "still_correct_rows_included":true,"additional_gap_threshold":serde_json::Value::Null,
                "interior_log_gap":1e-6,"maximum_total_halfspaces":7,"observed_maximum_halfspaces":joint_maximum_constraints,
                "gradient_calls":joint_choice_gradient_calls,"gradient_seconds":joint_choice_gradient_seconds,
                "maximum_corrections":6,"maximum_trials_per_correction":6,"selected":joint_choice_trace});
        }
        if fresh_interior {
            self.last["fresh_interior"] = serde_json::json!({"diagnostic_only":true,
                "relative_margin":f64::EPSILON.sqrt(),"target_rule":"max(0,C-sqrt(epsilon)*max(1,abs(C),abs(current fresh)))",
                "acceptance_ceiling_unchanged":true,"acceptance_tolerance":1e-12,
                "maximum_corrections":6,"maximum_trials_per_correction":6,
                "fallback":"original affine target C only if stricter solve is infeasible",
                "additional_model_reads":0,"additional_gradients":0,"solves":interior_trace});
        }
        if continuous_choices {
            self.last["continuous_choice_merit"] = serde_json::json!({"diagnostic_only":true,
                "interior_log_gap":1e-6,"old_protected_mask":old_choices,"states_and_trials":decision_trace,
                "additional_forward_reads":0,"maximum_corrections":6,"maximum_trials_per_correction":6,
                "final_choice_criterion_unchanged":true});
        }
        if let Some(panel) = &self.validation_value {
            self.last["validation_value"] = serde_json::json!({
                "rows":panel.rows.len(),"class_counts":panel.counts,"anchor":validation_anchor,
                "attempted":validation_measured,"applied":if success {validation_measured} else {validation_anchor},
                "full_mse_not_half_loss":true,"acceptance_tolerance":FINITE_LOSS_TOLERANCE,
                "training_constraints":self.gradients.len(),"maximum_finite_constraints":if two_choice_constraints {7} else if fresh_active {6} else {5}});
        }
        self.seconds += paisho_platform::training_time::elapsed(t).as_secs_f64();
        Ok(())
    }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"loop_v3":self.loop_v3,"fresh_positions":self.fresh.len(),"training_profile":self.profile.progress(),"checkpoint":self.saved,"updates":self.updates,"retained_norm_sum":self.retained_norm_sum,"zero_steps":self.zero_steps,"checks":self.checks,"accepted_corrections":self.accepted,"reference_evaluations":self.reference_evaluations,"seconds":self.seconds,"last":self.last,"references":self.rows.len()})
    }
}
