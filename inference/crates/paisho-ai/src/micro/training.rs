use super::*;

#[derive(Clone, Copy)]
enum NeuralValueContext { Joint, DetachedCurrent, Frozen(f64) }
#[cfg(test)]
#[path = "training_context_tests.rs"]
mod context_tests;

#[derive(Clone, Debug)]
pub struct MicroExample {
    /// Explicit verified winning support. False retains categorical imitation.
    pub policy_support: bool,
    /// Optional per-action Q supervision; None means unresolved, never a loss.
    /// Provenance remains in the dense/compact evidence archive. Empty is legacy.
    pub action_values: Vec<Option<f64>>,
    /// Whole-game exclusion for retrieval, zero when no source is known.
    pub sequence_source: u64,
    /// Independent value-loss coefficient; zero supports policy-only consolidation.
    pub value_weight: f64,
    pub state: Vec<f64>,
    pub actions: Vec<[f64; MICRO_ACTION_INPUTS]>,
    /// Distribution over exactly `actions`; empty only for value-only examples.
    pub policy: Vec<f64>,
    pub value: f64,
    pub policy_weight: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct MicroLoss {
    pub value: f64,
    pub policy: f64,
}
impl MicroLoss {
    pub fn total(self, policy_weight: f64) -> f64 {
        self.value + policy_weight * self.policy
    }
}

#[cfg(test)]
mod reused_buffer_tests {
    use super::*;
    #[test]
    fn reusable_gradient_capacity_never_reuses_old_coefficients() {
        let r:paisho_core::GameRecord=include_str!("../../tests/fixtures/micro-alias-1-a.psr").parse().unwrap();
        let p=r.replay().unwrap();let actions=paisho_core::legal_actions(&p);
        for model in [MicroModel::seeded(957),MicroModel::seeded(957).with_neural_memory(193)] {
            let e=MicroExample {policy_support:false,state:model.state_features(&p),actions:actions.iter().map(|&a|micro_action_features(&p,a)).collect(),policy:vec![1./actions.len() as f64;actions.len()],value:0.8,policy_weight:1.,value_weight:1.,action_values:vec![Some(1.);actions.len()],sequence_source:0};
            let (loss,expected)=model.loss_gradient(&e).unwrap();
            for size in [0,19,expected.len(),expected.len()+17] {
                let buffer=vec![f64::from_bits(0xfff8_0000_0000_0001);size];
                let (actual,g)=model.loss_gradient_reusing(&e,buffer).unwrap();
                assert_eq!((loss.value.to_bits(),loss.policy.to_bits()),(actual.value.to_bits(),actual.policy.to_bits()));
                assert_eq!(g.len(),expected.len());assert!(g.iter().zip(&expected).all(|(a,b)|a.to_bits()==b.to_bits()));
                let mut value=e.clone();value.actions.clear();value.policy.clear();value.action_values.clear();value.policy_weight=0.;
                let (_,expected)=model.loss_gradient(&value).unwrap();let (_,actual)=model.loss_gradient_reusing(&value,g).unwrap();
                assert!(expected.iter().zip(&actual).all(|(a,b)|a.to_bits()==b.to_bits()));
            }
        }
    }
}

impl MicroExample {
    pub fn validate(&self) -> Result<(), String> {
        if ![MICRO_INPUTS, MICRO_SPATIAL_INPUTS].contains(&self.state.len())
            || !self.value.is_finite()
            || self.value.abs() > 1.0
            || !self.value_weight.is_finite() || !(0.0..=1.0).contains(&self.value_weight)
            || !self.policy_weight.is_finite()
            || self.policy_weight < 0.0
            || self
                .state
                .iter()
                .chain(self.actions.iter().flatten())
                .any(|x| !x.is_finite())
            || self.actions.len() != self.policy.len()
            || (!self.action_values.is_empty() && self.action_values.len()!=self.actions.len())
            || self.action_values.iter().flatten().any(|v|!v.is_finite() || v.abs()>1.+1e-9)
            || self.policy.iter().any(|x| !x.is_finite() || *x < 0.0)
            || (!self.policy.is_empty() && (self.policy.iter().sum::<f64>() - 1.0).abs() > 1e-8)
            || (self.policy.is_empty() && self.policy_weight != 0.0)
        {
            return Err("invalid micro training target or features".into());
        }
        Ok(())
    }
}
impl MicroModel {
    /// Exact shared-trunk gradient: squared value error plus the explicitly
    /// selected policy objective (categorical by default, verified support set).
    pub fn loss_gradient(&self, ex: &MicroExample) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_mode(ex, true)
    }
    /// Reuse only allocation capacity. Every gradient coefficient is initialized
    /// to positive zero before computing this example at the current weights.
    pub fn loss_gradient_reusing(&self, ex:&MicroExample, buffer:Vec<f64>) -> Result<(MicroLoss,Vec<f64>),String> {
        self.loss_gradient_impl::<true>(ex,true,None,NeuralValueContext::Joint,false,buffer)
    }
    fn loss_gradient_mode(
        &self,
        ex: &MicroExample,
        value_loss: bool,
    ) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_impl::<true>(ex, value_loss, None,NeuralValueContext::Joint,false,Vec::new())
    }
    /// The identical forward loss without allocating or backpropagating a gradient.
    pub fn loss(&self, ex: &MicroExample) -> Result<MicroLoss, String> {
        self.loss_gradient_impl::<false>(ex, true, None,NeuralValueContext::Joint,false,Vec::new()).map(|(loss, _)| loss)
    }
    /// Reproduce the existing recalled-support policy gradient, reusing this
    /// model's neural read while normalizing its probability on the given support.
    pub fn policy_mass_gradient(&self, ex: &MicroExample) -> Result<Vec<f64>, String> {
        ex.validate()?;
        let base=micro_softmax(&Self::logits(&self.embed(&ex.state),&ex.actions))?;
        let (p,cache)=self.memory_priors_cached(&ex.state,&ex.actions,&base,ex.sequence_source)?;
        let mass:f64=p.iter().zip(&ex.policy).filter(|(_,t)|**t>0.).map(|(p,_)|p).sum();
        if mass<=0. {return Err("zero recalled policy mass".into());}
        let mut target=ex.clone();
        target.policy_support=false;
        target.policy=p.iter().zip(&ex.policy).map(|(p,t)|if *t>0. {p/mass}else{0.}).collect();
        target.policy_weight=1.;target.value_weight=0.;
        self.loss_gradient_impl::<true>(&target,true,cache.as_ref(),NeuralValueContext::Joint,false,Vec::new()).map(|(_,g)|g)
    }
    /// Partial derivative with the current value held fixed only as the neural
    /// reader's context. Direct value supervision and policy/memory paths remain
    /// differentiable. This is deliberately distinct from the joint gradient.
    pub fn loss_gradient_detached_value(&self, ex: &MicroExample) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_detached_value_reusing(ex, Vec::new())
    }
    pub fn loss_gradient_detached_value_reusing(&self, ex: &MicroExample, buffer: Vec<f64>) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_impl::<true>(ex,true,None,NeuralValueContext::DetachedCurrent,false,buffer)
    }
    /// V3 retains the historical total Q budget shared over known labels after
    /// provenance masking. Sparse proven labels keep their strength; surviving
    /// estimates may receive a larger individual share of the same bounded budget.
    pub fn loss_gradient_loop_v3_reusing(&self, ex: &MicroExample, buffer: Vec<f64>) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_impl::<true>(ex,true,None,NeuralValueContext::DetachedCurrent,false,buffer)
    }
    /// Matching V3 forward objective; detachment changes its derivative only.
    pub fn loss_loop_v3(&self, ex: &MicroExample) -> Result<MicroLoss, String> {
        self.loss_gradient_impl::<false>(ex,true,None,NeuralValueContext::DetachedCurrent,false,Vec::new()).map(|(loss,_)|loss)
    }
    /// Diagnostic alternative only. Normalizing over all legal actions avoids
    /// redistributing Q mass, but can severely attenuate sparse regulatory proofs.
    pub fn loss_gradient_all_actions_auxiliary_reusing(&self, ex: &MicroExample, buffer: Vec<f64>) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_impl::<true>(ex,true,None,NeuralValueContext::DetachedCurrent,true,buffer)
    }
    /// Exact derivative of L(parameters; fixed reader value). Hold this same
    /// context fixed when checking the derivative by parameter perturbations.
    pub fn loss_gradient_with_value_context(&self, ex: &MicroExample, context: f64) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_impl::<true>(ex,true,None,NeuralValueContext::Frozen(context),false,Vec::new())
    }
    pub fn loss_with_value_context(&self, ex: &MicroExample, context: f64) -> Result<MicroLoss, String> {
        self.loss_gradient_impl::<false>(ex,true,None,NeuralValueContext::Frozen(context),false,Vec::new()).map(|(loss,_)|loss)
    }
    fn loss_gradient_impl<const GRADIENT: bool>(
        &self, ex: &MicroExample, value_loss: bool,
        cached_neural: Option<&neural_memory::Cache>,
        value_context: NeuralValueContext,
        dense_auxiliary_normalizer: bool,
        mut buffer: Vec<f64>,
    ) -> Result<(MicroLoss, Vec<f64>), String> {
        ex.validate()?;
        if let NeuralValueContext::Frozen(value) = value_context {
            if !value.is_finite() || value.abs()>1. {return Err("invalid frozen neural value context".into());}
        }
        let mut deep = self.has_deep_value().then(deep_value::DeepValueActivations::default);
        let emb = self.embed_with_deep(&ex.state,deep.as_mut());
        let reader_value = match value_context {NeuralValueContext::Frozen(value)=>value,_=>emb.value};
        let w = &self.parameters;
        let size=if GRADIENT {self.parameters.len()} else {0};
        let mut g=if size>0 && buffer.capacity()>=size {
            buffer.resize(size,0.);buffer.fill(0.);buffer
        } else {vec![0.;size]};
        let mut dh = [0.0; MICRO_HIDDEN];
        let delta = if value_loss {
            emb.value - ex.value
        } else {
            0.0
        };
        let mut dv = delta * (1.0 - emb.value * emb.value) * ex.value_weight;
        let value_hidden = emb.value_hidden.as_ref().unwrap_or(&emb.hidden);
        for j in 0..MICRO_HIDDEN {
            if !self.has_spatial() { dh[j] = dv * w[VALUE_W + j]; }
        }
        let mut policy_loss = 0.0;
        let mut auxiliary_loss = 0.0;
        if !ex.actions.is_empty() {
            let context = self.memory_context(&ex.state, ex.sequence_source)?;
            let vector = context.as_ref().map(|c| self.memory_vector(c));
            let mut memory_dc = [0.; 32];
            // The frozen forward and backward passes use exactly the same
            // residual activations. Keep them once for this example only.
            let activations: Option<Vec<_>> = emb.residual.as_ref().map(|r| {
                ex.actions.iter().map(|a|r.activations(a)).collect()
            });
            let mut logits = match (&emb.residual, &activations) {
                (Some(r), Some(values)) => ex.actions.iter().zip(values).map(|(a,h)| {
                    let base = Self::base_logit(&emb,a);
                    if r.weights.active { base+r.logit_from_activations(h) } else { base }
                }).collect(),
                _ => Self::logits(&emb,&ex.actions),
            };
            let computed_neural = (self.has_neural_memory() && cached_neural.is_none()).then(||self.neural_forward(&ex.state,&ex.actions,&vector.unwrap_or([0.;32]),reader_value,&logits));
            let neural=cached_neural.or(computed_neural.as_ref());
            let mut memory_derivatives = Vec::new();
            if let Some(v) = &vector {
                memory_derivatives.reserve(logits.len());
                for (l, a) in logits.iter_mut().zip(&ex.actions) {
                    let (value, derivative) = Self::memory_action(v, a);
                    *l += value;
                    memory_derivatives.push(derivative);
                }
            }
            if let Some(c)=&neural {for (l,y) in logits.iter_mut().zip(c.output.chunks_exact(2)){if y[0]!=0. {*l+=y[0];}}}
            let (probabilities, max, total) = micro_softmax_parts(&logits)?;
            let log_z = max + total.ln();
            // A singleton support is exactly the old CE path, including loss
            // reduction and every gradient coefficient. Multi-action supports
            // use a stable log-sum-exp and its exact analytic derivative.
            let support_target = if ex.policy_support
                && ex.policy.iter().filter(|p| **p > 0.).count() > 1 {
                let support_logits: Vec<_> = logits.iter().zip(&ex.policy)
                    .filter(|(_, t)| **t > 0.).map(|(l, _)| *l).collect();
                let (conditional, support_max, support_total) = micro_softmax_parts(&support_logits)?;
                policy_loss = log_z - (support_max + support_total.ln());
                let mut conditional = conditional.into_iter();
                Some(ex.policy.iter().map(|t| if *t > 0. {
                    conditional.next().unwrap()
                } else { 0. }).collect::<Vec<_>>())
            } else { None };
            let policy_target = support_target.as_deref().unwrap_or(&ex.policy);
            let side = if let Some(c)=&neural {
                let mut d=vec![0.;ex.actions.len()*2];
                let known=if dense_auxiliary_normalizer {ex.actions.len().max(1)} else {ex.action_values.iter().flatten().count().max(1)} as f64;
                let scale=if value_loss {0.25*ex.value_weight/known} else {0.};
                for (i,(p,t)) in probabilities.iter().zip(policy_target).enumerate() {
                    d[2*i]=ex.policy_weight*(p-t);
                    if let Some(Some(target))=ex.action_values.get(i) {
                        let q=c.output[2*i+1].tanh();let error=q-target;
                        auxiliary_loss+=0.5*scale*error*error;
                        d[2*i+1]=scale*error*(1.-q*q);
                    }
                }
                if GRADIENT {
                let side=self.neural_backward(c,&d,&mut g);
                if matches!(value_context,NeuralValueContext::Joint) && side.value!=0. {dv+=side.value*(1.-emb.value*emb.value);}
                memory_dc=side.memory;
                Some(side)
                } else {None}
            } else {None};
            let mut dc = [0.0; MICRO_ACTION_INPUTS];
            for (action_index, ((features, target), (logit, p))) in ex
                .actions
                .iter()
                .zip(policy_target)
                .zip(logits.iter().zip(probabilities))
                .enumerate()
            {
                if support_target.is_none() {policy_loss += target * (log_z - logit);}
                if !GRADIENT {continue;}
                let mut delta_policy=ex.policy_weight*(p-target);
                if let Some(s)=&side {if s.logits[action_index]!=0. {delta_policy+=s.logits[action_index];}}
                if vector.is_some() {
                    let derivative = memory_derivatives[action_index];
                    for k in 0..32 {
                        memory_dc[k] += ex.policy_weight * (p - target) * derivative * features[k];
                    }
                }
                if let Some(residual) = &emb.residual {
                    residual.accumulate_gradient(
                        features,
                        &activations.as_ref().expect("residual forward activations")[action_index],
                        &emb.hidden,
                        delta_policy,
                        &mut g,
                        &mut dh,
                    );
                }
                for k in 0..MICRO_ACTION_INPUTS {
                    dc[k] += delta_policy * features[k];
                }
            }
            if GRADIENT {
            if let Some(c) = &context {
                for (j, pattern) in c.patterns.iter().enumerate() {
                    g[MICRO_RESIDUAL_PARAMETERS + j] =
                        pattern.iter().zip(memory_dc).map(|(a, b)| a * b / 8.).sum();
                }
            }
            for k in 0..MICRO_ACTION_INPUTS {
                g[POLICY_B + k] = dc[k];
                for j in 0..MICRO_HIDDEN {
                    g[POLICY_W + k * MICRO_HIDDEN + j] = dc[k] * emb.hidden[j];
                    dh[j] += dc[k] * w[POLICY_W + k * MICRO_HIDDEN + j];
                }
            }
        }
        }
        if GRADIENT {
        g[VALUE_B]=dv;
        for j in 0..MICRO_HIDDEN {g[VALUE_W+j]=dv*value_hidden[j];}
        if self.has_spatial(){self.value_trunk_gradient(&ex.state,value_hidden,dv,&mut g);}
        if let Some(a)=&deep {self.deep_value_gradient(&ex.state,a,dv,&mut g);}
        for j in 0..MICRO_HIDDEN {
            let d = dh[j] * (1.0 - emb.hidden[j] * emb.hidden[j]);
            g[TRUNK_B + j] = d;
            self.spatial_gradient(&ex.state, j, d, &mut g);
            for i in 0..MICRO_INPUTS {
                g[j * MICRO_INPUTS + i] = d * ex.state[i];
            }
        }
        }
        let loss = MicroLoss {
            value: 0.5 * delta * delta * ex.value_weight + auxiliary_loss,
            policy: policy_loss,
        };
        if !loss.total(ex.policy_weight).is_finite() || g.iter().any(|x| !x.is_finite()) {
            return Err("non-finite micro forward/backward pass".into());
        }
        Ok((loss, g))
    }
    /// Updates all weights together only after validation; invalid steps cannot
    /// leave a partially changed model. Snapshot publication remains caller-owned.
    pub fn train_step(
        &mut self,
        ex: &MicroExample,
        rate: f64,
        l2: f64,
    ) -> Result<MicroLoss, String> {
        self.train_step_mode(ex, rate, l2, true)
    }
    /// Policy-only gradient without cloning the potentially large legal-action example.
    pub fn train_policy_step(&mut self, ex: &MicroExample, rate: f64) -> Result<MicroLoss, String> {
        self.train_step_mode(ex, rate, 0.0, false)
    }
    fn train_step_mode(
        &mut self,
        ex: &MicroExample,
        rate: f64,
        l2: f64,
        value_loss: bool,
    ) -> Result<MicroLoss, String> {
        if !rate.is_finite() || rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
            return Err("invalid micro learning settings".into());
        }
        let (loss, mut gradient) = self.loss_gradient_mode(ex, value_loss)?;
        for (g, w) in gradient.iter_mut().zip(self.parameters.iter()) {
            *g += l2 * w;
        }
        let norm = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err("non-finite micro gradient norm".into());
        }
        let scale = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        let candidate = self
            .parameters
            .iter()
            .zip(gradient)
            .map(|(w, g)| w - rate * scale * g)
            .collect();
        let mut model = Self::from_parameters(candidate)?;
        model.memory = self.memory.clone();
        *self = model;
        Ok(loss)
    }
}

impl MicroModel {
    /// Frozen minibatch gradients run in the caller's shared CPU pool. Reduction
    /// follows input order, so worker count cannot change the trained weights.
    pub fn train_batch(
        &mut self,
        batch: &[&MicroExample],
        rate: f64,
        l2: f64,
    ) -> Result<MicroLoss, String> {
        self.train_batch_mode(batch, rate, l2, true)
    }
    /// Same gradients and reduction order on this thread, without waiting for
    /// a busy search pool. Useful for small causal online learning batches.
    pub fn train_batch_inline(
        &mut self,
        batch: &[&MicroExample],
        rate: f64,
        l2: f64,
    ) -> Result<MicroLoss, String> {
        self.train_batch_mode(batch, rate, l2, false)
    }
    fn train_batch_mode(
        &mut self,
        batch: &[&MicroExample],
        rate: f64,
        l2: f64,
        parallel: bool,
    ) -> Result<MicroLoss, String> {
        use rayon::prelude::*;
        if batch.is_empty() || !rate.is_finite() || rate <= 0.0 || !l2.is_finite() || l2 < 0.0 {
            return Err("invalid micro minibatch".into());
        }
        let mut gradient = vec![0.0; self.parameters.len()];
        let mut loss = MicroLoss {
            value: 0.0,
            policy: 0.0,
        };
        let mut error = None;
        let mut accumulate = |result: Result<(MicroLoss, Vec<f64>), String>| {
            // Evaluate every example even after an error, matching the previous
            // collected path's retrieval/cache side effects and first error.
            if error.is_some() { return; }
            let (l, g) = match result {
                Ok(value) => value,
                Err(e) => { error = Some(e); return; }
            };
            loss.value += l.value / batch.len() as f64;
            loss.policy += l.policy / batch.len() as f64;
            for (sum, part) in gradient.iter_mut().zip(g) {
                *sum += part / batch.len() as f64;
            }
        };
        if parallel {
            let results: Vec<_> = batch.par_iter().map(|ex| self.loss_gradient(ex)).collect();
            for result in results { accumulate(result); }
        } else {
            // Frozen model and identical input/reduction order; keep only one
            // per-example gradient instead of the complete batch of vectors.
            for ex in batch { accumulate(self.loss_gradient(ex)); }
        }
        if let Some(e) = error { return Err(e); }
        for (g, w) in gradient.iter_mut().zip(self.parameters.iter()) {
            *g += l2 * w;
        }
        let norm = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() {
            return Err("non-finite minibatch gradient".into());
        }
        let scale = if norm > 10.0 { 10.0 / norm } else { 1.0 };
        let next = self
            .parameters
            .iter()
            .zip(gradient)
            .map(|(w, g)| w - rate * scale * g)
            .collect();
        let mut model = Self::from_parameters(next)?;
        model.memory = self.memory.clone();
        *self = model;
        Ok(loss)
    }
}
