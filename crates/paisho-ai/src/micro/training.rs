use super::*;

#[derive(Clone, Debug)]
pub struct MicroExample {
    /// Whole-game exclusion for retrieval, zero when no source is known.
    pub sequence_source: u64,
    pub state: [f64; MICRO_INPUTS],
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

impl MicroExample {
    pub fn validate(&self) -> Result<(), String> {
        if !self.value.is_finite()
            || self.value.abs() > 1.0
            || !self.policy_weight.is_finite()
            || self.policy_weight < 0.0
            || self
                .state
                .iter()
                .chain(self.actions.iter().flatten())
                .any(|x| !x.is_finite())
            || self.actions.len() != self.policy.len()
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
    /// Exact shared-trunk gradient of 0.5 squared value error + weighted policy CE.
    pub fn loss_gradient(&self, ex: &MicroExample) -> Result<(MicroLoss, Vec<f64>), String> {
        self.loss_gradient_mode(ex, true)
    }
    fn loss_gradient_mode(
        &self,
        ex: &MicroExample,
        value_loss: bool,
    ) -> Result<(MicroLoss, Vec<f64>), String> {
        ex.validate()?;
        let emb = self.embed(&ex.state);
        let w = &self.parameters;
        let mut g = vec![0.0; self.parameters.len()];
        let mut dh = [0.0; MICRO_HIDDEN];
        let delta = if value_loss {
            emb.value - ex.value
        } else {
            0.0
        };
        let dv = delta * (1.0 - emb.value * emb.value);
        g[VALUE_B] = dv;
        for j in 0..MICRO_HIDDEN {
            g[VALUE_W + j] = dv * emb.hidden[j];
            dh[j] = dv * w[VALUE_W + j];
        }
        let mut policy_loss = 0.0;
        if !ex.actions.is_empty() {
            let context = self.memory_context(&ex.state, ex.sequence_source);
            let vector = context.as_ref().map(|c| self.memory_vector(c));
            let mut memory_dc = [0.; 32];
            let mut logits = Self::logits(&emb, &ex.actions);
            if let Some(v) = &vector {
                for (l, a) in logits.iter_mut().zip(&ex.actions) {
                    *l += Self::memory_action(v, a).0;
                }
            }
            let probabilities = micro_softmax(&logits)?;
            let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_z = max + logits.iter().map(|l| (l - max).exp()).sum::<f64>().ln();
            let mut dc = [0.0; MICRO_ACTION_INPUTS];
            for ((features, target), (logit, p)) in ex
                .actions
                .iter()
                .zip(&ex.policy)
                .zip(logits.iter().zip(probabilities))
            {
                policy_loss += target * (log_z - logit);
                if let Some(v) = &vector {
                    let (_, derivative) = Self::memory_action(v, features);
                    for k in 0..32 {
                        memory_dc[k] += ex.policy_weight * (p - target) * derivative * features[k];
                    }
                }
                if let Some(residual) = &emb.residual {
                    residual.accumulate_gradient(
                        features,
                        &emb.hidden,
                        ex.policy_weight * (p - target),
                        &mut g,
                        &mut dh,
                    );
                }
                for k in 0..MICRO_ACTION_INPUTS {
                    dc[k] += ex.policy_weight * (p - target) * features[k];
                }
            }
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
        for j in 0..MICRO_HIDDEN {
            let d = dh[j] * (1.0 - emb.hidden[j] * emb.hidden[j]);
            g[TRUNK_B + j] = d;
            for i in 0..MICRO_INPUTS {
                g[j * MICRO_INPUTS + i] = d * ex.state[i];
            }
        }
        let loss = MicroLoss {
            value: 0.5 * delta * delta,
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
        for (g, w) in gradient.iter_mut().zip(&self.parameters) {
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
        let results: Vec<_> = if parallel {
            batch.par_iter().map(|ex| self.loss_gradient(ex)).collect()
        } else {
            batch.iter().map(|ex| self.loss_gradient(ex)).collect()
        };
        let mut gradient = vec![0.0; self.parameters.len()];
        let mut loss = MicroLoss {
            value: 0.0,
            policy: 0.0,
        };
        for result in results {
            let (l, g) = result?;
            loss.value += l.value / batch.len() as f64;
            loss.policy += l.policy / batch.len() as f64;
            for (sum, part) in gradient.iter_mut().zip(g) {
                *sum += part / batch.len() as f64;
            }
        }
        for (g, w) in gradient.iter_mut().zip(&self.parameters) {
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
