//! Optional versioned Gen3.4 value and retrieval configuration. Legacy remains exact.
use super::*;
use paisho_core::{GameOutcome, TurnPhase};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gen3MemoryScope {
    #[default]
    Root,
    BonusNodes,
    AllNodes,
}
impl Gen3MemoryScope {
    pub fn name(self) -> &'static str {
        match self {
            Self::Root => "root",
            Self::BonusNodes => "bonus-nodes",
            Self::AllNodes => "all-nodes",
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "root" => Ok(Self::Root),
            "bonus-nodes" => Ok(Self::BonusNodes),
            "all-nodes" => Ok(Self::AllNodes),
            _ => Err("invalid Gen3 memory scope".into()),
        }
    }
    pub(super) fn uses_memory(self, p: &Position) -> bool {
        self == Self::AllNodes || self == Self::BonusNodes && p.phase() == TurnPhase::HarmonyBonus
    }
}
impl Gen32Model {
    /// Explicit, idempotent upgrade. A fresh residual contributes exactly zero.
    pub fn with_value_residual(mut self, seed: u64) -> Self {
        if self.value_residual.is_none() {
            self.value_residual = Some(Arc::new(Gen3ValueResidual::seeded(seed)));
            if self.value_extra.is_none() {
                self.value_extra = Some([0.; 64]);
            }
        }
        self
    }
    pub fn with_value128(mut self, weights: [f64; 64]) -> Result<Self, String> {
        if weights.iter().any(|w| !w.is_finite())
            || !weights.iter().map(|w| w.abs()).sum::<f64>().is_finite()
        {
            return Err("invalid extra value weights".into());
        }
        self.value_extra = Some(weights);
        Ok(self)
    }
    pub fn predict_value_state(&self, state: &[f64; 128]) -> f64 {
        let raw: f64 = self
            .value
            .weights()
            .iter()
            .zip(state)
            .map(|(w, x)| w * x)
            .sum();
        let raw = if let Some(extra) = &self.value_extra {
            raw + extra
                .iter()
                .zip(&state[64..])
                .map(|(w, x)| w * x)
                .sum::<f64>()
        } else {
            raw
        };
        let raw = raw + self.value_residual.as_ref().map_or(0., |r| r.raw(state));
        raw / (1. + raw.abs())
    }
    pub fn value_at(&self, p: &Position, player: Player) -> f32 {
        let Some(extra) = &self.value_extra else {
            return self.value.evaluate(p, player);
        };
        match p.outcome() {
            GameOutcome::Win(w) => return if w == player { 1. } else { -1. },
            GameOutcome::Draw => return 0.,
            GameOutcome::Ongoing => {}
        }
        // Exact legacy path until new coordinates actually contribute.
        if extra.iter().all(|w| *w == 0.)
            && !self.value_residual.as_ref().is_some_and(|r| r.active())
        {
            return self.value.evaluate(p, player);
        }
        let sign = if player == p.to_move() { 1. } else { -1. };
        (sign * self.predict_value_state(&micro_state_features(p))) as f32
    }
    pub(super) fn trained_value(
        &self,
        example: &MicroExample,
        rate: f64,
    ) -> Result<
        (
            CompactValueModel,
            Option<[f64; 64]>,
            Option<Arc<Gen3ValueResidual>>,
        ),
        String,
    > {
        let Some(extra) = self.value_extra else {
            let f =
                CompactValueFeatures::from_values(example.state[..64].try_into().unwrap(), None)
                    .map_err(|e| e.to_string())?;
            let mut value = self.value.clone();
            value
                .train_step(&f, example.value, rate, 0.)
                .map_err(|e| e.to_string())?;
            if self.value_residual.is_some() {
                return Err("residual requires 128-input value".into());
            }
            return Ok((value, None, None));
        };
        if !rate.is_finite() || rate <= 0. {
            return Err("invalid learning rate".into());
        }
        let raw = self
            .value
            .weights()
            .iter()
            .zip(&example.state[..64])
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + extra
                .iter()
                .zip(&example.state[64..])
                .map(|(w, x)| w * x)
                .sum::<f64>()
            + self
                .value_residual
                .as_ref()
                .map_or(0., |r| r.raw(&example.state));
        let error = raw / (1. + raw.abs()) - example.value;
        let step = rate * error / (1. + raw.abs()).powi(2);
        let weights = std::array::from_fn(|i| self.value.weights()[i] - step * example.state[i]);
        let next = std::array::from_fn(|i| extra[i] - step * example.state[64 + i]);
        let residual = self
            .value_residual
            .as_ref()
            .map(|r| r.trained(&example.state, step).map(Arc::new))
            .transpose()?;
        if next.iter().any(|w: &f64| !w.is_finite()) {
            return Err("nonfinite extended update".into());
        }
        Ok((
            CompactValueModel::from_weights(weights).map_err(|e| e.to_string())?,
            Some(next),
            residual,
        ))
    }
}
