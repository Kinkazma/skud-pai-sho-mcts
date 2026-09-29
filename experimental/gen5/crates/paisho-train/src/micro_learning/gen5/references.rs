//! Frozen Gen3 inference only. Shared models/banks, private trees per match.
use super::*;
use paisho_core::{Action, Position, TurnPhase};
use rayon::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpponentSpec {
    pub generation: String,
    pub model: PathBuf,
    pub sha256: String,
    pub solver: bool,
}
#[derive(Debug)]
pub struct Frozen {
    pub spec: OpponentSpec,
    value: CompactValueModel,
    policy: Option<MicroModel>,
    extra: Option<[f64; 64]>,
    residual: Option<Vec<f64>>,
    scope: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Artifact {
    schema: String,
    generation: String,
    rules: String,
    compact: crate::compact_learning::ModelArtifact,
    policy: MicroArtifact,
    parent_sha256: String,
    updates: u64,
    memory_manifest: Option<PathBuf>,
    memory_manifest_sha256: Option<String>,
    #[serde(default)]
    value128_extra: Option<Vec<f64>>,
    #[serde(default)]
    value_residual: Option<Vec<f64>>,
    #[serde(default = "root_scope")]
    memory_scope: String,
}
fn root_scope() -> String {
    "root".into()
}
impl Frozen {
    pub fn load(spec: &OpponentSpec) -> Result<Self> {
        let bytes = fs::read(&spec.model)?;
        if sha256(&bytes) != spec.sha256 {
            return Err(invalid("reference model hash changed"));
        }
        if spec.generation == "3.1" {
            return Ok(Self {
                spec: spec.clone(),
                value: crate::compact_learning::load_model(&spec.model)?.model()?,
                policy: None,
                extra: None,
                residual: None,
                scope: root_scope(),
            });
        }
        let a: Artifact = serde_json::from_slice(&bytes)?;
        let expected = match spec.generation.as_str() {
            "3.2" | "3.3" => "paisho-gen3-policy-memory-v1",
            "3.4" => "paisho-gen34-value128-memory-v1",
            "3.5" => "paisho-gen3-value-residual16-memory-v1",
            _ => return Err(invalid("unsupported frozen generation")),
        };
        if a.schema != expected
            || a.rules != crate::gen32::RULES.as_str()
            || !["root", "bonus-nodes", "all-nodes"].contains(&a.memory_scope.as_str())
            || (a.generation != spec.generation
                && !(spec.generation == "3.3" && a.generation == "3.2"))
        {
            return Err(invalid("frozen generation/schema mismatch"));
        }
        let _metadata = (a.parent_sha256, a.updates);
        let policy = a.policy.model()?;
        match (
            &a.memory_manifest,
            &a.memory_manifest_sha256,
            policy.sequence_memory(),
        ) {
            (Some(path), Some(hash), Some(bank)) => {
                let bytes = fs::read(path)?;
                let m: serde_json::Value = serde_json::from_slice(&bytes)?;
                if sha256(&bytes) != *hash
                    || m["rules"] != a.rules
                    || m["sha256"] != bank.spec.sha256
                {
                    return Err(invalid("reference memory manifest mismatch"));
                }
            }
            (None, None, None) => {}
            _ => return Err(invalid("reference memory metadata missing")),
        }
        let extra = match a.value128_extra {
            Some(v) => Some(
                v.try_into()
                    .map_err(|_| invalid("reference value128 length"))?,
            ),
            None => None,
        };
        if let Some(r) = &a.value_residual {
            if r.len() != 2081 || r.iter().any(|v| !v.is_finite()) {
                return Err(invalid("reference residual length/values"));
            }
        }
        if extra
            .as_ref()
            .is_some_and(|v: &[f64; 64]| v.iter().any(|v| !v.is_finite()))
            || (spec.generation == "3.4" || spec.generation == "3.5") && extra.is_none()
            || (spec.generation == "3.5") != a.value_residual.is_some()
        {
            return Err(invalid("reference value structure mismatch"));
        }
        Ok(Self {
            spec: spec.clone(),
            value: a.compact.model()?,
            policy: Some(policy),
            extra,
            residual: a.value_residual,
            scope: a.memory_scope,
        })
    }
    pub fn name(&self) -> String {
        format!("Gen{}", self.spec.generation)
    }
    fn value_at(&self, p: &Position, player: Player) -> f32 {
        let Some(extra) = &self.extra else {
            return self.value.evaluate(p, player);
        };
        match p.outcome() {
            GameOutcome::Win(w) => return if w == player { 1. } else { -1. },
            GameOutcome::Draw => return 0.,
            _ => {}
        }
        let active = self
            .residual
            .as_ref()
            .is_some_and(|r| r[2064..].iter().any(|v| *v != 0.));
        if extra.iter().all(|v| *v == 0.) && !active {
            return self.value.evaluate(p, player);
        }
        let x = micro_state_features(p);
        let raw = self
            .value
            .weights()
            .iter()
            .zip(&x)
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + extra.iter().zip(&x[64..]).map(|(w, x)| w * x).sum::<f64>();
        let correction = if active {
            let r = self.residual.as_ref().unwrap();
            let h: [f64; 16] = std::array::from_fn(|j| {
                (r[2048 + j]
                    + r[j * 128..(j + 1) * 128]
                        .iter()
                        .zip(&x)
                        .map(|(w, x)| w * x)
                        .sum::<f64>())
                .tanh()
            });
            r[2080]
                + h.iter()
                    .zip(&r[2064..2080])
                    .map(|(h, w)| h * w)
                    .sum::<f64>()
        } else {
            0.
        };
        let raw = raw + correction;
        let sign = if player == p.to_move() { 1. } else { -1. };
        (sign * raw / (1. + raw.abs())) as f32
    }
    fn bias(
        &self,
        p: &Position,
        a: &[Action],
        memory: bool,
    ) -> std::result::Result<Option<Vec<f64>>, String> {
        let Some(model) = &self.policy else {
            return Ok(None);
        };
        if a.is_empty() {
            return Ok(None);
        }
        let state = micro_state_features(p);
        let embedding = model.embed(&state);
        let score = |a: &Action| {
            let f = micro_action_features(p, *a);
            (f, MicroModel::logit(&embedding, &f))
        };
        let rows: Vec<_> = if a.len() >= 64 {
            a.par_iter().map(score).collect()
        } else {
            a.iter().map(score).collect()
        };
        let (features, logits): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        let base = micro_softmax(&logits)?;
        let prob = if memory {
            model.memory_priors(&state, &features, &base, 0)?
        } else {
            base
        };
        let logs: Vec<_> = prob.iter().map(|p| p.max(1e-300).ln()).collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        if logs.iter().all(|v| *v == logs[0]) {
            return Ok(None);
        }
        Ok(Some(
            logs.iter().map(|v| ((v - mean) / 2.).tanh()).collect(),
        ))
    }
}
impl MctsEvaluator for Frozen {
    fn ordering_matches_leaf(&self) -> bool {
        true
    }
    fn evaluate(
        &self,
        p: &[Position],
        seat: Player,
        w: HeuristicWeights,
    ) -> std::result::Result<Vec<f32>, String> {
        if self.extra.is_none() {
            return MctsEvaluator::evaluate(&self.value, p, seat, w);
        }
        Ok(if p.len() >= 64 {
            p.par_iter().map(|p| self.value_at(p, seat)).collect()
        } else {
            p.iter().map(|p| self.value_at(p, seat)).collect()
        })
    }
    fn evaluate_leaf(
        &self,
        p: &Position,
        seat: Player,
        _: HeuristicWeights,
    ) -> std::result::Result<f32, String> {
        Ok(self.value_at(p, seat))
    }
    fn policy_bias(
        &self,
        p: &Position,
        a: &[Action],
    ) -> std::result::Result<Option<Vec<f64>>, String> {
        self.bias(
            p,
            a,
            self.scope == "all-nodes"
                || self.scope == "bonus-nodes" && p.phase() == TurnPhase::HarmonyBonus,
        )
    }
    fn root_policy_bias(
        &self,
        p: &Position,
        a: &[Action],
    ) -> std::result::Result<Option<Vec<f64>>, String> {
        if self.policy.as_ref().map_or(true, |m| {
            m.sequence_memory().is_none()
                || m.parameters()[MICRO_RESIDUAL_PARAMETERS..]
                    .iter()
                    .all(|w| *w == 0.)
        }) {
            return Ok(None);
        }
        self.bias(p, a, true)
    }
}
pub(super) fn load_all(specs: &[OpponentSpec]) -> Result<Vec<Arc<Frozen>>> {
    let mut models = vec![];
    for (i, s) in specs.iter().enumerate() {
        if s.generation != format!("3.{}", i + 1) {
            return Err(invalid("references must be ordered 3.1 through 3.5"));
        }
        models.push(Arc::new(Frozen::load(s)?));
    }
    Ok(models)
}
/// Offline parity evidence; never called by training.
pub fn probe(specs: &[OpponentSpec], records: &[GameRecord]) -> Result<serde_json::Value> {
    let models = load_all(specs)?;
    let mut rows = vec![];
    for model in &models {
        let mut positions = vec![];
        for record in records {
            let p = record.replay()?;
            let actions = paisho_core::legal_actions(&p);
            let values = [Player::Host, Player::Guest]
                .iter()
                .map(|s| model.value_at(&p, *s).to_bits())
                .collect::<Vec<_>>();
            let mut searches = vec![];
            for budget in [8, 32] {
                let mut session = MctsSession::new(
                    71,
                    MctsConfig {
                        simulations: budget,
                        ..Default::default()
                    },
                    model.as_ref(),
                )
                .map_err(invalid)?;
                session.set_solver(model.spec.solver);
                let r = session.search_until(&p, &actions, None).map_err(invalid)?;
                searches.push(serde_json::json!({"budget":budget,"selected":actions[r.selected_index].to_string(),"visits":r.actions.iter().map(|a|a.visits).collect::<Vec<_>>(),"means":r.actions.iter().map(|a|a.mean_value().to_bits()).collect::<Vec<_>>()}));
            }
            positions.push(serde_json::json!({"values":values,"bias":model.policy_bias(&p,&actions).map_err(invalid)?,"root_bias":model.root_policy_bias(&p,&actions).map_err(invalid)?,"searches":searches}));
        }
        rows.push(serde_json::json!({"generation":model.spec.generation,"sha256":model.spec.sha256,"positions":positions,"memory":model.policy.as_ref().and_then(|p|p.sequence_memory()).map(|b|&b.spec)}));
    }
    Ok(serde_json::json!({"models":rows}))
}
