//! Bounded resident choices acquired from fully consumed regulatory proofs.
//! This registry is an additional finite choice check,
//! never a substitute for the publication Guard or permission to publish.
use super::*;
use std::collections::{BTreeSet, VecDeque};
mod reads;

pub(super) const CAPACITY: usize = 64;
pub(super) const ADMISSIONS_PER_PUBLICATION: usize = 8;
const SCHEMA: &str = "paisho-gen5-learned-raw-choices-v1";

/// Owned proof material from the native, fully consumed receipt. The collector
/// and actual decision actor remain distinct, including reference opponents.
/// The caller supplies this only after its native provenance checks and SGD.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ConsumedProof {
    pub prefix: String,
    pub certificate: MicroProofCertificate,
    pub source_run: String,
    pub game_id: u64,
    pub decision: usize,
    pub collector: String,
    pub decision_actor: String,
    pub player: String,
    pub updates_consumed: u64,
}

impl ConsumedProof {
    /// Bind the exact Saved target emitted by collector::targets to the native
    /// game's certificate. Called after fully_learned and SGD, while both remain
    /// in RAM. Played alone cannot recover a reference actor's model hash: its
    /// samples were drained and its opponent field contains only the generation.
    /// TargetEvidence.actor is the canonical identity produced by the collector;
    /// do not reconstruct it from candidate_seat, lane or reanalysis heuristics.
    /// A measurement/evaluation label does not imply that its targets went
    /// unlearned. The caller must establish actual complete SGD consumption.
    pub fn from_played_target(
        game: &collector::Played,
        saved: &SavedMicroExample,
        decision: usize,
        certificate: &MicroProofCertificate,
        source_run: &str,
        updates_consumed: u64,
    ) -> Result<Option<Self>> {
        if game.error.is_some() || !game.loop_repair || game.record.rules() != RULES
            || decision <= game.prefix_decisions || decision > game.record.actions().len() + 1
            || saved.decision != decision || saved.game_id != game.id.to_string()
            || saved.rules != RULES.as_str() || source_run.is_empty()
            || saved.source_run != source_run || saved.collector.is_empty()
            || saved.collector != game.snapshot.identity
            || updates_consumed == 0
        {
            return Err(invalid("learned proof does not match its consumed native target"));
        }
        let matching = game.certificates.iter().filter(|(d, _)| *d == decision)
            .collect::<Vec<_>>();
        if matching.len() != 1 || &matching[0].1 != certificate {
            return Err(invalid("learned proof differs from the native decision certificate"));
        }
        let prefix = cases::prefix(&game.record, decision - 1);
        let position = prefix.replay()?;
        let outcome = certificate.verify(&position).map_err(invalid)?;
        if outcome != GameOutcome::Win(position.to_move()) { return Ok(None); }
        if position.outcome() != GameOutcome::Ongoing {
            return Err(invalid("learned native proof starts at a terminal position"));
        }
        let evidence = saved.evidence.as_ref()
            .ok_or_else(|| invalid("learned native proof lacks decision actor evidence"))?;
        evidence.validate()?;
        let legal = paisho_core::legal_actions(&position);
        if evidence.player != position.to_move().code().to_string()
            || !["verified-search", "verified-regulatory-win"].contains(&evidence.policy_source.as_str())
            || !saved.policy_weight.is_finite() || saved.policy_weight <= 0.
            || !saved.tactical.as_ref().is_some_and(|t| t.root_value == Some(1))
            || saved.actions != legal.iter().map(ToString::to_string).collect::<Vec<_>>()
            || saved.state != game.snapshot.model.state_features(&position)
            || saved.action_features != legal.iter()
                .map(|a| micro_action_features(&position, *a).to_vec()).collect::<Vec<_>>()
        {
            return Err(invalid("learned native proof/target player or position differs"));
        }
        Ok(Some(Self {
            prefix: prefix.to_string(), certificate: certificate.clone(),
            source_run: saved.source_run.clone(),
            game_id: u64::try_from(game.id).map_err(|_| invalid("native game id exceeds u64"))?,
            decision, collector: saved.collector.clone(), decision_actor: evidence.actor.clone(),
            player: evidence.player.clone(), updates_consumed,
        }))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Acquisition {
    actor: String,
    publication: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Stored {
    key: String,
    proof_sha256: String,
    support_sha256: String,
    consumed_order: u64,
    proof: ConsumedProof,
    acquisition: Option<Acquisition>,
}

#[derive(Clone)]
struct Resident {
    stored: Stored,
    row: Arc<Witness>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct State {
    schema: String,
    feature_schema: String,
    last_validated_actor: String,
    publications: u64,
    next_consumed_order: u64,
    pending_evictions: u64,
    active_retirements: u64,
    pending: Vec<Stored>,
    active: Vec<Stored>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub(super) struct Admission {
    pub key: String,
    pub inserted: bool,
    pub evicted_pending: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Acquired {
    pub key: String,
    pub proof_sha256: String,
    pub support_sha256: String,
    pub acquired_actor: String,
    pub publication: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Commit {
    pub actor: String,
    pub publication: u64,
    pub acquired: Vec<Acquired>,
    pub retired: Vec<Acquired>,
    pub active_reads: usize,
    pub pending_reads: usize,
}

#[derive(Clone)]
pub(super) struct Registry {
    feature_schema: String,
    last_validated_actor: String,
    publications: u64,
    next_consumed_order: u64,
    pending_evictions: u64,
    active_retirements: u64,
    pending: VecDeque<Resident>,
    // Ordered by acquisition, not by the age of its source receipt.
    active: VecDeque<Resident>,
    // Ephemeral only: neither cache nor dispatch affects the checkpoint schema.
    reads: reads::Reads,
}

fn verified(proof: ConsumedProof, model: &MicroModel, order: u64) -> Result<Resident> {
    let record: GameRecord = proof.prefix.parse()?;
    let position = record.replay()?;
    if record.rules() != RULES || position.outcome() != GameOutcome::Ongoing
        || proof.decision != record.actions().len() + 1
        || proof.player != position.to_move().code().to_string()
        || proof.source_run.is_empty() || proof.collector.is_empty()
        || proof.decision_actor.is_empty() || proof.updates_consumed == 0
    {
        return Err(invalid("learned choice lacks consumed native proof provenance"));
    }
    let legal = paisho_core::legal_actions(&position);
    // This verifies the exact certificate and adds all other immediate wins.
    let policy = action_values::verified_winning_policy(&position, &legal, &proof.certificate)?;
    let valid = policy.iter().map(|p| *p > 0.).collect::<Vec<_>>();
    let support = legal.iter().zip(&valid).filter(|(_, v)| **v)
        .map(|(action, _)| action.to_string()).collect::<Vec<_>>();
    let stored = Stored {
        key: sha256(proof.prefix.as_bytes()),
        proof_sha256: sha256(&serde_json::to_vec(&proof.certificate)?),
        support_sha256: sha256(&serde_json::to_vec(&support)?),
        consumed_order: order,
        proof,
        acquisition: None,
    };
    let example = Arc::new(MicroExample {
        state: model.state_features(&position),
        actions: legal.iter().map(|a| micro_action_features(&position, *a)).collect(),
        policy,
        value: 1.,
        policy_weight: 1.,
        value_weight: 0.,
        action_values: vec![],
        sequence_source: 0,
        policy_support: true,
    });
    Ok(Resident {
        stored,
        row: Arc::new(Witness { position, valid, example, successors: Default::default() }),
    })
}

fn raw_wins(model: &MicroModel, row: &Witness) -> Result<bool> {
    let base = micro_softmax(&MicroModel::logits(
        &model.embed(&row.example.state), &row.example.actions,
    )).map_err(invalid)?;
    let p = model.memory_priors(&row.example.state, &row.example.actions, &base, 0)
        .map_err(invalid)?;
    if p.len() != row.valid.len() || p.is_empty()
        || p.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(invalid("invalid learned-choice policy/support"));
    }
    let best = (0..p.len()).max_by(|&a, &b| p[a].total_cmp(&p[b]).then_with(|| b.cmp(&a)))
        .ok_or_else(|| invalid("empty learned-choice policy"))?;
    Ok(row.valid[best])
}

fn acquired(entry: &Stored) -> Result<Acquired> {
    let a = entry.acquisition.as_ref().ok_or_else(|| invalid("unacquired retention entry"))?;
    Ok(Acquired { key: entry.key.clone(), proof_sha256: entry.proof_sha256.clone(),
        support_sha256: entry.support_sha256.clone(), acquired_actor: a.actor.clone(),
        publication: a.publication })
}

impl Registry {
    pub fn new(actor: String, model: &MicroModel) -> Result<Self> {
        if actor.is_empty() { return Err(invalid("learned choices require an accepted actor")); }
        Ok(Self { feature_schema: model.feature_schema().into(), last_validated_actor: actor,
            publications: 0, next_consumed_order: 0, pending_evictions: 0,
            active_retirements: 0, pending: VecDeque::new(), active: VecDeque::new(),
            reads: Default::default() })
    }

    fn check_model(&self, model: &MicroModel) -> Result<()> {
        if self.feature_schema != model.feature_schema() {
            return Err(invalid("learned-choice feature schema changed"));
        }
        Ok(())
    }

    /// Called after the receipt's full fresh batch has actually been consumed.
    /// A duplicate retains its first exact proof, provenance and chronological
    /// position. Its ignored incoming certificate is not replayed or verified:
    /// none of its fields can replace the already verified resident witness.
    /// Pending overflow rotates only pending entries, never obligations.
    pub fn admit_consumed(&mut self, proof: ConsumedProof, fully_learned: bool,
        model: &MicroModel) -> Result<Admission>
    {
        self.check_model(model)?;
        if !fully_learned { return Err(invalid("unconsumed proof cannot enter learned choices")); }
        let key = sha256(proof.prefix.as_bytes());
        if self.active.iter().chain(&self.pending).any(|r| r.stored.key == key) {
            return Ok(Admission { key, inserted: false, evicted_pending: None });
        }
        let resident = verified(proof, model, self.next_consumed_order)?;
        let next_order = self.next_consumed_order.checked_add(1)
            .ok_or_else(|| invalid("learned-choice order overflow"))?;
        let evictions = self.pending_evictions.checked_add(u64::from(self.pending.len() == CAPACITY))
            .ok_or_else(|| invalid("pending eviction counter overflow"))?;
        let evicted_pending = if self.pending.len() == CAPACITY {
            self.pending.pop_front().map(|r| r.stored.key)
        } else { None };
        self.pending.push_back(resident);
        self.next_consumed_order = next_order;
        self.pending_evictions = evictions;
        Ok(Admission { key, inserted: true, evicted_pending })
    }

    /// Immutable source order. External row-indexed caches must be replaced
    /// when this population changes; the private raw cache keys the exact Arcs.
    pub fn pending_rows(&self) -> Vec<(String, Arc<Witness>)> {
        self.pending.iter().map(|r| (r.stored.key.clone(), r.row.clone())).collect()
    }
    pub fn active_rows(&self) -> Vec<(String, Arc<Witness>)> {
        self.active.iter().map(|r| (r.stored.key.clone(), r.row.clone())).collect()
    }
    pub fn last_validated_actor(&self) -> &str { &self.last_validated_actor }
    pub fn progress(&self) -> serde_json::Value {
        serde_json::json!({"schema":SCHEMA,"last_validated_actor":self.last_validated_actor,
            "publications":self.publications,"pending":self.pending.len(),"active":self.active.len(),
            "pending_evictions":self.pending_evictions,"active_retirements":self.active_retirements,
            "maximum_admissions_per_publication":ADMISSIONS_PER_PUBLICATION})
    }

    /// Empty-cache switch for runtime wiring and isolated diagnostic arms.
    /// Each arm gets an empty cache. Later transactional clones may share it:
    /// the full model/storage and ordered witness identity are always checked.
    pub fn configure_read_optimization(&mut self, enabled: bool, parallel: Option<&cpu::Ordered>) {
        self.reads = reads::Reads::new(enabled, parallel);
    }

    pub fn read_optimization_progress(&self) -> serde_json::Value {
        self.reads.progress()
    }

    /// Read-only finite obligations; no failed candidate can clear its own loss.
    pub fn retention_losses(&self, model: &MicroModel) -> Result<Vec<String>> {
        self.check_model(model)?;
        if self.reads.enabled() {
            let rows = self.active.iter().map(|r| r.row.clone()).collect::<Vec<_>>();
            let wins = self.reads.evaluate(model, rows)?;
            return Ok(self.active.iter().zip(wins.iter())
                .filter(|(_, won)| !**won).map(|(r, _)| r.stored.key.clone()).collect());
        }
        self.active.iter().filter_map(|r| match raw_wins(model, &r.row) {
            Ok(true) => None, Ok(false) => Some(Ok(r.stored.key.clone())),
            Err(e) => Some(Err(e)),
        }).collect()
    }

    /// Invoke only for the FINAL candidate that passes the unchanged Guard.
    /// Re-read all obligations, then admit up to eight actual pending raw wins.
    /// A successfully relayed key has priority; the remaining order is FIFO.
    /// Natural gains require no relay to become protected. All reads complete
    /// before mutation. Commit a cloned registry with the actor transaction, or
    /// discard it if actor/checkpoint persistence fails.
    pub fn commit_validated(&mut self, actor: String, model: &MicroModel,
        relayed_key: Option<&str>) -> Result<Commit>
    {
        if actor.is_empty() { return Err(invalid("empty validated actor identity")); }
        let losses = self.retention_losses(model)?;
        if !losses.is_empty() {
            return Err(invalid(format!("active learned raw choices lost: {losses:?}")));
        }
        let mut order = (0..self.pending.len()).collect::<Vec<_>>();
        if let Some(key) = relayed_key {
            let index = self.pending.iter().position(|r| r.stored.key == key)
                .ok_or_else(|| invalid("relayed acquisition is not pending"))?;
            order.retain(|i| *i != index);
            order.insert(0, index);
        }
        let mut selected = vec![];
        let mut reads = 0;
        for i in order {
            let won = raw_wins(model, &self.pending[i].row)?;
            reads += 1;
            if !won && relayed_key == Some(self.pending[i].stored.key.as_str()) {
                return Err(invalid("relayed acquisition is not a real raw winning choice"));
            }
            if won { selected.push(i); }
            if selected.len() == ADMISSIONS_PER_PUBLICATION { break; }
        }
        self.commit_measured(actor, &selected, reads)
    }

    // State mutation is separate from finite reads so no later error can retire
    // an obligation. Tests may exercise capacity bookkeeping without inference.
    fn commit_measured(&mut self, actor: String, selected: &[usize], reads: usize) -> Result<Commit> {
        if actor.is_empty() || selected.len() > ADMISSIONS_PER_PUBLICATION
            || selected.iter().any(|i| *i >= self.pending.len())
            || selected.iter().copied().collect::<BTreeSet<_>>().len() != selected.len()
        { return Err(invalid("invalid learned acquisition selection")); }
        let publication = self.publications.checked_add(1)
            .ok_or_else(|| invalid("learned-choice publication counter overflow"))?;
        let retirement_count = (self.active.len() + selected.len()).saturating_sub(CAPACITY);
        let total_retired = self.active_retirements.checked_add(retirement_count as u64)
            .ok_or_else(|| invalid("learned retirement counter overflow"))?;
        let retired = self.active.iter().take(retirement_count)
            .map(|r| acquired(&r.stored)).collect::<Result<Vec<_>>>()?;
        let mut additions = selected.iter().map(|&i| {
            let mut r = self.pending[i].clone();
            r.stored.acquisition = Some(Acquisition { actor: actor.clone(), publication });
            r
        }).collect::<Vec<_>>();
        let acquisitions = additions.iter().map(|r| acquired(&r.stored)).collect::<Result<Vec<_>>>()?;
        let active_reads = self.active.len();
        for _ in 0..retirement_count { self.active.pop_front(); }
        for r in additions.drain(..) { self.active.push_back(r); }
        let selected_orders = selected.iter().map(|&i| self.pending[i].stored.consumed_order)
            .collect::<BTreeSet<_>>();
        self.pending.retain(|r| !selected_orders.contains(&r.stored.consumed_order));
        self.last_validated_actor = actor.clone();
        self.publications = publication;
        self.active_retirements = total_retired;
        Ok(Commit { actor, publication, acquired: acquisitions, retired, active_reads, pending_reads: reads })
    }

    pub fn checkpoint(&self) -> Result<serde_json::Value> {
        let state = State { schema: SCHEMA.into(), feature_schema: self.feature_schema.clone(),
            last_validated_actor: self.last_validated_actor.clone(), publications: self.publications,
            next_consumed_order: self.next_consumed_order, pending_evictions: self.pending_evictions,
            active_retirements: self.active_retirements,
            pending: self.pending.iter().map(|r| r.stored.clone()).collect(),
            active: self.active.iter().map(|r| r.stored.clone()).collect() };
        let payload = serde_json::to_value(state)?;
        Ok(serde_json::json!({"sha256":sha256(&serde_json::to_vec(&payload)?),"payload":payload}))
    }

    /// Certificates/support are restored from their EXACT checkpoint material,
    /// never from a prefix-addressed proof file that may contain an older proof.
    pub fn restore(saved: &serde_json::Value, actor: &str, model: &MicroModel) -> Result<Self> {
        let payload = &saved["payload"];
        if saved["sha256"].as_str() != Some(sha256(&serde_json::to_vec(payload)?).as_str()) {
            return Err(invalid("learned-choice checkpoint digest mismatch"));
        }
        let state: State = serde_json::from_value(payload.clone())?;
        if state.schema != SCHEMA || state.last_validated_actor != actor || actor.is_empty()
            || state.feature_schema != model.feature_schema()
            || state.pending.len() > CAPACITY || state.active.len() > CAPACITY
        { return Err(invalid("learned-choice checkpoint contract changed")); }
        let mut keys = BTreeSet::new();
        let mut orders = BTreeSet::new();
        for (is_active, rows) in [(false, &state.pending), (true, &state.active)] {
            let mut previous_order = None;
            let mut previous_publication = 0;
            for row in rows {
                if !keys.insert(row.key.clone()) || !orders.insert(row.consumed_order)
                    || row.consumed_order >= state.next_consumed_order
                    || row.acquisition.is_some() != is_active
                { return Err(invalid("duplicate or unordered learned-choice entry")); }
                if let Some(a) = &row.acquisition {
                    if a.actor.is_empty() || a.publication == 0 || a.publication > state.publications
                        || a.publication < previous_publication
                    { return Err(invalid("invalid historical acquisition")); }
                    previous_publication = a.publication;
                } else if previous_order.is_some_and(|old| row.consumed_order <= old) {
                    return Err(invalid("pending learned choices changed FIFO order"));
                }
                previous_order = Some(row.consumed_order);
            }
        }
        let reload = |stored: Stored| -> Result<Resident> {
            let mut r = verified(stored.proof.clone(), model, stored.consumed_order)?;
            if r.stored.key != stored.key || r.stored.proof_sha256 != stored.proof_sha256
                || r.stored.support_sha256 != stored.support_sha256
            { return Err(invalid("learned-choice proof or winning support changed")); }
            r.stored = stored;
            Ok(r)
        };
        let out = Self { feature_schema: state.feature_schema, last_validated_actor: state.last_validated_actor,
            publications: state.publications, next_consumed_order: state.next_consumed_order,
            pending_evictions: state.pending_evictions, active_retirements: state.active_retirements,
            pending: state.pending.into_iter().map(&reload).collect::<Result<VecDeque<_>>>()?,
            active: state.active.into_iter().map(reload).collect::<Result<VecDeque<_>>>()?,
            reads: Default::default() };
        if !out.retention_losses(model)?.is_empty() {
            return Err(invalid("restored actor does not retain its learned choices"));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn fixture() -> (MicroModel, ConsumedProof) {
        let prefix = include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/gen5-multiple-proved-wins.psr")).to_owned();
        let record: GameRecord = prefix.parse().unwrap();
        let p = record.replay().unwrap();
        let legal = paisho_core::legal_actions(&p);
        let action = legal.into_iter().find(|a| {
            let mut next = p.clone(); next.apply(*a).unwrap();
            next.outcome() == GameOutcome::Win(p.to_move())
        }).unwrap();
        let outcome = if p.to_move() == Player::Host { 1 } else { -1 };
        (MicroModel::seeded(17).with_spatial_policy(), ConsumedProof {
            prefix, certificate: MicroProofCertificate { outcome, children: vec![
                (action.to_string(), MicroProofCertificate { outcome, children: vec![] })] },
            source_run: "native-fixture".into(), game_id: 1, decision: record.actions().len() + 1,
            collector: "v5-collector".into(), decision_actor: "Gen3.3:reference".into(),
            player: p.to_move().code().into(), updates_consumed: 5,
        })
    }

    fn native_fixture() -> (collector::Played, SavedMicroExample, MicroProofCertificate) {
        let (model, proof) = fixture();
        let record: GameRecord = proof.prefix.parse().unwrap();
        let position = record.replay().unwrap();
        let legal = paisho_core::legal_actions(&position);
        let snapshot = Arc::new(Snapshot { artifact: None, version: 1,
            identity: "a".repeat(64), model: Arc::new(model), path: "unused".into() });
        let pool = cpu::Executor::direct(cpu::build_pool(1, None).unwrap().0);
        // Zero search decisions: construct the native episode metadata without
        // playing/training. The following Saved is the exact fixture target.
        let mut game = collector::play_from(31, snapshot.clone(), snapshot, None,
            &Options { decision_limit: 0, learning_loop_repair: true, ..Default::default() },
            paisho_platform::training_time::now(), &pool, "native-fixture", Some(&record), false, None);
        game.certificates = vec![(proof.decision, proof.certificate.clone())];
        let mut saved = super::super::super::super::tactics::fixture();
        saved.source_run = "native-fixture".into();
        saved.game_id = game.id.to_string();
        saved.decision = proof.decision;
        saved.collector = game.snapshot.identity.clone();
        saved.actions = legal.iter().map(ToString::to_string).collect();
        saved.state = game.snapshot.model.state_features(&position);
        saved.action_features = legal.iter().map(|a| micro_action_features(&position, *a).to_vec()).collect();
        saved.policy = action_values::verified_winning_policy(&position, &legal, &proof.certificate).unwrap();
        saved.new_visits.clear(); saved.policy_raw_visits.clear(); saved.policy_pruned_visits.clear();
        saved.tactical.as_mut().unwrap().action_values = saved.policy.iter()
            .map(|p| (*p > 0.).then_some(1)).collect();
        saved.evidence = Some(TargetEvidence { policy_support: true, policy_coordinates: String::new(),
            search_prior: vec![], coupling_strength: None, observed_value: None, observed_psr: None,
            estimated_value: None, value_weight: 1., policy_source: "verified-regulatory-win".into(),
            completed_action_values: vec![], action_value_visits: vec![], target_prior: vec![],
            excluded_actions: vec![], player: position.to_move().code().to_string(),
            actor: format!("Gen3.3:{}", "b".repeat(64)) });
        (game, saved, proof.certificate)
    }

    #[test]
    fn learned_native_binder_preserves_recorded_reference_and_reanalysis_actors() {
        let (mut game, mut saved, certificate) = native_fixture();
        let player = game.record.replay().unwrap().to_move();
        game.lane = Lane::Historical;
        game.opponent = "Gen3.3".into();
        game.reference_budget = Some(8);
        game.candidate_seat = player.opponent();
        let reference = ConsumedProof::from_played_target(&game, &saved, saved.decision,
            &certificate, "native-fixture", 11).unwrap().unwrap();
        assert_eq!(reference.collector, game.snapshot.identity);
        assert_eq!(reference.decision_actor, saved.evidence.as_ref().unwrap().actor);
        assert_ne!(reference.decision_actor, reference.collector);
        assert_eq!(reference.game_id, 31);
        assert_eq!(reference.player, player.code().to_string());
        assert_eq!(reference.updates_consumed, 11);

        game.lane = Lane::Reanalysis;
        game.reanalysis = true;
        game.opponent = game.snapshot.identity.clone();
        game.reference_budget = None;
        saved.evidence.as_mut().unwrap().actor = game.snapshot.identity.clone();
        saved.evidence.as_mut().unwrap().policy_source = "verified-search".into();
        // The seat is deliberately unchanged: the recorded native actor governs
        // this reanalysis, not a fresh opponent/player heuristic in the binder.
        let reanalysis = ConsumedProof::from_played_target(&game, &saved, saved.decision,
            &certificate, "native-fixture", 12).unwrap().unwrap();
        assert_eq!(reanalysis.decision_actor, game.snapshot.identity);
        assert_eq!(reanalysis.certificate, certificate);
    }

    #[test]
    fn learned_native_binder_rejects_mismatched_receipt_player_and_certificate() {
        let (game, saved, certificate) = native_fixture();
        let bind = |s: &SavedMicroExample, c: &MicroProofCertificate| {
            ConsumedProof::from_played_target(&game, s, saved.decision, c, "native-fixture", 11)
        };
        let mut changed = saved.clone(); changed.game_id = "different".into();
        assert!(bind(&changed, &certificate).is_err());
        changed = saved.clone(); changed.source_run = "other-run".into();
        assert!(bind(&changed, &certificate).is_err());
        changed = saved.clone(); changed.collector = "c".repeat(64);
        assert!(bind(&changed, &certificate).is_err());
        changed = saved.clone(); changed.evidence.as_mut().unwrap().player = "other-player".into();
        assert!(bind(&changed, &certificate).is_err());
        changed = saved.clone(); changed.state[0] += 0.01;
        assert!(bind(&changed, &certificate).is_err());
        let mut wrong_certificate = certificate.clone(); wrong_certificate.outcome = 0;
        assert!(bind(&saved, &wrong_certificate).is_err());
    }

    #[test]
    fn learned_native_binder_accepts_consumed_measurement_and_evaluation_targets() {
        let (mut game, saved, certificate) = native_fixture();
        game.measurement = true;
        game.evaluation = Some(evaluation::Attempt {
            reference: "Gen3.3".into(), budget: 8,
            model: game.snapshot.identity.clone(), version: game.snapshot.version,
            lot: 0, slot: 0, prefix: "consumed-fixture".into(),
        });
        let proof = ConsumedProof::from_played_target(
            &game, &saved, saved.decision, &certificate, "native-fixture", 11,
        ).unwrap().unwrap();
        assert_eq!(proof.decision_actor, saved.evidence.as_ref().unwrap().actor);
        assert_eq!(proof.certificate, certificate);
        // Binding is not admission: a caller without complete consumption still
        // cannot put this regulatory proof in the learned-choice catalogue.
        let mut registry = Registry::new(game.snapshot.identity.clone(), &game.snapshot.model).unwrap();
        assert!(registry.admit_consumed(proof.clone(), false, &game.snapshot.model).is_err());
        assert!(registry.admit_consumed(proof, true, &game.snapshot.model).unwrap().inserted);
    }

    #[test]
    fn learned_choices_require_consumption_and_expand_regulatory_support() {
        let (model, proof) = fixture();
        let mut r = Registry::new("actor-a".into(), &model).unwrap();
        let unchanged = r.checkpoint().unwrap();
        assert!(r.admit_consumed(proof.clone(), false, &model).is_err());
        assert_eq!(r.checkpoint().unwrap(), unchanged);
        let added = r.admit_consumed(proof.clone(), true, &model).unwrap();
        assert!(added.inserted);
        let row = &r.pending[0];
        assert_eq!(row.row.valid.iter().filter(|v| **v).count(), 4);
        assert_eq!(row.row.example.sequence_source, 0);
        assert!(row.row.example.policy_support);
        assert!(row.row.example.action_values.is_empty());
        assert_eq!(row.stored.proof.collector, "v5-collector");
        assert_eq!(row.stored.proof.decision_actor, "Gen3.3:reference");
        let before_duplicate = r.checkpoint().unwrap();
        let mut ignored_duplicate = proof;
        ignored_duplicate.certificate.outcome = 0;
        ignored_duplicate.decision_actor = "ignored-new-actor".into();
        assert!(!r.admit_consumed(ignored_duplicate, true, &model).unwrap().inserted);
        assert_eq!(r.checkpoint().unwrap(), before_duplicate);
    }

    #[test]
    fn learned_choices_restore_exact_embedded_proof_and_pending_order() {
        let (model, proof) = fixture();
        let mut r = Registry::new("actor-a".into(), &model).unwrap();
        r.admit_consumed(proof, true, &model).unwrap();
        let saved = r.checkpoint().unwrap();
        let restored = Registry::restore(&saved, "actor-a", &model).unwrap();
        assert_eq!(restored.checkpoint().unwrap(), saved);
        assert!(Registry::restore(&saved, "actor-b", &model).is_err());
        let mut changed = saved.clone();
        changed["payload"]["pending"][0]["proof"]["certificate"]["outcome"] = 0.into();
        assert!(Registry::restore(&changed, "actor-a", &model).is_err());
        // Even a newly computed envelope cannot make an invalid proof valid.
        changed["sha256"] = sha256(&serde_json::to_vec(&changed["payload"]).unwrap()).into();
        assert!(Registry::restore(&changed, "actor-a", &model).is_err());
    }

    // Synthetic identities below isolate registry bookkeeping; they deliberately
    // bypass proof ingestion and are never used as native proof/strength evidence.
    fn fill_pending(r: &mut Registry, template: &Resident, start: u64, n: usize) {
        for id in start..start + n as u64 {
            let mut row = template.clone();
            row.stored.key = format!("synthetic-{id}");
            row.stored.consumed_order = id;
            row.stored.acquisition = None;
            r.pending.push_back(row);
        }
        r.next_consumed_order = start + n as u64;
    }

    #[test]
    fn learned_choice_retirement_requires_a_new_acquisition_and_preserves_history() {
        let (model, proof) = fixture();
        let template = verified(proof, &model, 0).unwrap();
        let mut r = Registry::new("initial".into(), &model).unwrap();
        for batch in 0..8 {
            fill_pending(&mut r, &template, batch * 8, 8);
            let c = r.commit_measured(format!("actor-{batch}"), &(0..8).collect::<Vec<_>>(), 8).unwrap();
            assert!(c.retired.is_empty());
        }
        assert_eq!(r.active.len(), CAPACITY);
        let first = r.active[0].stored.key.clone();
        let historical = r.active[0].stored.acquisition.as_ref().unwrap().actor.clone();
        let c = r.commit_measured("no-new-acquisition".into(), &[], 0).unwrap();
        assert!(c.retired.is_empty());
        assert_eq!(r.active[0].stored.key, first);
        assert_eq!(r.active[0].stored.acquisition.as_ref().unwrap().actor, historical);
        assert_eq!(r.last_validated_actor(), "no-new-acquisition");
        fill_pending(&mut r, &template, 64, 9);
        let before = r.checkpoint().unwrap();
        assert!(r.commit_measured("too-many".into(), &(0..9).collect::<Vec<_>>(), 9).is_err());
        assert_eq!(r.checkpoint().unwrap(), before);
        let c = r.commit_measured("next".into(), &[8, 0], 9).unwrap();
        assert_eq!(c.retired.len(), 2);
        assert_eq!(c.retired[0].key, first);
        assert_eq!(c.retired[0].acquired_actor, historical);
        assert_eq!(c.acquired[0].key, "synthetic-72");
        assert_eq!(c.acquired[1].key, "synthetic-64");
        assert_eq!(r.active.len(), CAPACITY);
        assert_eq!(r.pending.len(), 7);
        assert_eq!(r.active_retirements, 2);
    }

    #[test]
    fn learned_choices_pending_overflow_rotates_without_touching_obligations() {
        let (model, proof) = fixture();
        let template = verified(proof.clone(), &model, 0).unwrap();
        let mut r = Registry::new("initial".into(), &model).unwrap();
        fill_pending(&mut r, &template, 0, 1);
        r.commit_measured("acquired".into(), &[0], 1).unwrap();
        fill_pending(&mut r, &template, 1, CAPACITY);
        let added = r.admit_consumed(proof, true, &model).unwrap();
        assert_eq!(added.evicted_pending.as_deref(), Some("synthetic-1"));
        assert_eq!(r.pending.len(), CAPACITY);
        assert_eq!(r.pending_evictions, 1);
        assert_eq!(r.active.len(), 1);
        assert_eq!(r.active[0].stored.key, "synthetic-0");
        assert_eq!(r.active_retirements, 0);
    }

    #[test]
    fn learned_choices_commit_uses_real_policy_and_errors_leave_state_unchanged() {
        let (model, proof) = fixture();
        let mut r = Registry::new("initial".into(), &model).unwrap();
        r.admit_consumed(proof, true, &model).unwrap();
        let won = raw_wins(&model, &r.pending[0].row).unwrap();
        let before = r.checkpoint().unwrap();
        assert!(r.commit_validated("bad".into(), &model, Some("unknown-key")).is_err());
        assert_eq!(r.checkpoint().unwrap(), before);
        let c = r.commit_validated("next".into(), &model, None).unwrap();
        assert_eq!(c.acquired.len(), usize::from(won));
        assert_eq!(c.pending_reads, 1);
        assert_eq!(r.active.len(), usize::from(won));
        let saved = r.checkpoint().unwrap();
        let restored = Registry::restore(&saved, "next", &model).unwrap();
        assert_eq!(restored.checkpoint().unwrap(), saved);
    }

    // Artificial supports isolate read ordering/cache bookkeeping, not native
    // proof validity. Production residents can only enter through verified().
    fn forced_row(template: &Resident, key: &str, won: bool) -> Resident {
        let mut row = template.clone();
        row.stored.key = key.into();
        row.row = Arc::new(Witness { position: template.row.position.clone(),
            valid: vec![won; template.row.example.actions.len()],
            example: template.row.example.clone(), successors: Default::default() });
        row
    }

    #[test]
    fn learned_raw_cache_parallel_reads_and_forked_populations_preserve_order() {
        let (model, proof) = fixture();
        let template = verified(proof.clone(), &model, 0).unwrap();
        let mut r = Registry::new("initial".into(), &model).unwrap();
        r.active.extend([forced_row(&template, "first", false),
            forced_row(&template, "middle", true), forced_row(&template, "last", false)]);
        let expected = r.retention_losses(&model).unwrap();
        assert_eq!(expected, vec!["first", "last"]);
        let checkpoint = r.checkpoint().unwrap();
        let pool = cpu::Ordered::new(&[cpu::build_pool(2, None).unwrap().0]);
        r.configure_read_optimization(true, Some(&pool));
        assert_eq!(r.retention_losses(&model).unwrap(), expected);
        assert_eq!(r.checkpoint().unwrap(), checkpoint);
        let mut fork = r.clone();
        assert_eq!(fork.retention_losses(&model).unwrap(), expected);
        assert_eq!(r.read_optimization_progress()["hits"], 1);
        fork.active.rotate_left(1);
        assert_eq!(fork.retention_losses(&model).unwrap(), vec!["last", "first"]);
        assert_eq!(r.retention_losses(&model).unwrap(), expected);
        assert_eq!(r.read_optimization_progress()["misses"], 2);
        // Pending admission does not change the active population or stale-hit it.
        fork.admit_consumed(proof, true, &model).unwrap();
        let misses = fork.read_optimization_progress()["misses"].clone();
        assert_eq!(fork.retention_losses(&model).unwrap(), vec!["last", "first"]);
        assert_eq!(fork.read_optimization_progress()["misses"], misses);
        fork.commit_measured("fork-next".into(), &[0], 1).unwrap();
        fork.retention_losses(&model).unwrap();
        assert_eq!(fork.read_optimization_progress()["misses"], 3);
        assert_eq!(r.retention_losses(&model).unwrap(), expected);
    }

    #[test]
    fn learned_raw_cache_restore_and_diagnostic_arms_start_empty() {
        let (model, proof) = fixture();
        let mut r = Registry::new("initial".into(), &model).unwrap();
        r.admit_consumed(proof, true, &model).unwrap();
        let saved = r.checkpoint().unwrap();
        r.configure_read_optimization(true, None);
        r.retention_losses(&model).unwrap();
        r.retention_losses(&model).unwrap();
        assert_eq!(r.read_optimization_progress()["hits"], 1);
        let original_arm = r.clone();
        r.configure_read_optimization(true, None);
        assert_eq!(r.read_optimization_progress()["entries"], 0);
        assert_eq!(original_arm.read_optimization_progress()["hits"], 1);
        let restored = Registry::restore(&saved, "initial", &model).unwrap();
        assert_eq!(restored.read_optimization_progress()["enabled"], false);
        assert_eq!(restored.read_optimization_progress()["entries"], 0);
        assert_eq!(restored.checkpoint().unwrap(), saved);
        r.configure_read_optimization(false, None);
        assert_eq!(r.read_optimization_progress()["enabled"], false);
        assert_eq!(r.checkpoint().unwrap(), saved);
    }

    #[test]
    fn learned_raw_cache_does_not_evaluate_pending_after_eighth_success() {
        let (model, proof) = fixture();
        let template = verified(proof, &model, 0).unwrap();
        let winner = forced_row(&template, "unused", true);
        let mut baseline = Registry::new("initial".into(), &model).unwrap();
        fill_pending(&mut baseline, &winner, 0, ADMISSIONS_PER_PUBLICATION + 1);
        // This malformed final row would error if a new parallel path greedily
        // read every pending entry. The native commit never reaches it.
        let last = baseline.pending.back_mut().unwrap();
        last.row = Arc::new(Witness { position: last.row.position.clone(), valid: vec![],
            example: last.row.example.clone(), successors: Default::default() });
        let mut optimized = baseline.clone();
        let pool = cpu::Ordered::new(&[cpu::build_pool(2, None).unwrap().0]);
        optimized.configure_read_optimization(true, Some(&pool));
        let old = baseline.commit_validated("next".into(), &model, None).unwrap();
        let new = optimized.commit_validated("next".into(), &model, None).unwrap();
        assert_eq!(serde_json::to_value(old).unwrap(), serde_json::to_value(new).unwrap());
        assert_eq!(baseline.checkpoint().unwrap(), optimized.checkpoint().unwrap());
        assert_eq!(optimized.pending.len(), 1);
    }
}
