use super::*;
pub struct Teacher {
    pub model: usize,
    pub ex: MicroExample,
    pub corrected: MicroExample,
    pub q: Vec<f64>,
    pub excluded: Vec<bool>,
    pub group: String,
    pub game: u64,
    pub decision: usize,
    pub position: Position,
}
pub struct Proof {
    pub ex: MicroExample,
    pub certificate: Vec<bool>,
    pub union: Vec<bool>,
    pub group: String,
    pub source: String,
    pub position: Position,
    pub successors: Vec<Successor>,
}
pub struct Successor {
    pub state: Vec<f64>,
    pub sign: f64,
    pub terminal: Option<f64>,
}
impl Successor {
    pub fn value(&self, m: &MicroModel) -> f64 {
        self.terminal
            .unwrap_or_else(|| m.embed(&self.state).value * self.sign)
    }
}
pub struct Data {
    pub models: Vec<MicroModel>,
    pub fresh: Vec<MicroExample>,
    pub teachers: Vec<Teacher>,
    pub proofs: Vec<Proof>,
}
pub fn successors(m: &MicroModel, p: &Position) -> Vec<Successor> {
    legal_actions(p)
        .into_iter()
        .map(|a| {
            let mut n = p.clone();
            n.apply(a).unwrap();
            let terminal = match n.outcome() {
                GameOutcome::Win(w) => Some(if w == p.to_move() { 1. } else { -1. }),
                GameOutcome::Draw => Some(0.),
                _ => None,
            };
            Successor {
                state: m.state_features(&n),
                sign: if n.to_move() == p.to_move() { 1. } else { -1. },
                terminal,
            }
        })
        .collect()
}
pub fn load(spec: &Value) -> Result<Data> {
    let mut models = vec![];
    let mut ids = vec![];
    for s in spec["models"].as_array().unwrap() {
        let a: MicroArtifact = serde_json::from_slice(&fs::read(s["path"].as_str().unwrap())?)?;
        assert_eq!(a.identity(), s["identity"]);
        ids.push(a.identity());
        models.push(a.model()?);
    }
    for m in &models {
        assert!(Arc::ptr_eq(
            models[0].sequence_memory().unwrap(),
            m.sequence_memory().unwrap()
        ));
    }
    let mut proofs = vec![];
    for s in spec["positions"].as_array().unwrap() {
        if s["kind"] != "certificate" {
            continue;
        }
        let bytes = fs::read(s["path"].as_str().unwrap())?;
        assert_eq!(hash(&bytes), s["sha256"]);
        let v: Value = serde_json::from_slice(&bytes)?;
        let r: GameRecord = v["prefix"].as_str().unwrap().parse()?;
        let p = r.replay()?;
        let cert: MicroProofCertificate = serde_json::from_value(v["certificate"].clone())?;
        cert.verify(&p)?;
        let sign = if p.to_move() == Player::Host { 1 } else { -1 };
        let z = (cert.outcome * sign) as f64;
        let legal = legal_actions(&p);
        let mut valid = vec![false; legal.len()];
        if z == 1. {
            for (a, c) in cert.children {
                if c.outcome == sign {
                    let a: Action = a.parse()?;
                    valid[legal.iter().position(|x| *x == a).unwrap()] = true;
                }
            }
        }
        let successors = if z == 1. {
            successors(&models[0], &p)
        } else {
            vec![]
        };
        let union = valid
            .iter()
            .enumerate()
            .map(|(i, v)| *v || z == 1. && successors[i].terminal == Some(1.))
            .collect::<Vec<_>>();
        let n = valid.iter().filter(|v| **v).count();
        let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
            state: models[0].state_features(&p),
            actions: if z == 1. {
                legal
                    .iter()
                    .map(|a| micro_action_features(&p, *a))
                    .collect()
            } else {
                vec![]
            },
            policy: if z == 1. {
                valid
                    .iter()
                    .map(|v| if *v { 1. / n as f64 } else { 0. })
                    .collect()
            } else {
                vec![]
            },
            value: z,
            policy_weight: if z == 1. { 1. } else { 0. },
            value_weight: 1.,
            sequence_source: 0,
        };
        ex.validate()?;
        proofs.push(Proof {
            ex,
            certificate: valid,
            union,
            group: s["group"].as_str().unwrap().into(),
            source: v["human_source"]
                .as_str()
                .unwrap_or(v["prefix"].as_str().unwrap())
                .into(),
            position: p,
            successors,
        });
    }
    let mut fresh = vec![];
    let mut teachers = vec![];
    for path in spec["bundles"].as_array().unwrap() {
        let raw = fs::read(path.as_str().unwrap())?;
        assert!(path
            .as_str()
            .unwrap()
            .ends_with(&format!("{}.json.gz", hash(&raw))));
        let b: Value = serde_json::from_reader(flate2::read::GzDecoder::new(raw.as_slice()))?;
        let record: GameRecord = b["psr"].as_str().unwrap().parse()?;
        let mut position = record.initial_position();
        let mut next = 0;
        let mi = ids
            .iter()
            .position(|id| Some(id.as_str()) == b["source"].as_str());
        for l in b["lessons"].as_array().unwrap() {
            let decision = l["decision"].as_u64().unwrap() as usize;
            while next < decision - 1 {
                position.apply(record.actions()[next])?;
                next += 1;
            }
            let pw = l["policy_weight"].as_f64().unwrap();
            let legal = if pw > 0. {
                legal_actions(&position)
            } else {
                vec![]
            };
            let mut policy = vec![0.; legal.len()];
            for t in l["policy"].as_array().unwrap() {
                let a: Action = t[0].as_str().unwrap().parse()?;
                policy[legal.iter().position(|x| *x == a).unwrap()] = t[1].as_f64().unwrap();
            }
            let ex = MicroExample { structured: Vec::new(), policy_support: false, action_values: vec![], 
                state: models[0].state_features(&position),
                actions: legal
                    .iter()
                    .map(|a| micro_action_features(&position, *a))
                    .collect(),
                policy,
                value: l["value"].as_f64().unwrap(),
                value_weight: l["evidence"]["value_weight"].as_f64().unwrap(),
                policy_weight: pw,
                sequence_source: sequence_source(&format!(
                    "{}/{}",
                    b["source_run"].as_str().unwrap(),
                    b["game_id"].as_u64().unwrap()
                )),
            };
            ex.validate()?;
            if let Some(mi) = mi.filter(|&i| {
                (i == 0 || i == 4) && l["evidence"]["policy_source"] == "full-search-estimate"
            }) {
                let mut base = ex.clone();
                base.value_weight = 0.;
                base.sequence_source = 0;
                let raw = prior(&models[mi], &base);
                let q: Vec<f64> =
                    serde_json::from_value(l["evidence"]["completed_action_values"].clone())?;
                let excluded: Vec<bool> =
                    serde_json::from_value(l["evidence"]["excluded_actions"].clone())?;
                let mut corrected = base.clone();
                corrected.policy = micro_softmax(
                    &raw.iter()
                        .zip(&q)
                        .enumerate()
                        .map(|(i, (p, q))| {
                            if excluded[i] {
                                -1e100
                            } else {
                                p.max(1e-300).ln() + q
                            }
                        })
                        .collect::<Vec<_>>(),
                )?;
                teachers.push(Teacher {
                    model: mi,
                    ex: base,
                    corrected,
                    q,
                    excluded,
                    group: b["case"]["human_source"].as_str().unwrap().into(),
                    game: b["game_id"].as_u64().unwrap(),
                    decision,
                    position: position.clone(),
                });
            }
            fresh.push(ex);
        }
    }
    Ok(Data {
        models,
        fresh,
        teachers,
        proofs,
    })
}
