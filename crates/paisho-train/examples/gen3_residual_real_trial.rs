//! Predeclared, isolated value-only experiment. Never installs campaign weights.
use paisho_ai::{micro_state_features, Gen32Model, MicroExample};
use paisho_core::{GameOutcome, GameRecord, Player, RuleProfileId};
use paisho_train::{compact_learning::load_dataset, gen32::Artifact};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, fs, path::Path, time::Instant};
struct Source {
    id: String,
    held: bool,
    rows: Vec<MicroExample>,
}
fn mse(m: &Gen32Model, sources: &[Source], held: bool) -> f64 {
    let groups: Vec<_> = sources.iter().filter(|s| s.held == held).collect();
    groups
        .iter()
        .map(|s| {
            s.rows
                .iter()
                .map(|e| (m.predict_value_state(&e.state) - e.value).powi(2))
                .sum::<f64>()
                / s.rows.len() as f64
        })
        .sum::<f64>()
        / groups.len() as f64
}
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().collect();
    let parent = Artifact::load(Path::new(&a[1]))?;
    let base = parent.model()?;
    assert!(base.value_residual.is_none());
    let data = load_dataset(Path::new(&a[2]))?;
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let mut counts = [0usize; 2];
    let mut sources = vec![];
    let mut manifest = vec![];
    let mut identities = HashSet::new();
    for game in &data.games {
        let split = usize::from(game.held_out);
        if counts[split] >= [192, 64][split] || game.external_outcome.is_some() {
            continue;
        }
        let identity = game
            .split_identity_sha256
            .as_ref()
            .unwrap_or(&game.game_sha256);
        if !identities.insert(identity.clone()) {
            continue;
        }
        let raw = fs::read(&game.originals[0].path)?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            game.originals[0].sha256
        );
        let record: GameRecord = std::str::from_utf8(&raw)?.parse()?;
        let mut p = GameRecord::with_rules(record.setup(), RuleProfileId::SkudPaiSho2022V2)
            .initial_position();
        let end = record.replay()?.outcome();
        assert_ne!(end, GameOutcome::Ongoing);
        let outcome = match end {
            GameOutcome::Win(Player::Host) => "H",
            GameOutcome::Win(Player::Guest) => "G",
            GameOutcome::Draw => "draw",
            _ => unreachable!(),
        };
        assert_eq!(outcome, game.outcome);
        let mut rows = vec![];
        for (decision, action) in record.actions().iter().enumerate() {
            if decision % 3 == 0 && game.examples.iter().any(|e| e.decision_index == decision) {
                let value = match end {
                    GameOutcome::Win(w) if w == p.to_move() => 1.,
                    GameOutcome::Win(_) => -1.,
                    GameOutcome::Draw => 0.,
                    _ => unreachable!(),
                };
                rows.push(MicroExample {
                    sequence_source: 0,
                    state: micro_state_features(&p),
                    actions: vec![],
                    policy: vec![],
                    value,
                    policy_weight: 0.,
                });
            }
            p.apply(*action)?;
        }
        assert_eq!(p.outcome(), end);
        assert!(!rows.is_empty());
        manifest.push(json!({"source":game.game_sha256,"split_identity":identity,"held_out":game.held_out,"path":game.originals[0].path,"sha256":game.originals[0].sha256,"outcome":outcome,"positions":rows.len()}));
        sources.push(Source {
            id: game.game_sha256.clone(),
            held: game.held_out,
            rows,
        });
        counts[split] += 1;
    }
    assert_eq!(counts, [192, 64]);
    fs::write(
        out.join("sources.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    let rows: Vec<_> = sources.iter().flat_map(|s|s.rows.iter().map(move |e|json!({"source":s.id,"held_out":s.held,"x":e.state.as_slice(),"y":e.value}))).collect();
    fs::write(out.join("examples.json"), serde_json::to_vec(&rows)?)?;
    let train: Vec<_> = sources.iter().filter(|s| !s.held).collect();
    let mut rng = 20260911;
    let stream: Vec<_> = (0..100000)
        .map(|_| {
            let i = random(&mut rng) as usize % train.len();
            let j = random(&mut rng) as usize % train[i].rows.len();
            (i, j)
        })
        .collect();
    let mut metrics = vec![
        json!({"model":"parent","step":0,"train_mse":mse(&base,&sources,false),"validation_mse":mse(&base,&sources,true)}),
    ];
    for (name, seed) in [
        ("linear", None),
        ("residual7", Some(7)),
        ("residual71", Some(71)),
        ("residual731", Some(731)),
    ] {
        let mut m = match seed {
            Some(s) => base.clone().with_value_residual(s),
            None => base.clone(),
        };
        let start = Instant::now();
        assert_eq!(mse(&m, &sources, false), mse(&base, &sources, false));
        for (step, (i, j)) in stream.iter().enumerate() {
            m.train(&train[*i].rows[*j], 0.01)?;
            if [1000, 10000, 100000].contains(&(step + 1)) {
                let row = json!({"model":name,"step":step+1,"train_mse":mse(&m,&sources,false),"validation_mse":mse(&m,&sources,true),"elapsed_seconds_contended":start.elapsed().as_secs_f64()});
                println!("{row}");
                metrics.push(row);
            }
        }
        assert_eq!(m.policy.parameters(), base.policy.parameters());
        parent
            .updated(&m, parent.updates + 100000)
            .save(&out.join(format!("{name}.json")))?;
    }
    fs::write(
        out.join("fit.json"),
        serde_json::to_vec_pretty(
            &json!({"parent":a[1],"parent_sha256":format!("{:x}",Sha256::digest(fs::read(&a[1])?)),"counts":counts,"positions":rows.len(),"stream_seed":20260911,"updates":100000,"rate":0.01,"metrics":metrics}),
        )?,
    )?;
    Ok(())
}
