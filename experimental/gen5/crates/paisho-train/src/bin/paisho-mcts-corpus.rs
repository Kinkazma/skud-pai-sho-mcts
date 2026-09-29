//! Durable corpus worker and shared Metal service. Orchestrated by tools/paisho_mcts_corpus.py.
use paisho_ai::{
    CpuMctsEvaluator, HeuristicWeights, MctsAgent, MctsConfig, MctsEvaluator, MetalMctsEvaluator,
    RemoteMetalMctsEvaluator, StableRng,
};
use paisho_core::{
    legal_actions, BasicFlower, GameOutcome, GameRecord, Player, Position, StandardSetup, TurnPhase,
};
use rayon::prelude::*;
use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

fn serve_connection(mut stream: UnixStream, gpu: &MetalMctsEvaluator) -> Result<(), String> {
    // Darwin can inherit O_NONBLOCK from the listening socket.
    stream.set_nonblocking(false).map_err(|e| e.to_string())?;
    stream
        .set_read_timeout(Some(Duration::from_secs(60)))
        .map_err(|e| e.to_string())?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|e| e.to_string())?;
    loop {
        let mut header = [0; 4];
        match stream.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.to_string()),
        }
        let count = u32::from_le_bytes(header) as usize;
        if count == 0 || count > 32768 {
            return Err("invalid board count".into());
        }
        let mut bytes = vec![0; count * 289];
        stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.iter().any(|&b| b > 24) {
            return Err("invalid tile code".into());
        }
        let features = gpu.board_features(bytes)?;
        let bytes: Vec<u8> = features
            .into_iter()
            .flatten()
            .flat_map(i32::to_le_bytes)
            .collect();
        stream.write_all(&bytes).map_err(|e| e.to_string())?;
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("serve") if a.len()==6 => {
            let gpu=Arc::new(MetalMctsEvaluator::launch(Path::new(&a[1]),Path::new(&a[2]),a[4].parse()?,Duration::from_micros(a[5].parse()?))?);
            let socket=UnixListener::bind(&a[3])?;
            println!("ready");
            socket.set_nonblocking(true)?;
            let mut last=Instant::now();
            loop {
                match socket.accept() {
                    Ok((stream,_)) => { let gpu=gpu.clone(); std::thread::spawn(move|| { if let Err(e)=serve_connection(stream,&gpu) { eprintln!("GPU client closed: {e}"); } }); },
                    Err(e) if e.kind()==std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(1)),
                    Err(e) => return Err(e.into()),
                }
                if last.elapsed()>=Duration::from_secs(1) {
                    use std::sync::atomic::Ordering::Relaxed;
                    let row=json!({"batches":gpu.telemetry.batches.load(Relaxed),"positions":gpu.telemetry.positions.load(Relaxed),"maximum_batch":gpu.telemetry.max_batch.load(Relaxed)});
                    fs::write(format!("{}.metrics.tmp",a[3]),serde_json::to_vec(&row)?)?;
                    fs::rename(format!("{}.metrics.tmp",a[3]),format!("{}.metrics.json",a[3]))?;
                    last=Instant::now();
                }
            }
        }
        Some("game") if a.len()==6 => {
            // Deadline enforced externally by killing this process, including its socket.
            let simulations:usize=a[1].parse()?; let ordinal:u64=a[2].parse()?;
            let mut seed_rng=StableRng::new(70332026 ^ ordinal.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let seed=seed_rng.next_u64();
            let mut agent=MctsAgent::new(seed,MctsConfig{simulations,..MctsConfig::default()})?;
            let backend:Box<dyn MctsEvaluator>=if a[3]=="cpu" { Box::new(CpuMctsEvaluator) } else { Box::new(RemoteMetalMctsEvaluator::connect(Path::new(&a[3]))?) };
            let limit:usize=a[5].parse()?;
            if limit==0 { return Err("source limit must be positive".into()); }
            let pool=rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
            let setup=StandardSetup::balanced(BasicFlower::Red3);
            let mut position=Position::from_standard_setup(setup);
            let mut record=GameRecord::new(setup);
            let mut cuts=vec![0usize];
            let begin=Instant::now();
            while position.outcome()==GameOutcome::Ongoing {
                if record.actions().len()>=limit && position.phase()==TurnPhase::Main { break; }
                let actions=legal_actions(&position);
                let report=pool.install(||agent.search_with_evaluator(&position,&actions,backend.as_ref()))?;
                position.apply(actions[report.selected_index])?;
                record.push(actions[report.selected_index]);
                if position.outcome()==GameOutcome::Ongoing && position.phase()==TurnPhase::Main { cuts.push(record.actions().len()); }
            }
            let terminal=position.outcome()!=GameOutcome::Ongoing;
            // Store all legal main-turn cut boundaries. Prefixes are reconstructed
            // from the full source, so provenance always reaches the opening.
            let row=json!({"ordinal":ordinal,"seed":seed,"simulations":simulations,"terminal":terminal,
                "outcome":format!("{:?}",position.outcome()),"decisions":record.actions().len(),
                "generation_seconds":begin.elapsed().as_secs_f64(),"cuts":cuts});
            fs::create_dir(&a[4])?;
            if terminal { fs::write(Path::new(&a[4]).join("game.psr"),record.to_string())?; }
            fs::write(Path::new(&a[4]).join("result.json"),serde_json::to_vec_pretty(&row)?)?;
        }
        Some("cut") if a.len()==4 => {
            let source:GameRecord=fs::read_to_string(&a[1])?.parse()?;
            let count:usize=a[2].parse()?;
            if count>=source.actions().len() { return Err("cut must precede terminal action".into()); }
            let mut prefix=GameRecord::with_rules(source.setup(), source.rules());
            for &action in &source.actions()[..count] { prefix.push(action); }
            let position=prefix.replay()?;
            if position.phase()!=TurnPhase::Main || position.outcome()!=GameOutcome::Ongoing { return Err("cut must be a nonterminal Main boundary".into()); }
            let mut file=fs::OpenOptions::new().write(true).create_new(true).open(&a[3])?;
            file.write_all(prefix.to_string().as_bytes())?;
        }
        Some("bench") if a.len()==5 => throughput(Path::new(&a[1]),Path::new(&a[2]),a[3].parse()?,a[4].parse()?)?,
        Some("verify") if a.len()==5 => verify(Path::new(&a[1]),Path::new(&a[2]),a[3].parse()?,a[4].parse()?)?,
        _=>return Err("usage: cut SOURCE DECISIONS NEW_PSR | serve SERVICE KERNEL SOCKET BATCH WAIT_US | game SIMULATIONS ORDINAL cpu|SOCKET NEW_OUTPUT SOURCE_LIMIT | verify SERVICE KERNEL BATCH WAIT_US".into())
    }
    Ok(())
}
fn verify(
    service: &Path,
    kernel: &Path,
    batch: usize,
    wait: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let gpu = MetalMctsEvaluator::launch(service, kernel, batch, Duration::from_micros(wait))?;
    let mut positions = Vec::new();
    for seed in 0..16 {
        let mut rng = StableRng::new(seed + 8321);
        let mut p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        for step in 0..1024 {
            if step % 8 == 0 || p.outcome() != GameOutcome::Ongoing {
                positions.push(p.clone());
            }
            if p.outcome() != GameOutcome::Ongoing {
                break;
            }
            let actions = legal_actions(&p);
            p.apply(actions[rng.index(actions.len())])?;
        }
    }
    for perspective in [Player::Host, Player::Guest] {
        let expected =
            CpuMctsEvaluator.evaluate(&positions, perspective, HeuristicWeights::default())?;
        let actual = gpu.evaluate(&positions, perspective, HeuristicWeights::default())?;
        if expected != actual {
            let i = expected
                .iter()
                .zip(&actual)
                .position(|(a, b)| a != b)
                .unwrap();
            return Err(format!(
                "heuristic mismatch at {i}: {} != {}",
                expected[i], actual[i]
            )
            .into());
        }
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(10).build()?;
    let samples: Vec<_> = positions
        .iter()
        .filter(|p| p.outcome() == GameOutcome::Ongoing)
        .step_by(29)
        .take(32)
        .collect();
    for simulations in [8, 32, 128, 512] {
        let config = MctsConfig {
            simulations,
            ..MctsConfig::default()
        };
        let reports: Vec<_> = pool.install(|| {
            samples
                .par_iter()
                .enumerate()
                .map(|(i, p)| {
                    let actions = legal_actions(p);
                    let expected = MctsAgent::new(i as u64, config)
                        .unwrap()
                        .search(p, &actions);
                    let actual = MctsAgent::new(i as u64, config)
                        .unwrap()
                        .search_with_evaluator(p, &actions, &gpu)?;
                    if expected.actions != actual.actions
                        || expected.selected_index != actual.selected_index
                        || expected.simulations != actual.simulations
                    {
                        return Err(format!("MCTS-{simulations} mismatch at position {i}"));
                    }
                    Ok(())
                })
                .collect()
        });
        for result in reports {
            result?;
        }
        println!(
            "parity simulations={simulations} searches={}",
            samples.len()
        );
    }
    use std::sync::atomic::Ordering::Relaxed;
    println!(
        "{}",
        json!({"positions":positions.len(),"parity":"exact","gpu_batches":gpu.telemetry.batches.load(Relaxed),"gpu_positions":gpu.telemetry.positions.load(Relaxed),"maximum_gpu_batch":gpu.telemetry.max_batch.load(Relaxed)})
    );
    Ok(())
}

fn throughput(
    service: &Path,
    kernel: &Path,
    batch: usize,
    wait: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let gpu = MetalMctsEvaluator::launch(service, kernel, batch, Duration::from_micros(wait))?;
    let mut positions = Vec::new();
    for seed in 0..32 {
        let mut rng = StableRng::new(seed + 8321);
        let mut p = Position::from_standard_setup(StandardSetup::balanced(BasicFlower::Red3));
        for step in 0..128 {
            if step >= 32 && step % 16 == 0 {
                positions.push(p.clone());
            }
            if p.outcome() != GameOutcome::Ongoing {
                break;
            }
            let a = legal_actions(&p);
            p.apply(a[rng.index(a.len())])?;
        }
    }
    for (label, backend, workers) in [
        ("cpu-a", &CpuMctsEvaluator as &dyn MctsEvaluator, 10),
        ("metal10", &gpu, 10),
        ("metal20", &gpu, 20),
        ("metal40", &gpu, 40),
        ("cpu-b", &CpuMctsEvaluator, 10),
    ] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()?;
        let begin = Instant::now();
        let results: Vec<_> = pool.install(|| {
            positions
                .par_iter()
                .enumerate()
                .map(|(i, p)| {
                    MctsAgent::new(
                        i as u64,
                        MctsConfig {
                            simulations: 32,
                            ..MctsConfig::default()
                        },
                    )
                    .unwrap()
                    .search_with_evaluator(p, &legal_actions(p), backend)
                })
                .collect()
        });
        for result in results {
            result?;
        }
        println!(
            "{}",
            json!({"backend":label,"workers":workers,"searches":positions.len(),"seconds":begin.elapsed().as_secs_f64()})
        );
    }
    Ok(())
}
