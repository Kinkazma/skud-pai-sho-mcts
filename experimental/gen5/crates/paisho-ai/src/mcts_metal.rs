//! Shared GPU batch broker for independent MCTS trees. Rules and UCT stay on CPU.
use crate::{HeuristicWeights, MctsEvaluator};
use paisho_core::{GameOutcome, Player, Position};
use std::{
    io::{Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicU64, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

struct Request {
    boards: Vec<u8>,
    reply: mpsc::Sender<Result<Vec<[i32; 4]>, String>>,
}
#[derive(Default)]
pub struct MetalMctsTelemetry {
    pub batches: AtomicU64,
    pub positions: AtomicU64,
    pub max_batch: AtomicU64,
}
pub struct MetalMctsEvaluator {
    sender: Option<mpsc::SyncSender<Request>>,
    child: Arc<Mutex<Child>>,
    worker: Option<thread::JoinHandle<()>>,
    pub telemetry: Arc<MetalMctsTelemetry>,
}
impl MetalMctsEvaluator {
    pub fn launch(
        binary: &Path,
        kernel: &Path,
        max_batch: usize,
        wait: Duration,
    ) -> Result<Self, String> {
        if !(1..=32768).contains(&max_batch) || wait > Duration::from_millis(10) {
            return Err("invalid GPU batch capacity/wait".into());
        }
        let mut child = Command::new(binary)
            .arg(kernel)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut input = child.stdin.take().unwrap();
        let mut output = child.stdout.take().unwrap();
        let child = Arc::new(Mutex::new(child));
        let (sender, receiver) = mpsc::sync_channel::<Request>(64);
        let (ready_tx, ready_rx) = mpsc::channel();
        let telemetry = Arc::new(MetalMctsTelemetry::default());
        let stats = telemetry.clone();
        let worker = thread::spawn(move || {
            let mut magic = [0; 4];
            let handshake = output
                .read_exact(&mut magic)
                .map_err(|e| e.to_string())
                .and_then(|_| {
                    if &magic == b"PMG1" {
                        Ok(())
                    } else {
                        Err("invalid GPU handshake".into())
                    }
                });
            let ok = handshake.is_ok();
            let _ = ready_tx.send(handshake);
            if !ok {
                return;
            }
            while let Ok(first) = receiver.recv() {
                let until = Instant::now() + wait;
                let mut pending = vec![first];
                let mut count = pending[0].boards.len() / 289;
                while count < max_batch {
                    let next = if wait.is_zero() {
                        receiver.try_recv().ok()
                    } else {
                        receiver
                            .recv_timeout(until.saturating_duration_since(Instant::now()))
                            .ok()
                    };
                    let Some(next) = next else {
                        break;
                    };
                    count += next.boards.len() / 289;
                    pending.push(next);
                }
                let bytes: Vec<u8> = pending
                    .iter()
                    .flat_map(|r| r.boards.iter().copied())
                    .collect();
                let result = (|| -> Result<Vec<[i32; 4]>, String> {
                    let mut features = Vec::with_capacity(count);
                    for chunk in bytes.chunks(max_batch * 289) {
                        let n = chunk.len() / 289;
                        input
                            .write_all(&(n as u32).to_le_bytes())
                            .and_then(|_| input.write_all(chunk))
                            .and_then(|_| input.flush())
                            .map_err(|e| e.to_string())?;
                        let mut reply = vec![0; n * 16];
                        output.read_exact(&mut reply).map_err(|e| e.to_string())?;
                        features.extend(reply.chunks_exact(16).map(|v| {
                            std::array::from_fn(|i| {
                                i32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap())
                            })
                        }));
                        stats.batches.fetch_add(1, Ordering::Relaxed);
                        stats.positions.fetch_add(n as u64, Ordering::Relaxed);
                        stats.max_batch.fetch_max(n as u64, Ordering::Relaxed);
                    }
                    Ok(features)
                })();
                let mut offset = 0;
                for request in pending {
                    let n = request.boards.len() / 289;
                    let reply = result
                        .as_ref()
                        .map(|f| f[offset..offset + n].to_vec())
                        .map_err(Clone::clone);
                    let _ = request.reply.send(reply);
                    offset += n;
                }
                if result.is_err() {
                    break;
                }
            }
        });
        let ready = ready_rx
            .recv_timeout(Duration::from_secs(30))
            .map_err(|e| e.to_string())
            .and_then(|r| r);
        if let Err(error) = ready {
            let mut c = child.lock().unwrap();
            let _ = c.kill();
            let _ = c.wait();
            drop(c);
            let _ = worker.join();
            return Err(error);
        }
        Ok(Self {
            sender: Some(sender),
            child,
            worker: Some(worker),
            telemetry,
        })
    }
}
impl MetalMctsEvaluator {
    pub fn board_features(&self, boards: Vec<u8>) -> Result<Vec<[i32; 4]>, String> {
        if boards.is_empty() || boards.len() % 289 != 0 || boards.len() > 32768 * 289 {
            return Err("invalid GPU board request size".into());
        }
        let (reply, rx) = mpsc::channel();
        self.sender
            .as_ref()
            .unwrap()
            .send(Request { boards, reply })
            .map_err(|_| "GPU broker stopped")?;
        rx.recv_timeout(Duration::from_secs(30)).map_err(|e| {
            let _ = self.child.lock().unwrap().kill();
            format!("GPU response deadline: {e}")
        })?
    }
}
impl MctsEvaluator for MetalMctsEvaluator {
    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        evaluate_boards(positions, perspective, weights, |b| self.board_features(b))
    }
}
fn evaluate_boards(
    positions: &[Position],
    perspective: Player,
    weights: HeuristicWeights,
    fetch: impl Fn(Vec<u8>) -> Result<Vec<[i32; 4]>, String>,
) -> Result<Vec<f32>, String> {
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    // Bound each message even if a caller supplies an entire corpus.
    let mut values = Vec::with_capacity(positions.len());
    for positions in positions.chunks(32768) {
        let mut boards = vec![0; positions.len() * 289];
        for (i, p) in positions.iter().enumerate() {
            for (c, t) in p.board().occupied() {
                boards[i * 289 + c.dense_index()] =
                    (1 + t.kind.index() + 12 * t.owner.index()) as u8;
            }
        }
        let features = fetch(boards)?;
        for (p, f) in positions.iter().zip(features) {
            let value = match p.outcome() {
                GameOutcome::Win(w) => {
                    if w == perspective {
                        1.0
                    } else {
                        -1.0
                    }
                }
                GameOutcome::Draw => 0.0,
                GameOutcome::Ongoing => {
                    let sign = if perspective.index() == 0 { 1.0 } else { -1.0 };
                    let reserve = f64::from(p.reserve(perspective.opponent()).basic_count())
                        - f64::from(p.reserve(perspective).basic_count());
                    let raw = f64::from(weights.harmony) * (f64::from(f[0]) * sign)
                        + f64::from(weights.midline_harmony) * (f64::from(f[1]) * sign)
                        + f64::from(weights.blooming_flower) * (f64::from(f[2]) * sign)
                        + f64::from(weights.total_flower) * (f64::from(f[3]) * sign)
                        + f64::from(weights.basic_reserve_progress) * reserve;
                    (raw / (1.0 + raw.abs())) as f32
                }
            };
            values.push(value);
        }
    }
    Ok(values)
}
impl Drop for MetalMctsEvaluator {
    fn drop(&mut self) {
        self.sender.take();
        // Kill also unblocks a broken service read; no child is left behind.
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// One game connection to a shared GPU service. Killing this process never kills
/// the other trees. Unix socket lives in the private corpus work directory.
#[cfg(unix)]
pub struct RemoteMetalMctsEvaluator {
    stream: Mutex<std::os::unix::net::UnixStream>,
}
#[cfg(unix)]
impl RemoteMetalMctsEvaluator {
    pub fn connect(path: &Path) -> Result<Self, String> {
        let stream = std::os::unix::net::UnixStream::connect(path).map_err(|e| e.to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(30)))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            stream: Mutex::new(stream),
        })
    }
}
#[cfg(unix)]
impl MctsEvaluator for RemoteMetalMctsEvaluator {
    fn evaluate(
        &self,
        positions: &[Position],
        perspective: Player,
        weights: HeuristicWeights,
    ) -> Result<Vec<f32>, String> {
        evaluate_boards(positions, perspective, weights, |boards| {
            let count = boards.len() / 289;
            let mut stream = self.stream.lock().map_err(|e| e.to_string())?;
            stream
                .write_all(&(count as u32).to_le_bytes())
                .and_then(|_| stream.write_all(&boards))
                .map_err(|e| e.to_string())?;
            let mut bytes = vec![0; count * 16];
            stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
            Ok(bytes
                .chunks_exact(16)
                .map(|v| {
                    std::array::from_fn(|i| {
                        i32::from_le_bytes(v[i * 4..i * 4 + 4].try_into().unwrap())
                    })
                })
                .collect())
        })
    }
}
